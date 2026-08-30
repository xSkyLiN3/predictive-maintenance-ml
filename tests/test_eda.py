"""Smoke tests for deterministic training-only EDA artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from predictive_maintenance.config import HOLDOUT_FRACTION, RANDOM_SEED
from predictive_maintenance.dataset import OFFICIAL_DATASET, sha256_file
from predictive_maintenance.eda import FIGURE_FILENAMES, generate_eda
from predictive_maintenance.splitting import SPLIT_MANIFEST_SCHEMA_VERSION
from predictive_maintenance.validation import FEATURE_COLUMNS, MODELING_COLUMNS, TARGET_COLUMN


def _training_frame(row_count: int = 8_000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Type": [["L", "M", "H"][index % 3] for index in range(row_count)],
            "Air temperature [K]": [295.3 + (index % 90) / 10 for index in range(row_count)],
            "Process temperature [K]": [305.7 + (index % 80) / 10 for index in range(row_count)],
            "Rotational speed [rpm]": [1168 + (index * 17) % 1700 for index in range(row_count)],
            "Torque [Nm]": [3.8 + ((index * 7) % 700) / 10 for index in range(row_count)],
            "Tool wear [min]": [(index * 11) % 254 for index in range(row_count)],
            TARGET_COLUMN: [1 if index % 10 == 0 else 0 for index in range(row_count)],
        }
    )


def _write_training_and_manifest(tmp_path: Path) -> tuple[Path, Path]:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    training_path = processed_dir / "train.csv"
    training_frame = _training_frame()
    training_frame.to_csv(training_path, index=False, lineterminator="\n")
    manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "source": {
            "filename": OFFICIAL_DATASET.csv_filename,
            "sha256": OFFICIAL_DATASET.csv_sha256,
            "rows": 10_000,
        },
        "split": {
            "method": "sklearn.model_selection.train_test_split",
            "random_seed": RANDOM_SEED,
            "holdout_fraction": HOLDOUT_FRACTION,
            "stratified_by": TARGET_COLUMN,
            "rows_sorted_within_partitions": True,
        },
        "training": {
            "filename": "train.csv",
            "rows": len(training_frame),
            "sha256": sha256_file(training_path),
            "columns": list(MODELING_COLUMNS),
        },
        "holdout": {
            "filename": "holdout.csv",
            "rows": 2_000,
            "sha256": "0" * 64,
            "columns": list(MODELING_COLUMNS),
            "access_policy": "Do not inspect until the final M3 evaluation.",
        },
        "versions": {
            "python": "3.12.0",
            "pandas": pd.__version__,
            "scikit_learn": "1.9.0",
        },
    }
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return processed_dir, manifest_path


def test_eda_generates_six_idempotent_artifacts_without_holdout(tmp_path: Path) -> None:
    processed_dir, manifest_path = _write_training_and_manifest(tmp_path)
    report_dir = tmp_path / "eda"
    assert not (processed_dir / "holdout.csv").exists()

    artifacts = generate_eda(processed_dir, manifest_path, report_dir)

    assert len(artifacts.figures) == 6
    assert tuple(path.name for path in artifacts.figures) == FIGURE_FILENAMES
    assert all(path.stat().st_size > 10_000 for path in artifacts.figures)
    summary = json.loads(artifacts.summary.read_text(encoding="utf-8"))
    assert summary["scope"] == "training_only"
    assert summary["schema_version"] == 2
    assert summary["feature_columns"] == list(FEATURE_COLUMNS)
    assert summary["target_column"] == TARGET_COLUMN
    assert "modeling_columns" not in summary
    assert summary["training_rows"] == 8_000
    assert summary["target"] == {"negative": 7_200, "positive": 800, "prevalence": 0.1}
    assert summary["protocol"]["holdout_profiled"] is False
    assert summary["protocol"]["threshold_score"] == "predict_proba[:, 1]"
    assert summary["protocol"]["threshold_decision_rule"] == "score >= threshold"
    assert summary["protocol"]["pooled_oof_average_precision_role"] == "diagnostic_only"
    for column, rows in summary["positive_rate_by_quintile"].items():
        _, expected_edges = pd.qcut(_training_frame()[column], q=5, duplicates="drop", retbins=True)
        assert [row["lower_bound"] for row in rows] == pytest.approx(expected_edges[:-1])
        assert [row["upper_bound"] for row in rows] == pytest.approx(expected_edges[1:])
        assert rows[0]["lower_inclusive"] is True
        assert all(row["upper_inclusive"] is True for row in rows)
    report = artifacts.report.read_text(encoding="utf-8")
    assert "uses exclusively the **training** partition" in report
    assert "There are no trained models" in report
    assert "`predict_proba[:, 1]`" in report
    assert "`score >= threshold`" in report
    assert not (processed_dir / "holdout.csv").exists()

    tracked_paths = (artifacts.report, artifacts.summary, *artifacts.figures)
    mtimes = {path: path.stat().st_mtime_ns for path in tracked_paths}
    second_artifacts = generate_eda(processed_dir, manifest_path, report_dir)
    assert second_artifacts == artifacts
    assert mtimes == {path: path.stat().st_mtime_ns for path in tracked_paths}
