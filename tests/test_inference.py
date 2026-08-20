"""Read-only inference-loader tests built from synthetic M3 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import predictive_maintenance.evaluation as evaluation
import predictive_maintenance.inference as inference
import predictive_maintenance.modeling as modeling
from predictive_maintenance.evaluation import evaluate_final_holdout
from predictive_maintenance.inference import ModelArtifactError, load_inference_service
from predictive_maintenance.modeling import build_candidate_pipelines, run_training_selection
from predictive_maintenance.schemas import PredictionRequest
from predictive_maintenance.validation import MODELING_COLUMNS, TARGET_COLUMN


def _frame(row_count: int, *, positive_every: int = 5) -> pd.DataFrame:
    target = np.array(
        [1 if index % positive_every == 0 else 0 for index in range(row_count)], dtype=int
    )
    return pd.DataFrame(
        {
            "Type": [["L", "M", "H"][index % 3] for index in range(row_count)],
            "Air temperature [K]": [
                299.0 + 2.0 * target[index] + (index % 5) / 10 for index in range(row_count)
            ],
            "Process temperature [K]": [
                309.0 + target[index] + (index % 7) / 10 for index in range(row_count)
            ],
            "Rotational speed [rpm]": [
                1500 - 120 * target[index] + index % 30 for index in range(row_count)
            ],
            "Torque [Nm]": [
                35.0 + 20.0 * target[index] + (index % 4) / 10 for index in range(row_count)
            ],
            "Tool wear [min]": [
                50 + 100 * target[index] + index % 40 for index in range(row_count)
            ],
            TARGET_COLUMN: target,
        },
        columns=MODELING_COLUMNS,
    )


def _small_pipelines():
    pipelines = build_candidate_pipelines()
    pipelines["random_forest"].set_params(model__n_estimators=8, model__max_depth=4)
    return pipelines


@pytest.fixture
def evaluated_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, object]:
    training = _frame(150)
    holdout = _frame(30)
    manifest = {
        "source": {"filename": "ai4i2020.csv", "sha256": "a" * 64, "rows": 180},
        "split": {"random_seed": 42},
        "training": {"filename": "train.csv", "rows": 150, "sha256": "b" * 64},
        "holdout": {"filename": "holdout.csv", "rows": 30, "sha256": "c" * 64},
    }
    report_dir = tmp_path / "reports" / "modeling"
    artifact_root = tmp_path / "artifacts" / "m3"
    ledger_dir = tmp_path / "reports" / "holdout_access"
    monkeypatch.setattr(modeling, "load_training_partition", lambda *_: (training.copy(), manifest))
    monkeypatch.setattr(modeling, "build_candidate_pipelines", _small_pipelines)
    artifacts = run_training_selection(
        tmp_path / "processed",
        tmp_path / "split.json",
        report_dir,
        artifact_root,
    )
    monkeypatch.setattr(
        evaluation,
        "load_training_partition",
        lambda *_: (training.copy(), manifest),
    )
    evaluate_final_holdout(
        tmp_path / "processed",
        tmp_path / "split.json",
        report_dir,
        artifact_root,
        confirm_final_evaluation=True,
        holdout_ledger_dir=ledger_dir,
        holdout_loader=lambda *_: (holdout.copy(), manifest),
    )
    return report_dir, artifact_root, ledger_dir, artifacts


def test_loader_is_read_only_and_never_resolves_dataset_files(
    evaluated_bundle: tuple[Path, Path, Path, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_dir, artifact_root, ledger_dir, artifacts = evaluated_bundle
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.name in {"ai4i2020.csv", "train.csv", "holdout.csv"}:
            raise AssertionError("inference startup must not resolve any dataset partition")
        return original_open(path, *args, **kwargs)

    def forbidden_write(*_args, **_kwargs):
        raise AssertionError("inference startup must not mutate local state")

    monkeypatch.setattr(Path, "open", guarded_open)
    for method in ("mkdir", "replace", "touch", "unlink", "write_bytes", "write_text"):
        monkeypatch.setattr(Path, method, forbidden_write)

    service = load_inference_service(report_dir, artifact_root, ledger_dir)
    result = service.predict(
        PredictionRequest(
            type="M",
            air_temperature_k=300.0,
            process_temperature_k=310.0,
            rotational_speed_rpm=1500,
            torque_nm=40.0,
            tool_wear_min=100,
        )
    )

    assert service.metadata.run_id == artifacts.run_id
    assert service.metadata.pipeline_sha256
    assert 0.0 <= result.risk_score <= 1.0


def test_pipeline_tampering_fails_before_deserialization(
    evaluated_bundle: tuple[Path, Path, Path, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_dir, artifact_root, ledger_dir, artifacts = evaluated_bundle
    pipeline_path = artifact_root / artifacts.run_id / "pipeline.joblib"
    with pipeline_path.open("ab") as stream:
        stream.write(b"tampered")
    called = False

    def forbidden_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("hash failure must happen before joblib.load")

    monkeypatch.setattr(inference.joblib, "load", forbidden_load)

    with pytest.raises(ModelArtifactError, match="SHA-256"):
        load_inference_service(report_dir, artifact_root, ledger_dir)
    assert called is False


def test_final_receipt_tampering_is_rejected(
    evaluated_bundle: tuple[Path, Path, Path, object],
) -> None:
    report_dir, artifact_root, ledger_dir, artifacts = evaluated_bundle
    receipt_path = report_dir / artifacts.run_id / "final_evaluation.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["threshold"] = 0.0
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="receipt|ledger"):
        load_inference_service(report_dir, artifact_root, ledger_dir)


def test_runtime_mismatch_fails_before_deserialization(
    evaluated_bundle: tuple[Path, Path, Path, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_dir, artifact_root, ledger_dir, _ = evaluated_bundle
    called = False

    def forbidden_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("runtime mismatch must happen before joblib.load")

    monkeypatch.setattr(inference, "_runtime_is_compatible", lambda _: False)
    monkeypatch.setattr(inference.joblib, "load", forbidden_load)

    with pytest.raises(ModelArtifactError, match="frozen M3 model"):
        load_inference_service(report_dir, artifact_root, ledger_dir)
    assert called is False
