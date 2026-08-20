"""Tests for the fixed, leakage-safe M2 partition."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import predictive_maintenance.splitting as splitting
from predictive_maintenance.dataset import DatasetPaths
from predictive_maintenance.splitting import (
    SplitIntegrityError,
    create_stratified_partitions,
    load_training_partition,
    materialize_split,
)
from predictive_maintenance.validation import MODELING_COLUMNS, TARGET_COLUMN


def _small_modeling_frame(row_count: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Type": [["L", "M", "H"][index % 3] for index in range(row_count)],
            "Air temperature [K]": [295.3 + (index % 90) / 10 for index in range(row_count)],
            "Process temperature [K]": [305.7 + (index % 80) / 10 for index in range(row_count)],
            "Rotational speed [rpm]": [1100 + index for index in range(row_count)],
            "Torque [Nm]": [5.0 + (index % 60) for index in range(row_count)],
            "Tool wear [min]": [index % 254 for index in range(row_count)],
            TARGET_COLUMN: [1 if index % 10 == 0 else 0 for index in range(row_count)],
        }
    )


def _full_source_frame() -> pd.DataFrame:
    row_numbers = list(range(1, 10_001))
    types = [["L", "M", "H"][(row_number - 1) % 3] for row_number in row_numbers]
    return pd.DataFrame(
        {
            "UDI": row_numbers,
            "Product ID": [
                f"{product_type}{row_number:05d}"
                for product_type, row_number in zip(types, row_numbers, strict=True)
            ],
            "Type": types,
            "Air temperature [K]": [295.3 + (value % 90) / 10 for value in row_numbers],
            "Process temperature [K]": [305.7 + (value % 80) / 10 for value in row_numbers],
            "Rotational speed [rpm]": [1168 + value % 1000 for value in row_numbers],
            "Torque [Nm]": [3.8 + (value % 700) / 10 for value in row_numbers],
            "Tool wear [min]": [value % 254 for value in row_numbers],
            TARGET_COLUMN: [1 if value <= 339 else 0 for value in row_numbers],
            "TWF": [0] * 10_000,
            "HDF": [0] * 10_000,
            "PWF": [0] * 10_000,
            "OSF": [0] * 10_000,
            "RNF": [0] * 10_000,
        }
    )


def _prepare_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_csv = raw_dir / "source.csv"
    _full_source_frame().to_csv(source_csv, index=False)
    paths = DatasetPaths(
        archive=raw_dir / "source.zip",
        csv=source_csv,
        metadata=raw_dir / "metadata.json",
    )
    monkeypatch.setattr(splitting, "verify_dataset_files", lambda _: paths)
    return raw_dir


def test_stratified_partitions_are_deterministic_disjoint_and_exhaustive() -> None:
    frame = _small_modeling_frame()

    first_training, first_holdout = create_stratified_partitions(frame)
    second_training, second_holdout = create_stratified_partitions(frame)

    assert len(first_training) == 80
    assert len(first_holdout) == 20
    assert int(first_training[TARGET_COLUMN].sum()) == 8
    assert int(first_holdout[TARGET_COLUMN].sum()) == 2
    training_ids = set(first_training["Rotational speed [rpm]"])
    holdout_ids = set(first_holdout["Rotational speed [rpm]"])
    assert training_ids.isdisjoint(holdout_ids)
    assert training_ids | holdout_ids == set(frame["Rotational speed [rpm]"])
    pd.testing.assert_frame_equal(first_training, second_training)
    pd.testing.assert_frame_equal(first_holdout, second_holdout)


def test_materialized_split_is_canonical_idempotent_and_training_only_loadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = _prepare_source(tmp_path, monkeypatch)
    processed_dir = tmp_path / "processed"
    manifest_path = tmp_path / "split_manifest.json"

    artifacts = materialize_split(raw_dir, processed_dir, manifest_path)
    training = pd.read_csv(artifacts.training_csv)
    holdout = pd.read_csv(artifacts.holdout_csv)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(training) == 8_000
    assert len(holdout) == 2_000
    assert int(training[TARGET_COLUMN].sum()) == 271
    assert int(holdout[TARGET_COLUMN].sum()) == 68
    assert tuple(training.columns) == MODELING_COLUMNS
    assert tuple(holdout.columns) == MODELING_COLUMNS
    assert "UDI" not in training and "TWF" not in training
    assert "generated_at" not in manifest
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")

    mtimes = {
        path: path.stat().st_mtime_ns
        for path in (artifacts.training_csv, artifacts.holdout_csv, manifest_path)
    }
    second_artifacts = materialize_split(raw_dir, processed_dir, manifest_path)
    assert second_artifacts == artifacts
    assert mtimes == {path: path.stat().st_mtime_ns for path in mtimes}

    original_manifest = manifest_path.read_bytes()
    monkeypatch.setattr(splitting.platform, "python_version", lambda: "3.12.99")
    patch_portable_artifacts = materialize_split(raw_dir, processed_dir, manifest_path)
    assert patch_portable_artifacts == artifacts
    assert manifest_path.read_bytes() == original_manifest

    artifacts.holdout_csv.unlink()
    loaded_training, _ = load_training_partition(processed_dir, manifest_path)
    pd.testing.assert_frame_equal(loaded_training, training)
    restored_artifacts = materialize_split(raw_dir, processed_dir, manifest_path)
    assert restored_artifacts.holdout_csv.is_file()


def test_materialized_split_refuses_to_overwrite_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = _prepare_source(tmp_path, monkeypatch)
    processed_dir = tmp_path / "processed"
    manifest_path = tmp_path / "split_manifest.json"
    artifacts = materialize_split(raw_dir, processed_dir, manifest_path)
    artifacts.training_csv.write_text("corrupted\n", encoding="utf-8")

    with pytest.raises(SplitIntegrityError, match="differs"):
        materialize_split(raw_dir, processed_dir, manifest_path)

    assert artifacts.training_csv.read_text(encoding="utf-8") == "corrupted\n"


def test_training_loader_rejects_mutated_contract_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = _prepare_source(tmp_path, monkeypatch)
    processed_dir = tmp_path / "processed"
    manifest_path = tmp_path / "split_manifest.json"
    materialize_split(raw_dir, processed_dir, manifest_path)
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutations = (
        ("source", "filename", "other.csv"),
        ("source", "rows", 9_999),
        ("split", "method", "other_splitter"),
        ("split", "rows_sorted_within_partitions", False),
        ("training", "rows", 7_999),
        ("training", "sha256", "not-a-sha256"),
        ("holdout", "filename", "other.csv"),
        ("holdout", "rows", 1_999),
        ("holdout", "columns", [TARGET_COLUMN]),
        ("holdout", "sha256", "not-a-sha256"),
        ("holdout", "access_policy", "Inspect freely."),
        ("versions", "python", "3.11.9"),
    )

    for section, field, bad_value in mutations:
        mutated = json.loads(json.dumps(original))
        mutated[section][field] = bad_value
        manifest_path.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(SplitIntegrityError):
            load_training_partition(processed_dir, manifest_path)

    mutated = json.loads(json.dumps(original))
    mutated["holdout"] = {"unexpected": "metadata"}
    manifest_path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(SplitIntegrityError):
        load_training_partition(processed_dir, manifest_path)

    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    loaded_training, _ = load_training_partition(processed_dir, manifest_path)
    assert len(loaded_training) == 8_000
