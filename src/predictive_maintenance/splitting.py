"""Reproducible train/holdout materialization without inspecting holdout outcomes."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from predictive_maintenance.config import (
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_SPLIT_MANIFEST_PATH,
    HOLDOUT_FILENAME,
    HOLDOUT_FRACTION,
    RANDOM_SEED,
    TRAIN_FILENAME,
)
from predictive_maintenance.dataset import (
    DEFAULT_RAW_DATA_DIR,
    OFFICIAL_DATASET,
    sha256_file,
    verify_dataset_files,
)
from predictive_maintenance.validation import (
    EXPECTED_ROW_COUNT,
    MODELING_COLUMNS,
    TARGET_COLUMN,
    read_dataset,
    select_modeling_data,
    validate_dataset,
    validate_modeling_partition,
)

SPLIT_MANIFEST_SCHEMA_VERSION = 1
SPLIT_METHOD = "sklearn.model_selection.train_test_split"
HOLDOUT_ACCESS_POLICY = "Do not inspect until the final M3 evaluation."
EXPECTED_HOLDOUT_ROWS = round(EXPECTED_ROW_COUNT * HOLDOUT_FRACTION)
EXPECTED_TRAINING_ROWS = EXPECTED_ROW_COUNT - EXPECTED_HOLDOUT_ROWS
MANIFEST_TOP_LEVEL_KEYS = {
    "schema_version",
    "source",
    "split",
    "training",
    "holdout",
    "versions",
}


class SplitIntegrityError(RuntimeError):
    """Raised when a materialized split differs from the reviewed configuration."""


@dataclass(frozen=True)
class SplitArtifacts:
    """Paths and stable metadata for a materialized split."""

    training_csv: Path
    holdout_csv: Path
    manifest: Path
    training_rows: int
    holdout_rows: int
    training_sha256: str
    holdout_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary without holdout target statistics."""
        return {
            "training_csv": str(self.training_csv),
            "holdout_csv": str(self.holdout_csv),
            "manifest": str(self.manifest),
            "training_rows": self.training_rows,
            "holdout_rows": self.holdout_rows,
            "training_sha256": self.training_sha256,
            "holdout_sha256": self.holdout_sha256,
        }


def create_stratified_partitions(
    frame: pd.DataFrame,
    *,
    holdout_fraction: float = HOLDOUT_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split row positions reproducibly and keep original order inside each partition."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1.")
    if TARGET_COLUMN not in frame:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    positions = list(range(len(frame)))
    training_positions, holdout_positions = train_test_split(
        positions,
        test_size=holdout_fraction,
        random_state=random_seed,
        shuffle=True,
        stratify=frame[TARGET_COLUMN],
    )
    training_positions = sorted(training_positions)
    holdout_positions = sorted(holdout_positions)

    if set(training_positions).intersection(holdout_positions):
        raise SplitIntegrityError("Training and holdout positions overlap.")
    if sorted((*training_positions, *holdout_positions)) != positions:
        raise SplitIntegrityError("Training and holdout positions do not cover the source.")

    training = frame.iloc[training_positions].reset_index(drop=True)
    holdout = frame.iloc[holdout_positions].reset_index(drop=True)
    return training, holdout


def _serialize_csv(frame: pd.DataFrame, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=".split-",
        suffix=".csv",
        dir=directory,
        delete=False,
    ) as temporary_file:
        frame.to_csv(temporary_file, index=False, lineterminator="\n")
        return Path(temporary_file.name)


def _expected_manifest(
    training_sha256: str,
    holdout_sha256: str,
    training_rows: int,
    holdout_rows: int,
) -> dict[str, Any]:
    return {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "source": {
            "filename": OFFICIAL_DATASET.csv_filename,
            "sha256": OFFICIAL_DATASET.csv_sha256,
            "rows": EXPECTED_ROW_COUNT,
        },
        "split": {
            "method": SPLIT_METHOD,
            "random_seed": RANDOM_SEED,
            "holdout_fraction": HOLDOUT_FRACTION,
            "stratified_by": TARGET_COLUMN,
            "rows_sorted_within_partitions": True,
        },
        "training": {
            "filename": TRAIN_FILENAME,
            "rows": training_rows,
            "sha256": training_sha256,
            "columns": list(MODELING_COLUMNS),
        },
        "holdout": {
            "filename": HOLDOUT_FILENAME,
            "rows": holdout_rows,
            "sha256": holdout_sha256,
            "columns": list(MODELING_COLUMNS),
            "access_policy": HOLDOUT_ACCESS_POLICY,
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SplitIntegrityError(f"Invalid split manifest: {path}") from error
    if not isinstance(payload, dict):
        raise SplitIntegrityError(f"Invalid split manifest structure: {path}")
    return payload


def _validate_manifest_contract(manifest: dict[str, Any], manifest_path: Path) -> None:
    """Validate invariant metadata while treating runtime patch versions as provenance."""
    if set(manifest) != MANIFEST_TOP_LEVEL_KEYS:
        raise SplitIntegrityError(f"Unexpected split manifest fields: {manifest_path}")
    if manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise SplitIntegrityError("Split manifest uses an unexpected schema version.")

    expected_source = {
        "filename": OFFICIAL_DATASET.csv_filename,
        "sha256": OFFICIAL_DATASET.csv_sha256,
        "rows": EXPECTED_ROW_COUNT,
    }
    if manifest.get("source") != expected_source:
        raise SplitIntegrityError("Split manifest references unexpected source metadata.")

    expected_split = {
        "method": SPLIT_METHOD,
        "random_seed": RANDOM_SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "stratified_by": TARGET_COLUMN,
        "rows_sorted_within_partitions": True,
    }
    if manifest.get("split") != expected_split:
        raise SplitIntegrityError("Split manifest contains unexpected partition metadata.")

    versions = manifest.get("versions")
    expected_version_keys = {"python", "pandas", "scikit_learn"}
    if not isinstance(versions, dict) or set(versions) != expected_version_keys:
        raise SplitIntegrityError("Split manifest contains invalid version metadata.")
    if any(not isinstance(value, str) or not value.strip() for value in versions.values()):
        raise SplitIntegrityError("Split manifest contains invalid version metadata.")
    python_parts = versions["python"].split(".")
    if len(python_parts) < 2 or python_parts[:2] != ["3", "12"]:
        raise SplitIntegrityError("Split manifest was not produced with Python 3.12.")

    _validate_partition_metadata(
        manifest.get("training"),
        label="training",
        expected_filename=TRAIN_FILENAME,
        expected_rows=EXPECTED_TRAINING_ROWS,
        include_access_policy=False,
    )
    _validate_partition_metadata(
        manifest.get("holdout"),
        label="holdout",
        expected_filename=HOLDOUT_FILENAME,
        expected_rows=EXPECTED_HOLDOUT_ROWS,
        include_access_policy=True,
    )


def _validate_partition_metadata(
    metadata: object,
    *,
    label: str,
    expected_filename: str,
    expected_rows: int,
    include_access_policy: bool,
) -> None:
    expected_keys = {"filename", "rows", "sha256", "columns"}
    if include_access_policy:
        expected_keys.add("access_policy")
    if not isinstance(metadata, dict) or set(metadata) != expected_keys:
        raise SplitIntegrityError(f"Invalid {label} metadata in split manifest.")
    if metadata.get("filename") != expected_filename or metadata.get("rows") != expected_rows:
        raise SplitIntegrityError(f"Unexpected {label} identity in split manifest.")
    if metadata.get("columns") != list(MODELING_COLUMNS):
        raise SplitIntegrityError(f"Unexpected {label} columns in split manifest.")
    checksum = metadata.get("sha256")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise SplitIntegrityError(f"Invalid {label} SHA-256 in split manifest.")
    if include_access_policy and metadata.get("access_policy") != HOLDOUT_ACCESS_POLICY:
        raise SplitIntegrityError("Unexpected holdout access policy in split manifest.")


def _manifest_invariants(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return fields that must match independently of informative runtime versions."""
    return {key: value for key, value in manifest.items() if key != "versions"}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=".split-manifest-",
        suffix=".json",
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        json.dump(payload, temporary_file, ensure_ascii=False, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _verify_existing_file(path: Path, expected_sha256: str) -> None:
    if not path.exists():
        return
    if not path.is_file():
        raise SplitIntegrityError(f"Expected a regular split file: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise SplitIntegrityError(
            f"Existing split file differs from the reviewed manifest: {path}; "
            f"expected {expected_sha256}, got {actual_sha256}."
        )


def materialize_split(
    raw_dir: Path = DEFAULT_RAW_DATA_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DATA_DIR,
    manifest_path: Path = DEFAULT_SPLIT_MANIFEST_PATH,
) -> SplitArtifacts:
    """Create or verify the fixed split and its versionable manifest."""
    source_paths = verify_dataset_files(raw_dir)
    source_frame = read_dataset(source_paths.csv)
    validate_dataset(source_frame)
    features, target = select_modeling_data(source_frame)
    modeling_frame = pd.concat([features, target], axis="columns")
    training, holdout = create_stratified_partitions(modeling_frame)

    temporary_training = _serialize_csv(training, processed_dir)
    temporary_holdout = _serialize_csv(holdout, processed_dir)
    created_paths: list[Path] = []
    try:
        training_sha256 = sha256_file(temporary_training)
        holdout_sha256 = sha256_file(temporary_holdout)
        expected_manifest = _expected_manifest(
            training_sha256,
            holdout_sha256,
            len(training),
            len(holdout),
        )

        if manifest_path.exists():
            existing_manifest = _load_json(manifest_path)
            _validate_manifest_contract(existing_manifest, manifest_path)
            if _manifest_invariants(existing_manifest) != _manifest_invariants(expected_manifest):
                raise SplitIntegrityError(
                    f"Existing manifest differs from the configured split: {manifest_path}"
                )

        training_path = processed_dir / TRAIN_FILENAME
        holdout_path = processed_dir / HOLDOUT_FILENAME
        _verify_existing_file(training_path, training_sha256)
        _verify_existing_file(holdout_path, holdout_sha256)

        if training_path.exists():
            temporary_training.unlink()
        else:
            temporary_training.replace(training_path)
            created_paths.append(training_path)
        if holdout_path.exists():
            temporary_holdout.unlink()
        else:
            temporary_holdout.replace(holdout_path)
            created_paths.append(holdout_path)
        if not manifest_path.exists():
            _write_json_atomic(manifest_path, expected_manifest)
            created_paths.append(manifest_path)

        return SplitArtifacts(
            training_csv=training_path,
            holdout_csv=holdout_path,
            manifest=manifest_path,
            training_rows=len(training),
            holdout_rows=len(holdout),
            training_sha256=training_sha256,
            holdout_sha256=holdout_sha256,
        )
    except Exception:
        for created_path in reversed(created_paths):
            created_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_training.unlink(missing_ok=True)
        temporary_holdout.unlink(missing_ok=True)


def load_training_partition(
    processed_dir: Path = DEFAULT_PROCESSED_DATA_DIR,
    manifest_path: Path = DEFAULT_SPLIT_MANIFEST_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only training after verifying its manifest; never open the holdout file."""
    manifest = _load_json(manifest_path)
    _validate_manifest_contract(manifest, manifest_path)
    training_metadata = manifest.get("training")
    assert isinstance(training_metadata, dict)

    training_filename = training_metadata.get("filename")
    if not isinstance(training_filename, str) or Path(training_filename).name != training_filename:
        raise SplitIntegrityError("Invalid training filename in split manifest.")
    training_path = processed_dir / training_filename
    expected_sha256 = training_metadata.get("sha256")
    expected_rows = training_metadata.get("rows")
    if not isinstance(expected_sha256, str) or not isinstance(expected_rows, int):
        raise SplitIntegrityError("Invalid training metadata in split manifest.")
    if training_metadata.get("columns") != list(MODELING_COLUMNS):
        raise SplitIntegrityError("Split manifest contains unexpected modeling columns.")
    _verify_existing_file(training_path, expected_sha256)
    if not training_path.is_file():
        raise SplitIntegrityError(f"Missing materialized training partition: {training_path}")

    training_frame = read_dataset(training_path)
    validate_modeling_partition(training_frame, expected_row_count=expected_rows)
    return training_frame, manifest


def load_holdout_partition_for_final_evaluation(
    processed_dir: Path = DEFAULT_PROCESSED_DATA_DIR,
    manifest_path: Path = DEFAULT_SPLIT_MANIFEST_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read holdout bytes once for the explicitly authorized final M3 evaluation."""
    manifest = _load_json(manifest_path)
    _validate_manifest_contract(manifest, manifest_path)
    holdout_metadata = manifest["holdout"]
    assert isinstance(holdout_metadata, dict)
    holdout_path = processed_dir / HOLDOUT_FILENAME
    if not holdout_path.is_file():
        raise SplitIntegrityError(f"Missing materialized holdout partition: {holdout_path}")

    try:
        content = holdout_path.read_bytes()
    except OSError as error:
        raise SplitIntegrityError(
            f"Could not read final holdout partition: {holdout_path}"
        ) from error
    actual_sha256 = hashlib.sha256(content).hexdigest()
    expected_sha256 = holdout_metadata["sha256"]
    if actual_sha256 != expected_sha256:
        raise SplitIntegrityError(
            "Final holdout differs from the reviewed manifest; "
            f"expected {expected_sha256}, got {actual_sha256}."
        )
    try:
        holdout_frame = pd.read_csv(io.BytesIO(content))
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise SplitIntegrityError("Could not parse the final holdout partition.") from error
    validate_modeling_partition(holdout_frame, expected_row_count=EXPECTED_HOLDOUT_ROWS)
    return holdout_frame, manifest
