"""Executable data contract and leakage-safe feature selection for AI4I 2020."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_string_dtype

EXPECTED_ROW_COUNT = 10_000
TARGET_COLUMN = "Machine failure"
IDENTIFIER_COLUMNS = ("UDI", "Product ID")
FEATURE_COLUMNS = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
FAILURE_MODE_COLUMNS = ("TWF", "HDF", "PWF", "OSF", "RNF")
EXCLUDED_COLUMNS = (*IDENTIFIER_COLUMNS, *FAILURE_MODE_COLUMNS)
NON_FEATURE_COLUMNS = (*EXCLUDED_COLUMNS, TARGET_COLUMN)
MODELING_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)
EXPECTED_COLUMNS = (
    *IDENTIFIER_COLUMNS,
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
    *FAILURE_MODE_COLUMNS,
)

STRING_COLUMNS = ("Product ID", "Type")
FLOAT_COLUMNS = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Torque [Nm]",
)
INTEGER_COLUMNS = (
    "UDI",
    "Rotational speed [rpm]",
    "Tool wear [min]",
    TARGET_COLUMN,
    *FAILURE_MODE_COLUMNS,
)
BINARY_COLUMNS = (TARGET_COLUMN, *FAILURE_MODE_COLUMNS)
EXPECTED_TYPE_CATEGORIES = frozenset({"L", "M", "H"})
SOURCE_GUARDRAIL_RANGES: dict[str, tuple[float, float]] = {
    "Air temperature [K]": (295.0, 305.0),
    "Process temperature [K]": (305.0, 315.0),
    "Rotational speed [rpm]": (1000.0, 3000.0),
    "Torque [Nm]": (0.0, 80.0),
    "Tool wear [min]": (0.0, 260.0),
}


class DataValidationError(ValueError):
    """Raised with all detected violations of the AI4I data contract."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("Data contract violations:\n- " + "\n- ".join(self.errors))


class UnsafeFeatureError(ValueError):
    """Raised when a proposed feature set violates the strict allowlist."""


@dataclass(frozen=True)
class DatasetSummary:
    """Small, serializable validation result without performing EDA."""

    row_count: int
    column_count: int
    missing_cells: int
    duplicate_rows: int
    duplicate_feature_rows: int
    type_counts: dict[str, int]
    target_counts: dict[int, int]
    failure_mode_target_disagreements: int
    observed_numeric_ranges: dict[str, dict[str, float]]
    feature_columns: tuple[str, ...]
    target_column: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class ModelingPartitionSummary:
    """Validation result for a leakage-safe modeling partition."""

    row_count: int
    missing_cells: int
    duplicate_feature_rows: int
    type_counts: dict[str, int]
    target_counts: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a snapshot or partition CSV while preserving pandas' inferred dtypes."""
    if not path.is_file():
        raise DataValidationError([f"Dataset CSV does not exist: {path}"])
    try:
        return pd.read_csv(path)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        raise DataValidationError([f"Could not parse dataset CSV {path}: {error}"]) from error


def _validate_columns(frame: pd.DataFrame, errors: list[str]) -> bool:
    actual_columns = tuple(str(column) for column in frame.columns)
    if actual_columns == EXPECTED_COLUMNS:
        return True

    missing = sorted(set(EXPECTED_COLUMNS) - set(actual_columns))
    unexpected = sorted(set(actual_columns) - set(EXPECTED_COLUMNS))
    if not frame.columns.is_unique:
        errors.append("Column names must be unique.")
    if missing:
        errors.append(f"Missing columns: {missing}")
    if unexpected:
        errors.append(f"Unexpected columns: {unexpected}")
    if not missing and not unexpected:
        errors.append("Columns are not in the official order.")
    return False


def _validate_dtypes(frame: pd.DataFrame, errors: list[str]) -> None:
    for column in STRING_COLUMNS:
        if column in frame and not is_string_dtype(frame[column].dtype):
            errors.append(f"Column {column!r} must be string, got {frame[column].dtype}.")
    for column in FLOAT_COLUMNS:
        if column in frame and not is_float_dtype(frame[column].dtype):
            errors.append(f"Column {column!r} must be floating point, got {frame[column].dtype}.")
    for column in INTEGER_COLUMNS:
        if column in frame and not is_integer_dtype(frame[column].dtype):
            errors.append(f"Column {column!r} must be integer, got {frame[column].dtype}.")


def _validate_categories_and_binary_values(frame: pd.DataFrame, errors: list[str]) -> None:
    if "Type" in frame:
        actual_types = frozenset(frame["Type"].dropna().astype(str).unique())
        if actual_types != EXPECTED_TYPE_CATEGORIES:
            errors.append(
                "Column 'Type' must contain exactly "
                f"{sorted(EXPECTED_TYPE_CATEGORIES)}, got {sorted(actual_types)}."
            )

    for column in BINARY_COLUMNS:
        if column not in frame:
            continue
        values = frozenset(frame[column].dropna().unique())
        if not values.issubset({0, 1}):
            formatted_values = sorted(repr(value) for value in values)
            errors.append(f"Column {column!r} must be binary, got {formatted_values}.")

    if TARGET_COLUMN in frame:
        target_values = frozenset(frame[TARGET_COLUMN].dropna().unique())
        if target_values != {0, 1}:
            formatted_values = sorted(repr(value) for value in target_values)
            errors.append(f"Target must contain both classes 0 and 1, got {formatted_values}.")


def _validate_ranges(frame: pd.DataFrame, errors: list[str]) -> None:
    for column, (minimum, maximum) in SOURCE_GUARDRAIL_RANGES.items():
        if column not in frame:
            continue
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        out_of_range = numeric_values.notna() & ~numeric_values.between(minimum, maximum)
        if out_of_range.any():
            observed_minimum = numeric_values.min()
            observed_maximum = numeric_values.max()
            errors.append(
                f"Column {column!r} must stay within [{minimum}, {maximum}], "
                f"got [{observed_minimum}, {observed_maximum}]."
            )


def _validate_identifiers(
    frame: pd.DataFrame,
    errors: list[str],
) -> None:
    if "UDI" in frame:
        if frame["UDI"].duplicated().any():
            errors.append("'UDI' must be unique.")
        if is_integer_dtype(frame["UDI"].dtype) and frame["UDI"].le(0).any():
            errors.append("'UDI' values must be positive.")

    if "Product ID" not in frame:
        return
    product_ids = frame["Product ID"]
    if product_ids.duplicated().any():
        errors.append("'Product ID' must be unique.")
    if not product_ids.astype("string").str.fullmatch(r"[LMH]\d{5}", na=False).all():
        errors.append("'Product ID' values must match one type letter followed by five digits.")
    if "Type" in frame:
        type_matches_id = frame["Type"].astype("string").eq(product_ids.astype("string").str[0])
        if not type_matches_id.all():
            errors.append("'Type' must match the first character of 'Product ID'.")


def _build_summary(frame: pd.DataFrame) -> DatasetSummary:
    mode_disagreements = 0
    if all(column in frame for column in (TARGET_COLUMN, *FAILURE_MODE_COLUMNS)):
        derived_failure = frame.loc[:, FAILURE_MODE_COLUMNS].max(axis="columns")
        mode_disagreements = int(frame[TARGET_COLUMN].ne(derived_failure).sum())

    type_counts = {
        str(key): int(value) for key, value in frame["Type"].value_counts().sort_index().items()
    }
    target_counts = {
        int(key): int(value)
        for key, value in frame[TARGET_COLUMN].value_counts().sort_index().items()
    }
    observed_numeric_ranges = {
        column: {
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
        }
        for column in SOURCE_GUARDRAIL_RANGES
    }
    return DatasetSummary(
        row_count=len(frame),
        column_count=len(frame.columns),
        missing_cells=int(frame.isna().sum().sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        duplicate_feature_rows=int(frame.duplicated(subset=FEATURE_COLUMNS).sum()),
        type_counts=type_counts,
        target_counts=target_counts,
        failure_mode_target_disagreements=mode_disagreements,
        observed_numeric_ranges=observed_numeric_ranges,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )


def validate_dataset(
    frame: pd.DataFrame,
    *,
    expected_row_count: int = EXPECTED_ROW_COUNT,
) -> DatasetSummary:
    """Validate schema, dtypes, values, identifiers and the binary target."""
    errors: list[str] = []
    columns_are_valid = _validate_columns(frame, errors)

    if len(frame) != expected_row_count:
        errors.append(f"Expected {expected_row_count} rows, got {len(frame)}.")
    missing_cells = int(frame.isna().sum().sum())
    if missing_cells:
        errors.append(f"Dataset contains {missing_cells} missing cells.")
    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        errors.append(f"Dataset contains {duplicate_rows} duplicated rows.")

    if not columns_are_valid:
        raise DataValidationError(errors)

    for column in STRING_COLUMNS:
        empty_values = int(frame[column].astype("string").str.strip().eq("").sum())
        if empty_values:
            errors.append(f"Column {column!r} contains {empty_values} empty strings.")
    for column in (*FLOAT_COLUMNS, *INTEGER_COLUMNS):
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        infinite_values = int((numeric_values.notna() & ~numeric_values.map(math.isfinite)).sum())
        if infinite_values:
            errors.append(f"Column {column!r} contains {infinite_values} infinite values.")

    _validate_dtypes(frame, errors)
    _validate_categories_and_binary_values(frame, errors)
    _validate_ranges(frame, errors)
    _validate_identifiers(frame, errors)

    if errors:
        raise DataValidationError(errors)
    return _build_summary(frame)


def validate_feature_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """Require the exact six-feature allowlist and reject leakage explicitly."""
    proposed = tuple(columns)
    proposed_set = set(proposed)
    leakage = sorted(proposed_set.intersection(EXCLUDED_COLUMNS))
    target_included = TARGET_COLUMN in proposed_set
    missing = sorted(set(FEATURE_COLUMNS) - proposed_set)
    unknown = sorted(proposed_set - set(FEATURE_COLUMNS) - set(NON_FEATURE_COLUMNS))

    problems: list[str] = []
    if leakage:
        problems.append(f"Identifiers or failure-mode leakage columns are forbidden: {leakage}.")
    if target_included:
        problems.append(f"Target column {TARGET_COLUMN!r} cannot be used as a feature.")
    if missing:
        problems.append(f"Required feature columns are missing: {missing}.")
    if unknown:
        problems.append(f"Unknown feature columns are not allowed: {unknown}.")
    if len(proposed) != len(proposed_set):
        problems.append("Feature columns must not be duplicated.")
    if not problems and proposed != FEATURE_COLUMNS:
        problems.append(f"Feature columns must use the declared order: {list(FEATURE_COLUMNS)}.")

    if problems:
        raise UnsafeFeatureError(" ".join(problems))
    return proposed


def validate_modeling_partition(
    frame: pd.DataFrame,
    *,
    expected_row_count: int,
) -> ModelingPartitionSummary:
    """Validate a materialized partition containing only allowlisted features and target."""
    errors: list[str] = []
    actual_columns = tuple(str(column) for column in frame.columns)
    if actual_columns != MODELING_COLUMNS:
        errors.append(
            f"Modeling columns must be exactly {list(MODELING_COLUMNS)}, "
            f"got {list(actual_columns)}."
        )
        raise DataValidationError(errors)

    validate_feature_columns(actual_columns[:-1])
    if len(frame) != expected_row_count:
        errors.append(f"Expected {expected_row_count} rows, got {len(frame)}.")
    missing_cells = int(frame.isna().sum().sum())
    if missing_cells:
        errors.append(f"Modeling partition contains {missing_cells} missing cells.")

    for column in ("Type",):
        empty_values = int(frame[column].astype("string").str.strip().eq("").sum())
        if empty_values:
            errors.append(f"Column {column!r} contains {empty_values} empty strings.")
    for column in (*FLOAT_COLUMNS, "Rotational speed [rpm]", "Tool wear [min]", TARGET_COLUMN):
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        infinite_values = int((numeric_values.notna() & ~numeric_values.map(math.isfinite)).sum())
        if infinite_values:
            errors.append(f"Column {column!r} contains {infinite_values} infinite values.")

    _validate_dtypes(frame, errors)
    _validate_categories_and_binary_values(frame, errors)
    _validate_ranges(frame, errors)
    if errors:
        raise DataValidationError(errors)

    return ModelingPartitionSummary(
        row_count=len(frame),
        missing_cells=missing_cells,
        duplicate_feature_rows=int(frame.duplicated(subset=FEATURE_COLUMNS).sum()),
        type_counts={
            str(key): int(value) for key, value in frame["Type"].value_counts().sort_index().items()
        },
        target_counts={
            int(key): int(value)
            for key, value in frame[TARGET_COLUMN].value_counts().sort_index().items()
        },
    )


def select_modeling_data(
    frame: pd.DataFrame,
    *,
    expected_row_count: int = EXPECTED_ROW_COUNT,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return only allowlisted features and target after full validation."""
    validate_dataset(frame, expected_row_count=expected_row_count)
    validate_feature_columns(FEATURE_COLUMNS)
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    target = frame.loc[:, TARGET_COLUMN].copy()
    return features, target
