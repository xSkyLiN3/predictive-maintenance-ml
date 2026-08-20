"""Regression tests for schema validation and leakage prevention."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from predictive_maintenance.validation import (
    EXCLUDED_COLUMNS,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    MODELING_COLUMNS,
    TARGET_COLUMN,
    DataValidationError,
    UnsafeFeatureError,
    read_dataset,
    select_modeling_data,
    validate_dataset,
    validate_feature_columns,
    validate_modeling_partition,
)


def test_valid_dataset_returns_a_compact_summary(valid_frame: pd.DataFrame) -> None:
    summary = validate_dataset(valid_frame, expected_row_count=3)

    assert summary.row_count == 3
    assert summary.column_count == 14
    assert summary.missing_cells == 0
    assert summary.duplicate_rows == 0
    assert summary.duplicate_feature_rows == 0
    assert summary.type_counts == {"H": 1, "L": 1, "M": 1}
    assert summary.target_counts == {0: 2, 1: 1}
    assert summary.failure_mode_target_disagreements == 0


def test_read_dataset_round_trip(tmp_path: Path, valid_frame: pd.DataFrame) -> None:
    csv_path = tmp_path / "dataset.csv"
    valid_frame.to_csv(csv_path, index=False)

    loaded_frame = read_dataset(csv_path)

    validate_dataset(loaded_frame, expected_row_count=3)
    pd.testing.assert_frame_equal(loaded_frame, valid_frame)


def test_read_dataset_rejects_an_empty_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(DataValidationError, match="Could not parse"):
        read_dataset(csv_path)


@pytest.mark.parametrize("excluded_column", EXCLUDED_COLUMNS)
def test_feature_allowlist_rejects_identifiers_and_failure_modes(excluded_column: str) -> None:
    with pytest.raises(UnsafeFeatureError, match="forbidden"):
        validate_feature_columns((*FEATURE_COLUMNS, excluded_column))


def test_feature_allowlist_rejects_target() -> None:
    with pytest.raises(UnsafeFeatureError, match="Target column"):
        validate_feature_columns((*FEATURE_COLUMNS, TARGET_COLUMN))


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (FEATURE_COLUMNS[:-1], "missing"),
        ((*FEATURE_COLUMNS, "unexpected"), "Unknown"),
        ((*FEATURE_COLUMNS, FEATURE_COLUMNS[0]), "duplicated"),
        (tuple(reversed(FEATURE_COLUMNS)), "declared order"),
    ],
)
def test_feature_allowlist_rejects_any_deviation(
    columns: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(UnsafeFeatureError, match=message):
        validate_feature_columns(columns)


def test_modeling_selector_uses_only_allowlisted_features(valid_frame: pd.DataFrame) -> None:
    original = valid_frame.copy(deep=True)

    features, target = select_modeling_data(valid_frame, expected_row_count=3)

    assert tuple(features.columns) == FEATURE_COLUMNS
    assert set(features.columns).isdisjoint(EXCLUDED_COLUMNS)
    assert target.name == TARGET_COLUMN
    pd.testing.assert_frame_equal(valid_frame, original)


def test_modeling_partition_rejects_any_leakage_column(valid_frame: pd.DataFrame) -> None:
    features, target = select_modeling_data(valid_frame, expected_row_count=3)
    modeling_frame = pd.concat([features, target], axis="columns")
    summary = validate_modeling_partition(modeling_frame, expected_row_count=3)
    assert tuple(modeling_frame.columns) == MODELING_COLUMNS
    assert summary.target_counts == {0: 2, 1: 1}

    leaked_frame = modeling_frame.assign(TWF=valid_frame["TWF"])
    with pytest.raises(DataValidationError, match="Modeling columns must be exactly"):
        validate_modeling_partition(leaked_frame, expected_row_count=3)


def test_target_failure_mode_disagreement_is_reported_but_not_rewritten(
    valid_frame: pd.DataFrame,
) -> None:
    valid_frame.loc[0, "RNF"] = 1

    summary = validate_dataset(valid_frame, expected_row_count=3)

    assert summary.failure_mode_target_disagreements == 1
    assert valid_frame.loc[0, TARGET_COLUMN] == 0


def test_repeated_operational_features_are_reported_without_rejecting_source(
    valid_frame: pd.DataFrame,
) -> None:
    repeated_row = valid_frame.iloc[[0]].copy()
    repeated_row.loc[:, "UDI"] = 4
    repeated_row.loc[:, "Product ID"] = "L00004"
    frame_with_repeated_observation = pd.concat(
        [valid_frame, repeated_row],
        ignore_index=True,
    )

    summary = validate_dataset(frame_with_repeated_observation, expected_row_count=4)

    assert summary.duplicate_rows == 0
    assert summary.duplicate_feature_rows == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns=["Torque [Nm]"]), "Missing columns"),
        (lambda frame: frame.assign(unexpected=1), "Unexpected columns"),
        (lambda frame: frame.assign(Type=["L", "M", "X"]), "must contain exactly"),
        (
            lambda frame: frame.assign(**{"Air temperature [K]": [295.3, 300.0, 999.0]}),
            "must stay within",
        ),
        (lambda frame: frame.assign(**{"Machine failure": [0, 2, 0]}), "must be binary"),
        (lambda frame: frame.assign(**{"Machine failure": [0, 0, 0]}), "both classes"),
        (lambda frame: frame.assign(RNF=[0, 2, 0]), "must be binary"),
        (lambda frame: frame.assign(UDI=[1, 1, 3]), "must be unique"),
        (lambda frame: frame.assign(UDI=[0, 2, 3]), "must be positive"),
        (lambda frame: frame.assign(UDI=["1", "2", "3"]), "must be integer"),
        (
            lambda frame: frame.assign(**{"Product ID": ["L00001", "L00001", "H00003"]}),
            "must be unique",
        ),
        (
            lambda frame: frame.assign(**{"Product ID": ["L0000X", "M00002", "H00003"]}),
            "must match",
        ),
        (
            lambda frame: frame.assign(**{"Product ID": ["M00001", "M00002", "H00003"]}),
            "first character",
        ),
        (lambda frame: frame.assign(Type=["L", "", "H"]), "empty strings"),
        (lambda frame: frame.assign(**{"Torque [Nm]": [4, 40, 77]}), "floating point"),
        (lambda frame: frame.loc[:, tuple(reversed(EXPECTED_COLUMNS))], "official order"),
    ],
)
def test_dataset_contract_rejects_invalid_values(
    valid_frame: pd.DataFrame,
    mutation: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    invalid_frame = mutation(valid_frame)

    with pytest.raises(DataValidationError, match=message):
        validate_dataset(invalid_frame, expected_row_count=3)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("Air temperature [K]", 294.9),
        ("Process temperature [K]", 315.1),
        ("Rotational speed [rpm]", 999),
        ("Torque [Nm]", 80.1),
        ("Tool wear [min]", 261),
    ],
)
def test_dataset_contract_enforces_each_source_guardrail(
    valid_frame: pd.DataFrame,
    column: str,
    invalid_value: float,
) -> None:
    invalid_frame = valid_frame.copy()
    invalid_frame.loc[1, column] = invalid_value

    with pytest.raises(DataValidationError, match="must stay within"):
        validate_dataset(invalid_frame, expected_row_count=3)


def test_dataset_contract_rejects_wrong_row_count(valid_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="Expected 4 rows"):
        validate_dataset(valid_frame, expected_row_count=4)


def test_dataset_contract_rejects_a_fully_duplicated_row(valid_frame: pd.DataFrame) -> None:
    duplicated_frame = pd.concat([valid_frame, valid_frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(DataValidationError, match="duplicated rows"):
        validate_dataset(duplicated_frame, expected_row_count=4)


def test_dataset_contract_rejects_missing_and_infinite_values(
    valid_frame: pd.DataFrame,
) -> None:
    invalid_frame = valid_frame.copy()
    invalid_frame.loc[0, "Torque [Nm]"] = float("nan")
    invalid_frame.loc[1, "Air temperature [K]"] = float("inf")

    with pytest.raises(DataValidationError) as error:
        validate_dataset(invalid_frame, expected_row_count=3)

    assert "missing cells" in str(error.value)
    assert "infinite values" in str(error.value)
