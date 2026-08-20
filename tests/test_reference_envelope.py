"""Regression proof for the training-only abstention-envelope provenance."""

from __future__ import annotations

import json
from pathlib import Path

from predictive_maintenance.schemas import AI4I_TRAINING_REFERENCE_ENVELOPE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_EDA_SUMMARY = PROJECT_ROOT / "reports" / "eda" / "summary.json"

API_TO_TRAINING_COLUMN = {
    "air_temperature_k": "Air temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
}


def test_reference_envelope_matches_only_the_versioned_training_eda() -> None:
    summary = json.loads(TRAINING_EDA_SUMMARY.read_text(encoding="utf-8"))

    assert summary["scope"] == "training_only"
    assert summary["training_rows"] == 8_000
    assert summary["protocol"]["holdout_profiled"] is False
    assert set(AI4I_TRAINING_REFERENCE_ENVELOPE) == set(API_TO_TRAINING_COLUMN)

    for api_field, training_column in API_TO_TRAINING_COLUMN.items():
        minimum, maximum, _unit = AI4I_TRAINING_REFERENCE_ENVELOPE[api_field]
        observed = summary["numeric_summary"][training_column]["overall"]
        assert minimum == observed["min"]
        assert maximum == observed["max"]
