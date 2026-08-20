"""Tests for intervals derived from reviewed aggregate counts only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from predictive_maintenance.uncertainty import (
    classification_wilson_intervals,
    wilson_score_interval,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_RECEIPT = PROJECT_ROOT / "reports" / "modeling" / "b15bab7b54bc2e1f" / "final_evaluation.json"


def test_final_wilson_intervals_are_derived_from_the_versioned_matrix_counts() -> None:
    receipt = json.loads(FINAL_RECEIPT.read_text(encoding="utf-8"))
    raw_matrix = receipt["metrics"]["confusion_matrix"]
    matrix = (
        (raw_matrix[0][0], raw_matrix[0][1]),
        (raw_matrix[1][0], raw_matrix[1][1]),
    )

    precision_interval, recall_interval = classification_wilson_intervals(matrix)

    assert receipt["receipt"] == "final_holdout_evaluation_complete"
    assert receipt["holdout"]["rows"] == sum(sum(row) for row in matrix)
    assert matrix == ((1897, 35), (18, 50))
    assert precision_interval == pytest.approx(
        (0.4820101461448797, 0.6868299449467584),
        abs=1e-15,
    )
    assert recall_interval == pytest.approx(
        (0.619922660101109, 0.825502593301211),
        abs=1e-15,
    )


@pytest.mark.parametrize(
    ("successes", "trials", "expected"),
    [
        (0, 10, (0.0, 0.2775327998628892)),
        (10, 10, (0.7224672001371107, 1.0)),
    ],
)
def test_wilson_interval_handles_boundary_proportions(
    successes: int,
    trials: int,
    expected: tuple[float, float],
) -> None:
    assert wilson_score_interval(successes, trials) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize(
    ("successes", "trials", "error"),
    [
        (True, 10, TypeError),
        (1, False, TypeError),
        (1.0, 10, TypeError),
        (1, 10.0, TypeError),
        (0, 0, ValueError),
        (-1, 10, ValueError),
        (11, 10, ValueError),
    ],
)
def test_wilson_interval_rejects_invalid_counts(
    successes: object,
    trials: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        wilson_score_interval(successes, trials)  # type: ignore[arg-type]
