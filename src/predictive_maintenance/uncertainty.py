"""Small, dependency-free uncertainty helpers for reviewed count metrics."""

from __future__ import annotations

import math
from statistics import NormalDist

WILSON_CONFIDENCE_LEVEL = 0.95
WILSON_95_Z = NormalDist().inv_cdf(0.5 + WILSON_CONFIDENCE_LEVEL / 2.0)


def wilson_score_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return the two-sided 95% Wilson score interval for a binomial proportion."""
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, int)
        or not isinstance(trials, int)
    ):
        raise TypeError("Wilson counts must be integers.")
    if trials <= 0:
        raise ValueError("Wilson trials must be positive.")
    if not 0 <= successes <= trials:
        raise ValueError("Wilson successes must be within [0, trials].")

    proportion = successes / trials
    z_squared = WILSON_95_Z**2
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        WILSON_95_Z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def classification_wilson_intervals(
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Derive precision and recall intervals only from ``[[TN, FP], [FN, TP]]``."""
    (_, false_positive), (false_negative, true_positive) = confusion_matrix
    precision_interval = wilson_score_interval(
        true_positive,
        true_positive + false_positive,
    )
    recall_interval = wilson_score_interval(
        true_positive,
        true_positive + false_negative,
    )
    return precision_interval, recall_interval
