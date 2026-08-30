"""Tests for the frozen training-only M3 selection protocol."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

import predictive_maintenance.modeling as modeling
from predictive_maintenance.modeling import (
    BaselineGateError,
    ModelingError,
    build_candidate_pipelines,
    choose_candidate,
    cv_plan_sha256,
    evaluate_candidates,
    generate_oof_probabilities,
    make_cv_plan,
    run_training_selection,
    select_threshold,
)
from predictive_maintenance.validation import FEATURE_COLUMNS, MODELING_COLUMNS, TARGET_COLUMN


def modeling_frame(row_count: int = 150) -> pd.DataFrame:
    """Create a valid small frame with a strong but non-leaking operational signal."""
    target = np.array([1 if index % 5 == 0 else 0 for index in range(row_count)], dtype=int)
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


def small_candidate_pipelines() -> dict[str, Pipeline]:
    pipelines = build_candidate_pipelines()
    pipelines["random_forest"].set_params(model__n_estimators=8, model__max_depth=4)
    return pipelines


def test_candidate_pipelines_use_exact_allowlist_and_fixed_parameters() -> None:
    pipelines = build_candidate_pipelines()

    assert tuple(pipelines) == ("dummy", "logistic_regression", "random_forest")
    for pipeline in pipelines.values():
        assert isinstance(pipeline, Pipeline)
        transformer = pipeline.named_steps["preprocess"]
        assert transformer.remainder == "drop"
        declared_columns = tuple(
            column for _, _, columns in transformer.transformers for column in columns
        )
        assert declared_columns == FEATURE_COLUMNS
        assert TARGET_COLUMN not in declared_columns
        assert not {"UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"}.intersection(
            declared_columns
        )
    assert pipelines["logistic_regression"].named_steps["model"].class_weight == "balanced"
    forest = pipelines["random_forest"].named_steps["model"]
    assert forest.n_estimators == 300
    assert forest.max_depth == 8
    assert forest.n_jobs == 1


def test_cv_plan_is_deterministic_exhaustive_and_stratified() -> None:
    target = modeling_frame()[TARGET_COLUMN]
    first = make_cv_plan(target)
    second = make_cv_plan(target)

    assert len(first) == 5
    assert cv_plan_sha256(first, len(target)) == cv_plan_sha256(second, len(target))
    validation_rows = np.concatenate([validation for _, validation in first])
    assert sorted(validation_rows.tolist()) == list(range(len(target)))
    for (first_train, first_valid), (second_train, second_valid) in zip(first, second, strict=True):
        assert np.array_equal(first_train, second_train)
        assert np.array_equal(first_valid, second_valid)
        assert set(first_train).isdisjoint(first_valid)
        assert set(target.iloc[first_valid]) == {0, 1}


def test_preprocessing_is_fitted_inside_each_fold() -> None:
    frame = modeling_frame()
    features = frame.loc[:, FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]
    plan = make_cv_plan(target)
    pipeline = small_candidate_pipelines()["logistic_regression"]

    result = cross_validate(
        pipeline,
        features,
        target,
        cv=plan,
        scoring="average_precision",
        return_estimator=True,
        error_score="raise",
    )

    global_mean = features.loc[:, FEATURE_COLUMNS[1:]].mean().to_numpy()
    saw_difference_from_global = False
    for estimator, (training_indices, _) in zip(result["estimator"], plan, strict=True):
        scaler_mean = estimator.named_steps["preprocess"].named_transformers_["numeric"].mean_
        fold_mean = features.iloc[training_indices].loc[:, FEATURE_COLUMNS[1:]].mean().to_numpy()
        assert scaler_mean == pytest.approx(fold_mean)
        saw_difference_from_global |= not np.allclose(scaler_mean, global_mean)
    assert saw_difference_from_global


def test_candidate_evaluation_and_oof_use_shared_plan() -> None:
    frame = modeling_frame()
    features = frame.loc[:, FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]
    plan = make_cv_plan(target)
    pipelines = small_candidate_pipelines()

    results = evaluate_candidates(features, target, plan, pipelines)
    assert tuple(results) == ("dummy", "logistic_regression", "random_forest")
    assert all(len(result["average_precision_by_fold"]) == 5 for result in results.values())
    assert "average_precision_delta_vs_dummy_by_fold" in results["logistic_regression"]

    probabilities = generate_oof_probabilities(
        pipelines["logistic_regression"], features, target, plan
    )
    assert probabilities.shape == (len(frame),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_candidate_selection_tie_and_strict_dummy_gate() -> None:
    assert (
        choose_candidate({"dummy": 0.1, "logistic_regression": 0.3, "random_forest": 0.3 + 5e-13})
        == "logistic_regression"
    )
    assert (
        choose_candidate({"dummy": 0.1, "logistic_regression": 0.3, "random_forest": 0.3 + 2e-12})
        == "random_forest"
    )
    with pytest.raises(BaselineGateError):
        choose_candidate({"dummy": 0.3, "logistic_regression": 0.3, "random_forest": 0.2})
    assert (
        choose_candidate(
            {
                "dummy": 0.3,
                "logistic_regression": np.nextafter(0.3, 1.0),
                "random_forest": 0.2,
            }
        )
        == "logistic_regression"
    )


def test_threshold_uses_both_tie_breakers_and_inclusive_decision() -> None:
    balance_tie = select_threshold(
        np.array([0, 1, 0, 0, 0, 1]),
        np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    )
    assert balance_tie["value"] == 0.8

    lower_threshold_tie = select_threshold(
        np.array([1, 0, 0, 1]),
        np.array([0.9, 0.8, 0.7, 0.6]),
    )
    assert lower_threshold_tie["value"] == 0.6
    assert lower_threshold_tie["oof_metrics"]["tp"] == 2

    with pytest.raises(ModelingError):
        select_threshold(np.array([0, 1]), np.array([0.1, np.nan]))
    with pytest.raises(ModelingError):
        make_cv_plan(pd.Series([0, 0, 0, 0, 1, 1, 1, 1]))


@pytest.mark.parametrize(
    "invalid_target",
    [
        pytest.param(
            np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1, 1]),
            id="fractional",
        ),
        pytest.param(
            np.array(["0", "0", "0", "0", "0", "1", "1", "1", "1", "1"]),
            id="strings",
        ),
        pytest.param(
            np.array([0, 0, 0, 0, np.nan, 1, 1, 1, 1, 1]),
            id="nan",
        ),
    ],
)
def test_modeling_rejects_invalid_targets_before_integer_conversion(
    invalid_target: np.ndarray,
) -> None:
    scores = np.linspace(0.05, 0.95, len(invalid_target))

    with pytest.raises(ModelingError):
        make_cv_plan(pd.Series(invalid_target))
    with pytest.raises(ModelingError):
        select_threshold(invalid_target, scores)


def test_training_selection_is_holdout_independent_idempotent_and_serializable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = modeling_frame()
    manifest = {
        "source": {"sha256": "a" * 64},
        "split": {"random_seed": 42},
        "training": {"filename": "train.csv", "rows": len(frame), "sha256": "b" * 64},
        "holdout": {"filename": "holdout.csv", "rows": 30, "sha256": "c" * 64},
    }
    monkeypatch.setattr(modeling, "load_training_partition", lambda *_: (frame.copy(), manifest))
    monkeypatch.setattr(modeling, "build_candidate_pipelines", small_candidate_pipelines)
    processed_dir = tmp_path / "processed"
    report_dir = tmp_path / "reports"
    artifact_dir = tmp_path / "artifacts"
    assert not (processed_dir / "holdout.csv").exists()

    first = run_training_selection(
        processed_dir,
        tmp_path / "split.json",
        report_dir,
        artifact_dir,
    )
    report_bytes = {
        path: path.read_bytes()
        for path in (
            *first.figures,
            first.cv_results,
            first.threshold_selection,
            first.run_manifest,
        )
    }
    second = run_training_selection(
        processed_dir,
        tmp_path / "split.json",
        report_dir,
        artifact_dir,
    )

    assert second == first
    assert not (processed_dir / "holdout.csv").exists()
    assert report_bytes == {path: path.read_bytes() for path in report_bytes}
    selection_report = first.selection_report.read_text(encoding="utf-8")
    assert "# M3 selection on training data" in selection_report
    assert "Logistic regression" in selection_report
    run_manifest = json.loads(first.run_manifest.read_text(encoding="utf-8"))
    assert run_manifest["baseline_gate_passed"] is True
    assert run_manifest["holdout_accessed_during_selection"] is False
    restored = joblib.load(first.pipeline)
    probabilities = restored.predict_proba(frame.loc[:, FEATURE_COLUMNS])[:, 1]
    assert np.isfinite(probabilities).all()
