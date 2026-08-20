"""Leakage-safe M3 candidate comparison, OOF thresholding and model fitting."""

from __future__ import annotations

import io
import math
import platform
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictive_maintenance.artifact_io import (
    ArtifactIntegrityError,
    canonical_json_bytes,
    load_json_object,
    sha256_bytes,
    sha256_path,
    write_bytes_atomic_if_changed,
    write_json_atomic_if_changed,
)
from predictive_maintenance.config import (
    CV_FOLDS,
    CV_STANDARD_DEVIATION_DDOF,
    DEFAULT_MODEL_ARTIFACT_DIR,
    DEFAULT_MODEL_REPORT_DIR,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_SPLIT_MANIFEST_PATH,
    LOGISTIC_REGRESSION_PARAMETERS,
    PRIMARY_METRIC,
    RANDOM_FOREST_PARAMETERS,
    RANDOM_SEED,
    SECONDARY_METRIC,
    THRESHOLD_STRATEGY,
    TIE_TOLERANCE,
)
from predictive_maintenance.splitting import load_training_partition
from predictive_maintenance.validation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_columns,
)

MODEL_RUN_SCHEMA_VERSION = 1
CANDIDATE_ORDER = ("dummy", "logistic_regression", "random_forest")
MODEL_CANDIDATES = ("logistic_regression", "random_forest")
NUMERIC_FEATURES = FEATURE_COLUMNS[1:]
TYPE_CATEGORIES = ("L", "M", "H")
PIPELINE_FILENAME = "pipeline.joblib"
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
ACTIVE_RUN_FILENAME = "active_run.json"
CV_RESULTS_FILENAME = "cv_results.json"
THRESHOLD_SELECTION_FILENAME = "threshold_selection.json"
SELECTION_REPORT_FILENAME = "SELECTION_REPORT.md"
CV_FIGURE_FILENAME = "01_cv_average_precision.png"
OOF_FIGURE_FILENAME = "02_oof_precision_recall.png"


class ModelingError(RuntimeError):
    """Raised when M3 selection cannot be completed safely."""


class BaselineGateError(ModelingError):
    """Raised when no real candidate beats the dummy in mean CV AP."""


def _validated_binary_target(target: pd.Series | np.ndarray, *, context: str) -> np.ndarray:
    """Validate binary integer values before narrowing their dtype."""
    series = pd.Series(target, copy=False)
    if series.ndim != 1 or series.isna().any() or not pd.api.types.is_integer_dtype(series.dtype):
        raise ModelingError(f"{context} target must be a non-null integer vector.")
    if set(int(value) for value in series.unique()) != {0, 1}:
        raise ModelingError(f"{context} target must contain exactly classes 0 and 1.")
    return series.to_numpy(dtype=np.int8, copy=True)


@dataclass(frozen=True)
class TrainingArtifacts:
    """Local binary and versionable selection outputs for one deterministic run."""

    run_id: str
    selected_model: str
    threshold: float
    pipeline: Path
    artifact_manifest: Path
    run_manifest: Path
    cv_results: Path
    threshold_selection: Path
    selection_report: Path
    figures: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "selected_model": self.selected_model,
            "threshold": self.threshold,
            "pipeline": str(self.pipeline),
            "artifact_manifest": str(self.artifact_manifest),
            "run_manifest": str(self.run_manifest),
            "cv_results": str(self.cv_results),
            "threshold_selection": str(self.threshold_selection),
            "selection_report": str(self.selection_report),
            "figures": [str(path) for path in self.figures],
        }


def modeling_configuration() -> dict[str, Any]:
    """Return the pre-registered, JSON-safe M3 configuration."""
    return {
        "features": list(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "preprocessing": {
            "categorical": {
                "columns": ["Type"],
                "transformer": "OneHotEncoder",
                "categories": [list(TYPE_CATEGORIES)],
                "handle_unknown": "error",
                "sparse_output": False,
            },
            "numeric": {
                "columns": list(NUMERIC_FEATURES),
                "transformer": "StandardScaler",
            },
            "remainder": "drop",
        },
        "candidates": {
            "dummy": {"strategy": "prior"},
            "logistic_regression": dict(LOGISTIC_REGRESSION_PARAMETERS),
            "random_forest": dict(RANDOM_FOREST_PARAMETERS),
        },
        "cv": {
            "class": "StratifiedKFold",
            "n_splits": CV_FOLDS,
            "shuffle": True,
            "random_state": RANDOM_SEED,
            "primary_metric": PRIMARY_METRIC,
            "secondary_metric": SECONDARY_METRIC,
            "standard_deviation_ddof": CV_STANDARD_DEVIATION_DDOF,
        },
        "selection": {
            "candidate_tie_tolerance": TIE_TOLERANCE,
            "candidate_tie_winner": "logistic_regression",
            "dummy_gate": "selected_mean_ap > dummy_mean_ap",
        },
        "threshold": {
            "strategy": THRESHOLD_STRATEGY,
            "score": "predict_proba[:, 1]",
            "decision_rule": "score >= threshold",
            "tie_tolerance": TIE_TOLERANCE,
            "tie_breakers": ["minimum_abs_precision_minus_recall", "lower_threshold"],
        },
    }


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=(
            (
                "type",
                OneHotEncoder(
                    categories=[list(TYPE_CATEGORIES)],
                    handle_unknown="error",
                    sparse_output=False,
                ),
                ["Type"],
            ),
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
        ),
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_candidate_pipelines() -> dict[str, Pipeline]:
    """Build the three fixed candidates with all transforms inside each pipeline."""
    validate_feature_columns(FEATURE_COLUMNS)
    return {
        "dummy": Pipeline(
            steps=(
                ("preprocess", _build_preprocessor()),
                ("model", DummyClassifier(strategy="prior", random_state=RANDOM_SEED)),
            )
        ),
        "logistic_regression": Pipeline(
            steps=(
                ("preprocess", _build_preprocessor()),
                ("model", LogisticRegression(**LOGISTIC_REGRESSION_PARAMETERS)),
            )
        ),
        "random_forest": Pipeline(
            steps=(
                ("preprocess", _build_preprocessor()),
                ("model", RandomForestClassifier(**RANDOM_FOREST_PARAMETERS)),
            )
        ),
    }


def make_cv_plan(target: pd.Series) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Materialize the shared deterministic folds and validate their coverage."""
    target_array = _validated_binary_target(target, context="Training")
    _, counts = np.unique(target_array, return_counts=True)
    if int(counts.min()) < CV_FOLDS:
        raise ModelingError(f"Each class needs at least {CV_FOLDS} rows for stratified CV.")

    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    plan = tuple(
        (training_indices, validation_indices)
        for training_indices, validation_indices in splitter.split(
            np.zeros(len(target_array), dtype=np.int8), target_array
        )
    )
    validation_counts = np.zeros(len(target_array), dtype=np.int8)
    for training_indices, validation_indices in plan:
        if np.intersect1d(training_indices, validation_indices).size:
            raise ModelingError("A CV fold contains overlapping training and validation rows.")
        validation_counts[validation_indices] += 1
        if np.unique(target_array[validation_indices]).size != 2:
            raise ModelingError("Every validation fold must contain both target classes.")
    if not np.all(validation_counts == 1):
        raise ModelingError("Every training row must appear in exactly one validation fold.")
    return plan


def cv_plan_sha256(plan: tuple[tuple[np.ndarray, np.ndarray], ...], row_count: int) -> str:
    """Hash the fold assignment without serializing row-level targets or features."""
    assignment = np.full(row_count, -1, dtype="<i2")
    for fold_index, (_, validation_indices) in enumerate(plan):
        assignment[validation_indices] = fold_index
    if np.any(assignment < 0):
        raise ModelingError("CV plan does not assign every row.")
    return sha256_bytes(assignment.tobytes())


def evaluate_candidates(
    features: pd.DataFrame,
    target: pd.Series,
    plan: tuple[tuple[np.ndarray, np.ndarray], ...],
    pipelines: Mapping[str, Pipeline] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate fixed candidates on the exact same materialized folds."""
    candidates = dict(pipelines or build_candidate_pipelines())
    if tuple(candidates) != CANDIDATE_ORDER:
        raise ModelingError(f"Candidate order must be exactly {list(CANDIDATE_ORDER)}.")
    results: dict[str, dict[str, Any]] = {}
    for name in CANDIDATE_ORDER:
        scores = cross_validate(
            candidates[name],
            features,
            target,
            cv=plan,
            scoring={PRIMARY_METRIC: PRIMARY_METRIC, SECONDARY_METRIC: SECONDARY_METRIC},
            error_score="raise",
            n_jobs=1,
            return_train_score=False,
        )
        ap_by_fold = np.asarray(scores[f"test_{PRIMARY_METRIC}"], dtype=float)
        roc_by_fold = np.asarray(scores[f"test_{SECONDARY_METRIC}"], dtype=float)
        if len(ap_by_fold) != CV_FOLDS or len(roc_by_fold) != CV_FOLDS:
            raise ModelingError("Candidate evaluation returned an unexpected fold count.")
        if not np.all(np.isfinite(ap_by_fold)) or not np.all(np.isfinite(roc_by_fold)):
            raise ModelingError(f"Candidate {name!r} produced non-finite CV metrics.")
        results[name] = {
            "average_precision_by_fold": ap_by_fold.tolist(),
            "average_precision_mean": float(ap_by_fold.mean()),
            "average_precision_std": float(ap_by_fold.std(ddof=CV_STANDARD_DEVIATION_DDOF)),
            "roc_auc_by_fold": roc_by_fold.tolist(),
            "roc_auc_mean": float(roc_by_fold.mean()),
            "roc_auc_std": float(roc_by_fold.std(ddof=CV_STANDARD_DEVIATION_DDOF)),
        }

    dummy_ap = np.asarray(results["dummy"]["average_precision_by_fold"], dtype=float)
    for name in MODEL_CANDIDATES:
        candidate_ap = np.asarray(results[name]["average_precision_by_fold"], dtype=float)
        results[name]["average_precision_delta_vs_dummy_by_fold"] = (
            candidate_ap - dummy_ap
        ).tolist()
        results[name]["average_precision_mean_delta_vs_dummy"] = float(
            candidate_ap.mean() - dummy_ap.mean()
        )
    return results


def choose_candidate(mean_average_precision: Mapping[str, float]) -> str:
    """Apply the frozen AP-only tie rule and strict baseline gate."""
    if set(mean_average_precision) != set(CANDIDATE_ORDER):
        raise ModelingError(f"Selection requires exactly {list(CANDIDATE_ORDER)}.")
    if not all(math.isfinite(float(value)) for value in mean_average_precision.values()):
        raise ModelingError("Selection metrics must be finite.")
    logistic_ap = float(mean_average_precision["logistic_regression"])
    forest_ap = float(mean_average_precision["random_forest"])
    if abs(logistic_ap - forest_ap) <= TIE_TOLERANCE or logistic_ap > forest_ap:
        selected = "logistic_regression"
    else:
        selected = "random_forest"
    if float(mean_average_precision[selected]) <= float(mean_average_precision["dummy"]):
        raise BaselineGateError(
            "No model candidate beats DummyClassifier in mean cross-validated Average Precision."
        )
    return selected


def generate_oof_probabilities(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    plan: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Generate exactly one positive-class probability for every training row."""
    probabilities = cross_val_predict(
        pipeline,
        features,
        target,
        cv=plan,
        method="predict_proba",
        n_jobs=1,
    )
    if probabilities.shape != (len(features), 2):
        raise ModelingError("OOF predict_proba returned an unexpected shape.")
    positive_probabilities = np.asarray(probabilities[:, 1], dtype=float)
    if not np.all(np.isfinite(positive_probabilities)):
        raise ModelingError("OOF probabilities must be finite.")
    if np.any((positive_probabilities < 0.0) | (positive_probabilities > 1.0)):
        raise ModelingError("OOF probabilities must stay within [0, 1].")
    return positive_probabilities


def select_threshold(target: pd.Series | np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Maximize OOF F1 with the pre-registered deterministic tie breakers."""
    target_array = _validated_binary_target(target, context="Threshold selection")
    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim != 1 or len(target_array) != len(score_array):
        raise ModelingError("Target and scores must be one-dimensional with equal lengths.")
    if not np.all(np.isfinite(score_array)):
        raise ModelingError("Threshold scores must be finite.")
    if np.any((score_array < 0.0) | (score_array > 1.0)):
        raise ModelingError("Threshold scores must stay within [0, 1].")

    precision_values, recall_values, thresholds = precision_recall_curve(target_array, score_array)
    threshold_precision = precision_values[:-1]
    threshold_recall = recall_values[:-1]
    denominator = threshold_precision + threshold_recall
    f1_values = np.divide(
        2.0 * threshold_precision * threshold_recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    best_f1 = float(f1_values.max())
    f1_candidates = np.flatnonzero(f1_values >= best_f1 - TIE_TOLERANCE)
    balance = np.abs(threshold_precision - threshold_recall)
    best_balance = float(balance[f1_candidates].min())
    balanced_candidates = f1_candidates[balance[f1_candidates] <= best_balance + TIE_TOLERANCE]
    selected_index = int(balanced_candidates[np.argmin(thresholds[balanced_candidates])])
    threshold = float(thresholds[selected_index])
    predictions = (score_array >= threshold).astype(np.int8)
    matrix = confusion_matrix(target_array, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )
    return {
        "strategy": THRESHOLD_STRATEGY,
        "score": "predict_proba[:, 1]",
        "decision_rule": "score >= threshold",
        "value": threshold,
        "oof_metrics": {
            "average_precision_pooled_diagnostic": float(
                average_precision_score(target_array, score_array)
            ),
            "roc_auc_pooled_diagnostic": float(roc_auc_score(target_array, score_array)),
            "precision": float(precision_score(target_array, predictions, zero_division=0)),
            "recall": float(recall_score(target_array, predictions, zero_division=0)),
            "f1": float(f1_score(target_array, predictions, zero_division=0)),
            "predicted_positive": int(predictions.sum()),
            "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
            "tp": true_positive,
        },
    }


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _new_figure(width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _save_figure(figure: Figure, path: Path) -> None:
    image = io.BytesIO()
    figure.savefig(
        image,
        format="png",
        dpi=160,
        facecolor="white",
        metadata={"Software": "Machine Failure Risk Classifier"},
    )
    content = image.getvalue()
    if path.is_file() and path.read_bytes() != content:
        raise ArtifactIntegrityError(f"Reproducibility drift in existing figure: {path}")
    write_bytes_atomic_if_changed(path, content)


def _write_reproducible_json(path: Path, payload: Any) -> None:
    content = canonical_json_bytes(payload)
    if path.is_file() and path.read_bytes() != content:
        raise ArtifactIntegrityError(f"Reproducibility drift in existing JSON: {path}")
    write_bytes_atomic_if_changed(path, content)


def _write_reproducible_text(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    if path.is_file() and path.read_bytes() != encoded:
        raise ArtifactIntegrityError(f"Reproducibility drift in existing report: {path}")
    write_bytes_atomic_if_changed(path, encoded)


def _plot_cv_average_precision(results: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    labels = ("Dummy", "Logística", "Random forest")
    means = [float(results[name]["average_precision_mean"]) for name in CANDIDATE_ORDER]
    deviations = [float(results[name]["average_precision_std"]) for name in CANDIDATE_ORDER]
    figure = _new_figure(8.4, 5.2)
    axis = figure.subplots()
    positions = np.arange(len(CANDIDATE_ORDER))
    axis.bar(positions, means, yerr=deviations, capsize=6, color=("#7A8793", "#2A9D8F", "#4C78A8"))
    for position, name in zip(positions, CANDIDATE_ORDER, strict=True):
        fold_values = np.asarray(results[name]["average_precision_by_fold"], dtype=float)
        offsets = np.linspace(-0.10, 0.10, len(fold_values))
        axis.scatter(
            np.full(len(fold_values), position) + offsets,
            fold_values,
            color="#263238",
            s=24,
            zorder=3,
        )
    axis.set_xticks(positions, labels=labels)
    axis.set_ylabel("Average Precision")
    axis.set_title("AP por fold y media ± desviación — solo training", weight="bold")
    axis.set_ylim(0.0, min(1.0, max(means) * 1.3 + 0.02))
    axis.grid(axis="y", color="#DCE3E8", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    _save_figure(figure, path)


def _plot_oof_precision_recall(
    target: pd.Series,
    scores: np.ndarray,
    threshold_result: Mapping[str, Any],
    path: Path,
) -> None:
    precision_values, recall_values, _ = precision_recall_curve(target, scores)
    metrics = threshold_result["oof_metrics"]
    figure = _new_figure(7.2, 5.4)
    axis = figure.subplots()
    axis.plot(recall_values, precision_values, color="#2A9D8F", linewidth=2.0, label="Curva OOF")
    axis.scatter(
        [metrics["recall"]],
        [metrics["precision"]],
        color="#D1495B",
        s=60,
        zorder=3,
        label=f"Umbral {threshold_result['value']:.4f}",
    )
    axis.axhline(float(np.mean(target)), color="#7A8793", linestyle="--", label="Prevalencia")
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel="Recall", ylabel="Precision")
    axis.set_title("Precision–recall OOF del modelo elegido — solo training", weight="bold")
    axis.grid(color="#DCE3E8", linewidth=0.8)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    _save_figure(figure, path)


def _selection_report(
    run_id: str,
    results: Mapping[str, Mapping[str, Any]],
    selected_model: str | None,
    threshold_result: Mapping[str, Any] | None,
) -> str:
    table_rows = []
    display_names = {
        "dummy": "Dummy prior",
        "logistic_regression": "Regresión logística",
        "random_forest": "Random forest",
    }
    for name in CANDIDATE_ORDER:
        result = results[name]
        table_rows.append(
            f"| {display_names[name]} | {result['average_precision_mean']:.6f} | "
            f"{result['average_precision_std']:.6f} | {result['roc_auc_mean']:.6f} |"
        )
    if selected_model is None or threshold_result is None:
        selection_text = (
            "Ningún candidato superó estrictamente al dummy en AP media. La puerta se cerró y el "
            "holdout permanece sin consultar."
        )
    else:
        metrics = threshold_result["oof_metrics"]
        selection_text = f"""Se eligió `{selected_model}` exclusivamente por AP media de CV.
El umbral OOF congelado es
`{threshold_result["value"]:.12g}` con la regla `score >= threshold`: precision
{metrics["precision"]:.4f}, recall {metrics["recall"]:.4f} y F1 {metrics["f1"]:.4f}. Estas son
estimaciones de selección sobre training, no resultados finales."""
    return f"""# Selección M3 sobre training

> Run `{run_id}`. AI4I 2020 es sintético; los scores no están calibrados ni validan uso industrial.

| Candidato | AP media | AP std (ddof=0) | ROC-AUC media |
|---|---:|---:|---:|
{chr(10).join(table_rows)}

{selection_text}

![Comparación CV](figures/01_cv_average_precision.png)
{"" if threshold_result is None else "![Curva OOF](figures/02_oof_precision_recall.png)"}
"""


def _publish_pipeline(
    pipeline: Pipeline,
    artifact_root: Path,
    run_id: str,
    training_sha256: str,
    configuration_sha256: str,
    fold_plan_sha256: str,
    selected_model: str,
    threshold: float,
    versions: Mapping[str, str],
) -> tuple[Path, Path, dict[str, Any]]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_directory = artifact_root / run_id
    pipeline_path = run_directory / PIPELINE_FILENAME
    manifest_path = run_directory / ARTIFACT_MANIFEST_FILENAME
    expected_identity = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "training_sha256": training_sha256,
        "configuration_sha256": configuration_sha256,
        "fold_plan_sha256": fold_plan_sha256,
        "selected_model": selected_model,
        "threshold": threshold,
        "pipeline_filename": PIPELINE_FILENAME,
        "versions": dict(versions),
    }
    if run_directory.exists():
        if not run_directory.is_dir() or not manifest_path.is_file():
            raise ArtifactIntegrityError(f"Incomplete local model bundle: {run_directory}")
        manifest = load_json_object(manifest_path)
        if set(manifest) != {*expected_identity, "pipeline_sha256"} or any(
            manifest.get(key) != value for key, value in expected_identity.items()
        ):
            raise ArtifactIntegrityError("Existing local model bundle has unexpected identity.")
        expected_pipeline_sha = manifest.get("pipeline_sha256")
        if (
            not isinstance(expected_pipeline_sha, str)
            or sha256_path(pipeline_path) != expected_pipeline_sha
        ):
            raise ArtifactIntegrityError("Existing serialized pipeline differs from its manifest.")
        return pipeline_path, manifest_path, manifest

    with tempfile.TemporaryDirectory(prefix=f".{run_id}-", dir=artifact_root) as temporary_name:
        staging = Path(temporary_name)
        staged_pipeline = staging / PIPELINE_FILENAME
        joblib.dump(pipeline, staged_pipeline, compress=3, protocol=5)
        pipeline_sha256 = sha256_path(staged_pipeline)
        artifact_manifest = {
            **expected_identity,
            "pipeline_sha256": pipeline_sha256,
        }
        write_json_atomic_if_changed(staging / ARTIFACT_MANIFEST_FILENAME, artifact_manifest)
        staging.replace(run_directory)
    return pipeline_path, manifest_path, artifact_manifest


def run_training_selection(
    processed_dir: Path = DEFAULT_PROCESSED_DATA_DIR,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST_PATH,
    report_dir: Path = DEFAULT_MODEL_REPORT_DIR,
    artifact_root: Path = DEFAULT_MODEL_ARTIFACT_DIR,
) -> TrainingArtifacts:
    """Select and fit from training only; this function never resolves the holdout path."""
    training_frame, split_manifest = load_training_partition(processed_dir, split_manifest_path)
    features = training_frame.loc[:, FEATURE_COLUMNS]
    target = training_frame.loc[:, TARGET_COLUMN]
    configuration = modeling_configuration()
    configuration_sha256 = sha256_bytes(canonical_json_bytes(configuration))
    training_sha256 = split_manifest["training"]["sha256"]
    versions = _runtime_versions()
    plan = make_cv_plan(target)
    fold_plan_sha256 = cv_plan_sha256(plan, len(training_frame))
    run_identity = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "training_sha256": training_sha256,
        "configuration_sha256": configuration_sha256,
        "fold_plan_sha256": fold_plan_sha256,
        "versions": versions,
    }
    run_id = sha256_bytes(canonical_json_bytes(run_identity))[:16]
    run_report_dir = report_dir / run_id
    active_run_path = report_dir / ACTIVE_RUN_FILENAME
    pipelines = build_candidate_pipelines()
    results = evaluate_candidates(features, target, plan, pipelines)
    mean_ap = {name: float(results[name]["average_precision_mean"]) for name in CANDIDATE_ORDER}

    run_report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = run_report_dir / "figures"
    cv_figure = figure_dir / CV_FIGURE_FILENAME
    oof_figure = figure_dir / OOF_FIGURE_FILENAME
    cv_payload = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "scope": "training_cross_validation_only",
        "fold_plan_sha256": fold_plan_sha256,
        "folds": CV_FOLDS,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "standard_deviation_ddof": CV_STANDARD_DEVIATION_DDOF,
        "candidates": results,
    }
    cv_results_path = run_report_dir / CV_RESULTS_FILENAME
    run_manifest_path = run_report_dir / "run_manifest.json"
    selection_report_path = run_report_dir / SELECTION_REPORT_FILENAME
    threshold_path = run_report_dir / THRESHOLD_SELECTION_FILENAME

    try:
        selected_model = choose_candidate(mean_ap)
    except BaselineGateError:
        run_manifest = {
            "schema_version": MODEL_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "configuration": configuration,
            "configuration_sha256": configuration_sha256,
            "versions": versions,
            "source": split_manifest["source"],
            "split": split_manifest["split"],
            "training": split_manifest["training"],
            "holdout": split_manifest["holdout"],
            "fold_plan_sha256": fold_plan_sha256,
            "baseline_gate_passed": False,
            "selected_model": None,
            "selection_receipts": {
                "cv_results_filename": CV_RESULTS_FILENAME,
                "cv_results_sha256": sha256_bytes(canonical_json_bytes(cv_payload)),
                "threshold_selection_filename": None,
                "threshold_selection_sha256": None,
            },
            "holdout_accessed_during_selection": False,
        }
        _plot_cv_average_precision(results, cv_figure)
        _write_reproducible_json(cv_results_path, cv_payload)
        _write_reproducible_json(run_manifest_path, run_manifest)
        _write_reproducible_text(
            selection_report_path,
            _selection_report(run_id, results, None, None),
        )
        write_json_atomic_if_changed(
            active_run_path,
            {
                "schema_version": MODEL_RUN_SCHEMA_VERSION,
                "run_id": run_id,
                "directory": run_id,
                "baseline_gate_passed": False,
            },
        )
        raise

    oof_probabilities = generate_oof_probabilities(
        pipelines[selected_model], features, target, plan
    )
    threshold_result = select_threshold(target, oof_probabilities)
    threshold_payload = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "scope": "out_of_fold_training_selection",
        "selected_model": selected_model,
        **threshold_result,
    }
    fitted_pipeline = pipelines[selected_model].fit(features, target)
    pipeline_path, artifact_manifest_path, artifact_manifest = _publish_pipeline(
        fitted_pipeline,
        artifact_root,
        run_id,
        training_sha256,
        configuration_sha256,
        fold_plan_sha256,
        selected_model,
        float(threshold_result["value"]),
        versions,
    )
    _plot_cv_average_precision(results, cv_figure)
    _plot_oof_precision_recall(target, oof_probabilities, threshold_result, oof_figure)
    _write_reproducible_json(cv_results_path, cv_payload)
    _write_reproducible_json(threshold_path, threshold_payload)
    _write_reproducible_text(
        selection_report_path,
        _selection_report(run_id, results, selected_model, threshold_result),
    )
    run_manifest = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "versions": versions,
        "source": split_manifest["source"],
        "split": split_manifest["split"],
        "training": split_manifest["training"],
        "holdout": split_manifest["holdout"],
        "fold_plan_sha256": fold_plan_sha256,
        "baseline_gate_passed": True,
        "selected_model": selected_model,
        "selected_mean_average_precision": mean_ap[selected_model],
        "dummy_mean_average_precision": mean_ap["dummy"],
        "threshold": float(threshold_result["value"]),
        "artifact": {
            "directory": run_id,
            "pipeline_filename": PIPELINE_FILENAME,
            "pipeline_sha256": artifact_manifest["pipeline_sha256"],
        },
        "selection_receipts": {
            "cv_results_filename": CV_RESULTS_FILENAME,
            "cv_results_sha256": sha256_path(cv_results_path),
            "threshold_selection_filename": THRESHOLD_SELECTION_FILENAME,
            "threshold_selection_sha256": sha256_path(threshold_path),
            "selection_report_filename": SELECTION_REPORT_FILENAME,
            "selection_report_sha256": sha256_path(selection_report_path),
            "cv_figure_filename": f"figures/{CV_FIGURE_FILENAME}",
            "cv_figure_sha256": sha256_path(cv_figure),
            "oof_figure_filename": f"figures/{OOF_FIGURE_FILENAME}",
            "oof_figure_sha256": sha256_path(oof_figure),
        },
        "holdout_accessed_during_selection": False,
    }
    _write_reproducible_json(run_manifest_path, run_manifest)
    write_json_atomic_if_changed(
        active_run_path,
        {
            "schema_version": MODEL_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "directory": run_id,
            "baseline_gate_passed": True,
        },
    )
    return TrainingArtifacts(
        run_id=run_id,
        selected_model=selected_model,
        threshold=float(threshold_result["value"]),
        pipeline=pipeline_path,
        artifact_manifest=artifact_manifest_path,
        run_manifest=run_manifest_path,
        cv_results=cv_results_path,
        threshold_selection=threshold_path,
        selection_report=selection_report_path,
        figures=(cv_figure, oof_figure),
    )
