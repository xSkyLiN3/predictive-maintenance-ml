"""Guarded, recoverable and one-time final holdout evaluation for M3."""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

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
    DEFAULT_HOLDOUT_LEDGER_DIR,
    DEFAULT_MODEL_ARTIFACT_DIR,
    DEFAULT_MODEL_REPORT_DIR,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_SPLIT_MANIFEST_PATH,
    PRIMARY_METRIC,
    SECONDARY_METRIC,
    THRESHOLD_STRATEGY,
)
from predictive_maintenance.modeling import (
    ACTIVE_RUN_FILENAME,
    ARTIFACT_MANIFEST_FILENAME,
    CANDIDATE_ORDER,
    CV_FIGURE_FILENAME,
    CV_RESULTS_FILENAME,
    MODEL_CANDIDATES,
    MODEL_RUN_SCHEMA_VERSION,
    OOF_FIGURE_FILENAME,
    PIPELINE_FILENAME,
    SELECTION_REPORT_FILENAME,
    THRESHOLD_SELECTION_FILENAME,
    choose_candidate,
    modeling_configuration,
)
from predictive_maintenance.splitting import (
    load_holdout_partition_for_final_evaluation,
    load_training_partition,
)
from predictive_maintenance.validation import FEATURE_COLUMNS, TARGET_COLUMN

FINAL_EVALUATION_FILENAME = "final_evaluation.json"
FINAL_REPORT_FILENAME = "M3_REPORT.md"
CONFUSION_FIGURE_FILENAME = "03_holdout_confusion_matrix.png"
FINAL_BUNDLE_DIRECTORY = "final_evaluation_bundle"
FINAL_BUNDLE_MANIFEST_FILENAME = "bundle_manifest.json"
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{16}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class FinalEvaluationError(RuntimeError):
    """Raised when the final holdout cannot be consumed safely."""


@dataclass(frozen=True)
class FinalEvaluationArtifacts:
    """Versionable receipt and local model for the single final evaluation."""

    run_id: str
    selected_model: str
    threshold: float
    pipeline: Path
    evaluation: Path
    report: Path
    confusion_figure: Path
    cached: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "selected_model": self.selected_model,
            "threshold": self.threshold,
            "pipeline": str(self.pipeline),
            "evaluation": str(self.evaluation),
            "report": str(self.report),
            "confusion_figure": str(self.confusion_figure),
            "cached_without_holdout_read": self.cached,
        }


def compute_final_metrics(
    target: pd.Series | np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute all declared final metrics with an inclusive decision threshold."""
    target_series = pd.Series(target, copy=False)
    if (
        target_series.ndim != 1
        or target_series.isna().any()
        or not pd.api.types.is_integer_dtype(target_series.dtype)
    ):
        raise FinalEvaluationError("Final targets must be a non-null integer vector.")
    if set(int(value) for value in target_series.unique()) != {0, 1}:
        raise FinalEvaluationError("Final evaluation requires both binary target classes.")
    target_array = target_series.to_numpy(dtype=np.int8, copy=True)
    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim != 1 or len(target_array) != len(score_array):
        raise FinalEvaluationError(
            "Final targets and scores must be aligned one-dimensional arrays."
        )
    if not np.all(np.isfinite(score_array)) or np.any((score_array < 0.0) | (score_array > 1.0)):
        raise FinalEvaluationError("Final scores must be finite values within [0, 1].")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise FinalEvaluationError("Final threshold must be finite and within [0, 1].")

    predictions = (score_array >= threshold).astype(np.int8)
    matrix = confusion_matrix(target_array, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )
    prevalence = float(target_array.mean())
    return {
        "average_precision": float(average_precision_score(target_array, score_array)),
        "roc_auc": float(roc_auc_score(target_array, score_array)),
        "threshold": float(threshold),
        "decision_rule": "score >= threshold",
        "precision": float(precision_score(target_array, predictions, zero_division=0)),
        "recall": float(recall_score(target_array, predictions, zero_division=0)),
        "f1": float(f1_score(target_array, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(target_array, predictions)),
        "target_prevalence": prevalence,
        "majority_class_accuracy": max(prevalence, 1.0 - prevalence),
        "predicted_positive": int(predictions.sum()),
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
    }


def _confusion_matrix_png(metrics: Mapping[str, Any]) -> bytes:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    figure = Figure(figsize=(6.2, 5.2), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                fontsize=13,
                color="white" if matrix[row, column] > matrix.max() / 2 else "#263238",
            )
    axis.set_xticks((0, 1), labels=("Pred. sin fallo", "Pred. fallo"))
    axis.set_yticks((0, 1), labels=("Real sin fallo", "Real fallo"))
    axis.set_xlabel("Predicción")
    axis.set_ylabel("Target")
    axis.set_title("Matriz de confusión — evaluación final holdout", weight="bold")
    output = io.BytesIO()
    figure.savefig(
        output,
        format="png",
        dpi=160,
        facecolor="white",
        metadata={"Software": "Machine Failure Risk Classifier"},
    )
    return output.getvalue()


def _create_exclusive_ticket(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise FinalEvaluationError(
            "A holdout access ticket already exists without a recoverable final bundle; "
            "manual review is required."
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as ticket:
            ticket.write(canonical_json_bytes(dict(payload)))
            ticket.flush()
            os.fsync(ticket.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _load_active_run(path: Path) -> tuple[dict[str, Any], str]:
    try:
        content = path.read_bytes()
        payload = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"Invalid active run artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError("Active run artifact must contain an object.")
    return payload, sha256_bytes(content)


def _safe_run_directory(root: Path, run_id: str) -> Path:
    resolved_root = root.resolve()
    candidate = root / run_id
    if candidate.resolve().parent != resolved_root:
        raise ArtifactIntegrityError("Run directory escapes its configured root.")
    return candidate


def _load_holdout_ledger(
    path: Path,
    *,
    holdout_sha256: str,
    run_id: str,
    selection_hashes: Mapping[str, str],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = load_json_object(path)
    ledger_run_id = payload.get("run_id")
    if ledger_run_id != run_id:
        raise FinalEvaluationError(
            "This holdout was already reserved or evaluated by another model run "
            f"({ledger_run_id!r}); it will not be opened again."
        )
    status = payload.get("status")
    required = {
        "schema_version",
        "holdout_sha256",
        "run_id",
        "status",
        "selection_receipts",
    }
    if status == "holdout_evaluation_complete":
        required.add("final_receipt_sha256")
    if (
        set(payload) != required
        or payload.get("schema_version") != MODEL_RUN_SCHEMA_VERSION
        or payload.get("holdout_sha256") != holdout_sha256
        or payload.get("selection_receipts") != dict(selection_hashes)
        or status not in {"holdout_access_started", "holdout_evaluation_complete"}
        or (
            status == "holdout_evaluation_complete"
            and not RUN_ID_PATTERN.fullmatch(str(ledger_run_id))
        )
        or (
            status == "holdout_evaluation_complete"
            and not SHA256_PATTERN.fullmatch(str(payload.get("final_receipt_sha256")))
        )
    ):
        raise ArtifactIntegrityError("Global holdout ledger is malformed or inconsistent.")
    return payload


def _validate_local_pipeline_bundle(
    run_manifest: Mapping[str, Any], artifact_root: Path
) -> tuple[Path, Path, dict[str, Any]]:
    run_id = run_manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ArtifactIntegrityError("Run manifest has an invalid run id.")
    artifact_directory = _safe_run_directory(artifact_root, run_id)
    pipeline_path = artifact_directory / PIPELINE_FILENAME
    artifact_manifest_path = artifact_directory / ARTIFACT_MANIFEST_FILENAME
    artifact_manifest = load_json_object(artifact_manifest_path)
    expected = {
        "schema_version": run_manifest.get("schema_version"),
        "run_id": run_id,
        "training_sha256": run_manifest.get("training", {}).get("sha256"),
        "configuration_sha256": run_manifest.get("configuration_sha256"),
        "fold_plan_sha256": run_manifest.get("fold_plan_sha256"),
        "selected_model": run_manifest.get("selected_model"),
        "threshold": run_manifest.get("threshold"),
        "pipeline_filename": PIPELINE_FILENAME,
        "versions": run_manifest.get("versions"),
    }
    if set(artifact_manifest) != {*expected, "pipeline_sha256"} or any(
        artifact_manifest.get(key) != value for key, value in expected.items()
    ):
        raise ArtifactIntegrityError("Local pipeline bundle does not match the selected run.")
    pipeline_sha256 = artifact_manifest.get("pipeline_sha256")
    if not isinstance(pipeline_sha256, str) or sha256_path(pipeline_path) != pipeline_sha256:
        raise ArtifactIntegrityError("Serialized pipeline differs from its local manifest.")
    expected_artifact = run_manifest.get("artifact")
    if (
        not isinstance(expected_artifact, dict)
        or expected_artifact.get("directory") != run_id
        or expected_artifact.get("pipeline_filename") != PIPELINE_FILENAME
        or expected_artifact.get("pipeline_sha256") != pipeline_sha256
    ):
        raise ArtifactIntegrityError("Run manifest and local pipeline identity disagree.")
    return pipeline_path, artifact_manifest_path, artifact_manifest


def _validate_run_manifest_identity(run_manifest: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "run_id",
        "configuration",
        "configuration_sha256",
        "versions",
        "source",
        "split",
        "training",
        "holdout",
        "fold_plan_sha256",
        "baseline_gate_passed",
        "selected_model",
        "selected_mean_average_precision",
        "dummy_mean_average_precision",
        "threshold",
        "artifact",
        "selection_receipts",
        "holdout_accessed_during_selection",
    }
    if set(run_manifest) != expected_keys:
        raise ArtifactIntegrityError("Active run manifest has an unexpected schema.")
    configuration = run_manifest.get("configuration")
    configuration_sha256 = run_manifest.get("configuration_sha256")
    versions = run_manifest.get("versions")
    training = run_manifest.get("training")
    if (
        run_manifest.get("schema_version") != MODEL_RUN_SCHEMA_VERSION
        or configuration != modeling_configuration()
        or not isinstance(configuration_sha256, str)
        or configuration_sha256 != sha256_bytes(canonical_json_bytes(configuration))
        or not isinstance(versions, dict)
        or not versions
        or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in versions.items()
        )
        or not isinstance(training, dict)
    ):
        raise ArtifactIntegrityError("Active run manifest has an invalid frozen configuration.")
    run_identity = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "training_sha256": training.get("sha256"),
        "configuration_sha256": configuration_sha256,
        "fold_plan_sha256": run_manifest.get("fold_plan_sha256"),
        "versions": versions,
    }
    expected_run_id = sha256_bytes(canonical_json_bytes(run_identity))[:16]
    selected_model = run_manifest.get("selected_model")
    selected_ap = run_manifest.get("selected_mean_average_precision")
    dummy_ap = run_manifest.get("dummy_mean_average_precision")
    threshold = run_manifest.get("threshold")
    if (
        run_manifest.get("run_id") != expected_run_id
        or run_manifest.get("baseline_gate_passed") is not True
        or run_manifest.get("holdout_accessed_during_selection") is not False
        or selected_model not in MODEL_CANDIDATES
        or not isinstance(selected_ap, (int, float))
        or not isinstance(dummy_ap, (int, float))
        or not np.isfinite(selected_ap)
        or not np.isfinite(dummy_ap)
        or not 0.0 <= float(dummy_ap) < float(selected_ap) <= 1.0
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ArtifactIntegrityError("Active run identity or selection gate is inconsistent.")


def _validate_split_identity(
    run_manifest: Mapping[str, Any], split_manifest: Mapping[str, Any]
) -> None:
    for section in ("source", "split", "training", "holdout"):
        if run_manifest.get(section) != split_manifest.get(section):
            raise ArtifactIntegrityError(
                f"Requested split metadata differs from the selected run: {section}."
            )


def _selection_receipt_hashes(
    run_manifest_path: Path,
    cv_results_path: Path,
    threshold_path: Path,
    selection_report_path: Path,
    cv_figure_path: Path,
    oof_figure_path: Path,
) -> dict[str, str]:
    return {
        "run_manifest_sha256": sha256_path(run_manifest_path),
        "cv_results_sha256": sha256_path(cv_results_path),
        "threshold_selection_sha256": sha256_path(threshold_path),
        "selection_report_sha256": sha256_path(selection_report_path),
        "cv_figure_sha256": sha256_path(cv_figure_path),
        "oof_figure_sha256": sha256_path(oof_figure_path),
    }


def _finite_vector(value: object, *, name: str, length: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError(f"{name} must contain numeric values.") from error
    if array.ndim != 1 or len(array) != length or not np.all(np.isfinite(array)):
        raise ArtifactIntegrityError(f"{name} must contain {length} finite values.")
    return array


def _validate_cv_payload(cv_results: Mapping[str, Any], run_manifest: Mapping[str, Any]) -> None:
    if set(cv_results) != {
        "schema_version",
        "run_id",
        "scope",
        "fold_plan_sha256",
        "folds",
        "primary_metric",
        "secondary_metric",
        "standard_deviation_ddof",
        "candidates",
    }:
        raise ArtifactIntegrityError("CV receipt has an unexpected schema.")
    candidates = cv_results.get("candidates")
    if (
        cv_results.get("schema_version") != MODEL_RUN_SCHEMA_VERSION
        or cv_results.get("run_id") != run_manifest.get("run_id")
        or cv_results.get("scope") != "training_cross_validation_only"
        or cv_results.get("fold_plan_sha256") != run_manifest.get("fold_plan_sha256")
        or cv_results.get("folds") != CV_FOLDS
        or cv_results.get("primary_metric") != PRIMARY_METRIC
        or cv_results.get("secondary_metric") != SECONDARY_METRIC
        or cv_results.get("standard_deviation_ddof") != CV_STANDARD_DEVIATION_DDOF
        or not isinstance(candidates, dict)
        or tuple(candidates) != CANDIDATE_ORDER
    ):
        raise ArtifactIntegrityError("CV receipt does not match the frozen protocol.")

    ap_by_candidate: dict[str, np.ndarray] = {}
    means: dict[str, float] = {}
    base_keys = {
        "average_precision_by_fold",
        "average_precision_mean",
        "average_precision_std",
        "roc_auc_by_fold",
        "roc_auc_mean",
        "roc_auc_std",
    }
    for name in CANDIDATE_ORDER:
        result = candidates[name]
        expected_keys = (
            base_keys
            if name == "dummy"
            else {
                *base_keys,
                "average_precision_delta_vs_dummy_by_fold",
                "average_precision_mean_delta_vs_dummy",
            }
        )
        if not isinstance(result, dict) or set(result) != expected_keys:
            raise ArtifactIntegrityError(f"CV result for {name!r} has an unexpected schema.")
        ap = _finite_vector(
            result["average_precision_by_fold"], name=f"{name} AP folds", length=CV_FOLDS
        )
        roc = _finite_vector(result["roc_auc_by_fold"], name=f"{name} ROC folds", length=CV_FOLDS)
        scalars = (
            result["average_precision_mean"],
            result["average_precision_std"],
            result["roc_auc_mean"],
            result["roc_auc_std"],
        )
        if (
            np.any((ap < 0.0) | (ap > 1.0))
            or np.any((roc < 0.0) | (roc > 1.0))
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0.0 <= float(value) <= 1.0
                for value in scalars
            )
            or not np.isclose(result["average_precision_mean"], ap.mean(), rtol=0.0, atol=1e-15)
            or not np.isclose(
                result["average_precision_std"],
                ap.std(ddof=CV_STANDARD_DEVIATION_DDOF),
                rtol=0.0,
                atol=1e-15,
            )
            or not np.isclose(result["roc_auc_mean"], roc.mean(), rtol=0.0, atol=1e-15)
            or not np.isclose(
                result["roc_auc_std"],
                roc.std(ddof=CV_STANDARD_DEVIATION_DDOF),
                rtol=0.0,
                atol=1e-15,
            )
        ):
            raise ArtifactIntegrityError(f"CV aggregates for {name!r} are inconsistent.")
        ap_by_candidate[name] = ap
        means[name] = float(result["average_precision_mean"])

    dummy_ap = ap_by_candidate["dummy"]
    for name in MODEL_CANDIDATES:
        result = candidates[name]
        delta = _finite_vector(
            result["average_precision_delta_vs_dummy_by_fold"],
            name=f"{name} AP deltas",
            length=CV_FOLDS,
        )
        expected_delta = ap_by_candidate[name] - dummy_ap
        if not np.allclose(delta, expected_delta, rtol=0.0, atol=1e-15) or not np.isclose(
            result["average_precision_mean_delta_vs_dummy"],
            expected_delta.mean(),
            rtol=0.0,
            atol=1e-15,
        ):
            raise ArtifactIntegrityError(f"CV dummy deltas for {name!r} are inconsistent.")
    try:
        expected_selection = choose_candidate(means)
    except RuntimeError as error:
        raise ArtifactIntegrityError("CV receipt no longer passes the dummy gate.") from error
    if expected_selection != run_manifest.get("selected_model"):
        raise ArtifactIntegrityError("CV receipt does not reproduce the selected model.")


def _validate_threshold_payload(
    threshold_selection: Mapping[str, Any], run_manifest: Mapping[str, Any]
) -> None:
    if set(threshold_selection) != {
        "schema_version",
        "run_id",
        "scope",
        "selected_model",
        "strategy",
        "score",
        "decision_rule",
        "value",
        "oof_metrics",
    }:
        raise ArtifactIntegrityError("Threshold receipt has an unexpected schema.")
    threshold = threshold_selection.get("value")
    metrics = threshold_selection.get("oof_metrics")
    if (
        threshold_selection.get("schema_version") != MODEL_RUN_SCHEMA_VERSION
        or threshold_selection.get("run_id") != run_manifest.get("run_id")
        or threshold_selection.get("scope") != "out_of_fold_training_selection"
        or threshold_selection.get("selected_model") != run_manifest.get("selected_model")
        or threshold_selection.get("strategy") != THRESHOLD_STRATEGY
        or threshold_selection.get("score") != "predict_proba[:, 1]"
        or threshold_selection.get("decision_rule") != "score >= threshold"
        or not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not np.isfinite(threshold)
        or not 0.0 <= float(threshold) <= 1.0
        or threshold != run_manifest.get("threshold")
        or not isinstance(metrics, dict)
        or set(metrics)
        != {
            "average_precision_pooled_diagnostic",
            "roc_auc_pooled_diagnostic",
            "precision",
            "recall",
            "f1",
            "predicted_positive",
            "confusion_matrix",
            "tn",
            "fp",
            "fn",
            "tp",
        }
    ):
        raise ArtifactIntegrityError("Threshold receipt does not match the frozen protocol.")
    rate_keys = (
        "average_precision_pooled_diagnostic",
        "roc_auc_pooled_diagnostic",
        "precision",
        "recall",
        "f1",
    )
    count_keys = ("predicted_positive", "tn", "fp", "fn", "tp")
    if any(
        isinstance(metrics[key], bool)
        or not isinstance(metrics[key], (int, float))
        or not np.isfinite(metrics[key])
        or not 0.0 <= float(metrics[key]) <= 1.0
        for key in rate_keys
    ) or any(
        isinstance(metrics[key], bool) or not isinstance(metrics[key], int) or metrics[key] < 0
        for key in count_keys
    ):
        raise ArtifactIntegrityError("OOF threshold metrics contain invalid values.")
    expected_matrix = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    training_rows = run_manifest["training"].get("rows")
    if (
        metrics["confusion_matrix"] != expected_matrix
        or not isinstance(training_rows, int)
        or sum(sum(row) for row in expected_matrix) != training_rows
        or metrics["predicted_positive"] != metrics["fp"] + metrics["tp"]
    ):
        raise ArtifactIntegrityError("OOF threshold confusion counts are inconsistent.")
    expected_precision = metrics["tp"] / max(metrics["tp"] + metrics["fp"], 1)
    expected_recall = metrics["tp"] / max(metrics["tp"] + metrics["fn"], 1)
    expected_f1 = (
        0.0
        if expected_precision + expected_recall == 0.0
        else 2.0 * expected_precision * expected_recall / (expected_precision + expected_recall)
    )
    if not (
        np.isclose(metrics["precision"], expected_precision, rtol=0.0, atol=1e-15)
        and np.isclose(metrics["recall"], expected_recall, rtol=0.0, atol=1e-15)
        and np.isclose(metrics["f1"], expected_f1, rtol=0.0, atol=1e-15)
    ):
        raise ArtifactIntegrityError("OOF precision, recall or F1 is inconsistent.")


def _validate_selection_receipts(
    run_manifest: Mapping[str, Any],
    run_manifest_path: Path,
    cv_results_path: Path,
    threshold_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    run_report_dir = run_manifest_path.parent
    selection_report_path = run_report_dir / SELECTION_REPORT_FILENAME
    cv_figure_path = run_report_dir / "figures" / CV_FIGURE_FILENAME
    oof_figure_path = run_report_dir / "figures" / OOF_FIGURE_FILENAME
    cv_results = load_json_object(cv_results_path)
    threshold_selection = load_json_object(threshold_path)
    run_id = run_manifest.get("run_id")
    if cv_results.get("run_id") != run_id or threshold_selection.get("run_id") != run_id:
        raise ArtifactIntegrityError("Selection reports do not belong to the active run.")
    declared = run_manifest.get("selection_receipts")
    if not isinstance(declared, dict) or set(declared) != {
        "cv_results_filename",
        "cv_results_sha256",
        "threshold_selection_filename",
        "threshold_selection_sha256",
        "selection_report_filename",
        "selection_report_sha256",
        "cv_figure_filename",
        "cv_figure_sha256",
        "oof_figure_filename",
        "oof_figure_sha256",
    }:
        raise ArtifactIntegrityError("Run manifest does not fix its selection receipts.")
    if (
        declared.get("cv_results_filename") != CV_RESULTS_FILENAME
        or declared.get("threshold_selection_filename") != THRESHOLD_SELECTION_FILENAME
        or declared.get("selection_report_filename") != SELECTION_REPORT_FILENAME
        or declared.get("cv_figure_filename") != f"figures/{CV_FIGURE_FILENAME}"
        or declared.get("oof_figure_filename") != f"figures/{OOF_FIGURE_FILENAME}"
        or declared.get("cv_results_sha256") != sha256_path(cv_results_path)
        or declared.get("threshold_selection_sha256") != sha256_path(threshold_path)
        or declared.get("selection_report_sha256") != sha256_path(selection_report_path)
        or declared.get("cv_figure_sha256") != sha256_path(cv_figure_path)
        or declared.get("oof_figure_sha256") != sha256_path(oof_figure_path)
    ):
        raise ArtifactIntegrityError("Selection report hashes differ from the run manifest.")
    _validate_cv_payload(cv_results, run_manifest)
    _validate_threshold_payload(threshold_selection, run_manifest)
    selected_model = run_manifest["selected_model"]
    candidates = cv_results.get("candidates")
    if (
        cv_results.get("schema_version") != MODEL_RUN_SCHEMA_VERSION
        or cv_results.get("scope") != "training_cross_validation_only"
        or cv_results.get("fold_plan_sha256") != run_manifest.get("fold_plan_sha256")
        or not isinstance(candidates, dict)
        or set(candidates) != {"dummy", *MODEL_CANDIDATES}
        or candidates[selected_model].get("average_precision_mean")
        != run_manifest.get("selected_mean_average_precision")
        or candidates["dummy"].get("average_precision_mean")
        != run_manifest.get("dummy_mean_average_precision")
        or threshold_selection.get("schema_version") != MODEL_RUN_SCHEMA_VERSION
        or threshold_selection.get("scope") != "out_of_fold_training_selection"
        or threshold_selection.get("selected_model") != selected_model
        or threshold_selection.get("value") != run_manifest.get("threshold")
        or threshold_selection.get("score") != "predict_proba[:, 1]"
        or threshold_selection.get("decision_rule") != "score >= threshold"
    ):
        raise ArtifactIntegrityError("Selection receipts are internally inconsistent.")
    return (
        cv_results,
        threshold_selection,
        _selection_receipt_hashes(
            run_manifest_path,
            cv_results_path,
            threshold_path,
            selection_report_path,
            cv_figure_path,
            oof_figure_path,
        ),
    )


def _validate_metrics(metrics: object, expected_rows: int, expected_threshold: float) -> None:
    required = {
        "average_precision",
        "roc_auc",
        "threshold",
        "decision_rule",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "target_prevalence",
        "majority_class_accuracy",
        "predicted_positive",
        "confusion_matrix",
        "tn",
        "fp",
        "fn",
        "tp",
    }
    if not isinstance(metrics, dict) or set(metrics) != required:
        raise ArtifactIntegrityError("Final evaluation contains incomplete metrics.")
    scalar_metrics = (
        "average_precision",
        "roc_auc",
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "target_prevalence",
        "majority_class_accuracy",
    )
    if any(
        isinstance(metrics[key], bool)
        or not isinstance(metrics[key], (int, float))
        or not np.isfinite(metrics[key])
        or not 0.0 <= float(metrics[key]) <= 1.0
        for key in scalar_metrics
    ):
        raise ArtifactIntegrityError("Final evaluation contains non-finite metrics.")
    if (
        metrics["decision_rule"] != "score >= threshold"
        or float(metrics["threshold"]) != expected_threshold
    ):
        raise ArtifactIntegrityError("Final evaluation uses an unexpected decision rule.")
    count_keys = ("predicted_positive", "tn", "fp", "fn", "tp")
    if any(
        isinstance(metrics[key], bool) or not isinstance(metrics[key], int) or metrics[key] < 0
        for key in count_keys
    ):
        raise ArtifactIntegrityError("Final evaluation contains invalid confusion counts.")
    matrix = metrics["confusion_matrix"]
    expected_matrix = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    if (
        matrix != expected_matrix
        or sum(sum(row) for row in matrix) != expected_rows
        or metrics["predicted_positive"] != metrics["fp"] + metrics["tp"]
    ):
        raise ArtifactIntegrityError("Final confusion matrix is inconsistent.")
    expected_prevalence = (metrics["fn"] + metrics["tp"]) / expected_rows
    expected_accuracy = (metrics["tn"] + metrics["tp"]) / expected_rows
    expected_majority = max(expected_prevalence, 1.0 - expected_prevalence)
    if not (
        np.isclose(metrics["target_prevalence"], expected_prevalence, rtol=0.0, atol=1e-15)
        and np.isclose(metrics["accuracy"], expected_accuracy, rtol=0.0, atol=1e-15)
        and np.isclose(metrics["majority_class_accuracy"], expected_majority, rtol=0.0, atol=1e-15)
    ):
        raise ArtifactIntegrityError("Final contextual metrics disagree with the confusion matrix.")


def _validate_evaluation_payload(
    payload: Mapping[str, Any],
    *,
    run_manifest: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    selection_hashes: Mapping[str, str],
    report_bytes: bytes,
    figure_bytes: bytes,
) -> None:
    expected_keys = {
        "schema_version",
        "run_id",
        "receipt",
        "selected_model",
        "threshold",
        "score",
        "holdout",
        "selection_receipts",
        "pipeline_sha256",
        "metrics",
        "outputs",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != MODEL_RUN_SCHEMA_VERSION
        or payload.get("run_id") != run_manifest.get("run_id")
        or payload.get("receipt") != "final_holdout_evaluation_complete"
        or payload.get("selected_model") != run_manifest.get("selected_model")
        or payload.get("threshold") != run_manifest.get("threshold")
        or payload.get("score") != "predict_proba[:, 1] (not assessed for calibration)"
        or payload.get("pipeline_sha256") != artifact_manifest.get("pipeline_sha256")
        or payload.get("selection_receipts") != dict(selection_hashes)
    ):
        raise ArtifactIntegrityError("Final evaluation identity is inconsistent.")
    expected_holdout = {
        "filename": split_manifest["holdout"]["filename"],
        "rows": split_manifest["holdout"]["rows"],
        "sha256": split_manifest["holdout"]["sha256"],
        "application_reads_for_evaluation": 1,
    }
    if payload.get("holdout") != expected_holdout:
        raise ArtifactIntegrityError("Final evaluation references unexpected holdout metadata.")
    expected_outputs = {
        "report_filename": FINAL_REPORT_FILENAME,
        "report_sha256": sha256_bytes(report_bytes),
        "confusion_figure_filename": f"figures/{CONFUSION_FIGURE_FILENAME}",
        "confusion_figure_sha256": sha256_bytes(figure_bytes),
    }
    if payload.get("outputs") != expected_outputs:
        raise ArtifactIntegrityError("Final report or figure hash differs from its receipt.")
    _validate_metrics(
        payload.get("metrics"),
        int(expected_holdout["rows"]),
        float(run_manifest["threshold"]),
    )


def _build_final_report(
    run_manifest: Mapping[str, Any],
    cv_results: Mapping[str, Any],
    threshold_selection: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> str:
    candidates = cv_results["candidates"]
    rows = []
    labels = {
        "dummy": "Dummy prior",
        "logistic_regression": "Regresión logística",
        "random_forest": "Random forest",
    }
    for name in ("dummy", "logistic_regression", "random_forest"):
        result = candidates[name]
        rows.append(
            f"| {labels[name]} | {result['average_precision_mean']:.6f} | "
            f"{result['average_precision_std']:.6f} | {result['roc_auc_mean']:.6f} |"
        )
    oof = threshold_selection["oof_metrics"]
    final = evaluation["metrics"]
    return f"""# Resultado M3 — selección y evaluación final

> Run `{run_manifest["run_id"]}`. AI4I 2020 es sintético. Este resultado no valida uso industrial
> y los scores de `predict_proba` no se evaluaron como probabilidades calibradas.

## Selección exclusivamente sobre training

| Candidato | AP media CV | AP std (ddof=0) | ROC-AUC media CV |
|---|---:|---:|---:|
{chr(10).join(rows)}

Modelo elegido: **`{evaluation["selected_model"]}`**. El dummy obtuvo AP media
`{run_manifest["dummy_mean_average_precision"]:.6f}` y el candidato elegido
`{run_manifest["selected_mean_average_precision"]:.6f}`.

El umbral `{evaluation["threshold"]:.12g}` se congeló con predicciones OOF de training antes de
abrir holdout. En OOF: precision `{oof["precision"]:.4f}`, recall `{oof["recall"]:.4f}` y F1
`{oof["f1"]:.4f}`. Estas cifras fueron parte de la selección, no son el resultado final.

![AP por fold](figures/01_cv_average_precision.png)

![Curva OOF](figures/02_oof_precision_recall.png)

## Evaluación final única sobre holdout

- Average Precision: **{final["average_precision"]:.6f}**.
- ROC-AUC: {final["roc_auc"]:.6f}.
- Precision al umbral: {final["precision"]:.6f}.
- Recall al umbral: {final["recall"]:.6f}.
- F1 al umbral: {final["f1"]:.6f}.
- Matriz `[[TN, FP], [FN, TP]]`: `{final["confusion_matrix"]}`.
- Accuracy: {final["accuracy"]:.6f}, mostrada con prevalencia positiva
  {final["target_prevalence"]:.4%} y referencia de clase mayoritaria
  {final["majority_class_accuracy"]:.6f}.

![Matriz de confusión final](figures/{CONFUSION_FIGURE_FILENAME})

## Límites

- El holdout se consultó una sola vez después de congelar modelo y umbral; ejecuciones posteriores
  reutilizan el recibo versionado.
- El split aleatorio estima generalización IID dentro del generador sintético, no generalización
  temporal, entre máquinas o en industria.
- `class_weight` mejora el tratamiento de la minoría, pero los scores no están calibrados y no
  deben interpretarse como frecuencias industriales de fallo.
- No se probaron más algoritmos, hiperparámetros ni features después de observar el resultado.
"""


def _publish_local_bundle(
    artifact_run_dir: Path,
    evaluation_payload: Mapping[str, Any],
    report_bytes: bytes,
    figure_bytes: bytes,
) -> Path:
    bundle_path = artifact_run_dir / FINAL_BUNDLE_DIRECTORY
    if bundle_path.exists():
        raise ArtifactIntegrityError("A local final evaluation bundle already exists unexpectedly.")
    with tempfile.TemporaryDirectory(
        prefix=".final-evaluation-", dir=artifact_run_dir
    ) as temporary_name:
        staging = Path(temporary_name)
        write_bytes_atomic_if_changed(staging / FINAL_REPORT_FILENAME, report_bytes)
        write_bytes_atomic_if_changed(staging / "figures" / CONFUSION_FIGURE_FILENAME, figure_bytes)
        evaluation_bytes = canonical_json_bytes(evaluation_payload)
        write_bytes_atomic_if_changed(staging / FINAL_EVALUATION_FILENAME, evaluation_bytes)
        bundle_manifest = {
            "schema_version": MODEL_RUN_SCHEMA_VERSION,
            "run_id": evaluation_payload["run_id"],
            "files": {
                FINAL_REPORT_FILENAME: sha256_bytes(report_bytes),
                f"figures/{CONFUSION_FIGURE_FILENAME}": sha256_bytes(figure_bytes),
                FINAL_EVALUATION_FILENAME: sha256_bytes(evaluation_bytes),
            },
        }
        write_json_atomic_if_changed(staging / FINAL_BUNDLE_MANIFEST_FILENAME, bundle_manifest)
        staging.replace(bundle_path)
    return bundle_path


def _load_local_bundle(bundle_path: Path) -> tuple[dict[str, Any], bytes, bytes] | None:
    if not bundle_path.exists():
        return None
    if not bundle_path.is_dir():
        raise ArtifactIntegrityError("Local final evaluation bundle is not a directory.")
    manifest = load_json_object(bundle_path / FINAL_BUNDLE_MANIFEST_FILENAME)
    evaluation_path = bundle_path / FINAL_EVALUATION_FILENAME
    report_path = bundle_path / FINAL_REPORT_FILENAME
    figure_path = bundle_path / "figures" / CONFUSION_FIGURE_FILENAME
    files = manifest.get("files")
    expected_files = {
        FINAL_REPORT_FILENAME: sha256_path(report_path),
        f"figures/{CONFUSION_FIGURE_FILENAME}": sha256_path(figure_path),
        FINAL_EVALUATION_FILENAME: sha256_path(evaluation_path),
    }
    evaluation_payload = load_json_object(evaluation_path)
    if (
        set(manifest) != {"schema_version", "run_id", "files"}
        or manifest.get("schema_version") != MODEL_RUN_SCHEMA_VERSION
        or manifest.get("run_id") != evaluation_payload.get("run_id")
        or not isinstance(files, dict)
        or files != expected_files
    ):
        raise ArtifactIntegrityError("Local final evaluation bundle failed integrity checks.")
    return evaluation_payload, report_path.read_bytes(), figure_path.read_bytes()


def _publish_versioned_bundle(
    run_report_dir: Path,
    evaluation_payload: Mapping[str, Any],
    report_bytes: bytes,
    figure_bytes: bytes,
) -> tuple[Path, Path, Path]:
    report_path = run_report_dir / FINAL_REPORT_FILENAME
    figure_path = run_report_dir / "figures" / CONFUSION_FIGURE_FILENAME
    evaluation_path = run_report_dir / FINAL_EVALUATION_FILENAME
    write_bytes_atomic_if_changed(report_path, report_bytes)
    write_bytes_atomic_if_changed(figure_path, figure_bytes)
    write_json_atomic_if_changed(evaluation_path, evaluation_payload)
    return evaluation_path, report_path, figure_path


def _final_artifacts(
    run_id: str,
    selected_model: str,
    threshold: float,
    pipeline_path: Path,
    run_report_dir: Path,
    *,
    cached: bool,
) -> FinalEvaluationArtifacts:
    return FinalEvaluationArtifacts(
        run_id=run_id,
        selected_model=selected_model,
        threshold=threshold,
        pipeline=pipeline_path,
        evaluation=run_report_dir / FINAL_EVALUATION_FILENAME,
        report=run_report_dir / FINAL_REPORT_FILENAME,
        confusion_figure=run_report_dir / "figures" / CONFUSION_FIGURE_FILENAME,
        cached=cached,
    )


def evaluate_final_holdout(
    processed_dir: Path = DEFAULT_PROCESSED_DATA_DIR,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST_PATH,
    report_dir: Path = DEFAULT_MODEL_REPORT_DIR,
    artifact_root: Path = DEFAULT_MODEL_ARTIFACT_DIR,
    *,
    confirm_final_evaluation: bool = False,
    holdout_ledger_dir: Path = DEFAULT_HOLDOUT_LEDGER_DIR,
    holdout_loader: Callable[[Path, Path], tuple[pd.DataFrame, dict[str, Any]]] = (
        load_holdout_partition_for_final_evaluation
    ),
) -> FinalEvaluationArtifacts:
    """Evaluate once after confirmation; recover publication without rereading holdout."""
    active_run_path = report_dir / ACTIVE_RUN_FILENAME
    active_run, active_run_sha256 = _load_active_run(active_run_path)
    run_id = active_run.get("run_id")
    if (
        set(active_run) != {"schema_version", "run_id", "directory", "baseline_gate_passed"}
        or active_run.get("schema_version") != MODEL_RUN_SCHEMA_VERSION
        or not isinstance(run_id, str)
        or RUN_ID_PATTERN.fullmatch(run_id) is None
        or active_run.get("directory") != run_id
        or active_run.get("baseline_gate_passed") is not True
    ):
        raise FinalEvaluationError(
            "Training selection is incomplete or did not pass the dummy gate."
        )
    run_report_dir = _safe_run_directory(report_dir, run_id)
    run_manifest_path = run_report_dir / "run_manifest.json"
    cv_results_path = run_report_dir / "cv_results.json"
    threshold_path = run_report_dir / "threshold_selection.json"
    run_manifest = load_json_object(run_manifest_path)
    _validate_run_manifest_identity(run_manifest)
    if run_manifest.get("run_id") != run_id:
        raise ArtifactIntegrityError("Active pointer and run manifest disagree.")
    pipeline_path, _, artifact_manifest = _validate_local_pipeline_bundle(
        run_manifest, artifact_root
    )
    cv_results, threshold_selection, selection_hashes = _validate_selection_receipts(
        run_manifest,
        run_manifest_path,
        cv_results_path,
        threshold_path,
    )
    selected_model = run_manifest.get("selected_model")
    threshold = run_manifest.get("threshold")
    if (
        not isinstance(selected_model, str)
        or not isinstance(threshold, (int, float))
        or threshold_selection.get("selected_model") != selected_model
        or threshold_selection.get("value") != threshold
    ):
        raise ArtifactIntegrityError("Selected model or threshold differs across artifacts.")

    training_frame, split_manifest = load_training_partition(processed_dir, split_manifest_path)
    _validate_split_identity(run_manifest, split_manifest)
    artifact_run_dir = _safe_run_directory(artifact_root, run_id)
    bundle_path = artifact_run_dir / FINAL_BUNDLE_DIRECTORY
    holdout_sha256 = split_manifest["holdout"]["sha256"]
    if not isinstance(holdout_sha256, str) or SHA256_PATTERN.fullmatch(holdout_sha256) is None:
        raise ArtifactIntegrityError("Split manifest contains an invalid holdout SHA-256.")
    ledger_path = holdout_ledger_dir / f"{holdout_sha256}.json"
    ledger = _load_holdout_ledger(
        ledger_path,
        holdout_sha256=holdout_sha256,
        run_id=run_id,
        selection_hashes=selection_hashes,
    )
    ledger_identity = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "holdout_sha256": holdout_sha256,
        "run_id": run_id,
        "selection_receipts": selection_hashes,
    }

    local_bundle = _load_local_bundle(bundle_path)
    if local_bundle is not None:
        if ledger is None:
            raise ArtifactIntegrityError("A final bundle exists without the global holdout ledger.")
        evaluation_payload, report_bytes, figure_bytes = local_bundle
        _validate_evaluation_payload(
            evaluation_payload,
            run_manifest=run_manifest,
            artifact_manifest=artifact_manifest,
            split_manifest=split_manifest,
            selection_hashes=selection_hashes,
            report_bytes=report_bytes,
            figure_bytes=figure_bytes,
        )
        expected_receipt_sha256 = sha256_bytes(canonical_json_bytes(evaluation_payload))
        if (
            ledger.get("status") == "holdout_evaluation_complete"
            and ledger.get("final_receipt_sha256") != expected_receipt_sha256
        ):
            raise ArtifactIntegrityError("Global ledger differs from the recoverable receipt.")
        _publish_versioned_bundle(run_report_dir, evaluation_payload, report_bytes, figure_bytes)
        write_json_atomic_if_changed(
            ledger_path,
            {
                **ledger_identity,
                "status": "holdout_evaluation_complete",
                "final_receipt_sha256": expected_receipt_sha256,
            },
        )
        return _final_artifacts(
            run_id,
            selected_model,
            float(threshold),
            pipeline_path,
            run_report_dir,
            cached=True,
        )

    evaluation_path = run_report_dir / FINAL_EVALUATION_FILENAME
    report_path = run_report_dir / FINAL_REPORT_FILENAME
    figure_path = run_report_dir / "figures" / CONFUSION_FIGURE_FILENAME
    if evaluation_path.exists():
        if ledger is None:
            raise ArtifactIntegrityError(
                "A final receipt exists without the global holdout ledger."
            )
        evaluation_payload = load_json_object(evaluation_path)
        if not report_path.is_file() or not figure_path.is_file():
            raise ArtifactIntegrityError(
                "Final receipt exists but its report bundle is incomplete."
            )
        report_bytes = report_path.read_bytes()
        figure_bytes = figure_path.read_bytes()
        _validate_evaluation_payload(
            evaluation_payload,
            run_manifest=run_manifest,
            artifact_manifest=artifact_manifest,
            split_manifest=split_manifest,
            selection_hashes=selection_hashes,
            report_bytes=report_bytes,
            figure_bytes=figure_bytes,
        )
        final_receipt_sha256 = sha256_path(evaluation_path)
        if (
            ledger.get("status") == "holdout_evaluation_complete"
            and ledger.get("final_receipt_sha256") != final_receipt_sha256
        ):
            raise ArtifactIntegrityError("Global ledger differs from the final receipt.")
        write_json_atomic_if_changed(
            ledger_path,
            {
                **ledger_identity,
                "status": "holdout_evaluation_complete",
                "final_receipt_sha256": final_receipt_sha256,
            },
        )
        return _final_artifacts(
            run_id,
            selected_model,
            float(threshold),
            pipeline_path,
            run_report_dir,
            cached=True,
        )
    if ledger is not None:
        raise FinalEvaluationError(
            "A global holdout ledger exists without a recoverable bundle; "
            "manual review is required."
        )
    if not confirm_final_evaluation:
        raise FinalEvaluationError(
            "Final holdout evaluation requires explicit --confirm-final-evaluation."
        )

    try:
        fitted_pipeline = joblib.load(pipeline_path)
    except Exception as error:
        raise ArtifactIntegrityError("Could not load the selected local pipeline.") from error
    if not isinstance(fitted_pipeline, Pipeline):
        raise ArtifactIntegrityError("Serialized model is not a scikit-learn Pipeline.")
    training_probe = fitted_pipeline.predict_proba(training_frame.loc[:4, list(FEATURE_COLUMNS)])[
        :, 1
    ]
    if not np.all(np.isfinite(training_probe)):
        raise ArtifactIntegrityError("Selected pipeline cannot produce finite probabilities.")

    if sha256_path(active_run_path) != active_run_sha256:
        raise FinalEvaluationError(
            "The active training run changed before holdout reservation; evaluation was aborted."
        )
    ticket_payload = {
        **ledger_identity,
        "status": "holdout_access_started",
    }
    _create_exclusive_ticket(ledger_path, ticket_payload)

    holdout_frame, holdout_manifest = holdout_loader(processed_dir, split_manifest_path)
    if holdout_manifest != split_manifest:
        raise ArtifactIntegrityError("Final holdout manifest differs from the reviewed split.")
    holdout_features = holdout_frame.loc[:, FEATURE_COLUMNS]
    holdout_target = holdout_frame.loc[:, TARGET_COLUMN]
    holdout_scores = np.asarray(fitted_pipeline.predict_proba(holdout_features)[:, 1], dtype=float)
    metrics = compute_final_metrics(holdout_target, holdout_scores, float(threshold))
    figure_bytes = _confusion_matrix_png(metrics)
    base_evaluation_payload: dict[str, Any] = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "receipt": "final_holdout_evaluation_complete",
        "selected_model": selected_model,
        "threshold": float(threshold),
        "score": "predict_proba[:, 1] (not assessed for calibration)",
        "holdout": {
            "filename": split_manifest["holdout"]["filename"],
            "rows": split_manifest["holdout"]["rows"],
            "sha256": split_manifest["holdout"]["sha256"],
            "application_reads_for_evaluation": 1,
        },
        "selection_receipts": selection_hashes,
        "pipeline_sha256": artifact_manifest["pipeline_sha256"],
        "metrics": metrics,
    }
    report_bytes = _build_final_report(
        run_manifest,
        cv_results,
        threshold_selection,
        base_evaluation_payload,
    ).encode("utf-8")
    evaluation_payload = {
        **base_evaluation_payload,
        "outputs": {
            "report_filename": FINAL_REPORT_FILENAME,
            "report_sha256": sha256_bytes(report_bytes),
            "confusion_figure_filename": f"figures/{CONFUSION_FIGURE_FILENAME}",
            "confusion_figure_sha256": sha256_bytes(figure_bytes),
        },
    }
    _publish_local_bundle(
        artifact_run_dir,
        evaluation_payload,
        report_bytes,
        figure_bytes,
    )
    _publish_versioned_bundle(
        run_report_dir,
        evaluation_payload,
        report_bytes,
        figure_bytes,
    )
    write_json_atomic_if_changed(
        ledger_path,
        {
            **ledger_identity,
            "status": "holdout_evaluation_complete",
            "final_receipt_sha256": sha256_path(evaluation_path),
        },
    )
    return _final_artifacts(
        run_id,
        selected_model,
        float(threshold),
        pipeline_path,
        run_report_dir,
        cached=False,
    )
