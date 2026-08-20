"""Integrity-checked local loading and inference for the frozen M3 pipeline."""

from __future__ import annotations

import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from predictive_maintenance.artifact_io import (
    ArtifactIntegrityError,
    canonical_json_bytes,
    load_json_object,
    sha256_bytes,
    sha256_path,
)
from predictive_maintenance.config import (
    DEFAULT_HOLDOUT_LEDGER_DIR,
    DEFAULT_MODEL_ARTIFACT_DIR,
    DEFAULT_MODEL_REPORT_DIR,
)
from predictive_maintenance.modeling import (
    ACTIVE_RUN_FILENAME,
    ARTIFACT_MANIFEST_FILENAME,
    MODEL_CANDIDATES,
    MODEL_RUN_SCHEMA_VERSION,
    PIPELINE_FILENAME,
    modeling_configuration,
)
from predictive_maintenance.schemas import (
    API_FEATURE_NAMES,
    DOMAIN_OUTSIDE_REFERENCE,
    DOMAIN_WITHIN_REFERENCE,
    NO_SCORE_WARNINGS,
    PUBLIC_WARNINGS,
    EvaluationMetricsResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from predictive_maintenance.uncertainty import classification_wilson_intervals
from predictive_maintenance.validation import FEATURE_COLUMNS

RUN_ID_PATTERN = re.compile(r"[0-9a-f]{16}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ModelArtifactError(RuntimeError):
    """Raised when the frozen model bundle cannot be trusted or loaded."""


class InferenceError(RuntimeError):
    """Raised when a loaded pipeline cannot score a valid observation."""


@dataclass(frozen=True)
class FinalMetricSnapshot:
    average_precision: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    holdout_rows: int

    def to_response(self) -> EvaluationMetricsResponse:
        precision_wilson_95, recall_wilson_95 = classification_wilson_intervals(
            self.confusion_matrix
        )
        return EvaluationMetricsResponse(
            average_precision=self.average_precision,
            roc_auc=self.roc_auc,
            precision=self.precision,
            precision_wilson_95=precision_wilson_95,
            recall=self.recall,
            recall_wilson_95=recall_wilson_95,
            f1=self.f1,
            confusion_matrix=self.confusion_matrix,
            holdout_rows=self.holdout_rows,
        )


@dataclass(frozen=True)
class ModelMetadata:
    run_id: str
    model_name: str
    threshold: float
    pipeline_sha256: str
    metrics: FinalMetricSnapshot


@dataclass(frozen=True)
class InferenceService:
    """Loaded immutable pipeline plus the reviewed metadata used by the API."""

    pipeline: Pipeline
    metadata: ModelMetadata

    def health(self) -> dict[str, str | bool]:
        return {"status": "ok", "model_loaded": True, "run_id": self.metadata.run_id}

    def model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            service_name="Machine Failure Risk Classifier",
            run_id=self.metadata.run_id,
            model_name=self.metadata.model_name,
            threshold=self.metadata.threshold,
            decision_rule="risk_score >= threshold",
            score_calibrated=False,
            dataset="AI4I 2020 Predictive Maintenance (synthetic)",
            request_fields=API_FEATURE_NAMES,
            evaluation=self.metadata.metrics.to_response(),
            warnings=PUBLIC_WARNINGS,
        )

    def predict(self, observation: PredictionRequest) -> PredictionResponse:
        domain_warnings = observation.reference_domain_warnings()
        if domain_warnings:
            return PredictionResponse(
                run_id=self.metadata.run_id,
                risk_score=None,
                threshold=self.metadata.threshold,
                predicted_failure=None,
                decision_rule="risk_score >= threshold",
                domain_status=DOMAIN_OUTSIDE_REFERENCE,
                decision_applicable=False,
                warnings=(*NO_SCORE_WARNINGS, *domain_warnings),
            )

        try:
            frame = pd.DataFrame(
                [observation.to_model_row()],
                columns=list(FEATURE_COLUMNS),
            )
            probabilities = np.asarray(self.pipeline.predict_proba(frame))
        except Exception as error:
            raise InferenceError("The local pipeline could not score this observation.") from error
        if probabilities.shape != (1, 2):
            raise InferenceError("The local pipeline returned an unexpected probability shape.")
        try:
            has_floating_dtype = np.issubdtype(probabilities.dtype, np.floating)
        except TypeError:
            has_floating_dtype = False
        if (
            not has_floating_dtype
            or not np.isfinite(probabilities).all()
            or (probabilities < 0.0).any()
            or (probabilities > 1.0).any()
            or not math.isclose(float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise InferenceError("The local pipeline returned invalid class probabilities.")
        score = float(probabilities[0, 1])
        return PredictionResponse(
            run_id=self.metadata.run_id,
            risk_score=score,
            threshold=self.metadata.threshold,
            predicted_failure=score >= self.metadata.threshold,
            decision_rule="risk_score >= threshold",
            domain_status=DOMAIN_WITHIN_REFERENCE,
            decision_applicable=True,
            warnings=PUBLIC_WARNINGS,
        )


def _safe_run_directory(root: Path, run_id: str) -> Path:
    resolved_root = root.resolve()
    candidate = root / run_id
    if candidate.resolve().parent != resolved_root:
        raise ModelArtifactError("Run directory escapes its configured root.")
    return candidate


def _is_probability(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0.0 <= float(value) <= 1.0
    )


def _runtime_is_compatible(expected: object) -> bool:
    """Require the serialization stack exactly and allow only Python patch-level drift."""
    if not isinstance(expected, dict) or set(expected) != {
        "python",
        "joblib",
        "matplotlib",
        "numpy",
        "pandas",
        "scikit_learn",
    }:
        return False
    expected_python = expected.get("python")
    if not isinstance(expected_python, str):
        return False
    expected_python_family = tuple(expected_python.split(".")[:2])
    current_python_family = tuple(platform.python_version().split(".")[:2])
    return expected_python_family == current_python_family and {
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    } == {key: expected.get(key) for key in ("joblib", "numpy", "pandas", "scikit_learn")}


def _validate_final_metrics(
    payload: object, holdout_rows: int, threshold: float
) -> FinalMetricSnapshot:
    if not isinstance(payload, dict):
        raise ModelArtifactError("Final evaluation metrics are missing.")
    required_rates = ("average_precision", "roc_auc", "precision", "recall", "f1")
    if any(not _is_probability(payload.get(key)) for key in required_rates):
        raise ModelArtifactError("Final evaluation metrics contain invalid rates.")
    matrix = payload.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for row in matrix
            for value in row
        )
        or sum(sum(row) for row in matrix) != holdout_rows
    ):
        raise ModelArtifactError("Final confusion matrix is invalid.")
    confusion_matrix = (
        (int(matrix[0][0]), int(matrix[0][1])),
        (int(matrix[1][0]), int(matrix[1][1])),
    )
    true_negative, false_positive = confusion_matrix[0]
    false_negative, true_positive = confusion_matrix[1]
    expected_precision = true_positive / max(true_positive + false_positive, 1)
    expected_recall = true_positive / max(true_positive + false_negative, 1)
    expected_f1 = (
        0.0
        if expected_precision + expected_recall == 0.0
        else 2.0 * expected_precision * expected_recall / (expected_precision + expected_recall)
    )
    expected_accuracy = (true_negative + true_positive) / holdout_rows
    expected_prevalence = (false_negative + true_positive) / holdout_rows
    if (
        payload.get("decision_rule") != "score >= threshold"
        or payload.get("threshold") != threshold
        or not _is_probability(payload.get("accuracy"))
        or not _is_probability(payload.get("target_prevalence"))
        or not math.isclose(payload["precision"], expected_precision, abs_tol=1e-15)
        or not math.isclose(payload["recall"], expected_recall, abs_tol=1e-15)
        or not math.isclose(payload["f1"], expected_f1, abs_tol=1e-15)
        or not math.isclose(payload["accuracy"], expected_accuracy, abs_tol=1e-15)
        or not math.isclose(payload["target_prevalence"], expected_prevalence, abs_tol=1e-15)
    ):
        raise ModelArtifactError("Final metrics disagree with their confusion matrix.")
    return FinalMetricSnapshot(
        average_precision=float(payload["average_precision"]),
        roc_auc=float(payload["roc_auc"]),
        precision=float(payload["precision"]),
        recall=float(payload["recall"]),
        f1=float(payload["f1"]),
        confusion_matrix=confusion_matrix,
        holdout_rows=holdout_rows,
    )


def _validate_selection_hashes(
    run_manifest: dict[str, Any],
    final_evaluation: dict[str, Any],
    ledger: dict[str, Any],
    run_manifest_path: Path,
) -> None:
    declared = run_manifest.get("selection_receipts")
    expected_final_hashes = {
        "run_manifest_sha256": sha256_path(run_manifest_path),
        "cv_results_sha256": declared.get("cv_results_sha256")
        if isinstance(declared, dict)
        else None,
        "threshold_selection_sha256": (
            declared.get("threshold_selection_sha256") if isinstance(declared, dict) else None
        ),
        "selection_report_sha256": (
            declared.get("selection_report_sha256") if isinstance(declared, dict) else None
        ),
        "cv_figure_sha256": declared.get("cv_figure_sha256")
        if isinstance(declared, dict)
        else None,
        "oof_figure_sha256": (
            declared.get("oof_figure_sha256") if isinstance(declared, dict) else None
        ),
    }
    if (
        not isinstance(declared, dict)
        or any(
            not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
            for value in expected_final_hashes.values()
        )
        or final_evaluation.get("selection_receipts") != expected_final_hashes
        or ledger.get("selection_receipts") != expected_final_hashes
    ):
        raise ModelArtifactError("Selection receipt hashes are inconsistent.")

    expected_files = {
        "cv_results_sha256": "cv_results_filename",
        "threshold_selection_sha256": "threshold_selection_filename",
        "selection_report_sha256": "selection_report_filename",
        "cv_figure_sha256": "cv_figure_filename",
        "oof_figure_sha256": "oof_figure_filename",
    }
    run_directory = run_manifest_path.parent.resolve()
    for hash_key, filename_key in expected_files.items():
        filename = declared.get(filename_key)
        if not isinstance(filename, str):
            raise ModelArtifactError("Selection receipt filenames are invalid.")
        path = run_manifest_path.parent / filename
        try:
            if not path.is_file() or not path.resolve().is_relative_to(run_directory):
                raise ModelArtifactError("Selection receipt path is invalid.")
        except OSError as error:
            raise ModelArtifactError("Selection receipt path is invalid.") from error
        if sha256_path(path) != declared.get(hash_key):
            raise ModelArtifactError("A versioned selection artifact failed SHA-256 validation.")


def _validate_final_outputs(payload: object, run_report_dir: Path) -> None:
    if not isinstance(payload, dict):
        raise ModelArtifactError("Final evaluation outputs are missing.")
    expected_filenames = {
        "report_filename": "M3_REPORT.md",
        "confusion_figure_filename": "figures/03_holdout_confusion_matrix.png",
    }
    if set(payload) != {
        "report_filename",
        "report_sha256",
        "confusion_figure_filename",
        "confusion_figure_sha256",
    } or any(payload.get(key) != value for key, value in expected_filenames.items()):
        raise ModelArtifactError("Final evaluation output identity is invalid.")
    resolved_root = run_report_dir.resolve()
    for filename_key, filename in expected_filenames.items():
        hash_key = filename_key.replace("filename", "sha256")
        expected_hash = payload.get(hash_key)
        path = run_report_dir / filename
        if (
            not isinstance(expected_hash, str)
            or SHA256_PATTERN.fullmatch(expected_hash) is None
            or not path.is_file()
            or not path.resolve().is_relative_to(resolved_root)
            or sha256_path(path) != expected_hash
        ):
            raise ModelArtifactError("A final evaluation output failed SHA-256 validation.")


def _load_inference_service(
    report_dir: Path = DEFAULT_MODEL_REPORT_DIR,
    artifact_root: Path = DEFAULT_MODEL_ARTIFACT_DIR,
    ledger_dir: Path = DEFAULT_HOLDOUT_LEDGER_DIR,
) -> InferenceService:
    """Load the evaluated pipeline without reading training, raw data or holdout files."""
    active = load_json_object(report_dir / ACTIVE_RUN_FILENAME)
    run_id = active.get("run_id")
    if (
        set(active) != {"schema_version", "run_id", "directory", "baseline_gate_passed"}
        or active.get("schema_version") != MODEL_RUN_SCHEMA_VERSION
        or not isinstance(run_id, str)
        or RUN_ID_PATTERN.fullmatch(run_id) is None
        or active.get("directory") != run_id
        or active.get("baseline_gate_passed") is not True
    ):
        raise ModelArtifactError("Active model pointer is invalid or incomplete.")

    run_report_dir = _safe_run_directory(report_dir, run_id)
    artifact_run_dir = _safe_run_directory(artifact_root, run_id)
    run_manifest_path = run_report_dir / "run_manifest.json"
    final_evaluation_path = run_report_dir / "final_evaluation.json"
    artifact_manifest_path = artifact_run_dir / ARTIFACT_MANIFEST_FILENAME
    pipeline_path = artifact_run_dir / PIPELINE_FILENAME
    run_manifest = load_json_object(run_manifest_path)
    artifact_manifest = load_json_object(artifact_manifest_path)
    final_evaluation = load_json_object(final_evaluation_path)

    configuration = run_manifest.get("configuration")
    configuration_sha256 = run_manifest.get("configuration_sha256")
    threshold = run_manifest.get("threshold")
    selected_model = run_manifest.get("selected_model")
    artifact = run_manifest.get("artifact")
    holdout = run_manifest.get("holdout")
    training = run_manifest.get("training")
    versions = run_manifest.get("versions")
    threshold_contract = configuration.get("threshold") if isinstance(configuration, dict) else None
    if (
        run_manifest.get("schema_version") != MODEL_RUN_SCHEMA_VERSION
        or run_manifest.get("run_id") != run_id
        or run_manifest.get("baseline_gate_passed") is not True
        or run_manifest.get("holdout_accessed_during_selection") is not False
        or selected_model not in MODEL_CANDIDATES
        or not _is_probability(threshold)
        or configuration != modeling_configuration()
        or not isinstance(configuration_sha256, str)
        or configuration_sha256 != sha256_bytes(canonical_json_bytes(configuration))
        or not isinstance(threshold_contract, dict)
        or threshold_contract.get("score") != "predict_proba[:, 1]"
        or threshold_contract.get("decision_rule") != "score >= threshold"
        or not isinstance(artifact, dict)
        or not isinstance(holdout, dict)
        or not isinstance(training, dict)
        or not isinstance(versions, dict)
        or not _runtime_is_compatible(versions)
    ):
        raise ModelArtifactError("Run manifest does not describe the frozen M3 model.")

    run_identity = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "training_sha256": training.get("sha256"),
        "configuration_sha256": configuration_sha256,
        "fold_plan_sha256": run_manifest.get("fold_plan_sha256"),
        "versions": versions,
    }
    if sha256_bytes(canonical_json_bytes(run_identity))[:16] != run_id:
        raise ModelArtifactError("Run ID does not match the frozen model identity.")

    pipeline_sha256 = artifact.get("pipeline_sha256")
    expected_artifact = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "training_sha256": training.get("sha256"),
        "configuration_sha256": configuration_sha256,
        "fold_plan_sha256": run_manifest.get("fold_plan_sha256"),
        "selected_model": selected_model,
        "threshold": threshold,
        "pipeline_filename": PIPELINE_FILENAME,
        "versions": run_manifest.get("versions"),
        "pipeline_sha256": pipeline_sha256,
    }
    if (
        artifact
        != {
            "directory": run_id,
            "pipeline_filename": PIPELINE_FILENAME,
            "pipeline_sha256": pipeline_sha256,
        }
        or artifact_manifest != expected_artifact
        or not isinstance(pipeline_sha256, str)
        or SHA256_PATTERN.fullmatch(pipeline_sha256) is None
        or sha256_path(pipeline_path) != pipeline_sha256
    ):
        raise ModelArtifactError("Pipeline identity or SHA-256 is inconsistent.")

    final_holdout = final_evaluation.get("holdout")
    holdout_sha256 = holdout.get("sha256")
    holdout_rows = holdout.get("rows")
    if (
        set(final_evaluation)
        != {
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
        or set(final_holdout or {})
        != {"filename", "rows", "sha256", "application_reads_for_evaluation"}
        or final_evaluation.get("schema_version") != MODEL_RUN_SCHEMA_VERSION
        or final_evaluation.get("receipt") != "final_holdout_evaluation_complete"
        or final_evaluation.get("run_id") != run_id
        or final_evaluation.get("selected_model") != selected_model
        or final_evaluation.get("threshold") != threshold
        or final_evaluation.get("pipeline_sha256") != pipeline_sha256
        or final_evaluation.get("score") != "predict_proba[:, 1] (not assessed for calibration)"
        or not isinstance(holdout_sha256, str)
        or SHA256_PATTERN.fullmatch(holdout_sha256) is None
        or isinstance(holdout_rows, bool)
        or not isinstance(holdout_rows, int)
        or holdout_rows <= 0
        or not isinstance(final_holdout, dict)
        or final_holdout.get("sha256") != holdout_sha256
        or final_holdout.get("rows") != holdout_rows
        or final_holdout.get("filename") != holdout.get("filename")
        or final_holdout.get("application_reads_for_evaluation") != 1
    ):
        raise ModelArtifactError("Final evaluation receipt is inconsistent.")

    ledger_path = ledger_dir / f"{holdout_sha256}.json"
    ledger = load_json_object(ledger_path)
    if (
        set(ledger)
        != {
            "schema_version",
            "holdout_sha256",
            "run_id",
            "selection_receipts",
            "status",
            "final_receipt_sha256",
        }
        or ledger.get("schema_version") != MODEL_RUN_SCHEMA_VERSION
        or ledger.get("holdout_sha256") != holdout_sha256
        or ledger.get("run_id") != run_id
        or ledger.get("status") != "holdout_evaluation_complete"
        or ledger.get("final_receipt_sha256") != sha256_path(final_evaluation_path)
    ):
        raise ModelArtifactError("Global holdout ledger is inconsistent.")
    _validate_selection_hashes(run_manifest, final_evaluation, ledger, run_manifest_path)
    _validate_final_outputs(final_evaluation.get("outputs"), run_report_dir)
    metrics = _validate_final_metrics(
        final_evaluation.get("metrics"), holdout_rows, float(threshold)
    )

    try:
        pipeline = joblib.load(pipeline_path)
    except Exception as error:
        raise ModelArtifactError("Could not deserialize the verified local pipeline.") from error
    if not isinstance(pipeline, Pipeline):
        raise ModelArtifactError("Serialized model is not a scikit-learn Pipeline.")
    classes = np.asarray(getattr(pipeline, "classes_", []))
    if classes.shape != (2,) or not np.array_equal(classes, np.array([0, 1])):
        raise ModelArtifactError("Pipeline classes are not exactly [0, 1].")
    feature_names = tuple(str(value) for value in getattr(pipeline, "feature_names_in_", ()))
    if feature_names != FEATURE_COLUMNS:
        raise ModelArtifactError("Pipeline feature order differs from the frozen allowlist.")
    expected_estimator_class = {
        "logistic_regression": "LogisticRegression",
        "random_forest": "RandomForestClassifier",
    }[str(selected_model)]
    estimator = pipeline.named_steps.get("model")
    if type(estimator).__name__ != expected_estimator_class:
        raise ModelArtifactError("Pipeline estimator differs from the selected model.")

    metadata = ModelMetadata(
        run_id=run_id,
        model_name=str(selected_model),
        threshold=float(threshold),
        pipeline_sha256=pipeline_sha256,
        metrics=metrics,
    )
    service = InferenceService(pipeline=pipeline, metadata=metadata)
    probe = PredictionRequest(
        type="L",
        air_temperature_k=300.0,
        process_temperature_k=310.0,
        rotational_speed_rpm=1_500,
        torque_nm=40.0,
        tool_wear_min=100,
    )
    try:
        service.predict(probe)
    except InferenceError as error:
        raise ModelArtifactError("Verified pipeline failed its inference probe.") from error
    return service


def load_inference_service(
    report_dir: Path = DEFAULT_MODEL_REPORT_DIR,
    artifact_root: Path = DEFAULT_MODEL_ARTIFACT_DIR,
    ledger_dir: Path = DEFAULT_HOLDOUT_LEDGER_DIR,
) -> InferenceService:
    """Translate all filesystem and schema failures into one startup-safe error type."""
    try:
        return _load_inference_service(report_dir, artifact_root, ledger_dir)
    except ModelArtifactError:
        raise
    except (
        ArtifactIntegrityError,
        OSError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise ModelArtifactError("Could not validate the frozen local model bundle.") from error
