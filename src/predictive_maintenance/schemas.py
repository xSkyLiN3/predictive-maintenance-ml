"""Strict public schemas for the local M4 inference application."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

API_FEATURE_NAMES = (
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)

SYNTHETIC_DATA_WARNING = (
    "AI4I 2020 is a synthetic dataset; this result has not been validated for real-world "
    "industrial use."
)
UNCALIBRATED_SCORE_WARNING = (
    "risk_score is a predict_proba score that has not been evaluated as a calibrated probability."
)
REFERENCE_ENVELOPE_WARNING = (
    "The domain check uses only marginal ranges from the synthetic AI4I training data; falling "
    "within them does not demonstrate that an observation is realistic."
)
PUBLIC_WARNINGS = (
    SYNTHETIC_DATA_WARNING,
    UNCALIBRATED_SCORE_WARNING,
    REFERENCE_ENVELOPE_WARNING,
)
NO_SCORE_WARNINGS = (
    SYNTHETIC_DATA_WARNING,
    REFERENCE_ENVELOPE_WARNING,
)

DOMAIN_WITHIN_REFERENCE = "within_reference_envelope"
DOMAIN_OUTSIDE_REFERENCE = "outside_reference_envelope"

# These extrema come exclusively from the versioned training-only EDA summary at
# reports/eda/summary.json (scope=training_only, training_rows=8000, holdout_profiled=false).
# They are hard-coded so inference remains data-independent. They are only an educational
# abstention rule, not physical limits, an industrial schema or a complete OOD detector.
AI4I_TRAINING_REFERENCE_ENVELOPE: dict[str, tuple[float | int, float | int, str]] = {
    "air_temperature_k": (295.3, 304.5, "K"),
    "process_temperature_k": (305.7, 313.8, "K"),
    "rotational_speed_rpm": (1168, 2886, "rpm"),
    "torque_nm": (3.8, 76.6, "Nm"),
    "tool_wear_min": (0, 253, "min"),
}
REFERENCE_FIELD_LABELS = {
    "air_temperature_k": "Air temperature",
    "process_temperature_k": "Process temperature",
    "rotational_speed_rpm": "Rotational speed",
    "torque_nm": "Torque",
    "tool_wear_min": "Tool wear",
}


class StrictResponseModel(BaseModel):
    """Forbid accidental response fields and keep serialization predictable."""

    model_config = ConfigDict(extra="forbid")


class PredictionRequest(BaseModel):
    """One local-API observation; this is not the source CSV contract."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["L", "M", "H"] = Field(description="AI4I product type.")
    air_temperature_k: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description="Finite air temperature greater than zero kelvin.",
    )
    process_temperature_k: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description="Finite process temperature greater than zero kelvin.",
    )
    rotational_speed_rpm: int = Field(
        ge=0,
        description="Non-negative integer rotational speed, in rpm.",
    )
    torque_nm: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        description="Finite non-negative torque, in Nm.",
    )
    tool_wear_min: int = Field(
        ge=0,
        description="Non-negative integer tool wear, in minutes.",
    )

    def to_model_row(self) -> dict[str, str | float | int]:
        """Map API names to the exact frozen model feature names."""
        return {
            "Type": self.type,
            "Air temperature [K]": self.air_temperature_k,
            "Process temperature [K]": self.process_temperature_k,
            "Rotational speed [rpm]": self.rotational_speed_rpm,
            "Torque [Nm]": self.torque_nm,
            "Tool wear [min]": self.tool_wear_min,
        }

    def reference_domain_warnings(self) -> tuple[str, ...]:
        """Describe marginal AI4I-envelope violations without claiming physical limits."""
        warnings: list[str] = []
        for field_name, (
            minimum,
            maximum,
            unit,
        ) in AI4I_TRAINING_REFERENCE_ENVELOPE.items():
            value = getattr(self, field_name)
            if value < minimum or value > maximum:
                label = REFERENCE_FIELD_LABELS[field_name]
                warnings.append(
                    f"{label} ({field_name}) is outside the marginal AI4I training envelope "
                    f"[{minimum}, {maximum}] {unit}; no score or classification was generated."
                )
        return tuple(warnings)


class HealthResponse(StrictResponseModel):
    status: Literal["ok"]
    model_loaded: Literal[True]
    run_id: str


class EvaluationMetricsResponse(StrictResponseModel):
    average_precision: float = Field(ge=0.0, le=1.0)
    roc_auc: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    precision_wilson_95: tuple[float, float]
    recall: float = Field(ge=0.0, le=1.0)
    recall_wilson_95: tuple[float, float]
    f1: float = Field(ge=0.0, le=1.0)
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    holdout_rows: int = Field(gt=0)


class ModelInfoResponse(StrictResponseModel):
    service_name: Literal["Machine Failure Risk Classifier"]
    run_id: str
    model_name: str
    threshold: float = Field(ge=0.0, le=1.0)
    decision_rule: Literal["risk_score >= threshold"]
    score_calibrated: Literal[False]
    dataset: Literal["AI4I 2020 Predictive Maintenance (synthetic)"]
    request_fields: tuple[str, ...]
    evaluation: EvaluationMetricsResponse
    warnings: tuple[str, ...]


class PredictionResponse(StrictResponseModel):
    run_id: str
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    predicted_failure: bool | None
    decision_rule: Literal["risk_score >= threshold"]
    domain_status: Literal[
        "within_reference_envelope",
        "outside_reference_envelope",
    ]
    decision_applicable: bool
    warnings: tuple[str, ...]

    @model_validator(mode="after")
    def validate_domain_decision_consistency(self) -> PredictionResponse:
        """Make it impossible to publish an actionable decision outside the reference domain."""
        if self.domain_status == DOMAIN_WITHIN_REFERENCE:
            if (
                self.risk_score is None
                or self.predicted_failure is None
                or self.decision_applicable is not True
            ):
                raise ValueError("An in-domain response requires a score and decision.")
        elif (
            self.risk_score is not None
            or self.predicted_failure is not None
            or self.decision_applicable is not False
        ):
            raise ValueError("An out-of-domain response must withhold its score and decision.")
        return self
