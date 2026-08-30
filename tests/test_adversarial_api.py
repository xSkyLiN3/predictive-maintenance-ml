"""Adversarial API checks that never load project data or the real model artifact."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import predictive_maintenance.inference as inference_module
from predictive_maintenance.api import create_app
from predictive_maintenance.inference import (
    FinalMetricSnapshot,
    InferenceError,
    InferenceService,
    ModelMetadata,
)
from predictive_maintenance.schemas import (
    DOMAIN_OUTSIDE_REFERENCE,
    DOMAIN_WITHIN_REFERENCE,
    PredictionRequest,
)

RUN_ID = "0123456789abcdef"
THRESHOLD = 0.6
MAX_REQUEST_BODY_BYTES = 16_384


def _valid_payload() -> dict[str, object]:
    return {
        "type": "M",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1_500,
        "torque_nm": 40.0,
        "tool_wear_min": 100,
    }


class SyntheticPipeline:
    def __init__(
        self,
        probabilities: object = np.array([[0.25, 0.75]], dtype=np.float64),
        *,
        error: Exception | None = None,
    ) -> None:
        self.probabilities = probabilities
        self.error = error
        self.calls = 0
        self.frames: list[pd.DataFrame] = []

    def predict_proba(self, frame: pd.DataFrame) -> object:
        self.calls += 1
        self.frames.append(frame.copy())
        if self.error is not None:
            raise self.error
        return self.probabilities


def _metadata() -> ModelMetadata:
    return ModelMetadata(
        run_id=RUN_ID,
        model_name="random_forest",
        threshold=THRESHOLD,
        pipeline_sha256="a" * 64,
        metrics=FinalMetricSnapshot(
            average_precision=0.65,
            roc_auc=0.90,
            precision=0.60,
            recall=0.75,
            f1=2.0 / 3.0,
            confusion_matrix=((18, 2), (1, 3)),
            holdout_rows=24,
        ),
    )


def _service(
    probabilities: object = np.array([[0.25, 0.75]], dtype=np.float64),
    *,
    error: Exception | None = None,
) -> tuple[InferenceService, SyntheticPipeline]:
    pipeline = SyntheticPipeline(probabilities, error=error)
    return InferenceService(pipeline=pipeline, metadata=_metadata()), pipeline


@pytest.fixture
def client_and_pipeline() -> Iterator[tuple[TestClient, SyntheticPipeline]]:
    service, pipeline = _service()
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        yield client, pipeline


def _assert_security_headers(response: Any) -> None:
    assert response.headers["cache-control"] == "no-store"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert "object-src 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


def _assert_within_response(payload: dict[str, object]) -> None:
    assert payload["domain_status"] == DOMAIN_WITHIN_REFERENCE
    assert payload["decision_applicable"] is True
    assert payload["risk_score"] == pytest.approx(0.75)
    assert payload["predicted_failure"] is True
    assert payload["threshold"] == THRESHOLD
    assert payload["decision_rule"] == "risk_score >= threshold"


def _assert_outside_response(payload: dict[str, object]) -> None:
    assert payload["domain_status"] == DOMAIN_OUTSIDE_REFERENCE
    assert payload["decision_applicable"] is False
    assert payload["risk_score"] is None
    assert payload["predicted_failure"] is None
    warnings = payload["warnings"]
    assert isinstance(warnings, list)
    assert any(
        "ai4i" in warning.lower()
        and "outside" in warning.lower()
        and "no score or classification was generated" in warning.lower()
        for warning in warnings
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("air_temperature_k", 295.3),
        ("air_temperature_k", 304.5),
        ("process_temperature_k", 305.7),
        ("process_temperature_k", 313.8),
        ("rotational_speed_rpm", 1_168),
        ("rotational_speed_rpm", 2_886),
        ("torque_nm", 3.8),
        ("torque_nm", 76.6),
        ("tool_wear_min", 0),
        ("tool_wear_min", 253),
    ],
)
def test_exact_reference_boundaries_are_scored(
    client_and_pipeline: tuple[TestClient, SyntheticPipeline],
    field: str,
    value: object,
) -> None:
    client, pipeline = client_and_pipeline
    payload = _valid_payload()
    payload[field] = value

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    _assert_within_response(response.json())
    assert pipeline.calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("air_temperature_k", 295.29999999999995),
        ("air_temperature_k", 304.50000000000006),
        ("process_temperature_k", 305.69999999999993),
        ("process_temperature_k", 313.80000000000007),
        ("rotational_speed_rpm", 1_167),
        ("rotational_speed_rpm", 2_887),
        ("torque_nm", 3.7999999999999994),
        ("torque_nm", 76.60000000000001),
        ("tool_wear_min", 254),
    ],
)
def test_outside_reference_domain_returns_no_decision_without_scoring(
    field: str,
    value: object,
) -> None:
    service, pipeline = _service(error=AssertionError("outside inputs must bypass the pipeline"))
    payload = _valid_payload()
    payload[field] = value

    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    _assert_outside_response(response.json())
    assert pipeline.calls == 0


@pytest.mark.parametrize("field", ["rotational_speed_rpm", "tool_wear_min"])
def test_arbitrarily_large_integers_are_outside_not_internal_errors(field: str) -> None:
    service, pipeline = _service(error=AssertionError("huge outside values must not be scored"))
    payload = _valid_payload()
    payload[field] = 10**309

    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        response = client.post("/predict", json=payload)
        health = client.get("/health")

    assert response.status_code == 200
    _assert_outside_response(response.json())
    assert pipeline.calls == 0
    assert health.status_code == 200


@pytest.mark.parametrize("duplicate_field", ["type", "torque_nm"])
def test_duplicate_json_keys_are_rejected_with_400_and_security_headers(
    client_and_pipeline: tuple[TestClient, SyntheticPipeline], duplicate_field: str
) -> None:
    client, pipeline = client_and_pipeline
    serialized = json.dumps(_valid_payload(), separators=(",", ":"))
    original = json.dumps(_valid_payload()[duplicate_field])
    duplicate = json.dumps("H" if duplicate_field == "type" else 41.0)
    body = serialized.replace(
        f'"{duplicate_field}":{original}',
        f'"{duplicate_field}":{original},"{duplicate_field}":{duplicate}',
    )

    response = client.post("/predict", content=body, headers={"content-type": "application/json"})

    assert response.status_code == 400
    assert response.json()["detail"] == f"The JSON object repeats the key {duplicate_field!r}."
    _assert_security_headers(response)
    assert pipeline.calls == 0


def test_request_body_limit_accepts_exact_boundary_and_rejects_next_byte(
    client_and_pipeline: tuple[TestClient, SyntheticPipeline],
) -> None:
    client, pipeline = client_and_pipeline
    serialized = json.dumps(_valid_payload(), separators=(",", ":")).encode()
    exact = serialized + b" " * (MAX_REQUEST_BODY_BYTES - len(serialized))
    oversized = exact + b" "

    accepted = client.post("/predict", content=exact, headers={"content-type": "application/json"})
    calls_after_accepted = pipeline.calls
    rejected = client.post(
        "/predict", content=oversized, headers={"content-type": "application/json"}
    )

    assert len(exact) == MAX_REQUEST_BODY_BYTES
    assert len(oversized) == MAX_REQUEST_BODY_BYTES + 1
    assert accepted.status_code == 200
    _assert_within_response(accepted.json())
    assert rejected.status_code == 413
    assert rejected.json()["detail"] == "The /predict request body exceeds the 16 KiB limit."
    _assert_security_headers(rejected)
    assert pipeline.calls == calls_after_accepted


def test_body_limit_measures_bytes_instead_of_trusting_content_length(
    client_and_pipeline: tuple[TestClient, SyntheticPipeline],
) -> None:
    client, pipeline = client_and_pipeline
    serialized = json.dumps(_valid_payload(), separators=(",", ":")).encode()
    oversized = serialized + b" " * (MAX_REQUEST_BODY_BYTES + 1 - len(serialized))

    response = client.post(
        "/predict",
        content=oversized,
        headers={"content-type": "application/json", "content-length": "1"},
    )

    assert response.status_code == 413
    _assert_security_headers(response)
    assert pipeline.calls == 0


def test_dataframe_failure_is_inference_error_directly_and_503_over_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, pipeline = _service()
    observation = PredictionRequest.model_validate(_valid_payload())

    def broken_dataframe(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise OverflowError("synthetic dataframe failure")

    monkeypatch.setattr(inference_module.pd, "DataFrame", broken_dataframe)

    with pytest.raises(InferenceError, match="could not score"):
        service.predict(observation)
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 503
    assert "synthetic dataframe failure" not in response.text
    _assert_security_headers(response)
    assert pipeline.calls == 0


def test_pipeline_failure_is_inference_error_directly_and_503_over_http() -> None:
    service, pipeline = _service(error=RuntimeError("synthetic private pipeline failure"))
    observation = PredictionRequest.model_validate(_valid_payload())

    with pytest.raises(InferenceError, match="could not score"):
        service.predict(observation)
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 503
    assert "synthetic private pipeline failure" not in response.text
    _assert_security_headers(response)
    assert pipeline.calls == 2


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([["0.25", "0.75"]]),
        np.array([[False, True]], dtype=bool),
        np.array([[0, 1]], dtype=np.int64),
    ],
    ids=["string", "bool", "integer"],
)
def test_non_floating_probability_dtypes_are_rejected_directly_and_over_http(
    probabilities: np.ndarray,
) -> None:
    service, _ = _service(probabilities)
    observation = PredictionRequest.model_validate(_valid_payload())

    with pytest.raises(InferenceError, match="probabilit"):
        service.predict(observation)
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 503
    _assert_security_headers(response)


def test_float32_probability_output_remains_supported() -> None:
    service, _ = _service(np.array([[0.25, 0.75]], dtype=np.float32))

    response = service.predict(PredictionRequest.model_validate(_valid_payload()))

    assert response.risk_score == pytest.approx(0.75)
    assert response.predicted_failure is True


def test_404_and_422_responses_also_receive_security_and_no_store_headers(
    client_and_pipeline: tuple[TestClient, SyntheticPipeline],
) -> None:
    client, pipeline = client_and_pipeline

    not_found = client.get("/not-a-real-route")
    invalid = client.post("/predict", json={})

    assert not_found.status_code == 404
    assert invalid.status_code == 422
    _assert_security_headers(not_found)
    _assert_security_headers(invalid)
    assert pipeline.calls == 0
