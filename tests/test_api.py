from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import predictive_maintenance.api as api_module
from predictive_maintenance.api import create_app
from predictive_maintenance.inference import (
    FinalMetricSnapshot,
    InferenceError,
    InferenceService,
    ModelArtifactError,
    ModelMetadata,
)
from predictive_maintenance.schemas import API_FEATURE_NAMES, PUBLIC_WARNINGS, PredictionRequest
from predictive_maintenance.validation import FEATURE_COLUMNS

RUN_ID = "0123456789abcdef"
THRESHOLD = 0.6


def _valid_payload() -> dict[str, object]:
    return {
        "type": "M",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1_500,
        "torque_nm": 40.0,
        "tool_wear_min": 100,
    }


class RecordingPipeline:
    def __init__(
        self,
        probabilities: object = ((0.25, 0.75),),
        *,
        error: Exception | None = None,
    ) -> None:
        self.probabilities = probabilities
        self.error = error
        self.frames: list[pd.DataFrame] = []

    def predict_proba(self, frame: pd.DataFrame) -> object:
        self.frames.append(frame.copy())
        if self.error is not None:
            raise self.error
        return self.probabilities


def _metadata(*, threshold: float = THRESHOLD) -> ModelMetadata:
    return ModelMetadata(
        run_id=RUN_ID,
        model_name="random_forest",
        threshold=threshold,
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
    probabilities: object = ((0.25, 0.75),),
    *,
    threshold: float = THRESHOLD,
    error: Exception | None = None,
) -> tuple[InferenceService, RecordingPipeline]:
    pipeline = RecordingPipeline(probabilities, error=error)
    return InferenceService(pipeline=pipeline, metadata=_metadata(threshold=threshold)), pipeline


@pytest.fixture
def service() -> InferenceService:
    loaded, _ = _service()
    return loaded


@pytest.fixture
def client(service: InferenceService) -> Iterator[TestClient]:
    with TestClient(create_app(service=service)) as test_client:
        yield test_client


def test_prediction_request_accepts_categories_numeric_json_and_unbounded_finite_values() -> None:
    for product_type in ("L", "M", "H"):
        payload = {
            "type": product_type,
            "air_temperature_k": 1_000_000,
            "process_temperature_k": 1_000_001,
            "rotational_speed_rpm": 1_000_002,
            "torque_nm": 1_000_003,
            "tool_wear_min": 1_000_004,
        }

        observation = PredictionRequest.model_validate(payload)

        assert observation.type == product_type
        assert isinstance(observation.air_temperature_k, float)
        assert isinstance(observation.process_temperature_k, float)
        assert isinstance(observation.torque_nm, float)
        assert observation.rotational_speed_rpm == 1_000_002
        assert observation.tool_wear_min == 1_000_004


@pytest.mark.parametrize("value", ["l", " L", "H ", "X", 1, None])
def test_prediction_request_rejects_unknown_or_non_string_type(value: object) -> None:
    payload = _valid_payload()
    payload["type"] = value

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("air_temperature_k", "300.0"),
        ("air_temperature_k", True),
        ("process_temperature_k", "310.0"),
        ("process_temperature_k", False),
        ("rotational_speed_rpm", "1500"),
        ("rotational_speed_rpm", 1500.0),
        ("rotational_speed_rpm", True),
        ("torque_nm", "40.0"),
        ("torque_nm", False),
        ("tool_wear_min", "100"),
        ("tool_wear_min", 100.0),
        ("tool_wear_min", False),
    ],
)
def test_prediction_request_rejects_numeric_coercion(field: str, value: object) -> None:
    payload = _valid_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


@pytest.mark.parametrize("field", ["air_temperature_k", "process_temperature_k", "torque_nm"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_prediction_request_rejects_non_finite_numbers(field: str, value: float) -> None:
    payload = _valid_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("air_temperature_k", 0.0),
        ("air_temperature_k", -0.1),
        ("process_temperature_k", 0.0),
        ("process_temperature_k", -0.1),
        ("rotational_speed_rpm", -1),
        ("torque_nm", -0.1),
        ("tool_wear_min", -1),
    ],
)
def test_prediction_request_enforces_only_semantic_lower_bounds(field: str, value: object) -> None:
    payload = _valid_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


def test_prediction_request_accepts_zero_for_non_negative_features() -> None:
    payload = _valid_payload()
    payload.update(rotational_speed_rpm=0, torque_nm=0, tool_wear_min=0)

    observation = PredictionRequest.model_validate(payload)

    assert observation.rotational_speed_rpm == 0
    assert observation.torque_nm == 0.0
    assert observation.tool_wear_min == 0


@pytest.mark.parametrize("missing", API_FEATURE_NAMES)
def test_prediction_request_requires_every_exact_field(missing: str) -> None:
    payload = _valid_payload()
    del payload[missing]

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


def test_prediction_request_forbids_extra_and_source_column_names() -> None:
    extra = _valid_payload()
    extra["Machine failure"] = 0
    source_names = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1_500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(extra)
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(source_names)


def test_prediction_request_maps_to_the_frozen_feature_names_and_order() -> None:
    observation = PredictionRequest.model_validate(_valid_payload())

    row = observation.to_model_row()

    assert tuple(row) == FEATURE_COLUMNS
    assert row == {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1_500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }


def test_inference_service_maps_one_row_and_uses_an_inclusive_threshold() -> None:
    loaded, pipeline = _service(((0.4, THRESHOLD),), threshold=THRESHOLD)
    observation = PredictionRequest.model_validate(_valid_payload())

    response = loaded.predict(observation)

    assert response.risk_score == THRESHOLD
    assert response.threshold == THRESHOLD
    assert response.predicted_failure is True
    assert response.decision_rule == "risk_score >= threshold"
    assert response.warnings == PUBLIC_WARNINGS
    assert len(pipeline.frames) == 1
    scored = pipeline.frames[0]
    assert tuple(scored.columns) == FEATURE_COLUMNS
    assert scored.shape == (1, len(FEATURE_COLUMNS))
    assert scored.iloc[0].to_dict() == observation.to_model_row()


def test_inference_service_classifies_a_score_immediately_below_threshold_as_negative() -> None:
    score = float(np.nextafter(THRESHOLD, 0.0))
    loaded, _ = _service(((1.0 - score, score),), threshold=THRESHOLD)

    response = loaded.predict(PredictionRequest.model_validate(_valid_payload()))

    assert response.risk_score == score
    assert response.predicted_failure is False


@pytest.mark.parametrize(
    "probabilities",
    [
        (0.25, 0.75),
        ((0.75,),),
        ((0.25, 0.75), (0.5, 0.5)),
        ((math.nan, 0.75),),
        ((0.25, math.nan),),
        ((0.25, math.inf),),
        ((-0.01, 0.75),),
        ((0.25, -0.01),),
        ((0.25, 1.01),),
        ((0.10, 0.75),),
    ],
)
def test_inference_service_rejects_invalid_probability_outputs(probabilities: object) -> None:
    loaded, _ = _service(probabilities)

    with pytest.raises(InferenceError):
        loaded.predict(PredictionRequest.model_validate(_valid_payload()))


def test_inference_service_wraps_pipeline_errors() -> None:
    loaded, _ = _service(error=RuntimeError("synthetic pipeline failure"))

    with pytest.raises(InferenceError, match="could not score"):
        loaded.predict(PredictionRequest.model_validate(_valid_payload()))


def test_health_model_info_and_predict_have_exact_public_shapes(client: TestClient) -> None:
    health = client.get("/health")
    model_info = client.get("/model-info")
    prediction = client.post("/predict", json=_valid_payload())

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_loaded": True, "run_id": RUN_ID}

    assert model_info.status_code == 200
    assert model_info.json() == {
        "service_name": "Machine Failure Risk Classifier",
        "run_id": RUN_ID,
        "model_name": "random_forest",
        "threshold": THRESHOLD,
        "decision_rule": "risk_score >= threshold",
        "score_calibrated": False,
        "dataset": "AI4I 2020 Predictive Maintenance (synthetic)",
        "request_fields": list(API_FEATURE_NAMES),
        "evaluation": {
            "average_precision": 0.65,
            "roc_auc": 0.90,
            "precision": 0.60,
            "precision_wilson_95": [0.23072428127601297, 0.8823792257673521],
            "recall": 0.75,
            "recall_wilson_95": [0.300641842582402, 0.9544127391902995],
            "f1": 2.0 / 3.0,
            "confusion_matrix": [[18, 2], [1, 3]],
            "holdout_rows": 24,
        },
        "warnings": list(PUBLIC_WARNINGS),
    }

    assert prediction.status_code == 200
    assert prediction.json() == {
        "run_id": RUN_ID,
        "risk_score": 0.75,
        "threshold": THRESHOLD,
        "predicted_failure": True,
        "decision_rule": "risk_score >= threshold",
        "domain_status": "within_reference_envelope",
        "decision_applicable": True,
        "warnings": list(PUBLIC_WARNINGS),
    }
    joined_warnings = " ".join(prediction.json()["warnings"]).lower()
    assert "sintético" in joined_warnings
    assert "no evaluado como probabilidad calibrada" in joined_warnings
    assert "no valida uso industrial" in joined_warnings


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {**_valid_payload(), "unexpected": 1},
        {**_valid_payload(), "rotational_speed_rpm": 1500.0},
        {**_valid_payload(), "air_temperature_k": "300"},
    ],
)
def test_predict_returns_422_for_invalid_contract(
    client: TestClient, payload: dict[str, object]
) -> None:
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert isinstance(response.json().get("detail"), list)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_predict_sanitizes_non_finite_validation_errors_as_json(
    client: TestClient, token: str
) -> None:
    body = json.dumps(_valid_payload()).replace("300.0", token, 1)

    response = client.post(
        "/predict",
        content=body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    payload = response.json()
    assert isinstance(payload.get("detail"), list)
    assert any(detail.get("loc")[-1] == "air_temperature_k" for detail in payload["detail"])
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("method", "path"),
    [("get", "/predict"), ("post", "/health"), ("post", "/model-info")],
)
def test_api_rejects_unsupported_methods(client: TestClient, method: str, path: str) -> None:
    response = client.request(method, path)

    assert response.status_code == 405


def test_inference_failure_is_reported_as_service_unavailable() -> None:
    loaded, _ = _service(error=RuntimeError("synthetic pipeline failure"))

    with TestClient(create_app(service=loaded)) as test_client:
        response = test_client.post("/predict", json=_valid_payload())

    assert response.status_code == 503
    assert isinstance(response.json().get("detail"), str)


def test_openapi_publishes_the_strict_unbounded_request_contract(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    schemas = document["components"]["schemas"]
    request_schema = schemas["PredictionRequest"]
    assert request_schema["additionalProperties"] is False
    assert set(request_schema["required"]) == set(API_FEATURE_NAMES)
    assert set(request_schema["properties"]) == set(API_FEATURE_NAMES)

    properties = request_schema["properties"]
    assert properties["type"]["enum"] == ["L", "M", "H"]
    assert properties["air_temperature_k"]["type"] == "number"
    assert properties["air_temperature_k"]["exclusiveMinimum"] == 0
    assert properties["process_temperature_k"]["type"] == "number"
    assert properties["process_temperature_k"]["exclusiveMinimum"] == 0
    assert properties["rotational_speed_rpm"]["type"] == "integer"
    assert properties["rotational_speed_rpm"]["minimum"] == 0
    assert properties["torque_nm"]["type"] == "number"
    assert properties["torque_nm"]["minimum"] == 0
    assert properties["tool_wear_min"]["type"] == "integer"
    assert properties["tool_wear_min"]["minimum"] == 0
    for feature in API_FEATURE_NAMES[1:]:
        assert "maximum" not in properties[feature]

    for response_schema in (
        "HealthResponse",
        "ModelInfoResponse",
        "PredictionResponse",
    ):
        assert schemas[response_schema]["additionalProperties"] is False


def _control_tag(document: str, field: str) -> str:
    match = re.search(
        rf'<(?:input|select)\b[^>]*\bname="{re.escape(field)}"[^>]*>',
        document,
        flags=re.DOTALL,
    )
    assert match is not None, f"Missing form control for {field}"
    return match.group(0)


def test_local_ui_and_static_assets_expose_the_json_prediction_flow(client: TestClient) -> None:
    index = client.get("/")
    stylesheet = client.get("/static/app.css")
    script = client.get("/static/app.js")
    favicon = client.get("/static/favicon.svg")

    assert index.status_code == 200
    assert index.headers["content-type"].startswith("text/html")
    assert '<form id="prediction-form"' in index.text
    assert "/static/app.css?v=1.0.0" in index.text
    assert "/static/app.js?v=1.0.0" in index.text
    assert "/static/favicon.svg?v=1.0.0" in index.text
    for field in API_FEATURE_NAMES:
        assert "required" in _control_tag(index.text, field)
    for field in API_FEATURE_NAMES[1:]:
        assert "max=" not in _control_tag(index.text, field)
    for field in ("rotational_speed_rpm", "torque_nm", "tool_wear_min"):
        assert 'min="0"' in _control_tag(index.text, field)
    page_text = index.text.lower()
    assert "dataset sintético" in page_text
    assert "no está calibrado" in page_text
    assert "no valida uso industrial" in page_text

    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert len(stylesheet.content) > 1_000
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert 'fetch("/predict"' in script.text
    assert "JSON.stringify" in script.text
    assert favicon.status_code == 200
    assert favicon.headers["content-type"].startswith("image/svg+xml")


@pytest.mark.parametrize("path", ["/", "/health", "/static/app.css"])
def test_security_headers_cover_ui_api_and_static_assets(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert "object-src 'none'" in response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "access-control-allow-origin" not in response.headers


def test_dynamic_responses_are_not_cached(client: TestClient) -> None:
    responses = (
        client.get("/"),
        client.get("/health"),
        client.get("/model-info"),
        client.post("/predict", json=_valid_payload()),
        client.get("/openapi.json"),
    )

    assert all(response.status_code == 200 for response in responses)
    assert all(response.headers["cache-control"] == "no-store" for response in responses)


def test_docs_are_disabled_or_have_a_csp_compatible_with_their_assets(client: TestClient) -> None:
    response = client.get("/docs")

    assert response.status_code in {200, 404}
    if response.status_code == 404:
        return
    policy = response.headers.get("content-security-policy")
    if policy is None:
        return
    external_urls = re.findall(r'https://[^"\s<]+', response.text)
    origins = {
        f"{parsed.scheme}://{parsed.netloc}" for url in external_urls if (parsed := urlsplit(url))
    }
    assert all(origin in policy for origin in origins)
    if re.search(r"<script(?:\s[^>]*)?>\s*[^<]", response.text):
        assert any(token in policy for token in ("'unsafe-inline'", "'nonce-", "'sha256-"))


def test_untrusted_host_is_rejected_before_reaching_the_local_service(client: TestClient) -> None:
    response = client.get("/health", headers={"host": "evil.example"})

    assert response.status_code == 400


def test_trusted_hosts_can_be_configured_for_a_reverse_proxy(
    monkeypatch: pytest.MonkeyPatch,
    service: InferenceService,
) -> None:
    monkeypatch.setenv(
        api_module.ALLOWED_HOSTS_ENV,
        "ml.example.test, localhost,ml.example.test",
    )

    with TestClient(create_app(service=service)) as configured_client:
        assert (
            configured_client.get("/health", headers={"host": "ml.example.test"}).status_code == 200
        )
        assert configured_client.get("/health", headers={"host": "evil.example"}).status_code == 400


@pytest.mark.parametrize("value", ["", "*", "https://ml.example.test", "ml.example.test/path"])
def test_invalid_trusted_host_configuration_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    service: InferenceService,
    value: str,
) -> None:
    monkeypatch.setenv(api_module.ALLOWED_HOSTS_ENV, value)

    with pytest.raises(RuntimeError, match=api_module.ALLOWED_HOSTS_ENV):
        create_app(service=service)


def test_run_uses_host_and_port_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    invocation: dict[str, object] = {}

    def record_run(application: str, **options: object) -> None:
        invocation.update(application=application, **options)

    monkeypatch.setenv(api_module.SERVER_HOST_ENV, "0.0.0.0")
    monkeypatch.setenv(api_module.SERVER_PORT_ENV, "8080")
    monkeypatch.setattr(api_module.uvicorn, "run", record_run)

    api_module.run()

    assert invocation == {
        "application": "predictive_maintenance.api:app",
        "host": "0.0.0.0",
        "port": 8080,
        "reload": False,
        "workers": 1,
    }


@pytest.mark.parametrize(
    "name,value", [("host", " "), ("port", "0"), ("port", "65536"), ("port", "x")]
)
def test_invalid_server_bind_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    if name == "host":
        monkeypatch.setenv(api_module.SERVER_HOST_ENV, value)
    else:
        monkeypatch.setenv(api_module.SERVER_PORT_ENV, value)

    with pytest.raises(RuntimeError):
        api_module._server_bind_from_environment()


def test_lifespan_uses_an_injected_service_without_loading_local_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    service: InferenceService,
) -> None:
    def forbidden_loader(*_args: Any, **_kwargs: Any) -> InferenceService:
        raise AssertionError("an injected synthetic service must bypass the artifact loader")

    monkeypatch.setattr(api_module, "load_inference_service", forbidden_loader)
    application = create_app(service=service)

    with TestClient(application) as test_client:
        assert application.state.inference_service is service
        assert test_client.get("/health").status_code == 200
        assert test_client.post("/predict", json=_valid_payload()).status_code == 200


def test_lifespan_loads_the_default_service_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    service: InferenceService,
) -> None:
    calls: list[str] = []

    def synthetic_loader() -> InferenceService:
        calls.append("load")
        return service

    monkeypatch.setattr(api_module, "load_inference_service", synthetic_loader)
    application = create_app()

    with TestClient(application) as test_client:
        assert test_client.get("/health").status_code == 200
        assert test_client.get("/model-info").status_code == 200
        assert test_client.post("/predict", json=_valid_payload()).status_code == 200

    assert calls == ["load"]


def test_lifespan_fails_closed_when_the_model_bundle_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_loader() -> InferenceService:
        raise ModelArtifactError("synthetic artifact mismatch")

    monkeypatch.setattr(api_module, "load_inference_service", invalid_loader)

    with pytest.raises(ModelArtifactError, match="artifact mismatch"), TestClient(create_app()):
        pass
