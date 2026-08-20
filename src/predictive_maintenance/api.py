"""FastAPI application for the Machine Failure Risk Classifier demo."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from predictive_maintenance import __version__
from predictive_maintenance.inference import (
    InferenceError,
    InferenceService,
    load_inference_service,
)
from predictive_maintenance.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

WEB_DIRECTORY = Path(__file__).with_name("web")
MAX_PREDICT_BODY_BYTES = 16 * 1024
LOGGER = logging.getLogger(__name__)
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000
DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "testserver")
SERVER_HOST_ENV = "MACHINE_FAILURE_HOST"
SERVER_PORT_ENV = "MACHINE_FAILURE_PORT"
ALLOWED_HOSTS_ENV = "MACHINE_FAILURE_ALLOWED_HOSTS"


class DuplicateJSONKeyError(ValueError):
    """Raised while decoding a JSON object that repeats a key."""


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(key)
        result[key] = value
    return result


def _is_json_media_type(value: str) -> bool:
    media_type = value.split(";", 1)[0].strip().lower()
    return media_type == "application/json" or (
        media_type.startswith("application/") and media_type.endswith("+json")
    )


async def _guard_predict_body(request: Request) -> JSONResponse | None:
    """Bound and inspect prediction JSON before FastAPI discards duplicate-key evidence."""
    if request.method != "POST" or request.url.path != "/predict":
        return None

    declared_length = request.headers.get("content-length")
    if declared_length is not None:
        try:
            parsed_length = int(declared_length)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Content-Length no es válido."},
            )
        if parsed_length < 0:
            return JSONResponse(
                status_code=400,
                content={"detail": "Content-Length no es válido."},
            )
        if parsed_length > MAX_PREDICT_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "El cuerpo de /predict supera el máximo de 16 KiB."},
            )

    try:
        body = await request.body()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"detail": "No se pudo leer el cuerpo de la solicitud."},
        )
    if len(body) > MAX_PREDICT_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": "El cuerpo de /predict supera el máximo de 16 KiB."},
        )

    if not _is_json_media_type(request.headers.get("content-type", "")):
        return None
    try:
        json.loads(body, object_pairs_hook=_unique_json_object)
    except DuplicateJSONKeyError as error:
        key = str(error)[:80]
        return JSONResponse(
            status_code=400,
            content={"detail": f"El JSON repite la clave {key!r}."},
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        # Preserve FastAPI's normal malformed-body response and its sanitized validation shape.
        return None
    return None


def get_service(request: Request) -> InferenceService:
    """Return the model loaded once during lifespan startup."""
    loaded = getattr(request.app.state, "inference_service", None)
    if not isinstance(loaded, InferenceService):
        raise HTTPException(status_code=503, detail="El servicio del modelo no está disponible.")
    return loaded


def _allowed_hosts_from_environment() -> tuple[str, ...]:
    """Return explicit trusted hosts while refusing an allow-all deployment."""
    raw_value = os.getenv(ALLOWED_HOSTS_ENV)
    if raw_value is None:
        return DEFAULT_ALLOWED_HOSTS

    hosts = tuple(dict.fromkeys(item.strip() for item in raw_value.split(",") if item.strip()))
    if not hosts:
        raise RuntimeError(f"{ALLOWED_HOSTS_ENV} must contain at least one host.")
    if any(
        host == "*"
        or "://" in host
        or "/" in host
        or any(character.isspace() for character in host)
        for host in hosts
    ):
        raise RuntimeError(
            f"{ALLOWED_HOSTS_ENV} must contain comma-separated hostnames without schemes or paths."
        )
    return hosts


def _server_bind_from_environment() -> tuple[str, int]:
    """Parse the Uvicorn bind address without weakening local defaults."""
    host = os.getenv(SERVER_HOST_ENV, DEFAULT_SERVER_HOST).strip()
    if not host or any(character.isspace() for character in host):
        raise RuntimeError(f"{SERVER_HOST_ENV} must be a non-empty host or IP address.")

    raw_port = os.getenv(SERVER_PORT_ENV, str(DEFAULT_SERVER_PORT)).strip()
    try:
        port = int(raw_port)
    except ValueError as error:
        raise RuntimeError(f"{SERVER_PORT_ENV} must be an integer from 1 to 65535.") from error
    if not 1 <= port <= 65_535:
        raise RuntimeError(f"{SERVER_PORT_ENV} must be an integer from 1 to 65535.")
    return host, port


def create_app(service: InferenceService | None = None) -> FastAPI:
    """Create the application; inject a service to keep tests data-independent."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.inference_service = service or load_inference_service()
        yield

    application = FastAPI(
        title="Machine Failure Risk Classifier",
        version=__version__,
        description=(
            "Demo educativa sobre AI4I 2020 sintético. "
            "El risk_score no es una probabilidad calibrada."
        ),
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(_allowed_hosts_from_environment()),
    )
    application.mount("/static", StaticFiles(directory=WEB_DIRECTORY), name="static")

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        # FastAPI's default body echoes the rejected input. Besides being unnecessary for clients,
        # a raw JSON NaN/Infinity then makes JSON serialization fail and turns a 422 into a 500.
        detail = [
            {key: item[key] for key in ("type", "loc", "msg") if key in item}
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    @application.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await _guard_predict_body(request)
        if response is None:
            try:
                response = await call_next(request)
            except Exception:
                LOGGER.exception("Unhandled exception while processing an HTTP request.")
                response = JSONResponse(
                    status_code=500,
                    content={"detail": "Error interno inesperado."},
                )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; base-uri 'none'; "
            "object-src 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        return response

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_DIRECTORY / "index.html")

    @application.get("/health", response_model=HealthResponse)
    async def health(
        loaded: Annotated[InferenceService, Depends(get_service)],
    ) -> HealthResponse:
        return HealthResponse.model_validate(loaded.health())

    @application.get("/model-info", response_model=ModelInfoResponse)
    async def model_info(
        loaded: Annotated[InferenceService, Depends(get_service)],
    ) -> ModelInfoResponse:
        return loaded.model_info()

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(
        observation: PredictionRequest,
        loaded: Annotated[InferenceService, Depends(get_service)],
    ) -> PredictionResponse:
        try:
            return loaded.predict(observation)
        except InferenceError as error:
            raise HTTPException(
                status_code=503,
                detail="El modelo local no pudo procesar esta observación.",
            ) from error

    return application


app = create_app()


def run() -> None:
    """Run one Uvicorn worker using the configured bind address."""
    host, port = _server_bind_from_environment()
    uvicorn.run(
        "predictive_maintenance.api:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
    )
