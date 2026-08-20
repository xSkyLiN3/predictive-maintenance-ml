# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# The wheel builder remains pinned independently from the shipped runtime.
ARG BUILD_IMAGE=python:3.12.0-slim-bookworm@sha256:19a6235339a74eca01227b03629f63b6f5020abc21142436eced6ec3a9839a76
ARG RUNTIME_IMAGE=python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

FROM ${BUILD_IMAGE} AS wheel-builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install \
        "pip==26.2.1" \
        "build==1.5.0" \
        "packaging==26.3" \
        "pyproject-hooks==1.2.0" \
        "setuptools==84.0.0" \
        "wheel==0.48.0" \
    && python -m build --wheel --no-isolation --outdir /wheels

FROM ${RUNTIME_IMAGE} AS runtime

LABEL org.opencontainers.image.title="Machine Failure Risk Classifier" \
      org.opencontainers.image.description="Educational AI4I 2020 machine-failure risk demo" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

ENV MACHINE_FAILURE_ALLOWED_HOSTS="127.0.0.1,localhost" \
    MACHINE_FAILURE_HOST="0.0.0.0" \
    MACHINE_FAILURE_PORT="8000" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin app

COPY requirements/constraints-py312.txt /tmp/constraints-py312.txt
COPY --from=wheel-builder /wheels /wheels

RUN python -m pip install "pip==26.2.1" \
    && python -m pip install \
        --constraint /tmp/constraints-py312.txt \
        /wheels/machine_failure_risk_classifier-1.0.0-py3-none-any.whl \
    && python -m pip check \
    && rm -rf /wheels /tmp/constraints-py312.txt

WORKDIR /app

COPY --chown=10001:10001 reports/modeling reports/modeling
COPY --chown=10001:10001 reports/holdout_access reports/holdout_access
COPY --chown=10001:10001 artifacts/m3/b15bab7b54bc2e1f/artifact_manifest.json artifacts/m3/b15bab7b54bc2e1f/artifact_manifest.json
COPY --chown=10001:10001 artifacts/m3/b15bab7b54bc2e1f/pipeline.joblib artifacts/m3/b15bab7b54bc2e1f/pipeline.joblib

USER 10001:10001

# The repository-owned artifact is checked against the frozen selection and
# single-use holdout receipts before it can be loaded. The runtime never
# downloads data, retrains, or accepts user-supplied model files.
RUN test ! -e /app/data \
    && ! find /app -type f \( -name 'train.csv' -o -name 'holdout.csv' -o -name 'ai4i2020.csv' \) -print -quit | grep -q . \
    && python -c "from predictive_maintenance.inference import load_inference_service; service = load_inference_service(); assert service.metadata.run_id == 'b15bab7b54bc2e1f'"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['MACHINE_FAILURE_PORT'] + '/health', timeout=3).read()"]

CMD ["machine-failure-app"]
