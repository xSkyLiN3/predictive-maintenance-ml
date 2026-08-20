# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# The model identity includes the Python patch version used for training. This
# historical builder image reproduces the frozen 3.12.0 run, while the shipped
# runtime uses a current, separately pinned Python 3.12 image.
ARG MODEL_BUILDER_IMAGE=python:3.12.0-slim-bookworm@sha256:19a6235339a74eca01227b03629f63b6f5020abc21142436eced6ec3a9839a76
ARG RUNTIME_IMAGE=python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

FROM ${MODEL_BUILDER_IMAGE} AS wheel-builder

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

FROM ${MODEL_BUILDER_IMAGE} AS model-builder

ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements/constraints-py312.txt /tmp/constraints-py312.txt
COPY --from=wheel-builder /wheels /wheels

RUN python -m pip install "pip==26.2.1" \
    && python -m pip install \
        --constraint /tmp/constraints-py312.txt \
        /wheels/machine_failure_risk_classifier-1.0.0-py3-none-any.whl \
    && python -m pip check

# Only the split contract and curated receipts enter the builder. The source
# snapshot and both partitions are downloaded/materialized inside this stage.
COPY data/split_manifest.json data/split_manifest.json
COPY reports/modeling reports/modeling
COPY reports/holdout_access reports/holdout_access

# `train` reads only train.csv. It reconstructs the frozen pipeline and checks
# the existing versioned receipts; evaluate-holdout is intentionally absent.
RUN mkdir -p data/raw data/processed artifacts/m3 \
    && machine-failure-data download \
    && machine-failure-data split \
    && machine-failure-data train \
    && python -c "from predictive_maintenance.inference import load_inference_service; service = load_inference_service(); assert service.metadata.run_id == 'b15bab7b54bc2e1f'" \
    && rm -rf data/raw data/processed

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

COPY --from=model-builder --chown=10001:10001 /build/reports/modeling reports/modeling
COPY --from=model-builder --chown=10001:10001 /build/reports/holdout_access reports/holdout_access
COPY --from=model-builder --chown=10001:10001 /build/artifacts/m3 artifacts/m3

RUN test ! -e /app/data \
    && ! find /app -type f \( -name 'train.csv' -o -name 'holdout.csv' -o -name 'ai4i2020.csv' \) -print -quit | grep -q .

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['MACHINE_FAILURE_PORT'] + '/health', timeout=3).read()"]

CMD ["machine-failure-app"]
