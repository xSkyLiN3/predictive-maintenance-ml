# Demo container and deployment

This document covers the `v1.0.1` image and its deployment procedure. It does not claim that this
release candidate is already running publicly. Service availability does not turn the educational
classifier into an industrial system.

The public endpoint is <https://ml.nightstrike.cloud>, behind Nginx and TLS. Until `v1.0.1` is
deployed, that endpoint may still serve the previous release. After deployment, verify the version,
English interface, API messages, and asset cache keys before announcing the release. The host must
publish the container port only on loopback. The container must run as UID/GID `10001`, use a
read-only filesystem, have no capabilities, and have explicit CPU, memory, and process limits.

## Build guarantees

The `Dockerfile` uses two stages:

1. builds the package wheel;
2. installs the wheel and the exact evaluated pipeline in a separate runtime image.

The build does not download the dataset, run `split`, `train`, or `evaluate-holdout`, and contains
no CSV files. Re-training or re-rendering across platforms is not expected to reproduce identical
Joblib and Matplotlib bytes. Therefore, the 1.25 MB evaluated pipeline and its manifest are
versioned alongside the receipts, reports, and ledgers instead of being regenerated in the image.
Before loading the pipeline, the loader reconciles its SHA-256, run, versions, selection receipts,
final evaluation, and global ledger. The API does not accept user-supplied artifacts.

Both base images are pinned by digest and the Python dependencies by version in
`requirements/constraints-py312.txt`. The build needs outbound access to Docker Hub and PyPI, but
not to UCI. `pip` and `setuptools` are removed after installing and verifying the wheel because the
service does not need package managers at runtime. Reproducing training remains available as a
separate workflow and never reopens the consumed holdout.

## Build and run locally

From the repository root:

```bash
docker build --tag machine-failure-risk-classifier:1.0.1 .
docker run --rm \
  --name machine-failure-demo \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --cpus 1.0 \
  --memory 512m \
  --pids-limit 128 \
  --publish 127.0.0.1:8000:8000 \
  machine-failure-risk-classifier:1.0.1
```

Open <http://127.0.0.1:8000>. The image runs as UID/GID `10001`, declares a healthcheck, and does not
need a writable volume.

Useful checks:

```bash
docker inspect --format '{{.State.Health.Status}}' machine-failure-demo
docker exec machine-failure-demo id
docker exec machine-failure-demo test ! -e /app/data
```

## Environment variables

| Variable | Value in the image | Purpose |
|---|---|---|
| `MACHINE_FAILURE_HOST` | `0.0.0.0` | Address on which Uvicorn listens inside the container. |
| `MACHINE_FAILURE_PORT` | `8000` | Internal port, between `1` and `65535`. |
| `MACHINE_FAILURE_ALLOWED_HOSTS` | `127.0.0.1,localhost` | Explicit list of accepted `Host` values. |

To use another internal port:

```bash
docker run --rm \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --cpus 1.0 \
  --memory 512m \
  --pids-limit 128 \
  --publish 127.0.0.1:8080:8080 \
  --env MACHINE_FAILURE_PORT=8080 \
  machine-failure-risk-classifier:1.0.1
```

`MACHINE_FAILURE_ALLOWED_HOSTS` does not accept `*`, schemes, or paths. For example, a proxy for
`ml.nightstrike.cloud` must include the domain:

```bash
--env MACHINE_FAILURE_ALLOWED_HOSTS=ml.nightstrike.cloud,127.0.0.1,localhost
```

The proxy must preserve `Host`, terminate HTTPS, and apply rate and size limits. The API has no
authentication, persistence, or built-in rate limiting; therefore, the container must not be
published directly to the Internet. The application does not store inputs, but proxy and platform
logs must be configured consistently with that policy.

## CI

`.github/workflows/ci.yml` runs:

- Ruff, formatting, `pip check`, and pytest on Python 3.12 on Ubuntu and Windows;
- `sdist` and `wheel` builds, isolated installation, and the full suite on each distribution;
- a real image build and Trivy scan for `HIGH`/`CRITICAL` vulnerabilities with a fix available;
- waiting for the healthcheck, a `/health` smoke test, a non-root user check, and verification that
  no CSV files are present at runtime.

Official Actions are pinned by SHA and annotated with their version. The workflow requests only
`contents: read`; it does not publish packages or images or deploy infrastructure.

## Operational limits

- This is an educational demo with a synthetic dataset, not a maintenance system.
- There is no guarantee of availability, calibration, joint OOD detection, or drift monitoring.
- A startup failure indicates that the pipeline or one of its receipts did not pass integrity
  validation; this fail-closed behavior must not be disabled.
- Changing NumPy, pandas, scikit-learn, or joblib dependencies requires rebuilding and explicitly
  reviewing artifact compatibility.
