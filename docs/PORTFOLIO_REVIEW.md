# Portfolio and publication review

**Date:** 2026-08-17
**Adversarial update:** 2026-08-18
**Release update:** 2026-08-19
**Scope:** evidence from M5 closure and the technical gate for the educational M6 release.

## M6 update

The project owner approved publication and deployment after local review. MIT was adopted for the
code, separately from the dataset's `CC BY 4.0`, and public version `1.0.0` was pinned. The model,
features, split, threshold, and evaluation remain frozen.

M6 adds Windows/Linux CI, packaging, a data-free runtime container, and a stateless educational
demo. The abstention envelope was tied to the versioned `training_only` summary; the Wilson
intervals for precision and recall are derived from the already recorded final matrix. None of
these changes rereads the holdout or modifies the M3 result.

The first Linux validation showed that Matplotlib PNGs and Joblib serialization are not
byte-for-byte identical between Windows and Linux. The runtime does not relax hashes or replace the
model: it distributes the exact 1.25 MB evaluated pipeline, checks its SHA-256 before loading it,
and keeps the dataset, partitions, and user-supplied artifacts outside the image.

The authorization/status conclusions written during M5 and retained below are historical evidence
of that gate; they are superseded by D-029 and the final M6 audit.

Local release evidence for the M6 snapshot:

- `pip check`, Ruff, and formatting pass; Ruff verified `46` files.
- pytest passes `227/227` tests.
- The wheel and sdist built with `setuptools 84.0.0` each pass `227/227` tests.
- OSV reports no active advisories in the `45` audited pins; the PyPI pins are not yanked.
- Gitleaks `8.30.1`, downloaded from the official release and verified by SHA-256, found no secrets
  in the snapshot or the entire reachable history.
- The public screenshot is a `1440 × 1100` PNG, with no detected metadata or sensitive strings.
- GitHub Actions passed quality checks on Windows/Linux, wheel/sdist installation, the Docker
  build, Trivy with no fixable `HIGH`/`CRITICAL` vulnerabilities, and hardened runtime smoke tests.
- The repository is public, and the HTTPS demo was verified over IPv4 and IPv6, including health,
  inference, abstention, body/rate limits, and regression of the existing services.

## M5 closure verdict

The local MVP is reproducible and suitable for technical review. The documentation, receipts,
metrics, and application reconcile with frozen run `b15bab7b54bc2e1f`. The holdout was not
reevaluated during M5.

At M5 closure, the repository was not yet authorized or ready for publication: its license still
had to be selected, and the MVP commit had to be curated. Those conditions motivated M6; they do
not describe its current status.

## Clean reconstruction

An isolated temporary copy was created without `.venv`, caches, local data, or binary artifacts.
The sequence was executed using only the README, with CPython `3.12.0` and pip `26.2.1`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade "pip==26.2.1"
.\.venv\Scripts\python.exe -m pip install -c requirements\constraints-win-py312.txt -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m predictive_maintenance download
.\.venv\Scripts\python.exe -m predictive_maintenance validate
.\.venv\Scripts\python.exe -m predictive_maintenance split
.\.venv\Scripts\python.exe -m predictive_maintenance eda
.\.venv\Scripts\python.exe -m predictive_maintenance train
```

Results:

- the `40` versions in the constraints file matched, and `pip check` found no conflicts;
- Ruff and the formatting check passed;
- pytest passed `174/174` tests;
- all eight EDA outputs were byte-for-byte identical;
- run `b15bab7b54bc2e1f` was reproduced;
- the pipeline SHA-256 was reproduced exactly:
  `8f383492fff0a1199a7f62289651a29da39f4c6a149762aa9b75c099efc1568a`;
- the versioned reports, active pointer, and ledger remained byte-for-byte identical;
- Uvicorn started with the holdout inaccessible to the application, and `/health`, `/model-info`,
  and `/predict` returned `200` with the correct identity and threshold.

`evaluate-holdout` was not run: the existing receipt and ledger remained the source of truth. The
holdout file in the clean copy remained under an exclusive lock during training and the application
smoke test.

Indicative times on that machine, not benchmarks: venv creation `8.472 s`, installation `88.836 s`,
pytest `27.769 s`, download `3.340 s`, validation `1.993 s`, split `2.067 s`, EDA `4.025 s`, and
training `13.114 s`.

Primary identities:

| Item | SHA-256 |
|---|---|
| Source CSV | `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e` |
| Training | `3b114192f249951632f4c700c07b5edf4306fcff89ac90abe556f15687cf803a` |
| Holdout | `50a1c9c07a57afbc6f34dd112852b61a44f81b6a83341241dd1bc7079f3ac4b7` |
| Pipeline | `8f383492fff0a1199a7f62289651a29da39f4c6a149762aa9b75c099efc1568a` |

## Subsequent adversarial hardening

On 2026-08-18, unexpected inputs were reviewed without opening data, retraining, or reevaluating the
holdout. The local API initially moved to version `0.2.0`; M6 pins it as `1.0.0`. The new regressions
cover inclusive marginal endpoints, abstention without invoking the pipeline, arbitrary-size
integers, duplicate JSON keys, bodies larger than `16 KiB`, non-finite values, internal exceptions,
invalid probability matrices, and defensive headers.

The smoke test with the real artifact confirmed controlled responses for all these cases:
observations outside the marginal envelope return `200` with null score and decision; an invalid
schema returns `422`, ambiguous JSON returns `400`, and an excessive body returns `413`, always as
JSON and without exposing tracebacks. The UI was verified against the real API in the
applicable/not-applicable states, and its assets use a version in the URL to prevent old JavaScript
from being mixed with a new contract. This review did not alter the M3 run, model, features,
threshold, or metrics.

## Claims and evidence

The README and Model Card were checked against the versioned JSON files, manifests, figures, and
ledger. The model, threshold, CV, OOF, final metrics, and confusion matrix match. The texts state
that AI4I is synthetic, the score was not evaluated as a calibrated probability, and the result
does not establish industrial use, RUL, or causality.

The dataset attribution records the title, UCI, DOI, `CC BY 4.0`, recommended citation, and the
transformations performed in [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). The original CSV remains
outside Git. The exact evaluated pipeline is versioned at
`artifacts/m3/b15bab7b54bc2e1f/pipeline.joblib`, linked to its manifest and verified by SHA-256.

## Interface and accessibility

The actual flow was reviewed in the local browser both on desktop and in a `390 × 844` mobile
viewport:

- valid prediction, score, threshold, decision, and warnings visible;
- invalid validation with focus on the summary and `aria-invalid` on the field;
- no horizontal overflow or console errors;
- structure with a skip link, landmarks, headings, labels, and a live region;
- visible focus, touch targets, and reduced-motion support;
- sampled contrast ratios between `5.14:1` and `17.56:1`.

This review is not a WCAG certification. A complete screen-reader session and a manual audit at
`200%` zoom were not performed; they are advisable checks before public exposure. The alternative
text for the EDA figures was made descriptive in M5.

## Hygiene, secrets, and artifacts

- During M5 there was not yet a remote or publication; M6 configured the public repository and
  completed release `v1.0.0` after the checks recorded at the beginning of this document.
- The initial static search found no keys, tokens, credentials, email addresses, or personal paths.
  Before publication, Gitleaks reviewed the snapshot and the entire reachable history without
  detecting secrets.
- `.gitignore` excludes `.venv`, caches, raw/processed data, local metadata, and generated bundles.
- `*.joblib`, `*.pkl`, and `*.pickle` remain globally ignored; only the nominal path of the
  evaluated pipeline is explicitly allowed.
- Curated reports and figures are versionable; together they total approximately `990 KB`, and the
  largest PNG is approximately `247 KB`.
- No large or unexpected binaries suitable for Git were detected.

## Licenses

The dataset uses `CC BY 4.0`; its attribution and the derived changes are documented separately.
The following expressions were reviewed in the installed metadata of the direct dependencies:

| Direct dependency | Declared license |
|---|---|
| FastAPI | MIT |
| Pydantic | MIT |
| Uvicorn | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| pandas | BSD-3-Clause |
| joblib | BSD-3-Clause |
| httpx2 (dev) | BSD-3-Clause |
| pytest (dev) | MIT |
| Ruff (dev) | MIT |
| Matplotlib | Matplotlib/PSF's own permissive license |
| NumPy | composite expression of permissive licenses according to its metadata |

No evident incompatibility was observed; this is not legal advice. M6 repeats the review against
the effective release environment. The project uses MIT, declared in `LICENSE` and
`pyproject.toml`, without replacing the data's `CC BY 4.0`.

## Publication gate

Final M6 status: **GO completed and published**. The model remains frozen, and the four gates
defined during M5 were satisfied:

1. clean suite and package build on Windows/Linux;
2. audit of the snapshot, secrets, dependencies, and artifacts;
3. green CI on the public commit;
4. HTTPS smoke tests before creating tag `v1.0.0`.

The evidence is summarized at the beginning of this document and remains accessible in release
[`v1.0.0`](https://github.com/xSkyLiN3/predictive-maintenance-ml/releases/tag/v1.0.0) and the
[educational demo](https://ml.nightstrike.cloud). This publication does not turn the system into an
industrial deployment or validate production use.
