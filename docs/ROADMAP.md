# Roadmap

Each milestone has a review gate. Work advances only after the preceding gate criteria are met.

## M0 — Environment and structure

**Outcome:** reproducible Python 3.12 project, without modeling yet.

- Create `.venv` explicitly with Python 3.12.
- Define `pyproject.toml` and minimal dependencies.
- Create the `src/`, `tests/`, `data/`, `notebooks/`, `reports/`, and `artifacts/` structure.
- Configure Ruff and pytest.
- Document PowerShell commands.

**Gate:** a clean installation imports the package and runs a minimal test.

## M1 — Data ingestion and contract

**Outcome:** official dataset downloaded and validated through code.

- Download from UCI.
- Store metadata and checksum.
- Verify schema, types, categories, nulls, duplicates, and target.
- Implement an explicit list of allowed features and prohibited columns.
- Create schema-regression tests.

**Gate:** Ruff and pytest pass; no leakage column can accidentally enter training.

## M2 — EDA and closed protocol

**Outcome:** brief report supporting the modeling decisions.

- First materialize the reproducible stratified partition and seal the holdout.
- Measure prevalence and review ranges.
- Create 5–7 relevant visualizations.
- Identify anomalies and limitations.
- Confirm the metrics and threshold strategy in writing.

**Gate:** human review of the report before training candidate models.

## M3 — Baseline, models, and evaluation

**Status:** completed and verified in run `b15bab7b54bc2e1f`.

**Outcome:** model selected and evaluated cleanly.

- Train Dummy, logistic regression, and random forest.
- Compare them through cross-validation on training only.
- Choose the threshold without consulting the holdout.
- Evaluate once on the holdout.
- Save the pipeline, configuration, metrics, and plots.

**Gate:** repeated execution with the same configuration reproduces the results within the stated
tolerances.

## M4 — API and local interface

**Status:** completed and verified against run `b15bab7b54bc2e1f`.

**Outcome:** functional localhost demo.

- Implement `/health`, `/model-info`, and `/predict`.
- Validate inputs with strict schemas.
- Create a simple locally served HTML/CSS interface.
- Show the score, decision, and educational warning without claiming calibration.
- Add API and inference tests.
- Explicitly abstain outside the educational marginal envelope without presenting it as a physical
  limit or complete OOD detector.
- Reject ambiguous or excessive JSON in a controlled way and prevent precision loss in the UI.

**Gate:** a user can start the application by following the README, complete the main flow, and
receive controlled responses to adversarial inputs.

## M5 — Closure and publication review

**Status:** completed and verified locally.

**Outcome:** technical candidate approved for publication preparation.

- Complete the README, model card, and architecture.
- Verify installation from scratch.
- Run the full suite and review secrets/licenses.
- Review interface accessibility and clarity.
- Compare README claims with generated evidence.

**Gate:** the local MVP passed review. The license, publication, and demo were authorized for M6
on 2026-08-19 and subsequently passed the snapshot and history audit.

## M6 — Publication and educational demo

**Status:** completed and verified.

**Outcome:** public release `v1.0.0`, verifiable CI, and a stateless demo behind HTTPS.

- Adopt MIT without mixing it with the dataset's `CC BY 4.0` license.
- Keep the model, features, threshold, and holdout frozen; only correct envelope traceability and
  contextualize precision/recall from the already versioned matrix.
- Verify the package on Windows and Linux, and build a non-root runtime without data partitions.
- Curate the README, changelog, screenshot, metadata, and honest commits.
- Audit secrets, dependencies, staged files, and sizes before the push.
- Publish the repository and release on GitHub.
- Deploy an educational demo with TLS, resource limits, and no input persistence.
- Add the link only after successful external smoke tests.

**Gate passed:** green CI, verifiable release, healthy demo, and public claims reconciled with the
versioned evidence. A subsequent ML project will be a separate milestone, not an expansion of this
MVP.
