# Decision record

## D-001 — Local development before publication

- **Status:** accepted.
- **Decision:** complete and review the MVP locally before creating a remote or deploying.
- **Rationale:** separate learning and experimentation from any public claim.

## D-002 — Python 3.12 and CPU

- **Status:** accepted.
- **Decision:** use Python 3.12 in `.venv` and train on CPU.
- **Rationale:** the equipment has Python 3.12 available, and the tabular dataset is small; a GPU
  would add complexity without relevant value.

## D-003 — Per-observation classification scope

- **Status:** accepted.
- **Decision:** estimate `Machine failure` for one observation, not remaining useful life or time
  series.
- **Rationale:** this is what the selected data can honestly support.

## D-004 — Operational features without failure indicators

- **Status:** accepted.
- **Decision:** exclude identifiers and `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
- **Rationale:** prevent non-generalizable signals and leakage with respect to the target.

## D-005 — Average Precision as the primary metric

- **Status:** accepted.
- **Decision:** use Average Precision, complemented by precision, recall, F1, a confusion matrix,
  and ROC-AUC.
- **Rationale:** the positive class is a minority, and accuracy alone would be uninformative.

## D-006 — Minimal packaging with dependencies by phase

- **Status:** accepted.
- **Decision:** use `setuptools` with a `src/` structure; keep `pandas` as the only runtime
  dependency during M1 and isolate pytest and Ruff in the `dev` extra.
- **Rationale:** M1 needs tabular reading and validation, while scikit-learn, FastAPI, and their
  dependencies are not needed until later milestones. This keeps the environment small and avoids
  implementing phases in advance.

## D-007 — Immutable, verified data snapshot

- **Status:** accepted.
- **Decision:** download the official ZIP with the standard library, require known size and
  SHA-256, extract only `ai4i2020.csv`, and verify the CSV again. Keep both in `data/raw/` without
  silently replacing existing files. Record the date, source, DOI, license, sizes, and hashes in
  local metadata.
- **Rationale:** this ensures traceability and detects source changes without adding an HTTP
  dependency for a single download. The data and local metadata are not versioned; the expected
  identities are pinned in code and documentation.

## D-008 — Preserve the official target without deriving it from failure modes

- **Status:** accepted.
- **Decision:** use the original `Machine failure` column without correcting it or recalculating it
  from `TWF`, `HDF`, `PWF`, `OSF`, or `RNF`. The validator reports, but does not reject, the 27 rows
  where the target differs from the OR of those indicators.
- **Rationale:** the official CSV contains 9 failures without an active indicator and 18 cases with
  `RNF = 1` and target 0. Rewriting the target would alter the source. The five indicators remain
  mandatorily excluded from the features because of leakage.

## D-009 — Source guardrails separate from industrial limits

- **Status:** accepted.
- **Decision:** validate the numerical variables against rounded envelopes that contain the AI4I
  snapshot, and separately record the observed minima and maxima. Do not reuse these ranges as a
  future API contract or present them as universal physical limits.
- **Rationale:** M1 needs to detect corrupt or incompatible files, but AI4I is synthetic, and its
  observed endpoints do not justify rules about real machinery.

## D-010 — Dependency resolution recorded for Windows and Python 3.12

- **Status:** accepted.
- **Decision:** keep direct dependency ranges in `pyproject.toml` and record the tested transitive
  resolution in `requirements/constraints-win-py312.txt`.
- **Rationale:** this prevents accidental drift when rebuilding the environment without adding
  another dependency manager. The file records versions but does not promise binary
  reproducibility across platforms or replace functional verification with Ruff and pytest.

## D-011 — Seal the holdout before EDA

- **Status:** accepted.
- **Decision:** at the beginning of M2, materialize a stratified 80/20 split with seed `42` before
  calculating decision-oriented statistics or plots. The seed was an arbitrary choice made before
  results and will not be compared with others. The EDA loads training only.
- **Rationale:** `ROADMAP.md` and the evaluation contract already required separating the holdout
  before selection; `TASKS.md` mistakenly placed it in M3. Moving the task to M2 aligns the
  checklist without changing the target, features, or ratio. M1 did verify the global aggregates
  needed for the source contract; holdout-specific blindness begins with this materialization.

## D-012 — Derivatives without leakage columns

- **Status:** accepted.
- **Decision:** store `train.csv` and `holdout.csv` with exactly the six allowed features and
  `Machine failure`; do not retain `UDI`, `Product ID`, or mode indicators in the derivatives.
  Version a deterministic manifest, not the CSV files.
- **Rationale:** a physical allowlist reduces the risk that M3 incorporates prohibited columns
  through an incomplete `drop`. The manifest hashes and configuration allow the partition to be
  reconstructed and audited without publishing derived data.

## D-013 — Closed selection and threshold protocol in M2

- **Status:** accepted.
- **Decision:** use five stratified folds with shuffle and seed `42`; select by mean AP, with
  secondary ROC-AUC; use a prior dummy baseline. Pooled OOF AP will be diagnostic only. A tie within
  `1e-12` between logistic regression and random forest favors logistic regression. The threshold
  will maximize F1 over the selected model's OOF `predict_proba[:, 1]` using
  `score >= threshold`, with the same tie margin and the documented deterministic tiebreaker. If no
  candidate beats the dummy on mean AP, the holdout will not be evaluated.
- **Rationale:** this closes decisions before model or holdout results are observed, prevents
  optimizing a later narrative, and does not invent nonexistent industrial costs.

## D-014 — Minimal dependencies for M2

- **Status:** accepted.
- **Decision:** add scikit-learn for the split and future modeling, and matplotlib for six
  reproducible figures. Do not add seaborn, statsmodels, or notebook tooling.
- **Rationale:** both dependencies have a direct role in the milestone and will be sufficient for
  the EDA. Wilson intervals are calculated with a small formula, avoiding another dependency.

## D-015 — Informational versions and invariant split hashes

- **Status:** accepted.
- **Decision:** retain in the manifest the exact versions of the environment that created it, but
  do not require a reconstruction to use the same Python 3.12 patch release. Source, configuration,
  columns, sizes, and CSV hashes are strict invariants.
- **Rationale:** `pyproject.toml` allows any Python 3.12.x. If another revision reconstructs
  identical bytes, rejecting it only because the informational version differs would prevent
  reproducibility without improving integrity; an actual splitter difference remains detectable by
  the hashes.

## D-016 — Pipelines and fixed complexity before M3

- **Status:** accepted.
- **Decision:** compare exactly a prior dummy, balanced L2 logistic regression (`l1_ratio=0`, `C=1`,
  `liblinear`), and a balanced random forest with 300 trees, depth 8, and minimum leaf size 5.
  `Type` is encoded with fixed L/M/H categories, and the numerical variables are standardized
  within each `Pipeline`. No tuning will be performed. All stochastic parameters use seed `42`,
  and the forest uses a single process.
- **Rationale:** these are small, interpretable, and sufficiently different candidates for the
  MVP. Fixing them before calculating CV prevents optimizing the narrative after seeing results and
  keeps the cost reproducible on CPU.

## D-017 — Final operationalization of CV and baseline gate

- **Status:** accepted.
- **Decision:** materialize a single tuple of five folds and reuse it for all three candidates and
  the OOF predictions. Report population standard deviation (`ddof=0`) and paired per-fold deltas.
  The gate is strict: `winner_mean_ap > dummy_mean_ap`; the tie tolerance does not apply to it.
  Reruns will reuse final results and will not reopen the holdout.
- **Rationale:** this removes operational ambiguity without changing the metric or criterion frozen
  in M2 and reconciles the single evaluation with an idempotent CLI.

## D-018 — Direct dependencies of M3 artifacts

- **Status:** accepted.
- **Decision:** declare NumPy and joblib as direct dependencies, even though scikit-learn also
  installs them transitively. NumPy implements explicit validation and aggregation; joblib
  serializes the selected pipeline. Also record matplotlib in the run's version identity.
- **Rationale:** the project code imports and uses these libraries directly. Declaring them avoids
  accidental reliance on the transitive graph and makes the identity of artifacts and figures
  auditable.

## D-019 — Global ledger and recoverable publication of the evaluation

- **Status:** accepted.
- **Decision:** isolate each run in `reports/modeling/<run_id>/` and `artifacts/m3/<run_id>/`, link
  the pipeline, configuration, folds, receipts, and figures by SHA-256, and claim the evaluation
  with an exclusive ledger indexed by the holdout SHA-256 in `reports/holdout_access/`. Publish a
  recoverable local bundle first and the versionable receipt last. A different run, even if output
  roots change, cannot consume the same holdout again.
- **Rationale:** a per-run flag did not protect the same test against configuration changes. The
  global ledger preserves single-evaluation semantics; the bundle can repair an interrupted
  publication without a second read. The guarantee covers the application's sequential workflow
  from the root, not a deliberate manual read of the CSV.

## D-020 — Frozen M3 result without post-holdout iteration

- **Status:** accepted.
- **Decision:** preserve run `b15bab7b54bc2e1f` as the M3 result. Random forest won with mean CV AP
  `0.643812`; the OOF threshold was `0.6965799216184142`. In the single holdout evaluation: AP
  `0.649538`, ROC-AUC `0.965458`, precision `0.588235`, recall `0.735294`, F1 `0.653595`, and matrix
  `[[1897, 35], [18, 50]]`. Do not tune models, features, or the threshold after observing these
  metrics.
- **Rationale:** publishing the real result preserves the preregistered protocol and avoids turning
  the holdout into a hidden validation set. AI4I is synthetic, and these values do not validate
  industrial performance or probabilistic calibration.

## D-021 — Read-only inference on the final run

- **Status:** accepted.
- **Decision:** for M4, load only the evaluated pipeline from run `b15bab7b54bc2e1f` through an
  inference loader that validates the active pointer, manifests, hashes, final receipt, ledger,
  versions, classes, and feature order before deserializing. Load it once in the FastAPI lifespan,
  and do not invoke the evaluation workflow or resolve data files.
- **Rationale:** the application must serve the frozen result without reopening the holdout,
  recalculating metrics, repairing artifacts, or mixing in a different selection. Failing closed on
  an inconsistency is preferable to serving a model whose identity cannot be demonstrated.

## D-022 — Semantic API schema without implying industrial support

- **Status:** accepted.
- **Decision:** separate public names from sklearn columns and require strict JSON with category
  L/M/H, finite numbers, temperatures greater than zero kelvin, and non-negative speed, torque, and
  wear; speed and wear are integers. Do not impose as API limits the
  295–305/305–315/1,000–3,000/0–80/0–260 envelopes used by M1 to validate the source.
- **Rationale:** this preserves D-009 and avoids presenting endpoints from a synthetic generator as
  physical limits. This contract validates shape, units, and signs; it does not detect
  out-of-distribution inputs. The API and interface warn that extrapolation may produce unreliable
  scores.
- **Evolution:** D-026 retains this schema and adds marginal-support abstention without treating
  observed endpoints as physical validation or a complete OOD detector.

## D-023 — Minimal local application and closed surface

- **Status:** accepted.
- **Decision:** add FastAPI, Pydantic, and Uvicorn as direct dependencies; use `httpx2` only in the
  `dev` extra for `TestClient`. Serve static HTML/CSS/JavaScript without Jinja2 or multipart, bind
  Uvicorn to `127.0.0.1`, accept only local hosts, and apply CSP and defensive headers. Do not enable
  CDN-dependent web documentation; retain only the JSON OpenAPI schema.
- **Rationale:** this covers the demo and its tests with the smallest functional graph, avoids
  external resources, and keeps M4 within local scope without authentication, a database,
  telemetry, or deployment.

## D-024 — Reproducible closure of the local MVP

- **Status:** accepted.
- **Decision:** close M5 after reproducing from a clean copy the dependencies, the eight EDA
  outputs, run `b15bab7b54bc2e1f`, and the pipeline's exact SHA-256. The test ran `174/174` tests and
  started the application with the holdout inaccessible, without invoking another evaluation.
- **Rationale:** a documented reproduction provides stronger evidence than repeating commands in
  the development environment and preserves the single-evaluation contract.

## D-025 — Separate data attribution, code license, and authorizations

- **Status:** accepted.
- **Decision:** document AI4I and its transformations under `CC BY 4.0` without automatically
  applying that license to the code. The code license remains pending a user choice. Publication on
  GitHub will additionally require a curated commit, final secrets review, and explicit
  authorization. Deployment will be a later, separate decision.
- **Rationale:** attribution, code licensing, publication, and operation are distinct permissions.
  M5 can close the local product without presuming any of them.
- **Evolution:** D-029 records the subsequent MIT choice and the separate M6 authorization.

## D-026 — Marginal abstention and adversarial hardening of the API contract

- **Status:** accepted.
- **Decision:** retain the semantic schema from D-022 and add, before inference, an inclusive check
  against the exact endpoints observed in AI4I: air `295.3`–`304.5` K, process `305.7`–`313.8` K,
  speed `1168`–`2886` rpm, torque `3.8`–`76.6` Nm, and wear `0`–`253` min. An interior observation
  retains `domain_status = "within_reference_envelope"`, `decision_applicable = true`, and normal
  inference. If one or more fields fall outside, respond with `200` without invoking the model or
  making a decision: `domain_status = "outside_reference_envelope"`,
  `decision_applicable = false`, `risk_score = null`, and `predicted_failure = null`, with one
  warning per offending field. This envelope is an educational marginal-support reference, not a
  physical limit or joint OOD detection. It does not contradict D-009: an out-of-envelope
  observation remains valid for the schema and receives explicit abstention, not a rejection
  presented as an industrial rule.

  Also harden transport: reject duplicate JSON keys with `400`, limit the body to `16 KiB`
  (`16,384` bytes) with `413`, return validation and inference errors as controlled JSON, and
  require JavaScript-safe integers in the UI for speed and wear (`Number.isSafeInteger`, from `0`
  to `9,007,199,254,740,991`). This last maximum protects browser serialization and does not express
  industrial support.
- **Rationale:** do not assign a score or decision to obvious marginal extrapolations, and address
  adversarial cases that could produce ambiguity, precision loss, or uncontrolled errors. This is
  a service-contract change: it does not change features, target, pipeline, run, threshold, or
  metrics, does not reopen the holdout, and does not authorize publication or deployment.

## D-027 — Envelope provenance exclusively from training

- **Status:** accepted; clarifies the provenance stated in D-026 without changing its values.
- **Decision:** define the abstention envelope from `reports/eda/summary.json`, which declares
  `scope = "training_only"`, `training_rows = 8000`, and `holdout_profiled = false`. Retain the five
  already published pairs because they exactly match those training endpoints. Pin them in code so
  inference continues without reading data or reports, and add a regression that compares them
  with the versioned training summary.
- **Rationale:** D-026 described the values as AI4I endpoints but did not clearly distinguish their
  provenance from the complete snapshot. Although the training endpoints in this split match the
  global ones, an applicability rule must not obtain information from the holdout. The
  correction does not reopen any CSV, change responses, pipeline, run, threshold, or metrics, or
  turn the marginal rule into an OOD detector.

## D-028 — Wilson intervals derived from the frozen final matrix

- **Status:** accepted.
- **Decision:** accompany precision and recall with two-sided 95% Wilson intervals, calculated
  exclusively from `[[1897, 35], [18, 50]]`, the matrix in the versioned final receipt. Precision
  uses `50/85` and yields `0.4820101461448797`–`0.6868299449467584`; recall uses `50/68` and yields
  `0.619922660101109`–`0.825502593301211`. `/model-info` derives them in memory from the matrix whose
  integrity the loader already validates; the final receipt and its ledger remain immutable.
- **Rationale:** the holdout contains only `68` positives and `85` positive predictions. Showing
  finite-support uncertainty avoids an overly precise reading of the estimates without reopening
  observations or scores. The intervals do not correct for bias, shift, dependence, or synthetic
  nature; they are not per-prediction uncertainty and are not used to select or modify the model,
  features, threshold, or claims.

## D-029 — Separate license and authorization for M6

- **Status:** accepted on `2026-08-19`; evolves D-025.
- **Decision:** license the code under MIT, keeping AI4I's attribution and `CC BY 4.0` license
  separate. Prepare and execute publication on GitHub and a public, educational, stateless demo
  with a minimal surface. Do not include datasets, partitions, or stored inputs in the image, or
  present the deployment as an industrial or production system.
- **Rationale:** the project owner approved both actions after local closure. Licensing,
  publication, and deployment remain conceptually separate decisions; the authorization does not
  unfreeze the model, features, split, threshold, or metrics, and it does not permit reopening the
  holdout. The demo will be declared available only after verifying the deployed endpoint.

## D-030 — Distribute the exact evaluated artifact

- **Status:** accepted on `2026-08-19` after Linux CI validation.
- **Decision:** version the 1.25 MB Joblib pipeline and its manifest, and copy those exact bytes into
  the runtime. Verify SHA-256, run identity, versions, selection receipts, final evaluation, and
  ledger before `joblib.load`. Do not accept uploaded models or include datasets or partitions in
  the image. Keep `train` as a separate reproducible workflow, not as a mechanism that silently
  replaces the deployed artifact.
- **Rationale:** CI confirmed that regenerating Matplotlib PNGs and Joblib serialization on Windows
  and Linux can produce different bytes. Regenerating the pipeline in the build correctly failed
  validation against the evaluated model's receipt; relaxing the hash would have served a different
  artifact. Distributing the project's own small, pinned binary preserves evaluation identity and
  reduces the build surface, which no longer downloads data or materializes the holdout.
