# Model Card — Machine Failure Risk Classifier

> **Educational model.** AI4I 2020 is a synthetic dataset. This model is not validated for
> industrial use, safety, real maintenance, or operational decisions. Its `risk_score` was not
> evaluated as a calibrated probability.

## 1. Identity and summary

| Field | Value |
|---|---|
| Product | `Machine Failure Risk Classifier` |
| Model run | `b15bab7b54bc2e1f` |
| Artifact schema version | `1` |
| Task | Binary classification of `Machine failure` for one observation |
| Selected model | `random_forest` (`RandomForestClassifier`) |
| Continuous output | `predict_proba[:, 1]`, exposed as `risk_score` only when the decision applies |
| Frozen threshold | `0.6965799216184142` |
| Decision rule | `risk_score >= threshold` |
| Applicability layer | Marginal envelope derived only from training; abstention outside it |
| Status | Educational software release `1.0.1`; M3 result frozen in `v1.0.0` |
| Availability | Public educational demo at `https://ml.nightstrike.cloud`; re-verified after every deployment |

The service receives six variables from one operational observation. It invokes the model and
returns a score for the positive class `Machine failure` only when the five numerical variables are
within the marginal envelope obtained exclusively from AI4I training. In that case, the Boolean
classification is derived by applying the threshold selected with out-of-fold training
predictions. Outside that reference, the service abstains. The product demonstrates a reproducible
machine-learning engineering workflow; it does not demonstrate usefulness on real machinery.

## 2. Purpose and intended uses

Intended uses:

- demonstrate, in a portfolio project, data traceability, leakage prevention, cross-validation
  selection, final evaluation, and a reproducible inference service;
- score individual observations compatible with the schema and within the marginal reference, and
  explicitly abstain when any field falls outside;
- enable technical review and educational experimentation, locally or through a public, stateless,
  minimal-surface demo;
- compare the real result against a baseline without selecting a number for marketing.

Intended users: people reviewing or studying the project and developers running the demo. It is not
intended for industrial operators.

Explicitly unintended uses:

- making maintenance, safety, shutdown, inventory, or staffing decisions;
- predicting remaining useful life, time to failure, or a future sequence;
- inferring the cause or physical mode of a failure;
- evaluating unrepresented real machinery, plants, manufacturers, or conditions;
- treating the score as a calibrated frequency or probability of a real failure;
- replacing diagnosis, engineering analysis, or human oversight;
- industrial integration or production use without independent validation.

## 3. Data

### 3.1 Source and license

- Dataset: **AI4I 2020 Predictive Maintenance Dataset**.
- Source: UCI Machine Learning Repository, dataset ID 601.
- DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Attribution and transformations: [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).
- Snapshot size: `10,000` observations.
- Nature: synthetic data inspired by predictive-maintenance scenarios.
- Personal data: the project documents that it contains no personal data.
- Source CSV SHA-256: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.

The dataset's synthetic nature limits any conclusion about conditions, noise, drift, temporal
dependencies, and failure mechanisms in real equipment.

### 3.2 Target and features

Official target: `Machine failure`, with values `0` and `1`. The CSV label was preserved without
recalculating it from the failure modes.

| sklearn feature | API field | Type or unit |
|---|---|---|
| `Type` | `type` | Category `L`, `M`, or `H` |
| `Air temperature [K]` | `air_temperature_k` | Kelvin |
| `Process temperature [K]` | `process_temperature_k` | Kelvin |
| `Rotational speed [rpm]` | `rotational_speed_rpm` | Revolutions per minute |
| `Torque [Nm]` | `torque_nm` | Newton-meter |
| `Tool wear [min]` | `tool_wear_min` | Minutes |

Mandatorily excluded columns:

- identifiers `UDI` and `Product ID`, because they do not represent generalizable operational
  signals;
- `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`, because they are failure-mode indicators linked to the
  target and would cause information leakage.

The snapshot contains `27` disagreements between `Machine failure` and the OR of the five
indicators: `9` positives without an active mode and `18` negatives with `RNF = 1`. The official
target was not corrected. This anomaly is a possible source of label noise and reinforces the
exclusion of the modes.

### 3.3 Partition

- Method: `sklearn.model_selection.train_test_split`.
- Seed: `42`.
- Stratification: `Machine failure`.
- Training: `8,000` rows (`80%`).
- Holdout: `2,000` rows (`20%`).
- Training SHA-256:
  `3b114192f249951632f4c700c07b5edf4306fcff89ac90abe556f15687cf803a`.
- Holdout SHA-256:
  `50a1c9c07a57afbc6f34dd112852b61a44f81b6a83341241dd1bc7079f3ac4b7`.

The holdout was materialized before the EDA, did not participate in selection or threshold tuning,
and was read once for the final evaluation of the frozen run.

## 4. Training and selection

### 4.1 Protocol

- Validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Primary metric: unweighted mean Average Precision across the five folds.
- Secondary metric: ROC-AUC.
- CV standard deviation: `ddof=0`.
- Baseline: `DummyClassifier(strategy="prior")`.
- Candidates: balanced L2 logistic regression and balanced random forest.
- No hyperparameter search was performed, and no additional algorithms were tested.
- The same folds were reused for all candidates.
- Preprocessing and estimator remained inside a scikit-learn `Pipeline`.

`Type` was transformed with `OneHotEncoder`, fixed categories `L`, `M`, `H`,
`handle_unknown="error"`, and dense output. The five numerical variables passed through
`StandardScaler`; `ColumnTransformer(remainder="drop")` removed every other column.

The gate against the baseline literally required the winning candidate's mean AP to be greater
than the Dummy's. Candidate ties within `1e-12` favored logistic regression. The random forest won
without needing that tiebreaker.

### 4.2 Frozen random forest

| Parameter | Value |
|---|---:|
| `n_estimators` | `300` |
| `criterion` | `gini` |
| `max_depth` | `8` |
| `min_samples_split` | `10` |
| `min_samples_leaf` | `5` |
| `max_features` | `sqrt` |
| `bootstrap` | `true` |
| `class_weight` | `balanced` |
| `random_state` | `42` |
| `n_jobs` | `1` |

### 4.3 Threshold selection

After selecting the model, one out-of-fold `predict_proba[:, 1]` prediction was generated for each
training row. Threshold `0.6965799216184142` maximizes F1 over those predictions under the inclusive
rule `score >= threshold`. Ties within `1e-12` were resolved by the smallest absolute difference
between precision and recall, then by the lower threshold. The holdout was not used to select or
modify this value.

## 5. Results

The following values are reproduced from the versioned JSON receipts. Average Precision is the
primary metric; accuracy is shown only with prevalence and the majority baseline.

### 5.1 Cross-validation on training

| Candidate | Mean AP | AP std (`ddof=0`) | Mean ROC-AUC |
|---|---:|---:|---:|
| Dummy prior | `0.033875` | `0.0002500000000000002` | `0.5` |
| Logistic regression | `0.44143285508642044` | `0.06850845339545959` | `0.8992749262131948` |
| Random forest | `0.6438124425485383` | `0.02247288575462437` | `0.9699351406866447` |

The random forest's mean AP improvement over the Dummy was `0.6099374425485383`. These figures are
selection estimates, not holdout results.

### 5.2 OOF threshold-selection metrics

| Metric | Value |
|---|---:|
| Pooled Average Precision, diagnostic | `0.6332800463156792` |
| Pooled ROC-AUC, diagnostic | `0.968384753067352` |
| Precision | `0.5878787878787879` |
| Recall | `0.7158671586715867` |
| F1 | `0.6455906821963394` |
| Positive predictions | `330` |
| Matrix `[[TN, FP], [FN, TP]]` | `[[7593, 136], [77, 194]]` |

The pooled metrics were recorded as diagnostics; model selection used mean AP across the five
folds, not pooled AP.

### 5.3 Single final evaluation on holdout

| Metric | Value |
|---|---:|
| Average Precision | `0.6495379423468456` |
| ROC-AUC | `0.9654579222993546` |
| Precision at threshold | `0.5882352941176471` |
| 95% Wilson CI for precision (`50/85`) | `0.4820101461448797`–`0.6868299449467584` |
| Recall at threshold | `0.7352941176470589` |
| 95% Wilson CI for recall (`50/68`) | `0.619922660101109`–`0.825502593301211` |
| F1 at threshold | `0.6535947712418301` |
| Accuracy | `0.9735` |
| Positive prevalence | `0.034` |
| Majority-class accuracy | `0.966` |
| Positive predictions | `85` |
| True negatives | `1897` |
| False positives | `35` |
| False negatives | `18` |
| True positives | `50` |

Confusion matrix, with rows as actual class and columns as predicted class:

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | `1897` | `35` |
| Actual 1 | `18` | `50` |

The result covers `2,000` observations at the same frozen threshold `0.6965799216184142`. The model,
features, and threshold were not modified after observing it.

The intervals are two-sided 95% Wilson intervals and were calculated later only from the already
versioned final matrix: precision uses `TP / (TP + FP) = 50/85`, and recall uses
`TP / (TP + FN) = 50/68`. Holdout observations, targets, and scores were not accessed again. The
`/model-info` endpoint exposes them as `precision_wilson_95` and `recall_wilson_95`, derived from the
same matrix verified when the application loads; it does not persist them as a second evaluation.

These intervals describe finite-support uncertainty conditional on this holdout. They do not
measure per-observation uncertainty, do not correct for synthetic nature or a distribution change,
and were not used to select or tune the system.

## 6. Educational demo inference contract

The application exposes:

- `GET /health`: availability and identity of the loaded model;
- `GET /model-info`: model, threshold, final metrics, and warnings;
- `POST /predict`: applicability status and, only when appropriate, score and classification;
- `GET /`: static interface.

`POST /predict` requires exactly:

| Field | Contract |
|---|---|
| `type` | Exact string `L`, `M`, or `H` |
| `air_temperature_k` | Finite number greater than `0` |
| `process_temperature_k` | Finite number greater than `0` |
| `rotational_speed_rpm` | Non-negative integer |
| `torque_nm` | Finite non-negative number |
| `tool_wear_min` | Non-negative integer |

Extra fields, `null`, booleans, numerical strings, `NaN`, and infinities are rejected. These limits
validate structure, units, and signs; they are not physical limits and do not prove membership in
the AI4I domain. The JSON body admits at most `16 KiB` (`16,384` bytes): exceeding that returns
`413`. JSON member names must be unique; a duplicate key returns `400` instead of applying the
ambiguous "last value wins" semantics. Other schema rejections and inference failures are returned
as controlled, serializable JSON errors, without unnecessary input echoing or details of the
internal exception.

The web interface restricts `rotational_speed_rpm` and `tool_wear_min` to JavaScript-safe integers
through `Number.isSafeInteger` (`0` to `9,007,199,254,740,991`). This maximum prevents precision loss
in the browser; it is neither a physical limit nor the model's applicability envelope.

After validating the schema, the service applies this training marginal reference, with inclusive
endpoints:

| Field | Exact envelope observed in AI4I training |
|---|---:|
| `air_temperature_k` | `295.3`–`304.5` K |
| `process_temperature_k` | `305.7`–`313.8` K |
| `rotational_speed_rpm` | `1168`–`2886` rpm |
| `torque_nm` | `3.8`–`76.6` Nm |
| `tool_wear_min` | `0`–`253` min |

With all fields inside, the response declares `domain_status = "within_reference_envelope"` and
`decision_applicable = true`; `risk_score` is the score for the positive class, and
`predicted_failure` applies the frozen threshold. If any field falls outside, the response retains
HTTP status `200`, but the model is not invoked and no decision is issued:
`domain_status = "outside_reference_envelope"`,
`decision_applicable = false`, `risk_score = null`, and `predicted_failure = null`. `warnings`
individually identify each field that exceeds its interval.

The intervals are the univariate minima and maxima from the `8,000` training rows, obtained from the
versioned EDA summary that declares `scope = "training_only"` and `holdout_profiled = false`. They
are neither physical limits nor a complete OOD detector. They do not evaluate combinations among
variables, density, drift, sequence, or causal plausibility. Therefore,
`within_reference_envelope` means only "within all observed marginal ranges"; an observation marked
that way may still be atypical or unrealistic. If a score exists, it is also not a calibrated
estimate of failure probability.

This abstention layer did not retrain the pipeline, change the run, features, threshold, or M3
metrics, or require another holdout read.

Local mode retains `127.0.0.1:8000` and local hosts as defaults. M6 makes the bind address, port, and
an explicit host allowlist configurable through environment variables for operation behind a
proxy; it does not allow a `*` allowlist. The application loads the pipeline once after validating
the active pointer, manifests, hashes, final receipt, ledger, versions, classes, and feature order.
Inference startup must not read raw, training, or holdout data. Public availability was originally
established for `v1.0.0` after completing the M6 deployment checks. The operational gate for every
release, including `v1.0.1`, requires the same post-deployment checks; every version remains an
educational demo, not industrial validation.

## 7. Limitations and responsible-use considerations

### 7.1 Data and generalization

- AI4I 2020 is synthetic and does not represent a verified population of industrial machinery.
- The random split estimates IID generalization within the generator; it does not measure temporal
  generalization or generalization across machines, plants, manufacturers, or operating regimes.
- There is no external validation with real industrial data.
- The official label has the `27` documented disagreements with the failure modes.
- Holdout prevalence was `0.034`; metrics and errors must be interpreted in that context.

### 7.2 Score, threshold, and OOD

- Calibration, calibration error, and probabilistic reliability were not evaluated.
- Optimizing F1 does not represent real false-positive and false-negative costs.
- The threshold tiebreaker is deterministic, not an operational preference.
- The API abstains outside the training marginal envelope, but it does not detect joint OOD, drift,
  or statistically atypical inputs within those intervals.
- Per-prediction uncertainty is not quantified.
- Each request is an independent observation; the model has no temporal history.

### 7.3 Use risks and ethics

- A false negative could conceal a failure under improper interpretation; a false positive could
  prompt unnecessary interventions. Neither cost was modeled.
- Presenting the score as a probability, diagnosis, or recommendation would be misleading.
- Performance by operational subgroup, fairness, economic impacts, and automation risks were not
  evaluated. The absence of personal data does not demonstrate the absence of impact.
- Any real use would require engineering, safety, governance, and human-oversight review, in
  addition to data representative of the target environment.
- The Joblib binary must be loaded only from the verified local bundle: deserializing artifacts of
  untrusted provenance can execute code.

## 8. Reproducibility and traceability

### 8.1 Cryptographic identities

| Artifact or configuration | SHA-256 |
|---|---|
| M3 configuration | `d880c8048fcb3c09395e38702fd9ca04b1d6e3e0b53fd882e7dd728bdb1b9065` |
| Fold plan | `2c24c5165e54481a6eb35ac08f579c78601ea191eee8a0d0a76537b034eddf48` |
| Local pipeline | `8f383492fff0a1199a7f62289651a29da39f4c6a149762aa9b75c099efc1568a` |
| Run manifest | `01c3c72a75df64922470ee163166fbb2437fad0b6c279ca54f0b5687ccb02a2a` |
| CV results | `ce9f62b79834c0da8d6a44311ae3714c18922820fc33f61a8ca9b3ab9f9e12f2` |
| Threshold selection | `9790eedc834f5624029a24c2e64a544392c171995bde51d33f89cf7cbe70d412` |
| Final evaluation receipt | `f3c947fe38fca0053c3f14e75c01681e5cef1dbcbc09e57fddb15409fd1e26c8` |

The global ledger records `holdout_evaluation_complete` for the indicated holdout and run. A later
execution must reuse the receipt, not reopen that holdout.

### 8.2 Environment recorded by the run

| Component | Version |
|---|---|
| Python | `3.12.0` |
| scikit-learn | `1.9.0` |
| pandas | `3.0.5` |
| NumPy | `2.5.2` |
| joblib | `1.5.3` |
| matplotlib | `3.11.1` |

The exact evaluated pipeline is versioned alongside its SHA-256 manifest and copied unchanged into
the Linux image. Re-training or re-serializing the model on another platform is not expected to
reproduce byte-for-byte identical Joblib output, even though training is deterministic; replacing
the pinned artifact would therefore break the final receipt.
`python -m predictive_maintenance train` makes it possible to reproduce the workflow from training
without reading the holdout, while the demo loads only the repository's own artifact after
validating the entire receipt chain. The exact tested resolution is retained in
`requirements/constraints-win-py312.txt`.

Primary versioned artifacts:

- [pipeline manifest](../artifacts/m3/b15bab7b54bc2e1f/artifact_manifest.json);
- [run manifest](../reports/modeling/b15bab7b54bc2e1f/run_manifest.json);
- [cross-validation results](../reports/modeling/b15bab7b54bc2e1f/cv_results.json);
- [OOF threshold selection](../reports/modeling/b15bab7b54bc2e1f/threshold_selection.json);
- [final receipt](../reports/modeling/b15bab7b54bc2e1f/final_evaluation.json);
- [M3 report](../reports/modeling/b15bab7b54bc2e1f/M3_REPORT.md);
- [holdout-access ledger](../reports/holdout_access/50a1c9c07a57afbc6f34dd112852b61a44f81b6a83341241dd1bc7079f3ac4b7.json);
- [data and evaluation contract](DATA_EVALUATION.md);
- [decision record](DECISIONS.md).

## 9. Maintenance and change management

There is no scheduled retraining, monitoring, telemetry, or drift detector. The model and threshold
are frozen for this local MVP.

Maintenance rules:

- if the versioned binary is missing, restore it from a verified Git revision; `train` can
  reproduce the workflow, but reserialization on another operating system does not silently
  replace the evaluated artifact or authorize repeating the consumed holdout;
- always validate the run, hashes, final receipt, ledger, versions, classes, and feature order
  before serving inference;
- treat changes to the target, features, dataset, split, protocol, models, threshold, or API
  contract as a new documented version, not a silent correction;
- keep the marginal applicability envelope separate from physical limits: changing it is a
  service-contract decision and does not by itself modify the model or its metrics;
- do not use metrics from the already observed holdout to select a new variant;
- update this card if the model identity or any of its assumptions changes;
- review compatibility and rerun tests after updates to Python, scikit-learn, joblib, NumPy,
  pandas, FastAPI, or Pydantic;
- treat the public M6 demo as educational and stateless, without storing inputs or including data
  or partitions; do not confuse its availability with industrial validation or production
  maturity.

Before considering real use, the minimum requirements would include representative industrial
data, temporal and per-machine splits, external validation, calibration evaluation, error costs,
expert-defined operating limits, OOD and drift detection, monitoring, service security, and an
explicit governance and withdrawal process.

## 10. Provenance of this card

All figures, parameters, and identities in this card come from the versioned artifacts linked above
and the project's accepted documentation. The envelope comes from the versioned training EDA
summary, and the Wilson intervals are derived from the versioned final matrix. Raw, training, and
holdout CSV files were not consulted, predictions were not recomputed, and no new evaluation was
produced to write it.
