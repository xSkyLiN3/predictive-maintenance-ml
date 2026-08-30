# Data and evaluation contract

This document establishes the rules before results are observed. Changing them afterward requires
recording the change in `docs/DECISIONS.md` and explaining the reason.

## Source and traceability

- Dataset: AI4I 2020 Predictive Maintenance.
- Canonical source: UCI Machine Learning Repository, ID 601.
- License: CC BY 4.0.
- The download must store the URL, date, size, and SHA-256.
- The original file is immutable; every transformation generates a separate output.

## Target

`Machine failure`, binary classification.

## Initially allowed features

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

These names were confirmed against the official CSV pinned in M1 and form the exact allowlist.

## Prohibited columns

- Identifiers: `UDI`, `Product ID`.
- Failure-mode indicators: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.

The failure-mode indicators are directly tied to the target definition. Including them would
artificially inflate performance and prevent the project from demonstrating useful inference from
operational signals.

## Snapshot contract confirmed in M1

The strict contract applies to the complete source dataset, not to a future API observation:

- 10,000 rows and 14 columns in the official order;
- strings: `Product ID` and `Type`;
- floats: air and process temperatures, and torque;
- integers: `UDI`, rotational speed, wear, target, and five failure modes;
- zero nulls, empty strings, infinities, and fully duplicated raw rows;
- `Type` contains exactly `L`, `M`, and `H`;
- the target and failure modes are binary, and the target contains both classes;
- `UDI` is positive and unique; `Product ID` is unique, follows `[LMH]` plus five digits, and its
  prefix matches `Type`.

The numerical regression guardrails are: air 295–305 K, process 305–315 K, speed 1,000–3,000 rpm,
torque 0–80 Nm, and wear 0–260 min. They contain the observed snapshot, but they are neither
universal physical limits nor the M4 API input contract.

The exact endpoints of the complete snapshot were inspected in M1 as source quality control, but
they are **not the provenance of the abstention rule**. The service envelope was frozen exclusively
from `reports/eda/summary.json`, a versioned artifact that declares
`scope = "training_only"`, `training_rows = 8000`, and `holdout_profiled = false`:

| Variable | Observed minimum and maximum, inclusive |
|---|---:|
| `Air temperature [K]` | `295.3`–`304.5` K |
| `Process temperature [K]` | `305.7`–`313.8` K |
| `Rotational speed [rpm]` | `1168`–`2886` rpm |
| `Torque [Nm]` | `3.8`–`76.6` Nm |
| `Tool wear [min]` | `0`–`253` min |

These are the univariate endpoints of **training**, not of the holdout. They describe marginal
support in synthetic data; they are not industrial limits, physical rules, or evidence that every
interior combination belongs to the generator's domain. The fact that these five pairs match the
global endpoints observed during M1 does not change their provenance: the contract and automated
regression use only the training EDA summary.

## M4 local inference contract

The API defines an independent contract for one observation; it does not reuse the CSV validator or
the source guardrails. It requires exactly these JSON fields:

- `type`: exact string `L`, `M`, or `H`;
- `air_temperature_k` and `process_temperature_k`: finite numbers greater than zero;
- `rotational_speed_rpm`: non-negative integer;
- `torque_nm`: finite non-negative number;
- `tool_wear_min`: non-negative integer.

Extra fields, `null`, booleans, numerical strings, `NaN`, and infinities are not accepted. The five
numerical constraints express semantic units and signs, not statistical support or industrial
limits. JSON names must be unique: a duplicate key returns `400`. The body has a maximum size of
`16 KiB` (`16,384` bytes), and exceeding it returns `413`. Other validation rejections and inference
failures are represented by controlled, serializable JSON responses.

The UI additionally requires `Number.isSafeInteger` for speed and wear, between `0` and
`9,007,199,254,740,991`, solely to prevent precision loss when converting the form to JavaScript
numbers. This client-side protection is not a physical limit and does not replace server
validation.

An observation that passes the schema is compared with the training marginal envelope recorded
above. The values are pinned in code so that inference does not read datasets or reports at runtime;
`tests/test_reference_envelope.py` demonstrates that they exactly match the `training_only` EDA
summary and that the summary declares the holdout was not profiled. If all variables remain within
their inclusive endpoints,
`domain_status = "within_reference_envelope"`,
`decision_applicable = true`, and the pipeline produces `risk_score` and `predicted_failure`
normally. If any variable falls outside, the response remains HTTP `200`, but the pipeline is not
invoked: `domain_status = "outside_reference_envelope"`, `decision_applicable = false`,
`risk_score = null`, and `predicted_failure = null`. Warnings separately identify each out-of-range
field.

This abstention does not turn the API into a complete OOD detector. It checks only five univariate
projections; it does not validate joint support, correlations, density, drift, temporal order, or
physical plausibility. An input marked `within_reference_envelope` may still be out of distribution.
Adopting this service layer does not modify the model, threshold, or any M3 metric, and it does not
reuse the holdout to select or tune the system.

In addition to raw duplicates, the validator reports observations repeated across the six features
without rejecting them: identical operational measurements may be legitimate. In the pinned
snapshot, both counts are zero.

The file contains 27 disagreements between `Machine failure` and the OR of the five indicators: 9
positives without an active mode and 18 negatives with `RNF = 1`. The original target is preserved,
no row is imputed or corrected, and the modes are not required to be mutually exclusive. This
anomaly reinforces that the indicators should be audited but never enter the feature set.

## Partition

1. Designate a stratified 20% holdout once.
2. Keep the remaining 80% for training and cross-validation.
3. Fit preprocessing within each fold through `Pipeline`.
4. Choose the model and threshold without consulting the holdout.
5. Run one final evaluation on the holdout and preserve it as the MVP result.

The fixed seed, chosen arbitrarily before modeling and without testing alternatives, is `42`; it is
reused in all compatible components. The partition was materialized at the beginning of M2, before
the EDA, with scikit-learn 1.9.0 `train_test_split`. Rows are sorted by their original position
within each partition to produce canonical CSV files.

The local derivatives contain exclusively the six allowed features and the target. The versioned
`data/split_manifest.json` manifest records the source hash, algorithm, seed, ratio, columns,
versions, sizes, and hashes of both partitions, but no holdout-specific statistics. The EDA loads
and verifies only `train.csv`; it does not resolve `holdout.csv`. The manifest versions record the
environment that created the file; they are not integrity invariants across Python 3.12 revisions.
The derived hashes must match exactly.

M1 necessarily inspected global counts and ranges to validate the source contract. Since the M2
partition was materialized, no holdout-specific distribution, example, or result has participated
in decisions.

## Cross-validation and selection

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` shared by the candidates.
- All preprocessing is fitted within each fold through `Pipeline`.
- Selection uses the unweighted mean Average Precision across the five folds; per-fold values and
  standard deviation are also published.
- Average Precision pooled over the OOF predictions may be shown only as a diagnostic; it does not
  replace the mean of the five folds for selection.
- ROC-AUC is secondary and does not break a selection tie.
- `DummyClassifier(strategy="prior")` is the baseline.
- Between logistic regression and random forest, the higher mean AP wins. A numerical tie within
  `1e-12` favors logistic regression for simplicity.
- If no candidate beats the dummy on mean AP, the holdout is not opened, and M3 stops for review.

## Initial models

1. `DummyClassifier` as the minimum reference.
2. Logistic regression with preprocessing and class balancing where applicable.
3. Random forest with controlled complexity.

No additional algorithms will be added until the error profiles of these models are understood.

The configuration was frozen before running M3, and no hyperparameter search will be performed:

- all three estimators live inside a `Pipeline`;
- `Type` is encoded with `OneHotEncoder` using fixed categories `L`, `M`, `H`; the five numerical
  variables pass through `StandardScaler`; the `ColumnTransformer` removes every other column;
- dummy: `DummyClassifier(strategy="prior")`;
- logistic regression: L2 through `l1_ratio=0`, `C=1`, solver `liblinear`,
  `class_weight="balanced"`, `max_iter=1000`, and seed `42`;
- random forest: 300 trees, Gini, maximum depth 8, `min_samples_split=10`,
  `min_samples_leaf=5`, `max_features="sqrt"`, bootstrap, `class_weight="balanced"`, seed `42`, and
  `n_jobs=1`.

The CV standard deviation will be calculated with `ddof=0`. The gate against the dummy preserves
the protocol's literal interpretation: the best candidate's mean AP must be strictly greater than
the dummy's mean AP; the `1e-12` tolerance is used only for ties between logistic regression and
random forest and for threshold ties.

## Metrics

### Primary

**Average Precision**, suitable for summarizing precision-recall when the positive class is a
minority.

### At the selected threshold

- precision;
- recall;
- F1;
- confusion matrix;
- number of false positives and false negatives.

### Secondary

- ROC-AUC.

Accuracy will be reported only together with prevalence, the baseline, and the metrics above.

## Threshold

The threshold will not necessarily be 0.5. After selecting the model, exactly one out-of-fold
`predict_proba[:, 1]` probability will be generated per training row using the same five folds. The
threshold that maximizes F1 under the rule `score >= threshold` will be selected. Ties within
`1e-12` will first be resolved by the smallest absolute difference between precision and recall,
then by the lower threshold. This last tiebreaker is deterministic only and does not represent
industrial costs. The value will be frozen before fitting the selected pipeline on all training and
evaluating the holdout once.

OOF precision, recall, and F1 will be labeled as estimates used for selection, not final results.
The report will show the complete tradeoff without inventing operating costs.

## Recorded M3 execution

The contract above was executed without tuning or feature changes in run `b15bab7b54bc2e1f`.
Random forest won on mean CV AP (`0.643812`), ahead of logistic regression (`0.441433`) and Dummy
(`0.033875`). The frozen OOF threshold was `0.6965799216184142`; on OOF training predictions it
produced precision `0.587879`, recall `0.715867`, and F1 `0.645591`.

After freezing the model and threshold, the holdout was read once. The final result was AP
`0.649538`, ROC-AUC `0.965458`, precision `0.588235`, recall `0.735294`, F1 `0.653595`, and matrix
`[[1897, 35], [18, 50]]`. Accuracy was `0.973500`, alongside prevalence `0.034` and majority
reference `0.966`.

As a subsequent closure step, without reopening the holdout or recomputing predictions, two-sided
95% Wilson intervals derived **only** from that versioned matrix were added. For precision, the
successes are `TP = 50` out of `TP + FP = 85`: CI `0.482010`–`0.686830`. For recall, the successes
are `TP = 50` out of `TP + FN = 68`: CI `0.619923`–`0.825503`. The normal quantile
`z = 1.9599639845400536` is used; the formula and its regressions live in
`src/predictive_maintenance/uncertainty.py` and `tests/test_uncertainty.py`.

These intervals quantify only binomial uncertainty from finite support under the pinned holdout.
They do not correct for selection bias, synthetic nature, distribution shift, dependence among
observations, or per-prediction uncertainty. They were not used to select or modify the model,
features, threshold, or claims.

The receipt and plots are in `reports/modeling/b15bab7b54bc2e1f/`. A SHA-256-indexed versioned
ledger in `reports/holdout_access/` prevents another run from consuming that holdout again. These
scores were not evaluated as calibrated probabilities, and the synthetic result does not establish
industrial usefulness.

## Acceptable result

There is no preset minimum portfolio figure. A result is valid if:

- it beats the baseline on mean CV AP and reports per-fold deltas, although those are not an
  additional gate;
- it was obtained without leakage;
- it is reproducible;
- it is reported in full, even if modest;
- its limitations are clear.
