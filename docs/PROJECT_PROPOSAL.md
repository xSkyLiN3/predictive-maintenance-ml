# Project proposal

## Executive summary

`Machine Failure Risk Classifier` will be a local binary-classification project that estimates the
failure risk associated with one operational observation. Its portfolio value will not come from
presenting a model as an industrial product, but from demonstrating a complete process: traceable
data, leakage prevention, a baseline, reproducible evaluation, an inference service, and honest
documentation.

## Professional objective

Create credible public evidence of foundational machine-learning engineering capability without
overstating professional seniority. The project must allow an external reviewer to verify:

- a clear understanding of the problem and the dataset's limits;
- data preparation that does not contaminate the evaluation;
- model comparison against a baseline;
- metrics appropriate for a minority class;
- a tested, executable software implementation;
- precise communication of results and limitations.

## Problem definition

### Input

One observation with:

- product type (`L`, `M`, or `H`);
- air temperature;
- process temperature;
- rotational speed;
- torque;
- tool wear.

### Output

- a `Machine failure` risk score without claiming probabilistic calibration;
- a classification according to a documented threshold;
- the model version and an educational-use warning.

### What it does not predict

- remaining time until a breakdown;
- a future sequence of states;
- the real cause of a failure;
- performance on real industrial machinery.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**, UCI Machine Learning Repository, ID 601.

- 10,000 observations.
- Synthetic data inspired by predictive-maintenance scenarios.
- Primary target: `Machine failure`.
- License: CC BY 4.0.
- No personal data.

Official source: <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>

The `UDI` and `Product ID` identifiers do not provide generalizable signal. The `TWF`, `HDF`,
`PWF`, `OSF`, and `RNF` indicators describe failure modes related to the target and will be excluded
to prevent information leakage.

## Local MVP

1. Automated dataset download and traceability.
2. Explicit schema validation.
3. Brief, reproducible EDA.
4. Stratified holdout and closed evaluation protocol.
5. `DummyClassifier` as the baseline.
6. Logistic regression and random forest within pipelines.
7. Selection through cross-validation on training only.
8. Reproducibly stored metrics and model artifact.
9. Local FastAPI API with health, information, and prediction endpoints.
10. Local HTML/CSS page for testing observations.
11. Automated tests and static analysis.
12. README, model card, results, and limitations.

## Outside the initial scope

- publication on GitHub;
- VPS, domain, and public demo;
- Docker;
- authentication, accounts, or database;
- monitoring and retraining;
- MLflow, DVC, Airflow, or Kubernetes;
- deep learning and GPU use;
- LLM, chatbot, or explanation generation;
- per-prediction SHAP;
- extensive hyperparameter search;
- temporal prediction or remaining useful life.

## Implemented architecture

```text
predictive-maintenance-ml/
├── src/predictive_maintenance/
│   ├── artifact_io.py
│   ├── config.py
│   ├── dataset.py
│   ├── validation.py
│   ├── splitting.py
│   ├── eda.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── schemas.py
│   ├── api.py
│   └── web/
├── tests/
├── notebooks/
├── data/
├── artifacts/
└── reports/
```

The EDA was implemented as a reproducible module and CLI; `notebooks/` is reserved for optional
exploratory views. All reusable logic lives in `src/`.

## Realistic estimate

- Preparation: 2–3 hours.
- Data and audit: 4–5 hours.
- Modeling and evaluation: 6–8 hours.
- API and local interface: 4–6 hours.
- Testing and documentation: 4–6 hours.

Estimated total: **20–28 hours**, spread across one or two weeks of focused work.

## Local MVP acceptance criteria

- The environment can be rebuilt from scratch by following the README.
- The dataset is obtained and validated without manual steps.
- No outcome column enters the feature set.
- Training is reproducible.
- There is an explicit comparison against a baseline.
- The holdout does not participate in model or threshold decisions.
- Real metrics are published, not a number selected for marketing.
- The API rejects invalid inputs.
- The interface works on localhost.
- Ruff and pytest pass.
- Limitations are visible.
- There are no secrets, personal data, or production claims.

Meeting these criteria enables a separate review to decide whether publication and deployment are
appropriate.
