# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-19

### Added

- Reproducible acquisition and validation of the UCI AI4I 2020 snapshot.
- Leakage-safe feature allowlist and deterministic stratified train/holdout split.
- Dummy, logistic-regression and random-forest comparison using five-fold cross-validation.
- Out-of-fold threshold selection and a sealed, single-use final holdout evaluation receipt.
- Strict FastAPI inference service and responsive browser interface.
- Explicit abstention outside the training-reference marginal envelope.
- Model card, data attribution, decision log and reproducible reports.
- Adversarial API tests for malformed JSON, oversized bodies, non-finite values and invalid model
  outputs.
- MIT license and public-release metadata.

### Results

- Selected model: random forest.
- Cross-validation Average Precision: `0.643812`.
- Final holdout Average Precision: `0.649538` on 2,000 synthetic observations.
- At the frozen threshold: 50 true positives, 18 false negatives and 35 false positives.

### Limitations

- AI4I 2020 is synthetic; these results do not establish safety or usefulness on real machinery.
- The score is not calibrated as a real-world failure probability.
- The demo is educational and must not be used to make maintenance or safety decisions.

[1.0.0]: https://github.com/xSkyLiN3/predictive-maintenance-ml/releases/tag/v1.0.0
