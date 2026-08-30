# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026-08-30

### Changed

- Made English the canonical public language across the browser demo, API descriptions, warnings,
  validation messages, documentation, and generated reports.
- Added an English public evaluation report and presentation figures derived from the
  training-only workflow and frozen final receipt.
- Clarified the reference-envelope and industrial-validation language to avoid overstating OOD
  detection or real-world applicability.
- Added an automated language regression gate for editable public surfaces.

### Integrity

- The evaluated model, run `b15bab7b54bc2e1f`, threshold, metrics, pipeline bytes, manifests,
  single-use holdout receipt, and archived M3 evidence remain unchanged.
- The final holdout was not reopened for this localization release.

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
- Exact SHA-verified evaluated pipeline for portable Linux deployment without redistributing data.
- Hardened non-root Docker runtime, cross-platform CI and Trivy vulnerability scanning.
- Public stateless HTTPS demo with request-size, rate and resource limits.

### Results

- Selected model: random forest.
- Cross-validation Average Precision: `0.643812`.
- Final holdout Average Precision: `0.649538` on 2,000 synthetic observations.
- At the frozen threshold: 50 true positives, 18 false negatives and 35 false positives.

### Limitations

- AI4I 2020 is synthetic; these results do not establish safety or usefulness on real machinery.
- The score is not calibrated as a real-world failure probability.
- The demo is educational and must not be used to make maintenance or safety decisions.

[Unreleased]: https://github.com/xSkyLiN3/predictive-maintenance-ml/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/xSkyLiN3/predictive-maintenance-ml/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/xSkyLiN3/predictive-maintenance-ml/releases/tag/v1.0.0
