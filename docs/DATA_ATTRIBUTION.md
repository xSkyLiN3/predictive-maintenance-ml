# Data attribution and transformations

## Source dataset

- Title: **AI4I 2020 Predictive Maintenance Dataset**.
- Source: [UCI Machine Learning Repository, dataset 601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maint).
- DOI: <https://doi.org/10.24432/C5HS5C>.
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
  (`CC BY 4.0`).
- Introductory article listed by UCI: S. Matzka (2020), *Explainable Artificial
  Intelligence for Predictive Maintenance Applications*.

Recommended citation from UCI:

> AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository.
> <https://doi.org/10.24432/C5HS5C>.

AI4I is a synthetic dataset. The attribution to UCI and the license above apply to the dataset;
they do not constitute validation of this project or its results.

## Changes and derived material

This project neither modifies nor redistributes the original CSV in Git. The ingestion command
keeps an immutable local copy and verifies its identity by size and SHA-256. The following
transformations are performed from that copy:

- schema validation and descriptive profiling;
- selection through a six-variable allowlist and exclusion of identifiers/failure modes;
- reproducible 80/20 stratified partition into local files;
- EDA statistics and figures computed only on training;
- cross-validation, OOF predictions, model selection, and threshold selection;
- metrics and figures from one final holdout evaluation;
- pipeline training and exposure through an educational demo.

The reports, figures, code, and pipeline produced by this project are released under the MIT
license while retaining this attribution to the source dataset. The exact evaluated pipeline is
copied unchanged into deployment. Re-training or re-serializing it on another platform is not
expected to reproduce byte-for-byte identical Joblib output, and replacing the pinned artifact
would break the SHA-256 receipt. The project's MIT license is independent of the
`CC BY 4.0` license that UCI declares for the AI4I dataset; the original data is not redistributed.

## Snapshot used

- ZIP: 522,170 bytes; SHA-256
  `f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252`.
- CSV: 522,048 bytes; SHA-256
  `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.

The original files remain ignored by Git. Technical download and verification details are in
[../data/README.md](../data/README.md).
