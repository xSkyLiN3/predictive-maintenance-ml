# Data

Data is neither added manually nor versioned.

- `raw/`: immutable copy obtained from UCI with the command implemented in M1.
- `processed/`: `train.csv` and `holdout.csv`, reproducible derivatives ignored by Git. They
  contain only the six allowed features and the target.

The download process must record the source, license, date, size, and SHA-256.

## Official snapshot pinned in M1

- UCI page: <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
- DOI: <https://doi.org/10.24432/C5HS5C>
- Data license: CC BY 4.0.
- Recommended citation: *AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine
  Learning Repository.* <https://doi.org/10.24432/C5HS5C>.
- ZIP: 522,170 bytes; SHA-256
  `f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252`.
- Extracted CSV: 522,048 bytes; SHA-256
  `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.

The actual retrieval date and this same provenance information are stored in
`data/raw/download_metadata.json`. That file, the ZIP, and the CSV are local and ignored by Git.
The versioned constants live in `predictive_maintenance.dataset`.

From the repository root, download, safely extract, and validate the M1 contract for the pinned
snapshot:

```powershell
.\.venv\Scripts\python.exe -m predictive_maintenance download
```

Subsequent fully offline validation:

```powershell
.\.venv\Scripts\python.exe -m predictive_maintenance validate
```

The code never silently replaces a local snapshot: if its size or checksum does not match, it
stops and requires explicit review.

The project neither modifies nor versions the original CSV. It does create local partitions,
profiles, figures, metrics, and derived models; these changes and the complete attribution are
documented in
[../docs/DATA_ATTRIBUTION.md](../docs/DATA_ATTRIBUTION.md).

## M2 partition

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m predictive_maintenance split
```

The partition uses 80% training and 20% stratified holdout with fixed seed `42`. The versioned
[split_manifest.json](split_manifest.json) makes it possible to reconstruct and verify both CSVs.
The holdout remained local, ignored, and unprofiled from partitioning through the final M3
evaluation. That evaluation has already been consumed once; its receipt is in `reports/modeling/`,
and the global ledger that prevents another consumption is in `reports/holdout_access/`.

The subsequent EDA verifies and loads only `train.csv`:

```powershell
.\.venv\Scripts\python.exe -m predictive_maintenance eda
```

Its versionable artifacts are described in [../reports/README.md](../reports/README.md).
