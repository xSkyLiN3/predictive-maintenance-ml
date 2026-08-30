# Reports

This directory contains reproducible, reviewed results, plots, and documentation. Temporary files
must be stored in `reports/tmp/`, which Git ignores.

The M2 EDA is in [eda/EDA_REPORT.md](eda/EDA_REPORT.md), with a JSON summary and six figures
generated exclusively from the training partition.

It is regenerated from the repository root, after materializing the split, with:

```powershell
.\.venv\Scripts\python.exe -m predictive_maintenance eda
```

M3 is in [modeling/b15bab7b54bc2e1f/M3_REPORT.md](modeling/b15bab7b54bc2e1f/M3_REPORT.md). It
includes the CV and OOF-threshold receipts, two selection figures, and the final confusion matrix.
The run selected random forest, achieved mean CV AP `0.643812`, and final AP `0.649538`.

`holdout_access/<sha256>.json` is the global, versioned ledger for the single evaluation. It must
not be deleted or edited: it prevents another run or artifact directory from reopening the same
holdout. Compatible reruns validate and reuse the final receipt.
