"""Fixed, reviewable configuration shared by data and modeling milestones."""

from pathlib import Path

RANDOM_SEED = 42
HOLDOUT_FRACTION = 0.20
CV_FOLDS = 5
PRIMARY_METRIC = "average_precision"
SECONDARY_METRIC = "roc_auc"
THRESHOLD_STRATEGY = "maximize_f1_on_out_of_fold_training_predictions"
TIE_TOLERANCE = 1e-12
CV_STANDARD_DEVIATION_DDOF = 0

LOGISTIC_REGRESSION_PARAMETERS = {
    "C": 1.0,
    "l1_ratio": 0.0,
    "solver": "liblinear",
    "class_weight": "balanced",
    "max_iter": 1_000,
    "random_state": RANDOM_SEED,
}
RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 300,
    "criterion": "gini",
    "max_depth": 8,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}

DEFAULT_PROCESSED_DATA_DIR = Path("data/processed")
DEFAULT_SPLIT_MANIFEST_PATH = Path("data/split_manifest.json")
DEFAULT_EDA_REPORT_DIR = Path("reports/eda")
DEFAULT_MODEL_REPORT_DIR = Path("reports/modeling")
DEFAULT_MODEL_ARTIFACT_DIR = Path("artifacts/m3")
DEFAULT_HOLDOUT_LEDGER_DIR = Path("reports/holdout_access")
TRAIN_FILENAME = "train.csv"
HOLDOUT_FILENAME = "holdout.csv"
