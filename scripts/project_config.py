from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "openml_cache"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TARGET_COLUMN = "__target__"
RANDOM_STATE = 42


DATASETS = {
    "credit-g": {
        "openml_name": "credit-g",
        "version": None,
        "description": "Small binary classification dataset with numeric and categorical features.",
    },
    "mfeat-factors": {
        "openml_name": "mfeat-factors",
        "version": None,
        "description": "Small multiclass dataset with mostly numeric features.",
    },
    "kc1": {
        "openml_name": "kc1",
        "version": None,
        "description": "Software defect classification dataset, often imbalanced.",
    },
    "kr-vs-kp": {
        "openml_name": "kr-vs-kp",
        "version": None,
        "description": "Chess endgame dataset with many categorical features.",
    },
    "adult": {
        "openml_name": "adult",
        "version": None,
        "description": "Larger binary classification dataset with mixed feature types.",
    },
    "bank-marketing": {
        "openml_name": "bank-marketing",
        "version": None,
        "description": "Larger marketing dataset with mixed feature types.",
    },
}


MODEL_NAMES = ("tabpfn", "tabicl", "lightgbm", "xgboost")
DEFAULT_SAMPLE_SIZES = (10000,)
SCALABILITY_SAMPLE_SIZES = (500, 1000, 2000, 5000, 10000, 30000)
MISSING_RATES = (0.0, 0.1, 0.3, 0.5)


def ensure_project_dirs() -> None:
    for path in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)

