from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "openml_cache"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TARGET_COLUMN = "__target__"
RANDOM_STATE = 42


DATASETS = {
    "pc1": {
        "openml_name": "pc1",
        "version": None,
        "description": "Small binary classification dataset for sample-size scaling.",
    },
    "kc1": {
        "openml_name": "kc1",
        "version": None,
        "description": "Binary classification dataset for sample-size scaling.",
    },
    "sylvine": {
        "openml_name": "sylvine",
        "version": None,
        "description": "Medium binary classification dataset for sample-size scaling.",
    },
    "ringnorm": {
        "openml_name": "ringnorm",
        "version": None,
        "description": "Medium binary classification dataset for sample-size scaling.",
    },
    "jm1": {
        "openml_name": "jm1",
        "version": None,
        "description": "Medium binary classification dataset for sample-size scaling.",
    },
    "default_of_credit_card_clients": {
        "openml_name": "default_of_credit_card_clients",
        "version": None,
        "description": "Larger binary classification dataset for sample-size scaling.",
    },
    "mfeat-morphological": {
        "openml_name": "mfeat-morphological",
        "version": None,
        "description": "Low-dimensional multiclass dataset for feature-count scaling.",
    },
    "mfeat-zernike": {
        "openml_name": "mfeat-zernike",
        "version": None,
        "description": "Medium-dimensional multiclass dataset for feature-count scaling.",
    },
    "mfeat-karhunen": {
        "openml_name": "mfeat-karhunen",
        "version": None,
        "description": "Higher-dimensional multiclass dataset for feature-count scaling.",
    },
    "mfeat-fourier": {
        "openml_name": "mfeat-fourier",
        "version": None,
        "description": "Higher-dimensional multiclass dataset for feature-count scaling.",
    },
    "mfeat-factors": {
        "openml_name": "mfeat-factors",
        "version": None,
        "description": "High-dimensional multiclass dataset for feature-count scaling.",
    },
    "mfeat-pixel": {
        "openml_name": "mfeat-pixel",
        "version": None,
        "description": "High-dimensional multiclass dataset for feature-count scaling.",
    },
}

SAMPLE_SCALE_DATASETS = ("pc1", "kc1", "sylvine", "ringnorm", "jm1", "default_of_credit_card_clients")
FEATURE_SCALE_DATASETS = (
    "mfeat-morphological",
    "mfeat-zernike",
    "mfeat-karhunen",
    "mfeat-fourier",
    "mfeat-factors",
    "mfeat-pixel",
)


MODEL_NAMES = ("tabpfn", "tabicl", "lightgbm", "xgboost")
DEFAULT_SAMPLE_SIZES = (1000, 3000, 10000)
SCALABILITY_SAMPLE_SIZES = (1000, 3000, 10000, 30000)
MISSING_RATES = (0.0, 0.1, 0.3, 0.5)


def ensure_project_dirs() -> None:
    for path in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
