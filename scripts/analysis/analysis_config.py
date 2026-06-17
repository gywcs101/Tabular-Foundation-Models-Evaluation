from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_RAW_DIR = PROJECT_ROOT / "results" / "raw"
RESULTS_FINAL_DIR = PROJECT_ROOT / "results" / "final"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
SELECTED_TALENT_DIR = PROJECT_ROOT / "data" / "selected_talent"

MODELS = ["tabicl", "tabpfn", "lightgbm", "xgboost"]
MODEL_LABELS = {
    "tabicl": "TabICL",
    "tabpfn": "TabPFN",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}

# Nature-style, low-saturation palette. Keep colors stable across all figures.
MODEL_COLORS = {
    "tabicl": "#272727",
    "tabpfn": "#7C6CCF",
    "lightgbm": "#33B5A5",
    "xgboost": "#B64342",
}

DEVICE_LABELS = {
    "light_laptop": "Light laptop",
    "gaming_laptop": "Gaming laptop",
    "unknown": "Unknown device",
}

DEVICE_ORDER = ["light_laptop", "gaming_laptop", "unknown"]
FIGURE_FORMATS = ("png", "svg")

PERFORMANCE_METRICS = ["accuracy", "balanced_accuracy", "macro_f1"]
CONFIDENCE_METRICS = [
    "mean_confidence",
    "correct_mean_confidence",
    "wrong_mean_confidence",
    "confidence_gap",
]
TIME_MEMORY_METRICS = [
    "fit_time_seconds",
    "predict_time_seconds",
    "wall_time_seconds",
    "seconds_per_test_sample",
    "peak_memory_mb",
    "training_throughput",
]

REQUIRED_RUN_FILES = [
    "metrics.csv",
    "run_config.json",
    "predictions.csv",
    "confusion_matrix.csv",
    "run.log",
]

CONFUSION_MATRIX_DATASETS = [
    "mfeat-morphological_2000rows_6feat_multiclass",
    "mfeat-pixel_2000rows_240feat_multiclass",
    "pc1_1109rows_21feat_binclass",
    "default_of_credit_card_clients_30000rows_23feat_binclass",
]

CONFUSION_MATRIX_DATASET_REASONS = {
    "mfeat-morphological_2000rows_6feat_multiclass": (
        "10-class handwritten digit task with only 6 morphological features. "
        "It represents low-dimensional multiclass behavior."
    ),
    "mfeat-pixel_2000rows_240feat_multiclass": (
        "The same handwritten digit family with 240 pixel features. "
        "It contrasts high-dimensional multiclass behavior against the low-dimensional mfeat variant."
    ),
    "pc1_1109rows_21feat_binclass": (
        "Small binary software-defect dataset from the sample-scale group. "
        "It represents small-data binary classification and likely class imbalance effects."
    ),
    "default_of_credit_card_clients_30000rows_23feat_binclass": (
        "Largest selected binary dataset. It represents large-sample behavior and practical financial-risk style classification."
    ),
}

SCALABILITY_SIZE_COLUMN = {
    "sample_scale": "fit_rows",
    "feature_scale": "n_features",
}

NATURE_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.1,
    "axes.labelsize": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
}
