from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_config import (
    DEVICE_LABELS,
    DEVICE_ORDER,
    FIGURE_FORMATS,
    FIGURES_DIR,
    MODEL_COLORS,
    MODELS,
    NATURE_RCPARAMS,
    RESULTS_FINAL_DIR,
    RESULTS_RAW_DIR,
    SELECTED_TALENT_DIR,
)


def ensure_output_dirs() -> None:
    RESULTS_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def apply_nature_style() -> None:
    mpl.rcParams.update(NATURE_RCPARAMS)


def save_figure(fig: plt.Figure, output_base: Path, formats: Iterable[str] = FIGURE_FORMATS) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", width=1.0, length=4, color="#272727")


def place_legend(ax: plt.Axes, ncols: int = 4, y: float = 1.16) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        title="",
        ncols=ncols,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
    )


def place_legend_right(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        title="",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        handlelength=1.6,
        borderaxespad=0,
    )


def device_label(device_type: str) -> str:
    return DEVICE_LABELS.get(device_type, device_type.replace("_", " ").title())


def parse_runner_dir(name: str) -> tuple[str, str]:
    for suffix in ("_light_laptop", "_gaming_laptop"):
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix[1:]
    return name, "unknown"


def discover_selected_datasets() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for info_path in sorted(SELECTED_TALENT_DIR.rglob("info.json")):
        dataset_dir = info_path.parent
        rel = dataset_dir.relative_to(SELECTED_TALENT_DIR).parts
        if len(rel) < 3:
            continue
        records.append(
            {
                "experiment_axis": rel[0],
                "scale_group": rel[1],
                "dataset": rel[2],
                "dataset_dir": str(dataset_dir),
            }
        )
    return records


def discover_runner_dirs() -> list[Path]:
    if not RESULTS_RAW_DIR.exists():
        return []
    return sorted(
        [
            path
            for path in RESULTS_RAW_DIR.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ],
        key=lambda p: p.name,
    )


def read_single_metrics(metrics_path: Path) -> dict[str, object]:
    with metrics_path.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    run_dir = metrics_path.parent
    dataset_dir = run_dir.parent
    scale_group_dir = dataset_dir.parent
    axis_dir = scale_group_dir.parent
    model_dir = axis_dir.parent
    runner_dir = model_dir.parent
    runner_name, device_type = parse_runner_dir(runner_dir.name)

    row.update(
        {
            "runner_dir": runner_dir.name,
            "runner_name_from_dir": runner_name,
            "device_type": device_type,
            "model": row.get("model") or model_dir.name,
            "experiment_axis": row.get("experiment_axis") or axis_dir.name,
            "scale_group": row.get("scale_group") or scale_group_dir.name,
            "dataset": row.get("dataset") or dataset_dir.name,
            "run_id": row.get("run_id") or run_dir.name,
            "metrics_path": str(metrics_path),
            "run_dir": str(run_dir),
        }
    )
    return row


def load_all_metrics() -> pd.DataFrame:
    rows = [read_single_metrics(path) for path in RESULTS_RAW_DIR.rglob("metrics.csv")]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col in {
            "runner_dir",
            "runner_name",
            "runner_name_from_dir",
            "device_type",
            "model",
            "dataset",
            "dataset_original_name",
            "experiment_axis",
            "scale_group",
            "task_type",
            "encoding",
            "device",
            "model_path",
            "success",
            "error_type",
            "error_message",
            "run_id",
            "metrics_path",
            "run_dir",
            "timestamp",
        }:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted
    if "success" in df.columns:
        df["success_bool"] = df["success"].astype(str).str.lower().eq("true")
    else:
        df["success_bool"] = False
    return df


def latest_successful_metrics() -> pd.DataFrame:
    df = load_all_metrics()
    if df.empty:
        return df
    df = df[df["success_bool"]].copy()
    if df.empty:
        return df
    sort_cols = ["runner_dir", "model", "experiment_axis", "scale_group", "dataset", "run_id"]
    df = df.sort_values(sort_cols)
    return df.drop_duplicates(
        subset=["runner_dir", "model", "experiment_axis", "scale_group", "dataset"],
        keep="last",
    ).reset_index(drop=True)


def ordered_models_present(df: pd.DataFrame) -> list[str]:
    present = set(df["model"].dropna().astype(str))
    return [model for model in MODELS if model in present]


def ordered_device_types_present(df: pd.DataFrame) -> list[str]:
    present = set(df["device_type"].dropna().astype(str))
    return [device for device in DEVICE_ORDER if device in present] + sorted(present - set(DEVICE_ORDER))


def add_model_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model_order"] = out["model"].map({m: i for i, m in enumerate(MODELS)}).fillna(99)
    out["model_label"] = out["model"].map(
        {"tabicl": "TabICL", "tabpfn": "TabPFN", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
    ).fillna(out["model"])
    return out


def relative_std(series: pd.Series) -> float:
    mean_value = series.mean()
    if not np.isfinite(mean_value) or abs(mean_value) < 1e-12:
        return np.nan
    return series.std(ddof=1) / abs(mean_value)


def fit_loglog_alpha(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float, int]:
    working = df[[x_col, y_col]].dropna().copy()
    working = working[(working[x_col] > 0) & (working[y_col] > 0)]
    n_points = len(working)
    if n_points < 2:
        return np.nan, np.nan, n_points
    x = np.log(working[x_col].astype(float).to_numpy())
    y = np.log(working[y_col].astype(float).to_numpy())
    alpha, intercept = np.polyfit(x, y, 1)
    return float(alpha), float(intercept), n_points


def shorten_dataset_name(name: str) -> str:
    replacements = {
        "_2000rows_": "\n2000 rows\n",
        "_1109rows_": "\n1109 rows\n",
        "_2109rows_": "\n2109 rows\n",
        "_5124rows_": "\n5124 rows\n",
        "_7400rows_": "\n7400 rows\n",
        "_10885rows_": "\n10885 rows\n",
        "_30000rows_": "\n30000 rows\n",
        "_multiclass": "\nmulticlass",
        "_binclass": "\nbinary",
        "default_of_credit_card_clients": "credit default",
        "mfeat-morphological": "mfeat morph.",
        "mfeat-fourier": "mfeat fourier",
        "mfeat-karhunen": "mfeat karh.",
        "mfeat-zernike": "mfeat zernike",
        "mfeat-factors": "mfeat factors",
        "mfeat-pixel": "mfeat pixel",
    }
    out = name
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out.replace("_", " ")


def pretty_dataset_title(name: str) -> str:
    base = shorten_dataset_name(name).replace("\n", " | ")
    return " ".join(base.split())


def metric_label(metric: str) -> str:
    labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "macro_f1": "Macro F1",
        "predict_time_seconds": "Predict time (s)",
        "wall_time_seconds": "Wall time (s)",
        "fit_time_seconds": "Fit time (s)",
        "peak_memory_mb": "Peak memory (MB)",
        "peak_memory_gb": "Peak memory (GB)",
        "confidence": "Confidence",
    }
    return labels.get(metric, metric.replace("_", " ").title())


def ordered_dataset_labels(df: pd.DataFrame, label_col: str = "dataset_short") -> list[str]:
    sort_cols = [col for col in ["experiment_axis", "scale_group", "fit_rows", "n_features", "dataset"] if col in df.columns]
    return (
        df.sort_values(sort_cols)[label_col]
        .drop_duplicates()
        .tolist()
    )


def model_palette() -> dict[str, str]:
    return MODEL_COLORS.copy()


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_final_metrics() -> pd.DataFrame:
    path = RESULTS_FINAL_DIR / "final_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/analysis/build_final_tables.py first."
        )
    return pd.read_csv(path)


def remove_nonfinite_rows(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([math.inf, -math.inf], np.nan).dropna(subset=list(columns))
