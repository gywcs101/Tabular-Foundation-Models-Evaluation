from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis_config import FIGURES_DIR, MODEL_LABELS, RESULTS_RAW_DIR
from analysis_utils import (
    add_model_display_columns,
    apply_nature_style,
    clean_axis,
    model_palette,
    place_legend,
    place_legend_right,
    parse_runner_dir,
    save_figure,
)


def read_predictions_file(path: Path) -> pd.DataFrame:
    run_dir = path.parent
    dataset_dir = run_dir.parent
    scale_group_dir = dataset_dir.parent
    axis_dir = scale_group_dir.parent
    model_dir = axis_dir.parent
    runner_dir = model_dir.parent
    runner_name, device_type = parse_runner_dir(runner_dir.name)
    df = pd.read_csv(path)
    df["runner_dir"] = runner_dir.name
    df["runner_name_from_dir"] = runner_name
    df["device_type"] = device_type
    df["model"] = model_dir.name
    df["experiment_axis"] = axis_dir.name
    df["scale_group"] = scale_group_dir.name
    df["dataset"] = dataset_dir.name
    df["run_id"] = run_dir.name
    return df


def load_predictions() -> pd.DataFrame:
    rows = [read_predictions_file(path) for path in RESULTS_RAW_DIR.rglob("predictions.csv")]
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["correct"] = df["correct"].astype(str).str.lower().eq("true")
    return df.dropna(subset=["confidence"])


def sample_for_plot(df: pd.DataFrame, max_rows_per_model: int = 6000) -> pd.DataFrame:
    parts = []
    for model, group in df.groupby("model", dropna=False):
        if len(group) > max_rows_per_model:
            parts.append(group.sample(max_rows_per_model, random_state=42))
        else:
            parts.append(group)
    return pd.concat(parts, ignore_index=True) if parts else df


def plot_histogram(df: pd.DataFrame) -> None:
    plot_df = add_model_display_columns(sample_for_plot(df))
    palette = {MODEL_LABELS.get(k, k): v for k, v in model_palette().items()}
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    sns.histplot(
        data=plot_df,
        x="confidence",
        hue="model_label",
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence distribution", pad=18)
    ax.set_xlim(0, 1)
    clean_axis(ax)
    if ax.legend_:
        ax.legend_.set_title("")
    save_figure(fig, FIGURES_DIR / "confidence_histogram_by_model")


def plot_correct_vs_wrong(df: pd.DataFrame) -> None:
    working = df.copy()
    working["prediction_status"] = working["correct"].map({True: "Correct", False: "Wrong"})
    run_means = (
        working.groupby(["runner_dir", "model", "dataset", "run_id", "prediction_status"], dropna=False)["confidence"]
        .mean()
        .reset_index()
    )
    plot_df = add_model_display_columns(run_means)
    palette = {"Correct": "#5F7FC4", "Wrong": "#E7A39D"}
    order = (
        plot_df[["model_label", "model_order"]]
        .drop_duplicates()
        .sort_values("model_order")["model_label"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    sns.barplot(
        data=plot_df,
        x="model_label",
        y="confidence",
        hue="prediction_status",
        order=order,
        hue_order=["Correct", "Wrong"],
        palette=palette,
        errorbar="sd",
        capsize=0.16,
        err_kws={"color": "#6F6F6F", "linewidth": 1.0},
        width=0.50,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean confidence")
    ax.set_title("Confidence reliability", pad=18)
    ax.set_ylim(0, 1.04)
    clean_axis(ax)
    place_legend_right(ax)
    save_figure(fig, FIGURES_DIR / "confidence_correct_vs_wrong")


def main() -> int:
    apply_nature_style()
    df = load_predictions()
    if df.empty:
        print("No prediction files with confidence values found.")
        return 1
    plot_histogram(df)
    plot_correct_vs_wrong(df)
    print(f"Saved confidence figures to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
