from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from analysis_config import FIGURES_DIR, MODEL_LABELS, RESULTS_RAW_DIR
from analysis_utils import (
    add_model_display_columns,
    apply_nature_style,
    clean_axis,
    model_palette,
    ordered_models_present,
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
    df = pd.read_csv(path, encoding="utf-8-sig")
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


def model_label_order(df: pd.DataFrame) -> list[str]:
    ordered_models = ordered_models_present(df)
    if not ordered_models:
        ordered_models = ["tabicl", "tabpfn", "lightgbm", "xgboost"]
    return [MODEL_LABELS.get(model, model) for model in ordered_models]


def palette_by_label(label_order: list[str]) -> dict[str, str]:
    colors = model_palette()
    reverse_labels = {MODEL_LABELS.get(model, model): model for model in colors}
    return {
        label: colors[reverse_labels[label]]
        for label in label_order
        if label in reverse_labels
    }


def plot_histogram_faceted(df: pd.DataFrame) -> None:
    plot_df = add_model_display_columns(sample_for_plot(df))
    label_order = model_label_order(plot_df)
    palette = palette_by_label(label_order)
    n_models = len(label_order)
    ncols = 2 if n_models > 1 else 1
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.8, 2.7 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    bins = np.linspace(0, 1, 21)
    for idx, label in enumerate(label_order):
        ax = axes_list[idx]
        subset = plot_df[plot_df["model_label"] == label]
        sns.histplot(
            data=subset,
            x="confidence",
            bins=bins,
            stat="density",
            common_norm=False,
            element="step",
            fill=True,
            color=palette[label],
            alpha=0.28,
            linewidth=1.5,
            ax=ax,
        )
        ax.set_title(label, pad=10)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Prediction confidence")
        ax.set_ylabel("Density")
        clean_axis(ax)
    for ax in axes_list[n_models:]:
        ax.axis("off")
    fig.suptitle("Confidence distribution by model", y=0.995, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, FIGURES_DIR / "confidence_distribution_faceted_by_model")


def plot_ecdf(df: pd.DataFrame) -> None:
    plot_df = add_model_display_columns(sample_for_plot(df))
    label_order = model_label_order(plot_df)
    palette = palette_by_label(label_order)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    sns.ecdfplot(
        data=plot_df,
        x="confidence",
        hue="model_label",
        hue_order=label_order,
        palette=palette,
        linewidth=2.0,
        ax=ax,
    )
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Cumulative share")
    ax.set_title("ECDF of confidence", pad=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    clean_axis(ax)
    if ax.legend_ is not None:
        ax.legend_.remove()
    handles = [Line2D([0], [0], color=palette[label], lw=2.5) for label in label_order]
    ax.legend(
        handles,
        label_order,
        title="",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        handlelength=1.8,
        borderaxespad=0,
    )
    save_figure(fig, FIGURES_DIR / "confidence_ecdf_by_model")


def plot_boxplot(df: pd.DataFrame) -> None:
    plot_df = add_model_display_columns(sample_for_plot(df))
    label_order = model_label_order(plot_df)
    palette = palette_by_label(label_order)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    sns.boxplot(
        data=plot_df,
        x="model_label",
        y="confidence",
        hue="model_label",
        hue_order=label_order,
        order=label_order,
        palette=palette,
        dodge=False,
        width=0.52,
        showfliers=False,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_xlabel("")
    ax.set_ylabel("Prediction confidence")
    ax.set_title("Confidence summary", pad=18)
    ax.set_ylim(0, 1.04)
    clean_axis(ax)
    save_figure(fig, FIGURES_DIR / "confidence_boxplot_by_model")


def main() -> int:
    apply_nature_style()
    df = load_predictions()
    if df.empty:
        print("No prediction files with confidence values found.")
        return 1
    plot_histogram_faceted(df)
    plot_ecdf(df)
    plot_boxplot(df)
    print(f"Saved confidence figures to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
