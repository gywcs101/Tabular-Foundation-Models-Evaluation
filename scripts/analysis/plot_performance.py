from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis_config import FIGURES_DIR, MODEL_LABELS
from analysis_utils import (
    add_model_display_columns,
    apply_nature_style,
    clean_axis,
    load_final_metrics,
    metric_label,
    model_palette,
    ordered_dataset_labels,
    place_legend_right,
    save_figure,
    shorten_dataset_name,
)


def plot_metric_by_model(df: pd.DataFrame, metric: str, output_name: str) -> None:
    plot_df = add_model_display_columns(df)
    point_df = (
        plot_df.groupby(["model", "model_label", "model_order", "dataset"], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    order = (
        plot_df[["model", "model_label", "model_order"]]
        .drop_duplicates()
        .sort_values("model_order")["model_label"]
        .tolist()
    )
    plot_df["model_label"] = pd.Categorical(plot_df["model_label"], categories=order, ordered=True)
    palette = {MODEL_LABELS.get(k, k): v for k, v in model_palette().items()}

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sns.barplot(
        data=point_df,
        x="model_label",
        y=metric,
        hue="model_label",
        order=order,
        palette=palette,
        errorbar=None,
        width=0.48,
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel(metric_label(metric))
    ax.set_title("Prediction performance", pad=14)
    ymin = 0.68 if metric == "macro_f1" else 0.74
    ax.set_ylim(ymin, 1.01)
    clean_axis(ax)
    save_figure(fig, FIGURES_DIR / output_name)


def plot_metric_by_dataset(df: pd.DataFrame, metric: str, output_name: str) -> None:
    plot_df = add_model_display_columns(df)
    plot_df["dataset_short"] = plot_df["dataset"].map(shorten_dataset_name)
    dataset_order = ordered_dataset_labels(plot_df, "dataset_short")
    model_order = (
        plot_df[["model", "model_label", "model_order"]]
        .drop_duplicates()
        .sort_values("model_order")["model_label"]
        .tolist()
    )
    palette = {MODEL_LABELS.get(k, k): v for k, v in model_palette().items()}

    fig, ax = plt.subplots(figsize=(11.2, 4.4))
    sns.barplot(
        data=plot_df,
        x="dataset_short",
        y=metric,
        hue="model_label",
        order=dataset_order,
        hue_order=model_order,
        palette=palette,
        errorbar=None,
        width=0.58,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(metric_label(metric))
    ax.set_title("Performance by dataset", pad=14)
    lower = max(0.0, min(float(plot_df[metric].min()) - 0.04, 0.68))
    ax.set_ylim(lower, 1.01)
    clean_axis(ax)
    ax.tick_params(axis="x", rotation=0)
    place_legend_right(ax)
    save_figure(fig, FIGURES_DIR / output_name)


def main() -> int:
    apply_nature_style()
    df = load_final_metrics()
    if df.empty:
        print("final_metrics.csv is empty.")
        return 1
    for metric in ["accuracy", "macro_f1"]:
        plot_metric_by_model(df, metric, f"performance_{metric}_by_model")
        plot_metric_by_dataset(df, metric, f"performance_{metric}_by_dataset")
    print(f"Saved performance figures to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
