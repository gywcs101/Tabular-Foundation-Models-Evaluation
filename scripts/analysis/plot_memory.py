from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis_config import FIGURES_DIR, MODEL_LABELS
from analysis_utils import (
    add_model_display_columns,
    apply_nature_style,
    clean_axis,
    device_label,
    load_final_metrics,
    model_palette,
    place_legend_right,
    ordered_device_types_present,
    remove_nonfinite_rows,
    save_figure,
    shorten_dataset_name,
)


AXIS_SPECS = {
    "sample_scale": ("fit_rows", "Sample scaling", "Samples"),
    "feature_scale": ("n_features", "Feature scaling", "Features"),
}


def plot_memory_bar(df: pd.DataFrame, axis: str, output_name: str, title: str) -> None:
    plot_df = df[df["experiment_axis"] == axis].copy()
    if plot_df.empty:
        return
    plot_df["dataset_short"] = plot_df["dataset"].map(shorten_dataset_name)
    plot_df = add_model_display_columns(plot_df)
    plot_df["peak_memory_gb"] = pd.to_numeric(plot_df["peak_memory_mb"], errors="coerce") / 1024.0
    plot_df = plot_df.dropna(subset=["peak_memory_gb"])
    if plot_df.empty:
        return
    palette = {MODEL_LABELS.get(k, k): v for k, v in model_palette().items()}
    model_order = (
        plot_df[["model", "model_label", "model_order"]]
        .drop_duplicates()
        .sort_values("model_order")["model_label"]
        .tolist()
    )
    devices = ordered_device_types_present(plot_df)
    fig, axes = plt.subplots(len(devices), 1, figsize=(10.6, 3.4 * len(devices)), sharex=True, sharey=True)
    if len(devices) == 1:
        axes = [axes]
    last_device = devices[-1]
    for ax, device in zip(axes, devices):
        sub = plot_df[plot_df["device_type"] == device]
        dataset_order = (
            sub[["scale_group", "dataset_short", "fit_rows", "n_features"]]
            .drop_duplicates()
            .sort_values(["scale_group", "fit_rows", "n_features", "dataset_short"])["dataset_short"]
            .tolist()
        )
        sns.barplot(
            data=sub,
            x="dataset_short",
            y="peak_memory_gb",
            hue="model_label",
            order=dataset_order,
            hue_order=model_order,
            palette=palette,
            errorbar="sd",
            capsize=0.18,
            err_kws={"color": "#8A8A8A", "linewidth": 1.0},
            width=0.50,
            ax=ax,
        )
        ax.text(
            0.01,
            0.98,
            device_label(device),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#666666",
        )
        ax.set_xlabel("")
        ax.set_ylabel("Peak memory (GB)")
        clean_axis(ax)
        ax.tick_params(axis="x", rotation=0)
        if device == last_device:
            place_legend_right(ax)
        elif ax.legend_:
            ax.legend_.remove()
    fig.suptitle(title, y=1.02, fontweight="bold")
    save_figure(fig, FIGURES_DIR / output_name)


def plot_loglog_memory(df: pd.DataFrame) -> None:
    rows = []
    for axis, x_col in [("sample_scale", "fit_rows"), ("feature_scale", "n_features")]:
        sub = df[df["experiment_axis"] == axis].copy()
        if sub.empty:
            continue
        grouped = (
            sub.groupby(["device_type", "model", "scale_group", "dataset"], dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        grouped["axis_label"] = "Sample scale" if axis == "sample_scale" else "Feature scale"
        grouped["x_value"] = grouped[x_col]
        rows.append(grouped)
    if not rows:
        return
    plot_df = pd.concat(rows, ignore_index=True)
    plot_df = add_model_display_columns(plot_df)
    plot_df["device_label"] = plot_df["device_type"].map(device_label)
    plot_df = remove_nonfinite_rows(plot_df, ["x_value", "peak_memory_mb"])
    plot_df = plot_df[(plot_df["x_value"] > 0) & (plot_df["peak_memory_mb"] > 0)]
    if plot_df.empty:
        return
    palette = {MODEL_LABELS.get(k, k): v for k, v in model_palette().items()}
    g = sns.relplot(
        data=plot_df,
        x="x_value",
        y="peak_memory_mb",
        hue="model_label",
        col="axis_label",
        row="device_label",
        kind="line",
        marker="o",
        markersize=4.2,
        linewidth=1.45,
        palette=palette,
        facet_kws={"sharex": False, "sharey": False},
        height=3.0,
        aspect=1.35,
    )
    for ax in g.axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
        clean_axis(ax)
        ax.set_xlabel("Scale variable")
        ax.set_ylabel("Peak memory (MB)")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    if g._legend:
        g._legend.set_title("")
    g.fig.suptitle("Memory scalability", y=1.04, fontweight="bold")
    save_figure(g.fig, FIGURES_DIR / "scalability_loglog_memory")


def plot_loglog_memory_axis_device(
    df: pd.DataFrame,
    axis: str,
    device_type: str,
    output_name: str,
) -> None:
    x_col, title, x_label = AXIS_SPECS[axis]
    sub = df[(df["experiment_axis"] == axis) & (df["device_type"] == device_type)].copy()
    if sub.empty:
        return
    plot_df = (
        sub.groupby(["model", "scale_group", "dataset"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    plot_df = add_model_display_columns(plot_df)
    plot_df = remove_nonfinite_rows(plot_df, [x_col, "peak_memory_mb"])
    plot_df = plot_df[(plot_df[x_col] > 0) & (plot_df["peak_memory_mb"] > 0)]
    if plot_df.empty:
        return

    palette = {MODEL_LABELS.get(k, k): v for k, v in model_palette().items()}
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    sns.lineplot(
        data=plot_df.sort_values([x_col, "model_order"]),
        x=x_col,
        y="peak_memory_mb",
        hue="model_label",
        marker="o",
        markersize=4.2,
        linewidth=1.45,
        dashes=False,
        palette=palette,
        ax=ax,
    )
    ax.text(
        0.01,
        0.98,
        device_label(device_type),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#666666",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title(title, pad=14)
    clean_axis(ax)
    place_legend_right(ax)
    save_figure(fig, FIGURES_DIR / output_name)


def plot_loglog_memory_single_axis(df: pd.DataFrame, axis: str, x_col: str, title: str, output_name: str) -> None:
    sub = df[df["experiment_axis"] == axis].copy()
    if sub.empty:
        return
    plot_df = (
        sub.groupby(["device_type", "model", "scale_group", "dataset"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    plot_df = add_model_display_columns(plot_df)
    plot_df = remove_nonfinite_rows(plot_df, [x_col, "peak_memory_mb"])
    plot_df = plot_df[(plot_df[x_col] > 0) & (plot_df["peak_memory_mb"] > 0)]
    if plot_df.empty:
        return

    palette = {MODEL_LABELS.get(k, k): v for k, v in model_palette().items()}
    devices = ordered_device_types_present(plot_df)
    fig, axes = plt.subplots(1, len(devices), figsize=(4.8 * len(devices), 3.3), sharey=True)
    if len(devices) == 1:
        axes = [axes]

    last_device = devices[-1]
    for ax, device in zip(axes, devices):
        device_df = plot_df[plot_df["device_type"] == device].sort_values([x_col, "model_order"])
        sns.lineplot(
            data=device_df,
            x=x_col,
            y="peak_memory_mb",
            hue="model_label",
            marker="o",
            markersize=4.2,
            linewidth=1.45,
            dashes=False,
            palette=palette,
            ax=ax,
        )
        ax.text(
            0.01,
            0.98,
            device_label(device),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#666666",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Samples" if x_col == "fit_rows" else "Features")
        ax.set_ylabel("Peak memory (MB)")
        clean_axis(ax)
        if device == last_device:
            place_legend_right(ax)
        elif ax.legend_:
            ax.legend_.remove()
    fig.suptitle(title, y=1.03, fontweight="bold")
    save_figure(fig, FIGURES_DIR / output_name)


def main() -> int:
    apply_nature_style()
    df = load_final_metrics()
    plot_memory_bar(
        df,
        "sample_scale",
        "sample_scale_peak_memory_bar_by_device",
        "Peak memory",
    )
    plot_memory_bar(
        df,
        "feature_scale",
        "feature_scale_peak_memory_bar_by_device",
        "Peak memory",
    )
    plot_loglog_memory(df)
    for axis in AXIS_SPECS:
        for device in ordered_device_types_present(df):
            plot_loglog_memory_axis_device(
                df,
                axis,
                device,
                f"{axis}_{device}_loglog_memory",
            )
    plot_loglog_memory_single_axis(
        df,
        "sample_scale",
        "fit_rows",
        "Sample scaling",
        "sample_scale_loglog_memory",
    )
    plot_loglog_memory_single_axis(
        df,
        "feature_scale",
        "n_features",
        "Feature scaling",
        "feature_scale_loglog_memory",
    )
    print(f"Saved memory figures to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
