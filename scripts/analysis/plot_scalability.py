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
    metric_label,
    model_palette,
    place_legend_right,
    remove_nonfinite_rows,
    save_figure,
    ordered_device_types_present,
)


AXIS_SPECS = {
    "sample_scale": ("fit_rows", "Sample scaling", "Samples"),
    "feature_scale": ("n_features", "Feature scaling", "Features"),
}


def aggregate_for_axis(df: pd.DataFrame, axis: str, metric: str, x_col: str, by_device: bool) -> pd.DataFrame:
    group_cols = ["model", "scale_group", x_col]
    if by_device:
        group_cols.insert(0, "device_type")
    out = (
        df[df["experiment_axis"] == axis]
        .groupby(group_cols, dropna=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": metric, "std": f"{metric}_std"})
    )
    return add_model_display_columns(out).sort_values([x_col, "model_order"])


def plot_line(df: pd.DataFrame, x_col: str, y_col: str, title: str, output_name: str, by_device: bool) -> None:
    plot_df = remove_nonfinite_rows(df, [x_col, y_col])
    if plot_df.empty:
        return
    palette = {MODEL_LABELS.get(k, k): v for k, v in model_palette().items()}
    if by_device:
        devices = ordered_device_types_present(plot_df)
        fig, axes = plt.subplots(1, len(devices), figsize=(4.8 * len(devices), 3.3), sharey=True)
        if len(devices) == 1:
            axes = [axes]
        last_device = devices[-1]
        for ax, device in zip(axes, devices):
            sub = plot_df[plot_df["device_type"] == device]
            sns.lineplot(
                data=sub,
                x=x_col,
                y=y_col,
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
            clean_axis(ax)
            ax.set_xlabel("Samples" if x_col == "fit_rows" else "Features")
            ax.set_ylabel(metric_label(y_col))
            if y_col.endswith("_time_seconds"):
                ax.set_yscale("log")
            if device == last_device:
                place_legend_right(ax)
            elif ax.legend_:
                ax.legend_.remove()
        fig.suptitle(title, y=1.03, fontweight="bold")
    else:
        fig, ax = plt.subplots(figsize=(5.0, 3.3))
        sns.lineplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue="model_label",
            marker="o",
            markersize=4.2,
            linewidth=1.45,
            dashes=False,
            palette=palette,
            ax=ax,
        )
        ax.set_title(title, pad=14)
        ax.set_xlabel("Samples" if x_col == "fit_rows" else "Features")
        ax.set_ylabel(metric_label(y_col))
        clean_axis(ax)
        place_legend_right(ax)
    save_figure(fig, FIGURES_DIR / output_name)


def plot_loglog(df: pd.DataFrame, output_name: str) -> None:
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
    plot_df = remove_nonfinite_rows(plot_df, ["x_value", "wall_time_seconds"])
    plot_df = plot_df[(plot_df["x_value"] > 0) & (plot_df["wall_time_seconds"] > 0)]
    if plot_df.empty:
        return
    palette = {MODEL_LABELS.get(k, k): v for k, v in model_palette().items()}

    g = sns.relplot(
        data=plot_df,
        x="x_value",
        y="wall_time_seconds",
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
        ax.set_ylabel("Wall time (s)")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    if g._legend:
        g._legend.set_title("")
    g.fig.suptitle("Time scalability", y=1.04, fontweight="bold")
    save_figure(g.fig, FIGURES_DIR / output_name)


def plot_loglog_axis_device(
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
    plot_df = remove_nonfinite_rows(plot_df, [x_col, "wall_time_seconds"])
    plot_df = plot_df[(plot_df[x_col] > 0) & (plot_df["wall_time_seconds"] > 0)]
    if plot_df.empty:
        return

    palette = {MODEL_LABELS.get(k, k): v for k, v in model_palette().items()}
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    sns.lineplot(
        data=plot_df.sort_values([x_col, "model_order"]),
        x=x_col,
        y="wall_time_seconds",
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
    ax.set_ylabel("Wall time (s)")
    ax.set_title(title, pad=14)
    clean_axis(ax)
    place_legend_right(ax)
    save_figure(fig, FIGURES_DIR / output_name)


def plot_loglog_single_axis(df: pd.DataFrame, axis: str, x_col: str, title: str, output_name: str) -> None:
    sub = df[df["experiment_axis"] == axis].copy()
    if sub.empty:
        return
    plot_df = (
        sub.groupby(["device_type", "model", "scale_group", "dataset"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    plot_df = add_model_display_columns(plot_df)
    plot_df = remove_nonfinite_rows(plot_df, [x_col, "wall_time_seconds"])
    plot_df = plot_df[(plot_df[x_col] > 0) & (plot_df["wall_time_seconds"] > 0)]
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
            y="wall_time_seconds",
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
        ax.set_ylabel("Wall time (s)")
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
    for metric in ["accuracy", "macro_f1"]:
        plot_line(
            aggregate_for_axis(df, "sample_scale", metric, "fit_rows", by_device=False),
            "fit_rows",
            metric,
            "Sample scaling",
            f"sample_scale_{metric}_line",
            by_device=False,
        )
        plot_line(
            aggregate_for_axis(df, "feature_scale", metric, "n_features", by_device=False),
            "n_features",
            metric,
            "Feature scaling",
            f"feature_scale_{metric}_line",
            by_device=False,
        )
    for metric in ["predict_time_seconds", "wall_time_seconds"]:
        plot_line(
            aggregate_for_axis(df, "sample_scale", metric, "fit_rows", by_device=True),
            "fit_rows",
            metric,
            "Sample scaling",
            f"sample_scale_{metric}_line_by_device",
            by_device=True,
        )
        plot_line(
            aggregate_for_axis(df, "feature_scale", metric, "n_features", by_device=True),
            "n_features",
            metric,
            "Feature scaling",
            f"feature_scale_{metric}_line_by_device",
            by_device=True,
        )
    plot_loglog(df, "scalability_loglog_time")
    for axis in AXIS_SPECS:
        for device in ordered_device_types_present(df):
            plot_loglog_axis_device(
                df,
                axis,
                device,
                f"{axis}_{device}_loglog_time",
            )
    plot_loglog_single_axis(
        df,
        "sample_scale",
        "fit_rows",
        "Sample scaling",
        "sample_scale_loglog_time",
    )
    plot_loglog_single_axis(
        df,
        "feature_scale",
        "n_features",
        "Feature scaling",
        "feature_scale_loglog_time",
    )
    print(f"Saved scalability figures to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
