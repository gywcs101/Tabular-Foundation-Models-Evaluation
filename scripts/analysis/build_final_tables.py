from __future__ import annotations

import pandas as pd

from analysis_config import (
    CONFIDENCE_METRICS,
    PERFORMANCE_METRICS,
    RESULTS_FINAL_DIR,
    SCALABILITY_SIZE_COLUMN,
    TIME_MEMORY_METRICS,
)
from analysis_utils import (
    fit_loglog_alpha,
    latest_successful_metrics,
    relative_std,
    write_dataframe,
)


def summarize_metrics(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    available_metrics = [metric for metric in metrics if metric in df.columns]
    grouped = df.groupby(group_cols, dropna=False)[available_metrics]
    summary = grouped.agg(["mean", "std", "min", "max", "count"]).reset_index()
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def build_device_consistency(df: pd.DataFrame) -> pd.DataFrame:
    metrics = PERFORMANCE_METRICS + CONFIDENCE_METRICS + TIME_MEMORY_METRICS
    rows: list[dict[str, object]] = []
    group_cols = ["model", "dataset", "experiment_axis", "scale_group"]
    for key, group in df.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key))
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    **key_dict,
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "min": values.min(),
                    "max": values.max(),
                    "relative_std": relative_std(values),
                    "n_runs": len(values),
                }
            )
    return pd.DataFrame(rows)


def build_scalability_alpha(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "predict_time_seconds",
        "wall_time_seconds",
        "peak_memory_mb",
    ]
    rows: list[dict[str, object]] = []
    group_cols = ["experiment_axis", "device_type", "model"]
    for (axis, device_type, model), group in df.groupby(group_cols, dropna=False):
        x_col = SCALABILITY_SIZE_COLUMN.get(axis)
        if not x_col or x_col not in group.columns:
            continue
        group_means = (
            group.groupby(["scale_group", "dataset"], dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        for metric in metrics:
            if metric not in group_means.columns:
                continue
            alpha, intercept, n_points = fit_loglog_alpha(group_means, x_col, metric)
            rows.append(
                {
                    "experiment_axis": axis,
                    "device_type": device_type,
                    "model": model,
                    "metric": metric,
                    "x_metric": x_col,
                    "alpha": alpha,
                    "intercept": intercept,
                    "n_points": n_points,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    RESULTS_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    df = latest_successful_metrics()
    if df.empty:
        print("No successful metrics found.")
        return 1

    write_dataframe(df, RESULTS_FINAL_DIR / "final_metrics.csv")

    consistency = build_device_consistency(df)
    write_dataframe(consistency, RESULTS_FINAL_DIR / "device_consistency.csv")

    model_dataset = summarize_metrics(
        df,
        ["model", "dataset", "experiment_axis", "scale_group"],
        PERFORMANCE_METRICS + CONFIDENCE_METRICS + TIME_MEMORY_METRICS,
    )
    write_dataframe(model_dataset, RESULTS_FINAL_DIR / "model_dataset_summary.csv")

    device_type = summarize_metrics(
        df,
        ["device_type", "model", "dataset", "experiment_axis", "scale_group"],
        TIME_MEMORY_METRICS,
    )
    write_dataframe(device_type, RESULTS_FINAL_DIR / "device_type_summary.csv")

    alpha = build_scalability_alpha(df)
    write_dataframe(alpha, RESULTS_FINAL_DIR / "scalability_alpha.csv")

    print(f"Saved final tables to {RESULTS_FINAL_DIR}")
    print(f"final_metrics rows: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
