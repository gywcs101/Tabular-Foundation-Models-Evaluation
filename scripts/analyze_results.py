from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from project_config import FIGURES_DIR, RESULTS_DIR


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    df = pd.read_csv(path)
    if "success" in df.columns:
        df = df[df["success"].astype(str).str.lower().isin(["true", "1"])]
    return df


def save_accuracy_by_dataset(df: pd.DataFrame, output_dir: Path) -> None:
    if not {"dataset", "model", "accuracy"}.issubset(df.columns):
        return
    plt.figure(figsize=(11, 6))
    sns.barplot(data=df, x="dataset", y="accuracy", hue="model")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("Accuracy by Dataset")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_dataset.png", dpi=200)
    plt.close()


def save_scalability_plots(df: pd.DataFrame, output_dir: Path) -> None:
    required = {"dataset", "model", "actual_sample_size", "accuracy", "predict_time_seconds"}
    if not required.issubset(df.columns):
        return
    plot_df = df.copy()
    plot_df = plot_df[plot_df["actual_sample_size"].astype(str) != "full"]
    plot_df["actual_sample_size"] = plot_df["actual_sample_size"].astype(int)

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=plot_df,
        x="actual_sample_size",
        y="accuracy",
        hue="model",
        style="dataset",
        marker="o",
    )
    plt.title("Accuracy vs Sample Size")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_sample_size.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=plot_df,
        x="actual_sample_size",
        y="predict_time_seconds",
        hue="model",
        style="dataset",
        marker="o",
    )
    plt.title("Inference Time vs Sample Size")
    plt.tight_layout()
    plt.savefig(output_dir / "inference_time_vs_sample_size.png", dpi=200)
    plt.close()


def save_missing_plot(path: Path, output_dir: Path) -> None:
    if not path.exists():
        return
    df = load_metrics(path)
    required = {"dataset", "model", "missing_rate", "accuracy"}
    if not required.issubset(df.columns):
        return
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="missing_rate",
        y="accuracy",
        hue="model",
        style="dataset",
        marker="o",
    )
    plt.title("Accuracy under Missing Values")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_value_robustness.png", dpi=200)
    plt.close()


def save_retrieval_plot(path: Path, output_dir: Path) -> None:
    if not path.exists():
        return
    df = load_metrics(path)
    required = {"dataset", "context_strategy", "accuracy"}
    if not required.issubset(df.columns):
        return
    plt.figure(figsize=(9, 6))
    sns.barplot(data=df, x="dataset", y="accuracy", hue="context_strategy")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("Random vs Retrieval Context")
    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_context_accuracy.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures from result CSV files.")
    parser.add_argument("--metrics", default=str(RESULTS_DIR / "metrics.csv"))
    parser.add_argument("--missing", default=str(RESULTS_DIR / "missing_values.csv"))
    parser.add_argument("--retrieval", default=str(RESULTS_DIR / "retrieval_icl.csv"))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    output_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics)
    if metrics_path.exists():
        metrics_df = load_metrics(metrics_path)
        save_accuracy_by_dataset(metrics_df, output_dir)
        save_scalability_plots(metrics_df, output_dir)

    save_missing_plot(Path(args.missing), output_dir)
    save_retrieval_plot(Path(args.retrieval), output_dir)
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

