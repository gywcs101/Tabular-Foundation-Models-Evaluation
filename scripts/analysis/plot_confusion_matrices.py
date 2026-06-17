from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis_config import CONFUSION_MATRIX_DATASETS, FIGURES_DIR, MODEL_LABELS, RESULTS_RAW_DIR
from analysis_utils import apply_nature_style, pretty_dataset_title, parse_runner_dir, save_figure


MULTICLASS_DATASET_MARKER = "_multiclass"


def label_sort_key(label: object) -> tuple[int, object]:
    text = str(label)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def short_matrix_title(dataset: str, model: str) -> str:
    dataset_title = pretty_dataset_title(dataset)
    dataset_title = dataset_title.replace("mfeat morph. |", "mfeat morph.")
    dataset_title = dataset_title.replace("mfeat pixel |", "mfeat pixel")
    dataset_title = dataset_title.replace(" | multiclass", "")
    return f"{MODEL_LABELS.get(model, model)}\n{dataset_title}"


def read_confusion_matrix(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    run_dir = path.parent
    dataset_dir = run_dir.parent
    scale_group_dir = dataset_dir.parent
    axis_dir = scale_group_dir.parent
    model_dir = axis_dir.parent
    runner_dir = model_dir.parent
    runner_name, device_type = parse_runner_dir(runner_dir.name)
    matrix = pd.read_csv(path)
    labels = matrix["true_label"].astype(str).tolist()
    values = matrix.drop(columns=["true_label"])
    values.columns = [col.replace("pred_", "") for col in values.columns]
    values.index = labels
    meta = {
        "runner_dir": runner_dir.name,
        "runner_name": runner_name,
        "device_type": device_type,
        "model": model_dir.name,
        "experiment_axis": axis_dir.name,
        "scale_group": scale_group_dir.name,
        "dataset": dataset_dir.name,
        "run_id": run_dir.name,
    }
    return values, meta


def collect_selected_matrices() -> dict[tuple[str, str], list[pd.DataFrame]]:
    selected: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for path in RESULTS_RAW_DIR.rglob("confusion_matrix.csv"):
        matrix, meta = read_confusion_matrix(path)
        if meta["dataset"] not in CONFUSION_MATRIX_DATASETS:
            continue
        if MULTICLASS_DATASET_MARKER not in meta["dataset"]:
            continue
        key = (meta["dataset"], meta["model"])
        selected.setdefault(key, []).append(matrix)
    return selected


def average_matrices(matrices: list[pd.DataFrame]) -> pd.DataFrame:
    aligned = []
    labels = sorted(set().union(*[set(m.index) for m in matrices]), key=label_sort_key)
    columns = sorted(set().union(*[set(m.columns) for m in matrices]), key=label_sort_key)
    for matrix in matrices:
        aligned.append(matrix.reindex(index=labels, columns=columns, fill_value=0).astype(float))
    return sum(aligned) / len(aligned)


def plot_matrix(matrix: pd.DataFrame, dataset: str, model: str) -> None:
    fig_size = max(4.8, 0.42 * len(matrix.columns) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_size + 0.7, fig_size))
    annotations = matrix.round(1).astype(str)
    sns.heatmap(
        matrix,
        cmap="Blues",
        square=True,
        cbar_kws={"label": "Mean count"},
        linewidths=0.2,
        linecolor="#FFFFFF",
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 6.5},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(short_matrix_title(dataset, model), pad=14)
    ax.tick_params(length=0, labelsize=7)
    output_dir = FIGURES_DIR / "confusion_matrix_selected"
    save_figure(fig, output_dir / f"confusion_matrix_{model}_{dataset}")


def main() -> int:
    apply_nature_style()
    selected = collect_selected_matrices()
    if not selected:
        print("No selected confusion matrices found.")
        return 1
    output_dir = FIGURES_DIR / "confusion_matrix_selected"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("confusion_matrix_*"):
        if old_file.suffix.lower() in {".png", ".svg"}:
            old_file.unlink()
    for (dataset, model), matrices in sorted(selected.items()):
        plot_matrix(average_matrices(matrices), dataset, model)
    print(f"Saved selected confusion matrix figures to {FIGURES_DIR / 'confusion_matrix_selected'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
