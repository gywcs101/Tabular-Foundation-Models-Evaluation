from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from core.run_talent_single_dataset import (
    PeakMemorySampler,
    collect_confidence_metrics,
    load_talent_split,
    preprocess_talent_arrays,
    progress,
    read_json,
    write_confusion_matrix,
    write_csv,
)
from model_utils import fit_classifier, make_classifier
from project_config import PROJECT_ROOT, RANDOM_STATE, TABICL_MODEL_PATH


DATASET_PRESETS = {
    "mfeat-morphological": {
        "dataset_dir": PROJECT_ROOT
        / "data"
        / "selected_talent"
        / "feature_scale"
        / "F1_6_20"
        / "mfeat-morphological_2000rows_6feat_multiclass",
        "experiment_axis": "feature_scale",
        "scale_group": "F1_6_20",
    },
    "mfeat-zernike": {
        "dataset_dir": PROJECT_ROOT
        / "data"
        / "selected_talent"
        / "feature_scale"
        / "F2_20_100"
        / "mfeat-zernike_2000rows_47feat_multiclass",
        "experiment_axis": "feature_scale",
        "scale_group": "F2_20_100",
    },
    "mfeat-fourier": {
        "dataset_dir": PROJECT_ROOT
        / "data"
        / "selected_talent"
        / "feature_scale"
        / "F2_20_100"
        / "mfeat-fourier_2000rows_76feat_multiclass",
        "experiment_axis": "feature_scale",
        "scale_group": "F2_20_100",
    },
    "pc1": {
        "dataset_dir": PROJECT_ROOT
        / "data"
        / "selected_talent"
        / "sample_scale"
        / "A_1000_3000"
        / "pc1_1109rows_21feat_binclass",
        "experiment_axis": "sample_scale",
        "scale_group": "A_1000_3000",
    },
    "sylvine": {
        "dataset_dir": PROJECT_ROOT
        / "data"
        / "selected_talent"
        / "sample_scale"
        / "B_3000_10000"
        / "sylvine_5124rows_20feat_binclass",
        "experiment_axis": "sample_scale",
        "scale_group": "B_3000_10000",
    },
}

DEFAULT_DATASETS = ("mfeat-morphological", "pc1")
DEFAULT_RETRIEVAL_METRICS = ("euclidean",)
RETRIEVAL_METRICS = ("euclidean", "cosine", "weighted_euclidean")


def package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def parse_dataset_names(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in names if name not in DATASET_PRESETS]
    if unknown:
        raise ValueError(f"Unknown dataset preset(s): {unknown}. Available: {sorted(DATASET_PRESETS)}")
    return names


def parse_retrieval_metrics(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in names if name not in RETRIEVAL_METRICS]
    if unknown:
        raise ValueError(f"Unknown retrieval metric(s): {unknown}. Available: {list(RETRIEVAL_METRICS)}")
    return names


def make_tabicl(seed: int, device: str, model_path: str, foundation_estimators: int):
    return make_classifier(
        model_name="tabicl",
        seed=seed,
        device=device,
        foundation_estimators=foundation_estimators,
        model_path=model_path,
    )


def predict_with_optional_proba(model: Any, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    predictions = np.asarray(model.predict(x_test), dtype=int)
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = np.asarray(model.predict_proba(x_test), dtype=float)
        except Exception:
            probabilities = None
    return predictions, probabilities


def align_probability_row(
    sample_probabilities: np.ndarray | None,
    context_labels: np.ndarray,
    n_classes: int,
) -> np.ndarray | None:
    if sample_probabilities is None:
        return None
    row = np.asarray(sample_probabilities[0], dtype=float)
    if row.shape[0] == n_classes:
        return row

    aligned = np.zeros(n_classes, dtype=float)
    present_classes = np.asarray(sorted(set(int(label) for label in context_labels)), dtype=int)
    if row.shape[0] == len(present_classes):
        aligned[present_classes] = row
        return aligned
    return None


def compute_neighbor_indices(
    strategy: str,
    x_context_scaled: np.ndarray,
    x_eval_scaled: np.ndarray,
    k: int,
    seed: int,
    retrieval_metric: str = "euclidean",
    feature_weights: np.ndarray | None = None,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    k = min(k, len(x_context_scaled))
    if strategy == "retrieval_k":
        if retrieval_metric == "weighted_euclidean":
            if feature_weights is None:
                raise ValueError("feature_weights must be provided for weighted_euclidean.")
            weights = np.asarray(feature_weights, dtype=float)
            x_context_for_search = x_context_scaled * np.sqrt(weights)
            x_eval_for_search = x_eval_scaled * np.sqrt(weights)
            nearest_neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean")
        else:
            x_context_for_search = x_context_scaled
            x_eval_for_search = x_eval_scaled
            nearest_neighbors = NearestNeighbors(n_neighbors=k, metric=retrieval_metric)
        nearest_neighbors.fit(x_context_for_search)
        return [
            np.asarray(indices, dtype=int)
            for indices in nearest_neighbors.kneighbors(x_eval_for_search, return_distance=False)
        ]
    if strategy == "random_k":
        return [
            rng.choice(len(x_context_scaled), size=k, replace=False).astype(int)
            for _ in range(len(x_eval_scaled))
        ]
    raise ValueError(f"Unsupported neighbor strategy: {strategy}")


def write_prediction_outputs(
    output_dir: Path,
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    confidence: np.ndarray,
    target_classes: list[str],
) -> None:
    label_decoder = LabelEncoder()
    label_decoder.fit(np.asarray(target_classes, dtype=str))
    y_true_labels = label_decoder.inverse_transform(y_true)
    y_pred_labels = label_decoder.inverse_transform(predictions)
    prediction_rows = []
    for index, (true_label, pred_label) in enumerate(zip(y_true_labels, y_pred_labels, strict=True)):
        row: dict[str, Any] = {
            "sample_index": index,
            "y_true": true_label,
            "y_pred": pred_label,
            "correct": str(true_label) == str(pred_label),
            "confidence": float(confidence[index]) if np.isfinite(confidence[index]) else np.nan,
        }
        if probabilities is not None:
            for class_index, class_label in enumerate(target_classes):
                row[f"prob_class_{class_label}"] = float(probabilities[index, class_index])
        prediction_rows.append(row)
    write_csv(output_dir / "predictions.csv", prediction_rows)

    matrix = confusion_matrix(y_true, predictions, labels=list(range(len(target_classes))))
    write_confusion_matrix(output_dir / "confusion_matrix.csv", target_classes, matrix)


def write_test_indices(output_dir: Path, test_indices: np.ndarray) -> None:
    rows = [
        {
            "evaluation_row": int(row_index),
            "original_test_index": int(test_index),
        }
        for row_index, test_index in enumerate(test_indices)
    ]
    write_csv(output_dir / "test_indices.csv", rows)


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_info: dict[str, Any],
    dataset_name: str,
    preset: dict[str, Any],
    strategy: str,
    k: int | None,
    retrieval_metric: str | None,
    seed: int,
    test_rows_used: int,
    target_classes: list[str],
    feature_info: dict[str, int],
    categorical_indices: list[int],
    memory_sampler: PeakMemorySampler,
) -> None:
    run_config = {
        "dataset_preset": dataset_name,
        "dataset": preset["dataset_dir"].name,
        "dataset_dir": str(preset["dataset_dir"]),
        "dataset_info": dataset_info,
        "model": "tabicl",
        "context_strategy": strategy,
        "k": k,
        "retrieval_metric": retrieval_metric,
        "seed": seed,
        "test_rows_used": test_rows_used,
        "test_selection": args.test_selection,
        "test_sample_size": args.test_sample_size,
        "test_sample_seed": args.test_sample_seed,
        "device": args.device,
        "model_path": args.model_path,
        "use_val_in_train": True,
        "data_split": "TALENT original train/val/test",
        "encoding": "talent_numeric_plus_ordinal_categorical",
        "target_classes": target_classes,
        "feature_info": feature_info,
        "categorical_feature_indices": categorical_indices,
        "foundation_estimators": args.foundation_estimators,
        "memory_sampler_available": memory_sampler.available,
        "memory_sampler_backend": memory_sampler.backend,
        "memory_sampler_error": memory_sampler.error,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            "numpy": package_version("numpy"),
            "scikit-learn": package_version("scikit-learn"),
            "tabicl": package_version("tabicl"),
            "psutil": package_version("psutil"),
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_strategy_result(
    output_dir: Path,
    metrics_row: dict[str, Any],
    log_lines: list[str],
) -> None:
    write_csv(output_dir / "metrics.csv", [metrics_row])
    (output_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def run_full_context(
    args: argparse.Namespace,
    output_dir: Path,
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    categorical_indices: list[int],
    target_classes: list[str],
) -> dict[str, Any]:
    fit_start = time.perf_counter()
    model = make_tabicl(args.seed, args.device, args.model_path, args.foundation_estimators)
    fit_classifier(model, x_context, y_context, categorical_feature_indices=categorical_indices)
    fit_time = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    predictions, probabilities = predict_with_optional_proba(model, x_eval)
    predict_time = time.perf_counter() - predict_start

    confidence_metrics, confidence = collect_confidence_metrics(probabilities, predictions, y_eval)
    write_prediction_outputs(output_dir, y_eval, predictions, probabilities, confidence, target_classes)
    return {
        "fit_time_seconds": fit_time,
        "predict_time_seconds": predict_time,
        "predictions": predictions,
        **confidence_metrics,
    }


def run_k_context(
    args: argparse.Namespace,
    output_dir: Path,
    strategy: str,
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    x_context_scaled: np.ndarray,
    x_eval_scaled: np.ndarray,
    categorical_indices: list[int],
    target_classes: list[str],
    k: int,
    seed: int,
    retrieval_metric: str = "euclidean",
    feature_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    neighbor_indices = compute_neighbor_indices(
        strategy,
        x_context_scaled,
        x_eval_scaled,
        k,
        seed,
        retrieval_metric=retrieval_metric,
        feature_weights=feature_weights,
    )
    fit_time_total = 0.0
    predict_time_total = 0.0
    predictions = []
    probability_rows = []
    probabilities_available = True
    n_classes = len(target_classes)

    for sample_index, context_indices in enumerate(neighbor_indices, start=1):
        if sample_index == 1 or sample_index % 25 == 0 or sample_index == len(neighbor_indices):
            progress(f"{strategy}: predicting sample {sample_index}/{len(neighbor_indices)}")
        model = make_tabicl(seed, args.device, args.model_path, args.foundation_estimators)

        fit_start = time.perf_counter()
        fit_classifier(
            model,
            x_context[context_indices],
            y_context[context_indices],
            categorical_feature_indices=categorical_indices,
        )
        fit_time_total += time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        sample_prediction, sample_probabilities = predict_with_optional_proba(
            model,
            x_eval[sample_index - 1 : sample_index],
        )
        predict_time_total += time.perf_counter() - predict_start
        predictions.append(int(sample_prediction[0]))
        probability_row = align_probability_row(
            sample_probabilities,
            y_context[context_indices],
            n_classes,
        )
        if probability_row is None:
            probabilities_available = False
        else:
            probability_rows.append(probability_row)

    predictions_array = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probability_rows, dtype=float) if probabilities_available else None
    confidence_metrics, confidence = collect_confidence_metrics(probabilities, predictions_array, y_eval)
    write_prediction_outputs(output_dir, y_eval, predictions_array, probabilities, confidence, target_classes)
    return {
        "fit_time_seconds": fit_time_total,
        "predict_time_seconds": predict_time_total,
        "predictions": predictions_array,
        **confidence_metrics,
    }


def run_dataset_strategy(
    args: argparse.Namespace,
    dataset_name: str,
    preset: dict[str, Any],
    strategy: str,
    k: int | None,
    seed: int,
    arrays: dict[str, Any],
    retrieval_metric: str | None = None,
) -> Path:
    dataset_dir = preset["dataset_dir"]
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if strategy == "retrieval_k" and retrieval_metric:
        strategy_dir = f"{strategy}_{retrieval_metric}_{k}"
    else:
        strategy_dir = strategy if k is None else f"{strategy}_{k}"
    output_dir = (
        PROJECT_ROOT
        / "results"
        / "innovation"
        / "retrieval_context"
        / args.runner_name
        / dataset_dir.name
        / strategy_dir
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    progress(f"Running {dataset_name} | {strategy_dir}")
    wall_start = time.perf_counter()
    start_time = datetime.now().isoformat(timespec="seconds")
    memory_sampler = PeakMemorySampler(args.memory_sample_interval)
    memory_sampler.start()
    log_lines = [
        f"run_id={run_id}",
        f"runner_name={args.runner_name}",
        f"dataset={dataset_dir.name}",
        f"context_strategy={strategy}",
        f"k={k}",
        f"retrieval_metric={retrieval_metric}",
        f"seed={seed}",
        f"start_time={start_time}",
    ]

    success = False
    error_type = ""
    error_message = ""
    result: dict[str, Any] = {
        "fit_time_seconds": np.nan,
        "predict_time_seconds": np.nan,
        "probabilities_available": False,
        "mean_confidence": np.nan,
        "correct_mean_confidence": np.nan,
        "wrong_mean_confidence": np.nan,
        "confidence_gap": np.nan,
        "predictions": None,
    }

    try:
        if strategy == "full":
            result = run_full_context(
                args,
                output_dir,
                arrays["x_context"],
                arrays["y_context"],
                arrays["x_eval"],
                arrays["y_eval"],
                arrays["categorical_indices"],
                arrays["target_classes"],
            )
        else:
            if k is None:
                raise ValueError("K must be provided for random_k or retrieval_k.")
            result = run_k_context(
                args,
                output_dir,
                strategy,
                arrays["x_context"],
                arrays["y_context"],
                arrays["x_eval"],
                arrays["y_eval"],
                arrays["x_context_scaled"],
                arrays["x_eval_scaled"],
                arrays["categorical_indices"],
                arrays["target_classes"],
                k,
                seed,
                retrieval_metric=retrieval_metric or "euclidean",
                feature_weights=arrays.get("feature_weights"),
            )
        success = True
    except Exception as error:
        error_type = type(error).__name__
        error_message = str(error)
        progress(f"{strategy_dir} failed: {error_type}: {error_message}")

    peak_memory_mb = memory_sampler.stop()
    wall_time_seconds = time.perf_counter() - wall_start
    predictions = result.get("predictions")
    accuracy = np.nan
    balanced_accuracy = np.nan
    macro_f1 = np.nan
    if success and predictions is not None:
        y_eval = arrays["y_eval"]
        accuracy = accuracy_score(y_eval, predictions)
        balanced_accuracy = balanced_accuracy_score(y_eval, predictions)
        macro_f1 = f1_score(y_eval, predictions, average="macro", zero_division=0)

    metrics_row: dict[str, Any] = {
        "run_id": run_id,
        "runner_name": args.runner_name,
        "timestamp": start_time,
        "dataset": dataset_dir.name,
        "dataset_preset": dataset_name,
        "experiment_axis": preset["experiment_axis"],
        "scale_group": preset["scale_group"],
        "model": "tabicl",
        "context_strategy": strategy,
        "k": k if k is not None else "",
        "retrieval_metric": retrieval_metric or "",
        "random_seed": seed,
        "fit_rows": len(arrays["y_context"]),
        "test_rows_total": arrays["test_rows_total"],
        "test_rows_used": len(arrays["y_eval"]),
        "n_features": arrays["feature_info"]["n_features_after_preprocessing"],
        "n_classes": len(arrays["target_classes"]),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "probabilities_available": result.get("probabilities_available", False),
        "mean_confidence": result.get("mean_confidence", np.nan),
        "correct_mean_confidence": result.get("correct_mean_confidence", np.nan),
        "wrong_mean_confidence": result.get("wrong_mean_confidence", np.nan),
        "confidence_gap": result.get("confidence_gap", np.nan),
        "fit_time_seconds": result.get("fit_time_seconds", np.nan),
        "predict_time_seconds": result.get("predict_time_seconds", np.nan),
        "wall_time_seconds": wall_time_seconds,
        "seconds_per_test_sample": wall_time_seconds / max(len(arrays["y_eval"]), 1),
        "peak_memory_mb": peak_memory_mb,
        "success": success,
        "error_type": error_type,
        "error_message": error_message,
    }
    save_strategy_result(output_dir, metrics_row, log_lines + [f"success={success}", f"wall_time_seconds={wall_time_seconds}"])
    write_test_indices(output_dir, arrays["test_indices"])
    write_run_config(
        output_dir,
        args,
        arrays["dataset_info"],
        dataset_name,
        preset,
        strategy,
        k,
        retrieval_metric,
        seed,
        len(arrays["y_eval"]),
        arrays["target_classes"],
        arrays["feature_info"],
        arrays["categorical_indices"],
        memory_sampler,
    )
    progress(
        f"{dataset_name} | {strategy_dir} | success={success} | "
        f"accuracy={accuracy:.4f} | macro_f1={macro_f1:.4f} | wall={wall_time_seconds:.2f}s"
    )
    return output_dir


def prepare_dataset_arrays(args: argparse.Namespace, preset: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = preset["dataset_dir"]
    dataset_info = read_json(dataset_dir / "info.json")
    n_train, c_train, y_train = load_talent_split(dataset_dir, "train")
    n_val, c_val, y_val = load_talent_split(dataset_dir, "val")
    n_test, c_test, y_test = load_talent_split(dataset_dir, "test")

    n_context = np.concatenate([n_train, n_val], axis=0)
    c_context = np.concatenate([c_train, c_val], axis=0)
    y_context_raw = np.concatenate([y_train, y_val], axis=0)

    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_context_raw, y_test]).astype(str))
    y_context = label_encoder.transform(y_context_raw.astype(str))
    y_test_encoded = label_encoder.transform(y_test.astype(str))
    target_classes = [str(label) for label in label_encoder.classes_]

    x_context, x_test, categorical_indices, feature_info = preprocess_talent_arrays(
        n_context,
        c_context,
        n_test,
        c_test,
    )
    if args.test_selection == "random":
        sample_size = min(args.test_sample_size, len(y_test_encoded))
        if sample_size <= 0:
            raise ValueError("--test-sample-size must be positive when --test-selection=random.")
        rng = np.random.default_rng(args.test_sample_seed)
        test_indices = np.sort(rng.choice(len(y_test_encoded), size=sample_size, replace=False))
    else:
        test_limit = len(y_test_encoded) if args.test_limit <= 0 else min(args.test_limit, len(y_test_encoded))
        test_indices = np.arange(test_limit)
    x_eval = x_test[test_indices]
    y_eval = y_test_encoded[test_indices]

    scaler = StandardScaler()
    x_context_scaled = scaler.fit_transform(x_context.astype(float))
    x_eval_scaled = scaler.transform(x_eval.astype(float))
    feature_weights = mutual_info_classif(
        x_context_scaled,
        y_context,
        discrete_features=False,
        random_state=args.seed,
    )
    if np.max(feature_weights) > 0:
        feature_weights = feature_weights / np.max(feature_weights)
    feature_weights = 0.1 + feature_weights

    return {
        "dataset_info": dataset_info,
        "x_context": x_context,
        "y_context": y_context,
        "x_eval": x_eval,
        "y_eval": y_eval,
        "x_context_scaled": x_context_scaled,
        "x_eval_scaled": x_eval_scaled,
        "feature_weights": feature_weights,
        "categorical_indices": categorical_indices,
        "feature_info": feature_info,
        "target_classes": target_classes,
        "test_rows_total": len(y_test_encoded),
        "test_indices": test_indices,
    }


def collect_metrics(output_root: Path) -> None:
    rows = []
    for path in sorted(output_root.rglob("metrics.csv")):
        with path.open(newline="", encoding="utf-8-sig") as file:
            row = next(csv.DictReader(file))
        row["metrics_path"] = str(path)
        rows.append(row)
    if rows:
        write_csv(output_root / "summary_metrics.csv", rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TabICL retrieval-context pilot experiments.")
    parser.add_argument("--runner-name", default="innovation_pilot")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--test-limit", type=int, default=100, help="0 means use the full test split.")
    parser.add_argument("--test-selection", choices=["head", "random"], default="head")
    parser.add_argument("--test-sample-size", type=int, default=100)
    parser.add_argument("--test-sample-seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--random-repeats", type=int, default=1)
    parser.add_argument("--retrieval-metrics", default=",".join(DEFAULT_RETRIEVAL_METRICS))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", default=str(TABICL_MODEL_PATH))
    parser.add_argument("--foundation-estimators", type=int, default=8)
    parser.add_argument("--memory-sample-interval", type=float, default=0.05)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    args = parser.parse_args()

    dataset_names = parse_dataset_names(args.datasets)
    retrieval_metrics = parse_retrieval_metrics(args.retrieval_metrics)
    output_root = PROJECT_ROOT / "results" / "innovation" / "retrieval_context" / args.runner_name
    for dataset_name in dataset_names:
        preset = DATASET_PRESETS[dataset_name]
        progress(f"Preparing dataset: {dataset_name}")
        arrays = prepare_dataset_arrays(args, preset)
        if not args.skip_full:
            run_dataset_strategy(args, dataset_name, preset, "full", None, args.seed, arrays)
        if not args.skip_random:
            for repeat_index in range(args.random_repeats):
                run_dataset_strategy(
                    args,
                    dataset_name,
                    preset,
                    "random_k",
                    args.k,
                    args.seed + repeat_index,
                    arrays,
                )
        for retrieval_metric in retrieval_metrics:
            run_dataset_strategy(
                args,
                dataset_name,
                preset,
                "retrieval_k",
                args.k,
                args.seed,
                arrays,
                retrieval_metric=retrieval_metric,
            )

    collect_metrics(output_root)
    progress(f"Summary saved under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
