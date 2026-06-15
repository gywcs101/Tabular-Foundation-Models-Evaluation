from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from model_utils import fit_classifier, make_classifier
from project_config import PROJECT_ROOT, RANDOM_STATE


DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "selected_talent"
    / "feature_scale"
    / "F1_6_20"
    / "mfeat-morphological_2000rows_6feat_multiclass"
)

DEFAULT_TABICL_MODEL_PATH = Path(
    r"E:\Software_Download\TabICL_models\tabicl-classifier-v2-20260212.ckpt"
)


def progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def memory_mb() -> float | None:
    try:
        import os

        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def load_optional_array(dataset_dir: Path, name: str, rows: int) -> np.ndarray:
    path = dataset_dir / name
    if not path.exists():
        return np.empty((rows, 0))
    array = np.load(path, allow_pickle=True)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def load_talent_split(dataset_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.load(dataset_dir / f"y_{split}.npy", allow_pickle=True)
    n_array = load_optional_array(dataset_dir, f"N_{split}.npy", len(y))
    c_array = load_optional_array(dataset_dir, f"C_{split}.npy", len(y))
    return n_array, c_array, y


def preprocess_talent_arrays(
    n_train: np.ndarray,
    c_train: np.ndarray,
    n_test: np.ndarray,
    c_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, int]]:
    parts_train: list[np.ndarray] = []
    parts_test: list[np.ndarray] = []

    if n_train.shape[1] > 0:
        numeric_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        parts_train.append(numeric_imputer.fit_transform(n_train.astype(float)))
        parts_test.append(numeric_imputer.transform(n_test.astype(float)))

    categorical_feature_indices: list[int] = []
    if c_train.shape[1] > 0:
        cat_imputer = SimpleImputer(
            strategy="constant",
            fill_value="__missing__",
            keep_empty_features=True,
        )
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        c_train_filled = cat_imputer.fit_transform(c_train.astype(object))
        c_test_filled = cat_imputer.transform(c_test.astype(object))
        c_train_encoded = encoder.fit_transform(c_train_filled)
        c_test_encoded = encoder.transform(c_test_filled)
        categorical_start = n_train.shape[1]
        categorical_feature_indices = list(
            range(categorical_start, categorical_start + c_train.shape[1])
        )
        parts_train.append(c_train_encoded)
        parts_test.append(c_test_encoded)

    if not parts_train:
        raise ValueError("No numeric or categorical feature arrays were found.")

    x_train = np.concatenate(parts_train, axis=1)
    x_test = np.concatenate(parts_test, axis=1)
    feature_info = {
        "n_numeric_features": int(n_train.shape[1]),
        "n_categorical_features": int(c_train.shape[1]),
        "n_features_after_preprocessing": int(x_train.shape[1]),
    }
    return x_train, x_test, categorical_feature_indices, feature_info


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    rows = []
    for label, values in zip(labels, matrix, strict=True):
        row = {"true_label": label}
        row.update({f"pred_{pred_label}": int(value) for pred_label, value in zip(labels, values, strict=True)})
        rows.append(row)
    write_csv(path, rows)


def result_directory(
    model: str,
    experiment_axis: str,
    scale_group: str,
    dataset_dir: Path,
    run_id: str,
) -> Path:
    return PROJECT_ROOT / "results" / model / experiment_axis / scale_group / dataset_dir.name / run_id


def run_experiment(args: argparse.Namespace) -> Path:
    dataset_dir = Path(args.dataset_dir)
    progress(f"Loading dataset metadata: {dataset_dir}")
    info = read_json(dataset_dir / "info.json")
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = result_directory(args.model, args.experiment_axis, args.scale_group, dataset_dir, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"Output folder: {output_dir}")

    start_wall = datetime.now().isoformat(timespec="seconds")
    log_lines = [
        f"run_id={run_id}",
        f"start_time={start_wall}",
        f"dataset_dir={dataset_dir}",
        f"model={args.model}",
        f"device={args.device}",
        f"model_path={args.model_path}",
    ]

    progress("Reading TALENT train/val/test .npy files")
    n_train, c_train, y_train = load_talent_split(dataset_dir, "train")
    n_val, c_val, y_val = load_talent_split(dataset_dir, "val")
    n_test, c_test, y_test = load_talent_split(dataset_dir, "test")
    progress(
        "Loaded splits: "
        f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, "
        f"numeric_features={n_train.shape[1]}, categorical_features={c_train.shape[1]}"
    )

    if args.use_val_in_train:
        progress("Using train + val as TabICL context/training samples")
        n_fit = np.concatenate([n_train, n_val], axis=0)
        c_fit = np.concatenate([c_train, c_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
    else:
        progress("Using train only as TabICL context/training samples")
        n_fit, c_fit, y_fit = n_train, c_train, y_train

    progress("Encoding labels")
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_fit, y_test]).astype(str))
    y_fit_encoded = label_encoder.transform(y_fit.astype(str))
    y_test_encoded = label_encoder.transform(y_test.astype(str))
    target_classes = [str(label) for label in label_encoder.classes_]

    progress("Preprocessing features")
    x_fit, x_test, categorical_indices, feature_info = preprocess_talent_arrays(
        n_fit,
        c_fit,
        n_test,
        c_test,
    )
    progress(f"Prepared arrays: fit={x_fit.shape}, test={x_test.shape}")

    memory_before = memory_mb()
    fit_start = time.perf_counter()
    predict_time = np.nan
    predictions = None
    probabilities = None
    success = False
    error_type = ""
    error_message = ""

    try:
        progress(f"Creating model: {args.model}")
        if args.model == "tabicl" and args.model_path:
            progress(f"Using local TabICL checkpoint: {args.model_path}")
        model = make_classifier(
            model_name=args.model,
            seed=args.seed,
            device=args.device,
            foundation_estimators=args.foundation_estimators,
            tree_estimators=args.tree_estimators,
            model_path=args.model_path,
        )
        progress("Fitting model")
        fit_classifier(model, x_fit, y_fit_encoded, categorical_feature_indices=categorical_indices)
        fit_time = time.perf_counter() - fit_start
        progress(f"Fit finished in {fit_time:.3f} seconds")

        progress("Predicting test set")
        predict_start = time.perf_counter()
        predictions = model.predict(x_test)
        predict_time = time.perf_counter() - predict_start
        progress(f"Prediction finished in {predict_time:.3f} seconds")
        if hasattr(model, "predict_proba"):
            try:
                progress("Trying to collect predicted probabilities")
                probabilities = model.predict_proba(x_test)
            except Exception:
                progress("Predicted probabilities are not available; continuing without them")
                probabilities = None
        success = True
    except Exception as error:
        fit_time = time.perf_counter() - fit_start
        error_type = type(error).__name__
        error_message = str(error)
        progress(f"Experiment failed: {error_type}: {error_message}")

    memory_after = memory_mb()
    end_wall = datetime.now().isoformat(timespec="seconds")

    rows_total = int(info.get("train_size", 0)) + int(info.get("val_size", 0)) + int(info.get("test_size", 0))
    metrics_row = {
        "run_id": run_id,
        "timestamp": start_wall,
        "dataset": dataset_dir.name,
        "dataset_original_name": info.get("name", dataset_dir.name),
        "experiment_axis": args.experiment_axis,
        "scale_group": args.scale_group,
        "model": args.model,
        "task_type": info.get("task_type", ""),
        "total_rows": rows_total,
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "fit_rows": int(len(y_fit)),
        "test_rows": int(len(y_test)),
        "n_features": int(info.get("n_num_features", 0)) + int(info.get("n_cat_features", 0)),
        "n_num_features": int(info.get("n_num_features", 0)),
        "n_cat_features": int(info.get("n_cat_features", 0)),
        "n_classes": len(target_classes),
        "encoding": "talent_numeric_plus_ordinal_categorical",
        "device": args.device,
        "model_path": args.model_path,
        "random_seed": args.seed,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "fit_time_seconds": fit_time,
        "predict_time_seconds": predict_time,
        "seconds_per_test_sample": np.nan,
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
        "success": success,
        "error_type": error_type,
        "error_message": error_message,
    }

    if success and predictions is not None:
        progress("Computing metrics and detailed prediction files")
        metrics_row.update(
            {
                "accuracy": accuracy_score(y_test_encoded, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test_encoded, predictions),
                "macro_f1": f1_score(y_test_encoded, predictions, average="macro", zero_division=0),
                "seconds_per_test_sample": predict_time / max(len(y_test_encoded), 1),
            }
        )

        prediction_rows = []
        y_true_labels = label_encoder.inverse_transform(y_test_encoded)
        y_pred_labels = label_encoder.inverse_transform(np.asarray(predictions, dtype=int))
        for index, (true_label, pred_label) in enumerate(zip(y_true_labels, y_pred_labels, strict=True)):
            row = {
                "sample_index": index,
                "y_true": true_label,
                "y_pred": pred_label,
                "correct": str(true_label) == str(pred_label),
            }
            if probabilities is not None:
                for class_index, class_label in enumerate(target_classes):
                    row[f"prob_class_{class_label}"] = float(probabilities[index, class_index])
            prediction_rows.append(row)
        write_csv(output_dir / "predictions.csv", prediction_rows)

        matrix = confusion_matrix(y_test_encoded, predictions, labels=list(range(len(target_classes))))
        write_confusion_matrix(output_dir / "confusion_matrix.csv", target_classes, matrix)

    progress("Saving metrics, config, and run log")
    write_csv(output_dir / "metrics.csv", [metrics_row])

    config = {
        "run_id": run_id,
        "dataset": dataset_dir.name,
        "dataset_dir": str(dataset_dir),
        "dataset_info": info,
        "model": args.model,
        "experiment_axis": args.experiment_axis,
        "scale_group": args.scale_group,
        "random_seed": args.seed,
        "device": args.device,
        "data_split": "TALENT original train/val/test",
        "use_val_in_train": args.use_val_in_train,
        "encoding": metrics_row["encoding"],
        "target_classes": target_classes,
        "feature_info": feature_info,
        "categorical_feature_indices": categorical_indices,
        "foundation_estimators": args.foundation_estimators,
        "tree_estimators": args.tree_estimators,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            "tabicl": package_version("tabicl"),
            "numpy": package_version("numpy"),
            "scikit-learn": package_version("scikit-learn"),
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log_lines.extend(
        [
            f"end_time={end_wall}",
            f"success={success}",
            f"fit_time_seconds={fit_time}",
            f"predict_time_seconds={predict_time}",
            f"accuracy={metrics_row['accuracy']}",
            f"balanced_accuracy={metrics_row['balanced_accuracy']}",
            f"macro_f1={metrics_row['macro_f1']}",
            f"error_type={error_type}",
            f"error_message={error_message}",
        ]
    )
    (output_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    progress(f"Done. Results saved to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one model on one selected TALENT dataset.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model", default="tabicl")
    parser.add_argument("--experiment-axis", default="feature_scale")
    parser.add_argument("--scale-group", default="F1_6_20")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_TABICL_MODEL_PATH),
        help="Local TabICL checkpoint path. Leave empty to allow automatic download.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--foundation-estimators", type=int, default=8)
    parser.add_argument("--tree-estimators", type=int, default=300)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--use-val-in-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TALENT validation rows together with train rows for fitting.",
    )
    args = parser.parse_args()
    if args.model_path == "":
        args.model_path = None
    output_dir = run_experiment(args)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
