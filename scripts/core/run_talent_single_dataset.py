from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import threading
import time
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from model_utils import fit_classifier, make_classifier
from project_config import PROJECT_ROOT, RANDOM_STATE, TABICL_MODEL_PATH, TABPFN_MODEL_PATH


DEFAULT_TABICL_MODEL_PATH = TABICL_MODEL_PATH
DEFAULT_TABPFN_MODEL_PATH = TABPFN_MODEL_PATH
DEFAULT_MEMORY_SAMPLE_INTERVAL = 0.05


class PeakMemorySampler:
    def __init__(self, interval_seconds: float = DEFAULT_MEMORY_SAMPLE_INTERVAL) -> None:
        self.interval_seconds = interval_seconds
        self.peak_memory_mb = np.nan
        self.available = False
        self.backend = ""
        self.error = ""
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        memory_reader = self._make_memory_reader()
        if memory_reader is None:
            return
        self.available = True

        def sample() -> None:
            while not self._stop_event.is_set():
                try:
                    rss_mb = memory_reader()
                    if np.isnan(self.peak_memory_mb) or rss_mb > self.peak_memory_mb:
                        self.peak_memory_mb = rss_mb
                except Exception as error:
                    self.error = f"{type(error).__name__}: {error}"
                    break
                self._stop_event.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def _make_memory_reader(self):
        try:
            import psutil

            process = psutil.Process(os.getpid())
            self.backend = "psutil_rss"
            return lambda: process.memory_info().rss / (1024 * 1024)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        if os.name != "nt":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_current_process.argtypes = []
            get_current_process.restype = wintypes.HANDLE
            get_process_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_process_memory_info.restype = wintypes.BOOL
            handle = get_current_process()
            self.backend = "windows_working_set"

            def read_working_set_mb() -> float:
                counters = ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(ProcessMemoryCounters)
                ok = get_process_memory_info(
                    handle,
                    ctypes.byref(counters),
                    counters.cb,
                )
                if not ok:
                    raise OSError("GetProcessMemoryInfo failed")
                return counters.WorkingSetSize / (1024 * 1024)

            self.error = ""
            return read_working_set_mb
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            return None

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        return float(self.peak_memory_mb)


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
        row.update(
            {
                f"pred_{pred_label}": int(value)
                for pred_label, value in zip(labels, values, strict=True)
            }
        )
        rows.append(row)
    write_csv(path, rows)


def result_directory(
    runner_name: str,
    model: str,
    experiment_axis: str,
    scale_group: str,
    dataset_dir: Path,
    run_id: str,
) -> Path:
    return (
        PROJECT_ROOT
        / "results"
        / "raw"
        / runner_name
        / model
        / experiment_axis
        / scale_group
        / dataset_dir.name
        / run_id
    )


def finite_or_nan(value: float | np.floating[Any]) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def collect_confidence_metrics(
    probabilities: np.ndarray | None,
    predictions: np.ndarray,
    y_true: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    confidence = np.full(len(predictions), np.nan)
    metrics = {
        "probabilities_available": False,
        "mean_confidence": np.nan,
        "correct_mean_confidence": np.nan,
        "wrong_mean_confidence": np.nan,
        "confidence_gap": np.nan,
    }
    if probabilities is None:
        return metrics, confidence

    predictions_int = np.asarray(predictions, dtype=int)
    row_indices = np.arange(len(predictions_int))
    confidence = probabilities[row_indices, predictions_int].astype(float)
    correct_mask = predictions_int == y_true
    correct_confidence = confidence[correct_mask]
    wrong_confidence = confidence[~correct_mask]
    correct_mean = np.mean(correct_confidence) if len(correct_confidence) else np.nan
    wrong_mean = np.mean(wrong_confidence) if len(wrong_confidence) else np.nan
    gap = correct_mean - wrong_mean if np.isfinite(correct_mean) and np.isfinite(wrong_mean) else np.nan
    metrics.update(
        {
            "probabilities_available": True,
            "mean_confidence": finite_or_nan(np.mean(confidence)),
            "correct_mean_confidence": finite_or_nan(correct_mean),
            "wrong_mean_confidence": finite_or_nan(wrong_mean),
            "confidence_gap": finite_or_nan(gap),
        }
    )
    return metrics, confidence


def run_experiment(args: argparse.Namespace) -> Path:
    dataset_dir = Path(args.dataset_dir)
    progress(f"Loading dataset metadata: {dataset_dir}")
    info = read_json(dataset_dir / "info.json")
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = result_directory(
        args.runner_name,
        args.model,
        args.experiment_axis,
        args.scale_group,
        dataset_dir,
        run_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"Output folder: {output_dir}")

    wall_start = time.perf_counter()
    start_wall = datetime.now().isoformat(timespec="seconds")
    memory_sampler = PeakMemorySampler(args.memory_sample_interval)
    memory_sampler.start()
    log_lines = [
        f"run_id={run_id}",
        f"runner_name={args.runner_name}",
        f"start_time={start_wall}",
        f"dataset_dir={dataset_dir}",
        f"model={args.model}",
        f"device={args.device}",
        f"model_path={args.model_path}",
        f"memory_sample_interval_seconds={args.memory_sample_interval}",
    ]

    fit_time = np.nan
    predict_time = np.nan
    probabilities = None
    predictions = None
    y_test_encoded = None
    target_classes: list[str] = []
    feature_info: dict[str, int] = {}
    categorical_indices: list[int] = []
    success = False
    error_type = ""
    error_message = ""
    n_train = c_train = y_train = n_val = c_val = y_val = n_test = c_test = y_test = None
    y_fit = None

    try:
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
            progress("Using train + val as fitting/context samples")
            n_fit = np.concatenate([n_train, n_val], axis=0)
            c_fit = np.concatenate([c_train, c_val], axis=0)
            y_fit = np.concatenate([y_train, y_val], axis=0)
        else:
            progress("Using train only as fitting/context samples")
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

        progress(f"Creating model: {args.model}")
        if args.model_path:
            progress(f"Using local model checkpoint/path: {args.model_path}")
        model = make_classifier(
            model_name=args.model,
            seed=args.seed,
            device=args.device,
            foundation_estimators=args.foundation_estimators,
            tree_estimators=args.tree_estimators,
            model_path=args.model_path,
        )

        progress("Fitting model")
        fit_start = time.perf_counter()
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
                progress("Collecting predicted probabilities")
                probabilities = model.predict_proba(x_test)
            except Exception as probability_error:
                progress(
                    "Predicted probabilities are not available; "
                    f"continuing without them ({type(probability_error).__name__})"
                )
                probabilities = None

        success = True
    except Exception as error:
        error_type = type(error).__name__
        error_message = str(error)
        progress(f"Experiment failed: {error_type}: {error_message}")

    peak_memory_mb = memory_sampler.stop()
    wall_time_seconds = time.perf_counter() - wall_start
    end_wall = datetime.now().isoformat(timespec="seconds")

    train_rows = int(len(y_train)) if y_train is not None else int(info.get("train_size", 0))
    val_rows = int(len(y_val)) if y_val is not None else int(info.get("val_size", 0))
    test_rows = int(len(y_test)) if y_test is not None else int(info.get("test_size", 0))
    fit_rows = int(len(y_fit)) if y_fit is not None else train_rows + (val_rows if args.use_val_in_train else 0)
    total_rows = train_rows + val_rows + test_rows
    n_features = int(info.get("n_num_features", 0)) + int(info.get("n_cat_features", 0))
    training_throughput = fit_rows * n_features / fit_time if np.isfinite(fit_time) and fit_time > 0 else np.nan

    metrics_row: dict[str, Any] = {
        "run_id": run_id,
        "runner_name": args.runner_name,
        "timestamp": start_wall,
        "dataset": dataset_dir.name,
        "dataset_original_name": info.get("name", dataset_dir.name),
        "experiment_axis": args.experiment_axis,
        "scale_group": args.scale_group,
        "model": args.model,
        "task_type": info.get("task_type", ""),
        "total_rows": total_rows,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "fit_rows": fit_rows,
        "test_rows": test_rows,
        "n_features": n_features,
        "n_num_features": int(info.get("n_num_features", 0)),
        "n_cat_features": int(info.get("n_cat_features", 0)),
        "n_classes": len(target_classes) or int(info.get("n_classes", 0)),
        "encoding": "talent_numeric_plus_ordinal_categorical",
        "device": args.device,
        "model_path": args.model_path,
        "random_seed": args.seed,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "probabilities_available": False,
        "mean_confidence": np.nan,
        "correct_mean_confidence": np.nan,
        "wrong_mean_confidence": np.nan,
        "confidence_gap": np.nan,
        "fit_time_seconds": fit_time,
        "predict_time_seconds": predict_time,
        "wall_time_seconds": wall_time_seconds,
        "seconds_per_test_sample": (
            predict_time / max(test_rows, 1) if np.isfinite(predict_time) else np.nan
        ),
        "peak_memory_mb": peak_memory_mb,
        "training_throughput": training_throughput,
        "success": success,
        "error_type": error_type,
        "error_message": error_message,
    }

    if success and predictions is not None and y_test_encoded is not None:
        progress("Computing metrics and detailed prediction files")
        predictions_array = np.asarray(predictions, dtype=int)
        confidence_metrics, confidence = collect_confidence_metrics(
            probabilities,
            predictions_array,
            y_test_encoded,
        )
        metrics_row.update(
            {
                "accuracy": accuracy_score(y_test_encoded, predictions_array),
                "balanced_accuracy": balanced_accuracy_score(y_test_encoded, predictions_array),
                "macro_f1": f1_score(
                    y_test_encoded,
                    predictions_array,
                    average="macro",
                    zero_division=0,
                ),
                **confidence_metrics,
            }
        )

        label_encoder = LabelEncoder()
        label_encoder.fit(np.asarray(target_classes, dtype=str))
        y_true_labels = label_encoder.inverse_transform(y_test_encoded)
        y_pred_labels = label_encoder.inverse_transform(predictions_array)
        prediction_rows = []
        for index, (true_label, pred_label) in enumerate(
            zip(y_true_labels, y_pred_labels, strict=True)
        ):
            row = {
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

        matrix = confusion_matrix(
            y_test_encoded,
            predictions_array,
            labels=list(range(len(target_classes))),
        )
        write_confusion_matrix(output_dir / "confusion_matrix.csv", target_classes, matrix)

    progress("Saving metrics, config, and run log")
    write_csv(output_dir / "metrics.csv", [metrics_row])

    run_config = {
        "run_id": run_id,
        "runner_name": args.runner_name,
        "dataset": dataset_dir.name,
        "dataset_dir": str(dataset_dir),
        "dataset_info": info,
        "model": args.model,
        "experiment_axis": args.experiment_axis,
        "scale_group": args.scale_group,
        "random_seed": args.seed,
        "device": args.device,
        "model_path": args.model_path,
        "data_split": "TALENT original train/val/test",
        "use_val_in_train": args.use_val_in_train,
        "encoding": metrics_row["encoding"],
        "target_classes": target_classes,
        "feature_info": feature_info,
        "categorical_feature_indices": categorical_indices,
        "foundation_estimators": args.foundation_estimators,
        "tree_estimators": args.tree_estimators,
        "memory_sample_interval_seconds": args.memory_sample_interval,
        "memory_sampler_available": memory_sampler.available,
        "memory_sampler_backend": memory_sampler.backend,
        "memory_sampler_error": memory_sampler.error,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "psutil": package_version("psutil"),
            "scikit-learn": package_version("scikit-learn"),
            "tabicl": package_version("tabicl"),
            "tabpfn": package_version("tabpfn"),
            "lightgbm": package_version("lightgbm"),
            "xgboost": package_version("xgboost"),
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log_lines.extend(
        [
            f"end_time={end_wall}",
            f"runner_name={args.runner_name}",
            f"success={success}",
            f"fit_time_seconds={fit_time}",
            f"predict_time_seconds={predict_time}",
            f"wall_time_seconds={wall_time_seconds}",
            f"peak_memory_mb={peak_memory_mb}",
            f"training_throughput={training_throughput}",
            f"accuracy={metrics_row['accuracy']}",
            f"balanced_accuracy={metrics_row['balanced_accuracy']}",
            f"macro_f1={metrics_row['macro_f1']}",
            f"mean_confidence={metrics_row['mean_confidence']}",
            f"correct_mean_confidence={metrics_row['correct_mean_confidence']}",
            f"wrong_mean_confidence={metrics_row['wrong_mean_confidence']}",
            f"confidence_gap={metrics_row['confidence_gap']}",
            f"probabilities_available={metrics_row['probabilities_available']}",
            f"error_type={error_type}",
            f"error_message={error_message}",
        ]
    )
    (output_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    progress(f"Done. Results saved to {output_dir}")
    return output_dir


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_dataset_dir: Path,
    default_model: str,
    default_experiment_axis: str,
    default_scale_group: str,
    default_model_path: str | None = None,
) -> None:
    parser.add_argument("--dataset-dir", default=str(default_dataset_dir))
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--experiment-axis", default=default_experiment_axis)
    parser.add_argument("--scale-group", default=default_scale_group)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", default=default_model_path or "")
    parser.add_argument("--runner-name", default="default_runner")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--foundation-estimators", type=int, default=8)
    parser.add_argument("--tree-estimators", type=int, default=300)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--memory-sample-interval", type=float, default=DEFAULT_MEMORY_SAMPLE_INTERVAL)
    parser.add_argument(
        "--use-val-in-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TALENT validation rows together with train rows for fitting.",
    )


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.model_path == "":
        args.model_path = None
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one model on one selected TALENT dataset.")
    add_common_arguments(
        parser,
        default_dataset_dir=PROJECT_ROOT / "data" / "selected_talent",
        default_model="lightgbm",
        default_experiment_axis="unknown_axis",
        default_scale_group="unknown_group",
    )
    output_dir = run_experiment(normalize_args(parser.parse_args()))
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
