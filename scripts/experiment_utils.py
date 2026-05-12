from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from data_utils import encode_labels, preprocess_features
from model_utils import fit_classifier, make_classifier
from project_config import RANDOM_STATE


def current_memory_mb() -> float | None:
    try:
        import os

        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def empty_result(error: Exception, extra: dict[str, Any]) -> dict[str, Any]:
    result = {
        "success": False,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "fit_time_seconds": np.nan,
        "predict_time_seconds": np.nan,
        "seconds_per_test_sample": np.nan,
        "memory_before_mb": np.nan,
        "memory_after_mb": np.nan,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback_tail": traceback.format_exc(limit=3),
    }
    result.update(extra)
    return result


def evaluate_preprocessed_classifier(
    model_name: str,
    X_train_array: np.ndarray,
    X_test_array: np.ndarray,
    y_train_encoded: np.ndarray,
    y_test_encoded: np.ndarray,
    categorical_feature_indices: list[int] | None,
    seed: int = RANDOM_STATE,
    device: str | None = None,
    foundation_estimators: int = 8,
    tree_estimators: int = 300,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = extra or {}
    memory_before = current_memory_mb()
    try:
        model = make_classifier(
            model_name=model_name,
            seed=seed,
            device=device,
            foundation_estimators=foundation_estimators,
            tree_estimators=tree_estimators,
        )

        fit_start = time.perf_counter()
        fit_classifier(
            model,
            X_train_array,
            y_train_encoded,
            categorical_feature_indices=categorical_feature_indices,
        )
        fit_time = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        predictions = model.predict(X_test_array)
        predict_time = time.perf_counter() - predict_start

        memory_after = current_memory_mb()
        result = {
            "success": True,
            "accuracy": accuracy_score(y_test_encoded, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test_encoded, predictions),
            "macro_f1": f1_score(y_test_encoded, predictions, average="macro", zero_division=0),
            "fit_time_seconds": fit_time,
            "predict_time_seconds": predict_time,
            "seconds_per_test_sample": predict_time / max(len(y_test_encoded), 1),
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "error_type": "",
            "error_message": "",
            "traceback_tail": "",
        }
        result.update(extra)
        return result
    except Exception as error:
        return empty_result(error, extra)


def evaluate_classifier(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int = RANDOM_STATE,
    device: str | None = None,
    encoding: str = "ordinal",
    foundation_estimators: int = 8,
    tree_estimators: int = 300,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = extra or {}
    try:
        y_train_encoded, y_test_encoded, target_classes = encode_labels(y_train, y_test)
        X_train_array, X_test_array, feature_info = preprocess_features(
            X_train,
            X_test,
            encoding=encoding,
        )
        model_extra = {
            "target_classes": "|".join(target_classes),
            "n_classes": len(target_classes),
            **feature_info,
            **extra,
        }
        return evaluate_preprocessed_classifier(
            model_name=model_name,
            X_train_array=X_train_array,
            X_test_array=X_test_array,
            y_train_encoded=y_train_encoded,
            y_test_encoded=y_test_encoded,
            categorical_feature_indices=feature_info["categorical_feature_indices"],
            seed=seed,
            device=device,
            foundation_estimators=foundation_estimators,
            tree_estimators=tree_estimators,
            extra=model_extra,
        )
    except Exception as error:
        return empty_result(error, extra)


def append_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if output_path.exists():
        old_df = pd.read_csv(output_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(output_path, index=False)

