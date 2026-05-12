from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from project_config import DATASETS, DATA_DIR, RANDOM_STATE, TARGET_COLUMN


COMMON_MISSING_MARKERS = ["?", "", "NA", "N/A", "na", "n/a", "None", "none", "null", "NULL"]


def parse_name_list(raw: str, allowed: Iterable[str]) -> list[str]:
    allowed_list = list(allowed)
    if raw.lower() == "all":
        return allowed_list
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(allowed_list))
    if unknown:
        raise ValueError(f"Unknown value(s): {unknown}. Allowed: {allowed_list}")
    return requested


def parse_sample_sizes(raw: str) -> list[int | str]:
    values: list[int | str] = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        if item.lower() == "full":
            values.append("full")
            continue
        size = int(item)
        if size <= 0:
            raise ValueError("Sample sizes must be positive integers or 'full'.")
        values.append(size)
    return values


def dataset_csv_path(dataset_name: str) -> Path:
    return DATA_DIR / f"{dataset_name}.csv"


def dataset_meta_path(dataset_name: str) -> Path:
    return DATA_DIR / f"{dataset_name}.meta.json"


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(COMMON_MISSING_MARKERS, np.nan)


def load_cached_dataset(dataset_name: str) -> tuple[pd.DataFrame, pd.Series]:
    path = dataset_csv_path(dataset_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached dataset: {path}. Run scripts/prepare_data.py first."
        )
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Cached dataset {path} does not contain {TARGET_COLUMN}.")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def save_cached_dataset(dataset_name: str, X: pd.DataFrame, y: pd.Series) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = X.copy()
    df[TARGET_COLUMN] = y.to_numpy()
    df = normalize_missing_values(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    df.to_csv(dataset_csv_path(dataset_name), index=False)

    metadata = {
        "dataset_name": dataset_name,
        "openml_name": DATASETS[dataset_name]["openml_name"],
        "rows": int(df.shape[0]),
        "features": int(df.shape[1] - 1),
        "target_column": TARGET_COLUMN,
        "target_classes": sorted([str(value) for value in df[TARGET_COLUMN].dropna().unique()]),
    }
    dataset_meta_path(dataset_name).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def valid_stratify_labels(y: pd.Series | np.ndarray) -> pd.Series | None:
    labels = pd.Series(y)
    counts = labels.value_counts(dropna=False)
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def subsample_frame(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int | str,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series, int | str]:
    if sample_size == "full" or int(sample_size) >= len(X):
        return X.reset_index(drop=True), y.reset_index(drop=True), "full"

    df = X.copy()
    df[TARGET_COLUMN] = y.to_numpy()
    stratify = valid_stratify_labels(df[TARGET_COLUMN])
    sampled, _ = train_test_split(
        df,
        train_size=int(sample_size),
        random_state=seed,
        stratify=stratify,
    )
    sampled = sampled.reset_index(drop=True)
    return sampled.drop(columns=[TARGET_COLUMN]), sampled[TARGET_COLUMN], int(sample_size)


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify = valid_stratify_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def detect_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_cols = list(
        X.select_dtypes(include=["object", "category", "bool"]).columns
    )
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    return numeric_cols, categorical_cols


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    encoding: str = "ordinal",
) -> tuple[np.ndarray, np.ndarray, dict]:
    if encoding not in {"ordinal", "onehot"}:
        raise ValueError("encoding must be 'ordinal' or 'onehot'.")

    X_train = X_train.copy()
    X_test = X_test.copy()
    numeric_cols, categorical_cols = detect_feature_types(X_train)

    for col in categorical_cols:
        X_train[col] = X_train[col].astype("object")
        X_test[col] = X_test[col].astype("object")

    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        if encoding == "ordinal":
            categorical_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        else:
            categorical_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__missing__",
                        keep_empty_features=True,
                    ),
                ),
                ("encoder", categorical_encoder),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_train_array = preprocessor.fit_transform(X_train)
    X_test_array = preprocessor.transform(X_test)

    if hasattr(X_train_array, "toarray"):
        X_train_array = X_train_array.toarray()
    if hasattr(X_test_array, "toarray"):
        X_test_array = X_test_array.toarray()

    if encoding == "ordinal":
        categorical_feature_indices = list(
            range(len(numeric_cols), len(numeric_cols) + len(categorical_cols))
        )
    else:
        categorical_feature_indices = []

    info = {
        "encoding": encoding,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "n_numeric_features": len(numeric_cols),
        "n_categorical_features": len(categorical_cols),
        "n_features_after_preprocessing": int(X_train_array.shape[1]),
        "categorical_feature_indices": categorical_feature_indices,
    }
    return np.asarray(X_train_array), np.asarray(X_test_array), info


def encode_labels(y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str]]:
    encoder = LabelEncoder()
    combined = pd.concat([y_train, y_test], axis=0).astype(str)
    encoder.fit(combined)
    return (
        encoder.transform(y_train.astype(str)),
        encoder.transform(y_test.astype(str)),
        [str(label) for label in encoder.classes_],
    )


def inject_missing_values(
    X: pd.DataFrame,
    rate: float,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("Missing rate must be between 0 and 1.")
    if rate == 0.0:
        return X.copy()
    rng = np.random.default_rng(seed)
    X_missing = X.copy()
    mask = rng.random(X_missing.shape) < rate
    return X_missing.mask(mask)

