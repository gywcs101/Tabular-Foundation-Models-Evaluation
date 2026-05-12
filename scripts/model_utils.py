from __future__ import annotations

import inspect
from typing import Any


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {}
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def make_classifier(
    model_name: str,
    seed: int,
    device: str | None = None,
    foundation_estimators: int = 8,
    tree_estimators: int = 300,
):
    if model_name == "tabpfn":
        from tabpfn import TabPFNClassifier

        kwargs = _filter_kwargs(
            TabPFNClassifier,
            {
                "device": device or "cpu",
                "seed": seed,
                "random_state": seed,
                "n_estimators": foundation_estimators,
                "show_progress": False,
            },
        )
        return TabPFNClassifier(**kwargs)

    if model_name == "tabicl":
        from tabicl import TabICLClassifier

        kwargs = _filter_kwargs(
            TabICLClassifier,
            {
                "device": device,
                "random_state": seed,
                "n_estimators": foundation_estimators,
                "kv_cache": False,
                "verbose": False,
                "allow_auto_download": True,
                "n_jobs": None,
            },
        )
        return TabICLClassifier(**kwargs)

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=tree_estimators,
            learning_rate=0.05,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=tree_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=tree_estimators,
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=1000,
            random_state=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


def fit_classifier(model: Any, X, y, categorical_feature_indices: list[int] | None = None) -> Any:
    categorical_feature_indices = categorical_feature_indices or []

    if categorical_feature_indices and hasattr(model, "set_categorical_features"):
        model.set_categorical_features(categorical_feature_indices)

    fit_kwargs = {}
    try:
        fit_signature = inspect.signature(model.fit)
        if "categorical_feature_indices" in fit_signature.parameters:
            fit_kwargs["categorical_feature_indices"] = categorical_feature_indices
        elif "categorical_features" in fit_signature.parameters:
            fit_kwargs["categorical_features"] = categorical_feature_indices
    except (TypeError, ValueError):
        fit_kwargs = {}

    return model.fit(X, y, **fit_kwargs)
