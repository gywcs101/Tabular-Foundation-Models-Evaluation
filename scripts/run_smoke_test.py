from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris

from data_utils import make_train_test_split
from experiment_utils import append_rows, evaluate_classifier
from project_config import RANDOM_STATE, RESULTS_DIR


def load_builtin_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    if name == "breast_cancer":
        bunch = load_breast_cancer(as_frame=True)
    elif name == "iris":
        bunch = load_iris(as_frame=True)
    else:
        raise ValueError(f"Unknown smoke dataset: {name}")

    return bunch.data, pd.Series(bunch.target, name="target")


def main() -> None:
    output_path = Path(RESULTS_DIR / "smoke_test.csv")
    rows = []
    for dataset_name in ("breast_cancer", "iris"):
        X, y = load_builtin_dataset(dataset_name)
        X_train, X_test, y_train, y_test = make_train_test_split(
            X,
            y,
            seed=RANDOM_STATE,
        )

        for model_name in ("random_forest", "logistic_regression"):
            row = evaluate_classifier(
                model_name=model_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                seed=RANDOM_STATE,
                extra={
                    "dataset": dataset_name,
                    "model": model_name,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "purpose": "offline_smoke_test",
                },
            )
            rows.append(row)
            print(
                f"{dataset_name} | {model_name} | success={row.get('success')} | "
                f"acc={row.get('accuracy', '')}"
            )

    append_rows(output_path, rows)
    print(f"Smoke test results saved to {output_path}")


if __name__ == "__main__":
    main()

