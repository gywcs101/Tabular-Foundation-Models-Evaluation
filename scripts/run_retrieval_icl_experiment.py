from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

from data_utils import (
    encode_labels,
    load_cached_dataset,
    make_train_test_split,
    parse_name_list,
    preprocess_features,
    subsample_frame,
)
from experiment_utils import append_rows
from project_config import RANDOM_STATE, RESULTS_DIR


DEFAULT_DATASETS = ("credit-g", "adult", "bank-marketing")


def run_strategy(
    strategy: str,
    X_train_array: np.ndarray,
    X_test_array: np.ndarray,
    y_train_encoded: np.ndarray,
    y_test_encoded: np.ndarray,
    categorical_feature_indices: list[int],
    context_size: int,
    test_limit: int,
    seed: int,
    device: str | None,
    foundation_estimators: int,
) -> dict:
    rng = np.random.default_rng(seed)
    test_limit = min(test_limit, len(X_test_array))
    context_size = min(context_size, len(X_train_array))

    if strategy == "retrieval":
        nearest_neighbors = NearestNeighbors(n_neighbors=context_size)
        nearest_neighbors.fit(X_train_array)
        neighbor_indices = nearest_neighbors.kneighbors(
            X_test_array[:test_limit],
            return_distance=False,
        )
    else:
        neighbor_indices = [
            rng.choice(len(X_train_array), size=context_size, replace=False)
            for _ in range(test_limit)
        ]

    start = time.perf_counter()
    direct_predictions = []
    from model_utils import fit_classifier, make_classifier

    try:
        for test_idx in range(test_limit):
            context_indices = neighbor_indices[test_idx]
            model = make_classifier(
                model_name="tabpfn",
                seed=seed,
                device=device,
                foundation_estimators=foundation_estimators,
            )
            fit_classifier(
                model,
                X_train_array[context_indices],
                y_train_encoded[context_indices],
                categorical_feature_indices=categorical_feature_indices,
            )
            direct_predictions.append(model.predict(X_test_array[test_idx : test_idx + 1])[0])
    except Exception as error:
        return {
            "success": False,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "predict_time_seconds": np.nan,
            "seconds_per_test_sample": np.nan,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

    elapsed = time.perf_counter() - start

    direct_predictions = np.asarray(direct_predictions)
    y_eval = y_test_encoded[:test_limit]
    return {
        "success": True,
        "accuracy": accuracy_score(y_eval, direct_predictions),
        "balanced_accuracy": balanced_accuracy_score(y_eval, direct_predictions),
        "macro_f1": f1_score(y_eval, direct_predictions, average="macro", zero_division=0),
        "predict_time_seconds": elapsed,
        "seconds_per_test_sample": elapsed / max(test_limit, 1),
        "error_type": "",
        "error_message": "",
    }


def run_retrieval_experiment(args: argparse.Namespace) -> None:
    dataset_names = parse_name_list(args.datasets, DEFAULT_DATASETS)
    output_path = Path(args.output)

    for dataset_name in dataset_names:
        X, y = load_cached_dataset(dataset_name)
        X_sample, y_sample, actual_sample_size = subsample_frame(
            X,
            y,
            args.sample_size,
            seed=args.seed,
        )
        X_train, X_test, y_train, y_test = make_train_test_split(
            X_sample,
            y_sample,
            seed=args.seed,
        )
        y_train_encoded, y_test_encoded, target_classes = encode_labels(y_train, y_test)
        X_train_array, X_test_array, feature_info = preprocess_features(
            X_train,
            X_test,
            encoding=args.encoding,
        )

        for strategy in ("random", "retrieval"):
            row = run_strategy(
                strategy=strategy,
                X_train_array=X_train_array,
                X_test_array=X_test_array,
                y_train_encoded=y_train_encoded,
                y_test_encoded=y_test_encoded,
                categorical_feature_indices=feature_info["categorical_feature_indices"],
                context_size=args.context_size,
                test_limit=args.test_limit,
                seed=args.seed,
                device=args.device,
                foundation_estimators=args.foundation_estimators,
            )
            row.update(
                {
                    "dataset": dataset_name,
                    "model": "tabpfn",
                    "context_strategy": strategy,
                    "actual_sample_size": actual_sample_size,
                    "train_rows": len(X_train),
                    "test_rows": min(args.test_limit, len(X_test)),
                    "context_size": min(args.context_size, len(X_train)),
                    "target_classes": "|".join(target_classes),
                    **feature_info,
                }
            )
            append_rows(output_path, [row])
            print(
                f"{dataset_name} | {strategy} | success={row.get('success')} | "
                f"acc={row.get('accuracy', '')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval-based in-context learning experiment with TabPFN."
    )
    parser.add_argument("--datasets", default="credit-g,adult,bank-marketing")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--output", default=str(RESULTS_DIR / "retrieval_icl.csv"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encoding", choices=["ordinal", "onehot"], default="ordinal")
    parser.add_argument("--foundation-estimators", type=int, default=4)
    args = parser.parse_args()
    run_retrieval_experiment(args)


if __name__ == "__main__":
    main()
