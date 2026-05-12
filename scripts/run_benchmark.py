from __future__ import annotations

import argparse
from pathlib import Path

from data_utils import (
    load_cached_dataset,
    make_train_test_split,
    parse_name_list,
    parse_sample_sizes,
    subsample_frame,
)
from experiment_utils import append_rows, evaluate_classifier
from project_config import DATASETS, DEFAULT_SAMPLE_SIZES, MODEL_NAMES, RANDOM_STATE, RESULTS_DIR


def run_benchmark(args: argparse.Namespace) -> None:
    dataset_names = parse_name_list(args.datasets, DATASETS.keys())
    model_names = parse_name_list(args.models, MODEL_NAMES)
    sample_sizes = parse_sample_sizes(args.sample_sizes)
    output_path = Path(args.output)

    for dataset_name in dataset_names:
        X, y = load_cached_dataset(dataset_name)
        for requested_sample_size in sample_sizes:
            X_sample, y_sample, actual_sample_size = subsample_frame(
                X,
                y,
                requested_sample_size,
                seed=args.seed,
            )
            X_train, X_test, y_train, y_test = make_train_test_split(
                X_sample,
                y_sample,
                seed=args.seed,
            )

            for model_name in model_names:
                if model_name == "tabpfn" and len(X_train) > args.tabpfn_max_train_rows:
                    row = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "requested_sample_size": requested_sample_size,
                        "actual_sample_size": actual_sample_size,
                        "train_rows": len(X_train),
                        "test_rows": len(X_test),
                        "success": False,
                        "error_type": "SkippedByLimit",
                        "error_message": (
                            f"TabPFN train rows {len(X_train)} exceed "
                            f"--tabpfn-max-train-rows={args.tabpfn_max_train_rows}"
                        ),
                    }
                else:
                    row = evaluate_classifier(
                        model_name=model_name,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        seed=args.seed,
                        device=args.device,
                        encoding=args.encoding,
                        foundation_estimators=args.foundation_estimators,
                        tree_estimators=args.tree_estimators,
                        extra={
                            "dataset": dataset_name,
                            "model": model_name,
                            "requested_sample_size": requested_sample_size,
                            "actual_sample_size": actual_sample_size,
                            "train_rows": len(X_train),
                            "test_rows": len(X_test),
                        },
                    )
                append_rows(output_path, [row])
                print(
                    f"{dataset_name} | {requested_sample_size} | {model_name} | "
                    f"success={row.get('success')} | acc={row.get('accuracy', '')}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tabular model benchmark.")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--models", default="all")
    parser.add_argument(
        "--sample-sizes",
        default=",".join(str(value) for value in DEFAULT_SAMPLE_SIZES),
        help="Comma-separated sample sizes, e.g. 500,1000,10000,full.",
    )
    parser.add_argument("--output", default=str(RESULTS_DIR / "metrics.csv"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--device", default=None, help="cpu, cuda, or None for model default.")
    parser.add_argument("--encoding", choices=["ordinal", "onehot"], default="ordinal")
    parser.add_argument("--foundation-estimators", type=int, default=8)
    parser.add_argument("--tree-estimators", type=int, default=300)
    parser.add_argument("--tabpfn-max-train-rows", type=int, default=10000)
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
