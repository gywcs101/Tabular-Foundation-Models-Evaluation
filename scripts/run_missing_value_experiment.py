from __future__ import annotations

import argparse
from pathlib import Path

from data_utils import (
    inject_missing_values,
    load_cached_dataset,
    make_train_test_split,
    parse_name_list,
    subsample_frame,
)
from experiment_utils import append_rows, evaluate_classifier
from project_config import MISSING_RATES, MODEL_NAMES, RANDOM_STATE, RESULTS_DIR


DEFAULT_DATASETS = ("adult", "bank-marketing")


def parse_rates(raw: str) -> list[float]:
    rates = [float(item.strip()) for item in raw.split(",") if item.strip()]
    for rate in rates:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("Missing rates must be between 0 and 1.")
    return rates


def run_missing_experiment(args: argparse.Namespace) -> None:
    dataset_names = parse_name_list(args.datasets, DEFAULT_DATASETS)
    model_names = parse_name_list(args.models, MODEL_NAMES)
    missing_rates = parse_rates(args.missing_rates)
    output_path = Path(args.output)

    for dataset_name in dataset_names:
        X, y = load_cached_dataset(dataset_name)
        X_sample, y_sample, actual_sample_size = subsample_frame(
            X,
            y,
            args.sample_size,
            seed=args.seed,
        )
        X_train_base, X_test_base, y_train, y_test = make_train_test_split(
            X_sample,
            y_sample,
            seed=args.seed,
        )

        for missing_rate in missing_rates:
            X_train = X_train_base
            X_test = X_test_base
            if args.apply_to in {"train", "both"}:
                X_train = inject_missing_values(X_train, missing_rate, seed=args.seed)
            if args.apply_to in {"test", "both"}:
                X_test = inject_missing_values(X_test, missing_rate, seed=args.seed + 1)

            for model_name in model_names:
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
                        "actual_sample_size": actual_sample_size,
                        "train_rows": len(X_train),
                        "test_rows": len(X_test),
                        "missing_rate": missing_rate,
                        "missing_apply_to": args.apply_to,
                    },
                )
                append_rows(output_path, [row])
                print(
                    f"{dataset_name} | missing={missing_rate} | {model_name} | "
                    f"success={row.get('success')} | acc={row.get('accuracy', '')}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run missing value robustness experiment.")
    parser.add_argument("--datasets", default="adult,bank-marketing")
    parser.add_argument("--models", default="all")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument(
        "--missing-rates",
        default=",".join(str(value) for value in MISSING_RATES),
    )
    parser.add_argument("--apply-to", choices=["train", "test", "both"], default="both")
    parser.add_argument("--output", default=str(RESULTS_DIR / "missing_values.csv"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encoding", choices=["ordinal", "onehot"], default="ordinal")
    parser.add_argument("--foundation-estimators", type=int, default=8)
    parser.add_argument("--tree-estimators", type=int, default=300)
    args = parser.parse_args()
    run_missing_experiment(args)


if __name__ == "__main__":
    main()

