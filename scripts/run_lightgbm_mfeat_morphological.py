from __future__ import annotations

import argparse

from run_tabicl_mfeat_morphological import DEFAULT_DATASET_DIR, run_experiment
from project_config import RANDOM_STATE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LightGBM on the mfeat-morphological TALENT dataset.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--experiment-axis", default="feature_scale")
    parser.add_argument("--scale-group", default="F1_6_20")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", default="")
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
    run_experiment(args)


if __name__ == "__main__":
    main()
