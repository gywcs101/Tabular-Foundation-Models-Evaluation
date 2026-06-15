from __future__ import annotations

import argparse

from run_tabicl_mfeat_morphological import run_experiment
from project_config import PROJECT_ROOT, RANDOM_STATE

DEFAULT_PIXEL_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "selected_talent"
    / "feature_scale"
    / "F3_100_300"
    / "mfeat-pixel_2000rows_240feat_multiclass"
)
DEFAULT_TABPFN_MODEL_PATH = (
    r"E:\Software_Download\TabPFN_models\tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabPFN on the mfeat-pixel TALENT dataset.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_PIXEL_DATASET_DIR))
    parser.add_argument("--model", default="tabpfn")
    parser.add_argument("--experiment-axis", default="feature_scale")
    parser.add_argument("--scale-group", default="F3_100_300")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", default=DEFAULT_TABPFN_MODEL_PATH)
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
