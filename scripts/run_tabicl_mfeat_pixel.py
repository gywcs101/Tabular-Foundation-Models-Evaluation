from __future__ import annotations

import argparse

from project_config import PROJECT_ROOT, TABICL_MODEL_PATH
from run_talent_single_dataset import add_common_arguments, normalize_args, run_experiment


DEFAULT_PIXEL_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "selected_talent"
    / "feature_scale"
    / "F3_100_300"
    / "mfeat-pixel_2000rows_240feat_multiclass"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabICL on the mfeat-pixel TALENT dataset.")
    add_common_arguments(
        parser,
        default_dataset_dir=DEFAULT_PIXEL_DATASET_DIR,
        default_model="tabicl",
        default_experiment_axis="feature_scale",
        default_scale_group="F3_100_300",
        default_model_path=str(TABICL_MODEL_PATH),
    )
    run_experiment(normalize_args(parser.parse_args()))


if __name__ == "__main__":
    main()
