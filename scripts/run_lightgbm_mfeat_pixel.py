from __future__ import annotations

import argparse

from run_tabicl_mfeat_pixel import DEFAULT_PIXEL_DATASET_DIR
from run_talent_single_dataset import add_common_arguments, normalize_args, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LightGBM on the mfeat-pixel TALENT dataset.")
    add_common_arguments(
        parser,
        default_dataset_dir=DEFAULT_PIXEL_DATASET_DIR,
        default_model="lightgbm",
        default_experiment_axis="feature_scale",
        default_scale_group="F3_100_300",
    )
    run_experiment(normalize_args(parser.parse_args()))


if __name__ == "__main__":
    main()
