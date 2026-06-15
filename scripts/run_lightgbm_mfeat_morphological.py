from __future__ import annotations

import argparse

from run_talent_single_dataset import add_common_arguments, normalize_args, run_experiment
from run_tabicl_mfeat_morphological import DEFAULT_DATASET_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LightGBM on the mfeat-morphological TALENT dataset.")
    add_common_arguments(
        parser,
        default_dataset_dir=DEFAULT_DATASET_DIR,
        default_model="lightgbm",
        default_experiment_axis="feature_scale",
        default_scale_group="F1_6_20",
    )
    run_experiment(normalize_args(parser.parse_args()))


if __name__ == "__main__":
    main()
