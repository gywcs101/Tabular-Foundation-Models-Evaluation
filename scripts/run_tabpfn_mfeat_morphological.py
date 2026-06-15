from __future__ import annotations

import argparse

from project_config import PROJECT_ROOT, TABPFN_MODEL_PATH
from run_talent_single_dataset import add_common_arguments, normalize_args, run_experiment


DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "selected_talent"
    / "feature_scale"
    / "F1_6_20"
    / "mfeat-morphological_2000rows_6feat_multiclass"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabPFN on the mfeat-morphological TALENT dataset.")
    add_common_arguments(
        parser,
        default_dataset_dir=DEFAULT_DATASET_DIR,
        default_model="tabpfn",
        default_experiment_axis="feature_scale",
        default_scale_group="F1_6_20",
        default_model_path=str(TABPFN_MODEL_PATH),
    )
    run_experiment(normalize_args(parser.parse_args()))


if __name__ == "__main__":
    main()
