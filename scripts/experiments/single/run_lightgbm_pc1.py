from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT_FOR_IMPORTS / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS / "scripts"))


from project_config import PROJECT_ROOT
from core.run_talent_single_dataset import add_common_arguments, normalize_args, run_experiment


DEFAULT_PC1_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "selected_talent"
    / "sample_scale"
    / "A_1000_3000"
    / "pc1_1109rows_21feat_binclass"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LightGBM on the pc1 TALENT dataset.")
    add_common_arguments(
        parser,
        default_dataset_dir=DEFAULT_PC1_DATASET_DIR,
        default_model="lightgbm",
        default_experiment_axis="sample_scale",
        default_scale_group="A_1000_3000",
    )
    run_experiment(normalize_args(parser.parse_args()))


if __name__ == "__main__":
    main()
