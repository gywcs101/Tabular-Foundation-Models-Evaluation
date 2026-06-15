from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT_FOR_IMPORTS / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS / "scripts"))


from project_config import PROJECT_ROOT, TABPFN_MODEL_PATH
from core.run_talent_single_dataset import add_common_arguments, normalize_args, run_experiment


DEFAULT_PIXEL_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "selected_talent"
    / "feature_scale"
    / "F3_100_300"
    / "mfeat-pixel_2000rows_240feat_multiclass"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabPFN on the mfeat-pixel TALENT dataset.")
    add_common_arguments(
        parser,
        default_dataset_dir=DEFAULT_PIXEL_DATASET_DIR,
        default_model="tabpfn",
        default_experiment_axis="feature_scale",
        default_scale_group="F3_100_300",
        default_model_path=str(TABPFN_MODEL_PATH),
    )
    run_experiment(normalize_args(parser.parse_args()))


if __name__ == "__main__":
    main()
