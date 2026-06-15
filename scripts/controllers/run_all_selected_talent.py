from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import (  # noqa: E402
    LIGHTGBM_PYTHON,
    SELECTED_TALENT_DIR,
    TABICL_MODEL_PATH,
    TABICL_PYTHON,
    TABPFN_MODEL_PATH,
    TABPFN_PYTHON,
    XGBOOST_PYTHON,
)


MODEL_CONFIGS = {
    "tabicl": {
        "python": TABICL_PYTHON,
        "model_path": TABICL_MODEL_PATH,
    },
    "tabpfn": {
        "python": TABPFN_PYTHON,
        "model_path": TABPFN_MODEL_PATH,
    },
    "lightgbm": {
        "python": LIGHTGBM_PYTHON,
        "model_path": None,
    },
    "xgboost": {
        "python": XGBOOST_PYTHON,
        "model_path": None,
    },
}


def progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def discover_datasets(root: Path) -> list[tuple[Path, str, str]]:
    datasets: list[tuple[Path, str, str]] = []
    for info_path in sorted(root.rglob("info.json")):
        dataset_dir = info_path.parent
        relative = dataset_dir.relative_to(root)
        if len(relative.parts) < 3:
            continue
        experiment_axis = relative.parts[0]
        scale_group = relative.parts[1]
        datasets.append((dataset_dir, experiment_axis, scale_group))
    return datasets


def parse_model_list(value: str) -> list[str]:
    if value == "all":
        return list(MODEL_CONFIGS)
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [model for model in models if model not in MODEL_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    return models


def run_one(
    *,
    runner_name: str,
    model: str,
    dataset_dir: Path,
    experiment_axis: str,
    scale_group: str,
    use_val_in_train: bool,
    dry_run: bool,
) -> int:
    config = MODEL_CONFIGS[model]
    python_path: Path = config["python"]
    model_path: Path | None = config["model_path"]

    command = [
        str(python_path),
        str(SCRIPTS_DIR / "core" / "run_talent_single_dataset.py"),
        "--runner-name",
        runner_name,
        "--model",
        model,
        "--dataset-dir",
        str(dataset_dir),
        "--experiment-axis",
        experiment_axis,
        "--scale-group",
        scale_group,
    ]
    if model_path is not None:
        command.extend(["--model-path", str(model_path)])
    if not use_val_in_train:
        command.append("--no-use-val-in-train")

    progress(f"Run {runner_name} | {model} | {experiment_axis}/{scale_group}/{dataset_dir.name}")
    if dry_run:
        print(" ".join(f'"{part}"' if " " in part else part for part in command))
        return 0

    if not python_path.exists():
        print(f"ERROR python not found for {model}: {python_path}", flush=True)
        return 1
    if model_path is not None and not model_path.exists():
        print(f"ERROR model checkpoint not found for {model}: {model_path}", flush=True)
        return 1

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all selected TALENT datasets with all configured models.")
    parser.add_argument("--runner-name", required=True, help="Name of the person running this experiment batch.")
    parser.add_argument("--models", default="all", help="Comma-separated models or 'all'.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them.")
    parser.add_argument(
        "--use-val-in-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TALENT validation rows together with train rows for fitting.",
    )
    args = parser.parse_args()

    datasets = discover_datasets(SELECTED_TALENT_DIR)
    models = parse_model_list(args.models)
    if not datasets:
        print(f"ERROR no selected TALENT datasets found under {SELECTED_TALENT_DIR}", flush=True)
        return 1

    progress(f"Runner: {args.runner_name}")
    progress(f"Datasets: {len(datasets)}")
    progress(f"Models: {', '.join(models)}")

    failures: list[tuple[str, Path, int]] = []
    total = len(datasets) * len(models)
    current = 0
    for dataset_dir, experiment_axis, scale_group in datasets:
        for model in models:
            current += 1
            progress(f"Progress {current}/{total}")
            return_code = run_one(
                runner_name=args.runner_name,
                model=model,
                dataset_dir=dataset_dir,
                experiment_axis=experiment_axis,
                scale_group=scale_group,
                use_val_in_train=args.use_val_in_train,
                dry_run=args.dry_run,
            )
            if return_code != 0:
                failures.append((model, dataset_dir, return_code))
                progress(f"FAILED return_code={return_code}")

    progress(f"Finished. failures={len(failures)}")
    if failures:
        print("\nFailed runs:")
        for model, dataset_dir, return_code in failures:
            print(f"- {model} | {dataset_dir.name} | return_code={return_code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
