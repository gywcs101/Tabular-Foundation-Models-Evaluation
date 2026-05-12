from __future__ import annotations

import argparse

from data_utils import parse_name_list, save_cached_dataset
from project_config import DATASETS, DATA_DIR, ensure_project_dirs


def fetch_dataset(dataset_name: str) -> None:
    from sklearn.datasets import fetch_openml

    spec = DATASETS[dataset_name]
    fetch_kwargs = {
        "name": spec["openml_name"],
        "as_frame": True,
        "data_home": str(DATA_DIR / "sklearn_openml"),
    }
    if spec.get("version") is not None:
        fetch_kwargs["version"] = spec["version"]

    bunch = fetch_openml(**fetch_kwargs)
    X = bunch.data
    y = bunch.target
    save_cached_dataset(dataset_name, X, y)
    print(f"Saved {dataset_name}: {len(X)} rows, {X.shape[1]} features")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache OpenML datasets.")
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset names, or 'all'.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    dataset_names = parse_name_list(args.datasets, DATASETS.keys())
    for dataset_name in dataset_names:
        fetch_dataset(dataset_name)


if __name__ == "__main__":
    main()
