from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from project_config import LOCAL_TABARENA_ROOT, LOCAL_TALENT_ROOT, TABICL_PYTHON, TABPFN_PYTHON

DEFAULT_TALENT_ROOT = LOCAL_TALENT_ROOT
DEFAULT_TABARENA_ROOT = LOCAL_TABARENA_ROOT
DEFAULT_TABPFN_PYTHON = TABPFN_PYTHON
DEFAULT_TABICL_PYTHON = TABICL_PYTHON


def status(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def check_python_env(name: str, python_path: Path, modules: list[str]) -> bool:
    print(f"\n[{name}]")
    if not python_path.exists():
        print(f"FAIL python executable not found: {python_path}")
        return False

    code = (
        "import sys, importlib.util\n"
        "print('python=' + sys.executable)\n"
        "print('version=' + sys.version.split()[0])\n"
        f"mods={modules!r}\n"
        "for m in mods:\n"
        "    print(m + '=' + ('OK' if importlib.util.find_spec(m) else 'MISSING'))\n"
    )
    completed = subprocess.run(
        [str(python_path), "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())
    return completed.returncode == 0


def list_talent_datasets(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted([path for path in data_dir.iterdir() if path.is_dir()])


def check_talent_dataset(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "info.json"
    result = {
        "dataset": dataset_dir.name,
        "has_info": info_path.exists(),
        "task_type": "",
        "n_num_features": "",
        "n_cat_features": "",
        "train_rows": "",
        "val_rows": "",
        "test_rows": "",
        "ok": False,
        "error": "",
    }
    try:
        if not info_path.exists():
            raise FileNotFoundError("missing info.json")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        result["task_type"] = info.get("task_type", "")
        result["n_num_features"] = info.get("n_num_features", "")
        result["n_cat_features"] = info.get("n_cat_features", "")

        y_train = np.load(dataset_dir / "y_train.npy", allow_pickle=True)
        y_val = np.load(dataset_dir / "y_val.npy", allow_pickle=True)
        y_test = np.load(dataset_dir / "y_test.npy", allow_pickle=True)
        result["train_rows"] = len(y_train)
        result["val_rows"] = len(y_val)
        result["test_rows"] = len(y_test)

        n_train = dataset_dir / "N_train.npy"
        c_train = dataset_dir / "C_train.npy"
        if n_train.exists():
            n_array = np.load(n_train, allow_pickle=True)
            if len(n_array) != len(y_train):
                raise ValueError("N_train row count differs from y_train")
        if c_train.exists():
            c_array = np.load(c_train, allow_pickle=True)
            if len(c_array) != len(y_train):
                raise ValueError("C_train row count differs from y_train")
        result["ok"] = True
    except Exception as error:
        result["error"] = str(error)
    return result


def check_talent(root: Path, max_datasets: int) -> bool:
    print("\n[TALENT dataset]")
    print(f"root={root}")
    data_dir = root / "benchmark_dataset" / "data"
    datasets = list_talent_datasets(data_dir)
    print(f"data_dir={data_dir}")
    print(f"dataset_count={len(datasets)}")
    if not datasets:
        print("FAIL no dataset folders found")
        return False

    rows = [check_talent_dataset(path) for path in datasets[:max_datasets]]
    for row in rows:
        print(
            f"{status(bool(row['ok']))} {row['dataset']} "
            f"task={row['task_type']} train={row['train_rows']} "
            f"val={row['val_rows']} test={row['test_rows']} error={row['error']}"
        )
    return all(bool(row["ok"]) for row in rows)


def check_tabarena(root: Path) -> bool:
    print("\n[TabArena repository]")
    print(f"root={root}")
    required = ["README.md", "pyproject.toml", "tabarena", "examples"]
    ok = True
    for name in required:
        path = root / name
        exists = path.exists()
        print(f"{status(exists)} {name}")
        ok = ok and exists

    metadata_candidates = [
        root / "tabarena" / "tabarena" / "nips2025_utils" / "metadata" / "curated_tabarena_dataset_metadata.csv",
        root / "tabarena" / "tabarena" / "nips2025_utils" / "metadata" / "task_metadata_tabarena51.csv",
    ]
    for path in metadata_candidates:
        exists = path.exists()
        print(f"{status(exists)} metadata {path.name}")
        if exists:
            df = pd.read_csv(path)
            print(f"rows={len(df)} columns={list(df.columns)[:8]}")
        ok = ok and exists
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local model environments and downloaded tabular datasets.")
    parser.add_argument("--talent-root", type=Path, default=DEFAULT_TALENT_ROOT)
    parser.add_argument("--tabarena-root", type=Path, default=DEFAULT_TABARENA_ROOT)
    parser.add_argument("--tabpfn-python", type=Path, default=DEFAULT_TABPFN_PYTHON)
    parser.add_argument("--tabicl-python", type=Path, default=DEFAULT_TABICL_PYTHON)
    parser.add_argument("--max-talent-datasets", type=int, default=5)
    args = parser.parse_args()

    checks = [
        check_python_env("TabPFN env", args.tabpfn_python, ["tabpfn", "torch", "sklearn", "pandas", "numpy"]),
        check_python_env("TabICL env", args.tabicl_python, ["tabicl", "torch", "sklearn", "numpy"]),
        check_talent(args.talent_root, args.max_talent_datasets),
        check_tabarena(args.tabarena_root),
    ]

    print(f"\nOVERALL={status(all(checks))}")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
