from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def should_print_all(total_rows: int, max_rows: int) -> bool:
    if total_rows <= max_rows:
        return True

    print(f"This file has {total_rows} rows, which is more than {max_rows}.")
    try:
        answer = input("Print the complete array? Enter y or n: ").strip().lower()
    except EOFError:
        print("No input received. Defaulting to n.")
        return False
    return answer == "y"


def inspect_npy(path: Path, max_rows: int, precision: int) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    array = np.load(path, allow_pickle=True)

    print(f"file: {path}")
    print(f"shape: {array.shape}")
    print(f"dtype: {array.dtype}")
    print(f"ndim: {array.ndim}")
    print()

    if array.ndim == 0:
        np.set_printoptions(precision=precision, suppress=False, threshold=np.inf)
        print("content:")
        print(array)
        return

    total_rows = len(array)
    print_all = should_print_all(total_rows, max_rows)
    display_array = array if print_all else array[:max_rows]

    np.set_printoptions(
        precision=precision,
        suppress=False,
        threshold=np.inf,
        linewidth=200,
    )
    if print_all:
        print(f"all {total_rows} row(s):")
    else:
        print(f"first {max_rows} row(s):")
    print(display_array)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a .npy file from the terminal.")
    parser.add_argument("path", type=Path, help="Path to the .npy file.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Print all rows up to this limit. Ask before printing more rows.",
    )
    parser.add_argument("--precision", type=int, default=4, help="Number precision for printing.")
    args = parser.parse_args()

    inspect_npy(args.path, args.max_rows, args.precision)


if __name__ == "__main__":
    main()
