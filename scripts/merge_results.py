from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from project_config import RESULTS_DIR


def merge_result_files(input_dir: Path, output_path: Path) -> None:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames = []
    for file_path in files:
        df = pd.read_csv(file_path)
        df["source_file"] = file_path.name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged {len(files)} files into {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge team result CSV files.")
    parser.add_argument("--input-dir", default=str(RESULTS_DIR / "raw"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "final" / "final_metrics.csv"))
    args = parser.parse_args()
    merge_result_files(Path(args.input_dir), Path(args.output))


if __name__ == "__main__":
    main()

