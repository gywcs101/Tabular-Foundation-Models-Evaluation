from __future__ import annotations

import csv
from pathlib import Path

from analysis_config import MODELS, REQUIRED_RUN_FILES, RESULTS_FINAL_DIR, RESULTS_RAW_DIR
from analysis_utils import discover_runner_dirs, discover_selected_datasets, write_dataframe

import pandas as pd


def is_successful_metrics(metrics_path: Path) -> bool:
    try:
        with metrics_path.open(newline="", encoding="utf-8") as file:
            row = next(csv.DictReader(file), None)
        return bool(row) and str(row.get("success", "")).lower() == "true"
    except Exception:
        return False


def build_report() -> pd.DataFrame:
    datasets = discover_selected_datasets()
    runner_dirs = discover_runner_dirs()
    rows: list[dict[str, object]] = []

    for runner_dir in runner_dirs:
        for dataset in datasets:
            for model in MODELS:
                result_root = (
                    RESULTS_RAW_DIR
                    / runner_dir.name
                    / model
                    / dataset["experiment_axis"]
                    / dataset["scale_group"]
                    / dataset["dataset"]
                )
                runs = sorted(result_root.glob("run_*")) if result_root.exists() else []
                success_runs = [run for run in runs if is_successful_metrics(run / "metrics.csv")]
                latest_success = success_runs[-1] if success_runs else None
                missing_files: list[str] = []
                if latest_success:
                    missing_files = [
                        file_name
                        for file_name in REQUIRED_RUN_FILES
                        if not (latest_success / file_name).exists()
                    ]
                status = "complete"
                if not runs:
                    status = "missing"
                elif not success_runs:
                    status = "no_success"
                elif len(success_runs) > 1:
                    status = "duplicate_success"
                if missing_files:
                    status = "incomplete_files"
                rows.append(
                    {
                        "runner_dir": runner_dir.name,
                        "model": model,
                        "experiment_axis": dataset["experiment_axis"],
                        "scale_group": dataset["scale_group"],
                        "dataset": dataset["dataset"],
                        "run_count": len(runs),
                        "success_count": len(success_runs),
                        "latest_success_run": latest_success.name if latest_success else "",
                        "missing_files": ";".join(missing_files),
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    report = build_report()
    RESULTS_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_FINAL_DIR / "input_completeness_report.csv"
    write_dataframe(report, output_path)
    if report.empty:
        print("No result directories found.")
        return 1
    summary = report.groupby(["runner_dir", "status"]).size().unstack(fill_value=0)
    print(summary)
    print(f"Saved completeness report to {output_path}")
    blocking_statuses = {"missing", "no_success", "incomplete_files"}
    has_blocking_issue = report["status"].isin(blocking_statuses).any()
    return 1 if has_blocking_issue else 0


if __name__ == "__main__":
    raise SystemExit(main())
