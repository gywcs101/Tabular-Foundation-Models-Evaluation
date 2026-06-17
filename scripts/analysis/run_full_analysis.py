from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STEPS = [
    "check_analysis_inputs.py",
    "build_final_tables.py",
    "plot_performance.py",
    "plot_scalability.py",
    "plot_memory.py",
    "plot_confidence.py",
    "plot_confusion_matrices.py",
]


def run_step(script_name: str) -> int:
    print(f"\n=== Running {script_name} ===", flush=True)
    completed = subprocess.run([sys.executable, str(SCRIPT_DIR / script_name)], check=False)
    if completed.returncode != 0:
        print(f"Step failed: {script_name} (exit code {completed.returncode})", flush=True)
    return completed.returncode


def main() -> int:
    failures = 0
    for script in STEPS:
        return_code = run_step(script)
        if return_code != 0:
            failures += 1
            if script in {"build_final_tables.py"}:
                break
    if failures:
        print(f"\nAnalysis finished with {failures} failed step(s).", flush=True)
        return 1
    print("\nAnalysis finished successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
