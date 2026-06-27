from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from innovation.run_tabicl_retrieval_context_pilot import main


if __name__ == "__main__":
    default_args = [
        "--datasets",
        "mfeat-zernike",
        "--k",
        "512",
        "--retrieval-metrics",
        "euclidean",
        "--test-selection",
        "random",
        "--test-sample-size",
        "50",
        "--test-sample-seed",
        "42",
        "--skip-random",
    ]
    sys.argv[1:1] = default_args
    raise SystemExit(main())
