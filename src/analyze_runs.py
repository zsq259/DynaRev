#!/usr/bin/env python3
"""Cross-run analysis CLI.

Writes derived tables and markdown under ``reports/``. Per-run scores stay in
``logs/runs/<run>/evaluations/``.

Usage (from the repository root):
    python -m src.analyze_runs matrix --prefix run_ --output-csv reports/run_matrix.csv
    python -m src.analyze_runs eval-report --run-dir logs/runs/my_run --output reports/evaluation.md
    python -m src.analyze_runs compare --runs run_a run_b --out-csv reports/task_matrix.csv
    python -m src.analyze_runs baseline --run-a logs/runs/run_a --run-b logs/runs/run_b --output reports/compare.md
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    cmd, rest = sys.argv[1], sys.argv[2:]
    sys.argv = [sys.argv[0], *rest]

    if cmd == "matrix":
        from src.analysis_lib.run_matrix import main as run
    elif cmd == "eval-report":
        from src.analysis_lib.eval_report import main as run
    elif cmd == "compare":
        from src.analysis_lib.compare_runs import main as run
    elif cmd == "baseline":
        from src.analysis_lib.baseline_compare import main as run
    elif cmd == "curves":
        from src.analysis_lib.plot_cost_quality_curves import main as run
    elif cmd == "score-thresholds":
        from src.analysis_lib.plot_score_thresholds import main as run
    elif cmd in {"watch-run", "watch"}:
        from src.analysis_lib.watch_run import main as run
    elif cmd == "watch-pipeline":
        from src.analysis_lib.watch_pipeline import main as run
    else:
        raise SystemExit(
            "unknown command: "
            f"{cmd}\n"
            "use: matrix | eval-report | compare | baseline | curves | "
            "score-thresholds | watch-run | watch-pipeline"
        )
    run()


if __name__ == "__main__":
    main()
