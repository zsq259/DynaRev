#!/usr/bin/env python3
"""Deferred site health postprocessing from run results.

Reads task results (*.summary.json) from a run directory and updates
site health inventory. Can be run after execution to backfill or re-process
site health states.

Usage:
    conda run -n dyna python -m src.data_prep.postprocess_site_health --run-dir logs/runs/my_run_name
"""
import argparse
import json
import sys
from pathlib import Path
from ..utils.site_health import SiteHealthStore


def postprocess_run(run_dir: Path) -> int:
    """Process all task results in a run directory and update site health."""
    if not run_dir.exists():
        print(f"Error: run_dir not found: {run_dir}", file=sys.stderr)
        return 1

    # Collect all task result files
    summary_files = sorted(run_dir.glob("*/summary.json"))
    if not summary_files:
        print(f"No *.summary.json files found in {run_dir}", file=sys.stderr)
        return 1

    site_health = SiteHealthStore()
    processed = 0

    for summary_path in summary_files:
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception as e:
            print(f"Failed to read {summary_path}: {e}", file=sys.stderr)
            continue

        # Extract task URLs and update site health
        website = result.get("website") or ""
        start_url = result.get("runtime", {}).get("start_url") or ""
        task_urls = [str(website), str(start_url)]

        site_health.register_urls(task_urls, source="postprocess", task_id=result.get("instance_id"))
        site_health.update_from_run_result(result, source="postprocess")
        processed += 1

    # Persist all changes
    site_health.save()
    site_health.export_feasible()

    print(f"Processed {processed} task results from {run_dir}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory to process (default: latest logs/runs/* dir)",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        # Use latest run directory
        runs_dir = Path("logs/runs")
        if not runs_dir.exists():
            print(f"Error: {runs_dir} not found", file=sys.stderr)
            sys.exit(1)
        dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if not dirs:
            print(f"No run directories found in {runs_dir}", file=sys.stderr)
            sys.exit(1)
        run_dir = max(dirs, key=lambda d: d.stat().st_mtime)

    sys.exit(postprocess_run(run_dir))


if __name__ == "__main__":
    main()
