#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path


def parse_total_tasks(run_dir: Path) -> int:
    totals = []

    summary = run_dir / "dry_run_summary.json"
    if summary.exists():
        try:
            with summary.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            v = int(obj.get("task_count", 0))
            if v > 0:
                totals.append(v)
        except Exception:
            pass

    log_path = run_dir / "runner.log"
    if log_path.exists():
        pat = re.compile(r"RUN_START \| tasks=(\d+)")
        try:
            with log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    m = pat.search(line)
                    if m:
                        totals.append(int(m.group(1)))
        except Exception:
            pass
    return max(totals) if totals else 0


def render_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    safe_done = min(done, total)
    filled = int((safe_done / total) * width)
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def count_done(run_dir: Path) -> int:
    # Support both current layout (<instance_id>/summary.json)
    # and legacy flat layout (*.summary.json).
    nested = len(list(run_dir.glob("*/summary.json")))
    if nested > 0:
        return nested
    return len(list(run_dir.glob("*.summary.json")))


def main() -> None:
    p = argparse.ArgumentParser(description="Watch run progress by counting task summary files")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    total = parse_total_tasks(run_dir)

    while True:
        done = count_done(run_dir)
        total = max(total, done)
        bar = render_bar(done, total)
        pct = (done / total * 100.0) if total > 0 else 0.0
        line = f"{run_dir.name} {bar} {done}/{total} ({pct:.1f}%)"
        print(line, flush=True)

        if (run_dir / "run_summary.json").exists() or args.once:
            break

        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    main()
