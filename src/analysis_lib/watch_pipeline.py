#!/usr/bin/env python3
"""Real-time monitor for dyna_pipeline runs.

Usage:
    python -m src.analyze_runs watch-pipeline --run <pipeline_run>
    python -m src.analyze_runs watch-pipeline --run <pipeline_run> --once
"""
import argparse
import re
import time
from pathlib import Path

LOG_DIR = Path("logs/pipeline")

# Patterns
P_START   = re.compile(r"PIPELINE_START \| tasks=(\d+)")
P_STEP1   = re.compile(r"STEP1 \| (\S+) \| Extracting")
P_PASSED4 = re.compile(r"STEP4 \| (\S+) \| PASSED")
P_FAILED4 = re.compile(r"STEP4 \| (\S+) \| FAILED")
P_PASSED5 = re.compile(r"STEP5 \| (\S+) \| PASSED")
P_FAILED5 = re.compile(r"STEP5 \| (\S+) \| FAILED")
P_FILTER  = re.compile(r"WARNING \| STEP\d+ \| (\S+) \| (Rejected|filtered)", re.I)
P_END     = re.compile(r"PIPELINE_END \| total=(\d+) \| passed=(\d+) \| filtered=(\d+)")
P_STEP_CUR = re.compile(r"STEP(\d) \| (\S+) \| (.+)$")


def parse_log(log_path: Path):
    total = 0
    seen_order = []   # task_ids in encounter order
    status = {}       # task_id -> "running" | "passed" | "filtered"
    step_current = {} # task_id -> (step, detail)
    finished = False
    final = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = P_START.search(line)
            if m:
                total = int(m.group(1))
                continue
            m = P_END.search(line)
            if m:
                finished = True
                final = {"total": int(m.group(1)), "passed": int(m.group(2)), "filtered": int(m.group(3))}
                continue
            m = P_PASSED5.search(line)
            if m:
                status[m.group(1)] = "passed"
                step_current[m.group(1)] = (5, "PASSED")
                continue
            m = P_FILTER.search(line)
            if m:
                status[m.group(1)] = "filtered"
                continue
            m = P_STEP1.search(line)
            if m:
                tid = m.group(1)
                if tid not in seen_order:
                    seen_order.append(tid)
                if status.get(tid) not in ("passed", "filtered"):
                    status[tid] = "running"
                step_current[tid] = (1, "extracting constraints")
                continue
            m = P_STEP_CUR.search(line)
            if m:
                step, tid, detail = m.group(1), m.group(2), m.group(3)[:60]
                if status.get(tid) not in ("passed", "filtered"):
                    step_current[tid] = (int(step), detail.strip())

    return total, seen_order, status, step_current, finished, final


def render(run_name: str, log_path: Path):
    total, seen, status, step_cur, finished, final = parse_log(log_path)

    passed  = sum(1 for s in status.values() if s == "passed")
    filtered = sum(1 for s in status.values() if s == "filtered")
    running  = sum(1 for s in status.values() if s == "running")
    done     = passed + filtered

    bar_width = 30
    filled = int((done / total) * bar_width) if total else 0
    bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
    pct = done / total * 100 if total else 0

    print(f"\n{'='*60}")
    print(f"  Pipeline: {run_name}")
    print(f"  {bar}  {done}/{total}  ({pct:.0f}%)")
    print(f"  passed={passed}  filtered={filtered}  running={running}")
    print(f"{'='*60}")

    for tid in seen:
        s = status.get(tid, "?")
        icon = {"passed": "✓", "filtered": "✗", "running": "→"}.get(s, "?")
        step, detail = step_cur.get(tid, (0, ""))
        step_str = f"  [step{step}] {detail}" if s == "running" else ""
        print(f"  {icon} {tid:<45}{step_str}")

    if finished:
        print(f"\n  DONE — passed={final['passed']}  filtered={final['filtered']}")
    print()
    return finished


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run name under logs/pipeline/")
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    log_path = LOG_DIR / args.run / "pipeline.log"
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")

    while True:
        done = render(args.run, log_path)
        if done or args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
