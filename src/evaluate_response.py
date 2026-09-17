#!/usr/bin/env python3
"""Compute response correctness + response efficiency metrics on top of an
existing answer-correctness evaluation.

Reads ``<run_dir>/evaluations/<task_id>.evaluation.json`` (produced by
``evaluate_logs.py``) plus the task's ``events.jsonl`` and ``summary.json``
under ``<run_dir>/<task_id>/``. Writes:

- ``<run_dir>/evaluations/<task_id>.response_metrics.json`` — per-task
  RCSR / MFFR / per-round combine_label distribution / CSD per round /
  WRA + adaptation per fired revision.
- ``<run_dir>/evaluations/response_summary.json`` — run-level aggregates.

Usage:
    python -m src.evaluate_response \\
        --run-dir logs/runs/<run> \\
        --dataset-path data/dyna-tasks/batches/batch0/tasks.jsonl \\
        [--task-ids id1,id2] [--judge-model deepseek-v4-flash] [--force]
        [--no-judge]   # skip LLM-backed WRA + adaptation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.agent_lib.data import load_dyna_jsonl
from src.evaluator_lib import EvaluatorError
from src.evaluator_lib.response_judge import judge_response_window
from src.evaluator_lib.response_metrics import (
    compute_csd_per_round,
    compute_demand_side_metrics,
    compute_injection_steps,
    slice_window,
)
from src.utils.client import get_client
from src.utils.log_reader import LogReader


def evaluate_task_response(
    task,
    log_reader: LogReader,
    eval_result: Dict[str, Any],
    client,
    judge_model: str,
    use_judge: bool,
    execution_date: str,
) -> Dict[str, Any]:
    """Compute all response metrics for a single task."""
    events = log_reader.get_events()
    summary = log_reader.get_summary()
    total_steps = summary.get("steps", events[-1]["step"] if events else 0)

    injection_steps = compute_injection_steps(events, summary)
    demand = compute_demand_side_metrics(eval_result, task)
    efficiency = compute_csd_per_round(eval_result, task, injection_steps, total_steps)

    # Behavior side: judge each fired revision (skip i0 — it's the
    # original task, not a correction).
    fired_in_order: List[str] = efficiency["fired_revisions_in_order"]
    revision_text: Dict[str, str] = {r.revision_id: r.content for r in task.revisions}

    window_judgments: Dict[str, Dict[str, Any]] = {}
    if use_judge:
        # Build next-window-start map; reuse efficiency boundaries.
        for rev_id in fired_in_order:
            csd_entry = efficiency["csd_per_round"][rev_id]
            window_start = csd_entry["window_start_step"]
            next_window_start = csd_entry["next_window_start_step"]
            window_events = slice_window(events, window_start, next_window_start)
            pre_url: Optional[str] = None
            for ev in events:
                if ev.get("step") == window_start - 1:
                    pre_url = ev.get("url")
                    break
            judgment = judge_response_window(
                instruction_id=rev_id,
                instruction_text=revision_text.get(rev_id, ""),
                window_events=window_events,
                client=client,
                model=judge_model,
                execution_date=execution_date,
                pre_injection_url=pre_url,
            )
            judgment["window_steps"] = [ev.get("step") for ev in window_events]
            judgment["window_start_step"] = window_start
            judgment["next_window_start_step"] = next_window_start
            window_judgments[rev_id] = judgment

    # Aggregate WRA + adaptation distribution
    wra_count = sum(j.get("wra", 0) for j in window_judgments.values())
    n_judged = len(window_judgments)
    wra = (wra_count / n_judged) if n_judged else None
    adaptation_dist: Dict[str, int] = {}
    for j in window_judgments.values():
        a = j.get("adaptation", "")
        if a:
            adaptation_dist[a] = adaptation_dist.get(a, 0) + 1

    return {
        "task_id": task.instance_id,
        "n_revisions_total": demand["n_rounds"],
        "fired_revisions_in_order": fired_in_order,
        "injection_steps": injection_steps,
        "total_steps": total_steps,
        "demand_side": {
            "rcsr_per_round": demand["rcsr_per_round"],
            "mffr": demand["mffr"],
            "label_distribution_per_round": demand["label_distribution_per_round"],
        },
        "efficiency": {
            "csd_per_round": efficiency["csd_per_round"],
            "mean_csd": efficiency["mean_csd"],
        },
        "behavior_side": {
            "wra": wra,
            "wra_count": wra_count,
            "n_judged": n_judged,
            "adaptation_distribution": adaptation_dist,
            "per_round": window_judgments,
        },
    }


def evaluate_run_dir(
    run_dir: Path,
    dataset_path: str,
    task_ids: Optional[Set[str]] = None,
    judge_model: str = "deepseek-v4-flash",
    use_judge: bool = True,
    force: bool = False,
    execution_date: str = "",
) -> int:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    eval_dir = run_dir / "evaluations"
    if not eval_dir.exists():
        print(
            f"Error: {eval_dir} not found. Run evaluate_logs.py first to "
            "produce answer-correctness evaluations."
        )
        sys.exit(1)

    eval_files = sorted(eval_dir.glob("*.evaluation.json"))
    if not eval_files:
        print(f"No *.evaluation.json under {eval_dir}. Run evaluate_logs.py first.")
        return 0

    try:
        tasks = load_dyna_jsonl(dataset_path)
    except Exception as e:
        print(f"Error: failed to load dataset from {dataset_path}: {e}")
        sys.exit(1)
    task_instances = {t.instance_id: t for t in tasks}

    if force:
        (eval_dir / "response_summary.json").unlink(missing_ok=True)
        for ef in eval_files:
            forced_task_id = ef.name.replace(".evaluation.json", "")
            if task_ids and forced_task_id not in task_ids:
                continue
            forced_out_file = eval_dir / f"{forced_task_id}.response_metrics.json"
            forced_out_file.unlink(missing_ok=True)
            forced_out_file.with_suffix(
                forced_out_file.suffix + ".tmp"
            ).unlink(missing_ok=True)

    try:
        client = get_client(judge_model) if use_judge else None
    except Exception as exc:
        print(f"EVALUATOR ERROR: initialization failed: {str(exc)[:200]}")
        return 1

    per_task: List[Dict[str, Any]] = []
    evaluated = skipped = errors = 0

    for ef in eval_files:
        task_id = ef.name.replace(".evaluation.json", "")
        if task_ids and task_id not in task_ids:
            continue

        out_file = eval_dir / f"{task_id}.response_metrics.json"
        if out_file.exists() and not force:
            print(f"  ⊘ {task_id}: response_metrics exists (use --force to re-run)")
            skipped += 1
            try:
                per_task.append(json.loads(out_file.read_text(encoding="utf-8")))
            except Exception:
                pass
            continue
        if task_id not in task_instances:
            print(f"  ✗ {task_id}: not found in dataset")
            errors += 1
            continue

        try:
            payload = json.loads(ef.read_text(encoding="utf-8"))
            eval_result = payload["result"]
            if eval_result.get("exclude_from_official") or eval_result.get(
                "eval_status"
            ) in {"environment_error", "evaluator_error"}:
                status = eval_result.get("eval_status") or "excluded"
                print(f"→ {task_id}: {status} (excluded from official metrics)")
                skipped += 1
                continue
            log_reader = LogReader(run_dir, task_id)
            task = task_instances[task_id]

            print(f"→ {task_id}...", end=" ", flush=True)
            metrics = evaluate_task_response(
                task=task,
                log_reader=log_reader,
                eval_result=eval_result,
                client=client,
                judge_model=judge_model,
                use_judge=use_judge,
                execution_date=execution_date,
            )
            temp_out_file = out_file.with_suffix(out_file.suffix + ".tmp")
            temp_out_file.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_out_file.replace(out_file)
            mffr = metrics["demand_side"]["mffr"]
            mean_csd = metrics["efficiency"]["mean_csd"]
            wra = metrics["behavior_side"]["wra"]
            mean_csd_str = f"{mean_csd:.4f}" if mean_csd is not None else "n/a"
            wra_str = f"{wra:.2f}" if wra is not None else "n/a"
            print(
                f"✓ MFFR={mffr} mean_CSD={mean_csd_str} "
                f"WRA={wra_str} (n_judged={metrics['behavior_side']['n_judged']})"
            )
            per_task.append(metrics)
            evaluated += 1
        except EvaluatorError as e:
            out_file.with_suffix(out_file.suffix + ".tmp").unlink(missing_ok=True)
            print(f"✗ EVALUATOR ERROR: {e}")
            errors += 1
        except Exception as e:
            out_file.with_suffix(out_file.suffix + ".tmp").unlink(missing_ok=True)
            print(f"✗ ERROR: {e}")
            errors += 1

    print("\n" + "=" * 60)
    print(f"Response metrics: evaluated={evaluated} skipped={skipped} errors={errors}")
    print("=" * 60)

    if per_task:
        _write_run_summary(eval_dir, per_task, judge_model, use_judge)

    return errors


def _write_run_summary(
    eval_dir: Path,
    per_task: List[Dict[str, Any]],
    judge_model: str,
    use_judge: bool,
) -> None:
    """Aggregate per-task response metrics into a run-level summary."""
    n = len(per_task)

    # Demand side aggregates
    mffr_values = [t["demand_side"]["mffr"] for t in per_task]
    mean_mffr = sum(mffr_values) / n if n else 0.0

    # Per-revision-ordinal RCSR (i1, i2, ...) — averaged across tasks
    # that have ≥k revisions defined. Not all tasks have the same number
    # of revisions, so per-ordinal gets a per-task-count denominator.
    rcsr_by_ordinal: Dict[int, List[float]] = {}
    for t in per_task:
        revisions = [k for k in t["demand_side"]["rcsr_per_round"].keys() if k != "i0"]
        for ord_idx, rev_id in enumerate(sorted(revisions, key=lambda r: int(r[1:])), start=1):
            entry = t["demand_side"]["rcsr_per_round"][rev_id]
            rcsr = entry.get("rcsr")
            if rcsr is not None:
                rcsr_by_ordinal.setdefault(ord_idx, []).append(rcsr)
    mean_rcsr_by_ordinal = {
        f"i{k}": (sum(vs) / len(vs)) if vs else None
        for k, vs in rcsr_by_ordinal.items()
    }
    rcsr_n_by_ordinal = {f"i{k}": len(vs) for k, vs in rcsr_by_ordinal.items()}

    # Efficiency aggregates
    csd_means = [
        t["efficiency"]["mean_csd"] for t in per_task
        if t["efficiency"]["mean_csd"] is not None
    ]
    macro_mean_csd = (sum(csd_means) / len(csd_means)) if csd_means else None
    csd_by_ordinal: Dict[int, List[float]] = {}
    for t in per_task:
        fired_order = t["fired_revisions_in_order"]
        for ord_idx, rev_id in enumerate(fired_order, start=1):
            csd = t["efficiency"]["csd_per_round"][rev_id]["csd"]
            csd_by_ordinal.setdefault(ord_idx, []).append(csd)
    mean_csd_by_ordinal = {
        f"i{k}": (sum(vs) / len(vs)) if vs else None
        for k, vs in csd_by_ordinal.items()
    }

    # Behavior side aggregates
    wra_values = [
        t["behavior_side"]["wra"] for t in per_task
        if t["behavior_side"]["wra"] is not None
    ]
    macro_mean_wra = (sum(wra_values) / len(wra_values)) if wra_values else None
    total_judged = sum(t["behavior_side"]["n_judged"] for t in per_task)
    total_wra = sum(t["behavior_side"]["wra_count"] for t in per_task)
    micro_wra = (total_wra / total_judged) if total_judged else None
    adaptation_total: Dict[str, int] = {}
    for t in per_task:
        for label, cnt in t["behavior_side"]["adaptation_distribution"].items():
            adaptation_total[label] = adaptation_total.get(label, 0) + cnt

    payload = {
        "judge_model": judge_model,
        "judge_used": use_judge,
        "task_count": n,
        "demand_side": {
            "mean_mffr": mean_mffr,
            "mean_rcsr_by_ordinal": mean_rcsr_by_ordinal,
            "n_tasks_with_ordinal": rcsr_n_by_ordinal,
        },
        "efficiency": {
            "macro_mean_csd": macro_mean_csd,
            "mean_csd_by_ordinal": mean_csd_by_ordinal,
        },
        "behavior_side": {
            "macro_mean_wra": macro_mean_wra,
            "micro_wra": micro_wra,
            "total_judged": total_judged,
            "adaptation_distribution": adaptation_total,
        },
        "per_task": [
            {
                "task_id": t["task_id"],
                "mffr": t["demand_side"]["mffr"],
                "mean_csd": t["efficiency"]["mean_csd"],
                "wra": t["behavior_side"]["wra"],
                "n_judged": t["behavior_side"]["n_judged"],
            }
            for t in per_task
        ],
    }
    out = eval_dir / "response_summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Response summary: {out}")
    if macro_mean_csd is not None:
        print(f"  macro_mean_CSD={macro_mean_csd:.5f}  mean_MFFR={mean_mffr:.2f}")
    if macro_mean_wra is not None:
        print(f"  macro_mean_WRA={macro_mean_wra:.3f}  micro_WRA={micro_wra:.3f}")
    if adaptation_total:
        print(f"  adaptation: {adaptation_total}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute response correctness + efficiency metrics."
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument(
        "--dataset-path", type=str, required=True,
        help="Path to JSONL task dataset (v2 with binding info).",
    )
    parser.add_argument(
        "--task-ids", type=str, default=None,
        help="Comma-separated task ids; default = all tasks with eval JSON.",
    )
    parser.add_argument("--judge-model", type=str, default="deepseek-v4-flash")
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM-backed WRA + adaptation; only emit deterministic metrics.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execution-date", type=str, default=None)
    args = parser.parse_args()

    task_ids = None
    if args.task_ids:
        task_ids = set(x.strip() for x in args.task_ids.split(",") if x.strip())

    execution_date = args.execution_date or ""
    if not execution_date:
        m = re.search(r"(\d{8})", Path(args.run_dir).name)
        if m:
            d = m.group(1)
            execution_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    error_count = evaluate_run_dir(
        run_dir=Path(args.run_dir),
        dataset_path=args.dataset_path,
        task_ids=task_ids,
        judge_model=args.judge_model,
        use_judge=not args.no_judge,
        force=args.force,
        execution_date=execution_date,
    )
    if error_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
