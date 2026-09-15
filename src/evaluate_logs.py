#!/usr/bin/env python3
"""
Independent evaluation tool: evaluates execution logs with configurable LLM model.

Operates independently of the execution phase, allowing re-evaluation with different
models or updated constraint descriptions.

Usage:
    python evaluate_logs.py --run-dir logs/runs/my_run --evaluator-model gpt-4o
    python evaluate_logs.py --run-dir logs/runs/my_run --task-ids task1,task2 --force
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Set

from src.evaluator_lib import EvaluatorError, TaskEvaluator
from src.agent_lib.data import load_dyna_jsonl
from src.utils.client import get_client
from src.utils.log_reader import LogReader


def evaluate_run_dir(
    run_dir: Path,
    dataset_path: Optional[str] = None,
    task_ids: Optional[Set[str]] = None,
    evaluator_model: str = "deepseek-v4-flash",
    force: bool = False,
    execution_date: str = "",
) -> int:
    """Evaluate all tasks in a run directory.

    Args:
        run_dir: Path to run directory containing task logs
        dataset_path: Path to task dataset file (needed to load full TaskInstance)
        task_ids: Set of specific task IDs to evaluate, or None for all
        evaluator_model: LLM model to use for constraint judgment
        force: If True, re-evaluate even if evaluation exists
        execution_date: Date the task was executed (YYYY-MM-DD)
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    # Find all task summary files (new layout: <task_id>/summary.json)
    summary_files = sorted(run_dir.glob("*/summary.json"))

    # Fallback to legacy flat layout (<task_id>.summary.json)
    if not summary_files:
        summary_files = sorted(run_dir.glob("*.summary.json"))

    if not summary_files:
        print(f"No tasks found in {run_dir}")
        return 0

    if not dataset_path:
        print("Error: --dataset-path is required (the evaluator needs v2 binding info).")
        sys.exit(1)

    try:
        tasks = load_dyna_jsonl(dataset_path)
    except Exception as e:
        print(f"Error: failed to load dataset from {dataset_path}: {e}")
        sys.exit(1)
    task_instances: Dict[str, any] = {t.instance_id: t for t in tasks}

    eval_dir = run_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if force:
        (eval_dir / "run_summary.json").unlink(missing_ok=True)
        for summary_file in summary_files:
            if summary_file.name == "summary.json":
                forced_task_id = summary_file.parent.name
            else:
                forced_task_id = summary_file.stem.replace(".summary", "")
            if task_ids and forced_task_id not in task_ids:
                continue
            forced_eval_file = eval_dir / f"{forced_task_id}.evaluation.json"
            forced_eval_file.unlink(missing_ok=True)
            forced_eval_file.with_suffix(
                forced_eval_file.suffix + ".tmp"
            ).unlink(missing_ok=True)

    try:
        client = get_client(evaluator_model)
    except Exception as exc:
        print(f"EVALUATOR ERROR: initialization failed: {str(exc)[:200]}")
        return 1

    evaluated_count = 0
    skipped_count = 0
    error_count = 0
    per_task_summary: list = []   # collected for run-level diagnostic aggregation

    for summary_file in summary_files:
        # New layout: <task_id>/summary.json → task_id is parent dir name
        # Legacy layout: <task_id>.summary.json → task_id is stem minus suffix
        if summary_file.name == "summary.json":
            task_id = summary_file.parent.name
        else:
            task_id = summary_file.stem.replace(".summary", "")

        # Filter by task_ids if specified
        if task_ids and task_id not in task_ids:
            continue

        # Check if already evaluated in standalone evaluation artifact
        eval_file = eval_dir / f"{task_id}.evaluation.json"
        if not force and eval_file.exists():
            print(f"  ⊘ {task_id}: already evaluated (use --force to re-evaluate)")
            skipped_count += 1
            continue
        print(f"→ Evaluating {task_id}...", end=" ", flush=True)

        try:
            if task_id not in task_instances:
                print(f"✗ ERROR: task {task_id} not found in dataset {dataset_path}")
                error_count += 1
                continue
            task = task_instances[task_id]

            # Create LogReader and TaskEvaluator
            log_reader = LogReader(run_dir, task_id)
            evaluator = TaskEvaluator(task, log_reader, client, evaluator_model, execution_date)

            # Evaluate
            evaluation_result = evaluator.evaluate()

            # Save standalone evaluation artifact (do not mutate execution summary)
            payload = {
                "task_id": task_id,
                "evaluator_model": evaluator_model,
                "result": evaluation_result,
            }
            temp_eval_file = eval_file.with_suffix(eval_file.suffix + ".tmp")
            temp_eval_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_eval_file.replace(eval_file)

            # Report
            is_correct = evaluation_result.get("is_correct", False)
            score = evaluation_result.get("correctness_score", 0.0)
            status = "✓ PASS" if is_correct else "✗ FAIL"
            print(f"{status} (score: {score:.2f})")
            per_task_summary.append({
                "task_id": task_id,
                "correctness_score": score,
                "is_correct": is_correct,
                "diagnostic_counts": evaluation_result.get("diagnostic_counts", {}),
            })
            evaluated_count += 1

        except EvaluatorError as e:
            eval_file.with_suffix(eval_file.suffix + ".tmp").unlink(missing_ok=True)
            print(f"✗ EVALUATOR ERROR: {e}")
            error_count += 1
        except Exception as e:
            eval_file.with_suffix(eval_file.suffix + ".tmp").unlink(missing_ok=True)
            print(f"✗ ERROR: {e}")
            error_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Evaluation complete:")
    print(f"  Evaluated: {evaluated_count}")
    print(f"  Skipped:   {skipped_count}")
    print(f"  Errors:    {error_count}")
    print(f"  Total:     {evaluated_count + skipped_count + error_count}")
    print("=" * 60)
    print(f"Evaluation artifacts: {eval_dir}")

    if per_task_summary:
        _write_run_summary(eval_dir, per_task_summary, evaluator_model)

    return error_count


def _write_run_summary(
    eval_dir: Path,
    per_task: list,
    evaluator_model: str,
) -> None:
    """Aggregate per-task results into a run-level summary file.

    Includes mean correctness score, total diagnostic-label counts, and
    rate metrics (confirmed_rate, claim_supported_rescued_rate,
    claim_partial_rate, hallucination_rate, evidence_only_rate,
    no_signal_rate, confirmed_failure_rate, ...) computed over the union
    of all per-task constraint outcomes.
    """
    n = len(per_task)
    mean_score = sum(t["correctness_score"] for t in per_task) / n if n else 0.0
    correct_count = sum(1 for t in per_task if t["is_correct"])

    label_totals: Dict[str, int] = {}
    for t in per_task:
        for label, cnt in (t.get("diagnostic_counts") or {}).items():
            label_totals[label] = label_totals.get(label, 0) + cnt
    total_constraints = sum(label_totals.values())
    rates = {
        f"{label}_rate": (cnt / total_constraints) if total_constraints else 0.0
        for label, cnt in label_totals.items()
    }

    payload = {
        "evaluator_model": evaluator_model,
        "task_count": n,
        "correct_count": correct_count,
        "success_rate": (correct_count / n) if n else 0.0,
        "mean_correctness_score": mean_score,
        "total_constraints": total_constraints,
        "diagnostic_label_totals": label_totals,
        "diagnostic_label_rates": rates,
        "per_task": per_task,
    }
    out = eval_dir / "run_summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Run summary: {out}")
    print(f"  mean_score={mean_score:.3f}  success_rate={correct_count}/{n}")
    if label_totals:
        top = sorted(label_totals.items(), key=lambda kv: -kv[1])[:5]
        print(f"  diagnostic top-5: " + ", ".join(f"{k}={v}" for k, v in top))


def main():
    parser = argparse.ArgumentParser(
        description="Independently evaluate task execution logs"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory (e.g., logs/runs/my_run)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        required=True,
        help="Path to JSONL task dataset (required; the evaluator needs v2 binding info)",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated task IDs to evaluate (default: all)",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default="deepseek-v4-flash",
        help="LLM model for constraint judgment (default: deepseek-v4-flash)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if evaluation already exists",
    )
    parser.add_argument(
        "--execution-date",
        type=str,
        default=None,
        help="Date the task was executed (YYYY-MM-DD). If omitted, auto-extracted from run dir name.",
    )

    args = parser.parse_args()

    # Parse task IDs if provided
    task_ids = None
    if args.task_ids:
        task_ids = set(x.strip() for x in args.task_ids.split(",") if x.strip())

    # Resolve execution date
    execution_date = args.execution_date or ""
    if not execution_date:
        # Try to extract YYYYMMDD from run dir name
        match = re.search(r"(\d{8})", Path(args.run_dir).name)
        if match:
            d = match.group(1)
            execution_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    error_count = evaluate_run_dir(
        run_dir=args.run_dir,
        dataset_path=args.dataset_path,
        task_ids=task_ids,
        evaluator_model=args.evaluator_model,
        force=args.force,
        execution_date=execution_date,
    )
    if error_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
