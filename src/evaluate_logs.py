#!/usr/bin/env python3
"""
Independent evaluation tool: evaluates execution logs with configurable LLM model.

Operates independently of the execution phase, allowing re-evaluation with different
models or updated constraint descriptions.

Episodes that hit max_steps without a valid final submit skip the LLM evaluator
and receive a zero-score ``no_final_submit`` artifact (see evaluator_lib.no_final_submit).
Website/browser environment failures skip the LLM evaluator and are excluded
from official score denominators (see evaluator_lib.environment_error).
Evaluator-model failures are retried from scratch; if they still fail, no
evaluation JSON is written, so the next ``evaluate_logs`` run retries them
without ``--force``.

Usage:
    python evaluate_logs.py --run-dir logs/runs/my_run --evaluator-model gpt-4o
    python evaluate_logs.py --run-dir logs/runs/my_run --task-ids task1,task2 --force
"""

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Optional, Set

from src.evaluator_lib import EvaluatorError, TaskEvaluator
from src.evaluator_lib.environment_error import (
    ENVIRONMENT_ERROR_LABEL,
    build_environment_error_result,
    is_environment_error,
)
from src.evaluator_lib.no_final_submit import (
    build_no_final_submit_result,
    is_no_final_submit,
)
from src.agent_lib.data import load_dyna_jsonl
from src.utils.client import get_client, make_client
from src.utils.log_reader import LogReader


def evaluate_run_dir(
    run_dir: Path,
    dataset_path: Optional[str] = None,
    task_ids: Optional[Set[str]] = None,
    *,
    evaluator_model: str,
    force: bool = False,
    execution_date: str = "",
    eval_retries: int = 2,
    eval_workers: int = 1,
) -> int:
    """Evaluate all tasks in a run directory.

    Args:
        run_dir: Path to run directory containing task logs
        dataset_path: Path to task dataset file (needed to load full TaskInstance)
        task_ids: Set of specific task IDs to evaluate, or None for all
        evaluator_model: LLM model to use for constraint judgment (required)
        force: If True, re-evaluate even if evaluation exists
        execution_date: Date the task was executed (YYYY-MM-DD)
        eval_retries: extra full-task attempts after an EvaluatorError
            (discard the failed attempt; do not continue from it)
        eval_workers: max tasks to score at once; 1 keeps the serial loop
    """
    if not evaluator_model:
        raise ValueError("evaluator_model is required")
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

    print_lock = threading.Lock()
    evaluated_count = 0
    skipped_count = 0
    error_count = 0
    jobs = []

    def _say(message: str) -> None:
        with print_lock:
            print(message, flush=True)

    for summary_file in summary_files:
        if summary_file.name == "summary.json":
            task_id = summary_file.parent.name
        else:
            task_id = summary_file.stem.replace(".summary", "")

        if task_ids and task_id not in task_ids:
            continue
        if task_id.startswith("_"):
            continue

        eval_file = eval_dir / f"{task_id}.evaluation.json"
        if not force and _is_reusable_eval_artifact(eval_file):
            _say(f"  ⊘ {task_id}: already evaluated (use --force to re-evaluate)")
            skipped_count += 1
            continue
        jobs.append((summary_file, task_id, eval_file))

    workers = max(1, int(eval_workers or 1))
    serial_client = None

    def serial_client_factory():
        nonlocal serial_client
        if serial_client is None:
            try:
                serial_client = get_client(evaluator_model)
            except Exception as exc:
                raise EvaluatorError(
                    f"initialization failed: {str(exc)[:200]}"
                ) from exc
        return serial_client

    def parallel_client_factory():
        try:
            return make_client(evaluator_model)
        except Exception as exc:
            raise EvaluatorError(
                f"initialization failed: {str(exc)[:200]}"
            ) from exc

    client_factory = serial_client_factory if workers == 1 else parallel_client_factory

    def run_job(summary_file, task_id, eval_file) -> str:
        return _evaluate_one_job(
            run_dir=run_dir,
            dataset_path=dataset_path,
            task_instances=task_instances,
            summary_file=summary_file,
            task_id=task_id,
            eval_file=eval_file,
            evaluator_model=evaluator_model,
            execution_date=execution_date,
            eval_retries=eval_retries,
            client_factory=client_factory,
            say=_say,
        )

    if workers == 1:
        outcomes = [run_job(*job) for job in jobs]
    else:
        outcomes = []
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs) or 1)) as pool:
            futs = [pool.submit(run_job, *job) for job in jobs]
            for fut in as_completed(futs):
                outcomes.append(fut.result())

    evaluated_count = sum(1 for item in outcomes if item == "evaluated")
    error_count = sum(1 for item in outcomes if item == "error")

    # Summary
    print("\n" + "=" * 60)
    print(f"Evaluation complete:")
    print(f"  Evaluated: {evaluated_count}")
    print(f"  Skipped:   {skipped_count}")
    print(f"  Errors:    {error_count}")
    print(f"  Total:     {evaluated_count + skipped_count + error_count}")
    print("=" * 60)
    print(f"Evaluation artifacts: {eval_dir}")

    disk_summary = _load_per_task_from_eval_dir(eval_dir)
    if disk_summary:
        _write_run_summary(eval_dir, disk_summary, evaluator_model)

    return error_count


def _evaluate_one_job(
    *,
    run_dir: Path,
    dataset_path: Optional[str],
    task_instances: Dict[str, any],
    summary_file: Path,
    task_id: str,
    eval_file: Path,
    evaluator_model: str,
    execution_date: str,
    eval_retries: int,
    client_factory: Callable,
    say: Callable[[str], None],
) -> str:
    """Score one task. Returns ``evaluated`` or ``error``. Does not write run_summary."""
    try:
        if task_id not in task_instances:
            say(
                f"→ Evaluating {task_id}... ✗ ERROR: task {task_id} not found in dataset {dataset_path}"
            )
            return "error"
        task = task_instances[task_id]
        summary = json.loads(summary_file.read_text(encoding="utf-8"))

        if is_environment_error(summary):
            prefix = f"→ {task_id}: environment_error (excluded from official scores)..."
            evaluation_result = build_environment_error_result(task, summary)
        elif is_no_final_submit(summary):
            prefix = f"→ {task_id}: no_final_submit (max_steps)..."
            evaluation_result = build_no_final_submit_result(task, summary)
        else:
            prefix = f"→ Evaluating {task_id}..."
            log_reader = LogReader(run_dir, task_id)
            try:
                evaluation_result = _evaluate_task_with_retries(
                    task=task,
                    log_reader=log_reader,
                    client=client_factory(),
                    evaluator_model=evaluator_model,
                    execution_date=execution_date,
                    eval_retries=eval_retries,
                )
            except EvaluatorError as exc:
                eval_file.unlink(missing_ok=True)
                eval_file.with_suffix(eval_file.suffix + ".tmp").unlink(missing_ok=True)
                say(f"{prefix} ✗ EVALUATOR ERROR: {exc}")
                return "error"

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

        is_correct = evaluation_result.get("is_correct", False)
        score = evaluation_result.get("correctness_score", 0.0)
        eval_status = evaluation_result.get("eval_status", "evaluated")
        if eval_status == ENVIRONMENT_ERROR_LABEL:
            say(f"{prefix} ⊘ EXCLUDED")
        elif eval_status == "no_final_submit":
            say(f"{prefix} ✗ FAIL (score: {score:.2f}, no_final_submit)")
        else:
            status = "✓ PASS" if is_correct else "✗ FAIL"
            say(f"{prefix} {status} (score: {score:.2f})")
        return "evaluated"
    except EvaluatorError as exc:
        eval_file.with_suffix(eval_file.suffix + ".tmp").unlink(missing_ok=True)
        say(f"→ Evaluating {task_id}... ✗ EVALUATOR ERROR: {exc}")
        return "error"
    except Exception as exc:
        eval_file.with_suffix(eval_file.suffix + ".tmp").unlink(missing_ok=True)
        say(f"→ Evaluating {task_id}... ✗ ERROR: {exc}")
        return "error"


def _evaluate_task_with_retries(
    *,
    task,
    log_reader,
    client,
    evaluator_model: str,
    execution_date: str,
    eval_retries: int,
):
    """Run one task evaluation from scratch on each attempt.

    An EvaluatorError discards that attempt's in-memory state and starts
    a new TaskEvaluator. Partial JSON from a failed LLM call is not reused.
    """
    extra = max(0, int(eval_retries))
    max_attempts = 1 + extra
    last_error: EvaluatorError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            evaluator = TaskEvaluator(
                task, log_reader, client, evaluator_model, execution_date
            )
            return evaluator.evaluate()
        except EvaluatorError as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            print(
                f"(evaluator error, retry {attempt + 1}/{max_attempts})...",
                end=" ",
                flush=True,
            )
    assert last_error is not None
    raise last_error


def _is_reusable_eval_artifact(eval_file: Path) -> bool:
    """True when an existing eval JSON should skip a later evaluate_logs run.

    Evaluator failures must not leave a reusable artifact. Leftover
    ``evaluator_error`` files from older code are treated as missing.
    """
    if not eval_file.exists():
        return False
    try:
        payload = json.loads(eval_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    result = payload.get("result") or {}
    return result.get("eval_status") != "evaluator_error"


def _load_per_task_from_eval_dir(eval_dir: Path) -> list:
    rows = []
    for path in sorted(eval_dir.glob("*.evaluation.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        result = payload.get("result") or {}
        rows.append(
            {
                "task_id": payload.get("task_id")
                or path.name.replace(".evaluation.json", ""),
                "correctness_score": result.get("correctness_score"),
                "is_correct": result.get("is_correct"),
                "eval_status": result.get("eval_status", "evaluated"),
                "exclude_from_official": bool(result.get("exclude_from_official")),
                "diagnostic_counts": result.get("diagnostic_counts", {}),
            }
        )
    return rows


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
    scored = [t for t in per_task if not t.get("exclude_from_official")]
    excluded = [t for t in per_task if t.get("exclude_from_official")]
    n = len(scored)
    mean_score = sum(float(t["correctness_score"] or 0.0) for t in scored) / n if n else 0.0
    correct_count = sum(1 for t in scored if t["is_correct"])
    no_final_submit_count = sum(
        1 for t in scored if t.get("eval_status") == "no_final_submit"
    )
    environment_excluded_count = sum(
        1 for t in excluded if t.get("eval_status") == ENVIRONMENT_ERROR_LABEL
    )
    evaluator_excluded_count = sum(
        1 for t in excluded if t.get("eval_status") == "evaluator_error"
    )

    label_totals: Dict[str, int] = {}
    for t in scored:
        for label, cnt in (t.get("diagnostic_counts") or {}).items():
            label_totals[label] = label_totals.get(label, 0) + cnt
    total_constraints = sum(label_totals.values())
    rates = {
        f"{label}_rate": (cnt / total_constraints) if total_constraints else 0.0
        for label, cnt in label_totals.items()
    }

    payload = {
        "evaluator_model": evaluator_model,
        "task_count": len(per_task),
        "scored_task_count": n,
        "environment_excluded_count": environment_excluded_count,
        "evaluator_excluded_count": evaluator_excluded_count,
        "correct_count": correct_count,
        "success_rate": (correct_count / n) if n else 0.0,
        "mean_correctness_score": mean_score,
        "no_final_submit_count": no_final_submit_count,
        "total_constraints": total_constraints,
        "diagnostic_label_totals": label_totals,
        "diagnostic_label_rates": rates,
        "per_task": per_task,
    }
    out = eval_dir / "run_summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Run summary: {out}")
    print(
        f"  mean_score={mean_score:.3f}  success_rate={correct_count}/{n}"
        f"  env_excluded={environment_excluded_count}"
        f"  eval_excluded={evaluator_excluded_count}"
    )
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
        required=True,
        help="LLM model for constraint judgment (no default; must be explicit)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if evaluation already exists",
    )
    parser.add_argument(
        "--eval-retries",
        type=int,
        default=2,
        help="extra full-task evaluation attempts after an evaluator error",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        required=True,
        help="max tasks to score at once; 1 keeps the serial loop",
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
        eval_retries=args.eval_retries,
        eval_workers=args.eval_workers,
    )
    if error_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
