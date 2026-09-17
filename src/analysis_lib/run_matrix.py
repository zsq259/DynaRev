#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def read_summary(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "run_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_eval_stats(run_dir: Path) -> Dict[str, float]:
    eval_dir = run_dir / "evaluations"
    if not eval_dir.exists():
        return {
            "evaluated_count": 0,
            "correct_count": 0,
            "avg_correctness_score": 0.0,
        }

    scores: List[float] = []
    correct_count = 0
    for path in sorted(eval_dir.glob("*.evaluation.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            result = obj.get("result") or {}
            if result.get("exclude_from_official") or result.get("eval_status") in {
                "environment_error",
                "evaluator_error",
            }:
                continue
            score = float(result.get("correctness_score", 0.0) or 0.0)
            is_correct = bool(result.get("is_correct", False))
            scores.append(score)
            if is_correct:
                correct_count += 1
        except Exception:
            continue

    return {
        "evaluated_count": len(scores),
        "correct_count": correct_count,
        "avg_correctness_score": (sum(scores) / len(scores)) if scores else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize multiple dyna run directories")
    p.add_argument("--log-dir", default="logs/runs")
    p.add_argument("--prefix", required=True, help="run_name prefix, e.g. scene29_ds_eval")
    p.add_argument("--output-csv", default="reports/run_matrix.csv")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    run_dirs = sorted([d for d in log_dir.glob(f"{args.prefix}*") if d.is_dir()])

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = read_summary(run_dir)
        if not summary:
            continue
        eval_stats = read_eval_stats(run_dir)
        task_count = int(summary.get("task_count", 0) or 0)
        evaluated_count = int(eval_stats.get("evaluated_count", 0) or 0)
        correct_count = int(eval_stats.get("correct_count", 0) or 0)
        submitted_count = summary.get("submitted_count")
        if submitted_count is None:
            submitted_count = sum(
                1
                for result in summary.get("results", [])
                if result.get("stopped_reason") == "env_done"
                and bool(result.get("done_called", False))
            )
        rows.append(
            {
                "run_name": summary.get("run_name", run_dir.name),
                "dataset_type": summary.get("dataset_type", ""),
                "policy_model": summary.get("policy_model", ""),
                "trigger_model": summary.get("trigger_model", ""),
                "evaluator_model": summary.get("evaluator_model", ""),
                "task_count": task_count,
                "evaluated_count": evaluated_count,
                "correct_count": correct_count,
                "correct_rate": (
                    float(correct_count) / float(evaluated_count)
                    if evaluated_count > 0
                    else 0.0
                ),
                "avg_correctness_score": eval_stats.get("avg_correctness_score", 0.0),
                "avg_steps": summary.get("avg_steps", 0.0),
                "done_called_count": summary.get("done_called_count", 0),
                "submitted_count": submitted_count,
            }
        )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_name",
        "dataset_type",
        "policy_model",
        "trigger_model",
        "evaluator_model",
        "task_count",
        "evaluated_count",
        "correct_count",
        "correct_rate",
        "avg_correctness_score",
        "avg_steps",
        "done_called_count",
        "submitted_count",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(str(out_path))
    print(f"RUNS={len(rows)}")


if __name__ == "__main__":
    main()
