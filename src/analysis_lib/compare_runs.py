#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_task_summaries(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    eval_map: Dict[str, Dict[str, Any]] = {}
    eval_dir = run_dir / "evaluations"
    if eval_dir.exists():
        for path in sorted(eval_dir.glob("*.evaluation.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                task_id = str(obj.get("task_id") or path.stem.replace(".evaluation", ""))
                eval_map[task_id] = obj.get("result") or {}
            except Exception:
                continue

    items: Dict[str, Dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*.summary.json")):
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        instance_id = str(obj.get("instance_id") or path.stem.replace(".summary", ""))
        obj["_evaluation"] = eval_map.get(instance_id, {})
        items[instance_id] = obj
    return items


def eval_info(task_summary: Dict[str, Any]) -> Dict[str, Any]:
    return task_summary.get("_evaluation") or {}


def is_excluded_from_official(task_summary: Dict[str, Any]) -> bool:
    info = eval_info(task_summary)
    status = str(info.get("eval_status") or "")
    if info.get("exclude_from_official") or status in {
        "environment_error",
        "evaluator_error",
    }:
        return True
    return completion_status(task_summary) == "environment_error"


def completion_status(task_summary: Dict[str, Any]) -> str:
    status = task_summary.get("completion_status")
    if status:
        return str(status)
    stopped_reason = str(task_summary.get("stopped_reason", "unknown"))
    if stopped_reason == "env_done" and bool(task_summary.get("done_called", False)):
        return "submitted"
    if stopped_reason == "env_done":
        return "terminated"
    if stopped_reason in {"env_error", "env_setup_error"}:
        return "environment_error"
    return stopped_reason


def slugify(name: str) -> str:
    chars: List[str] = []
    for ch in name.lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    s = "".join(chars)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "run"


def resolve_compare_ids(run_data: Dict[str, Dict[str, Dict[str, Any]]], id_mode: str) -> List[str]:
    id_sets = [set(d.keys()) for d in run_data.values()]
    if not id_sets:
        return []
    if id_mode == "shared":
        ids = set.intersection(*id_sets)
    else:
        ids = set.union(*id_sets)
    return sorted(ids)


def build_aggregate(items: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    arr = list(items.values())
    scored = [x for x in arr if not is_excluded_from_official(x)]
    n = len(scored)
    correct_count = sum(1 for x in scored if bool(eval_info(x).get("is_correct", False)))
    avg_correctness_score = (
        sum(float(eval_info(x).get("correctness_score", 0.0) or 0.0) for x in scored) / n
        if n
        else 0.0
    )
    avg_steps = sum(int(x.get("steps", 0)) for x in scored) / n if n else 0.0
    done_called_count = sum(1 for x in scored if bool(x.get("done_called", False)))
    submitted_count = sum(1 for x in arr if completion_status(x) == "submitted")

    stop_reason_counts: Dict[str, int] = {}
    completion_status_counts: Dict[str, int] = {}
    error_category_counts: Dict[str, int] = {}
    for x in arr:
        sr = str(x.get("stopped_reason", "unknown"))
        cs = completion_status(x)
        stop_reason_counts[sr] = stop_reason_counts.get(sr, 0) + 1
        completion_status_counts[cs] = completion_status_counts.get(cs, 0) + 1
        ec = x.get("error_category")
        if ec:
            category = str(ec)
            error_category_counts[category] = error_category_counts.get(category, 0) + 1

    return {
        "task_count": len(arr),
        "scored_task_count": n,
        "excluded_count": len(arr) - n,
        "correct_count": correct_count,
        "correct_rate": (correct_count / n) if n else 0.0,
        "avg_correctness_score": avg_correctness_score,
        "avg_steps": avg_steps,
        "done_called_count": done_called_count,
        "submitted_count": submitted_count,
        "stop_reason_counts": stop_reason_counts,
        "completion_status_counts": completion_status_counts,
        "error_category_counts": error_category_counts,
    }


def write_task_matrix_csv(
    out_csv: Path,
    run_order: List[str],
    run_labels: Dict[str, str],
    run_data: Dict[str, Dict[str, Dict[str, Any]]],
    instance_ids: List[str],
) -> None:
    fieldnames: List[str] = ["instance_id", "scene", "website"]
    for run_name in run_order:
        label = run_labels[run_name]
        fieldnames.extend(
            [
                f"{label}_is_correct",
                f"{label}_score",
                f"{label}_steps",
                f"{label}_completion_status",
                f"{label}_stop",
                f"{label}_error_category",
                f"{label}_done_called",
            ]
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for instance_id in instance_ids:
            row: Dict[str, Any] = {"instance_id": instance_id, "scene": "", "website": ""}

            for run_name in run_order:
                summary = run_data[run_name].get(instance_id)
                label = run_labels[run_name]
                if summary is None:
                    row[f"{label}_is_correct"] = ""
                    row[f"{label}_score"] = ""
                    row[f"{label}_steps"] = ""
                    row[f"{label}_completion_status"] = ""
                    row[f"{label}_stop"] = ""
                    row[f"{label}_error_category"] = ""
                    row[f"{label}_done_called"] = ""
                    continue

                if not row["scene"]:
                    row["scene"] = summary.get("scene", "")
                if not row["website"]:
                    row["website"] = summary.get("website", "")

                if is_excluded_from_official(summary):
                    status = str(eval_info(summary).get("eval_status") or "") or completion_status(summary)
                    row[f"{label}_is_correct"] = ""
                    row[f"{label}_score"] = ""
                    row[f"{label}_steps"] = int(summary.get("steps", 0))
                    row[f"{label}_completion_status"] = status
                    row[f"{label}_stop"] = str(summary.get("stopped_reason", ""))
                    row[f"{label}_error_category"] = str(summary.get("error_category") or "")
                    row[f"{label}_done_called"] = int(bool(summary.get("done_called", False)))
                    continue

                row[f"{label}_is_correct"] = int(bool(eval_info(summary).get("is_correct", False)))
                row[f"{label}_score"] = float(eval_info(summary).get("correctness_score", 0.0) or 0.0)
                row[f"{label}_steps"] = int(summary.get("steps", 0))
                row[f"{label}_completion_status"] = completion_status(summary)
                row[f"{label}_stop"] = str(summary.get("stopped_reason", ""))
                row[f"{label}_error_category"] = str(summary.get("error_category") or "")
                row[f"{label}_done_called"] = int(bool(summary.get("done_called", False)))

            writer.writerow(row)


def build_pairwise_score_wlt(
    run_order: List[str],
    run_data: Dict[str, Dict[str, Dict[str, Any]]],
    instance_ids: List[str],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for ra in run_order:
        out[ra] = {}
        for rb in run_order:
            if ra == rb:
                continue
            win = 0
            loss = 0
            tie = 0
            for instance_id in instance_ids:
                a = run_data[ra].get(instance_id)
                b = run_data[rb].get(instance_id)
                if a is None or b is None:
                    continue
                if is_excluded_from_official(a) or is_excluded_from_official(b):
                    continue
                score_a = float(eval_info(a).get("correctness_score", 0.0) or 0.0)
                score_b = float(eval_info(b).get("correctness_score", 0.0) or 0.0)
                if score_a > score_b:
                    win += 1
                elif score_a < score_b:
                    loss += 1
                else:
                    tie += 1
            out[ra][rb] = {"win": win, "loss": loss, "tie": tie}
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build reusable multi-run task comparison matrix")
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="run dir names under --log-dir, e.g. scene29_ds_eval_p4o_v3 scene29_ds_eval_pdsr",
    )
    p.add_argument(
        "--id-mode",
        choices=["shared", "union"],
        default="shared",
        help="shared: only ids present in all runs; union: ids present in any run",
    )
    p.add_argument("--log-dir", default="logs/runs")
    p.add_argument("--out-csv", default="reports/task_matrix.csv")
    p.add_argument("--out-json", default="reports/task_matrix_summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_order = args.runs
    run_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for run_name in run_order:
        run_dir = Path(args.log_dir) / run_name
        run_data[run_name] = load_task_summaries(run_dir)

    run_labels: Dict[str, str] = {}
    used_labels = set()
    for idx, run_name in enumerate(run_order, start=1):
        label = slugify(run_name)
        if label in used_labels:
            label = f"{label}_{idx}"
        used_labels.add(label)
        run_labels[run_name] = label

    compare_ids = resolve_compare_ids(run_data, args.id_mode)

    out_csv = Path(args.out_csv)
    write_task_matrix_csv(out_csv, run_order, run_labels, run_data, compare_ids)

    run_aggregates = {run_name: build_aggregate(run_data[run_name]) for run_name in run_order}

    union_ids = set().union(*(set(items.keys()) for items in run_data.values())) if run_data else set()
    ids_missing_per_run_vs_union = {
        run_name: sorted(union_ids - set(run_data[run_name].keys()))
        for run_name in run_order
    }

    pairwise_wlt = build_pairwise_score_wlt(run_order, run_data, compare_ids)

    payload = {
        "runs": run_order,
        "run_labels": run_labels,
        "id_mode": args.id_mode,
        "run_task_counts": {run_name: len(run_data[run_name]) for run_name in run_order},
        "compare_task_count": len(compare_ids),
        "ids_missing_per_run_vs_union": ids_missing_per_run_vs_union,
        "run_aggregates": run_aggregates,
        "pairwise_score_win_loss_tie": pairwise_wlt,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"OUT_CSV={out_csv}")
    print(f"OUT_JSON={out_json}")
    print(f"COMPARE_TASKS={len(compare_ids)}")
    for run_name in run_order:
        agg = run_aggregates[run_name]
        label = run_labels[run_name]
        print(f"RUN={run_name} LABEL={label} CORRECT={agg['correct_count']} RATE={agg['correct_rate']:.4f}")


if __name__ == "__main__":
    main()
