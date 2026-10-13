#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def classify_reason(reason: str) -> list[str]:
    text = reason.lower()
    cats: list[str] = []
    if any(k in text for k in ["no evidence", "not present", "not shown", "lacks any", "cannot be confirmed"]):
        cats.append("Missing verifiable evidence")
    if any(k in text for k in ["filtered", "filter", "list view", "single product", "product detail page"]):
        cats.append("Filter/list state not achieved")
    if any(k in text for k in ["review", "rating", "stars"]):
        cats.append("Review constraints not satisfied")
    if any(k in text for k in ["black", "leather", "size", "color", "material", "off-white"]):
        cats.append("Attribute constraints not matched")
    if any(k in text for k in ["cheapest", "lowest price", "price"]):
        cats.append("Lowest-price constraint not verified")
    if any(k in text for k in ["completion message", "done message"]):
        cats.append("Insufficient done_message evidence")
    if not cats:
        cats.append("Other")
    return cats


def short_reason(reason: str, limit: int = 180) -> str:
    s = re.sub(r"\s+", " ", reason).strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def completion_status(runtime: dict) -> str:
    status = runtime.get("completion_status")
    if status:
        return str(status)
    stopped_reason = str(runtime.get("stopped_reason", ""))
    if stopped_reason == "env_done" and bool(runtime.get("done_called", False)):
        return "submitted"
    if stopped_reason == "env_done":
        return "terminated"
    if stopped_reason in {"env_error", "env_setup_error"}:
        return "environment_error"
    return stopped_reason


def main() -> None:
    p = argparse.ArgumentParser(description="Generate detailed markdown evaluation report")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.output)

    run_summary_path = run_dir / "run_summary.json"
    if not run_summary_path.exists():
        raise SystemExit(f"run_summary.json not found: {run_summary_path}")

    eval_dir = run_dir / "evaluations"
    if not eval_dir.exists():
        raise SystemExit(f"evaluations dir not found: {eval_dir}")

    run_summary = load_json(run_summary_path)
    task_runtime = {r.get("instance_id", ""): r for r in run_summary.get("results", [])}

    eval_paths = sorted(eval_dir.glob("*.evaluation.json"))
    rows = []
    score_hist = Counter()
    fail_constraint_counter = Counter()
    reason_category_counter = Counter()
    top_fail_reasons = Counter()

    excluded_rows = []

    for ep in eval_paths:
        obj = load_json(ep)
        tid = obj.get("task_id", ep.stem.replace(".evaluation", ""))
        result = obj.get("result", {})
        eval_status = str(result.get("eval_status") or "")
        excluded = bool(result.get("exclude_from_official")) or eval_status in {
            "environment_error",
            "evaluator_error",
        }
        runtime = task_runtime.get(tid, {})
        if excluded:
            excluded_rows.append(
                {
                    "task_id": tid,
                    "eval_status": eval_status or "excluded",
                    "completion_status": completion_status(runtime),
                    "website": runtime.get("website", ""),
                }
            )
            continue
        score = float(result.get("correctness_score", 0.0) or 0.0)
        is_correct = bool(result.get("is_correct", False))
        constraints = result.get("constraint_results", []) or []
        failed_constraints = [c for c in constraints if not bool(c.get("satisfied", False))]

        score_hist[score] += 1

        categories = []
        for fc in failed_constraints:
            cid = str(fc.get("constraint_id", "unknown"))
            fail_constraint_counter[cid] += 1
            rs = str(fc.get("reason", ""))
            for cat in classify_reason(rs):
                categories.append(cat)
                reason_category_counter[cat] += 1
            top_fail_reasons[short_reason(rs, 120)] += 1

        rows.append(
            {
                "task_id": tid,
                "score": score,
                "is_correct": is_correct,
                "failed_constraints": failed_constraints,
                "steps": runtime.get("steps", 0),
                "done_called": runtime.get("done_called", False),
                "completion_status": completion_status(runtime),
                "stopped_reason": runtime.get("stopped_reason", ""),
                "error_category": runtime.get("error_category") or "",
                "scene": runtime.get("scene", ""),
                "website": runtime.get("website", ""),
                "categories": sorted(set(categories)),
            }
        )

    total = len(rows)
    passed = sum(1 for r in rows if r["is_correct"])
    failed = total - passed
    avg_score = (sum(r["score"] for r in rows) / total) if total else 0.0
    pass_rate = (passed / total * 100.0) if total else 0.0

    rows_sorted = sorted(rows, key=lambda x: (x["is_correct"], x["score"]), reverse=True)
    fails_sorted = [r for r in rows_sorted if not r["is_correct"]]

    lines: list[str] = []
    lines.append("# Detailed Evaluation Report")
    lines.append("")
    lines.append(f"- Run: `{run_summary.get('run_name', run_dir.name)}`")
    lines.append(f"- Directory: `{run_dir}`")
    lines.append(f"- Policy/Trigger/Evaluator: `{run_summary.get('policy_model')}` / `{run_summary.get('trigger_model')}` / `{run_summary.get('evaluator_model')}`")
    lines.append("")

    lines.append("## 1. Overall Results")
    lines.append("")
    lines.append(f"- Total tasks: {total + len(excluded_rows)}")
    lines.append(f"- Scored tasks: {total}")
    lines.append(f"- Excluded due to environment/evaluator issues: {len(excluded_rows)}")
    lines.append(f"- Passed tasks: {passed}")
    lines.append(f"- Failed tasks: {failed}")
    lines.append(f"- Pass rate: {pass_rate:.2f}%")
    lines.append(f"- Mean correctness_score: {avg_score:.4f}")
    lines.append(f"- Mean steps: {run_summary.get('avg_steps', 0.0):.4f}")
    lines.append(f"- done_called_count: {run_summary.get('done_called_count', 0)}")
    lines.append(f"- submitted_count: {run_summary.get('submitted_count', 0)}")
    lines.append("")

    lines.append("## 2. Score Distribution")
    lines.append("")
    for s, c in sorted(score_hist.items(), key=lambda x: x[0]):
        lines.append(f"- score={s}: {c} tasks")
    lines.append("")

    lines.append("## 3. Failure Categories (Constraint Level)")
    lines.append("")
    for cat, c in reason_category_counter.most_common():
        lines.append(f"- {cat}: {c}")
    lines.append("")

    lines.append("## 4. Most Frequently Failed Constraint IDs")
    lines.append("")
    for cid, c in fail_constraint_counter.most_common():
        lines.append(f"- {cid}: {c}")
    lines.append("")

    lines.append("## 5. Most Frequent Failure Reasons (Top 20)")
    lines.append("")
    for rs, c in top_fail_reasons.most_common(20):
        lines.append(f"- ({c}) {rs}")
    lines.append("")

    lines.append("## 6. Passed Tasks")
    lines.append("")
    pass_rows = [r for r in rows_sorted if r["is_correct"]]
    if pass_rows:
        for r in pass_rows:
            lines.append(f"- {r['task_id']} | score={r['score']:.2f} | steps={r['steps']} | done_called={r['done_called']}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## 7. Failed Task Overview")
    lines.append("")
    lines.append("| task_id | score | steps | done_called | Failed constraints | Main failure categories |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in fails_sorted:
        lines.append(
            "| {tid} | {score:.2f} | {steps} | {done} | {fcnt} | {cats} |".format(
                tid=r["task_id"],
                score=r["score"],
                steps=r["steps"],
                done=int(bool(r["done_called"])),
                fcnt=len(r["failed_constraints"]),
                cats=", ".join(r["categories"]) if r["categories"] else "-",
            )
        )
    lines.append("")

    lines.append("## 8. Detailed Failure Reasons by Task")
    lines.append("")
    for r in fails_sorted:
        lines.append(f"### {r['task_id']}")
        lines.append(f"- score: {r['score']:.2f}")
        lines.append(f"- steps: {r['steps']}")
        lines.append(f"- done_called: {r['done_called']}")
        lines.append(f"- website: {r['website']}")
        lines.append(f"- scene: {r['scene']}")
        for fc in r["failed_constraints"]:
            lines.append(f"- Constraint `{fc.get('constraint_id', 'unknown')}`: {fc.get('constraint_description', '')}")
            lines.append(f"  - Reason: {short_reason(str(fc.get('reason', '')), 360)}")
        lines.append("")

    lines.append("## 9. Excluded Tasks (Environment/Evaluator)")
    lines.append("")
    if excluded_rows:
        for r in excluded_rows:
            lines.append(
                f"- {r['task_id']} | {r['eval_status']} | website={r['website']}"
            )
    else:
        lines.append("- None")
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
