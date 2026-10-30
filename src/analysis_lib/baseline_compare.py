#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def classify_reason(reason: str) -> List[str]:
    text = (reason or "").lower()
    cats: List[str] = []
    if any(k in text for k in ["completion message", "done message"]):
        cats.append("done_message evidence issues")
    if any(k in text for k in ["no evidence", "cannot be verified", "not present", "not shown", "lacks"]):
        cats.append("Missing verifiable evidence")
    if any(k in text for k in ["filtered", "filter", "list view", "single product"]):
        cats.append("Filter/list state not achieved")
    if any(k in text for k in ["review", "rating", "stars"]):
        cats.append("Review constraints not satisfied")
    if any(k in text for k in ["black", "leather", "size", "color", "material", "off-white"]):
        cats.append("Attribute constraints not matched")
    if any(k in text for k in ["cheapest", "price", "lowest"]):
        cats.append("Lowest-price/price constraints not verified")
    if not cats:
        cats.append("Other")
    return cats


def short_text(s: str, n: int = 180) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 3] + "..."


def load_run(run_dir: Path) -> Dict[str, Any]:
    summary = load_json(run_dir / "run_summary.json")
    eval_dir = run_dir / "evaluations"

    runtime_by_task: Dict[str, Dict[str, Any]] = {}
    for r in summary.get("results", []) or []:
        runtime_by_task[str(r.get("instance_id", ""))] = r

    eval_by_task: Dict[str, Dict[str, Any]] = {}
    if eval_dir.exists():
        for p in sorted(eval_dir.glob("*.evaluation.json")):
            obj = load_json(p)
            tid = str(obj.get("task_id", p.stem.replace(".evaluation", "")))
            eval_by_task[tid] = obj

    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "runtime_by_task": runtime_by_task,
        "eval_by_task": eval_by_task,
    }


def metrics_block(data: Dict[str, Any]) -> Dict[str, Any]:
    summary = data["summary"]
    eval_by_task = data["eval_by_task"]
    scores: List[float] = []
    correct = 0
    for e in eval_by_task.values():
        res = e.get("result", {})
        s = float(res.get("correctness_score", 0.0) or 0.0)
        scores.append(s)
        if bool(res.get("is_correct", False)):
            correct += 1
    total = len(eval_by_task)
    hist = Counter(scores)
    return {
        "task_count": int(summary.get("task_count", 0) or 0),
        "evaluated_count": total,
        "correct_count": correct,
        "correct_rate": (correct / total) if total else 0.0,
        "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
        "avg_steps": float(summary.get("avg_steps", 0.0) or 0.0),
        "done_called_count": int(summary.get("done_called_count", 0) or 0),
        "score_hist": dict(sorted(hist.items(), key=lambda x: x[0])),
    }


def collect_reason_stats(eval_by_task: Dict[str, Dict[str, Any]]) -> Dict[str, Counter]:
    cat_counter: Counter = Counter()
    cid_counter: Counter = Counter()
    reason_counter: Counter = Counter()
    for e in eval_by_task.values():
        res = e.get("result", {})
        for c in (res.get("constraint_results", []) or []):
            if bool(c.get("satisfied", False)):
                continue
            cid = str(c.get("constraint_id", "unknown"))
            cid_counter[cid] += 1
            rs = str(c.get("reason", ""))
            reason_counter[short_text(rs, 140)] += 1
            for cat in classify_reason(rs):
                cat_counter[cat] += 1
    return {
        "cat": cat_counter,
        "cid": cid_counter,
        "reason": reason_counter,
    }


def build_task_universe(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    s = set(a["runtime_by_task"].keys()) | set(b["runtime_by_task"].keys()) | set(a["eval_by_task"].keys()) | set(b["eval_by_task"].keys())
    return sorted(s)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a detailed baseline comparison markdown report")
    p.add_argument("--run-a", required=True, help="run directory for baseline A")
    p.add_argument("--run-b", required=True, help="run directory for baseline B")
    p.add_argument("--label-a", default="gpt-4o")
    p.add_argument("--label-b", default="deepseek-v4-flash")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    run_a = load_run(Path(args.run_a))
    run_b = load_run(Path(args.run_b))
    m_a = metrics_block(run_a)
    m_b = metrics_block(run_b)
    rs_a = collect_reason_stats(run_a["eval_by_task"])
    rs_b = collect_reason_stats(run_b["eval_by_task"])

    tasks = build_task_universe(run_a, run_b)

    pass_a = {tid for tid, e in run_a["eval_by_task"].items() if bool((e.get("result") or {}).get("is_correct", False))}
    pass_b = {tid for tid, e in run_b["eval_by_task"].items() if bool((e.get("result") or {}).get("is_correct", False))}

    lines: List[str] = []
    lines.append("# Detailed Evaluation Comparison Report")
    lines.append("")
    lines.append("## 1. Execution and Evaluation Settings")
    lines.append("")
    lines.append(f"- Group A: {args.label_a}")
    lines.append(f"- Group B: {args.label_b}")
    lines.append(f"- Directory A: {args.run_a}")
    lines.append(f"- Directory B: {args.run_b}")
    lines.append(f"- A evaluator_model: {run_a['summary'].get('evaluator_model', '')}")
    lines.append(f"- B evaluator_model: {run_b['summary'].get('evaluator_model', '')}")
    lines.append("")

    lines.append("## 2. Overall Metric Comparison")
    lines.append("")
    lines.append("| Metric | Group A | Group B | Difference (B-A) |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| task_count | {m_a['task_count']} | {m_b['task_count']} | {m_b['task_count'] - m_a['task_count']} |")
    lines.append(f"| evaluated_count | {m_a['evaluated_count']} | {m_b['evaluated_count']} | {m_b['evaluated_count'] - m_a['evaluated_count']} |")
    lines.append(f"| correct_count | {m_a['correct_count']} | {m_b['correct_count']} | {m_b['correct_count'] - m_a['correct_count']} |")
    lines.append(f"| correct_rate | {m_a['correct_rate']*100:.2f}% | {m_b['correct_rate']*100:.2f}% | {(m_b['correct_rate']-m_a['correct_rate'])*100:.2f}pp |")
    lines.append(f"| avg_correctness_score | {m_a['avg_score']:.4f} | {m_b['avg_score']:.4f} | {m_b['avg_score']-m_a['avg_score']:+.4f} |")
    lines.append(f"| avg_steps | {m_a['avg_steps']:.4f} | {m_b['avg_steps']:.4f} | {m_b['avg_steps']-m_a['avg_steps']:+.4f} |")
    lines.append(f"| done_called_count | {m_a['done_called_count']} | {m_b['done_called_count']} | {m_b['done_called_count']-m_a['done_called_count']} |")
    lines.append("")

    lines.append("## 3. Passed Task Set Comparison")
    lines.append("")
    lines.append(f"- Group APassed tasks: {len(pass_a)}")
    lines.append(f"- Group BPassed tasks: {len(pass_b)}")
    lines.append(f"- Passed by both groups: {len(pass_a & pass_b)}")
    lines.append(f"- Passed only by A: {len(pass_a - pass_b)}")
    lines.append(f"- Passed only by B: {len(pass_b - pass_a)}")
    lines.append(f"- Tasks passed by both groups: {', '.join(sorted(pass_a & pass_b)) if (pass_a & pass_b) else 'None'}")
    lines.append(f"- Tasks passed only by A: {', '.join(sorted(pass_a - pass_b)) if (pass_a - pass_b) else 'None'}")
    lines.append(f"- Tasks passed only by B: {', '.join(sorted(pass_b - pass_a)) if (pass_b - pass_a) else 'None'}")
    lines.append("")

    lines.append("## 4. score Distribution Comparison")
    lines.append("")
    lines.append("| score | Group A count | Group B count |")
    lines.append("|---:|---:|---:|")
    all_scores = sorted(set(m_a["score_hist"].keys()) | set(m_b["score_hist"].keys()))
    for s in all_scores:
        lines.append(f"| {s} | {m_a['score_hist'].get(s, 0)} | {m_b['score_hist'].get(s, 0)} |")
    lines.append("")

    lines.append("## 5. Failure Category Comparison (Constraint Level)")
    lines.append("")
    lines.append("| Category | Group A | Group B | Difference (B-A) |")
    lines.append("|---|---:|---:|---:|")
    all_cats = sorted(set(rs_a["cat"].keys()) | set(rs_b["cat"].keys()))
    for c in all_cats:
        a = rs_a["cat"].get(c, 0)
        b = rs_b["cat"].get(c, 0)
        lines.append(f"| {c} | {a} | {b} | {b-a:+d} |")
    lines.append("")

    lines.append("## 6. Failed Constraint ID Comparison (Constraint Level)")
    lines.append("")
    lines.append("| constraint_id | Group A | Group B | Difference (B-A) |")
    lines.append("|---|---:|---:|---:|")
    all_cids = sorted(set(rs_a["cid"].keys()) | set(rs_b["cid"].keys()))
    for cid in all_cids:
        a = rs_a["cid"].get(cid, 0)
        b = rs_b["cid"].get(cid, 0)
        lines.append(f"| {cid} | {a} | {b} | {b-a:+d} |")
    lines.append("")

    lines.append("## 7. Detailed Task Comparison")
    lines.append("")
    lines.append("| task_id | A_score | B_score | delta(B-A) | A_pass | B_pass | A_steps | B_steps | A_done | B_done |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tid in tasks:
        ea = run_a["eval_by_task"].get(tid, {})
        eb = run_b["eval_by_task"].get(tid, {})
        ra = run_a["runtime_by_task"].get(tid, {})
        rb = run_b["runtime_by_task"].get(tid, {})
        sa = float(((ea.get("result") or {}).get("correctness_score", 0.0) or 0.0))
        sb = float(((eb.get("result") or {}).get("correctness_score", 0.0) or 0.0))
        pa = int(bool((ea.get("result") or {}).get("is_correct", False)))
        pb = int(bool((eb.get("result") or {}).get("is_correct", False)))
        lines.append(
            f"| {tid} | {sa:.2f} | {sb:.2f} | {sb-sa:+.2f} | {pa} | {pb} | {int(ra.get('steps', 0) or 0)} | {int(rb.get('steps', 0) or 0)} | {int(bool(ra.get('done_called', False)))} | {int(bool(rb.get('done_called', False)))} |"
        )
    lines.append("")

    lines.append("## 8. Task Failure Details (Side by Side)")
    lines.append("")
    for tid in tasks:
        ea = run_a["eval_by_task"].get(tid, {})
        eb = run_b["eval_by_task"].get(tid, {})
        ra = run_a["runtime_by_task"].get(tid, {})
        rb = run_b["runtime_by_task"].get(tid, {})
        resa = ea.get("result", {})
        resb = eb.get("result", {})
        sa = float(resa.get("correctness_score", 0.0) or 0.0)
        sb = float(resb.get("correctness_score", 0.0) or 0.0)
        lines.append(f"### {tid}")
        lines.append(f"- A: score={sa:.2f}, pass={bool(resa.get('is_correct', False))}, steps={int(ra.get('steps', 0) or 0)}, done_called={bool(ra.get('done_called', False))}")
        lines.append(f"- B: score={sb:.2f}, pass={bool(resb.get('is_correct', False))}, steps={int(rb.get('steps', 0) or 0)}, done_called={bool(rb.get('done_called', False))}")

        fca = [c for c in (resa.get("constraint_results", []) or []) if not bool(c.get("satisfied", False))]
        fcb = [c for c in (resb.get("constraint_results", []) or []) if not bool(c.get("satisfied", False))]

        lines.append("- A Failed constraints:")
        if fca:
            for c in fca:
                lines.append(f"  - {c.get('constraint_id', 'unknown')}: {short_text(str(c.get('constraint_description', '')), 120)}")
                lines.append(f"    - reason: {short_text(str(c.get('reason', '')), 220)}")
        else:
            lines.append("  - None")

        lines.append("- B Failed constraints:")
        if fcb:
            for c in fcb:
                lines.append(f"  - {c.get('constraint_id', 'unknown')}: {short_text(str(c.get('constraint_description', '')), 120)}")
                lines.append(f"    - reason: {short_text(str(c.get('reason', '')), 220)}")
        else:
            lines.append("  - None")
        lines.append("")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
