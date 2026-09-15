"""Protocol failure when the agent exhausts the step budget without submitting.

These episodes must not enter TaskEvaluator (no decomposer / verifier LLM
calls). The artifact keeps the same shape as a normal evaluation so
``evaluate_response.py`` can still compute WRA; demand-side metrics see
every constraint as unsatisfied.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.models import TaskInstance

NO_FINAL_SUBMIT_LABEL = "no_final_submit"
NO_FINAL_SUBMIT_REASON = (
    "Episode ended at max_steps without a valid final submit; "
    "answer evaluator was not run."
)


def is_no_final_submit(summary: Dict[str, Any]) -> bool:
    """True when runner stopped on the step budget without submitting."""
    status = summary.get("completion_status") or summary.get("stopped_reason")
    return status == "max_steps"


def build_no_final_submit_result(
    task: TaskInstance,
    summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Zero-score evaluation result compatible with TaskEvaluator output."""
    summary = summary or {}
    fired_revisions = set(summary.get("fired_revisions") or [])
    constraint_results = [
        _constraint_result(c, fired_revisions) for c in task.constraints
    ]
    n = len(constraint_results)
    return {
        "eval_status": NO_FINAL_SUBMIT_LABEL,
        "skip_reason": "max_steps",
        "correctness_score": 0.0,
        "is_correct": False,
        "constraint_results": constraint_results,
        "revision_results": _revision_results(task, constraint_results),
        "diagnostic_counts": {NO_FINAL_SUBMIT_LABEL: n} if n else {},
        "overall_reason": (
            f"No final submit (max_steps); {n}/{n} constraints scored 0"
            if n
            else "No final submit (max_steps); no constraints"
        ),
    }


def _constraint_result(constraint, fired_revisions: set) -> Dict[str, Any]:
    source = constraint.source_revision or "i0"
    fired = source == "i0" or source in fired_revisions
    cid = f"c{constraint.id}"
    return {
        "constraint_id": cid,
        "constraint_description": constraint.description,
        "source_revision": source,
        "fired": fired,
        "satisfied": False,
        "verdict": "not_satisfied",
        "confidence": 1.0,
        "reason": NO_FINAL_SUBMIT_REASON,
        "evidence_step": None,
        "evidence_steps": [],
        "subclaims": [],
        "subclaim_resolutions": [],
        "combine_label": NO_FINAL_SUBMIT_LABEL,
        "fail_reason_code": NO_FINAL_SUBMIT_LABEL,
    }


def _revision_results(
    task: TaskInstance,
    constraint_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rev_to_cids: Dict[str, List[int]] = {"i0": list(task.i0_constraint_ids)}
    for revision in task.revisions:
        rev_to_cids[revision.revision_id] = list(revision.constraint_ids)

    cid_to_result = {r["constraint_id"]: r for r in constraint_results}
    out: List[Dict[str, Any]] = []
    for rev_id, cids in rev_to_cids.items():
        bound = [
            cid_to_result[f"c{cid}"]
            for cid in cids
            if f"c{cid}" in cid_to_result
        ]
        if not bound:
            continue
        failed = [r["constraint_id"] for r in bound]
        out.append({
            "revision_id": rev_id,
            "satisfied": False,
            "constraint_ids": [r["constraint_id"] for r in bound],
            "failed_constraints": failed,
        })
    return out
