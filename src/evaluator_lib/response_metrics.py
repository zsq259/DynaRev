"""Response correctness & response efficiency metrics (deterministic part).

Computes:
- Demand-side response correctness: RCSR_k (per-round constraint satisfaction
  rate), MFFR (mean first failure round), and per-round combine_label
  distribution.
- Response efficiency: CSD_k (constraint satisfaction density between
  consecutive injection points), aggregated mean.

These functions consume an existing answer-correctness evaluation result
(from TaskEvaluator) plus the runner's events.jsonl + summary.json — no
LLM calls. The behavior-side WRA + adaptation classification lives in
``response_judge.py``.

Window-start semantics — for each fired revision i_k we need the step at
which the agent's response began. The runner records two flavors of
injection:

- mid-task injection: ``events[fired_step].injected_revision_ids``
  contains i_k AND the action at fired_step is taken with i_k already in
  context — the response window starts at fired_step.
- post-done injection: ``events[fired_step].injected_revision_ids`` is
  empty (the runner intentionally leaves it empty for done-step
  injections, see engine.py around ``injected_revision_ids=[]`` for
  ``send_msg_to_user`` branches). The agent's first response is the next
  step, so the window starts at fired_step + 1.

i0 is the original instruction; we treat it as injected at step 1 for
window purposes (the entire trajectory before i1's injection responds to
i0).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import TaskInstance


def compute_injection_steps(
    events: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, int]:
    """Resolve the response-window-start step for each fired revision.

    Returns a dict ``{revision_id: window_start_step}``. Includes "i0"
    mapped to step 1 (response window for i0 is the whole prefix before
    i1 fires). Only fired revisions appear.
    """
    out: Dict[str, int] = {"i0": 1}

    events_by_step: Dict[int, Dict[str, Any]] = {ev["step"]: ev for ev in events}

    for diag in summary.get("milestone_diagnostics", []):
        if not diag.get("fired"):
            continue
        rev_id = diag.get("inject_revision_id")
        fired_step = diag.get("fired_step")
        if not rev_id or fired_step is None:
            continue

        ev = events_by_step.get(fired_step, {})
        injected_at_step = ev.get("injected_revision_ids") or []
        if rev_id in injected_at_step:
            window_start = fired_step
        else:
            window_start = fired_step + 1
        out[rev_id] = window_start
    return out


def compute_demand_side_metrics(
    eval_result: Dict[str, Any],
    task: TaskInstance,
) -> Dict[str, Any]:
    """RCSR per round, MFFR, and per-round combine_label distribution.

    RCSR_k = |{c ∈ C_k : combine_label==confirmed}| / |C_k|
      where C_k is constraints whose source_revision == revision_k.
      not_fired counts in the denominator (stays unsatisfied).

    MFFR = first ordinal index k (1-based: i1=1, i2=2, ...) at which any
      bound constraint failed. n+1 if everything passed; i0 failures are
      reported separately and do not count as a "round" failure.
    """
    constraint_results = eval_result.get("constraint_results", [])
    by_rev: Dict[str, List[Dict[str, Any]]] = {}
    for cr in constraint_results:
        by_rev.setdefault(cr["source_revision"], []).append(cr)

    rounds = ["i0"] + [r.revision_id for r in task.revisions]

    rcsr_per_round: Dict[str, Dict[str, Any]] = {}
    label_dist_per_round: Dict[str, Dict[str, int]] = {}
    for rev_id in rounds:
        bound = by_rev.get(rev_id, [])
        # branch_skipped / conditional_unaddressed remove themselves from
        # both numerator and denominator — they don't apply to this run.
        effective = [
            cr for cr in bound
            if cr.get("combine_label") not in ("branch_skipped", "conditional_unaddressed")
        ]
        confirmed = sum(1 for cr in effective if cr.get("combine_label") == "confirmed")
        total = len(effective)
        rcsr_per_round[rev_id] = {
            "confirmed_count": confirmed,
            "total": total,
            "rcsr": (confirmed / total) if total else None,
            "constraint_ids": [cr["constraint_id"] for cr in bound],
        }
        labels: Dict[str, int] = {}
        for cr in bound:
            lbl = cr.get("combine_label", "")
            if lbl:
                labels[lbl] = labels.get(lbl, 0) + 1
        label_dist_per_round[rev_id] = labels

    # MFFR — first revision (in i1, i2, ...) ordinal whose bound
    # constraints contain any non-confirmed effective entry. All-pass → n+1.
    revision_order = [r.revision_id for r in task.revisions]
    mffr_ordinal: Optional[int] = None
    for ord_idx, rev_id in enumerate(revision_order, start=1):
        bound = by_rev.get(rev_id, [])
        effective = [
            cr for cr in bound
            if cr.get("combine_label") not in ("branch_skipped", "conditional_unaddressed")
        ]
        if any(cr.get("combine_label") != "confirmed" for cr in effective):
            mffr_ordinal = ord_idx
            break
    if mffr_ordinal is None:
        mffr_ordinal = len(revision_order) + 1

    return {
        "rcsr_per_round": rcsr_per_round,
        "mffr": mffr_ordinal,
        "n_rounds": len(revision_order),
        "label_distribution_per_round": label_dist_per_round,
    }


def compute_csd_per_round(
    eval_result: Dict[str, Any],
    task: TaskInstance,
    injection_steps: Dict[str, int],
    total_steps: int,
) -> Dict[str, Any]:
    """CSD_k = (confirmed/|C_k|) / max(1, t_{k+1} - t_k) for fired revisions.

    Excludes i0 (no associated correction round). t_k is the response
    window start step (see ``compute_injection_steps``). t_{k+1} is the
    next fired revision's window-start, or total_steps + 1 for the last
    fired revision (so the final-round denominator counts the trailing
    steps). Only fired revisions get a CSD entry.
    """
    constraint_results = eval_result.get("constraint_results", [])
    by_rev: Dict[str, List[Dict[str, Any]]] = {}
    for cr in constraint_results:
        by_rev.setdefault(cr["source_revision"], []).append(cr)

    revision_order = [r.revision_id for r in task.revisions]
    fired_in_order: List[str] = [r for r in revision_order if r in injection_steps]
    boundaries: Dict[str, int] = {}
    for idx, rev_id in enumerate(fired_in_order):
        if idx + 1 < len(fired_in_order):
            boundaries[rev_id] = injection_steps[fired_in_order[idx + 1]]
        else:
            boundaries[rev_id] = total_steps + 1

    csd_per_round: Dict[str, Dict[str, Any]] = {}
    csd_values: List[float] = []
    for rev_id in fired_in_order:
        bound = by_rev.get(rev_id, [])
        effective = [
            cr for cr in bound
            if cr.get("combine_label") not in ("branch_skipped", "conditional_unaddressed")
        ]
        confirmed = sum(1 for cr in effective if cr.get("combine_label") == "confirmed")
        total = len(effective)
        sat_rate = (confirmed / total) if total else 0.0
        t_k = injection_steps[rev_id]
        t_next = boundaries[rev_id]
        denom = max(1, t_next - t_k)
        csd = sat_rate / denom
        csd_per_round[rev_id] = {
            "confirmed_count": confirmed,
            "total": total,
            "satisfaction_rate": sat_rate,
            "window_start_step": t_k,
            "next_window_start_step": t_next,
            "denom_steps": denom,
            "csd": csd,
        }
        csd_values.append(csd)

    return {
        "csd_per_round": csd_per_round,
        "mean_csd": (sum(csd_values) / len(csd_values)) if csd_values else None,
        "fired_revisions_in_order": fired_in_order,
    }


def slice_window(
    events: List[Dict[str, Any]],
    window_start: int,
    next_window_start: int,
    max_window: int = 3,
) -> List[Dict[str, Any]]:
    """Pick the first ``min(max_window, next_window_start - window_start)``
    events at or after ``window_start``. Returns the raw event dicts.

    Caller is responsible for deciding what to expose to the judge
    (action, url, axtree_path).
    """
    if next_window_start <= window_start:
        return []
    available = next_window_start - window_start
    cap = min(max_window, available)
    out = []
    for ev in events:
        if ev.get("step", 0) < window_start:
            continue
        if ev.get("step", 0) >= window_start + cap:
            break
        out.append(ev)
    return out
