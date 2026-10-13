"""WRA + adaptation-mode judge for response correctness behavior side.

A single LLM call per fired revision returns:
- ``wra``: 1 if the agent's first ~3 actions after the instruction's
  injection clearly act on the new instruction's intent; 0 otherwise.
- ``adaptation``: ``"overwrite"`` (back out / re-navigate / restart),
  ``"additive"`` (extend on the same page / result in place), or
  ``"unrelated"`` (ignored the instruction). The label is informational
  per metrics.md §Response Efficiency / Adaptation Mode Classification.

The judge sees the new instruction text, a compact rendering of the
window's actions + URLs, and (optionally) a one-line page label per
step. We deliberately do NOT feed the answer-side decompose output —
keeps the response judgment decoupled from final-answer judgments
(metrics.md §Metric Complementarity cross-analysis only works if the two signals
are independently computed).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..utils.llm import llm_json_call
from .errors import evaluator_error


_VALID_ADAPTATIONS = {"overwrite", "additive", "unrelated"}


def _format_window(window_events: List[Dict[str, Any]]) -> str:
    """One line per step: ``step N | URL | ACTION``. Action truncated."""
    if not window_events:
        return "(empty window — no agent actions occurred after injection)"
    lines = []
    for ev in window_events:
        step = ev.get("step")
        url = ev.get("url", "") or ""
        action = ev.get("action", "") or ""
        if len(action) > 220:
            action = action[:217] + "..."
        if len(url) > 120:
            url = url[:117] + "..."
        lines.append(f"step {step} | {url} | {action}")
    return "\n".join(lines)


def judge_response_window(
    instruction_id: str,
    instruction_text: str,
    window_events: List[Dict[str, Any]],
    client: OpenAI,
    model: str = "deepseek-v4-flash",
    execution_date: str = "",
    pre_injection_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Judge whether the action window responds to the instruction.

    ``pre_injection_url`` is the agent's URL just before the instruction
    fired (events[fired_step - 1].url). It anchors the "overwrite vs
    additive" decision: backing out from this URL = overwrite; staying
    on / extending it = additive.

    Returns ``{wra, adaptation, reason, confidence}``. Evaluator failures
    raise instead of being recorded as poor agent adaptation.
    """
    system_msg = (
        "You judge whether a GUI agent's actions in a short window (1-3 "
        "steps) responded to a specific user instruction that was just "
        "injected. The instruction is one of a sequence of corrections "
        "the user gave the agent during execution.\n\n"
        "Decide TWO things, independently:\n\n"
        "1. wra ∈ {0, 1}\n"
        "   wra=1 — at least one action in the window concretely advances "
        "the new instruction's intent (issuing a relevant search, "
        "navigating to a relevant page, filtering / refining results to "
        "match the new constraint, opening a candidate that satisfies "
        "the new requirement, etc.).\n"
        "   wra=0 — none of the actions clearly serve the new "
        "instruction. Continuing prior unrelated work, looping on the "
        "same failed action, idling, or making changes orthogonal to "
        "the instruction all count as wra=0.\n"
        "   Be lenient on partial / exploratory progress (clicking a "
        "filter that partly addresses the new requirement is wra=1) "
        "but strict on irrelevance (any-website search that ignores the "
        "instruction's specifics is wra=0).\n\n"
        "2. adaptation ∈ {overwrite, additive, unrelated}\n"
        "   overwrite — the agent abandons the page / result it was on "
        "before injection and re-navigates from scratch (back to a "
        "search engine, a different site, or a fresh query). "
        "Characterized by goto() to a new domain, opening a new search, "
        "or discarding the current candidate.\n"
        "   additive — the agent extends or refines the work in place: "
        "applies a filter, scrolls, opens a sub-link of the current "
        "result, edits the existing query, etc. The previous progress "
        "is preserved.\n"
        "   unrelated — actions don't engage with the instruction at "
        "all (e.g. re-issuing a done call, looping on the same prior "
        "click, navigating to something orthogonal). Use this label "
        "freely when wra=0 and the actions don't fit overwrite/additive.\n\n"
        "RULES:\n"
        "- Judge only on the actions / URLs given. Do not assume the "
        "agent will do something later.\n"
        "- The instruction text is verbatim from the user. Read it "
        "carefully — 'exactly three' means the agent should be acting "
        "on the count requirement, not anything else in the prompt.\n"
        "- A done call inside the window is not a response by itself "
        "unless the prior actions already satisfied the instruction.\n"
        "- Reasoning must reference specific step actions / URLs.\n\n"
        "Output JSON only: "
        '{"wra": 0|1, "adaptation": "overwrite"|"additive"|"unrelated", '
        '"reason": "...", "confidence": 0.0-1.0}'
    )

    date_line = (
        f"\nTask was executed on: {execution_date}\n" if execution_date else ""
    )
    prior_line = (
        f"Agent URL just before injection: {pre_injection_url}\n"
        if pre_injection_url
        else ""
    )
    user_prompt = (
        f"Injected instruction ({instruction_id}):\n{instruction_text}\n"
        f"{date_line}\n"
        f"{prior_line}"
        f"Action window after injection (chronological):\n"
        f"{_format_window(window_events)}\n\n"
        "Judge wra and adaptation."
    )

    try:
        payload = llm_json_call(
            client=client,
            system=system_msg,
            user=user_prompt,
            model=model,
            max_tokens=600,
        )
    except Exception as exc:
        raise evaluator_error("response-window verification failed", exc) from exc

    wra_raw = payload.get("wra", 0)
    try:
        wra = 1 if int(wra_raw) == 1 else 0
    except (TypeError, ValueError):
        wra = 1 if str(wra_raw).strip().lower() in ("1", "true", "yes") else 0

    adaptation = str(payload.get("adaptation", "unrelated")).strip().lower()
    if adaptation not in _VALID_ADAPTATIONS:
        adaptation = "unrelated"

    return {
        "wra": wra,
        "adaptation": adaptation,
        "reason": str(payload.get("reason", "")).strip(),
        "confidence": float(payload.get("confidence", 0.0)),
    }
