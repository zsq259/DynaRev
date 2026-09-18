"""Step 3 routing for base vs override instruction tracks."""

from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from .models import ConstraintResult, I0Result, MultiturnResult, OverridePlanResult
from .step3_multiturn import design_multiturn
from .step3a_override_plan import build_override_plan
from .step3b_override_instructions import generate_override_instructions


def run_step3(
    constraints: ConstraintResult,
    i0: I0Result,
    client: OpenAI,
    model: str,
    instruction_track_mode: str = "base",
    override_planning_strategy: str = "rule_based",
    override_insertion_timing: str = "early",
    replacement_pairs: Optional[Dict[str, Any]] = None,
    scene: str = "",
    previous_step4_reason: str | None = None,
    attempt_history: Optional[list] = None,
    override_mode: str = "unconditional",
    switching_conditions: Optional[Dict[str, Any]] = None,
    website: str = "",
) -> Tuple[Optional[OverridePlanResult], Optional[MultiturnResult]]:
    """Route Step 3 generation based on instruction track mode.

    ``override_mode`` is only used when ``instruction_track_mode='override'``:
        - "unconditional" (default): Step 3b picks one candidate and writes
          a flat A → B replacement.
        - "conditional": Step 3b picks one candidate plus a page-observable
          (quantity) ≷ N condition and writes a ``conditional_override``.

    ``override_planning_strategy`` and ``switching_conditions`` are unused;
    kept so existing callers do not break.
    """
    del override_planning_strategy
    del switching_conditions

    if instruction_track_mode == "base":
        mt = design_multiturn(
            constraints,
            i0,
            client,
            model,
            previous_step4_reason=previous_step4_reason,
            attempt_history=attempt_history,
        )
        return None, mt

    if instruction_track_mode != "override":
        raise ValueError(f"Unknown instruction_track_mode: {instruction_track_mode}")

    if not replacement_pairs:
        return OverridePlanResult(enable_override=False, override_mode=override_mode), None

    plan = build_override_plan(
        constraints=constraints,
        i0=i0,
        replacement_pairs=replacement_pairs,
        scene=scene,
        override_mode=override_mode,
    )

    if not plan.candidates:
        plan.enable_override = False
        return plan, None

    mt = generate_override_instructions(
        constraints=constraints,
        i0=i0,
        plan=plan,
        insertion_timing=override_insertion_timing,
        client=client,
        model=model,
        previous_step4_reason=previous_step4_reason,
        website=website,
    )
    if mt is None or not plan.enable_override or not plan.overrides:
        return plan, None
    return plan, mt
