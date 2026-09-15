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
) -> Tuple[Optional[OverridePlanResult], MultiturnResult]:
    """Route Step 3 generation based on instruction track mode.

    ``override_mode`` is only used when ``instruction_track_mode='override'``:
        - "unconditional" (default): A → B replacement, single revision.
        - "conditional": A stays as default; B activates only when the
          page-observable condition holds. Step 3a attaches a
          ``SwitchingCondition`` to each override item; Step 3b emits a
          ``conditional_override`` revision.
    """
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
        return OverridePlanResult(enable_override=False), None

    plan = build_override_plan(
        constraints=constraints,
        i0=i0,
        replacement_pairs=replacement_pairs,
        scene=scene,
        strategy=override_planning_strategy,
        client=client,
        model=model,
        override_mode=override_mode,
        switching_conditions=switching_conditions,
    )

    if not plan.enable_override or not plan.overrides:
        return plan, None

    mt = generate_override_instructions(
        constraints=constraints,
        i0=i0,
        plan=plan,
        insertion_timing=override_insertion_timing,
        client=client,
        model=model,
        previous_step4_reason=previous_step4_reason,
    )
    return plan, mt
