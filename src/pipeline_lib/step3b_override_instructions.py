"""Step 3b: generate full override-aware instruction timeline."""

import json
from typing import Any, Dict

from openai import OpenAI

from utils.llm import llm_json_call
from .models import ConstraintResult, I0Result, Instruction, MultiturnResult, OverridePlanResult
from .prompts import STEP3_OVERRIDE_SYSTEM, STEP3_OVERRIDE_USER


def _timing_guidance(insertion_timing: str) -> str:
    if insertion_timing == "early":
        return (
            "Use EARLY mode: place the override revision immediately after the related "
            "preference is first introduced."
        )
    if insertion_timing == "late":
        return (
            "Use LATE mode: you may place the override revision at any natural later point "
            "in the timeline. Choose the timing freely to keep the interaction natural."
        )
    raise ValueError(f"Unknown override insertion timing: {insertion_timing}")


def _plan_to_prompt_payload(plan: OverridePlanResult) -> Dict[str, Any]:
    enriched = {
        "enable_override": plan.enable_override,
        "override_mode": plan.override_mode,
        "overrides": [],
    }

    for item in plan.overrides:
        entry: Dict[str, Any] = {
            "target_constraint_id": item.target_constraint_id,
            "target_constraint": item.target_constraint_text,
            "replacement_pair_id": item.replacement_pair_id,
            "replacement_constraint": item.replacement_constraint_text,
            "dimension": item.dimension,
        }
        if item.switching_condition is not None:
            sc = item.switching_condition
            entry["switching_condition"] = {
                "dimension": sc.dimension,
                "operator": sc.operator,
                "threshold": sc.threshold,
                "unit": sc.unit,
                "condition_text": sc.condition_text,
            }
        enriched["overrides"].append(entry)

    return enriched


def generate_override_instructions(
    constraints: ConstraintResult,
    i0: I0Result,
    plan: OverridePlanResult,
    insertion_timing: str,
    client: OpenAI,
    model: str = "gpt-4o",
    previous_step4_reason: str | None = None,
) -> MultiturnResult:
    """Generate complete instruction timeline for override mode."""
    if not plan.enable_override or not plan.overrides:
        raise ValueError("Override plan is disabled or empty")

    constraints_json = json.dumps(
        [{"id": c.id, "description": c.description} for c in constraints.constraints],
        ensure_ascii=False,
        indent=2,
    )
    override_plan_json = json.dumps(
        _plan_to_prompt_payload(plan),
        ensure_ascii=False,
        indent=2,
    )

    previous_context = ""
    if previous_step4_reason:
        previous_context = (
            "Previous review feedback:\n"
            f"{previous_step4_reason}\n"
            "Regenerate while fixing the above issues."
        )

    prompt = STEP3_OVERRIDE_USER.format(
        override_timing_guidance=_timing_guidance(insertion_timing),
        previous_step4_context=previous_context,
        retry_reflection_context="",
        original_instruction=constraints.original_instruction,
        constraints=constraints_json,
        i0=i0.i0,
        override_plan=override_plan_json,
    )

    payload = llm_json_call(
        client=client,
        system=STEP3_OVERRIDE_SYSTEM,
        user=prompt,
        model=model,
        max_tokens=2400,
    )

    instructions = []
    for item in payload.get("instructions", []):
        rtype = item.get("type", "additive")
        # When the plan is conditional, force the override revision's
        # type so runner emits the [CONDITIONAL] tag and evaluator
        # routes via Phase A branch resolution. We don't trust the LLM
        # to set "conditional_override" itself — it commonly emits
        # "override" — so coerce based on the plan.
        if (
            plan.override_mode == "conditional"
            and rtype == "override"
        ):
            rtype = "conditional_override"
        instructions.append(
            Instruction(
                id=item["id"],
                instruction=item["instruction"],
                milestone=item.get("milestone", ""),
                revision_type=rtype,
                condition_text=str(item.get("condition_text", "")).strip(),
            )
        )

    if len(instructions) <= 1:
        raise ValueError("Override instruction generation returned i0-only output")

    # Conditional mode: copy LLM-written condition_text back into the
    # plan's switching_condition, so downstream consumers (export.py,
    # branch_policy in final task JSON) get a clean LLM-authored
    # predicate instead of a template-rendered string.
    if plan.override_mode == "conditional":
        cond_inst = next(
            (
                inst for inst in instructions
                if inst.revision_type == "conditional_override" and inst.condition_text
            ),
            None,
        )
        if cond_inst is not None and plan.overrides:
            sc = plan.overrides[0].switching_condition
            if sc is not None:
                sc.condition_text = cond_inst.condition_text

    return MultiturnResult(instructions=instructions)
