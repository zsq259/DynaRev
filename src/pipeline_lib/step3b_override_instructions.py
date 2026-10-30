"""Step 3b: LLM picks one candidate and writes the override timeline."""

import copy
import json
from typing import Any, Dict, Optional

from openai import OpenAI

from ..utils.llm import llm_json_call
from .models import (
    ConstraintResult,
    I0Result,
    Instruction,
    MultiturnResult,
    OverrideItem,
    OverridePlanResult,
    SwitchingCondition,
)
from .prompts import (
    STEP3_OVERRIDE_CONDITIONAL_BLOCK,
    STEP3_OVERRIDE_SYSTEM,
    STEP3_OVERRIDE_UNCONDITIONAL_BLOCK,
    STEP3_OVERRIDE_USER,
)
from .step3a_override_plan import _normalize, ordered_pair_values

SWITCHING_QUANTITIES = {"count", "price", "rating", "time"}
SWITCHING_OPERATORS = {">", "<"}


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


def _candidates_payload(plan: OverridePlanResult) -> Dict[str, Any]:
    return {
        "candidates": [
            {
                "target_constraint_id": item.target_constraint_id,
                "anchor_constraint": item.target_constraint_text,
                "replacement_pair_id": item.replacement_pair_id,
                "pair_dimension": item.dimension,
                "pair_values": ordered_pair_values(item),
            }
            for item in plan.candidates
        ],
    }


def _normalize_pair_id(pair_id: Any) -> Optional[str]:
    if pair_id is None:
        return None
    if isinstance(pair_id, str):
        text = pair_id.strip()
        if not text or text.lower() == "null":
            return None
        return text
    return None


def _build_proposed_item(
    constraints: ConstraintResult,
    selected: Dict[str, Any],
    constraint_id: Any,
) -> OverrideItem:
    if not isinstance(constraint_id, int):
        raise ValueError("Proposed override requires target_constraint_id")
    texts = {c.id: c.description for c in constraints.constraints}
    target = texts.get(constraint_id, "")
    if not target:
        raise ValueError("Proposed override target_constraint_id is not in the constraint list")
    return OverrideItem(
        target_constraint_id=constraint_id,
        replacement_pair_id="proposed",
        target_constraint_text=target,
        replacement_constraint_text="",
        dimension=str(selected.get("pair_dimension") or "").strip(),
        score=0.0,
    )


def _apply_keep_change(chosen: OverrideItem, selected: Dict[str, Any]) -> Optional[OverrideItem]:
    """Write 3b keep/change_to onto the item. None = no-op / tautology."""
    keep = str(selected.get("keep") or "").strip()
    change_to = str(
        selected.get("change_to") or selected.get("proposed_replacement") or ""
    ).strip()
    if not keep or not change_to:
        raise ValueError("selected.keep and selected.change_to are required")
    if _normalize(keep) == _normalize(change_to):
        return None
    chosen.target_constraint_text = keep
    chosen.replacement_constraint_text = change_to
    return chosen


def _resolve_selected_item(
    plan: OverridePlanResult,
    constraints: ConstraintResult,
    selected: Dict[str, Any],
) -> OverrideItem:
    constraint_id = selected.get("target_constraint_id")
    if isinstance(constraint_id, str) and constraint_id.strip().lstrip("-").isdigit():
        constraint_id = int(constraint_id.strip())
    pair_id = _normalize_pair_id(selected.get("replacement_pair_id"))
    if plan.candidates:
        if pair_id is None:
            raise ValueError("Must copy replacement_pair_id from a candidate row")
        chosen = _find_candidate(plan, constraint_id, pair_id)
        if chosen is None:
            raise ValueError("Override instruction generation selected a pair that is not a candidate")
        return copy.copy(chosen)
    return _build_proposed_item(constraints, selected, constraint_id)


def _find_candidate(
    plan: OverridePlanResult,
    constraint_id: Any,
    pair_id: Any,
) -> Optional[OverrideItem]:
    if not isinstance(constraint_id, int) or not isinstance(pair_id, str) or not pair_id:
        return None
    for item in plan.candidates:
        if item.target_constraint_id == constraint_id and item.replacement_pair_id == pair_id:
            return item
    return None


def _parse_threshold(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _attach_conditional_choice(
    chosen: OverrideItem,
    selected: Dict[str, Any],
    condition_text: str,
) -> None:
    quantity = str(selected.get("quantity") or "").strip().lower()
    operator = str(selected.get("operator") or "").strip()
    threshold = _parse_threshold(selected.get("threshold"))
    unit = str(selected.get("unit") or "").strip()
    if quantity not in SWITCHING_QUANTITIES:
        raise ValueError(f"Invalid switching quantity: {quantity!r}")
    if operator not in SWITCHING_OPERATORS:
        raise ValueError(f"Invalid switching operator: {operator!r}")
    if threshold is None:
        raise ValueError("Conditional override requires a numeric threshold")
    if not condition_text:
        raise ValueError("Conditional override requires condition_text")
    chosen.switching_condition = SwitchingCondition(
        dimension=quantity,
        operator=operator,
        threshold=threshold,
        unit=unit,
        default_option=chosen.target_constraint_text,
        alt_option=chosen.replacement_constraint_text,
        condition_text=condition_text,
    )


def generate_override_instructions(
    constraints: ConstraintResult,
    i0: I0Result,
    plan: OverridePlanResult,
    insertion_timing: str,
    client: OpenAI,
    model: str = "gpt-4o",
    previous_step4_reason: str | None = None,
    website: str = "",
) -> Optional[MultiturnResult]:
    """Pick one candidate, or propose B when the candidate list is empty."""
    constraints_json = json.dumps(
        [{"id": c.id, "description": c.description} for c in constraints.constraints],
        ensure_ascii=False,
        indent=2,
    )
    candidates_json = json.dumps(
        _candidates_payload(plan),
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

    if plan.override_mode == "conditional":
        mode_block = STEP3_OVERRIDE_CONDITIONAL_BLOCK
    else:
        mode_block = STEP3_OVERRIDE_UNCONDITIONAL_BLOCK

    prompt = STEP3_OVERRIDE_USER.format(
        override_timing_guidance=_timing_guidance(insertion_timing),
        previous_step4_context=previous_context,
        original_instruction=constraints.original_instruction,
        website=website or "(not provided)",
        constraints=constraints_json,
        i0=i0.i0,
        override_candidates=candidates_json,
        mode_block=mode_block,
    )

    payload = llm_json_call(
        client=client,
        system=STEP3_OVERRIDE_SYSTEM,
        user=prompt,
        model=model,
        max_tokens=3200,
    )

    if payload.get("enable_override") is False:
        plan.enable_override = False
        plan.overrides = []
        return None

    selected = payload.get("selected") or {}
    if not isinstance(selected, dict):
        raise ValueError("Override instruction generation returned invalid selected object")

    chosen = _resolve_selected_item(plan, constraints, selected)
    chosen = _apply_keep_change(chosen, selected)
    if chosen is None:
        plan.enable_override = False
        plan.overrides = []
        return None

    raw_instructions = payload.get("instructions", [])
    if not isinstance(raw_instructions, list) or len(raw_instructions) <= 1:
        raise ValueError("Override instruction generation returned i0-only output")

    instructions = []
    for item in raw_instructions:
        rtype = item.get("type", "additive")
        condition_text = str(item.get("condition_text", "")).strip()
        if plan.override_mode == "conditional" and rtype == "override" and condition_text:
            rtype = "conditional_override"
        instructions.append(
            Instruction(
                id=item["id"],
                instruction=item["instruction"],
                milestone=item.get("milestone", ""),
                revision_type=rtype,
                condition_text=condition_text,
            )
        )

    if plan.override_mode == "conditional":
        cond_inst = next(
            (
                inst for inst in instructions
                if inst.revision_type == "conditional_override" and inst.condition_text
            ),
            None,
        )
        if cond_inst is None:
            raise ValueError(
                "Conditional override generation did not produce a conditional_override "
                "turn with condition_text"
            )
        _attach_conditional_choice(chosen, selected, cond_inst.condition_text)
        if any(inst.revision_type == "override" for inst in instructions):
            raise ValueError("Conditional track emitted a flat override revision")
    else:
        if not any(inst.revision_type == "override" for inst in instructions):
            raise ValueError("Unconditional override generation did not produce an override turn")
        chosen.switching_condition = None

    plan.enable_override = True
    plan.overrides = [chosen]
    return MultiturnResult(instructions=instructions)
