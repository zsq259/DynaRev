"""Step 3c: sync related constraints after an override.

Unconditional: rewrite related descriptions to the new requirement.
Conditional: split related constraints into default / alternative groups.
"""

from openai import OpenAI

from ..utils.llm import llm_json_call
from .models import (
    AlignedConstraint,
    AlignResult,
    ConstraintResult,
    OverrideItem,
)
from .prompts import STEP3C_CONDITIONAL_USER, STEP3C_USER


def align_related_constraints(
    constraints: ConstraintResult,
    instructions: list,
    item: OverrideItem,
    override_mode: str,
    client: OpenAI,
    model: str,
) -> AlignResult:
    """Return constraints after Step 3c alignment."""
    if override_mode == "conditional":
        return _align_conditional(constraints, instructions, item, client, model)
    return _align_unconditional(constraints, instructions, item, client, model)


def _turns(instructions: list) -> str:
    return "\n".join(f"i{inst.id}: {inst.instruction}" for inst in instructions)


def _align_unconditional(
    constraints: ConstraintResult,
    instructions: list,
    item: OverrideItem,
    client: OpenAI,
    model: str,
) -> AlignResult:
    now = item.replacement_constraint_text
    shown = []
    for c in constraints.constraints:
        text = now if c.id == item.target_constraint_id else c.description
        shown.append(f"{c.id}. {text}")

    user_msg = STEP3C_USER.format(
        instructions=_turns(instructions),
        constraints="\n".join(shown),
        target_id=item.target_constraint_id,
        keep=item.target_constraint_text,
        now=now,
    )
    payload = llm_json_call(client, "", user_msg, model=model)

    by_id = {c.id: c.description for c in constraints.constraints}
    by_id[item.target_constraint_id] = now
    for edit in payload.get("edits") or []:
        cid = edit.get("id")
        desc = (edit.get("description") or "").strip()
        if cid == item.target_constraint_id or cid not in by_id or not desc:
            continue
        by_id[cid] = desc

    return AlignResult(
        original_instruction=constraints.original_instruction,
        constraints=[
            AlignedConstraint(id=c.id, description=by_id[c.id])
            for c in constraints.constraints
        ],
    )


def _align_conditional(
    constraints: ConstraintResult,
    instructions: list,
    item: OverrideItem,
    client: OpenAI,
    model: str,
) -> AlignResult:
    keep = item.target_constraint_text
    change_to = item.replacement_constraint_text
    condition = ""
    if item.switching_condition is not None:
        condition = item.switching_condition.condition_text or ""

    shown = []
    for c in constraints.constraints:
        if c.id == item.target_constraint_id:
            shown.append(f"{c.id}. {keep}")
        else:
            shown.append(f"{c.id}. {c.description}")

    user_msg = STEP3C_CONDITIONAL_USER.format(
        instructions=_turns(instructions),
        constraints="\n".join(shown),
        target_id=item.target_constraint_id,
        keep=keep,
        change_to=change_to,
        condition=condition,
    )
    payload = llm_json_call(client, "", user_msg, model=model)

    next_id = max((c.id for c in constraints.constraints), default=0) + 1
    out: list[AlignedConstraint] = []
    split_ids = set()

    for c in constraints.constraints:
        if c.id == item.target_constraint_id:
            out.append(AlignedConstraint(id=c.id, description=keep, branch="default"))
            continue
        out.append(AlignedConstraint(id=c.id, description=c.description))

    # Target alternative
    out.append(
        AlignedConstraint(id=next_id, description=change_to, branch="alternative")
    )
    next_id += 1

    for split in payload.get("splits") or []:
        cid = split.get("id")
        default_text = (split.get("default") or "").strip()
        alt_text = (split.get("alternative") or "").strip()
        if (
            cid is None
            or cid == item.target_constraint_id
            or cid in split_ids
            or not default_text
            or not alt_text
        ):
            continue
        if not any(c.id == cid for c in constraints.constraints):
            continue
        split_ids.add(cid)
        for row in out:
            if row.id == cid:
                row.description = default_text
                row.branch = "default"
                break
        out.append(
            AlignedConstraint(id=next_id, description=alt_text, branch="alternative")
        )
        next_id += 1

    return AlignResult(
        original_instruction=constraints.original_instruction,
        constraints=out,
    )
