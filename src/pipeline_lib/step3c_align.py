"""Step 3c: sync constraint descriptions that still state the old requirement."""

from openai import OpenAI

from utils.llm import llm_json_call
from .models import Constraint, ConstraintResult, OverrideItem
from .prompts import STEP3C_USER


def updated_constraint_text(item: OverrideItem, override_mode: str) -> str:
    """Text of the constraint after the revision.

    Unconditional: the new requirement. Conditional: the old requirement
    still holds unless the condition selects the new one.
    """
    if override_mode == "conditional" and item.switching_condition and item.switching_condition.condition_text:
        return (
            f"{item.target_constraint_text} "
            f"If {item.switching_condition.condition_text}, {item.replacement_constraint_text}"
        )
    return item.replacement_constraint_text


def align_related_constraints(
    constraints: ConstraintResult,
    instructions: list,
    item: OverrideItem,
    override_mode: str,
    client: OpenAI,
    model: str,
) -> ConstraintResult:
    """Return a copy of the constraints with related descriptions brought in line."""
    now = updated_constraint_text(item, override_mode)
    shown = []
    for c in constraints.constraints:
        text = now if c.id == item.target_constraint_id else c.description
        shown.append(f"{c.id}. {text}")
    turns = [f"i{inst.id}: {inst.instruction}" for inst in instructions]

    user_msg = STEP3C_USER.format(
        instructions="\n".join(turns),
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

    return ConstraintResult(
        original_instruction=constraints.original_instruction,
        constraints=[Constraint(id=c.id, description=by_id[c.id]) for c in constraints.constraints],
    )
