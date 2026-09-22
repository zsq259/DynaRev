"""Step 5: Post-hoc constraint-to-instruction binding and coverage audit."""

import json

from openai import OpenAI

from utils.llm import llm_json_call
from .models import BindingEntry, BindingResult, ConstraintResult, I0Result, MultiturnResult
from .prompts import STEP5_SYSTEM, STEP5_USER


def bind_constraints(
    constraints: ConstraintResult,
    i0: I0Result,
    multiturn: MultiturnResult,
    client: OpenAI,
    model: str = "gpt-4o",
    override_context: dict | None = None,
) -> BindingResult:
    """Run Step 5: bind each constraint to the instruction that covers it."""
    constraints_json = json.dumps(
        [{"id": c.id, "description": c.description} for c in constraints.constraints],
        ensure_ascii=False,
        indent=2,
    )

    instructions_json = json.dumps(
        [
            {"id": f"i{inst.id}", "instruction": inst.instruction}
            for inst in multiturn.instructions
        ],
        ensure_ascii=False,
        indent=2,
    )

    override_context_block = ""
    if override_context:
        override_mode = override_context.get("override_mode", "unconditional")
        ctx_json = json.dumps(override_context, ensure_ascii=False, indent=2)
        if override_mode == "conditional":
            override_context_block = (
                "Override context:\n"
                f"{ctx_json}\n\n"
                "This is a conditional override (override_mode=conditional). Bind "
                "target_constraint to the conditional_override revision — that's where "
                "the dimension's outcome is decided.\n\n"
                "The replacement_constraint text (Step 3b change_to) only appears in "
                "override_context above. It is NOT a row in your constraints list, and no "
                "instruction needs to deliver it as a hard requirement. Export-time "
                "post-processing adds it to the final task as a synthetic alt-branch "
                "constraint after Step 5. Don't try to bind it. "
                "target_constraint is Step 3b keep; bind the listed constraint id to the "
                "conditional_override revision."
            )
        else:
            override_context_block = (
                "Override context:\n"
                f"{ctx_json}\n\n"
                "Binding rule: for any override item above, it is valid to bind the target constraint to the override "
                "instruction that explicitly replaces target with replacement."
            )

    user_msg = STEP5_USER.format(
        constraints=constraints_json,
        i0=i0.i0,
        instructions=instructions_json,
        override_context_block=override_context_block,
    )

    payload = llm_json_call(client, STEP5_SYSTEM, user_msg, model=model)

    mapping = [
        BindingEntry(
            constraint_id=entry["constraint_id"],
            instruction_id=entry["instruction_id"],
            confidence=entry.get("confidence", "medium"),
            reason=entry.get("reason", ""),
        )
        for entry in payload.get("mapping", [])
    ]

    return BindingResult(
        mapping=mapping,
        issues=payload.get("issues", []),
        passed=payload.get("passed", True),
        retry_from_step=payload.get("retry_from_step"),
    )
