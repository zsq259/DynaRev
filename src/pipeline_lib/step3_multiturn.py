"""Step 3: Arrange constraints into multi-turn instructions."""

import json

from openai import OpenAI

from ..utils.llm import llm_json_call
from .models import ConstraintResult, I0Result, Instruction, MultiturnResult
from .prompts import STEP3_SYSTEM, STEP3_USER, STEP3_RETRY_REFLECTION


def _format_attempt_history(attempt_history: list[dict]) -> str:
    lines = []
    for i, att in enumerate(attempt_history, start=1):
        lines.append(f"### Attempt {i}")
        lines.append(f"I0: {att.get('i0', '')}")
        lines.append("Instructions + milestones:")
        for inst in att.get("instructions", []):
            m = inst.get("milestone") or "(i0, no milestone)"
            lines.append(f"  i{inst['id']}: {inst['instruction']}")
            lines.append(f"       milestone: {m}")
        lines.append(f"Reviewer critique: {att.get('critique', '')}")
        lines.append("")
    return "\n".join(lines)


def design_multiturn(
    constraints: ConstraintResult,
    i0: I0Result,
    client: OpenAI,
    model: str = "gpt-4o",
    previous_step4_reason: str | None = None,
    attempt_history: list[dict] | None = None,
) -> MultiturnResult:
    constraints_json = json.dumps(
        [{"id": c.id, "description": c.description} for c in constraints.constraints],
        ensure_ascii=False,
        indent=2,
    )

    # Build context about previous failure (if retrying)
    previous_context = ""
    if previous_step4_reason and not attempt_history:
        previous_context = (
            f"**Previous attempt feedback:** {previous_step4_reason}\n\n"
            "Please address the issues identified above in your redesign of the multi-turn sequence."
        )

    retry_reflection_context = ""
    if attempt_history:
        retry_reflection_context = STEP3_RETRY_REFLECTION.format(
            n_attempts=len(attempt_history),
            history_block=_format_attempt_history(attempt_history),
        )

    user_msg = STEP3_USER.format(
        original_instruction=constraints.original_instruction,
        constraints=constraints_json,
        i0=i0.i0,
        previous_step4_context=previous_context,
        retry_reflection_context=retry_reflection_context,
    )
    payload = llm_json_call(client, STEP3_SYSTEM, user_msg, model=model)

    # Guardrail: some model outputs still collapse to a single i0-only instruction.
    # Do one targeted retry asking explicitly for i1+ when this happens.
    if len(payload.get("instructions", [])) <= 1:
        retry_msg = (
            user_msg
            + "\n\nYour previous output is invalid because it only contains i0. "
              "Regenerate and include at least i1 with a valid milestone. "
              "Return JSON only."
        )
        payload = llm_json_call(client, STEP3_SYSTEM, retry_msg, model=model)

    instructions = [
        Instruction(
            id=i["id"],
            instruction=i["instruction"],
            milestone=i.get("milestone", ""),
            revision_type=i.get("type", "additive"),
        )
        for i in payload["instructions"]
    ]
    result = MultiturnResult(instructions=instructions)
    if "reflection" in payload:
        result.reflection = payload["reflection"]
    return result
