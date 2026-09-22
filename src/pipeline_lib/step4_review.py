"""Step 4: Quality review — accept or reject the redesigned task."""

import json

from openai import OpenAI

from utils.llm import llm_json_call
from .models import MultiturnResult, ReviewResult
from .prompts import STEP4_SYSTEM, STEP4_USER


def review_quality(
    original_instruction: str,
    multiturn: MultiturnResult,
    client: OpenAI,
    model: str = "gpt-4o",
    previous_step4_reason: str | None = None,
    override_context: dict | None = None,
) -> ReviewResult:
    redesign_json = json.dumps(
        {
            "instructions": [
                {
                    "id": i.id,
                    "instruction": i.instruction,
                    **({"milestone": i.milestone} if i.milestone else {}),
                }
                for i in multiturn.instructions
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    
    # Build context about previous failure (if retrying)
    previous_context = ""
    if previous_step4_reason:
        previous_context = (
            f"**Previous attempt feedback:** {previous_step4_reason}\n\n"
            "Please re-evaluate the redesign, addressing the issues identified above."
        )

    override_context_block = ""
    if override_context:
        override_mode = override_context.get("override_mode", "unconditional")
        ctx_json = json.dumps(override_context, ensure_ascii=False, indent=2)
        if override_mode == "conditional":
            override_context_block = (
                "Override context:\n"
                f"{ctx_json}\n\n"
                "This is a conditional override (override_mode=conditional). The override "
                "revision (type=conditional_override) softens the user's original preference "
                "with a conditional clause: default = target_constraint (Step 3b keep), "
                "alt = replacement_constraint (Step 3b change_to) accepted only when a "
                "page-observable condition holds.\n\n"
                "Those two texts are written by Step 3b. They may differ from the "
                "library pair and from the original extracted constraint. Paraphrase is "
                "fine. Reject if they are the same preference (a no-op override).\n\n"
                "Accept the override revision when its text:\n"
                "  1. acknowledges target_constraint as the default preference;\n"
                "  2. states a page-observable comparison (quantity) > or < (number), where "
                "quantity is count, price, rating, or time, with a concrete number "
                "(paraphrase OK; the number must not drift from condition_text);\n"
                "  3. presents replacement_constraint as a conditional alternative, not a "
                "flat switch.\n\n"
                "target_constraint must be introduced earlier in the timeline (i0 or an "
                "earlier additive); the conditional_override revision should not be its "
                "first appearance."
            )
        else:
            override_context_block = (
                "Override context:\n"
                f"{ctx_json}\n\n"
                "Interpretation rule: if an override instruction explicitly performs the "
                "target->replacement change declared above, do not reject it solely for "
                "differing from the original extracted constraint or from the library pair. "
                "target_constraint / replacement_constraint are Step 3b keep / change_to. "
                "Reject if they are the same preference."
            )
    
    user_msg = STEP4_USER.format(
        original_instruction=original_instruction,
        redesign=redesign_json,
        previous_step4_context=previous_context,
        override_context_block=override_context_block,
    )
    payload = llm_json_call(client, STEP4_SYSTEM, user_msg, model=model)
    return ReviewResult(
        passed=bool(payload.get("passed", False)),
        worth_retry=payload.get("worth_retry"),  # can be True, False, or None
        retry_from_step=payload.get("retry_from_step"),  # can be "step2", "step3", or None
        reason=payload.get("reason", ""),
    )
