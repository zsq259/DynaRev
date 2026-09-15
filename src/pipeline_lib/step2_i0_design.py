"""Step 2: Design I0 based on constraints and the website homepage."""

import json

from openai import OpenAI

from utils.llm import llm_json_call
from .models import ConstraintResult, I0Result
from .prompts import STEP2_SYSTEM, STEP2_USER


def design_i0(
    constraints: ConstraintResult,
    homepage_description: str,
    client: OpenAI,
    model: str = "gpt-4o",
    previous_step4_reason: str | None = None,
) -> I0Result:
    constraints_json = json.dumps(
        [{"id": c.id, "description": c.description} for c in constraints.constraints],
        ensure_ascii=False,
        indent=2,
    )
    
    # Build context about previous failure (if retrying)
    previous_context = ""
    if previous_step4_reason:
        previous_context = (
            f"**Previous attempt feedback:** {previous_step4_reason}\n\n"
            "Please address the issues identified above in your redesign of I0."
        )
    
    user_msg = STEP2_USER.format(
        homepage_description=homepage_description,
        constraints=constraints_json,
        previous_step4_context=previous_context,
    )
    payload = None
    last_err: Exception | None = None
    # Try twice: second pass adds stricter JSON-only reminder to recover from format drift.
    for idx in range(2):
        attempt_msg = user_msg
        if idx == 1:
            attempt_msg += (
                "\n\nIMPORTANT: Return ONLY a valid JSON object matching the schema. "
                "Do not include explanations, headings, or markdown."
            )
        try:
            payload = llm_json_call(client, STEP2_SYSTEM, attempt_msg, model=model)
            break
        except Exception as e:  # pragma: no cover - upstream LLM/runtime failures
            last_err = e

    if payload is None:
        raise last_err if last_err else RuntimeError("Step2 returned no payload")

    classification = payload.get("classification", [])
    return I0Result(
        i0=payload["i0"],
        classification=classification,
    )
