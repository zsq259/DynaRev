"""Step 1: Extract constraints from a single-turn task instruction."""

from openai import OpenAI

from utils.llm import llm_json_call
from .models import Constraint, ConstraintResult
from .prompts import STEP1_SYSTEM, STEP1_USER

# Minimum number of constraints to proceed
MIN_CONSTRAINTS = 3


def extract_constraints(
    task_instruction: str,
    client: OpenAI,
    model: str = "gpt-4o",
) -> ConstraintResult:
    user_msg = STEP1_USER.format(task=task_instruction)
    payload = llm_json_call(client, STEP1_SYSTEM, user_msg, model=model)
    constraints = [
        Constraint(id=c["id"], description=c["description"])
        for c in payload.get("constraints", [])
    ]
    return ConstraintResult(
        original_instruction=task_instruction,
        constraints=constraints,
    )


def should_filter_step1(result: ConstraintResult) -> str:
    """Return filter reason if the task should be dropped, else empty string."""
    if len(result.constraints) < MIN_CONSTRAINTS:
        return f"Too few constraints ({len(result.constraints)} < {MIN_CONSTRAINTS})"
    return ""
