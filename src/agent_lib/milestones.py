from typing import Any, Dict, Tuple

from openai import OpenAI

from ..utils.llm import llm_call, strip_thinking, try_parse_json
from ..models import Milestone, Revision


def milestone_should_fire(
    milestone: Milestone,
    axtree: str,
    step: int,
    fired_step: Dict[str, int],
    client: OpenAI,
    model: str = "gpt-4o",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Semantic-stage-based milestone trigger judgment.
    Evaluates whether the current page state indicates that the milestone's
    semantic stage has been reached.
    """
    axtree_short = axtree[:20000] + ("\n...(truncated)" if len(axtree) > 20000 else "")
    system_msg = (
        "You are a milestone trigger judge for a dynamic GUI benchmark.\n"
        "Your job: decide whether the current page indicates that a specific "
        "semantic stage has been reached.\n\n"
        "RULES:\n"
        "- This is a factual judgment about the current page state.\n"
        "- Look for concrete evidence: listings, filtered results, "
        "specific content, navigation state, etc.\n"
        "- The semantic stage describes a CHECKPOINT in the user's task progress, "
        "not the exact UI method. For example, 'search results for sofas are "
        "displayed' is satisfied by ANY page showing a list of sofas — whether "
        "reached via search, category navigation, or direct URL. Focus on whether "
        "the current page state matches the described stage, not on how the agent "
        "got there.\n"
        "- The page must clearly show that the described stage has been achieved.\n"
        "- When genuinely uncertain, return should_fire=false. But do not be "
        "overly conservative — if the page clearly reaches the stage described "
        "by the milestone, fire even if minor details differ from the literal "
        "wording.\n"
        "- Existential vs exclusive: when the milestone describes a result state "
        "like 'a list of X is displayed' or 'results showing X', treat it as "
        "satisfied if the page contains X candidates the user could act on, EVEN "
        "IF the list also contains non-X items. Do not require the list to be "
        "exclusively X unless the milestone explicitly says 'only X' / "
        "'exclusively X' / 'purely X'.\n"
        "- Observable count vs explicit constraint marker: when the milestone "
        "describes a quantitative state like 'exactly N stops' or 'N items "
        "meeting criteria', you do NOT need the page to literally display "
        "'limited to N' or 'filter applied → N'. The state is satisfied if N "
        "matching items are observably present in the page content. The user "
        "does not need the page to advertise its own constraint application.\n"
        "- Junction vs terminal cleanliness: the milestone marks an intermediate "
        "junction on the task path, not the terminal completion of prior "
        "instructions. Fire as soon as the junction's observable evidence "
        "appears; do not wait for the prior task to be 'fully consummated' or "
        "for the page to settle into a single canonical end-state.\n"
        "- confidence below 0.8 → should_fire must be false.\n"
    )
    prompt = (
        f"Semantic stage to detect:\n{milestone.semantic_concept}\n\n"
        f"Current page content:\n{axtree_short}\n\n"
        "Has the current page reached the described semantic stage?\n"
        "List the specific evidence you found (or did not find), then decide.\n"
        "Return JSON only: {\"should_fire\": bool, \"confidence\": 0.0-1.0, \"reason\": \"...\"}"
    )
    try:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        raw = llm_call(
            client=client,
            messages=messages,
            model=model,
            temperature=0.0,
            max_tokens=250,
            usage_tag="trigger",
        )
        raw = strip_thinking(raw)
        payload = try_parse_json(raw)
        return bool(payload.get("should_fire", False)), {
            "confidence": payload.get("confidence", 0.0),
            "reason": payload.get("reason", ""),
        }
    except Exception as e:
        return False, {"confidence": 0.0, "reason": f"llm_error: {str(e)[:200]}"}


def build_instruction_block(i0: str, revisions: Dict[str, Revision], fired_revision_ids):
    lines = [f"1. {i0}"]
    idx = 2
    for rid in fired_revision_ids:
        rev = revisions.get(rid)
        if rev:
            if rev.type == "override":
                lines.append(f"{idx}. [CHANGED] {rev.content}")
            elif rev.type == "conditional_override":
                lines.append(
                    f"{idx}. [CONDITIONAL — report observation in done_message] "
                    f"{rev.content}"
                )
            else:
                lines.append(f"{idx}. {rev.content}")
            idx += 1
    return "\n".join(lines)
