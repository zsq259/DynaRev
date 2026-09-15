"""Phase 2: Filter Mind2Web-2 tasks by dynamic goal revision suitability.

Input: Phase 1 results (116 tasks that passed accessibility check).
Uses LLM to assess whether the task scenario naturally supports
a user dynamically adjusting their goals mid-task.
"""

import json
import logging
import time
from pathlib import Path

from ..utils.client import get_client
from ..utils.llm import llm_call, find_last_json_object

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "deepseek-v4-flash"

SYSTEM_PROMPT = """\
You are an expert at analyzing web task scenarios for a GUI Agent benchmark.

## Background

We are building a benchmark where users interact with a GUI agent across multiple turns,
dynamically adjusting their goals mid-task. The agent must respond to these changes during
execution. We need to identify tasks from an existing dataset that can be naturally converted
into this multi-turn dynamic adjustment format.

Every task in the existing dataset is written as a single instruction with all constraints
listed together. This is just the dataset format. Your job is to judge whether a task can be
naturally restructured as a multi-turn conversation where the user progressively adjusts
their goal.

## Core Principle

The conversion works by splitting the original task into I0, I1, I2, ... where each turn
states part of the content and subsequent turns progressively add or adjust.

The core question is: **After conversion, would each turn's instruction feel natural to say
at that moment?**

What does "natural" mean? A real user, during execution, would naturally say this at that
moment — perhaps because they saw an intermediate result and formed a new preference, perhaps
because they discovered a new constraint they hadn't considered. The key is that this
instruction is something the user decided to say at that moment, not something they planned
to say from the beginning but deliberately held back.

IMPORTANT CAVEAT: "The user might have forgotten to mention it" is NOT a valid justification
on its own. For any constraint, you could hypothetically argue the user forgot. The real test
is whether the constraint is INDEPENDENT — meaning the task still makes complete sense without
it. If removing a constraint makes the task meaningless or fundamentally different, then that
constraint is part of the task's core intent, and claiming "the user forgot" is just an excuse
for artificial splitting. The test is not just whether the task can still be executed without
the constraint, but whether what remains is genuinely what the user wanted in the first place.

What does "unnatural" mean? It means converting for the sake of converting — the result looks
like multiple turns on the surface, but no such interaction pattern exists in real life.
These constraints were part of the task's intent from the very beginning; without any one of
them the task becomes something else entirely. Splitting them up is just artificially revealing
a complete, pre-formed query piece by piece. Real users do not interact with agents this way.

## How to Judge

### Step 1: Imagine the converted multi-turn form.

Convert the original task into multiple turns: I0 states the broadest part, subsequent turns
progressively add constraints. Imagine a real user executing this task — at what moments would
they naturally say something new to the agent?

### Step 2: Check the naturalness of each turn.

For each constraint added in a new turn, ask: is this something the user would naturally think
to say at that moment, or is it something they knew they needed to say from the start and just
hadn't said yet?

If the former — a preference formed after seeing options, a new constraint the user hadn't
considered before — this constraint is natural. But ONLY if the constraint is independent:
the task without it must still be a complete, meaningful task on its own.

If the latter — this constraint was bundled together with the others from the very beginning,
the task intent is incomplete without it, and in a real scenario the user would never have left
it unsaid at the start — then pulling it out as a later turn is artificial, not genuine dynamic
adjustment. Do NOT justify it by saying "the user might have forgotten" — if the task doesn't
make sense without this constraint, the user would not have forgotten it.

### Step 3: Check the completeness at each split point.

At any split point, remove all subsequent constraints. Is what remains a task the user would
genuinely put forward on its own? If what remains is too vague — if no real user would actually
start executing with just that — then the subsequent constraints are part of the task's core
intent, not something that emerged dynamically.

## Examples

**Suitable (natural):**

House-hunting task — I0 "Help me find houses in San Antonio," then after seeing results
"keep it under $300k," then after seeing some without garages "I need a garage." Each turn
sounds like something a real user would naturally say at that moment. "Under $300k" and
"needs a garage" are independent preferences that can naturally surface after seeing specific
listings. Remove these constraints and I0 is still a task the user would genuinely put forward
on its own.

**Not suitable (converting for the sake of converting):**

Buzzer-beater task — if converted into "find NBA buzzer-beaters" → "from the 2019 playoffs" →
"conference semifinals and beyond" → "within the last 3 seconds, no free throws," none of these
turns sound like something a real user would say that way. In real life, no such interaction
exists where a user starts with "find NBA buzzer-beaters" and then gradually adds these
conditions. This converted multi-turn form has no counterpart in reality. These conditions are
the definition of the query itself from the start; splitting them up is just artificially
revealing a pre-formed query piece by piece.

## Output

Output a JSON object:
{
  "suitable": true or false,
  "reason": "1-2 sentences explaining why this task is or is not suitable for conversion"
}
"""


def assess_task(client, task: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Task ID: {task['task_id']}\n"
            f"Domain: {task['domain']}\n"
            f"Description: {task['task_description']}"
        )},
    ]
    raw = llm_call(
        client=client,
        messages=messages,
        model=MODEL,
        max_tokens=4096,
    )
    return find_last_json_object(raw)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    # Load Phase 1 results
    with open(PROJECT_ROOT / "data/sources/mind2web2/L2_filtered/m2w2_phase1_accessible.json") as f:
        phase1 = json.load(f)

    # Only keep tasks that passed Phase 1
    candidates = [t for t in phase1 if t["status"] in ("pass", "pass_no_website")]
    logger.info("Phase 2 input: %d tasks (passed accessibility)", len(candidates))

    client = get_client(MODEL)
    results = []

    for i, task in enumerate(candidates):
        logger.info("[%d/%d] Assessing: %s", i + 1, len(candidates), task["task_id"])
        try:
            assessment = assess_task(client, task)
        except Exception as e:
            logger.error("Failed: %s: %s", task["task_id"], e)
            assessment = {"suitable": None, "reason": f"LLM error: {e}"}
            time.sleep(3)

        results.append({
            "task_id": task["task_id"],
            "domain": task["domain"],
            "subdomain": task["subdomain"],
            "websites": task["websites"],
            "suitable": assessment.get("suitable"),
            "reason": assessment.get("reason", ""),
            "task_description": task["task_description"],
        })

        if (i + 1) % 5 == 0:
            time.sleep(1)

    # Save full results
    out_path = PROJECT_ROOT / "data/sources/mind2web2/L2_filtered/m2w2_phase2_revision.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved to %s", out_path)

    # Summary
    suitable = [r for r in results if r["suitable"] is True]
    not_suitable = [r for r in results if r["suitable"] is False]
    errored = [r for r in results if r["suitable"] is None]

    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS: Dynamic Goal Revision Suitability")
    print("=" * 60)
    print(f"Total assessed:    {len(results)}")
    print(f"SUITABLE:          {len(suitable)}")
    print(f"NOT SUITABLE:      {len(not_suitable)}")
    if errored:
        print(f"ERROR:             {len(errored)}")
    print()

    print(f"SUITABLE ({len(suitable)}):")
    for r in suitable:
        sites = ", ".join(r["websites"]) if r["websites"] else "(no specific site)"
        print(f"  {r['task_id']:40s} {sites}")
        print(f"  {'':40s} {r['reason'][:100]}")
        print()

    print(f"NOT SUITABLE ({len(not_suitable)}):")
    for r in not_suitable:
        print(f"  {r['task_id']:40s} {r['reason'][:100]}")


if __name__ == "__main__":
    main()
