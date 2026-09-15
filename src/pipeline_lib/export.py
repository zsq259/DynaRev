"""Convert pipeline output to dyna_tasks_v2.jsonl format."""

import json
from typing import Any, Dict

from .models import PipelineOutput


def to_dyna_task_json(output: PipelineOutput) -> Dict[str, Any]:
    """Convert a successful PipelineOutput to dyna_task_v2 JSON structure.

    Conditional override (override_mode='conditional'): the original
    constraint stays as ``branch='default'`` and a synthetic alternative
    constraint with ``branch='alternative'`` is appended, bound to the
    conditional revision. ``branch_policy`` carries the LLM-readable
    condition_text plus the constraint id pointers for evaluator Phase A.
    Unconditional override and base tracks emit ``branch_policy: []`` as
    before.
    """
    inp = output.input
    mt = output.step3
    if mt is None:
        raise ValueError("step3 result is required to convert to task JSON")

    i0_inst = mt.instructions[0]

    # Build constraint-to-instruction mapping from Step 5 (if available)
    constraint_id_to_revision: Dict[int, str] = {}
    if output.step5:
        for entry in output.step5.mapping:
            constraint_id_to_revision[entry.constraint_id] = entry.instruction_id

    is_conditional = bool(
        output.step3a
        and output.step3a.enable_override
        and output.step3a.override_mode == "conditional"
    )

    # In override mode, evaluation should use the replaced constraint
    # text — UNLESS we're in conditional mode, where the original stays
    # as the default branch and we instead append a synthetic
    # alternative-branch constraint.
    override_replacement_by_constraint_id: Dict[int, str] = {}
    if (
        output.step3a
        and output.step3a.enable_override
        and not is_conditional
    ):
        for item in output.step3a.overrides:
            if item.replacement_constraint_text:
                override_replacement_by_constraint_id[item.target_constraint_id] = item.replacement_constraint_text

    # Build revisions from instructions (skip i0 which is instructions[0])
    # Track which revision is the conditional override (only one per
    # task in v1 of conditional pipeline).
    conditional_revision_id: str = ""
    revisions = []
    for inst in mt.instructions[1:]:
        rev_id = f"i{inst.id}"
        if inst.revision_type == "conditional_override":
            conditional_revision_id = rev_id
        rev_constraint_ids = [
            cid for cid, iid in constraint_id_to_revision.items()
            if iid == rev_id
        ]
        revisions.append({
            "revision_id": rev_id,
            "type": inst.revision_type,
            "content": inst.instruction,
            "block_user_reply_until_applied": True,
            "constraint_ids": rev_constraint_ids,
        })

    # Build milestones from instruction-level milestone definitions
    milestones = []
    for inst in mt.instructions[1:]:
        milestones.append({
            "milestone_id": f"m{inst.id}",
            "inject_revision_id": f"i{inst.id}",
            "semantic_concept": inst.milestone,
        })

    # Build i0 constraint_ids from Step 5 mapping
    i0_constraint_ids = [
        cid for cid, iid in constraint_id_to_revision.items()
        if iid == "i0"
    ]

    # Build constraints with source_revision binding
    constraints_list = []
    if output.step1:
        for c in output.step1.constraints:
            description = override_replacement_by_constraint_id.get(c.id, c.description)
            entry: Dict[str, Any] = {
                "id": c.id,
                "description": description,
                "source_revision": constraint_id_to_revision.get(c.id, "unknown"),
            }
            constraints_list.append(entry)

    branch_policy: Any = []

    if is_conditional and output.step3a:
        # Pick first override item — current pipeline allows at most one.
        item = output.step3a.overrides[0]
        sc = item.switching_condition
        if sc is not None and conditional_revision_id:
            # Tag the original (default-branch) constraint
            for c_entry in constraints_list:
                if c_entry["id"] == item.target_constraint_id:
                    c_entry["branch"] = "default"
                    break
            # Append synthetic alternative-branch constraint
            existing_ids = {c["id"] for c in constraints_list}
            alt_id = max(existing_ids) + 1 if existing_ids else 1
            constraints_list.append({
                "id": alt_id,
                "description": item.replacement_constraint_text,
                "source_revision": conditional_revision_id,
                "branch": "alternative",
            })
            # Bind alt id to conditional revision in revision listing
            for r in revisions:
                if r["revision_id"] == conditional_revision_id:
                    if alt_id not in r["constraint_ids"]:
                        r["constraint_ids"].append(alt_id)
                    break
            branch_policy = {
                "condition_text": sc.condition_text,
                "default_constraint_id": item.target_constraint_id,
                "alternative_constraint_id": alt_id,
                "metadata": {
                    "switching_dimension": sc.dimension,
                    "operator": sc.operator,
                    "threshold": sc.threshold,
                    "unit": sc.unit,
                },
            }

    return {
        "format_version": "dyna_task_v2",
        "instance_id": f"dyna_{inp.task_id}",
        "source": {
            "dataset": inp.dataset,
            "task_id": inp.task_id,
            "website": inp.website,
            "scene": inp.scene,
            "base_task": inp.original_instruction,
        },
        "runtime": {
            "start_url": inp.website,
            "max_steps": 28,
            "action_space": ["goto", "click", "fill", "press", "scroll", "send_msg_to_user"],
        },
        "instruction_timeline": {
            "i0": {
                "content": i0_inst.instruction,
                "constraint_ids": i0_constraint_ids,
            },
            "revisions": revisions,
        },
        "milestones": milestones,
        "branch_policy": branch_policy,
        "constraints": constraints_list,
    }


def write_dyna_jsonl(outputs: list[PipelineOutput], path: str) -> int:
    """Write accepted pipeline outputs to a JSONL file. Returns count written."""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for output in outputs:
            if not output.filtered_at and output.step3:
                task_json = to_dyna_task_json(output)
                f.write(json.dumps(task_json, ensure_ascii=False) + "\n")
                count += 1
    return count
