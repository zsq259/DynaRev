from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from ..models import BranchPolicy, Constraint, Milestone, Revision, TaskInstance


TOP_KEYS = {
    "format_version",
    "instance_id",
    "source",
    "runtime",
    "instruction_timeline",
    "milestones",
    "branch_policy",
    "constraints",
}
SOURCE_KEYS = {"dataset", "task_id", "website", "scene", "base_task"}
RUNTIME_KEYS = {"start_url", "max_steps", "action_space"}
REVISION_KEYS = {
    "revision_id",
    "type",
    "content",
    "block_user_reply_until_applied",
    "constraint_ids",
}
MILESTONE_KEYS = {"milestone_id", "inject_revision_id", "semantic_concept"}
CONSTRAINT_KEYS = {"id", "description", "source_revision"}
BRANCH_POLICY_KEYS_REQUIRED = {"condition_text", "metadata"}
BRANCH_POLICY_ID_KEYS_OLD = {"default_constraint_id", "alternative_constraint_id"}
BRANCH_POLICY_ID_KEYS_NEW = {"default_constraint_ids", "alternative_constraint_ids"}
BRANCH_POLICY_KEYS = BRANCH_POLICY_KEYS_REQUIRED | BRANCH_POLICY_ID_KEYS_OLD | BRANCH_POLICY_ID_KEYS_NEW
REVISION_TYPES = {"additive", "override", "conditional_override"}
BRANCHES = {"default", "alternative"}


class TaskFormatError(ValueError):
    """A JSONL task does not conform to the supported task schema."""


def _fail(field: str, message: str) -> None:
    raise TaskFormatError(f"{field}: {message}")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(field, "must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(field, "must be a list")
    return value


def _string(value: Any, field: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        _fail(field, "must be a string" if empty else "must be a non-empty string")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(field, "must be an integer")
    return value


def _keys(
    obj: Mapping[str, Any],
    required: set[str],
    field: str,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - obj.keys())
    if missing:
        _fail(field, f"missing required keys: {', '.join(missing)}")
    unexpected = sorted(obj.keys() - required - optional)
    if unexpected:
        _fail(field, f"unexpected keys: {', '.join(unexpected)}")


def _constraint_ids(value: Any, field: str) -> list[int]:
    result = [
        _integer(item, f"{field}[{i}]")
        for i, item in enumerate(_list(value, field))
    ]
    if len(result) != len(set(result)):
        _fail(field, "must not contain duplicate IDs")
    return result


def _validate_task(raw: Any) -> None:
    """The one format check used by every JSONL task consumer."""
    task = _mapping(raw, "task")
    _keys(task, TOP_KEYS, "task")
    _string(task["format_version"], "format_version")
    _string(task["instance_id"], "instance_id")

    source = _mapping(task["source"], "source")
    _keys(source, SOURCE_KEYS, "source")
    for key in SOURCE_KEYS:
        _string(source[key], f"source.{key}", empty=(key == "scene"))

    runtime = _mapping(task["runtime"], "runtime")
    _keys(runtime, RUNTIME_KEYS, "runtime")
    _string(runtime["start_url"], "runtime.start_url")
    if _integer(runtime["max_steps"], "runtime.max_steps") <= 0:
        _fail("runtime.max_steps", "must be greater than zero")
    actions = _list(runtime["action_space"], "runtime.action_space")
    if not actions:
        _fail("runtime.action_space", "must not be empty")
    for i, action in enumerate(actions):
        _string(action, f"runtime.action_space[{i}]")

    timeline = _mapping(task["instruction_timeline"], "instruction_timeline")
    _keys(timeline, {"i0", "revisions"}, "instruction_timeline")
    i0 = _mapping(timeline["i0"], "instruction_timeline.i0")
    _keys(i0, {"content", "constraint_ids"}, "instruction_timeline.i0")
    _string(i0["content"], "instruction_timeline.i0.content")
    owners = {
        cid: "i0"
        for cid in _constraint_ids(
            i0["constraint_ids"], "instruction_timeline.i0.constraint_ids"
        )
    }

    revisions = _list(timeline["revisions"], "instruction_timeline.revisions")
    revision_ids: set[str] = set()
    conditional_ids: list[str] = []
    for i, revision_raw in enumerate(revisions):
        field = f"instruction_timeline.revisions[{i}]"
        revision = _mapping(revision_raw, field)
        _keys(revision, REVISION_KEYS, field)
        rid = _string(revision["revision_id"], f"{field}.revision_id")
        if rid == "i0" or rid in revision_ids:
            _fail(f"{field}.revision_id", f"duplicate or reserved ID: {rid}")
        revision_ids.add(rid)
        rtype = _string(revision["type"], f"{field}.type")
        if rtype not in REVISION_TYPES:
            _fail(f"{field}.type", f"unsupported revision type: {rtype}")
        if rtype == "conditional_override":
            conditional_ids.append(rid)
        _string(revision["content"], f"{field}.content")
        if not isinstance(revision["block_user_reply_until_applied"], bool):
            _fail(f"{field}.block_user_reply_until_applied", "must be a boolean")
        for cid in _constraint_ids(
            revision["constraint_ids"], f"{field}.constraint_ids"
        ):
            if cid in owners:
                _fail(
                    f"{field}.constraint_ids",
                    f"constraint ID {cid} is already owned by {owners[cid]}",
                )
            owners[cid] = rid

    milestones = _list(task["milestones"], "milestones")
    milestone_ids: set[str] = set()
    injected: list[str] = []
    for i, milestone_raw in enumerate(milestones):
        field = f"milestones[{i}]"
        milestone = _mapping(milestone_raw, field)
        _keys(milestone, MILESTONE_KEYS, field, {"notes"})
        mid = _string(milestone["milestone_id"], f"{field}.milestone_id")
        if mid in milestone_ids:
            _fail(f"{field}.milestone_id", f"duplicate ID: {mid}")
        milestone_ids.add(mid)
        rid = _string(milestone["inject_revision_id"], f"{field}.inject_revision_id")
        if rid not in revision_ids:
            _fail(f"{field}.inject_revision_id", f"unknown revision ID: {rid}")
        injected.append(rid)
        _string(milestone["semantic_concept"], f"{field}.semantic_concept")
        if "notes" in milestone:
            _string(milestone["notes"], f"{field}.notes", empty=True)
    if len(injected) != len(set(injected)):
        _fail("milestones", "each revision must be injected by at most one milestone")
    if set(injected) != revision_ids:
        missing = sorted(revision_ids - set(injected))
        _fail("milestones", f"must inject every revision exactly once; missing: {missing}")

    constraints = _list(task["constraints"], "constraints")
    if not constraints:
        _fail("constraints", "must not be empty")
    by_id: dict[int, Mapping[str, Any]] = {}
    for i, constraint_raw in enumerate(constraints):
        field = f"constraints[{i}]"
        constraint = _mapping(constraint_raw, field)
        _keys(constraint, CONSTRAINT_KEYS, field, {"branch"})
        cid = _integer(constraint["id"], f"{field}.id")
        if cid in by_id:
            _fail(f"{field}.id", f"duplicate ID: {cid}")
        by_id[cid] = constraint
        _string(constraint["description"], f"{field}.description")
        source_revision = _string(
            constraint["source_revision"], f"{field}.source_revision"
        )
        if source_revision != "i0" and source_revision not in revision_ids:
            _fail(f"{field}.source_revision", f"unknown revision ID: {source_revision}")
        branch = constraint.get("branch")
        if branch is not None and branch not in BRANCHES:
            _fail(f"{field}.branch", f"unsupported branch: {branch}")

    if set(owners) != set(by_id):
        unknown = sorted(set(owners) - set(by_id))
        unbound = sorted(set(by_id) - set(owners))
        _fail(
            "constraints",
            f"instruction bindings do not match constraints; "
            f"unknown={unknown}, unbound={unbound}",
        )
    for cid, owner in owners.items():
        declared = by_id[cid]["source_revision"]
        if owner != declared:
            _fail(
                f"constraint {cid}.source_revision",
                f"declares {declared}, but instruction_timeline binds it to {owner}",
            )

    policy_raw = task["branch_policy"]
    branch_tags = {
        cid: c.get("branch")
        for cid, c in by_id.items()
        if c.get("branch") is not None
    }
    if conditional_ids:
        if len(conditional_ids) != 1:
            _fail("instruction_timeline.revisions", "must contain at most one conditional override")
        policy = _mapping(policy_raw, "branch_policy")
        _keys(
            policy,
            BRANCH_POLICY_KEYS_REQUIRED,
            "branch_policy",
            BRANCH_POLICY_ID_KEYS_OLD | BRANCH_POLICY_ID_KEYS_NEW,
        )
        _string(policy["condition_text"], "branch_policy.condition_text")
        default_ids, alternative_ids = _branch_policy_id_lists(policy)
        if not default_ids or not alternative_ids:
            _fail("branch_policy", "default and alternative groups must be non-empty")
        if set(default_ids) & set(alternative_ids):
            _fail("branch_policy", "default and alternative groups must be disjoint")
        expected_tags = {
            **{cid: "default" for cid in default_ids},
            **{cid: "alternative" for cid in alternative_ids},
        }
        if branch_tags != expected_tags:
            _fail(
                "branch_policy",
                "constraint branch tags must match its default and alternative ID groups",
            )
        for cid in default_ids + alternative_ids:
            if cid not in by_id:
                _fail("branch_policy", f"unknown constraint ID: {cid}")
        _mapping(policy["metadata"], "branch_policy.metadata")
    else:
        if policy_raw != []:
            _fail("branch_policy", "must be [] when there is no conditional override")
        if branch_tags:
            _fail("constraints", "branch tags require a conditional override")


def _branch_policy_id_lists(policy: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    """Normalize singular or plural branch_policy id fields into two lists."""
    has_new = (
        "default_constraint_ids" in policy or "alternative_constraint_ids" in policy
    )
    has_old = (
        "default_constraint_id" in policy or "alternative_constraint_id" in policy
    )
    if has_new and has_old:
        _fail(
            "branch_policy",
            "use either default/alternative_constraint_id or *_ids, not both",
        )
    if has_new:
        default_ids = _constraint_ids(
            policy.get("default_constraint_ids", []),
            "branch_policy.default_constraint_ids",
        )
        alternative_ids = _constraint_ids(
            policy.get("alternative_constraint_ids", []),
            "branch_policy.alternative_constraint_ids",
        )
        return default_ids, alternative_ids
    if has_old:
        default_id = _integer(
            policy["default_constraint_id"], "branch_policy.default_constraint_id"
        )
        alternative_id = _integer(
            policy["alternative_constraint_id"],
            "branch_policy.alternative_constraint_id",
        )
        return [default_id], [alternative_id]
    _fail(
        "branch_policy",
        "missing default/alternative constraint id fields",
    )
    return [], []


def _parse_task(raw: Mapping[str, Any]) -> TaskInstance:
    timeline = raw["instruction_timeline"]
    i0 = timeline["i0"]
    revisions = [
        Revision(
            revision_id=r["revision_id"],
            type=r["type"],
            content=r["content"],
            block_user_reply_until_applied=r[
                "block_user_reply_until_applied"
            ],
            constraint_ids=r["constraint_ids"],
        )
        for r in timeline["revisions"]
    ]
    constraints = [
        Constraint(
            id=c["id"],
            description=c["description"],
            source_revision=c["source_revision"],
            branch=c.get("branch"),
        )
        for c in raw["constraints"]
    ]
    milestones = [Milestone(**m) for m in raw["milestones"]]
    bp_raw = raw["branch_policy"]
    branch_policy: Optional[BranchPolicy] = None
    if isinstance(bp_raw, dict):
        default_ids, alternative_ids = _branch_policy_id_lists(bp_raw)
        branch_policy = BranchPolicy(
            condition_text=bp_raw["condition_text"],
            default_constraint_ids=default_ids,
            alternative_constraint_ids=alternative_ids,
            metadata=bp_raw["metadata"],
        )
    return TaskInstance(
        format_version=raw["format_version"],
        instance_id=raw["instance_id"],
        source=raw["source"],
        runtime=raw["runtime"],
        i0=i0["content"],
        i0_constraint_ids=i0["constraint_ids"],
        revisions=revisions,
        milestones=milestones,
        constraints=constraints,
        branch_policy=branch_policy,
    )


def load_dyna_jsonl(path: str | Path) -> List[TaskInstance]:
    """Load JSONL tasks; this is the sole task-format validation boundary."""
    tasks: List[TaskInstance] = []
    seen: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TaskFormatError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            try:
                _validate_task(raw)
            except TaskFormatError as exc:
                raise TaskFormatError(f"{path}:{line_number}: {exc}") from exc
            instance_id = raw["instance_id"]
            if instance_id in seen:
                raise TaskFormatError(
                    f"{path}:{line_number}: instance_id: duplicate "
                    f"{instance_id!r}; first seen on line {seen[instance_id]}"
                )
            seen[instance_id] = line_number
            tasks.append(_parse_task(raw))
    return tasks


def load_om2w_hf(dataset_name: str, split: str, default_max_steps: int) -> List[TaskInstance]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name)[split]
    tasks: List[TaskInstance] = []
    for row in ds:
        tasks.append(
            TaskInstance(
                format_version="om2w_compat",
                instance_id=f"om2w_{row['task_id']}",
                source={
                    "dataset": dataset_name,
                    "task_id": row["task_id"],
                    "website": row["website"],
                    "scene": "unknown",
                    "base_task": row["confirmed_task"],
                },
                runtime={
                    "start_url": row["website"],
                    "max_steps": default_max_steps,
                    "action_space": [
                        "goto",
                        "click",
                        "fill",
                        "press",
                        "scroll",
                        "send_msg_to_user",
                    ],
                },
                i0=row["confirmed_task"],
                i0_constraint_ids=[],
                revisions=[],
                milestones=[],
                constraints=[Constraint(id=0, description="Satisfy task description")],
            )
        )
    return tasks


def select_tasks(tasks: Sequence[TaskInstance], args) -> List[TaskInstance]:
    filtered = list(tasks)
    if args.scene:
        filtered = [t for t in filtered if t.source.get("scene") == args.scene]
    if args.instance_id:
        wanted = {x.strip() for x in args.instance_id.split(",") if x.strip()}
        filtered = [t for t in filtered if t.instance_id in wanted]
    if args.shuffle:
        random.Random(args.seed).shuffle(filtered)
    if args.offset > 0:
        filtered = filtered[args.offset :]
    if args.limit > 0:
        filtered = filtered[: args.limit]
    return filtered
