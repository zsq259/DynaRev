import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.agent_lib.data import TaskFormatError, load_dyna_jsonl


def make_static_task(instance_id: str = "task_static") -> dict:
    return {
        "format_version": "dyna_task_v2",
        "instance_id": instance_id,
        "source": {
            "dataset": "test",
            "task_id": "source_1",
            "website": "https://example.test",
            "scene": "test",
            "base_task": "Find the answer.",
        },
        "runtime": {
            "start_url": "https://example.test",
            "max_steps": 10,
            "action_space": ["goto", "click", "send_msg_to_user"],
        },
        "instruction_timeline": {
            "i0": {"content": "Find the answer.", "constraint_ids": [1]},
            "revisions": [],
        },
        "milestones": [],
        "branch_policy": [],
        "constraints": [
            {"id": 1, "description": "The answer must be found", "source_revision": "i0"}
        ],
    }


def write_tasks(directory: str, tasks: list[dict]) -> Path:
    path = Path(directory) / "tasks.jsonl"
    path.write_text("".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8")
    return path


class TaskDataValidationTests(unittest.TestCase):
    def test_static_task_allows_empty_revisions_and_milestones(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = load_dyna_jsonl(write_tasks(tmp, [make_static_task()]))

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].revisions, [])

    def test_constraint_reference_must_exist(self):
        task = make_static_task()
        task["instruction_timeline"]["i0"]["constraint_ids"] = [99]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TaskFormatError, "instruction bindings do not match"):
                load_dyna_jsonl(write_tasks(tmp, [task]))

    def test_duplicate_instance_id_is_rejected(self):
        task = make_static_task()

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TaskFormatError, "first seen on line 1"):
                load_dyna_jsonl(write_tasks(tmp, [task, copy.deepcopy(task)]))

    def test_legacy_shape_is_not_silently_parsed(self):
        task = make_static_task()
        task["format_version"] = "dyna_task_v1"
        task["instruction_timeline"]["i0"] = "Find the answer."
        task["success_criteria"] = task.pop("constraints")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TaskFormatError, "missing required keys: constraints"):
                load_dyna_jsonl(write_tasks(tmp, [task]))

    def test_conditional_branch_references_are_checked(self):
        task = make_static_task("task_conditional")
        task["instruction_timeline"]["revisions"] = [
            {
                "revision_id": "i1",
                "type": "conditional_override",
                "content": "Use the alternative if the condition holds.",
                "block_user_reply_until_applied": True,
                "constraint_ids": [2, 3],
            }
        ]
        task["milestones"] = [
            {
                "milestone_id": "m1",
                "inject_revision_id": "i1",
                "semantic_concept": "The condition can be observed.",
            }
        ]
        task["constraints"].extend(
            [
                {"id": 2, "description": "Use default", "source_revision": "i1", "branch": "default"},
                {"id": 3, "description": "Use alternative", "source_revision": "i1", "branch": "alternative"},
            ]
        )
        task["branch_policy"] = {
            "condition_text": "the condition holds",
            "default_constraint_id": 2,
            "alternative_constraint_id": 99,
            "metadata": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TaskFormatError, "branch tags must match"):
                load_dyna_jsonl(write_tasks(tmp, [task]))


if __name__ == "__main__":
    unittest.main()
