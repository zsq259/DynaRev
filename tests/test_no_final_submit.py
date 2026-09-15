import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.evaluate_logs import evaluate_run_dir
from src.evaluator_lib.no_final_submit import (
    NO_FINAL_SUBMIT_LABEL,
    build_no_final_submit_result,
    is_no_final_submit,
)
from src.evaluator_lib.response_metrics import compute_demand_side_metrics
from src.models import Constraint, Revision, TaskInstance


def _task() -> TaskInstance:
    return TaskInstance(
        format_version="dyna_task_v2",
        instance_id="task_1",
        source={"website": "https://example.test"},
        runtime={"max_steps": 10},
        i0="Find the answer.",
        i0_constraint_ids=[1],
        revisions=[
            Revision(
                revision_id="i1",
                type="additive",
                content="Also require X.",
                block_user_reply_until_applied=False,
                constraint_ids=[2],
            )
        ],
        milestones=[],
        constraints=[
            Constraint(id=1, description="c1", source_revision="i0"),
            Constraint(id=2, description="c2", source_revision="i1"),
        ],
    )


class NoFinalSubmitProtocolTests(unittest.TestCase):
    def test_detects_max_steps_from_completion_status(self):
        self.assertTrue(is_no_final_submit({"completion_status": "max_steps"}))
        self.assertTrue(is_no_final_submit({"stopped_reason": "max_steps"}))
        self.assertFalse(is_no_final_submit({"completion_status": "submitted"}))
        self.assertFalse(is_no_final_submit({}))

    def test_zero_result_uses_new_label_not_face_value_or_not_fired(self):
        result = build_no_final_submit_result(
            _task(),
            {"fired_revisions": ["i1"], "completion_status": "max_steps"},
        )
        self.assertEqual(result["eval_status"], NO_FINAL_SUBMIT_LABEL)
        self.assertEqual(result["skip_reason"], "max_steps")
        self.assertFalse(result["is_correct"])
        self.assertEqual(result["correctness_score"], 0.0)
        self.assertEqual(result["diagnostic_counts"], {NO_FINAL_SUBMIT_LABEL: 2})
        labels = {r["combine_label"] for r in result["constraint_results"]}
        self.assertEqual(labels, {NO_FINAL_SUBMIT_LABEL})
        self.assertTrue(all(r["satisfied"] is False for r in result["constraint_results"]))
        fired = {r["constraint_id"]: r["fired"] for r in result["constraint_results"]}
        self.assertTrue(fired["c1"])
        self.assertTrue(fired["c2"])

    def test_mffr_includes_no_final_submit_as_first_round_failure(self):
        result = build_no_final_submit_result(_task(), {"fired_revisions": ["i1"]})
        demand = compute_demand_side_metrics(result, _task())
        self.assertEqual(demand["mffr"], 1)
        self.assertEqual(demand["rcsr_per_round"]["i0"]["rcsr"], 0.0)
        self.assertEqual(demand["rcsr_per_round"]["i1"]["rcsr"], 0.0)


class EvaluateLogsNoFinalSubmitTests(unittest.TestCase):
    def test_max_steps_skips_evaluator_and_writes_zero_artifact(self):
        task = _task()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            task_dir = run_dir / task.instance_id
            task_dir.mkdir(parents=True)
            (task_dir / "summary.json").write_text(
                json.dumps({
                    "completion_status": "max_steps",
                    "stopped_reason": "max_steps",
                    "done_called": True,
                    "fired_revisions": ["i1"],
                }),
                encoding="utf-8",
            )

            def fail_client(*args, **kwargs):
                raise AssertionError("evaluator client must not be created")

            class BoomEvaluator:
                def __init__(self, *args, **kwargs):
                    raise AssertionError("TaskEvaluator must not run")

            output = io.StringIO()
            with (
                patch("src.evaluate_logs.load_dyna_jsonl", return_value=[task]),
                patch("src.evaluate_logs.get_client", side_effect=fail_client),
                patch("src.evaluate_logs.TaskEvaluator", BoomEvaluator),
                redirect_stdout(output),
            ):
                errors = evaluate_run_dir(
                    run_dir=run_dir,
                    dataset_path="unused.jsonl",
                    force=True,
                )

            self.assertEqual(errors, 0)
            self.assertIn("no_final_submit", output.getvalue())
            eval_path = run_dir / "evaluations" / f"{task.instance_id}.evaluation.json"
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            result = payload["result"]
            self.assertEqual(result["eval_status"], NO_FINAL_SUBMIT_LABEL)
            self.assertFalse(result["is_correct"])
            self.assertEqual(result["correctness_score"], 0.0)
            run_summary = json.loads(
                (run_dir / "evaluations" / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_summary["no_final_submit_count"], 1)
            self.assertEqual(run_summary["success_rate"], 0.0)
            self.assertEqual(
                run_summary["diagnostic_label_totals"].get(NO_FINAL_SUBMIT_LABEL), 2
            )


if __name__ == "__main__":
    unittest.main()
