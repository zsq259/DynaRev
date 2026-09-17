import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.evaluate_logs import evaluate_run_dir
from src.evaluator_lib.constraint_judge import simple_verify, verify_claim_on_frame
from src.evaluator_lib.core import (
    ScanState,
    TaskEvaluator,
    _frame_match_info,
    page_candidate_steps,
)
from src.evaluator_lib.decomposer import SubClaim, decompose
from src.evaluator_lib.errors import EvaluatorError
from src.evaluator_lib.response_judge import judge_response_window


class EvaluatorErrorPropagationTests(unittest.TestCase):
    def test_decomposer_raises_on_llm_failure(self):
        with patch(
            "src.evaluator_lib.decomposer.llm_json_call",
            side_effect=ConnectionError("network unavailable"),
        ):
            with self.assertRaisesRegex(EvaluatorError, "constraint decomposition failed"):
                decompose("must match", "done", object(), "test-model")

    def test_simple_verifier_raises_on_llm_failure(self):
        with patch(
            "src.evaluator_lib.constraint_judge.llm_json_call",
            side_effect=TimeoutError("request timed out"),
        ):
            with self.assertRaisesRegex(
                EvaluatorError, "simple constraint verification failed"
            ):
                simple_verify("constraint", "predicate", "done", object())

    def test_page_verifier_raises_on_llm_failure(self):
        with patch(
            "src.evaluator_lib.constraint_judge.llm_json_call",
            side_effect=ConnectionError("connection reset"),
        ):
            with self.assertRaisesRegex(
                EvaluatorError, "page evidence verification failed"
            ):
                verify_claim_on_frame(
                    "constraint", "subject", "assertion", "axtree", object()
                )

    def test_response_judge_raises_on_llm_failure(self):
        with patch(
            "src.evaluator_lib.response_judge.llm_json_call",
            side_effect=ConnectionError("connection reset"),
        ):
            with self.assertRaisesRegex(
                EvaluatorError, "response-window verification failed"
            ):
                judge_response_window("i1", "change", [], object())


class EvaluateLogsErrorHandlingTests(unittest.TestCase):
    def test_exhausted_evaluator_error_writes_no_result(self):
        class FailingTaskEvaluator:
            def __init__(self, *args, **kwargs):
                pass

            def evaluate(self):
                raise EvaluatorError("constraint decomposition failed: timeout")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            task_id = "task_1"
            task_dir = run_dir / task_id
            eval_dir = run_dir / "evaluations"
            task_dir.mkdir(parents=True)
            eval_dir.mkdir()
            (task_dir / "summary.json").write_text("{}", encoding="utf-8")

            eval_file = eval_dir / f"{task_id}.evaluation.json"
            eval_file.write_text(json.dumps({"old": True}), encoding="utf-8")
            run_summary = eval_dir / "run_summary.json"
            run_summary.write_text(json.dumps({"old": True}), encoding="utf-8")

            output = io.StringIO()
            with (
                patch(
                    "src.evaluate_logs.load_dyna_jsonl",
                    return_value=[SimpleNamespace(instance_id=task_id)],
                ),
                patch("src.evaluate_logs.get_client", return_value=object()),
                patch("src.evaluate_logs.LogReader", return_value=object()),
                patch("src.evaluate_logs.TaskEvaluator", FailingTaskEvaluator),
                redirect_stdout(output),
            ):
                errors = evaluate_run_dir(
                    run_dir=run_dir,
                    dataset_path="unused.jsonl",
                    force=True,
                )

            self.assertEqual(errors, 1)
            self.assertIn("EVALUATOR ERROR", output.getvalue())
            self.assertIn("retry 2/3", output.getvalue())
            self.assertFalse(eval_file.exists())
            self.assertFalse(run_summary.exists())

    def test_evaluator_error_retries_without_force(self):
        calls = {"n": 0}

        class FlakyTaskEvaluator:
            def __init__(self, *args, **kwargs):
                pass

            def evaluate(self):
                calls["n"] += 1
                if calls["n"] <= 3:
                    raise EvaluatorError("constraint decomposition failed: timeout")
                return {
                    "is_correct": True,
                    "correctness_score": 1.0,
                    "eval_status": "evaluated",
                    "diagnostic_counts": {"confirmed": 1},
                }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            task_id = "task_1"
            task_dir = run_dir / task_id
            eval_dir = run_dir / "evaluations"
            task_dir.mkdir(parents=True)
            eval_dir.mkdir()
            (task_dir / "summary.json").write_text("{}", encoding="utf-8")
            leftover = eval_dir / f"{task_id}.evaluation.json"
            leftover.write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "result": {
                            "eval_status": "evaluator_error",
                            "exclude_from_official": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "src.evaluate_logs.load_dyna_jsonl",
                    return_value=[SimpleNamespace(instance_id=task_id)],
                ),
                patch("src.evaluate_logs.get_client", return_value=object()),
                patch("src.evaluate_logs.LogReader", return_value=object()),
                patch("src.evaluate_logs.TaskEvaluator", FlakyTaskEvaluator),
                redirect_stdout(io.StringIO()),
            ):
                first = evaluate_run_dir(
                    run_dir=run_dir,
                    dataset_path="unused.jsonl",
                )
                self.assertEqual(first, 1)
                self.assertFalse(leftover.exists())

                second = evaluate_run_dir(
                    run_dir=run_dir,
                    dataset_path="unused.jsonl",
                )

            self.assertEqual(second, 0)
            self.assertEqual(calls["n"], 4)
            payload = json.loads(leftover.read_text(encoding="utf-8"))
            self.assertTrue(payload["result"]["is_correct"])

    def test_successful_eval_is_skipped_without_force(self):
        class BoomEvaluator:
            def __init__(self, *args, **kwargs):
                raise AssertionError("should skip existing evaluation")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            task_id = "task_1"
            task_dir = run_dir / task_id
            eval_dir = run_dir / "evaluations"
            task_dir.mkdir(parents=True)
            eval_dir.mkdir()
            (task_dir / "summary.json").write_text("{}", encoding="utf-8")
            eval_file = eval_dir / f"{task_id}.evaluation.json"
            eval_file.write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "result": {
                            "eval_status": "evaluated",
                            "is_correct": True,
                            "correctness_score": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "src.evaluate_logs.load_dyna_jsonl",
                    return_value=[SimpleNamespace(instance_id=task_id)],
                ),
                patch("src.evaluate_logs.TaskEvaluator", BoomEvaluator),
                redirect_stdout(io.StringIO()),
            ):
                errors = evaluate_run_dir(
                    run_dir=run_dir,
                    dataset_path="unused.jsonl",
                )

            self.assertEqual(errors, 0)


class EvaluateLogsRetryTests(unittest.TestCase):
    def test_evaluator_error_retries_from_scratch_then_succeeds(self):
        calls = {"n": 0}

        class FlakyTaskEvaluator:
            def __init__(self, *args, **kwargs):
                pass

            def evaluate(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise EvaluatorError("page evidence verification failed: no json")
                return {
                    "is_correct": True,
                    "correctness_score": 1.0,
                    "eval_status": "evaluated",
                    "diagnostic_counts": {"confirmed": 1},
                }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            task_id = "task_1"
            task_dir = run_dir / task_id
            eval_dir = run_dir / "evaluations"
            task_dir.mkdir(parents=True)
            eval_dir.mkdir()
            (task_dir / "summary.json").write_text("{}", encoding="utf-8")

            output = io.StringIO()
            with (
                patch(
                    "src.evaluate_logs.load_dyna_jsonl",
                    return_value=[SimpleNamespace(instance_id=task_id)],
                ),
                patch("src.evaluate_logs.get_client", return_value=object()),
                patch("src.evaluate_logs.LogReader", return_value=object()),
                patch("src.evaluate_logs.TaskEvaluator", FlakyTaskEvaluator),
                redirect_stdout(output),
            ):
                errors = evaluate_run_dir(
                    run_dir=run_dir,
                    dataset_path="unused.jsonl",
                    force=True,
                )

            self.assertEqual(errors, 0)
            self.assertEqual(calls["n"], 2)
            self.assertIn("retry 2/3", output.getvalue())
            self.assertTrue((eval_dir / f"{task_id}.evaluation.json").exists())


class JsonCallRetryTests(unittest.TestCase):
    def test_discards_empty_content_and_retries_without_thinking(self):
        calls = {"n": 0}

        def fake_llm_call(**kwargs):
            calls["n"] += 1
            self.assertFalse(kwargs.get("allow_reasoning_fallback"))
            if calls["n"] == 1:
                self.assertIsNone(kwargs.get("thinking"))
                return ""
            self.assertIs(kwargs.get("thinking"), False)
            return '{"ok": true}'

        with patch("src.utils.llm.llm_call", side_effect=fake_llm_call):
            from src.utils.llm import llm_json_call

            payload = llm_json_call(object(), "sys", "user", model="deepseek-flash")
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(calls["n"], 2)


class MultiLevelPageMatchingTests(unittest.TestCase):
    def test_token_match_handles_semantically_same_page_title(self):
        level, score = _frame_match_info(
            subject="DOJ Archives Press Releases page",
            assertion="page is filtered to Component: Antitrust Division",
            axtree=(
                "RootWebArea 'Archives | Press Releases Archive | United States "
                "Department of Justice'\n"
                "checkbox 'Antitrust Division (51)', checked='true'"
            ),
            url="https://www.justice.gov/archives/press-releases-archive",
        )
        self.assertEqual(level, "token")
        self.assertGreaterEqual(score, 0.6)

    def test_semantic_fallback_is_judged_instead_of_immediate_failure(self):
        class FakeLogReader:
            def get_events(self):
                return [{"step": 1, "url": "https://example.test/results"}]

            def get_axtree(self, step):
                return "RootWebArea 'Results'\nStaticText 'relevant evidence'"

        evaluator = TaskEvaluator(object(), FakeLogReader(), object(), "test-model")
        state = ScanState("c1", "constraint", "i0", True)
        claim = SubClaim(
            claim_id=1,
            type="page",
            description="verify renamed destination",
            subject="completely different wording",
            assertion="relevant evidence",
        )

        with patch(
            "src.evaluator_lib.core.verify_claim_on_frame",
            return_value={
                "is_subject_page": True,
                "assertion_supported": True,
                "evidence_summary": "page contains relevant evidence",
                "confidence": 0.9,
                "reason": "semantic match",
            },
        ) as judge:
            result = evaluator._resolve_page_subclaim(state, claim, {"i0": 1}, 1)

        self.assertTrue(result["satisfied"])
        self.assertEqual(result["match_level"], "semantic_fallback")
        judge.assert_called_once()


class PageCandidateStepsTests(unittest.TestCase):
    def test_cited_url_steps_precede_last_step_and_injection_window(self):
        done = (
            "Year One: https://www.dc.com/graphic-novels/injustice-gods-among-us-2013/"
            "injustice-gods-among-us-year-one-the-complete-collection\n"
            "Pick: https://www.dc.com/comics/dceased-4"
        )
        events = [
            {"step": 9, "url": "https://www.dc.com/graphic-novels/injustice-gods-among-us-2013/injustice-gods-among-us-year-one-the-complete-collection"},
            {"step": 16, "url": "https://www.dc.com/talent/tom-taylor"},
            {"step": 17, "url": "https://www.dc.com/comics/detective-comics-2016/detective-comics-1017"},
            {"step": 20, "url": "https://www.dc.com/comics/dceased-4"},
        ]
        steps = page_candidate_steps(done, events, start_step=16, last_step=20)
        self.assertEqual(steps[0], 9)
        self.assertEqual(steps[1], 20)
        self.assertEqual(steps[2], 16)
        self.assertEqual(steps[3], 17)
        self.assertEqual(steps.count(20), 1)

    def test_trailing_slash_still_matches_cited_url(self):
        done = "https://www.dc.com/comics/dceased-4/"
        events = [{"step": 4, "url": "https://www.dc.com/comics/dceased-4"}]
        steps = page_candidate_steps(done, events, start_step=3, last_step=4)
        self.assertEqual(steps[0], 4)
        self.assertEqual(steps.count(4), 1)

    def test_no_cited_url_keeps_old_window_order(self):
        events = [
            {"step": 1, "url": "https://www.dc.com/comics"},
            {"step": 2, "url": "https://www.dc.com/comics"},
        ]
        steps = page_candidate_steps("no links here", events, start_step=1, last_step=2)
        self.assertEqual(steps, [2, 1])


if __name__ == "__main__":
    unittest.main()
