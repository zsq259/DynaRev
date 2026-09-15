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
from src.evaluator_lib.core import ScanState, TaskEvaluator, _frame_match_info
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
            "src.evaluator_lib.constraint_judge.llm_call",
            side_effect=TimeoutError("request timed out"),
        ):
            with self.assertRaisesRegex(
                EvaluatorError, "simple constraint verification failed"
            ):
                simple_verify("constraint", "predicate", "done", object())

    def test_page_verifier_raises_on_llm_failure(self):
        with patch(
            "src.evaluator_lib.constraint_judge.llm_call",
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
            "src.evaluator_lib.response_judge.llm_call",
            side_effect=ConnectionError("connection reset"),
        ):
            with self.assertRaisesRegex(
                EvaluatorError, "response-window verification failed"
            ):
                judge_response_window("i1", "change", [], object())


class EvaluateLogsErrorHandlingTests(unittest.TestCase):
    def test_evaluator_error_writes_no_result_and_returns_error(self):
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
            self.assertIn("Evaluated: 0", output.getvalue())
            self.assertIn("Errors:    1", output.getvalue())
            self.assertFalse(eval_file.exists())
            self.assertFalse(run_summary.exists())


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


if __name__ == "__main__":
    unittest.main()
