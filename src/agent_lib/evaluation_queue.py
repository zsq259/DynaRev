"""Bounded evaluation workers consuming finalized execution results.

Only the coordinator writes the run aggregate. Execution workers enqueue after
all environment retries; filesystem appearance is never a readiness signal.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

from src.evaluate_logs import (
    evaluate_completed_task,
    evaluation_run_lock,
    summarize_evaluations,
)


class EvaluationQueue:
    def __init__(self, run_dir, evaluator_model, workers, retries, logger,
                 execution_date=""):
        if workers < 1 or retries < 0 or not evaluator_model:
            raise ValueError("evaluation requires a model, workers >= 1 and retries >= 0")
        self.run_dir = Path(run_dir)
        self.model = evaluator_model
        self.retries = retries
        self.logger = logger
        self.execution_date = execution_date
        self._lock = threading.Lock()
        self._seen = set()
        self._futures = []
        self._finished = False
        self._errors = 0
        self._run_lock = evaluation_run_lock(self.run_dir)
        self._run_lock.__enter__()
        try:
            self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dyna-eval")
        except BaseException:
            self._run_lock.__exit__(None, None, None)
            raise
        logger.info("EVAL_START | workers=%d | model=%s", workers, evaluator_model)

    def submit(self, task):
        with self._lock:
            if self._finished:
                raise RuntimeError("evaluation queue is closed")
            if task.instance_id in self._seen:
                return
            self._seen.add(task.instance_id)
            self._futures.append(self._pool.submit(self._score, task))
        self.logger.info("EVAL_QUEUED | task=%s", task.instance_id)

    def _score(self, task):
        try:
            return evaluate_completed_task(
                run_dir=self.run_dir, task=task, evaluator_model=self.model,
                execution_date=self.execution_date, eval_retries=self.retries,
                say=lambda message: self.logger.info("%s", message),
            )
        except Exception:
            self.logger.exception("EVAL_ERROR | task=%s", task.instance_id)
            return "error"

    def finish(self, cancel_pending=False):
        """Drain once and release the run lock even if aggregation fails."""
        if self._finished:
            return self._errors
        with self._lock:
            self._finished = True
        try:
            self._pool.shutdown(wait=True, cancel_futures=cancel_pending)
            outcomes = ["cancelled" if f.cancelled() else f.result() for f in self._futures]
            self._errors = sum(outcome in {"error", "cancelled"} for outcome in outcomes)
            summarize_evaluations(self.run_dir, self.model)
            self.logger.info(
                "EVAL_END | evaluated=%d | skipped=%d | errors=%d",
                outcomes.count("evaluated"), outcomes.count("skipped"), self._errors,
            )
            return self._errors
        finally:
            self._run_lock.__exit__(None, None, None)
