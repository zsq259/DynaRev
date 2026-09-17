"""Protocol skip when a task dies from website/browser infrastructure.

These episodes are not agent failures. They must not enter TaskEvaluator
and must not enter official score denominators.
"""

from __future__ import annotations

from typing import Any, Dict

from src.models import TaskInstance

ENVIRONMENT_ERROR_LABEL = "environment_error"
ENVIRONMENT_ERROR_REASON = (
    "Episode ended in a website/browser environment failure; "
    "answer evaluator was not run and the task is excluded from official scores."
)


def is_environment_error(summary: Dict[str, Any]) -> bool:
    """True when the runner stopped on a website/browser failure."""
    status = summary.get("completion_status") or summary.get("stopped_reason")
    return status in {"environment_error", "env_error", "env_setup_error"}


def build_environment_error_result(
    task: TaskInstance,
    summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluation placeholder that is excluded from official aggregates."""
    summary = summary or {}
    return {
        "eval_status": ENVIRONMENT_ERROR_LABEL,
        "skip_reason": "environment_error",
        "exclude_from_official": True,
        "correctness_score": None,
        "is_correct": None,
        "constraint_results": [],
        "revision_results": [],
        "diagnostic_counts": {},
        "overall_reason": ENVIRONMENT_ERROR_REASON,
        "environment_attempt": summary.get("environment_attempt"),
        "environment_retry_exhausted": summary.get("environment_retry_exhausted"),
        "instance_id": task.instance_id,
    }
