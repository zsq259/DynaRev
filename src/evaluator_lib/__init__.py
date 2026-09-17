"""Evaluator library: independent task evaluation from execution logs."""

from .constraint_judge import simple_verify, verify_claim_on_frame
from .core import TaskEvaluator
from .decomposer import SubClaim, decompose, extract_done_message
from .environment_error import (
    ENVIRONMENT_ERROR_LABEL,
    build_environment_error_result,
    is_environment_error,
)
from .errors import EvaluatorError
from .no_final_submit import (
    NO_FINAL_SUBMIT_LABEL,
    build_no_final_submit_result,
    is_no_final_submit,
)
from .response_judge import judge_response_window
from .response_metrics import (
    compute_csd_per_round,
    compute_demand_side_metrics,
    compute_injection_steps,
    slice_window,
)

__all__ = [
    "TaskEvaluator",
    "EvaluatorError",
    "NO_FINAL_SUBMIT_LABEL",
    "build_no_final_submit_result",
    "is_no_final_submit",
    "ENVIRONMENT_ERROR_LABEL",
    "build_environment_error_result",
    "is_environment_error",
    "SubClaim",
    "decompose",
    "extract_done_message",
    "simple_verify",
    "verify_claim_on_frame",
    "compute_demand_side_metrics",
    "compute_csd_per_round",
    "compute_injection_steps",
    "slice_window",
    "judge_response_window",
]
