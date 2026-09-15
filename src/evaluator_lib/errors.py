"""Errors raised when the evaluator itself cannot produce a valid judgment."""


class EvaluatorError(RuntimeError):
    """Infrastructure/model failure that must not be scored as an agent failure."""


def evaluator_error(stage: str, exc: Exception) -> EvaluatorError:
    """Build a concise, stage-qualified evaluator error."""
    detail = str(exc).strip() or exc.__class__.__name__
    return EvaluatorError(f"{stage}: {detail[:200]}")
