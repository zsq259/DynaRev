"""Data models for the pipeline intermediate and final outputs."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Constraint:
    id: int
    description: str


@dataclass
class AlignedConstraint:
    """Constraint after Step 3c; may carry a conditional branch tag."""
    id: int
    description: str
    branch: Optional[str] = None  # None | "default" | "alternative"


@dataclass
class AlignResult:
    """Step 3c output: final constraint wording (and groups for conditional)."""
    original_instruction: str
    constraints: List[AlignedConstraint]


@dataclass
class ConstraintResult:
    """Step 1 output."""
    original_instruction: str
    constraints: List[Constraint]


@dataclass
class I0Result:
    """Step 2 output."""
    i0: str
    classification: List[Dict] = field(default_factory=list)
    i0_constraint_ids: List[int] = field(default_factory=list)


@dataclass
class Instruction:
    id: int
    instruction: str
    milestone: str = ""
    revision_type: str = "additive"
    constraint_ids: List[int] = field(default_factory=list)
    # Filled by Step 3b LLM only when revision_type == "conditional_override":
    # a clean predicate sentence used by the evaluator at runtime.
    condition_text: str = ""


@dataclass
class MultiturnResult:
    """Step 3 output."""
    instructions: List[Instruction]
    reflection: dict = field(default_factory=dict)


@dataclass
class SwitchingCondition:
    """Page-observable condition attached to a conditional override.

    Step 3b picks ``dimension`` from {count, price, rating, time} and fills
    operator / threshold / unit. ``condition_text`` is the evaluator predicate
    written in the same call.
    """
    dimension: str           # "count" | "price" | "rating" | "time"
    operator: str            # ">", "<", ">=", "<="
    threshold: Optional[float] = None
    unit: str = ""
    default_option: str = ""    # original (default-branch) constraint summary
    alt_option: str = ""        # alternative (alt-branch) constraint summary
    condition_text: str = ""    # natural-language predicate, written by Step 3b LLM


@dataclass
class OverrideItem:
    """One (anchor constraint, library pair) item.

    Step 3a fills candidates with the seed constraint and both pair texts
    as unlabeled references. It does not decide direction.
    After Step 3b, ``target_constraint_text`` / ``replacement_constraint_text``
    are the LLM's keep / change_to gold (wording may differ from the pair
    and from the seed constraint).
    """
    target_constraint_id: int
    replacement_pair_id: str
    target_constraint_text: str = ""
    replacement_constraint_text: str = ""
    pair_a: str = ""
    pair_b: str = ""
    dimension: str = ""
    score: float = 0.0
    # Conditional override only: present when override_mode == "conditional"
    switching_condition: Optional[SwitchingCondition] = None


@dataclass
class OverridePlanResult:
    """Step 3a candidate pool; Step 3b fills ``overrides`` after LLM selection."""
    enable_override: bool
    overrides: List[OverrideItem] = field(default_factory=list)
    candidates: List[OverrideItem] = field(default_factory=list)
    # "unconditional" (legacy default) | "conditional"
    override_mode: str = "unconditional"


@dataclass
class ReviewResult:
    """Step 4 output."""
    passed: bool
    worth_retry: Optional[bool] = None  # only set when passed=False
    retry_from_step: Optional[str] = None  # "step2" or "step3", only when worth_retry=True
    reason: str = ""


@dataclass
class BindingEntry:
    """A single constraint-to-instruction mapping."""
    constraint_id: int
    instruction_id: str  # "i0", "i1", etc.
    confidence: str  # "high", "medium", "low"
    reason: str = ""


@dataclass
class BindingResult:
    """Step 5 output: constraint-instruction binding + audit."""
    mapping: List[BindingEntry]
    issues: List[str] = field(default_factory=list)
    passed: bool = True
    retry_from_step: Optional[str] = None  # "step2" or "step3", only when passed=False


@dataclass
class PipelineInput:
    """A single task to be processed by the pipeline."""
    task_id: str
    website: str
    original_instruction: str
    scene: str = ""
    dataset: str = "osunlp/Online-Mind2Web"


@dataclass
class PipelineOutput:
    """Full pipeline result for one task."""
    input: PipelineInput
    step1: Optional[ConstraintResult] = None
    step2: Optional[I0Result] = None
    step3a: Optional['OverridePlanResult'] = None
    step3: Optional[MultiturnResult] = None
    step3c: Optional[AlignResult] = None
    step4: Optional[ReviewResult] = None
    step5: Optional['BindingResult'] = None
    attempt_records: List[Dict[str, Any]] = field(default_factory=list)
    filtered_at: Optional[str] = None  # which step filtered it out
    filter_reason: str = ""
    homepage_description: str = ""
