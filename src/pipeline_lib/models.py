"""Data models for the pipeline intermediate and final outputs."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Constraint:
    id: int
    description: str


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

    Step 3a picks the dimension/threshold/unit from
    ``data/pipeline/switching_conditions.yaml`` based on the pair dimension and
    task scene. ``condition_text`` is filled later by Step 3b's LLM —
    it writes a clean predicate that the evaluator will use against
    the agent's done_message.
    """
    dimension: str           # e.g. "price_diff", "availability_fail"
    operator: str            # ">", "<", ">=", "<=", "==true"
    threshold: Optional[float] = None
    unit: str = ""
    default_option: str = ""    # original (default-branch) constraint summary, fed to LLM
    alt_option: str = ""        # alternative (alt-branch) constraint summary, fed to LLM
    condition_text: str = ""    # natural-language predicate, written by Step 3b LLM


@dataclass
class OverrideItem:
    """Single override replacement decision from Step 3a."""
    target_constraint_id: int
    replacement_pair_id: str
    target_constraint_text: str = ""
    replacement_constraint_text: str = ""
    dimension: str = ""
    # Conditional override only: present when override_mode == "conditional"
    switching_condition: Optional[SwitchingCondition] = None


@dataclass
class OverridePlanResult:
    """Step 3a output: whether to apply override and which constraints to replace."""
    enable_override: bool
    overrides: List[OverrideItem] = field(default_factory=list)
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
    step4: Optional[ReviewResult] = None
    step5: Optional['BindingResult'] = None
    attempt_records: List[Dict[str, Any]] = field(default_factory=list)
    filtered_at: Optional[str] = None  # which step filtered it out
    filter_reason: str = ""
    homepage_description: str = ""
