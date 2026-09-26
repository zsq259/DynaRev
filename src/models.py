"""Shared data models for the Dyna project."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Constraint:
    id: int
    description: str
    source_revision: str = ""
    branch: Optional[str] = None  # None | "default" | "alternative" — set on conditional override branches


@dataclass
class Revision:
    revision_id: str
    type: str  # "additive" | "override" | "conditional_override"
    content: str
    block_user_reply_until_applied: bool
    constraint_ids: List[int] = field(default_factory=list)


@dataclass
class Milestone:
    milestone_id: str
    semantic_concept: str = ""
    inject_revision_id: str = ""
    notes: str = ""


@dataclass
class BranchPolicy:
    condition_text: str
    default_constraint_ids: List[int]
    alternative_constraint_ids: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def default_constraint_id(self) -> int:
        """First default id (compat for callers that expect a single id)."""
        return self.default_constraint_ids[0]

    @property
    def alternative_constraint_id(self) -> int:
        """First alternative id (compat for callers that expect a single id)."""
        return self.alternative_constraint_ids[0]


@dataclass
class TaskInstance:
    format_version: str
    instance_id: str
    source: Dict[str, Any]
    runtime: Dict[str, Any]
    i0: str
    i0_constraint_ids: List[int]
    revisions: List[Revision]
    milestones: List[Milestone]
    constraints: List[Constraint]
    branch_policy: Optional[BranchPolicy] = None


@dataclass
class StepEvent:
    step: int
    url: str
    action: str
    raw_output: str
    injected_revision_ids: List[str] = field(default_factory=list)
    trigger_checks: List[Dict[str, Any]] = field(default_factory=list)
    axtree_path: Optional[str] = None  # e.g., "step_001.axtree"
    instruction_policy: str = "structured"
    instruction_key: str = "i0"
    active_instruction: str = ""
    thought: Optional[str] = None          # ReAct thought text
    reflection: Optional[str] = None       # post-action reflection text
    reasoning_mode: str = "direct"         # which reasoning strategy was used
    token_usage: Optional[Dict[str, Any]] = None  # LLM tokens spent on this step
