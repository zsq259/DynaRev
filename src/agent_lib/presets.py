"""Preset agent configurations for ablation experiments.

Each preset is a validated combination of the three policy dimensions:
- history_policy: "full" | "kill"
  (full keeps dialogue across injections; kill pulse-clears the log only when
  a revision is injected, then accumulates again like a normal task)
- revision_policy: "append" | "reset"
  (browser only: stay put vs reopen at start_url; does not touch dialogue)
- instruction_policy: "structured" | "rewrite"

Use --mode <key> to select a preset, or use the three individual flags directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class AgentPreset:
    key: str
    description: str
    history_policy: str      # "full" | "kill"
    revision_policy: str     # "append" | "reset"
    instruction_policy: str  # "structured" | "rewrite"


PRESETS: Dict[str, AgentPreset] = {
    "baseline": AgentPreset(
        key="baseline",
        description="[Keep history + keep browser + structured instructions] Full baseline",
        history_policy="full",
        revision_policy="append",
        instruction_policy="structured",
    ),
    "merge": AgentPreset(
        key="merge",
        description="[Keep history + keep browser + rewrite instructions] Merge instructions with an LLM after injection",
        history_policy="full",
        revision_policy="append",
        instruction_policy="rewrite",
    ),
    "kill_structured": AgentPreset(
        key="kill_structured",
        description="[Clear history on injection + keep browser + structured instructions] Accumulate history normally after clearing",
        history_policy="kill",
        revision_policy="append",
        instruction_policy="structured",
    ),
    "kill_rewrite": AgentPreset(
        key="kill_rewrite",
        description="[Clear history on injection + keep browser + rewrite instructions] Accumulate history normally after clearing",
        history_policy="kill",
        revision_policy="append",
        instruction_policy="rewrite",
    ),
    "reset_structured": AgentPreset(
        key="reset_structured",
        description="[Clear history on injection + reset browser to start URL + structured instructions] Restart with structured instructions",
        history_policy="kill",
        revision_policy="reset",
        instruction_policy="structured",
    ),
    "reset_rewrite": AgentPreset(
        key="reset_rewrite",
        description="[Clear history on injection + reset browser to start URL + rewrite instructions] Restart with rewritten instructions",
        history_policy="kill",
        revision_policy="reset",
        instruction_policy="rewrite",
    ),
}


def list_presets() -> str:
    """Return a human-readable list of available presets."""
    lines = ["Available presets (--mode):", ""]
    for key, p in PRESETS.items():
        lines.append(f"  {key:20s}  {p.description}")
    return "\n".join(lines)


def apply_preset(args) -> None:
    """If --mode is specified, override individual policy flags.

    When --mode is not set (None), this is a no-op and the three individual
    flags (--history-policy, --revision-policy, --instruction-policy) are
    used as-is for backward compatibility.
    """
    if not getattr(args, "mode", None):
        return

    preset = PRESETS[args.mode]
    args.history_policy = preset.history_policy
    args.revision_policy = preset.revision_policy
    args.instruction_policy = preset.instruction_policy
