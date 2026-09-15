"""Preset agent configurations for ablation experiments.

Each preset is a validated combination of the three policy dimensions:
- history_policy: "full" | "kill"
- revision_policy: "append" | "reset"
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
        description="[保留+保留+多条] 完整版：完整对话历史 + 浏览器不重置 + 多条指令逐条呈现",
        history_policy="full",
        revision_policy="append",
        instruction_policy="structured",
    ),
    "merge": AgentPreset(
        key="merge",
        description="[保留+保留+合并] 完整对话历史 + 浏览器不重置 + 指令 LLM 合并为一条",
        history_policy="full",
        revision_policy="append",
        instruction_policy="rewrite",
    ),
    "kill_structured": AgentPreset(
        key="kill_structured",
        description="[保留+丢弃+多条] 丢弃对话历史 + 浏览器不重置 + 多条指令逐条呈现",
        history_policy="kill",
        revision_policy="append",
        instruction_policy="structured",
    ),
    "kill_rewrite": AgentPreset(
        key="kill_rewrite",
        description="[保留+丢弃+合并] Kill History：丢弃对话历史 + 浏览器不重置 + 指令合并",
        history_policy="kill",
        revision_policy="append",
        instruction_policy="rewrite",
    ),
    "reset_structured": AgentPreset(
        key="reset_structured",
        description="[重置+丢弃+多条] 浏览器重置 + 丢弃对话历史 + 多条指令逐条呈现",
        history_policy="kill",
        revision_policy="reset",
        instruction_policy="structured",
    ),
    "reset_rewrite": AgentPreset(
        key="reset_rewrite",
        description="[重置+丢弃+合并] 指令重写版：浏览器重置 + 丢弃对话历史 + 指令合并",
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
