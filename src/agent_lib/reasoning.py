"""Reasoning strategies for agent action generation.

Provides three reasoning modes:
- Direct:     output action directly (current baseline)
- ReAct:      Thought → Action in one LLM call
- Reflection: post-action self-critique via extra LLM call
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from openai import OpenAI

from ..utils.llm import llm_call


_THOUGHT_PATTERN = re.compile(r"^\s*Thought:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

_REFLECTION_PROMPT = """You are reviewing a browser agent's last action.

Last action taken: {action}
Current page state (AXTree excerpt):
{axtree}

Critique the last action:
1. Was it the right action to take at this point?
2. Did it move the task forward effectively?
3. What should the agent consider or be careful about next?

Output a brief, actionable reflection (2-4 sentences). Do NOT suggest specific next actions — only critique and warn about potential issues."""


class ReasoningStrategy(ABC):
    """Extension point for reasoning modes."""

    @abstractmethod
    def get_system_prompt_suffix(self) -> str:
        """Return text to append to the system prompt for this strategy."""
        ...

    @abstractmethod
    def process_output(self, raw: str) -> Tuple[str, Optional[str]]:
        """Extract (action, thought_or_none) from raw LLM output."""
        ...

    def post_action_reflect(
        self,
        *,
        action: str,
        axtree: str,
        client: OpenAI,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Optional[str]:
        """Post-action reflection. Returns reflection text or None.

        Only ReflectionStrategy overrides this. Direct and ReAct return None.
        """
        return None


class DirectStrategy(ReasoningStrategy):
    """Current baseline: output action directly, no reasoning step."""

    def get_system_prompt_suffix(self) -> str:
        return ""

    def process_output(self, raw: str) -> Tuple[str, Optional[str]]:
        from .action_parser import extract_action
        return extract_action(raw), None


class ReActStrategy(ReasoningStrategy):
    """Thought-then-Action in a single LLM call.

    System prompt is augmented to require a 'Thought:' line before the action.
    process_output extracts the thought from the raw output.
    No extra LLM call.
    """

    _SUFFIX = (
        "\n\nBefore each action, reason about what to do next. "
        'Output exactly one line starting with "Thought:" followed by your '
        "reasoning, then output the action call on the next line."
    )

    def get_system_prompt_suffix(self) -> str:
        return self._SUFFIX

    def process_output(self, raw: str) -> Tuple[str, Optional[str]]:
        from .action_parser import extract_action

        thought: Optional[str] = None
        match = _THOUGHT_PATTERN.search(raw)
        if match:
            thought = match.group(1).strip()
        return extract_action(raw), thought


class ReflectionStrategy(ReasoningStrategy):
    """Post-action self-critique via an extra LLM call per step.

    After each action (and before the next), an additional LLM call critiques
    the action. The reflection text is embedded into the next user message
    so the agent can learn from its mistakes within the trajectory.
    """

    def get_system_prompt_suffix(self) -> str:
        return ""

    def process_output(self, raw: str) -> Tuple[str, Optional[str]]:
        from .action_parser import extract_action
        return extract_action(raw), None

    def post_action_reflect(
        self,
        *,
        action: str,
        axtree: str,
        client: OpenAI,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Optional[str]:
        axtree_short = axtree[:6000]
        if len(axtree) > 6000:
            axtree_short += "\n...(truncated)"
        prompt = _REFLECTION_PROMPT.format(action=action, axtree=axtree_short)
        messages = [
            {"role": "system", "content": "You are a critical reviewer for a browser agent."},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = llm_call(
                client=client,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                usage_tag="reflection",
            )
            return raw.strip()
        except Exception:
            return None
