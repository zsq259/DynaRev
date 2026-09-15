"""Conversation history manager for agent execution.

Separates the authoritative conversation log from the LLM context window.
The full log records everything; build_llm_context() selects what the policy
model actually sees based on history_policy.
"""

from __future__ import annotations

from typing import Dict, List


class ConversationHistory:
    """Manages conversation messages for a single task episode.

    Core invariant:
    - _full_log: ALL messages ever recorded (always grows)
    - build_llm_context(): returns a NEW list built from the policy rules,
      never mutates internal state
    """

    def __init__(self, system_content: str, history_policy: str = "full") -> None:
        if history_policy not in ("full", "kill"):
            raise ValueError(f"Unsupported history_policy: {history_policy}")
        self._system_content = system_content
        self._policy = history_policy
        self._full_log: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Recording (always appends to _full_log)
    # ------------------------------------------------------------------

    def record_user_message(self, content: str) -> None:
        self._full_log.append({"role": "user", "content": content})

    def record_assistant_message(self, content: str) -> None:
        self._full_log.append({"role": "assistant", "content": content})

    def record_correction_round(
        self, correction: str, retry_response: str
    ) -> None:
        """Record a correction + retry pair (action parse failure)."""
        self._full_log.append({"role": "user", "content": correction})
        self._full_log.append({"role": "assistant", "content": retry_response})

    # ------------------------------------------------------------------
    # Context building (read-only)
    # ------------------------------------------------------------------

    def build_llm_context(
        self, current_user_msg: str
    ) -> List[Dict[str, str]]:
        """Return the message list to send to the policy model.

        In 'full' mode: system + all recorded turns + current user message.
        In 'kill' mode: system + current user message only.
        """
        ctx: List[Dict[str, str]] = [
            {"role": "system", "content": self._system_content}
        ]
        if self._policy == "full":
            ctx.extend(self._full_log)
        ctx.append({"role": "user", "content": current_user_msg})
        return ctx

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded messages (called on browser reset)."""
        self._full_log.clear()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def policy(self) -> str:
        return self._policy

    def get_full_log(self) -> List[Dict[str, str]]:
        return list(self._full_log)
