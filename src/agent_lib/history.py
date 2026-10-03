"""Conversation history manager for agent execution.

Separates the authoritative conversation log from the LLM context window.

history_policy:
- full: keep dialogue across revision injections
- kill: pulse-clear the log when a revision is injected; between injections
  the policy model still sees accumulated turns (same context shaping as full)
"""

from __future__ import annotations

from typing import Dict, List


PAGE_CONTENT_MARKER = "PAGE CONTENT:"
OMITTED_PAGE_CONTENT = "[Page observation omitted; subsequent action retained]"


def is_observation_message(content: str) -> bool:
    """True when a user turn carries a page snapshot (not a revision/correction)."""
    return PAGE_CONTENT_MARKER in content


def elide_page_content(content: str) -> str:
    """Replace the PAGE CONTENT block; keep instructions, URL, and errors."""
    idx = content.find(PAGE_CONTENT_MARKER)
    if idx < 0:
        return content
    return content[:idx] + OMITTED_PAGE_CONTENT


def elide_old_observations(
    messages: List[Dict[str, str]],
    keep: int,
) -> List[Dict[str, str]]:
    """Keep the last ``keep`` observation user turns intact; elide earlier pages.

    Assistant turns, revision injections, and correction prompts are never elided.
    ``keep`` counts observation user messages from the end of ``messages``,
    including the current page.
    """
    if keep < 1:
        raise ValueError(f"observation keep must be >= 1, got {keep}")

    observation_indexes = [
        i
        for i, message in enumerate(messages)
        if message.get("role") == "user"
        and is_observation_message(message.get("content") or "")
    ]
    if len(observation_indexes) <= keep:
        return [dict(message) for message in messages]

    drop = set(observation_indexes[:-keep])
    clipped: List[Dict[str, str]] = []
    for i, message in enumerate(messages):
        if i in drop:
            clipped.append(
                {
                    "role": "user",
                    "content": elide_page_content(message["content"]),
                }
            )
        else:
            clipped.append(dict(message))
    return clipped


class ConversationHistory:
    """Manages conversation messages for a single task episode.

    Core invariant:
    - _full_log: ALL messages ever recorded (always grows)
    - build_llm_context(): returns a NEW list built from the policy rules,
      never mutates internal state
    """

    def __init__(
        self,
        system_content: str,
        history_policy: str = "full",
        observation_keep: int = 5,
    ) -> None:
        if history_policy not in ("full", "kill"):
            raise ValueError(f"Unsupported history_policy: {history_policy}")
        if observation_keep < 1:
            raise ValueError(
                f"observation_keep must be >= 1, got {observation_keep}"
            )
        self._system_content = system_content
        self._policy = history_policy
        self._observation_keep = observation_keep
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

        Both policies use the same context shape: system + recorded turns +
        current user message, with page snapshots older than the last
        ``observation_keep`` replaced by a placeholder (actions, thoughts,
        and revision injections stay). ``kill`` only differs by pulse-clearing
        the log at revision injection (see ``clear_at_revision``).
        """
        ctx: List[Dict[str, str]] = [
            {"role": "system", "content": self._system_content}
        ]
        ctx.extend(self._full_log)
        ctx.append({"role": "user", "content": current_user_msg})
        return elide_old_observations(ctx, self._observation_keep)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded messages."""
        self._full_log.clear()

    def clear_at_revision(self) -> bool:
        """Pulse-clear dialogue when history_policy=kill. Returns whether cleared."""
        if self._policy != "kill":
            return False
        self.reset()
        return True

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def policy(self) -> str:
        return self._policy

    @property
    def observation_keep(self) -> int:
        return self._observation_keep

    def get_full_log(self) -> List[Dict[str, str]]:
        return list(self._full_log)
