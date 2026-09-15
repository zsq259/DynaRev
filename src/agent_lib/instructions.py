from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from openai import OpenAI

from ..models import Revision
from ..utils.llm import llm_call, strip_thinking
from .milestones import build_instruction_block


@dataclass
class InstructionSnapshot:
    text: str
    policy: str
    key: str
    rewritten: bool = False


class InstructionComposer:
    """Build the active task instruction shown to the policy model."""

    def __init__(
        self,
        policy: str,
        client: OpenAI,
        model: str,
        temperature: float = 0.0,
    ) -> None:
        self.policy = policy
        self.client = client
        self.model = model
        self.temperature = temperature
        self._cache: Dict[Tuple[str, ...], InstructionSnapshot] = {}
        self.rewrite_records: List[dict] = []

    def compose(
        self,
        i0: str,
        revisions: Dict[str, Revision],
        fired_revision_ids: List[str],
    ) -> InstructionSnapshot:
        key_tuple = tuple(fired_revision_ids)
        key = ",".join(key_tuple) if key_tuple else "i0"
        cached = self._cache.get(key_tuple)
        if cached is not None:
            return cached

        structured = build_instruction_block(i0, revisions, fired_revision_ids)
        if self.policy == "structured":
            snapshot = InstructionSnapshot(
                text=structured,
                policy=self.policy,
                key=key,
                rewritten=False,
            )
            self._cache[key_tuple] = snapshot
            return snapshot

        if self.policy != "rewrite":
            raise ValueError(f"Unsupported instruction_policy: {self.policy}")

        prompt = (
            "Rewrite the current multi-turn GUI task into ONE complete, "
            "user-facing task instruction.\n\n"
            "Rules:\n"
            "- Preserve every active requirement.\n"
            "- Do not add new requirements.\n"
            "- For [CHANGED] updates, replace or relax only the conflicting "
            "earlier requirement; keep unrelated requirements.\n"
            "- For [CONDITIONAL] updates, keep the condition in the rewritten "
            "instruction. Do not decide the branch during rewriting.\n"
            "- Do not mention revisions, turns, messages, or this rewrite step.\n"
            "- Output only the rewritten instruction, no bullets, no explanation.\n\n"
            f"Current active instructions:\n{structured}"
        )
        messages = [
            {
                "role": "system",
                "content": "You rewrite dynamic GUI task instructions faithfully.",
            },
            {"role": "user", "content": prompt},
        ]
        raw = llm_call(
            client=self.client,
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=512,
            usage_tag="rewrite",
        )
        rewritten = strip_thinking(raw).strip()
        snapshot = InstructionSnapshot(
            text=rewritten or structured,
            policy=self.policy,
            key=key,
            rewritten=True,
        )
        self._cache[key_tuple] = snapshot
        self.rewrite_records.append(
            {
                "key": key,
                "fired_revision_ids": list(fired_revision_ids),
                "model": self.model,
                "structured_instruction": structured,
                "raw_output": raw,
                "rewritten_instruction": snapshot.text,
            }
        )
        return snapshot
