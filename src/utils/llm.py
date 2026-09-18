"""Shared LLM call utilities for the Dyna project.

Merges pipeline's reasoning-model support + JSON parsing with runner's retry logic.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token counts from one chat.completions response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tag: str = "unspecified"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tag": self.tag,
        }


@dataclass
class TokenUsageMeter:
    """Accumulate token usage across LLM calls within one task episode."""

    calls: List[TokenUsage] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def add(self, usage: TokenUsage) -> None:
        self.calls.append(usage)

    def totals(self, start: int = 0) -> Dict[str, Any]:
        """Aggregate counts from ``calls[start:]`` (inclusive)."""
        slice_ = self.calls[start:]
        by_tag: Dict[str, Dict[str, int]] = {}
        prompt = completion = total = 0
        for item in slice_:
            prompt += item.prompt_tokens
            completion += item.completion_tokens
            total += item.total_tokens
            bucket = by_tag.setdefault(
                item.tag,
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "n_calls": 0},
            )
            bucket["prompt_tokens"] += item.prompt_tokens
            bucket["completion_tokens"] += item.completion_tokens
            bucket["total_tokens"] += item.total_tokens
            bucket["n_calls"] += 1
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "n_calls": len(slice_),
            "by_tag": by_tag,
        }


_TOKEN_METER: contextvars.ContextVar[Optional[TokenUsageMeter]] = contextvars.ContextVar(
    "dyna_token_meter", default=None
)


def get_token_meter() -> Optional[TokenUsageMeter]:
    return _TOKEN_METER.get()


@contextmanager
def using_token_meter(meter: TokenUsageMeter) -> Iterator[TokenUsageMeter]:
    """Bind ``meter`` for all ``llm_call`` invocations in this context."""
    token = _TOKEN_METER.set(meter)
    try:
        yield meter
    finally:
        _TOKEN_METER.reset(token)


def extract_token_usage(resp: Any, tag: str = "unspecified") -> TokenUsage:
    """Parse OpenAI-compatible ``usage`` from a chat completion response."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return TokenUsage(tag=tag)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        tag=tag,
    )


def strip_thinking(text: str) -> str:
    """Remove complete or orphan-closed thinking blocks from model output."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # Some OpenAI-compatible Qwen servers emit reasoning text followed by a
    # closing </think> without the opening tag. In that case, the actual answer
    # is the suffix after the final closing tag.
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1]
    return cleaned.strip()


def try_parse_json(raw: str) -> Dict[str, Any]:
    """Try to extract and parse the outermost JSON object from text."""
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidate = raw[start : end + 1]
        return json.loads(candidate)
    raise ValueError(f"No JSON object found in: {raw[:300]}")


def find_last_json_object(raw: str) -> Dict[str, Any]:
    """
    Find all top-level JSON objects in raw text and return the last one.
    Useful when reasoning content contains partial JSON before the real answer.
    """
    results = []
    i = 0
    while i < len(raw):
        if raw[i] == "{":
            depth = 0
            j = i
            in_string = False
            escape = False
            while j < len(raw):
                c = raw[j]
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = not in_string
                elif not in_string:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(raw[i : j + 1])
                                results.append(obj)
                            except json.JSONDecodeError:
                                pass
                            break
                j += 1
            i = j + 1
        else:
            i += 1
    if results:
        # Return the last (and usually largest/most complete) JSON object
        return max(results, key=lambda o: len(json.dumps(o)))
    raise ValueError(f"Could not parse any JSON object from: {raw[:300]}")


def _is_deepseek_model(model: str) -> bool:
    """Check if a model name indicates a DeepSeek model (needs thinking mode)."""
    return model.startswith("deepseek")


def llm_call(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    retries: int = 3,
    reasoning_effort: str = "high",
    usage_tag: str = "unspecified",
    allow_reasoning_fallback: bool = True,
    thinking: Optional[bool] = None,
) -> str:
    """General-purpose LLM call with retry logic and DeepSeek thinking support.

    - DeepSeek models: enables thinking mode via extra_body, skips temperature,
      increases max_tokens, handles reasoning_content fallback, strips <think>
      blocks. Pass ``thinking=False`` to disable thinking for this call.
    - Non-DeepSeek models: standard call with temperature.
    - Retries with exponential backoff on failure.
    - When a ``TokenUsageMeter`` is bound via ``using_token_meter``, records
      prompt/completion/total tokens under ``usage_tag``.
    - ``allow_reasoning_fallback``: if content is empty, parse
      ``reasoning_content`` (policy/agent). JSON judges should pass False
      and retry the call instead of treating a CoT dump as the answer.
    """
    is_deepseek = _is_deepseek_model(model)

    if is_deepseek:
        max_tokens = max(max_tokens, 16384)

    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

    if is_deepseek:
        enable_thinking = True if thinking is None else thinking
        if enable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            kwargs["temperature"] = temperature
    else:
        kwargs["temperature"] = temperature

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            meter = get_token_meter()
            if meter is not None:
                meter.add(extract_token_usage(resp, tag=usage_tag))

            msg = resp.choices[0].message
            raw = (msg.content or "").strip()

            # DeepSeek thinking mode may put content in reasoning_content
            if not raw and hasattr(msg, "reasoning_content") and msg.reasoning_content:
                if allow_reasoning_fallback:
                    logger.warning(
                        "LLM returned empty content (reasoning_content length=%d). "
                        "Attempting to parse reasoning_content as fallback.",
                        len(msg.reasoning_content),
                    )
                    raw = msg.reasoning_content.strip()
                else:
                    logger.warning(
                        "LLM returned empty content (reasoning_content length=%d). "
                        "Not using reasoning_content as the answer.",
                        len(msg.reasoning_content),
                    )

            # Safe for all providers and needed for Qwen endpoints that return
            # an orphan closing </think> tag in message.content.
            raw = strip_thinking(raw)

            return raw
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    return ""


def llm_json_call(
    client: OpenAI,
    system: str,
    user: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 16384,
    reasoning_effort: str = "high",
    parse_retries: int = 3,
) -> Dict[str, Any]:
    """Call LLM and parse the response as JSON.

    Failed parses discard the bad output and retry from the original
    messages. Empty ``content`` is not filled in from thinking/CoT.
    After the first failed attempt on a DeepSeek model, thinking is
    disabled so the next try can emit JSON in ``content``.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_err: Exception | None = None
    for attempt in range(1, parse_retries + 1):
        thinking = False if attempt > 1 and _is_deepseek_model(model) else None
        try:
            raw = llm_call(
                client=client,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                retries=1,
                allow_reasoning_fallback=False,
                thinking=thinking,
            )
        except Exception as exc:
            last_err = exc
            logger.warning(
                "JSON_CALL_RETRY | attempt=%d/%d | llm_call failed: %s",
                attempt,
                parse_retries,
                exc,
            )
            continue
        if not raw.strip():
            last_err = ValueError("empty model content (no JSON)")
            logger.warning(
                "JSON_CALL_RETRY | attempt=%d/%d | empty content",
                attempt,
                parse_retries,
            )
            continue
        try:
            return try_parse_json(raw)
        except (json.JSONDecodeError, ValueError):
            try:
                return find_last_json_object(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                last_err = exc
                logger.warning(
                    "JSON_CALL_RETRY | attempt=%d/%d | %s",
                    attempt,
                    parse_retries,
                    exc,
                )
    raise last_err or ValueError("JSON parse failed")
