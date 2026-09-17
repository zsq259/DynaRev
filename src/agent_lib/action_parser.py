"""Action extraction and normalization from agent LLM output."""

from __future__ import annotations

import re

ACTION_PATTERN = re.compile(
    r'^(goto|click|fill|press|scroll|send_msg_to_user|report_infeasible)\s*\(',
    re.IGNORECASE,
)
# A single call at the start of the fence body (optional wrapping backticks).
_CALL_HEAD = re.compile(
    r'^[ \t]*(?:`{1,3})?(goto|click|fill|press|scroll|send_msg_to_user|report_infeasible)\s*\(',
    re.IGNORECASE,
)
# Last ```action fence wins. Closing fence must be on its own line.
_ACTION_FENCE = re.compile(
    r"^[ \t]*```[ \t]*action[ \t]*\r?\n(.*?)(?:\r?\n)[ \t]*```[ \t]*$",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
NUMERIC_CLICK_PATTERN = re.compile(r'^click\((\d+)\)$', re.IGNORECASE)


def _strip_action_wrappers(text: str) -> str:
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
    if t.startswith("`") and t.endswith("`") and len(t) >= 2:
        t = t[1:-1].strip()
    return t


def escape_raw_newlines_in_strings(call: str) -> str:
    """Rewrite raw newlines inside quoted literals to ``\\n`` / ``\\r``.

    BrowserGym ``exec``s the action as Python. A quoted string that contains
    a real newline is a SyntaxError, so ``send_msg_to_user`` never runs and
    the episode does not terminate.
    """
    out: list[str] = []
    in_string = False
    quote_char = ""
    escape = False
    for ch in call:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
            elif ch == "\\":
                out.append(ch)
                escape = True
            elif ch == quote_char:
                out.append(ch)
                in_string = False
                quote_char = ""
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            else:
                out.append(ch)
            continue
        out.append(ch)
        if ch in ('"', "'"):
            in_string = True
            quote_char = ch
    return "".join(out)


def normalize_action(action: str) -> str:
    match = NUMERIC_CLICK_PATTERN.match(action.strip())
    if match:
        return f'click("{match.group(1)}")'
    return escape_raw_newlines_in_strings(action)


def _complete_action_span(text: str, start: int) -> tuple[str, int] | None:
    """Return ``(call, end_index)`` for the call starting at ``start``.

    Scans from the opening ``(`` with quote/escape awareness so newlines and
    ``)`` characters inside string literals do not truncate the call.
    """
    open_paren = text.find("(", start)
    if open_paren < 0:
        return None

    depth = 0
    in_string = False
    quote_char = ""
    escape = False
    i = open_paren
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False
                quote_char = ""
        else:
            if ch in ('"', "'"):
                in_string = True
                quote_char = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1].strip(), i + 1
        i += 1
    return None


def _exactly_one_call(body: str) -> str:
    """Return the single BrowserGym call in ``body``, or empty if invalid."""
    text = _strip_action_wrappers(body.strip())
    if not text:
        return ""

    match = _CALL_HEAD.match(text)
    if not match:
        return ""
    start = match.start(1)
    if text[:start].strip("`").strip():
        return ""

    span = _complete_action_span(text, start)
    if span is None:
        return ""
    call, end = span
    if text[end:].strip().strip("`").strip():
        return ""

    call = _strip_action_wrappers(call)
    if not ACTION_PATTERN.match(call):
        return ""
    return normalize_action(call)


def extract_action(raw: str) -> str:
    """Extract one executable action call from model output.

    Only the last closed `` ```action `` fence is considered. The fence body
    must contain exactly one complete BrowserGym call. Calls, URLs, XML, and
    DSML outside that fence are ignored so the runner can retry.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not cleaned:
        return ""

    fences = list(_ACTION_FENCE.finditer(cleaned))
    if not fences:
        return ""
    return _exactly_one_call(fences[-1].group(1))
