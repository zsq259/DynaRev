"""Action extraction and normalization from agent LLM output."""

from __future__ import annotations

import re

ACTION_PATTERN = re.compile(
    r'^(goto|click|fill|press|scroll|send_msg_to_user|report_infeasible)\s*\(',
    re.IGNORECASE,
)
# Line-leading action verbs (optional markdown fences around the call).
_ACTION_START = re.compile(
    r'(?m)^[ \t]*(?:`{1,3})?(goto|click|fill|press|scroll|send_msg_to_user|report_infeasible)\s*\(',
    re.IGNORECASE,
)
BARE_URL_PATTERN = re.compile(r"^https?://\S+$")
NUMERIC_CLICK_PATTERN = re.compile(r'^click\((\d+)\)$', re.IGNORECASE)


def _strip_action_wrappers(text: str) -> str:
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
    if t.startswith("`") and t.endswith("`") and len(t) >= 2:
        t = t[1:-1].strip()
    return t


def normalize_action(action: str) -> str:
    match = NUMERIC_CLICK_PATTERN.match(action.strip())
    if match:
        return f'click("{match.group(1)}")'
    return action


def _complete_action_call(text: str, start: int) -> str | None:
    """Return the full action call starting at ``start``, or None if incomplete.

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
                    return text[start : i + 1].strip()
        i += 1
    return None


def extract_action(raw: str) -> str:
    """Extract one executable action call from model output.

    Prefers the last complete action call whose verb starts a line. Supports
    multiline string arguments (e.g. ``send_msg_to_user("...\\n...")``). Incomplete
    calls are ignored so the runner can retry instead of executing a truncated
    action.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not cleaned:
        return ""

    # Candidate starts: line-leading action verbs, last match first.
    for match in reversed(list(_ACTION_START.finditer(cleaned))):
        # Point at the verb, not leading whitespace / fences.
        start = match.start(1)
        call = _complete_action_call(cleaned, start)
        if call is None:
            continue
        call = _strip_action_wrappers(call)
        if ACTION_PATTERN.match(call):
            return normalize_action(call)

    # Legacy fallbacks for single-line bare URLs / last line.
    lines = [
        _strip_action_wrappers(line.strip())
        for line in cleaned.splitlines()
        if line.strip()
    ]
    for line in reversed(lines):
        if BARE_URL_PATTERN.match(line):
            return f'goto("{line}")'
        # Single-line complete action only (do not accept truncated prefixes).
        if ACTION_PATTERN.match(line) and _complete_action_call(line, 0) == line:
            return normalize_action(line)

    return ""
