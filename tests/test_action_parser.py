"""Tests for action extraction from a strict ```action fence."""

import ast

from src.agent_lib.action_parser import ACTION_PATTERN, extract_action


def _fence(body: str, prefix: str = "", suffix: str = "") -> str:
    return f"{prefix}```action\n{body}\n```{suffix}"


def test_single_line_click():
    raw = _fence('click("42")', prefix="Thought: click search\n")
    assert extract_action(raw) == 'click("42")'


def test_numeric_click_normalized():
    assert extract_action(_fence("click(173)")) == 'click("173")'


def test_multiline_send_msg_keeps_full_string():
    raw = _fence(
        'send_msg_to_user("done: filtered under $50) and On Sale.\n'
        "\n"
        "[CONDITIONAL OBSERVATION]\n"
        '- Observation: VILINICE speaker is $45.99 (on sale, Save $16).")'
    )
    action = extract_action(raw)
    assert ACTION_PATTERN.match(action)
    assert action.startswith("send_msg_to_user(")
    assert action.endswith(")")
    assert "VILINICE" in action
    assert "$45.99" in action
    assert "[CONDITIONAL OBSERVATION]" in action
    # Must not stop at the first ')' inside the price text.
    assert "under $50) and On Sale" in action
    ast.parse(action)
    inner = action[len("send_msg_to_user(") : -1]
    msg = ast.literal_eval(inner)
    assert "\n[CONDITIONAL OBSERVATION]\n" in msg
    assert "VILINICE speaker is $45.99" in msg


def test_incomplete_send_msg_returns_empty_for_retry():
    raw = _fence(
        'send_msg_to_user("done: under $50) and On Sale.\n'
        "\n"
        "[CONDITIONAL OBSERVATION]\n"
        "- still open string, no closing quote"
    )
    assert extract_action(raw) == ""


def test_prefers_last_action_fence():
    raw = (
        _fence('click("1")')
        + "\nmore text\n"
        + _fence('fill("2", "hello")')
    )
    assert extract_action(raw) == 'fill("2", "hello")'


def test_ignores_calls_outside_fence():
    raw = 'click("1")\n' + _fence('fill("2", "hello")') + '\nsend_msg_to_user("done: skip")'
    assert extract_action(raw) == 'fill("2", "hello")'


def test_bare_call_without_fence_is_rejected():
    assert extract_action('Thought: click search\nclick("42")') == ""


def test_extra_call_inside_fence_is_rejected():
    raw = _fence('click("1")\nfill("2", "hello")')
    assert extract_action(raw) == ""


def test_unlabeled_fence_is_rejected():
    raw = '```\nclick("42")\n```'
    assert extract_action(raw) == ""


def test_python_fence_is_rejected():
    raw = '```python\nclick("42")\n```'
    assert extract_action(raw) == ""


def test_dsml_without_fence_is_rejected():
    raw = (
        '<｜DSML｜invoke name="goto">\n'
        '<｜DSML｜parameter name="url">https://www.ign.com/</parameter>\n'
    )
    assert extract_action(raw) == ""


def test_unclosed_fence_is_rejected():
    raw = '```action\nclick("42")'
    assert extract_action(raw) == ""


def test_escaped_quotes_inside_string():
    raw = _fence(r'send_msg_to_user("done: he said \"hi\" and left")')
    assert extract_action(raw) == r'send_msg_to_user("done: he said \"hi\" and left")'


def test_already_escaped_newline_is_left_alone():
    raw = _fence(r'send_msg_to_user("done: line1\nline2")')
    action = extract_action(raw)
    assert action == r'send_msg_to_user("done: line1\nline2")'
    ast.parse(action)
    assert ast.literal_eval(action[len("send_msg_to_user(") : -1]) == "done: line1\nline2"


def test_fill_with_raw_newline_is_python_executable():
    raw = _fence('fill("12", "line1\nline2")')
    action = extract_action(raw)
    call = ast.parse(action, mode="eval").body
    assert call.func.id == "fill"
    assert call.args[0].value == "12"
    assert call.args[1].value == "line1\nline2"


def test_bare_url_without_fence_is_rejected():
    assert extract_action("https://www.example.com/path") == ""
