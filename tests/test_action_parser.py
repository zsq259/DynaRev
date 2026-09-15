"""Tests for action extraction, including multiline send_msg_to_user."""

from src.agent_lib.action_parser import ACTION_PATTERN, extract_action


def test_single_line_click():
    raw = 'Thought: click search\nclick("42")'
    assert extract_action(raw) == 'click("42")'


def test_numeric_click_normalized():
    assert extract_action("click(173)") == 'click("173")'


def test_multiline_send_msg_keeps_full_string():
    raw = (
        "I am done.\n"
        'send_msg_to_user("done: filtered under $50) and On Sale.\n'
        "\n"
        "[CONDITIONAL OBSERVATION]\n"
        "- Observation: VILINICE speaker is $45.99 (on sale, Save $16).\")\n"
    )
    action = extract_action(raw)
    assert ACTION_PATTERN.match(action)
    assert action.startswith("send_msg_to_user(")
    assert action.endswith(")")
    assert "VILINICE" in action
    assert "$45.99" in action
    assert "[CONDITIONAL OBSERVATION]" in action
    # Must not stop at the first ')'' inside the price text.
    assert "under $50) and On Sale" in action


def test_incomplete_send_msg_returns_empty_for_retry():
    raw = (
        'send_msg_to_user("done: under $50) and On Sale.\n'
        "\n"
        "[CONDITIONAL OBSERVATION]\n"
        "- still open string, no closing quote\n"
    )
    assert extract_action(raw) == ""


def test_prefers_last_complete_action():
    raw = (
        'click("1")\n'
        "more text\n"
        'fill("2", "hello")\n'
    )
    assert extract_action(raw) == 'fill("2", "hello")'


def test_escaped_quotes_inside_string():
    raw = r'send_msg_to_user("done: he said \"hi\" and left")'
    assert extract_action(raw) == r'send_msg_to_user("done: he said \"hi\" and left")'


def test_bare_url_fallback():
    assert extract_action("https://www.example.com/path") == 'goto("https://www.example.com/path")'
