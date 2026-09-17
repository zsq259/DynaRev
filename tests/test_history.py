from src.agent_lib.history import (
    ConversationHistory,
    OMITTED_PAGE_CONTENT,
    elide_old_observations,
)


def _page(step: int, body: str) -> str:
    return (
        f"CURRENT INSTRUCTIONS (ALL active):\nFind a dataset.\n\n"
        f"CURRENT URL: https://example.com/p{step}\n\n"
        f"PAGE CONTENT:\n{body}"
    )


def test_elide_keeps_last_k_observations_and_all_actions():
    messages = [{"role": "system", "content": "sys"}]
    for step in range(1, 8):
        messages.append({"role": "user", "content": _page(step, f"TREE-{step}")})
        messages.append({"role": "assistant", "content": f'```action\nclick("{step}")\n```'})
    messages.append({"role": "user", "content": _page(8, "TREE-8")})

    clipped = elide_old_observations(messages, keep=5)
    user_obs = [m for m in clipped if m["role"] == "user"]
    assert len(user_obs) == 8
    omitted = [m for m in user_obs if "TREE-" not in m["content"]]
    kept = [m for m in user_obs if "TREE-" in m["content"]]
    assert len(omitted) == 3
    assert [m["content"] for m in kept] == [
        _page(4, "TREE-4"),
        _page(5, "TREE-5"),
        _page(6, "TREE-6"),
        _page(7, "TREE-7"),
        _page(8, "TREE-8"),
    ]
    assert all(OMITTED_PAGE_CONTENT in m["content"] for m in omitted)
    assert all("CURRENT URL:" in m["content"] for m in omitted)
    assistants = [m["content"] for m in clipped if m["role"] == "assistant"]
    assert assistants == [f'```action\nclick("{step}")\n```' for step in range(1, 8)]


def test_elide_does_not_drop_revision_or_correction_turns():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": _page(1, "TREE-1")},
        {"role": "assistant", "content": "click(1)"},
        {
            "role": "user",
            "content": "[INSTRUCTION UPDATE] Only keep Batman.",
        },
        {"role": "user", "content": _page(2, "TREE-2")},
        {"role": "assistant", "content": "click(2)"},
        {
            "role": "user",
            "content": "ERROR: wrap exactly one BrowserGym call in an ```action fence.",
        },
        {"role": "assistant", "content": "click(2b)"},
        {"role": "user", "content": _page(3, "TREE-3")},
    ]
    clipped = elide_old_observations(messages, keep=2)
    texts = [m["content"] for m in clipped if m["role"] == "user"]
    assert "[INSTRUCTION UPDATE] Only keep Batman." in texts
    assert any(t.startswith("ERROR:") for t in texts)
    assert "TREE-1" not in "".join(texts)
    assert "TREE-2" in "".join(texts)
    assert "TREE-3" in "".join(texts)


def test_full_policy_clips_pages_but_log_stays_complete():
    hist = ConversationHistory("sys", history_policy="full", observation_keep=5)
    for step in range(1, 8):
        hist.record_user_message(_page(step, f"TREE-{step}"))
        hist.record_assistant_message(f"action-{step}")
    ctx = hist.build_llm_context(_page(8, "TREE-8"))
    joined = "\n".join(m["content"] for m in ctx)
    assert "TREE-1" not in joined
    assert "TREE-3" not in joined
    assert "TREE-4" in joined
    assert "TREE-8" in joined
    assert all(f"action-{step}" in joined for step in range(1, 8))
    log = hist.get_full_log()
    assert "TREE-1" in log[0]["content"]
    assert len(log) == 14


def test_kill_policy_still_drops_prior_turns():
    hist = ConversationHistory("sys", history_policy="kill", observation_keep=5)
    hist.record_user_message(_page(1, "TREE-1"))
    hist.record_assistant_message("action-1")
    ctx = hist.build_llm_context(_page(2, "TREE-2"))
    assert [m["role"] for m in ctx] == ["system", "user"]
    assert "TREE-1" not in ctx[1]["content"]
    assert "TREE-2" in ctx[1]["content"]
