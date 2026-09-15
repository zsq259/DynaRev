"""Unit tests for LLM token usage metering."""

from types import SimpleNamespace

from src.utils.llm import (
    TokenUsage,
    TokenUsageMeter,
    extract_token_usage,
    llm_call,
    using_token_meter,
)


class _FakeCompletions:
    def __init__(self, usage):
        self._usage = usage

    def create(self, **kwargs):
        message = SimpleNamespace(content='click("1")', reasoning_content=None)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], usage=self._usage)


class _FakeClient:
    def __init__(self, usage):
        self.chat = SimpleNamespace(completions=_FakeCompletions(usage))


def test_extract_token_usage_from_response():
    resp = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    usage = extract_token_usage(resp, tag="policy")
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15
    assert usage.tag == "policy"


def test_meter_totals_by_tag_and_slice():
    meter = TokenUsageMeter()
    meter.add(TokenUsage(3, 1, 4, tag="trigger"))
    meter.add(TokenUsage(10, 2, 12, tag="policy"))
    meter.add(TokenUsage(7, 3, 10, tag="policy"))

    full = meter.totals()
    assert full["prompt_tokens"] == 20
    assert full["completion_tokens"] == 6
    assert full["total_tokens"] == 26
    assert full["n_calls"] == 3
    assert full["by_tag"]["policy"]["n_calls"] == 2
    assert full["by_tag"]["policy"]["total_tokens"] == 22

    since = meter.totals(start=1)
    assert since["n_calls"] == 2
    assert since["total_tokens"] == 22


def test_llm_call_records_usage_when_meter_bound():
    client = _FakeClient(
        SimpleNamespace(prompt_tokens=11, completion_tokens=4, total_tokens=15)
    )
    meter = TokenUsageMeter()
    with using_token_meter(meter):
        text = llm_call(
            client=client,
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
            usage_tag="policy",
        )
    assert text == 'click("1")'
    totals = meter.totals()
    assert totals["total_tokens"] == 15
    assert totals["by_tag"]["policy"]["prompt_tokens"] == 11


def test_llm_call_without_meter_still_returns_text():
    client = _FakeClient(
        SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    )
    text = llm_call(
        client=client,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o",
    )
    assert text == 'click("1")'
