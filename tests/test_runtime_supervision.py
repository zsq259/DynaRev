import argparse
import json
from types import SimpleNamespace

from src.dyna_agent import build_task_timeout_result, collect_completed_instance_ids
from src.agent_lib.runner import observation_after_revision
from src.utils.site_health import SiteHealthStore


def _args():
    return argparse.Namespace(
        policy_model="policy",
        history_policy="full",
        revision_policy="append",
        instruction_policy="structured",
        reasoning_mode="direct",
    )


def _task():
    return SimpleNamespace(
        instance_id="task-1",
        source={"task_id": "source-1", "scene": "test", "website": "https://example.com"},
        constraints=[SimpleNamespace(description="must work")],
    )


def test_browser_start_timeout_is_environment_failure():
    result = build_task_timeout_result(
        _task(),
        _args(),
        {"phase": "browser_start", "step": 0, "message": "timed out"},
    )
    assert result["completion_status"] == "environment_error"
    assert result["stopped_reason"] == "env_setup_error"
    assert result["error_category"] == "environment"


def test_runner_incompatible_overrides_old_positive_health(tmp_path):
    store = SiteHealthStore(tmp_path)
    store.register_urls(["https://example.com/"])
    row = store.get_health("https://example.com")
    row["runtime_accessible"] = True
    row["runtime_stable"] = True
    row["run_access_ok_count"] = 5
    store.mark_runner_incompatible("https://example.com", "iframe observation livelock")
    store.save()
    feasible = store.export_feasible()

    assert store.is_runner_incompatible("https://example.com/")
    assert "https://example.com" not in feasible["urls"]
    persisted = json.loads((tmp_path / "site_health.json").read_text())
    assert persisted["sites"]["https://example.com"]["runner_compatible"] is False


def test_resume_retries_system_exceptions(tmp_path):
    good = tmp_path / "good"
    retry = tmp_path / "retry"
    good.mkdir()
    retry.mkdir()
    (good / "summary.json").write_text(json.dumps({"completion_status": "submitted"}))
    (retry / "summary.json").write_text(json.dumps({"completion_status": "task_exception"}))

    assert collect_completed_instance_ids(tmp_path) == {"good"}


def test_append_revision_preserves_current_observation():
    current = {"axtree_object": {"nodes": []}, "url": "https://example.com/results"}
    assert observation_after_revision(current, {}, did_reset=False) is current


def test_reset_revision_uses_fresh_observation():
    current = {"axtree_object": {"nodes": []}, "url": "https://example.com/results"}
    fresh = {"axtree_object": {"nodes": []}, "url": "https://example.com"}
    assert observation_after_revision(current, fresh, did_reset=True) is fresh
