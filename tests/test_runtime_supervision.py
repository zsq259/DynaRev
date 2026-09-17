import argparse
import json
from types import SimpleNamespace

from src.dyna_agent import (
    archive_task_dir,
    build_task_timeout_result,
    collect_completed_instance_ids,
    is_retryable_environment_failure,
    stash_resume_environment_dir,
    validate_exec_workers,
)
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


def test_resume_retries_environment_errors(tmp_path):
    good = tmp_path / "good"
    env = tmp_path / "env"
    good.mkdir()
    env.mkdir()
    (good / "summary.json").write_text(json.dumps({"completion_status": "max_steps"}))
    (env / "summary.json").write_text(json.dumps({"completion_status": "environment_error"}))

    assert collect_completed_instance_ids(tmp_path) == {"good"}


def test_browser_timeout_is_retryable_wall_timeout_is_not():
    browser = {
        "completion_status": "environment_error",
        "watchdog": {"timeout_kind": "browser", "timeout_seconds": 90},
    }
    wall = {
        "completion_status": "environment_error",
        "watchdog": {"timeout_kind": "wall", "timeout_seconds": 7200},
    }
    submitted = {"completion_status": "submitted"}
    assert is_retryable_environment_failure(browser)
    assert not is_retryable_environment_failure(wall)
    assert not is_retryable_environment_failure(submitted)


def test_archive_task_dir_moves_attempt_aside(tmp_path):
    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "summary.json").write_text("{}", encoding="utf-8")
    dest = archive_task_dir(tmp_path, "task-1", 1)
    assert dest is not None
    assert not task_dir.exists()
    assert (dest / "summary.json").exists()


def test_stash_resume_clears_stale_eval_artifact(tmp_path):
    env = tmp_path / "env"
    env.mkdir()
    (env / "summary.json").write_text(
        json.dumps({"completion_status": "environment_error"}),
        encoding="utf-8",
    )
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()
    eval_file = eval_dir / "env.evaluation.json"
    eval_file.write_text("{}", encoding="utf-8")
    stash_resume_environment_dir(tmp_path, "env")
    assert not eval_file.exists()
    assert (tmp_path / "_env_attempts" / "env" / "attempt_1" / "summary.json").exists()


def test_append_revision_preserves_current_observation():
    current = {"axtree_object": {"nodes": []}, "url": "https://example.com/results"}
    assert observation_after_revision(current, {}, did_reset=False) is current


def test_reset_revision_uses_fresh_observation():
    current = {"axtree_object": {"nodes": []}, "url": "https://example.com/results"}
    fresh = {"axtree_object": {"nodes": []}, "url": "https://example.com"}
    assert observation_after_revision(current, fresh, did_reset=True) is fresh


def test_validate_exec_workers_keeps_serial_default():
    args = argparse.Namespace(exec_workers=1, no_task_isolation=True)
    assert validate_exec_workers(args, [_task(), _task()]) == 1


def test_validate_exec_workers_rejects_shared_process():
    args = argparse.Namespace(exec_workers=2, no_task_isolation=True)
    try:
        validate_exec_workers(args, [_task()])
    except SystemExit as exc:
        assert "task isolation" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_validate_exec_workers_rejects_duplicate_ids():
    args = argparse.Namespace(exec_workers=2, no_task_isolation=False)
    try:
        validate_exec_workers(args, [_task(), _task()])
    except SystemExit as exc:
        assert "unique instance_id" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
