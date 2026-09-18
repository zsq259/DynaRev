import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.agent_lib.data import load_dyna_jsonl, load_om2w_hf, select_tasks
from src.agent_lib.engine import run_one_task
from src.agent_lib.exec_pool import run_isolated_task_pool
from src.agent_lib.logging_utils import configure_logging
from src.agent_lib.presets import PRESETS, apply_preset
from src.agent_lib.task_supervisor import run_task_isolated
from src.utils.client import get_client
from src.utils.site_health import SiteHealthStore, normalize_url


DEFAULT_MODEL = "deepseek-v4-flash"
RESUME_RETRYABLE_STATUSES = {"task_exception", "environment_error"}


def collect_completed_instance_ids(run_dir: Path) -> set[str]:
    """Collect finished instance IDs from per-task summary artifacts.

    Resume behavior is keyed by run_name: if a run directory already exists,
    tasks with existing "<instance_id>/summary.json" files are skipped.
    System crashes and website/browser environment failures are not treated
    as completed attempts, so a later invocation can retry them.
    """
    completed: set[str] = set()
    if not run_dir.exists():
        return completed

    for path in run_dir.glob("*/summary.json"):
        instance_id = path.parent.name
        if not instance_id or instance_id.startswith("_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("completion_status") in RESUME_RETRYABLE_STATUSES:
            continue
        completed.add(instance_id)
    return completed


def is_retryable_environment_failure(result: Dict[str, Any]) -> bool:
    """Website/browser failures are retried; a 2h wall-clock kill is not."""
    if result.get("completion_status") != "environment_error":
        return False
    watchdog = result.get("watchdog") or {}
    return watchdog.get("timeout_kind") != "wall"


def archive_task_dir(run_dir: Path, task_id: str, attempt: int) -> Path | None:
    """Move a failed attempt aside so the next try writes a clean task dir."""
    src = run_dir / task_id
    if not src.exists():
        return None
    dest = run_dir / "_env_attempts" / task_id / f"attempt_{attempt}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(src), str(dest))
    return dest


def clear_task_eval_artifact(run_dir: Path, task_id: str) -> None:
    """Drop a stale evaluation JSON so evaluate_logs will score the new attempt."""
    eval_dir = run_dir / "evaluations"
    (eval_dir / f"{task_id}.evaluation.json").unlink(missing_ok=True)
    (eval_dir / f"{task_id}.evaluation.json.tmp").unlink(missing_ok=True)


def stash_resume_environment_dir(run_dir: Path, task_id: str) -> None:
    """Archive a previous environment/system failure before a fresh attempt."""
    summary_path = run_dir / task_id / "summary.json"
    if not summary_path.exists():
        return
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if payload.get("completion_status") not in RESUME_RETRYABLE_STATUSES:
        return
    n = 1
    while (run_dir / "_env_attempts" / task_id / f"attempt_{n}").exists():
        n += 1
    archive_task_dir(run_dir, task_id, n)
    clear_task_eval_artifact(run_dir, task_id)


def render_progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int((done / total) * width)
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def resolve_model_roles(args: argparse.Namespace) -> None:
    args.policy_model = args.policy_model or args.model
    args.trigger_model = args.trigger_model or args.policy_model


def build_run_aggregate(
    run_name: str,
    args: argparse.Namespace,
    results: List[Dict[str, Any]],
    exclusions: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    stop_reasons: Dict[str, int] = {}
    completion_statuses: Dict[str, int] = {}
    error_categories: Dict[str, int] = {}
    for r in results:
        sr = str(r.get("stopped_reason", "unknown"))
        stop_reasons[sr] = stop_reasons.get(sr, 0) + 1
        cs = str(r.get("completion_status", sr))
        completion_statuses[cs] = completion_statuses.get(cs, 0) + 1
        ec = r.get("error_category")
        if ec:
            category = str(ec)
            error_categories[category] = error_categories.get(category, 0) + 1
    scored_results = [
        r for r in results if r.get("completion_status") != "environment_error"
    ]
    return {
        "run_name": run_name,
        "dataset_type": args.dataset_type,
        "policy_model": args.policy_model,
        "trigger_model": args.trigger_model,
        "history_policy": args.history_policy,
        "history_obs_keep": getattr(args, "history_obs_keep", 5),
        "revision_policy": args.revision_policy,
        "instruction_policy": args.instruction_policy,
        "instruction_rewrite_model": args.policy_model if args.instruction_policy == "rewrite" else None,
        "reasoning_mode": getattr(args, "reasoning_mode", "direct"),
        "task_count": len(results),
        "health_excluded_count": len(exclusions or []),
        "health_exclusions": exclusions or [],
        "environment_error_count": sum(
            1 for r in results if r.get("completion_status") == "environment_error"
        ),
        "scored_task_count": sum(
            1 for r in results if r.get("completion_status") != "environment_error"
        ),
        "done_called_count": sum(1 for r in results if r["done_called"]),
        "submitted_count": sum(1 for r in results if r.get("completion_status") == "submitted"),
        "avg_steps": (
            sum(r["steps"] for r in scored_results) / len(scored_results)
            if scored_results
            else 0.0
        ),
        "viewport_only": getattr(args, "viewport_only", False),
        "milestone_fire_count": sum(len(r.get("fired_milestones", [])) for r in results),
        "stop_reason_counts": stop_reasons,
        "completion_status_counts": completion_statuses,
        "error_category_counts": error_categories,
        "results": results,
    }


def build_task_crash_result(task, args: argparse.Namespace, error_message: str) -> Dict[str, Any]:
    return {
        "instance_id": task.instance_id,
        "source_task_id": task.source.get("task_id"),
        "scene": task.source.get("scene"),
        "website": task.source.get("website"),
        "policy_model": args.policy_model,
        "history_policy": args.history_policy,
        "revision_policy": args.revision_policy,
        "instruction_policy": args.instruction_policy,
        "instruction_rewrite_model": args.policy_model if args.instruction_policy == "rewrite" else None,
        "reasoning_mode": getattr(args, "reasoning_mode", "direct"),
        "steps": 0,
        "done_called": False,
        "completion_status": "task_exception",
        "fired_milestones": [],
        "fired_revisions": [],
        "milestone_diagnostics": [],
        "stopped_reason": "task_exception",
        "error_category": "system",
        "error_message": error_message,
        "success_criteria": {"hard_constraints": [c.description for c in task.constraints]},
    }


def build_task_timeout_result(
    task,
    args: argparse.Namespace,
    timeout_payload: Dict[str, Any],
) -> Dict[str, Any]:
    phase = str(timeout_payload.get("phase", "unknown"))
    current_step = int(timeout_payload.get("step", 0) or 0)
    stopped_reason = "env_setup_error" if phase == "browser_start" else "env_error"
    return {
        "instance_id": task.instance_id,
        "source_task_id": task.source.get("task_id"),
        "scene": task.source.get("scene"),
        "website": task.source.get("website"),
        "policy_model": args.policy_model,
        "history_policy": args.history_policy,
        "revision_policy": args.revision_policy,
        "instruction_policy": args.instruction_policy,
        "instruction_rewrite_model": args.policy_model if args.instruction_policy == "rewrite" else None,
        "reasoning_mode": getattr(args, "reasoning_mode", "direct"),
        "steps": max(0, current_step - (1 if phase == "browser_step" else 0)),
        "done_called": False,
        "completion_status": "environment_error",
        "fired_milestones": [],
        "fired_revisions": [],
        "milestone_diagnostics": [],
        "stopped_reason": stopped_reason,
        "error_category": "environment",
        "error_message": str(timeout_payload.get("message", "task watchdog timeout"))[:500],
        "watchdog": timeout_payload,
        "success_criteria": {"hard_constraints": [c.description for c in task.constraints]},
    }


def build_task_environment_error_result(
    task,
    args: argparse.Namespace,
    error_message: str,
    phase: str,
    step: int,
) -> Dict[str, Any]:
    payload = {
        "phase": phase,
        "step": step,
        "message": error_message,
    }
    result = build_task_timeout_result(task, args, payload)
    result.pop("watchdog", None)
    result["error_message"] = error_message[:500]
    return result


def persist_task_result(run_dir: Path, task_id: str, result: Dict[str, Any]) -> None:
    """Write summary.json, merging into an existing runner summary when present."""
    task_dir = run_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / "summary.json"
    payload = dict(result)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                existing.update(result)
                payload = existing
        except (OSError, json.JSONDecodeError):
            pass
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    events_path = task_dir / "events.jsonl"
    if not events_path.exists():
        events_path.write_text("", encoding="utf-8")


def write_failure_artifacts(run_dir: Path, task_id: str, result: Dict[str, Any]) -> None:
    persist_task_result(run_dir, task_id, result)


def load_run_results(run_dir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("*/summary.json")):
        if path.parent.name.startswith("_"):
            continue
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return results


def run_single_attempt(
    task,
    args: argparse.Namespace,
    logger,
    run_dir: Path,
    policy_client,
    trigger_client,
) -> Dict[str, Any]:
    """Execute one isolated (or in-process) attempt and persist failure artifacts."""
    try:
        if args.no_task_isolation:
            return run_one_task(task, args, logger, run_dir, policy_client, trigger_client)

        worker_status, payload = run_task_isolated(task, args, run_dir)
        if worker_status == "result":
            return payload
        if worker_status == "timeout":
            result = build_task_timeout_result(task, args, payload)
            logger.error(
                "TASK_WATCHDOG_TIMEOUT | task=%s | phase=%s | step=%s | kind=%s | limit=%ss",
                task.instance_id,
                payload.get("phase"),
                payload.get("step"),
                payload.get("timeout_kind"),
                payload.get("timeout_seconds"),
            )
            write_failure_artifacts(run_dir, task.instance_id, result)
            return result

        err = str(payload.get("error", "task worker failed"))[:500]
        failed_phase = str(payload.get("phase", "unknown"))
        logger.error(
            "TASK_WORKER_ERROR | task=%s | phase=%s | error=%s",
            task.instance_id,
            failed_phase,
            err,
        )
        if failed_phase.startswith("browser_"):
            result = build_task_environment_error_result(
                task,
                args,
                err,
                failed_phase,
                int(payload.get("step", 0) or 0),
            )
        else:
            result = build_task_crash_result(task, args, err)
        write_failure_artifacts(run_dir, task.instance_id, result)
        return result
    except Exception as e:
        err = str(e)[:500]
        logger.exception("TASK_CRASH | task=%s | error=%s", task.instance_id, err)
        result = build_task_crash_result(task, args, err)
        write_failure_artifacts(run_dir, task.instance_id, result)
        return result


def run_task_with_env_retries(
    task,
    args: argparse.Namespace,
    logger,
    run_dir: Path,
    policy_client,
    trigger_client,
) -> Dict[str, Any]:
    """Retry website/browser failures; keep the last attempt as the task result."""
    extra = max(0, int(getattr(args, "env_retries", 2)))
    max_attempts = 1 + extra
    stash_resume_environment_dir(run_dir, task.instance_id)
    result: Dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            archive_task_dir(run_dir, task.instance_id, attempt - 1)
            logger.info(
                "ENV_RETRY | task=%s | attempt=%d/%d",
                task.instance_id,
                attempt,
                max_attempts,
            )
        result = run_single_attempt(
            task, args, logger, run_dir, policy_client, trigger_client
        )
        result["environment_attempt"] = attempt
        result["environment_attempts_max"] = max_attempts
        persist_task_result(run_dir, task.instance_id, result)
        if not is_retryable_environment_failure(result):
            return result
        logger.warning(
            "ENV_FAIL | task=%s | attempt=%d/%d | phase=%s | step=%s | %s",
            task.instance_id,
            attempt,
            max_attempts,
            (result.get("watchdog") or {}).get("phase", result.get("stopped_reason")),
            (result.get("watchdog") or {}).get("step", result.get("steps")),
            result.get("error_message", ""),
        )
    assert result is not None
    result["environment_retry_exhausted"] = True
    persist_task_result(run_dir, task.instance_id, result)
    return result


def task_health_urls(task: Any) -> List[str]:
    raw_urls = [task.source.get("website", ""), task.runtime.get("start_url", "")]
    normalized = [normalize_url(str(url)) for url in raw_urls]
    return sorted({url for url in normalized if url})

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dyna unified benchmark runner")
    p.add_argument("--dataset-type", choices=["dyna_jsonl", "om2w_hf"], default="dyna_jsonl")
    p.add_argument("--dataset-path", default="dyna_tasks_v1.jsonl")
    p.add_argument("--hf-dataset", default="osunlp/Online-Mind2Web")
    p.add_argument("--hf-split", default="test")
    p.add_argument("--instance-id", default=None, help="comma-separated instance ids")
    p.add_argument("--scene", default=None)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--model", default=DEFAULT_MODEL, help="shorthand default for policy/trigger/evaluator models")
    p.add_argument("--policy-model", default=None)
    p.add_argument("--trigger-model", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--stuck-steps",
        type=int,
        default=0,
        help="deprecated compatibility option; repeated actions no longer stop a task",
    )
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--history-policy",
        choices=["full", "kill"],
        default="full",
        help="policy context: full action/thought history or current turn only",
    )
    p.add_argument(
        "--history-obs-keep",
        type=int,
        default=5,
        metavar="K",
        help="when history-policy=full, keep the last K page snapshots; "
        "older PAGE CONTENT is omitted (actions and revisions stay)",
    )
    p.add_argument(
        "--revision-policy",
        choices=["append", "reset"],
        default="append",
        help="what to do after a revision fires: append in-place or reset browser to start_url",
    )
    p.add_argument(
        "--instruction-policy",
        choices=["structured", "rewrite"],
        default="structured",
        help="show numbered active revisions or rewrite them into one instruction",
    )
    p.add_argument(
        "--mode",
        choices=list(PRESETS.keys()),
        default=None,
        help="convenience preset that overrides individual policy flags",
    )
    p.add_argument(
        "--reasoning-mode",
        choices=["direct", "react", "reflection"],
        default="direct",
        help="agent reasoning strategy",
    )

    p.add_argument("--run-name", default=None)
    p.add_argument("--log-dir", default="logs/runs")
    p.add_argument("--viewport-only", action="store_true", help="filter AXTree to viewport-visible elements only")
    p.add_argument(
        "--browser-operation-timeout",
        type=float,
        default=90.0,
        help="hard limit in seconds for browser start/step/reset/close inside a task worker",
    )
    p.add_argument(
        "--env-retries",
        type=int,
        default=2,
        help="extra attempts after a website/browser environment failure; "
        "exhausted failures are excluded from official scores",
    )
    p.add_argument(
        "--exec-workers",
        type=int,
        default=3,
        help="max isolated task processes in flight; 1 keeps the serial loop",
    )
    p.add_argument(
        "--task-wall-timeout",
        type=float,
        default=7200.0,
        help="hard wall-clock limit in seconds for one isolated task",
    )
    p.add_argument(
        "--no-task-isolation",
        action="store_true",
        help="debug only: run tasks in the main process without watchdog isolation",
    )
    p.add_argument("--site-health-dir", default="data/websites")
    p.add_argument(
        "--include-runner-incompatible-sites",
        action="store_true",
        help="opt in to sites explicitly marked incompatible with the benchmark runner",
    )
    p.add_argument("--dry-run", action="store_true", help="validate loading and selection without executing browser")
    args = p.parse_args()
    if args.history_obs_keep < 1:
        raise SystemExit("--history-obs-keep must be >= 1")
    return args


def validate_exec_workers(args: argparse.Namespace, tasks: List[Any]) -> int:
    """Reject combinations that would share a process or dual-write a task dir."""
    workers = int(getattr(args, "exec_workers", 3) or 1)
    if workers < 1:
        raise SystemExit("--exec-workers must be >= 1")
    if workers > 1 and getattr(args, "no_task_isolation", False):
        raise SystemExit("--exec-workers > 1 requires task isolation")
    if workers > 1:
        ids = [task.instance_id for task in tasks]
        if len(ids) != len(set(ids)):
            raise SystemExit("parallel exec requires unique instance_id values")
    return workers


def log_run_progress(logger, done: int, total: int) -> None:
    bar = render_progress_bar(done, total)
    logger.info(
        "RUN_PROGRESS | %s %d/%d (%.1f%%)",
        bar,
        done,
        total,
        (done / total) * 100.0 if total else 0.0,
    )


def main() -> None:
    args = parse_args()
    resolve_model_roles(args)
    apply_preset(args)

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.log_dir) / run_name
    logger = configure_logging(run_dir)

    if args.dataset_type == "dyna_jsonl":
        tasks = load_dyna_jsonl(args.dataset_path)
    else:
        tasks = load_om2w_hf(args.hf_dataset, args.hf_split, args.max_steps or 30)

    tasks = select_tasks(tasks, args)
    if not tasks:
        raise SystemExit("No tasks selected.")

    exclusions: List[Dict[str, Any]] = []
    if not args.include_runner_incompatible_sites:
        health = SiteHealthStore(args.site_health_dir)
        kept = []
        for task in tasks:
            blocked_sites = []
            for url in task_health_urls(task):
                row = health.get_health(url)
                if row and row.get("runner_compatible") is False:
                    blocked_sites.append(
                        {
                            "url": url,
                            "reason": row.get("runner_incompatibility_reason", ""),
                        }
                    )
            if blocked_sites:
                exclusions.append({"instance_id": task.instance_id, "sites": blocked_sites})
                logger.warning(
                    "HEALTH_EXCLUDE | task=%s | sites=%s",
                    task.instance_id,
                    ",".join(item["url"] for item in blocked_sites),
                )
            else:
                kept.append(task)
        tasks = kept
        if exclusions:
            with open(run_dir / "health_exclusions.json", "w", encoding="utf-8") as f:
                json.dump(exclusions, f, ensure_ascii=False, indent=2)

    completed_ids = collect_completed_instance_ids(run_dir)
    if completed_ids:
        before = len(tasks)
        tasks = [t for t in tasks if t.instance_id not in completed_ids]
        logger.info(
            "RESUME_SKIP | run_name=%s | skipped=%d | remaining=%d",
            run_name,
            before - len(tasks),
            len(tasks),
        )

    logger.info(
        "RUN_START | tasks=%d | exec_workers=%s | policy_model=%s | trigger_model=%s | dataset_type=%s",
        len(tasks),
        getattr(args, "exec_workers", 3),
        args.policy_model,
        args.trigger_model,
        args.dataset_type,
    )
    logger.info(
        "AGENT_PROTOCOL | history_policy=%s | history_obs_keep=%s | revision_policy=%s | instruction_policy=%s | reasoning_mode=%s",
        args.history_policy,
        getattr(args, "history_obs_keep", 5),
        args.revision_policy,
        args.instruction_policy,
        getattr(args, "reasoning_mode", "direct"),
    )

    if args.dry_run:
        preview = [
            {
                "instance_id": t.instance_id,
                "scene": t.source.get("scene"),
                "website": t.source.get("website"),
                "revisions": len(t.revisions),
                "milestones": len(t.milestones),
            }
            for t in tasks[:10]
        ]
        payload = {
            "run_name": run_name,
            "dataset_type": args.dataset_type,
            "policy_model": args.policy_model,
            "history_policy": args.history_policy,
            "revision_policy": args.revision_policy,
            "instruction_policy": args.instruction_policy,
            "reasoning_mode": getattr(args, "reasoning_mode", "direct"),
            "task_count": len(tasks),
            "health_excluded_count": len(exclusions),
            "health_exclusions": exclusions,
            "preview": preview,
        }
        with open(run_dir / "dry_run_summary.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("DRY_RUN_END | run_dir=%s", str(run_dir))
        print(str(run_dir))
        return

    if not tasks:
        aggregate = build_run_aggregate(run_name, args, load_run_results(run_dir), exclusions)
        with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(aggregate, f, ensure_ascii=False, indent=2)
        logger.info("RUN_END | no remaining tasks | run_dir=%s", str(run_dir))
        print(str(run_dir))
        return

    exec_workers = validate_exec_workers(args, tasks)
    policy_client = trigger_client = None
    if args.no_task_isolation:
        policy_client = get_client(args.policy_model)
        trigger_client = get_client(args.trigger_model)

    total_tasks = len(tasks)
    if exec_workers == 1:
        results = []
        for idx, task in enumerate(tasks, start=1):
            results.append(
                run_task_with_env_retries(
                    task, args, logger, run_dir, policy_client, trigger_client
                )
            )
            log_run_progress(logger, idx, total_tasks)
    else:
        logger.info("EXEC_POOL | workers=%d | tasks=%d", exec_workers, total_tasks)

        def run_one(task):
            return run_task_with_env_retries(task, args, logger, run_dir, None, None)

        results = run_isolated_task_pool(
            tasks,
            exec_workers,
            run_one,
            on_progress=lambda done, total: log_run_progress(logger, done, total),
        )

    aggregate = build_run_aggregate(run_name, args, load_run_results(run_dir), exclusions)

    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    logger.info("RUN_END | run_dir=%s", str(run_dir))
    print(str(run_dir))


if __name__ == "__main__":
    main()
