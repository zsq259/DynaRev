"""Dispatch isolated tasks with a concurrency cap and site mixing.

The pool only schedules work. Each task still runs in its own isolated
process via the existing task supervisor; this module must not run two
agent episodes in one Python process.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, List
from urllib.parse import urlparse


def task_host(task: Any) -> str:
    """Hostname used only to mix sites; empty if the task has no URL."""
    runtime = getattr(task, "runtime", None) or {}
    source = getattr(task, "source", None) or {}
    url = ""
    if isinstance(runtime, dict):
        url = str(runtime.get("start_url") or "")
    if not url and isinstance(source, dict):
        url = str(source.get("website") or "")
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def pick_next_task(pending: List[Any], busy_hosts: set[str]) -> Any:
    """Prefer a task whose site is not already in flight; otherwise FIFO."""
    if not pending:
        raise IndexError("no pending tasks")
    for index, task in enumerate(pending):
        host = task_host(task)
        if host and host not in busy_hosts:
            return pending.pop(index)
    return pending.pop(0)


def run_isolated_task_pool(
    tasks: List[Any],
    workers: int,
    run_fn: Callable[[Any], Dict[str, Any]],
    on_progress: Callable[[int, int], None] | None = None,
) -> List[Dict[str, Any]]:
    """Run ``run_fn`` on each task with at most ``workers`` in flight.

    ``run_fn`` must itself isolate the episode (own process, browser, files).
    Results are collected in completion order.
    """
    pending = list(tasks)
    total = len(pending)
    if total == 0:
        return []

    workers = max(1, min(int(workers), total))
    inflight: Dict[Any, Any] = {}
    results: List[Dict[str, Any]] = []

    def submit(pool: ThreadPoolExecutor) -> None:
        if not pending:
            return
        busy = {task_host(task) for task in inflight.values()}
        task = pick_next_task(pending, busy)
        future = pool.submit(run_fn, task)
        inflight[future] = task

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(min(workers, len(pending))):
            submit(pool)
        while inflight:
            done, _ = wait(tuple(inflight), return_when=FIRST_COMPLETED)
            for future in done:
                inflight.pop(future)
                results.append(future.result())
                if on_progress is not None:
                    on_progress(len(results), total)
                submit(pool)
    return results
