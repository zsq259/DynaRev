"""Process-isolated execution and watchdog for one browser task."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

from .engine import run_one_task
from .logging_utils import configure_logging
from ..utils.client import get_client


BROWSER_PHASE_PREFIX = "browser_"


def _task_worker(task: Any, args: Any, run_dir: str, messages: Any) -> None:
    """Run one task in its own session so its browser tree can be killed safely."""
    try:
        if hasattr(os, "setsid"):
            os.setsid()
    except OSError:
        pass

    current = {"phase": "worker_start", "step": 0}

    def report_phase(phase: str, step: int) -> None:
        current["phase"] = phase
        current["step"] = step
        messages.put({"kind": "phase", "phase": phase, "step": step})

    try:
        path = Path(run_dir)
        logger = configure_logging(path)
        policy_client = get_client(args.policy_model)
        trigger_client = get_client(args.trigger_model)
        result = run_one_task(
            task,
            args,
            logger,
            path,
            policy_client,
            trigger_client,
            progress_callback=report_phase,
        )
        messages.put({"kind": "result", "result": result})
    except BaseException as exc:
        messages.put(
            {
                "kind": "error",
                "error": str(exc)[:1000],
                "traceback": traceback.format_exc()[-8000:],
                "phase": current["phase"],
                "step": current["step"],
            }
        )


def _stop_process_tree(process: mp.Process) -> None:
    """Stop the worker and, on POSIX, browser subprocesses in its session."""
    if not process.is_alive():
        process.join(timeout=1)
        return

    killed_group = False
    if hasattr(os, "killpg"):
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGTERM)
                killed_group = True
        except (OSError, ProcessLookupError):
            pass
    if not killed_group:
        process.terminate()

    process.join(timeout=3)
    if process.is_alive():
        if hasattr(os, "killpg"):
            try:
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
        else:
            process.kill()
        process.join(timeout=3)


def run_task_isolated(
    task: Any,
    args: Any,
    run_dir: Path,
) -> Tuple[str, Dict[str, Any]]:
    """Run a task with phase and wall-clock deadlines.

    Returns ``(status, payload)`` where status is ``result``, ``error``, or
    ``timeout``. Browser phases have a short deadline; all work also has a
    generous task-wide ceiling.
    """
    context = mp.get_context("spawn")
    messages = context.Queue()
    process = context.Process(
        target=_task_worker,
        args=(task, args, str(run_dir), messages),
        name=f"dyna-{task.instance_id}",
    )
    process.start()

    started = time.monotonic()
    wall_timeout = float(getattr(args, "task_wall_timeout", 7200.0))
    browser_timeout = float(getattr(args, "browser_operation_timeout", 90.0))
    current_phase = "worker_start"
    current_step = 0
    phase_started = started
    terminal: Dict[str, Any] | None = None

    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            phase_elapsed = now - phase_started

            if elapsed > wall_timeout:
                terminal = {
                    "phase": current_phase,
                    "step": current_step,
                    "elapsed_seconds": round(elapsed, 3),
                    "timeout_seconds": wall_timeout,
                    "message": f"task wall-clock limit exceeded during {current_phase}",
                }
                _stop_process_tree(process)
                return "timeout", terminal

            if current_phase.startswith(BROWSER_PHASE_PREFIX) and phase_elapsed > browser_timeout:
                terminal = {
                    "phase": current_phase,
                    "step": current_step,
                    "elapsed_seconds": round(phase_elapsed, 3),
                    "timeout_seconds": browser_timeout,
                    "message": f"browser operation timed out during {current_phase}",
                }
                _stop_process_tree(process)
                return "timeout", terminal

            try:
                message = messages.get(timeout=0.5)
            except queue.Empty:
                if not process.is_alive():
                    process.join(timeout=1)
                    return "error", {
                        "error": f"task worker exited with code {process.exitcode} without a result",
                        "phase": current_phase,
                        "step": current_step,
                    }
                continue

            kind = message.get("kind")
            if kind == "phase":
                current_phase = str(message.get("phase", "unknown"))
                current_step = int(message.get("step", 0) or 0)
                phase_started = time.monotonic()
            elif kind == "result":
                process.join(timeout=5)
                if process.is_alive():
                    _stop_process_tree(process)
                return "result", dict(message["result"])
            elif kind == "error":
                process.join(timeout=5)
                if process.is_alive():
                    _stop_process_tree(process)
                return "error", dict(message)
    finally:
        if process.is_alive():
            _stop_process_tree(process)
        messages.cancel_join_thread()
        messages.close()
