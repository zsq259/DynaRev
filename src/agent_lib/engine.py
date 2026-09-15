"""Agent engine entry point — thin wrapper that delegates to AgentRunner.

All execution logic lives in runner.py.  This module exists for backward
compatibility: external callers import run_one_task from here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from openai import OpenAI

from .runner import AgentRunner
from ..models import TaskInstance


def run_one_task(
    task: TaskInstance,
    args: Any,
    logger: logging.Logger,
    run_dir: Path,
    client: OpenAI,
    trigger_client: OpenAI | None = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    """Run a single task episode.  Delegates to AgentRunner."""
    runner = AgentRunner(
        task,
        args,
        logger,
        run_dir,
        client,
        trigger_client,
        progress_callback=progress_callback,
    )
    return runner.run()
