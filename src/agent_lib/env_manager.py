"""Browser environment lifecycle manager.

Wraps BrowserEnv creation, reset-on-revision logic, and cleanup.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from browsergym.core.env import BrowserEnv

from .history import ConversationHistory
from .task_env import GenericBrowserTask
from ..models import TaskInstance
from ..utils.browser import (
    get_pw_chromium_kwargs,
    get_pw_context_kwargs,
    wait_for_page_ready,
)


class DynaBrowserEnv(BrowserEnv):
    """BrowserGym environment with bounded, content-aware observation waits."""

    def _wait_dom_loaded(self) -> None:
        # BrowserGym's default implementation waits up to three seconds for
        # every frame, sequentially. Ad-heavy sites can have many never-settling
        # iframes, turning one observation into a minute-long wait. The active
        # top-level page is the state given to the agent, so settle that page
        # once with a single hard timeout shared by all readiness checks.
        page = getattr(self, "page", None)
        if page is None or page.is_closed():
            return
        try:
            wait_for_page_ready(page)
        except Exception:
            # Readiness waiting is best-effort and must never kill a task.
            return


def _build_task_env(task: TaskInstance, headless: bool) -> BrowserEnv:
    task_cls = type(
        f"DynTask_{task.instance_id}",
        (GenericBrowserTask,),
        {"goal": task.i0, "start_url": task.runtime["start_url"]},
    )
    return DynaBrowserEnv(
        task_entrypoint=task_cls,
        task_kwargs={},
        headless=headless,
        viewport={"width": 1280, "height": 720},
        locale="en-US",
        pw_chromium_kwargs=get_pw_chromium_kwargs(),
        pw_context_kwargs=get_pw_context_kwargs(),
    )


class EnvironmentManager:
    """Manages the BrowserEnv lifecycle for one task episode."""

    def __init__(self, task: TaskInstance, args: Any) -> None:
        self._task = task
        self._args = args
        self._env: BrowserEnv | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> Tuple[BrowserEnv, Dict[str, Any]]:
        """Create and reset an env with shared pre-navigation browser settings."""
        self._env = _build_task_env(self._task, self._args.headless)
        obs, _ = self._env.reset()
        return self._env, obs

    def handle_revision(
        self,
        revision_policy: str,
        history: ConversationHistory,
    ) -> Tuple[BrowserEnv, Dict[str, Any], bool]:
        """Handle browser state after a revision fires.

        If revision_policy == 'reset': close old env, create fresh one,
        clear conversation history. Returns (env, obs, did_reset=True).
        Otherwise returns (current_env, current_obs, did_reset=False).
        """
        if revision_policy != "reset":
            assert self._env is not None
            return self._env, {}, False

        # Reset: close old env, create new one
        self.close()
        history.reset()
        env, obs = self.start()
        self._env = env
        return env, obs, True

    def close(self) -> None:
        """Close the browser environment if open."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
