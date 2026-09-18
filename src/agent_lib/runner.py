"""AgentRunner — orchestrates one task episode with composable handlers.

Refactored from engine.py.  The four orthogonal dimensions are each managed
by a dedicated handler, and the run() loop calls them in a clean pipeline:

  ConversationHistory  — what the LLM sees (full vs. kill)
  EnvironmentManager   — browser lifecycle & reset-on-revision
  InstructionComposer  — how active instructions are presented
  ReasoningStrategy    — action generation pattern (direct / react / reflection)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from browsergym.core.env import BrowserEnv
from browsergym.utils.obs import flatten_axtree_to_str
from openai import OpenAI

from .action_parser import ACTION_PATTERN
from .env_manager import EnvironmentManager
from .history import ConversationHistory
from .instructions import InstructionComposer, InstructionSnapshot
from .milestones import milestone_should_fire
from .prompts import SYSTEM_PROMPT, VIEWPORT_HINT
from .reasoning import ReasoningStrategy, DirectStrategy
from ..models import Revision, StepEvent, TaskInstance
from ..utils.llm import TokenUsageMeter, llm_call, using_token_meter

# ---------------------------------------------------------------------------
# Injection-message helpers (preserved from original engine.py)
# ---------------------------------------------------------------------------

_CONDITIONAL_REMINDER = (
    "\n\nThis is a CONDITIONAL clause: observe the relevant page evidence "
    "(concrete numbers / comparisons / availability), decide which branch "
    "holds, and act accordingly. When you finish, your final "
    'send_msg_to_user("done: ...") MUST include a [CONDITIONAL OBSERVATION] '
    "section that states what you observed (with concrete values), whether "
    "the condition holds, and which branch you took."
)


def observation_after_revision(
    current_obs: Dict[str, Any],
    revision_obs: Dict[str, Any],
    did_reset: bool,
) -> Dict[str, Any]:
    """A non-reset revision must not discard the current browser observation."""
    return revision_obs if did_reset else current_obs


def derive_completion_status(stopped_reason: str, final_action: str) -> str:
    """Convert low-level environment termination into a user-facing status."""
    if stopped_reason == "env_done":
        if final_action.startswith("send_msg_to_user"):
            return "submitted"
        if final_action.startswith("report_infeasible"):
            return "reported_infeasible"
        return "terminated"
    if stopped_reason in {"env_error", "env_setup_error"}:
        return "environment_error"
    return stopped_reason


def derive_error_category(stopped_reason: str, completion_status: str) -> str | None:
    """Classify actual failures without treating normal submission as an error."""
    if stopped_reason in {"env_error", "env_setup_error"}:
        return "environment"
    if stopped_reason == "task_exception":
        return "system"
    if stopped_reason in {"stuck", "max_steps"}:
        return "agent"
    if completion_status in {"submitted", "reported_infeasible"}:
        return None
    return "unknown"


def _build_injection_message(revision: Revision, post_done: bool) -> str:
    """Compose the user message that delivers a revision to the agent."""
    if post_done:
        framing = (
            "[INSTRUCTION UPDATE — your previous done call was premature] "
            f"{revision.content}\n\nThere is more to do. After acting on this "
            'requirement, call send_msg_to_user("done: ...") AGAIN with an '
            "UPDATED summary that reflects all current requirements. Your "
            "previous done message is now stale — write a fresh one when "
            "finished."
        )
    else:
        framing = (
            f"[INSTRUCTION UPDATE] {revision.content}\n\n"
            "Important: this is now active. Apply it before finalizing."
        )
    if revision.type == "conditional_override":
        framing += _CONDITIONAL_REMINDER
    return framing


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------

class AgentRunner:
    """Orchestrates a single task episode.

    Four handlers are assembled at init time and the run() loop calls them
    in order.  The class has no mutable state beyond the handlers and
    step-level tracking — each stage is a method with a single responsibility.
    """

    def __init__(
        self,
        task: TaskInstance,
        args: Any,
        logger: logging.Logger,
        run_dir: Path,
        client: OpenAI,
        trigger_client: OpenAI | None = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        self._task = task
        self._args = args
        self._logger = logger
        self._run_dir = run_dir
        self._client = client
        self._trigger_client = trigger_client or client
        self._progress_callback = progress_callback

        # --- Handlers -------------------------------------------------------
        system_content = SYSTEM_PROMPT
        if getattr(args, "viewport_only", False):
            system_content += VIEWPORT_HINT

        self._history = ConversationHistory(
            system_content=system_content,
            history_policy=getattr(args, "history_policy", "full"),
            observation_keep=int(getattr(args, "history_obs_keep", 5)),
        )
        self._env_manager = EnvironmentManager(task, args)
        self._composer = InstructionComposer(
            policy=getattr(args, "instruction_policy", "structured"),
            client=client,
            model=args.policy_model,
            temperature=args.temperature,
        )
        self._reasoning = self._build_reasoning(client, args)

        # --- Episode state --------------------------------------------------
        self._revision_map = {r.revision_id: r for r in task.revisions}
        self._fired_milestones: List[str] = []
        self._fired_revision_ids: List[str] = []
        self._fired_step: Dict[str, int] = {}
        self._events: List[StepEvent] = []
        self._token_meter = TokenUsageMeter()

        # --- Per-task dir ---------------------------------------------------
        self._task_dir = run_dir / task.instance_id
        self._task_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Reasoning strategy factory
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reasoning(client: OpenAI, args: Any) -> ReasoningStrategy:
        mode = getattr(args, "reasoning_mode", "direct")
        if mode == "react":
            from .reasoning import ReActStrategy
            return ReActStrategy()
        elif mode == "reflection":
            from .reasoning import ReflectionStrategy
            return ReflectionStrategy()
        return DirectStrategy()

    # ==================================================================
    # Main loop
    # ==================================================================

    def run(self) -> Dict[str, Any]:
        args = self._args
        task = self._task
        logger = self._logger

        # --- Start browser --------------------------------------------------
        self._report_phase("browser_start", 0)
        env, obs = self._env_manager.start()
        self._report_phase("agent_processing", 0)

        max_steps = args.max_steps or int(task.runtime.get("max_steps", 30))
        history_policy = getattr(args, "history_policy", "full")
        revision_policy = getattr(args, "revision_policy", "append")

        stopped_reason = "max_steps"

        logger.info(
            "TASK_START | %s | %s | history=%s | revision=%s | instruction=%s | reasoning=%s",
            task.instance_id,
            task.source.get("website", ""),
            history_policy,
            revision_policy,
            self._composer.policy,
            getattr(args, "reasoning_mode", "direct"),
        )

        with using_token_meter(self._token_meter):
            for step in range(1, max_steps + 1):
                step_token_start = self._token_meter.n_calls
                # ---- Stage 1: Observe ------------------------------------------
                axtree, url, last_error = self._observe(obs)

                # ---- Stage 2: Milestones ---------------------------------------
                injected_now, trigger_checks = self._check_milestones(axtree, step)

                # ---- Stage 3: Revision Reset -----------------------------------
                if injected_now:
                    if revision_policy == "reset":
                        self._report_phase("browser_revision_reset", step)
                    next_env, next_obs, did_reset = self._env_manager.handle_revision(
                        revision_policy, self._history
                    )
                    env = next_env
                    obs = observation_after_revision(obs, next_obs, did_reset)
                    self._report_phase("agent_processing", step)
                    if did_reset:
                        axtree, url, last_error = self._observe(obs)
                        logger.info(
                            "REVISION_RESET | task=%s | step=%d | revisions=%s",
                            task.instance_id, step, ",".join(injected_now),
                        )

                # ---- Stage 4: Compose Instruction --------------------------------
                instruction_snapshot = self._composer.compose(
                    task.i0, self._revision_map, self._fired_revision_ids
                )

                # ---- Stage 5: Build User Message ---------------------------------
                user_msg = self._build_user_message(
                    instruction_snapshot, url, last_error, axtree
                )

                # ---- Stage 6-7: LLM Context + Call ------------------------------
                api_messages = self._history.build_llm_context(user_msg)
                suffix = self._reasoning.get_system_prompt_suffix()
                if suffix:
                    api_messages[0] = dict(api_messages[0])
                    api_messages[0]["content"] += suffix

                raw = llm_call(
                    client=self._client,
                    messages=api_messages,
                    model=args.policy_model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    usage_tag="policy",
                )

                # ---- Stage 8: Parse Output -------------------------------------
                action, thought = self._reasoning.process_output(raw)

                # ---- Record turn in history -----------------------------------
                self._history.record_user_message(user_msg)
                self._history.record_assistant_message(raw)

                # ---- Stage 9: Action Retry ------------------------------------
                if not ACTION_PATTERN.match(action):
                    action, raw, thought = self._retry_action(
                        api_messages, raw, thought
                    )

                # ---- Stage 10: Premature Done ---------------------------------
                if action.startswith("send_msg_to_user") and len(self._fired_revision_ids) < len(task.revisions):
                    self._handle_premature_done(
                        step, url, raw, trigger_checks, instruction_snapshot,
                        token_usage=self._token_meter.totals(step_token_start),
                    )
                    if revision_policy == "reset":
                        self._report_phase("browser_post_done_reset", step)
                        env, obs, _ = self._env_manager.handle_revision(
                            "reset", self._history
                        )
                        self._report_phase("agent_processing", step)
                        logger.info(
                            "POST_DONE_RESET | task=%s | step=%d | revisions=%s",
                            task.instance_id, step, ",".join(self._fired_revision_ids),
                        )
                    # Re-observe on next iteration
                    continue

                # ---- Stage 11: Execute Action ----------------------------------
                self._save_axtree(step, axtree)
                try:
                    self._report_phase("browser_step", step)
                    obs, _environment_reward, terminated, truncated, _ = env.step(action)
                    self._report_phase("agent_processing", step)
                except Exception as e:
                    self._report_phase("agent_processing", step)
                    logger.warning(
                        "STEP_ENV_ERROR | task=%s | step=%d | error=%s",
                        task.instance_id, step, str(e)[:200],
                    )
                    stopped_reason = "env_error"
                    self._record_step(
                        step, url, action, raw, injected_now, trigger_checks,
                        instruction_snapshot, thought,
                        token_usage=self._token_meter.totals(step_token_start),
                    )
                    break

                # ---- Stage 12: Post-Action Reflection --------------------------
                reflection = self._reasoning.post_action_reflect(
                    action=action,
                    axtree=axtree,
                    client=self._client,
                    model=args.policy_model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens or 512,
                )

                # ---- Stage 13: Record & Check Termination ----------------------
                self._record_step(
                    step, url, action, raw, injected_now, trigger_checks,
                    instruction_snapshot, thought, reflection,
                    token_usage=self._token_meter.totals(step_token_start),
                )
                logger.info(
                    "STEP | task=%s | step=%d | action=%s | url=%s",
                    task.instance_id, step, action, url[:120],
                )

                if terminated or truncated:
                    stopped_reason = "env_done"
                    break

        # --- Finalize -------------------------------------------------------
        self._report_phase("browser_close", len(self._events))
        result = self._finalize(env, stopped_reason)
        self._report_phase("finished", len(self._events))
        return result

    # ==================================================================
    # Stage methods
    # ==================================================================

    def _report_phase(self, phase: str, step: int) -> None:
        """Expose blocking browser phases to the parent-process watchdog."""
        if self._progress_callback is not None:
            self._progress_callback(phase, step)

    def _observe(self, obs: Dict[str, Any]) -> Tuple[str, str, str]:
        viewport_only = getattr(self._args, "viewport_only", False)
        if viewport_only:
            axtree = flatten_axtree_to_str(
                obs["axtree_object"],
                extra_properties=obs.get("extra_element_properties"),
                filter_visible_only=True,
            )
        else:
            axtree = flatten_axtree_to_str(obs["axtree_object"])
        url = obs.get("url", "")
        last_error = (obs.get("last_action_error", "") or "").strip()
        return axtree, url, last_error

    def _check_milestones(
        self, axtree: str, step: int
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        injected_now: List[str] = []
        trigger_checks: List[Dict[str, Any]] = []

        # Find first unfired milestone
        next_milestone = None
        for milestone in self._task.milestones:
            if milestone.milestone_id not in self._fired_milestones:
                next_milestone = milestone
                break

        if next_milestone is None:
            return injected_now, trigger_checks

        milestone = next_milestone
        should_fire, trigger_meta = milestone_should_fire(
            milestone=milestone,
            axtree=axtree,
            step=step,
            fired_step=self._fired_step,
            client=self._trigger_client,
            model=getattr(self._args, "trigger_model", "deepseek-v4-flash"),
        )
        trigger_checks.append({
            "milestone_id": milestone.milestone_id,
            "inject_revision_id": milestone.inject_revision_id,
            "fired": bool(should_fire),
            "confidence": trigger_meta.get("confidence"),
            "reason": trigger_meta.get("reason", ""),
        })

        if should_fire:
            self._fired_milestones.append(milestone.milestone_id)
            self._fired_step[milestone.milestone_id.lower()] = step
            revision = self._revision_map.get(milestone.inject_revision_id)
            if revision and revision.revision_id not in self._fired_revision_ids:
                self._fired_revision_ids.append(revision.revision_id)
                injected_now.append(revision.revision_id)
                self._history.record_user_message(
                    _build_injection_message(revision, post_done=False)
                )
                self._logger.info(
                    "MILESTONE_FIRE | task=%s | step=%d | milestone=%s | revision=%s | reason=%s",
                    self._task.instance_id, step,
                    milestone.milestone_id, revision.revision_id,
                    str(trigger_meta.get("reason", ""))[:160],
                )

        return injected_now, trigger_checks

    def _build_user_message(
        self,
        snapshot: InstructionSnapshot,
        url: str,
        last_error: str,
        axtree: str,
    ) -> str:
        parts = [
            f"CURRENT INSTRUCTIONS (ALL active):\n{snapshot.text}",
            f"CURRENT URL: {url}",
            "DECISION CHECKPOINT: If all active instructions are satisfied now, "
            "put send_msg_to_user('done: ...') inside an ```action fence.",
        ]
        if last_error:
            parts.append(f"LAST ERROR: {last_error}")
        axtree_short = axtree[:8000]
        if len(axtree) > 8000:
            axtree_short += "\n...(truncated)"
        parts.append(f"PAGE CONTENT:\n{axtree_short}")
        return "\n\n".join(parts)

    def _retry_action(
        self,
        api_messages: List[Dict[str, str]],
        raw: str,
        thought: Optional[str],
    ) -> Tuple[str, str, Optional[str]]:
        correction = (
            "ERROR: wrap exactly one BrowserGym call in an ```action fence. "
            "Do not use XML, DSML, JSON, unlabeled fences, or a bare last-line call. "
            "Example:\n\n```action\nclick(\"42\")\n```"
        )
        retry_msgs = list(api_messages) + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": correction},
        ]
        raw_retry = llm_call(
            client=self._client,
            messages=retry_msgs,
            model=self._args.policy_model,
            temperature=self._args.temperature,
            max_tokens=self._args.max_tokens,
            usage_tag="policy_retry",
        )
        action_retry, thought_retry = self._reasoning.process_output(raw_retry)
        self._history.record_correction_round(correction, raw_retry)
        return action_retry, raw_retry, thought_retry

    def _handle_premature_done(
        self,
        step: int,
        url: str,
        raw: str,
        trigger_checks: List[Dict[str, Any]],
        instruction_snapshot: InstructionSnapshot,
        token_usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Record the premature done step
        ev = StepEvent(
            step=step,
            url=url,
            action=f'send_msg_to_user("done: ...")',
            raw_output=raw,
            injected_revision_ids=[],
            trigger_checks=trigger_checks,
            instruction_policy=instruction_snapshot.policy,
            instruction_key=instruction_snapshot.key,
            active_instruction=instruction_snapshot.text,
            reasoning_mode=getattr(self._args, "reasoning_mode", "direct"),
            token_usage=token_usage,
        )
        self._events.append(ev)
        self._logger.info(
            "PREMATURE_DONE | task=%s | step=%d | revisions=%d/%d | injecting next",
            self._task.instance_id, step,
            len(self._fired_revision_ids), len(self._task.revisions),
        )

        # Force-fire next unfired milestone
        for milestone in self._task.milestones:
            if milestone.milestone_id not in self._fired_milestones:
                self._fired_milestones.append(milestone.milestone_id)
                self._fired_step[milestone.milestone_id.lower()] = step
                revision = self._revision_map.get(milestone.inject_revision_id)
                if revision and revision.revision_id not in self._fired_revision_ids:
                    self._fired_revision_ids.append(revision.revision_id)
                    self._history.record_user_message(
                        _build_injection_message(revision, post_done=True)
                    )
                    self._logger.info(
                        "POST_DONE_INJECT | task=%s | step=%d | milestone=%s | revision=%s",
                        self._task.instance_id, step,
                        milestone.milestone_id, revision.revision_id,
                    )
                break

    def _save_axtree(self, step: int, axtree: str) -> None:
        path = self._task_dir / f"step_{step:03d}.axtree"
        with open(path, "w", encoding="utf-8") as f:
            f.write(axtree)

    def _record_step(
        self,
        step: int,
        url: str,
        action: str,
        raw_output: str,
        injected_revision_ids: List[str],
        trigger_checks: List[Dict[str, Any]],
        instruction_snapshot: InstructionSnapshot,
        thought: Optional[str] = None,
        reflection: Optional[str] = None,
        token_usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        ev = StepEvent(
            step=step,
            url=url,
            action=action,
            raw_output=raw_output,
            injected_revision_ids=injected_revision_ids,
            trigger_checks=trigger_checks,
            axtree_path=f"step_{step:03d}.axtree",
            instruction_policy=instruction_snapshot.policy,
            instruction_key=instruction_snapshot.key,
            active_instruction=instruction_snapshot.text,
            thought=thought,
            reflection=reflection,
            reasoning_mode=getattr(self._args, "reasoning_mode", "direct"),
            token_usage=token_usage,
        )
        self._events.append(ev)

    def _finalize(
        self,
        env: BrowserEnv,
        stopped_reason: str,
    ) -> Dict[str, Any]:
        self._env_manager.close()

        task = self._task
        args = self._args
        done_called = any("send_msg_to_user" in e.action for e in self._events)
        final_action = self._events[-1].action if self._events else ""
        completion_status = derive_completion_status(stopped_reason, final_action)
        milestone_diagnostics = self._collect_milestone_diagnostics()
        error_category = derive_error_category(stopped_reason, completion_status)

        result = {
            "instance_id": task.instance_id,
            "source_task_id": task.source.get("task_id"),
            "scene": task.source.get("scene"),
            "website": task.source.get("website"),
            "policy_model": args.policy_model,
            "history_policy": getattr(args, "history_policy", "full"),
            "history_obs_keep": int(getattr(args, "history_obs_keep", 5)),
            "revision_policy": getattr(args, "revision_policy", "append"),
            "instruction_policy": self._composer.policy,
            "instruction_rewrite_model": args.policy_model if self._composer.policy == "rewrite" else None,
            "reasoning_mode": getattr(args, "reasoning_mode", "direct"),
            "steps": len(self._events),
            "token_usage": self._token_meter.totals(),
            "done_called": done_called,
            "completion_status": completion_status,
            "fired_milestones": self._fired_milestones,
            "fired_revisions": self._fired_revision_ids,
            "milestone_diagnostics": milestone_diagnostics,
            "stopped_reason": stopped_reason,
            "error_category": error_category,
            "success_criteria": {
                "hard_constraints": [c.description for c in task.constraints]
            },
        }

        # Write per-task artifacts
        with open(self._task_dir / "events.jsonl", "w", encoding="utf-8") as f:
            for e in self._events:
                f.write(json.dumps(e.__dict__, ensure_ascii=False) + "\n")

        if self._composer.rewrite_records:
            with open(self._task_dir / "instruction_rewrites.jsonl", "w", encoding="utf-8") as f:
                for rec in self._composer.rewrite_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with open(self._task_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        self._logger.info(
            "TASK_END | %s | steps=%d | done_called=%s | stop=%s",
            task.instance_id, len(self._events), done_called, stopped_reason,
        )
        return result

    def _collect_milestone_diagnostics(self) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []
        for milestone in self._task.milestones:
            checks = []
            for event in self._events:
                for check in event.trigger_checks:
                    if check.get("milestone_id") == milestone.milestone_id:
                        checks.append({"step": event.step, **check})
            diagnostics.append({
                "milestone_id": milestone.milestone_id,
                "inject_revision_id": milestone.inject_revision_id,
                "fired": milestone.milestone_id in self._fired_milestones,
                "fired_step": self._fired_step.get(milestone.milestone_id.lower()),
                "checks_count": len(checks),
                "last_check": checks[-1] if checks else None,
            })
        return diagnostics
