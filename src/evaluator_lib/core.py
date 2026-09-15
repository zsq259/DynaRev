"""Task evaluator: score atomic constraints from execution logs.

Pipeline (v2-bound tasks only — every constraint must carry a source_revision):

  Phase 0 — fired filter: each constraint inherits its source_revision; only
    fired constraints (i0, or revisions whose injection_step appeared in the
    run) are evaluated. Unfired constraints lock fail / "not_fired".

  Phase B — decompose (one LLM call per fired constraint): the decomposer
    takes (constraint, done_message) and emits a list of binary sub-claims
    (type ∈ {simple, page}) that, ANDed, cover the constraint completely.

  Phase C — resolve (per sub-claim): each sub-claim is independently
    judged.
      type=simple: simple_verify reads done_message + applies logic.
      type=page:   locate the subject's host frame in the trajectory
                   (last_step preferred; iterate earlier frames),
                   then verify_claim_on_frame on that frame.

  Phase D — combine (deterministic): all sub-claims satisfied → constraint
    satisfied. Any failure → constraint fail. Diagnostic label categorizes
    the failure mode (face_value_fail / hallucination_detected /
    subject_not_found / mixed_fail). Evaluator failures abort the task-level
    evaluation and never become constraint failures.

This replaces the older done_judge triage step. There is no longer a
global done_says classification; agent self-admitted failures fall out as
simple sub-claims that fail naturally, and process-style constraints
(e.g. "on sephora.com") are expressed as page sub-claims with empty
assertions. The "any frame satisfied passes" soft-semantic of the old
no_claim path is preserved by the host-frame search in Phase C — a page
sub-claim with assertion=null is satisfied as soon as one frame in the
trajectory shows the subject.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote_plus

from openai import OpenAI

from ..models import TaskInstance
from ..utils.log_reader import LogReader
from .constraint_judge import simple_verify, verify_branch_condition, verify_claim_on_frame
from .decomposer import SubClaim, decompose, extract_done_message
from .errors import EvaluatorError


# Punctuation-normalization table for grep — bridges trivial unicode
# differences between done_message text and AXTree display text
# (en-dash vs hyphen, typographic quotes vs straight, NBSP vs space).
_GREP_NORMALIZE = str.maketrans({
    "–": "-",   # en-dash
    "—": "-",   # em-dash
    "‐": "-",   # hyphen
    "‑": "-",   # non-breaking hyphen
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u00a0": " ",  # non-breaking space
})


def _normalize_for_grep(s: str) -> str:
    """Case + light-unicode normalization for substring matching."""
    return s.translate(_GREP_NORMALIZE).lower()


_MATCH_STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "is", "of",
    "on", "page", "site", "the", "to", "website", "with",
}


def _match_tokens(text: str) -> set[str]:
    """Return lightly stemmed content tokens for relaxed frame matching."""
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", _normalize_for_grep(text)):
        if token in _MATCH_STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def _frame_match_info(
    subject: str,
    assertion: Optional[str],
    axtree: str,
    url: str,
) -> Tuple[str, float]:
    """Classify a frame as exact, token-level, or semantic-fallback.

    The numeric score only ranks fallback frames; it never decides whether a
    claim is true. The page verifier remains the final authority.
    """
    searchable = f"{unquote_plus(url or '')}\n{axtree}"
    searchable_norm = _normalize_for_grep(searchable)
    subject_norm = _normalize_for_grep(subject).strip()
    if subject_norm and subject_norm in searchable_norm:
        return "exact", 1.0

    frame_tokens = _match_tokens(searchable)
    subject_tokens = _match_tokens(subject)
    assertion_tokens = _match_tokens(assertion or "")
    subject_hits = len(subject_tokens & frame_tokens)
    subject_coverage = subject_hits / len(subject_tokens) if subject_tokens else 0.0
    assertion_coverage = (
        len(assertion_tokens & frame_tokens) / len(assertion_tokens)
        if assertion_tokens
        else 0.0
    )

    if (
        (len(subject_tokens) == 1 and subject_hits == 1)
        or (len(subject_tokens) >= 2 and subject_hits >= 2 and subject_coverage >= 0.6)
    ):
        return "token", subject_coverage

    # Rank semantic fallback candidates using both the subject and what the
    # claim expects the page to show. This score is deliberately permissive;
    # only verify_claim_on_frame can turn the candidate into a pass.
    return "semantic_fallback", (2.0 * subject_coverage) + assertion_coverage


# (lock_state) → diagnostic label
_COMBINE: Dict[str, str] = {
    "confirmed":               "confirmed",
    "face_value_fail":         "face_value_fail",
    "hallucination_detected":  "hallucination_detected",
    "subject_not_found":       "subject_not_found",
    "mixed_fail":              "mixed_fail",
    "not_fired":               "not_fired",
    "no_final_submit":         "no_final_submit",           # max_steps, evaluator skipped
    # Conditional override:
    "branch_skipped":          "branch_skipped",            # constraint on inactive branch
    "conditional_unaddressed": "conditional_unaddressed",   # agent didn't address condition
}

# Lock states that exclude a constraint from task-level numerator AND
# denominator. Run-level diagnostic counts still surface them so the
# rates remain visible.
_SKIP_LOCK_STATES = {"branch_skipped", "conditional_unaddressed"}


@dataclass
class ScanState:
    """Per-constraint state mutated through Phase B / C / D."""

    constraint_id: str
    description: str
    source_revision: str
    fired: bool

    # Filled by Phase B
    subclaims: List[SubClaim] = field(default_factory=list)

    # Filled by Phase C (one resolution per sub-claim, aligned by index)
    subclaim_resolutions: List[Dict[str, Any]] = field(default_factory=list)

    # Trajectory steps that produced evidence
    evidence_steps: List[int] = field(default_factory=list)

    # Lock
    status: str = "scanning"   # "scanning" | "satisfied" | "fail" | "skipped"
    lock_state: Optional[str] = None
    fail_reason_code: Optional[str] = None


class TaskEvaluator:
    """Evaluate one task execution against its atomic constraints."""

    def __init__(
        self,
        task: TaskInstance,
        log_reader: LogReader,
        client: OpenAI,
        model: str,
        execution_date: str = "",
    ):
        self.task = task
        self.log_reader = log_reader
        self.client = client
        self.model = model
        self.execution_date = execution_date
        self._done_message: str = ""

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, Any]:
        events = self.log_reader.get_events()
        last_step = events[-1]["step"] if events else 0
        injection_map = self._build_injection_map(events)
        self._done_message = extract_done_message(events) if events else ""

        states = self._build_constraint_states()

        # Phase A — branch resolution (only if task has a conditional
        # override). Marks default- or alternative-branch constraints as
        # skipped so the per-constraint loop doesn't run B/C/D on them.
        branch_resolution = self._resolve_branch_if_any(states)

        for s in states:
            if s.status == "skipped":
                continue
            if not s.fired:
                self._lock_unfired(s)
                continue
            if last_step <= 0:
                raise EvaluatorError("execution log contains no events")
            self._scan_constraint(s, injection_map, last_step)

        return self._render(states, branch_resolution)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_injection_map(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        injection: Dict[str, int] = {"i0": 1}
        for ev in events:
            for rid in ev.get("injected_revision_ids", []):
                if rid not in injection:
                    injection[rid] = ev["step"]
        return injection

    def _build_constraint_states(self) -> List[ScanState]:
        summary = self.log_reader.get_summary()
        fired_revisions = set(summary.get("fired_revisions", []))
        states = []
        for c in self.task.constraints:
            if not c.source_revision:
                raise ValueError(
                    f"Constraint c{c.id} has no source_revision binding. "
                    "This evaluator requires v2-bound tasks (run Pipeline Step 5)."
                )
            fired = (c.source_revision == "i0") or (c.source_revision in fired_revisions)
            states.append(ScanState(
                constraint_id=f"c{c.id}",
                description=c.description,
                source_revision=c.source_revision,
                fired=fired,
            ))
        return states

    # ------------------------------------------------------------------
    # Phase A: branch resolution (conditional override)
    # ------------------------------------------------------------------

    def _resolve_branch_if_any(
        self, states: List[ScanState]
    ) -> Optional[Dict[str, Any]]:
        """Resolve which branch of a conditional override is active.

        Returns None when ``task.branch_policy`` is unset (non-conditional
        task). Otherwise:

        - If the conditional revision never fired, alternative is locked
          ``branch_skipped`` and default is evaluated normally — the
          conditional clause was never delivered to the agent.
        - Else, simple_verify on ``condition_text`` against done_message
          returns one of {"true", "false", "not_addressed"}:
            * "true"           → alternative active, default skipped
            * "false"          → default active, alternative skipped
            * "not_addressed"  → both branches skipped with
              ``conditional_unaddressed`` lock; not counted in task
              denominator, surfaced as a run-level diagnostic.

        Returns a dict embedded into the evaluation output describing
        the resolution (condition_state, active_branch, skipped ids,
        reason, confidence) so reviewers can audit the decision.
        """
        bp = self.task.branch_policy
        if bp is None:
            return None

        states_by_id: Dict[int, ScanState] = {}
        for s in states:
            try:
                states_by_id[int(s.constraint_id[1:])] = s
            except (ValueError, IndexError):
                pass

        default_state = states_by_id.get(bp.default_constraint_id)
        alt_state = states_by_id.get(bp.alternative_constraint_id)

        if default_state is None or alt_state is None:
            # Malformed binding — surface but don't crash.
            return {
                "condition_text": bp.condition_text,
                "condition_state": "not_addressed",
                "active_branch": "unaddressed",
                "active_constraint_id": None,
                "skipped_constraint_ids": [],
                "reason": (
                    "branch_policy references constraint id(s) not in task; "
                    f"default={bp.default_constraint_id} "
                    f"alt={bp.alternative_constraint_id}"
                ),
                "confidence": 0.0,
            }

        # Both default and alternative bind to the same conditional
        # revision (export-time invariant). If that revision did not
        # fire, both states are not_fired by Phase 0 — Phase A skips
        # branch resolution and lets not_fired locks stand.
        if not default_state.fired and not alt_state.fired:
            return {
                "condition_text": bp.condition_text,
                "condition_state": "not_delivered",
                "active_branch": "none",
                "active_constraint_id": None,
                "skipped_constraint_ids": [],
                "reason": (
                    "Conditional revision did not fire; both default and "
                    "alternative branches are not_fired. Branch resolution skipped."
                ),
                "confidence": 1.0,
            }

        result = verify_branch_condition(
            condition_text=bp.condition_text,
            done_message=self._done_message,
            client=self.client,
            model=self.model,
            execution_date=self.execution_date,
        )
        state = result.get("condition_state", "not_addressed")
        reason = result.get("reason", "")
        confidence = float(result.get("confidence", 0.0))

        if state == "true":
            self._lock_skipped(default_state, "branch_skipped")
            return {
                "condition_text": bp.condition_text,
                "condition_state": "true",
                "active_branch": "alternative",
                "active_constraint_id": bp.alternative_constraint_id,
                "skipped_constraint_ids": [bp.default_constraint_id],
                "reason": reason,
                "confidence": confidence,
            }
        if state == "false":
            self._lock_skipped(alt_state, "branch_skipped")
            return {
                "condition_text": bp.condition_text,
                "condition_state": "false",
                "active_branch": "default",
                "active_constraint_id": bp.default_constraint_id,
                "skipped_constraint_ids": [bp.alternative_constraint_id],
                "reason": reason,
                "confidence": confidence,
            }
        # not_addressed
        self._lock_skipped(default_state, "conditional_unaddressed")
        self._lock_skipped(alt_state, "conditional_unaddressed")
        return {
            "condition_text": bp.condition_text,
            "condition_state": "not_addressed",
            "active_branch": "unaddressed",
            "active_constraint_id": None,
            "skipped_constraint_ids": [
                bp.default_constraint_id, bp.alternative_constraint_id
            ],
            "reason": reason,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Phase B/C/D: per-constraint scan
    # ------------------------------------------------------------------

    def _scan_constraint(
        self,
        s: ScanState,
        injection_map: Dict[str, int],
        last_step: int,
    ) -> None:
        # Phase B: decompose
        s.subclaims = decompose(
            constraint_description=s.description,
            done_message=self._done_message,
            client=self.client,
            model=self.model,
            execution_date=self.execution_date,
        )

        if not s.subclaims:
            # Defensive guard: decompose() normally raises this itself.
            raise EvaluatorError(
                f"constraint {s.constraint_id} decomposition returned no sub-claims"
            )

        # Phase C: resolve each sub-claim
        for sc in s.subclaims:
            resolution = self._resolve_subclaim(s, sc, injection_map, last_step)
            s.subclaim_resolutions.append(resolution)
            host_step = resolution.get("host_step")
            if host_step is not None and host_step not in s.evidence_steps:
                s.evidence_steps.append(host_step)

        # Phase D: combine
        self._combine_lock(s)

    def _resolve_subclaim(
        self,
        s: ScanState,
        sc: SubClaim,
        injection_map: Dict[str, int],
        last_step: int,
    ) -> Dict[str, Any]:
        if sc.type == "simple":
            r = simple_verify(
                constraint_description=s.description,
                predicate=sc.description,
                done_message=self._done_message,
                client=self.client,
                model=self.model,
                execution_date=self.execution_date,
            )
            return {
                "claim_id": sc.claim_id,
                "type": "simple",
                "description": sc.description,
                "satisfied": bool(r.get("satisfied", False)),
                "reason": r.get("reason", ""),
                "confidence": r.get("confidence", 0.0),
                "host_step": None,
            }

        # type == "page"
        return self._resolve_page_subclaim(s, sc, injection_map, last_step)

    def _resolve_page_subclaim(
        self,
        s: ScanState,
        sc: SubClaim,
        injection_map: Dict[str, int],
        last_step: int,
    ) -> Dict[str, Any]:
        """Locate host frame for sc.subject and verify sc.assertion on it.

        Trial order: last_step first (so trajectories that end on the
        subject's page lock cleanly), then injection_step .. last_step-1
        in ascending order.

        Locking rule:
        - assertion=None: first frame with is_subject_page=true wins.
        - assertion!=None: keep iterating past is_subject_page=true frames
          whose assertion_supported is false, looking for a frame where
          BOTH is_subject_page and assertion_supported are true (the
          subject's page may render incrementally — early scrolls miss
          synopsis text that later scrolls show). Stop on the first
          full match.
        - If candidates exhaust without a full match but at least one
          frame had is_subject_page=true, fall back to that frame as
          host_step with fail_kind="hallucination" — the subject was
          really visited but the agent's assertion was not corroborated
          on any visited frame.
        - If no frame qualified as host at all, fail_kind=
          "subject_not_found".
        """
        subject = sc.subject or ""
        assertion = sc.assertion
        start_step = injection_map.get(s.source_revision, 1)

        candidate_steps: List[int] = [last_step]
        for st in range(start_step, last_step):
            if st != last_step:
                candidate_steps.append(st)

        last_attempt: Optional[Dict[str, Any]] = None
        last_attempt_step: Optional[int] = None

        # Remember the first is_subject_page=true frame whose assertion was
        # not supported, in case no later frame supports the assertion either
        # — that frame becomes the hallucination host.
        partial_match: Optional[Tuple[int, Dict[str, Any], str]] = None

        events_by_step = {
            ev.get("step"): ev
            for ev in self.log_reader.get_events()
            if ev.get("step") is not None
        }
        exact_candidates: List[Tuple[int, str, str, float]] = []
        token_candidates: List[Tuple[int, str, str, float]] = []
        fallback_candidates: List[Tuple[int, str, str, float]] = []

        for priority, step in enumerate(candidate_steps):
            axtree = self.log_reader.get_axtree(step)
            if not axtree:
                continue
            url = str(events_by_step.get(step, {}).get("url") or "")
            level, score = _frame_match_info(subject, assertion, axtree, url)
            candidate = (step, axtree, level, score - (priority * 1e-6))
            if level == "exact":
                exact_candidates.append(candidate)
            elif level == "token":
                token_candidates.append(candidate)
            else:
                fallback_candidates.append(candidate)

        # Exact hits retain the old fast path. Token matches cover harmless
        # wording variations. Finally, judge at most three best remaining
        # frames semantically so a grep miss cannot directly become failure.
        fallback_candidates.sort(key=lambda item: item[3], reverse=True)
        candidates = exact_candidates + token_candidates + fallback_candidates[:3]

        last_attempt_match_level: Optional[str] = None
        for step, axtree, match_level, _score in candidates:
            v = verify_claim_on_frame(
                constraint_description=s.description,
                subject=subject,
                agent_assertion=assertion,
                axtree=axtree,
                client=self.client,
                model=self.model,
                execution_date=self.execution_date,
            )
            last_attempt = v
            last_attempt_step = step
            last_attempt_match_level = match_level
            if v.get("is_subject_page"):
                if assertion is None or v.get("assertion_supported"):
                    return {
                        "claim_id": sc.claim_id,
                        "type": "page",
                        "description": sc.description,
                        "subject": subject,
                        "assertion": assertion,
                        "satisfied": True,
                        "host_step": step,
                        "is_subject_page": True,
                        "assertion_supported": v.get("assertion_supported"),
                        "evidence_summary": v.get("evidence_summary", ""),
                        "confidence": v.get("confidence", 0.0),
                        "reason": v.get("reason", ""),
                        "fail_kind": None,
                        "match_level": match_level,
                    }
                # is_subject_page but assertion not supported here. Remember
                # the first such frame and continue searching for a frame
                # where both hold.
                if partial_match is None:
                    partial_match = (step, v, match_level)

        # All candidates exhausted.
        if partial_match is not None:
            step, v, match_level = partial_match
            return {
                "claim_id": sc.claim_id,
                "type": "page",
                "description": sc.description,
                "subject": subject,
                "assertion": assertion,
                "satisfied": False,
                "host_step": step,
                "is_subject_page": True,
                "assertion_supported": False,
                "evidence_summary": v.get("evidence_summary", ""),
                "confidence": v.get("confidence", 0.0),
                "reason": v.get("reason", ""),
                "fail_kind": "hallucination",
                "match_level": match_level,
            }

        # No frame qualified as subject's host page.
        if last_attempt is None:
            return {
                "claim_id": sc.claim_id,
                "type": "page",
                "description": sc.description,
                "subject": subject,
                "assertion": assertion,
                "satisfied": False,
                "host_step": None,
                "is_subject_page": False,
                "assertion_supported": None,
                "evidence_summary": "subject not found in trajectory",
                "confidence": 0.0,
                "reason": "trajectory contains no readable candidate frames",
                "fail_kind": "subject_not_found",
                "match_level": None,
            }
        return {
            "claim_id": sc.claim_id,
            "type": "page",
            "description": sc.description,
            "subject": subject,
            "assertion": assertion,
            "satisfied": False,
            "host_step": None,
            "is_subject_page": False,
            "assertion_supported": last_attempt.get("assertion_supported"),
            "evidence_summary": (
                last_attempt.get("evidence_summary")
                or "no host frame for subject"
            ),
            "confidence": last_attempt.get("confidence", 0.0),
            "reason": last_attempt.get("reason", ""),
            "last_attempt_step": last_attempt_step,
            "fail_kind": "subject_not_found",
            "match_level": last_attempt_match_level,
        }

    def _combine_lock(self, s: ScanState) -> None:
        all_satisfied = all(r.get("satisfied") for r in s.subclaim_resolutions)
        if all_satisfied:
            anchor = s.evidence_steps[-1] if s.evidence_steps else 0
            self._lock_satisfied(s, anchor, "confirmed")
            return

        simple_fail = False
        hallucination = False
        subject_not_found = False
        for r in s.subclaim_resolutions:
            if r.get("satisfied"):
                continue
            if r.get("type") == "simple":
                simple_fail = True
            else:
                fk = r.get("fail_kind")
                if fk == "hallucination":
                    hallucination = True
                elif fk == "subject_not_found":
                    subject_not_found = True

        kinds = [k for k, present in (
            ("simple", simple_fail),
            ("hallucination", hallucination),
            ("subject_not_found", subject_not_found),
        ) if present]

        if len(kinds) == 0:
            # Degenerate: subclaims exist, none satisfied, but no kind
            # categorized — should not happen, but guard anyway.
            self._lock_fail(s, "mixed_fail", "uncategorized")
            return

        if len(kinds) > 1:
            self._lock_fail(s, "mixed_fail", "+".join(kinds))
            return

        kind = kinds[0]
        if kind == "simple":
            self._lock_fail(s, "face_value_fail", "simple_predicate_failed")
        elif kind == "hallucination":
            self._lock_fail(s, "hallucination_detected", "assertion_not_supported")
        else:
            self._lock_fail(s, "subject_not_found", "no_host_frame")

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------

    def _lock_satisfied(self, s: ScanState, step: int, lock_state: str) -> None:
        s.status = "satisfied"
        s.lock_state = lock_state
        if step and step not in s.evidence_steps:
            s.evidence_steps.append(step)

    def _lock_fail(self, s: ScanState, lock_state: str, fail_code: str) -> None:
        s.status = "fail"
        s.lock_state = lock_state
        s.fail_reason_code = fail_code

    def _lock_unfired(self, s: ScanState) -> None:
        s.status = "fail"
        s.lock_state = "not_fired"
        s.fail_reason_code = "not_fired"

    def _lock_skipped(self, s: ScanState, lock_state: str) -> None:
        s.status = "skipped"
        s.lock_state = lock_state
        s.fail_reason_code = lock_state

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(
        self,
        states: List[ScanState],
        branch_resolution: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        results = [self._render_constraint(s) for s in states]
        # Skipped constraints (branch_skipped / conditional_unaddressed)
        # are excluded from BOTH numerator and denominator. They remain
        # visible in constraint_results and diagnostic_counts.
        effective = [
            r for r in results
            if r.get("combine_label") not in _SKIP_LOCK_STATES
        ]
        satisfied = sum(1 for r in effective if r["satisfied"])
        total = len(effective)
        out: Dict[str, Any] = {
            "correctness_score": satisfied / total if total else 0.0,
            "is_correct": total > 0 and satisfied == total,
            "constraint_results": results,
            "revision_results": self._aggregate_revisions(results),
            "diagnostic_counts": self._diagnostic_counts(results),
            "overall_reason": (
                f"Satisfied {satisfied}/{total} effective constraints "
                f"(skipped {len(results) - total} via branch dispatch)"
                if len(results) != total
                else f"Satisfied {satisfied}/{total} constraints"
            ),
        }
        if branch_resolution is not None:
            out["branch_resolution"] = branch_resolution
        return out

    def _render_constraint(self, s: ScanState) -> Dict[str, Any]:
        if s.status == "skipped":
            label = s.lock_state or "branch_skipped"
        elif not s.fired:
            label = "not_fired"
        else:
            label = _COMBINE.get(s.lock_state or "", "ambiguous")

        if s.status == "satisfied":
            verdict = "satisfied"
        elif s.status == "skipped":
            verdict = "skipped"
        else:
            verdict = "not_satisfied"

        # Top-line reason: when satisfied, summarize sub-claim count;
        # when failed, surface the first failed sub-claim's reason.
        if s.status == "satisfied":
            reason = (
                f"all {len(s.subclaim_resolutions)} sub-claims satisfied"
                if s.subclaim_resolutions
                else ""
            )
            confidence = 1.0
        elif s.status == "skipped":
            if s.lock_state == "conditional_unaddressed":
                reason = (
                    "Agent did not address the conditional clause in "
                    "done_message; both branches skipped."
                )
            else:
                reason = (
                    "Branch dispatch — this constraint is on the inactive "
                    "branch for this run."
                )
            confidence = 1.0
        elif not s.fired:
            reason = "Revision not triggered during execution"
            confidence = 1.0
        elif not s.subclaim_resolutions:
            reason = "decomposer returned no sub-claims"
            confidence = 0.0
        else:
            failed = [r for r in s.subclaim_resolutions if not r.get("satisfied")]
            if failed:
                reason = failed[0].get("reason") or failed[0].get("description") or ""
                confidence = float(failed[0].get("confidence", 0.0))
            else:
                reason = ""
                confidence = 0.0

        # For skipped constraints, satisfied=null in JSON (None in dict)
        # — they are not satisfied OR unsatisfied; they don't apply to
        # this run.
        if s.status == "skipped":
            satisfied_field: Optional[bool] = None
        else:
            satisfied_field = s.status == "satisfied"

        return {
            "constraint_id": s.constraint_id,
            "constraint_description": s.description,
            "source_revision": s.source_revision,
            "fired": s.fired,
            "satisfied": satisfied_field,
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "evidence_step": s.evidence_steps[-1] if s.evidence_steps else None,
            "evidence_steps": list(s.evidence_steps),
            "subclaims": [
                {
                    "claim_id": sc.claim_id,
                    "type": sc.type,
                    "description": sc.description,
                    "subject": sc.subject,
                    "assertion": sc.assertion,
                }
                for sc in s.subclaims
            ],
            "subclaim_resolutions": list(s.subclaim_resolutions),
            "combine_label": label,
            "fail_reason_code": s.fail_reason_code,
        }

    def _aggregate_revisions(
        self, constraint_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        rev_to_cids: Dict[str, List[int]] = {"i0": list(self.task.i0_constraint_ids)}
        for r in self.task.revisions:
            rev_to_cids[r.revision_id] = list(r.constraint_ids)

        cid_to_result = {r["constraint_id"]: r for r in constraint_results}
        out = []
        for rev_id, cids in rev_to_cids.items():
            bound = [cid_to_result[f"c{cid}"] for cid in cids if f"c{cid}" in cid_to_result]
            if not bound:
                continue
            # Skipped constraints (satisfied=None) are excluded from the
            # revision-level AND aggregation — a revision whose only
            # binding is its inactive branch should not be marked
            # unsatisfied for that reason.
            effective_bound = [r for r in bound if r["satisfied"] is not None]
            failed = [
                r["constraint_id"] for r in effective_bound
                if not r["satisfied"]
            ]
            satisfied = (
                all(r["satisfied"] for r in effective_bound)
                if effective_bound
                else None
            )
            out.append({
                "revision_id": rev_id,
                "satisfied": satisfied,
                "constraint_ids": [r["constraint_id"] for r in bound],
                "failed_constraints": failed,
            })
        return out

    def _diagnostic_counts(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in results:
            label = r.get("combine_label", "")
            if label:
                counts[label] = counts.get(label, 0) + 1
        return counts
