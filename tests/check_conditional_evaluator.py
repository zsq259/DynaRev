"""End-to-end sanity check: conditional override evaluator with stubbed LLM.

Verifies that Phase A branch resolution correctly:
- locks default branch when condition is true
- locks alternative branch when condition is false
- locks both branches as conditional_unaddressed when not addressed
- excludes skipped constraints from numerator AND denominator
- writes branch_resolution into the evaluation output

Stubs the four LLM-touching evaluator functions (decompose,
simple_verify, verify_branch_condition, verify_claim_on_frame) so the
test is deterministic and runnable without API keys.

Run from repo root:
    conda run -n dyna python tests/check_conditional_evaluator.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Imports must come AFTER sys.path manipulation
from src.agent_lib.data import load_dyna_jsonl
from src.evaluator_lib import core
from src.evaluator_lib.decomposer import SubClaim
from src.utils.log_reader import LogReader


# ---------------------------------------------------------------------------
# Stub LLM verifiers — pure rule-based, deterministic
# ---------------------------------------------------------------------------

def stub_decompose(constraint_description, done_message, **kw):
    return [
        SubClaim(
            claim_id=1,
            type="simple",
            description=constraint_description,
            subject=None,
            assertion=None,
        )
    ]


def stub_simple_verify(constraint_description, predicate, done_message, **kw):
    pred_l = predicate.lower()
    dm_l = done_message.lower()
    if "origin is beijing" in pred_l:
        return {"satisfied": "beijing" in dm_l, "reason": "stub", "confidence": 0.9}
    if "destination is shanghai" in pred_l:
        return {"satisfied": "shanghai" in dm_l, "reason": "stub", "confidence": 0.9}
    if "day after tomorrow" in pred_l:
        return {
            "satisfied": "day after tomorrow" in dm_l,
            "reason": "stub",
            "confidence": 0.9,
        }
    if "non-red-eye" in pred_l:
        # Default branch — agent must have selected a non-red-eye flight
        return {
            "satisfied": "selected: ca1858" in dm_l or "non-red-eye, 880" in dm_l,
            "reason": "stub",
            "confidence": 0.9,
        }
    if "red-eye" in pred_l:
        # Alternative branch — agent must have selected a red-eye flight
        return {
            "satisfied": "selected: mu5101" in dm_l or "red-eye, 580" in dm_l,
            "reason": "stub",
            "confidence": 0.9,
        }
    return {"satisfied": False, "reason": "stub default", "confidence": 0.5}


def stub_verify_branch_condition(condition_text, done_message, **kw):
    dm = done_message.lower()
    has_observation = "[conditional observation]" in dm
    if not has_observation:
        return {
            "condition_state": "not_addressed",
            "reason": "stub: no observation block",
            "confidence": 0.6,
        }
    if "exceeding" in dm or "switched to" in dm:
        return {
            "condition_state": "true",
            "reason": "stub: agent says exceed",
            "confidence": 0.95,
        }
    if "below" in dm or "kept" in dm:
        return {
            "condition_state": "false",
            "reason": "stub: agent says below",
            "confidence": 0.95,
        }
    return {
        "condition_state": "not_addressed",
        "reason": "stub: ambiguous",
        "confidence": 0.5,
    }


def stub_verify_claim_on_frame(constraint_description, subject, agent_assertion, axtree, **kw):
    if subject and subject.lower() in axtree.lower():
        return {
            "is_subject_page": True,
            "assertion_supported": True if agent_assertion else None,
            "evidence_summary": "stub",
            "confidence": 0.9,
            "reason": "stub",
        }
    return {
        "is_subject_page": False,
        "assertion_supported": False if agent_assertion else None,
        "evidence_summary": "stub",
        "confidence": 0.9,
        "reason": "stub",
    }


# Patch all references — both at import site and at use site
from src.evaluator_lib import constraint_judge, decomposer
constraint_judge.simple_verify = stub_simple_verify
constraint_judge.verify_branch_condition = stub_verify_branch_condition
constraint_judge.verify_claim_on_frame = stub_verify_claim_on_frame
core.simple_verify = stub_simple_verify
core.verify_branch_condition = stub_verify_branch_condition
core.verify_claim_on_frame = stub_verify_claim_on_frame
decomposer.decompose = stub_decompose
core.decompose = stub_decompose


# ---------------------------------------------------------------------------
# Build a fake run directory
# ---------------------------------------------------------------------------

def build_fake_run(run_dir: Path, task_id: str, done_message: str) -> None:
    task_dir = run_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    events = [
        {"step": 1, "url": "https://www.ctrip.com/", "action": "click('1')",
         "raw_output": "...", "injected_revision_ids": [], "trigger_checks": []},
        {"step": 2, "url": "https://www.ctrip.com/list?from=BJS",
         "action": "click('2')", "raw_output": "...",
         "injected_revision_ids": ["i1"], "trigger_checks": []},
        {"step": 3, "url": "https://www.ctrip.com/list?from=BJS&filter=non-redeye",
         "action": "click('3')", "raw_output": "...",
         "injected_revision_ids": [], "trigger_checks": []},
        {"step": 4, "url": "https://www.ctrip.com/list?from=BJS&filter=non-redeye",
         "action": "scroll(0, 300)", "raw_output": "...",
         "injected_revision_ids": ["i2"], "trigger_checks": []},
        {"step": 5, "url": "https://www.ctrip.com/booking",
         "action": f"send_msg_to_user({json.dumps(done_message)})",
         "raw_output": "...", "injected_revision_ids": [], "trigger_checks": []},
    ]
    with (task_dir / "events.jsonl").open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    # summary.json: which revisions fired (i0 implicit; i1 from m1; i2 from m2)
    summary = {
        "instance_id": task_id,
        "fired_revisions": ["i1", "i2"],
        "fired_milestones": ["m1", "m2"],
        "steps": len(events),
    }
    with (task_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # axtrees — axtree content is irrelevant for simple sub_claims
    for s in range(1, len(events) + 1):
        with (task_dir / f"step_{s:03d}.axtree").open("w", encoding="utf-8") as f:
            f.write("dummy axtree content")


# ---------------------------------------------------------------------------
# Run one variant + assert
# ---------------------------------------------------------------------------

def run_variant(name, done_message, expected):
    print(f"\n{'='*68}\nVariant: {name}\n{'='*68}")
    print(f"done_message preview: {done_message[:140]}...\n")

    task_path = REPO_ROOT / "data" / "dyna-tasks" / "v1_conditional" / "sanity_flight_001.jsonl"
    tasks = load_dyna_jsonl(str(task_path))
    task = tasks[0]

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        build_fake_run(run_dir, task.instance_id, done_message)
        log_reader = LogReader(run_dir, task.instance_id)
        evaluator = core.TaskEvaluator(
            task=task,
            log_reader=log_reader,
            client=None,
            model="stub",
        )
        result = evaluator.evaluate()

    br = result.get("branch_resolution", {})
    print(f"branch_resolution:")
    print(f"  condition_state = {br.get('condition_state')}")
    print(f"  active_branch   = {br.get('active_branch')}")
    print(f"  active_cid      = {br.get('active_constraint_id')}")
    print(f"  skipped_cids    = {br.get('skipped_constraint_ids')}")
    print(f"  reason          = {br.get('reason')!r}")

    print(f"\ncorrectness_score = {result['correctness_score']}")
    print(f"is_correct        = {result['is_correct']}")
    print(f"overall_reason    = {result['overall_reason']!r}")
    print(f"diagnostic_counts = {result['diagnostic_counts']}")
    print(f"\nconstraint_results:")
    for cr in result["constraint_results"]:
        sat = cr["satisfied"]
        sat_disp = "SKIPPED" if sat is None else ("PASS" if sat else "FAIL")
        print(f"  {cr['constraint_id']} ({cr['source_revision']}): "
              f"{sat_disp:7s}  label={cr['combine_label']:24s}  verdict={cr['verdict']}")

    print(f"\nrevision_results:")
    for rv in result.get("revision_results", []):
        print(f"  {rv['revision_id']}: satisfied={rv['satisfied']}, cids={rv['constraint_ids']}, failed={rv['failed_constraints']}")

    # Assertions
    fails = []
    if br.get("condition_state") != expected["condition_state"]:
        fails.append(f"condition_state: expected {expected['condition_state']}, got {br.get('condition_state')}")
    if sorted(br.get("skipped_constraint_ids", [])) != sorted(expected["skipped"]):
        fails.append(f"skipped: expected {expected['skipped']}, got {br.get('skipped_constraint_ids')}")
    if abs(result["correctness_score"] - expected["score"]) > 1e-9:
        fails.append(f"score: expected {expected['score']}, got {result['correctness_score']}")
    if result["is_correct"] != expected["is_correct"]:
        fails.append(f"is_correct: expected {expected['is_correct']}, got {result['is_correct']}")
    for cid, want_label in expected["labels"].items():
        cr = next((r for r in result["constraint_results"] if r["constraint_id"] == cid), None)
        if cr is None:
            fails.append(f"{cid}: missing")
            continue
        if cr["combine_label"] != want_label:
            fails.append(f"{cid}: expected label={want_label}, got {cr['combine_label']}")

    if fails:
        print("\nFAILURES:")
        for f in fails:
            print(f"  - {f}")
        return False
    print("\nPASS")
    return True


# ---------------------------------------------------------------------------
# Variants per design doc §8
# ---------------------------------------------------------------------------

DONE_A = """done: I selected flight CA1858, Beijing->Shanghai, departing 14:00 the day after tomorrow.

[CONDITIONAL OBSERVATION] I compared prices: cheapest non-red-eye is 880 RMB (CA1858, 14:00), cheapest red-eye is 720 RMB (HU7605, 01:30). Price difference is 160 RMB, BELOW the 500 RMB threshold, so I kept the non-red-eye preference.

Selected: CA1858, 14:00, non-red-eye, 880 RMB."""

DONE_B = """done: I selected flight MU5101, Beijing->Shanghai, departing 02:30 the day after tomorrow.

[CONDITIONAL OBSERVATION] I compared prices: cheapest non-red-eye is 1480 RMB (CA1858, 14:00), cheapest red-eye is 580 RMB (MU5101, 02:30). Price difference is 900 RMB, EXCEEDING the 500 RMB threshold, so I switched to the cheapest red-eye option.

Selected: MU5101, 02:30, red-eye, 580 RMB."""

DONE_C = """done: I selected flight CA1858, Beijing->Shanghai, departing 14:00 the day after tomorrow. Booking complete.

Selected: CA1858, 14:00, non-red-eye, 880 RMB."""


def main() -> int:
    ok = True
    ok &= run_variant(
        "A: condition false (diff 160<500, agent kept default)",
        DONE_A,
        expected={
            "condition_state": "false",
            "skipped": [5],
            "score": 1.0,
            "is_correct": True,
            "labels": {
                "c1": "confirmed", "c2": "confirmed", "c3": "confirmed",
                "c4": "confirmed", "c5": "branch_skipped",
            },
        },
    )
    ok &= run_variant(
        "B: condition true (diff 900>500, agent switched to alternative)",
        DONE_B,
        expected={
            "condition_state": "true",
            "skipped": [4],
            "score": 1.0,
            "is_correct": True,
            "labels": {
                "c1": "confirmed", "c2": "confirmed", "c3": "confirmed",
                "c4": "branch_skipped", "c5": "confirmed",
            },
        },
    )
    ok &= run_variant(
        "C: agent did not address condition",
        DONE_C,
        expected={
            "condition_state": "not_addressed",
            "skipped": [4, 5],
            "score": 1.0,  # c1/c2/c3 only; c4/c5 excluded
            "is_correct": True,
            "labels": {
                "c1": "confirmed", "c2": "confirmed", "c3": "confirmed",
                "c4": "conditional_unaddressed",
                "c5": "conditional_unaddressed",
            },
        },
    )

    print("\n" + "=" * 68)
    if ok:
        print("ALL SANITY VARIANTS PASSED")
        return 0
    print("SOME SANITY VARIANTS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
