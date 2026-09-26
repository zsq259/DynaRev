"""
Dyna Task Generation Pipeline

Converts single-turn Online-Mind2Web tasks into multi-turn Dyna tasks.

Pipeline steps:
  1. Extract constraints from original instruction
  2. Design I0 based on constraints + website homepage
  3. Arrange into multi-turn instructions + milestones
  4. Quality review (accept / reject)
  5. Constraint binding (post-processing)

Usage:
  # Process a single task
  python dyna_pipeline.py --task "Find ..." --website "https://..."

  # Batch process from Online-Mind2Web CSV
  python dyna_pipeline.py --input-csv dyna_small_dataset_v1.csv

  # Dry run (step 1 only, no browser)
  python dyna_pipeline.py --task "Find ..." --website "https://..." --dry-run
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from pipeline_lib.export import to_dyna_task_json, write_dyna_jsonl
from pipeline_lib.homepage import fetch_homepage
from pipeline_lib.models import PipelineInput, PipelineOutput
from pipeline_lib.step1_constraints import extract_constraints, should_filter_step1
from pipeline_lib.step2_i0_design import design_i0
from pipeline_lib.step3_router import run_step3
from pipeline_lib.step3c_align import align_related_constraints
from pipeline_lib.step4_review import review_quality
from pipeline_lib.step5_binding import bind_constraints
from utils.client import get_client


def _build_override_context(
    step3a: Any,
) -> Dict[str, Any]:
    """Build override context from the chosen Step 3b item (stored on step3a).

    Step 4 (review) and Step 5 (binding) consume this. In conditional
    mode the additional ``switching_condition`` per item lets those
    LLM steps reason about the conditional clause; ``override_mode``
    lets them branch their interpretation rules.
    """
    overrides = []
    for item in step3a.overrides:
        entry: Dict[str, Any] = {
            "target_constraint_id": item.target_constraint_id,
            "target_constraint": item.target_constraint_text,
            "replacement_pair_id": item.replacement_pair_id,
            "replacement_constraint": item.replacement_constraint_text,
            "dimension": item.dimension,
        }
        if item.switching_condition is not None:
            sc = item.switching_condition
            entry["switching_condition"] = {
                "dimension": sc.dimension,
                "operator": sc.operator,
                "threshold": sc.threshold,
                "unit": sc.unit,
                "condition_text": sc.condition_text,
            }
        overrides.append(entry)

    return {
        "enable_override": bool(step3a.enable_override),
        "override_mode": step3a.override_mode,
        "overrides": overrides,
    }


def _configure_pipeline_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dyna_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def run_pipeline(
    task: PipelineInput,
    client: OpenAI,
    model: str = "gpt-4o",
    instruction_track_mode: str = "base",
    override_planning_strategy: str = "rule_based",
    override_insertion_timing: str = "early",
    replacement_pairs: Dict[str, Any] | None = None,
    dry_run: bool = False,
    logger: logging.Logger | None = None,
    override_mode: str = "unconditional",
    switching_conditions: Dict[str, Any] | None = None,
) -> PipelineOutput:
    """Run the full 5-step pipeline on a single task with conditional retry logic."""
    if logger is None:
        logger = logging.getLogger("dyna_pipeline")
    output = PipelineOutput(input=task)
    retry_count = 0
    max_retries = 2

    # ── Step 1: Constraint Extraction ──
    logger.info("STEP1 | %s | Extracting constraints...", task.task_id)
    try:
        output.step1 = extract_constraints(task.original_instruction, client, model)
        logger.info(
            "STEP1 | %s | Found %d constraints",
            task.task_id, len(output.step1.constraints),
        )
    except Exception as e:
        output.filtered_at = "step1"
        output.filter_reason = f"LLM error: {str(e)[:200]}"
        logger.error("STEP1 | %s | Failed: %s", task.task_id, output.filter_reason)
        return output

    filter_reason = should_filter_step1(output.step1)
    if filter_reason:
        output.filtered_at = "step1"
        output.filter_reason = filter_reason
        logger.warning("STEP1 | %s | Filtered: %s", task.task_id, filter_reason)
        return output

    if dry_run:
        logger.info("STEP1 | %s | Dry run — stopping after step 1", task.task_id)
        return output

    # ── Step 2: I0 Design ──
    logger.info("STEP2 | %s | Fetching homepage...", task.task_id)
    homepage_result = fetch_homepage(task.website)

    if not homepage_result.accessible:
        output.filtered_at = "step2"
        output.filter_reason = f"Website inaccessible: {homepage_result.block_reason}"
        logger.warning(
            "STEP2 | %s | Filtered — site inaccessible: %s",
            task.task_id, homepage_result.block_reason,
        )
        return output

    homepage_desc = homepage_result.description
    output.homepage_description = homepage_desc

    # Retry loop for Step 2-5
    previous_retry_reason: str | None = None
    next_retry_from = "step2"
    attempt_history: list[dict] = []
    while True:
        current_attempt: Dict[str, Any] = {
            "attempt": retry_count + 1,
            "retry_from": next_retry_from,
            "instruction_track_mode": instruction_track_mode,
        }

        # ── Step 2: I0 Design ──
        if next_retry_from == "step2" or output.step2 is None:
            logger.info("STEP2 | %s | Designing I0... (attempt %d)", task.task_id, retry_count + 1)
            try:
                output.step2 = design_i0(
                    output.step1,
                    homepage_desc,
                    client,
                    model,
                    previous_step4_reason=previous_retry_reason,
                )
                logger.info("STEP2 | %s | I0: %s", task.task_id, output.step2.i0[:100])
                current_attempt["step2_i0"] = output.step2.i0
            except Exception as e:
                output.filtered_at = "step2"
                output.filter_reason = f"LLM error: {str(e)[:200]}"
                logger.error("STEP2 | %s | Failed: %s", task.task_id, output.filter_reason)
                current_attempt["step2_error"] = output.filter_reason
                output.attempt_records.append(current_attempt)
                return output
        else:
            logger.info("STEP2 | %s | Reusing previous I0 for attempt %d", task.task_id, retry_count + 1)
            current_attempt["step2_i0"] = output.step2.i0 if output.step2 else ""

        # ── Step 3: Multi-turn Instructions ──
        logger.info("STEP3 | %s | Designing multi-turn sequence... (attempt %d)", task.task_id, retry_count + 1)
        try:
            output.step3a, output.step3 = run_step3(
                constraints=output.step1,
                i0=output.step2,
                client=client,
                model=model,
                instruction_track_mode=instruction_track_mode,
                override_planning_strategy=override_planning_strategy,
                override_insertion_timing=override_insertion_timing,
                replacement_pairs=replacement_pairs,
                scene=task.scene,
                previous_step4_reason=previous_retry_reason,
                attempt_history=attempt_history if attempt_history else None,
                override_mode=override_mode,
                switching_conditions=switching_conditions,
                website=task.website,
            )

            # In override mode, distinguish candidate miss from LLM selection miss.
            if instruction_track_mode == "override":
                if output.step3a is None:
                    output.filtered_at = "step3a"
                    output.filter_reason = "Override plan missing"
                    logger.warning(
                        "STEP3A | %s | Filtered: %s",
                        task.task_id,
                        output.filter_reason,
                    )
                    current_attempt["step3a_filtered"] = output.filter_reason
                    output.attempt_records.append(current_attempt)
                    return output
                logger.info(
                    "STEP3A | %s | %d override candidates",
                    task.task_id,
                    len(output.step3a.candidates),
                )
                if (
                    output.step3 is None
                    or not output.step3a.enable_override
                    or not output.step3a.overrides
                ):
                    output.filtered_at = "step3b"
                    output.filter_reason = "Override instruction generation declined or invalid"
                    logger.warning(
                        "STEP3B | %s | Filtered: %s",
                        task.task_id,
                        output.filter_reason,
                    )
                    current_attempt["step3b_filtered"] = output.filter_reason
                    output.attempt_records.append(current_attempt)
                    return output
                if override_mode == "conditional":
                    chosen = output.step3a.overrides[0]
                    sc = chosen.switching_condition
                    if (
                        output.step3a.override_mode != "conditional"
                        or sc is None
                        or not sc.condition_text
                    ):
                        output.filtered_at = "step3b"
                        output.filter_reason = (
                            "Conditional override missing switching condition; "
                            "refusing unconditional fallback"
                        )
                        logger.warning(
                            "STEP3B | %s | Filtered: %s",
                            task.task_id,
                            output.filter_reason,
                        )
                        current_attempt["step3b_filtered"] = output.filter_reason
                        output.attempt_records.append(current_attempt)
                        return output

            if output.step3 is None:
                raise RuntimeError("Step3 generation returned no output")

            n_inst = len(output.step3.instructions)
            logger.info("STEP3 | %s | %d instructions", task.task_id, n_inst)
            if output.step3.reflection:
                logger.info(
                    "STEP3 | %s | Reflection: loop=%s next_move=%s",
                    task.task_id,
                    output.step3.reflection.get("in_text_swap_loop"),
                    str(output.step3.reflection.get("next_move", ""))[:180],
                )
            current_attempt["step3_instruction_count"] = n_inst
            current_attempt["step3_instructions"] = [
                {
                    "id": inst.id,
                    "instruction": inst.instruction,
                    "milestone": inst.milestone,
                    "type": inst.revision_type,
                }
                for inst in output.step3.instructions
            ]
            current_attempt["override_instructions"] = [
                {
                    "id": inst.id,
                    "instruction": inst.instruction,
                    "milestone": inst.milestone,
                    "type": inst.revision_type,
                }
                for inst in output.step3.instructions
                if inst.revision_type in ("override", "conditional_override")
            ]
        except Exception as e:
            output.filtered_at = "step3"
            output.filter_reason = f"Step3 error: {str(e)[:200]}"
            logger.error("STEP3 | %s | Failed: %s", task.task_id, output.filter_reason)
            current_attempt["step3_error"] = output.filter_reason
            output.attempt_records.append(current_attempt)
            return output

        override_context: Dict[str, Any] | None = None
        if instruction_track_mode == "override":
            override_context = _build_override_context(output.step3a)
            current_attempt["override_context"] = override_context

        # ── Step 4: Quality Review ──
        logger.info("STEP4 | %s | Reviewing quality... (attempt %d)", task.task_id, retry_count + 1)
        try:
            output.step4 = review_quality(
                task.original_instruction,
                output.step3,
                client,
                model,
                previous_step4_reason=previous_retry_reason,
                override_context=override_context,
            )
            status = "PASSED" if output.step4.passed else "FAILED"
            logger.info(
                "STEP4 | %s | %s: %s", task.task_id, status, output.step4.reason[:100],
            )
            current_attempt["step4_passed"] = output.step4.passed
            current_attempt["step4_reason"] = output.step4.reason
            current_attempt["step4_retry_from_step"] = output.step4.retry_from_step

            if not output.step4.passed:
                # Failed — check if worth retrying
                if output.step4.worth_retry and retry_count < max_retries:
                    retry_from = output.step4.retry_from_step
                    logger.info(
                        "STEP4 | %s | Worth retry from %s: %s",
                        task.task_id, retry_from, output.step4.reason,
                    )
                    attempt_history.append({
                        "attempt": retry_count + 1,
                        "i0": output.step2.i0 if output.step2 else "",
                        "instructions": [
                            {"id": inst.id, "instruction": inst.instruction, "milestone": inst.milestone}
                            for inst in output.step3.instructions
                        ],
                        "critique": output.step4.reason,
                    })
                    previous_retry_reason = output.step4.reason
                    next_retry_from = retry_from or "step3"
                    if next_retry_from == "step2":
                        output.step2 = None
                    output.step3a = None
                    output.step3 = None
                    output.step3c = None
                    output.step4 = None
                    output.step5 = None
                    output.attempt_records.append(current_attempt)
                    retry_count += 1
                    continue

                # Not worth retrying or max retries reached
                output.filtered_at = "step4"
                output.filter_reason = output.step4.reason
                logger.warning(
                    "STEP4 | %s | Rejected (not worth retry or max retries): %s",
                    task.task_id, output.step4.reason[:100],
                )
                output.attempt_records.append(current_attempt)
                break

        except Exception as e:
            output.filtered_at = "step4"
            output.filter_reason = f"LLM error: {str(e)[:200]}"
            logger.error("STEP4 | %s | Failed: %s", task.task_id, output.filter_reason)
            current_attempt["step4_error"] = output.filter_reason
            output.attempt_records.append(current_attempt)
            return output

        # ── Step 3c: sync descriptions that still state the old requirement ──
        if (
            instruction_track_mode == "override"
            and output.step3a
            and output.step3a.enable_override
            and output.step3a.overrides
        ):
            logger.info("STEP3C | %s | Syncing related constraints...", task.task_id)
            try:
                output.step3c = align_related_constraints(
                    output.step1,
                    output.step3.instructions,
                    output.step3a.overrides[0],
                    output.step3a.override_mode,
                    client,
                    model,
                )
            except Exception as e:
                output.filtered_at = "step3c"
                output.filter_reason = f"LLM error: {str(e)[:200]}"
                logger.error("STEP3C | %s | Failed: %s", task.task_id, output.filter_reason)
                current_attempt["step3c_error"] = output.filter_reason
                output.attempt_records.append(current_attempt)
                return output

        # ── Step 5: Constraint Binding ──
        logger.info("STEP5 | %s | Binding constraints to instructions... (attempt %d)", task.task_id, retry_count + 1)
        try:
            output.step5 = bind_constraints(
                output.step3c or output.step1,
                output.step2,
                output.step3,
                client,
                model,
                override_context=override_context,
            )
            n_issues = len(output.step5.issues)
            status = "PASSED" if output.step5.passed else "FAILED"
            logger.info("STEP5 | %s | %s", task.task_id, status)
            if output.step5.issues:
                for issue in output.step5.issues:
                    logger.warning("STEP5 | %s | Issue: %s", task.task_id, issue)
            current_attempt["step5_passed"] = output.step5.passed
            current_attempt["step5_issues"] = output.step5.issues
            current_attempt["step5_retry_from_step"] = output.step5.retry_from_step

            if output.step5.passed:
                output.attempt_records.append(current_attempt)
                break

            # Step 5 failed — retry if possible
            if retry_count < max_retries:
                retry_from = output.step5.retry_from_step or "step3"
                logger.info(
                    "STEP5 | %s | Retry from %s: %s",
                    task.task_id, retry_from, "; ".join(output.step5.issues),
                )
                attempt_history.append({
                    "attempt": retry_count + 1,
                    "i0": output.step2.i0 if output.step2 else "",
                    "instructions": [
                        {"id": inst.id, "instruction": inst.instruction, "milestone": inst.milestone}
                        for inst in output.step3.instructions
                    ],
                    "critique": f"Constraint binding failed: {'; '.join(output.step5.issues)}",
                })
                previous_retry_reason = f"Constraint binding failed: {'; '.join(output.step5.issues)}"
                next_retry_from = retry_from
                if next_retry_from == "step2":
                    output.step2 = None
                output.step3a = None
                output.step3 = None
                output.step3c = None
                output.step4 = None
                output.step5 = None
                output.attempt_records.append(current_attempt)
                retry_count += 1
                continue

            # Max retries reached
            output.filtered_at = "step5"
            output.filter_reason = f"Binding failed: {'; '.join(output.step5.issues)}"
            logger.warning(
                "STEP5 | %s | Rejected (max retries): %s",
                task.task_id, output.filter_reason[:100],
            )
            output.attempt_records.append(current_attempt)
            break

        except Exception as e:
            output.filtered_at = "step5"
            output.filter_reason = f"LLM error: {str(e)[:200]}"
            logger.error("STEP5 | %s | Failed: %s", task.task_id, output.filter_reason)
            current_attempt["step5_error"] = output.filter_reason
            output.attempt_records.append(current_attempt)
            return output

    return output


def load_inputs_from_csv(csv_path: str) -> List[PipelineInput]:
    """Load pipeline inputs from dyna_small_dataset CSV."""
    inputs: List[PipelineInput] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inputs.append(PipelineInput(
                task_id=row.get("task_id", ""),
                website=row.get("website", ""),
                original_instruction=row.get("confirmed_task", row.get("base_task", "")),
                scene=row.get("primary_scene", row.get("scene_label", "")),
                dataset=row.get("source_dataset") or row.get("dataset") or "osunlp/Online-Mind2Web",
            ))
    return inputs


def load_replacement_pairs(path: str, logger: logging.Logger) -> Dict[str, Any]:
    """Load replacement-pairs JSON used by override track."""
    rp = Path(path)
    if not rp.exists():
        raise FileNotFoundError(f"replacement pairs file not found: {path}")
    try:
        payload = json.loads(rp.read_text(encoding="utf-8"))
        logger.info("OVERRIDE | Loaded replacement pairs from %s", path)
        return payload
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid replacement pairs JSON: {e}") from e


def collect_done_task_ids(log_dir: str) -> set:
    """Scan all report.json files under log_dir for passed task IDs."""
    done = set()
    log_path = Path(log_dir)
    if not log_path.exists():
        return done
    for report_file in log_path.glob("*/report.json"):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                if entry.get("step4_passed") is True:
                    done.add(entry["task_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def save_pipeline_report(outputs: List[PipelineOutput], path: str) -> None:
    """Save full pipeline trace for debugging.

    Atomic write: stages to <path>.tmp then renames so a crash mid-write can't
    corrupt an existing report (matters because we call this incrementally).
    """
    report: List[Dict[str, Any]] = []
    for o in outputs:
        entry: Dict[str, Any] = {
            "task_id": o.input.task_id,
            "website": o.input.website,
            "original_instruction": o.input.original_instruction,
            "filtered_at": o.filtered_at,
            "filter_reason": o.filter_reason,
        }
        if o.step1:
            entry["step1_constraints"] = [
                {"id": c.id, "description": c.description}
                for c in o.step1.constraints
            ]
        if o.step2:
            entry["step2_homepage_description"] = o.homepage_description
            entry["step2_classification"] = o.step2.classification
            entry["step2_i0"] = o.step2.i0
        if o.step3a:
            entry["step3a_override_plan"] = {
                "enable_override": o.step3a.enable_override,
                "override_mode": o.step3a.override_mode,
                "candidates": [
                    {
                        "target_constraint_id": item.target_constraint_id,
                        "target_constraint_text": item.target_constraint_text,
                        "replacement_pair_id": item.replacement_pair_id,
                        "pair_a": item.pair_a,
                        "pair_b": item.pair_b,
                        "dimension": item.dimension,
                        "score": item.score,
                    }
                    for item in o.step3a.candidates
                ],
                "overrides": [
                    {
                        "target_constraint_id": item.target_constraint_id,
                        "target_constraint_text": item.target_constraint_text,
                        "replacement_pair_id": item.replacement_pair_id,
                        "replacement_constraint_text": item.replacement_constraint_text,
                        "pair_a": item.pair_a,
                        "pair_b": item.pair_b,
                        "dimension": item.dimension,
                        **(
                            {"switching_condition": {
                                "dimension": item.switching_condition.dimension,
                                "operator": item.switching_condition.operator,
                                "threshold": item.switching_condition.threshold,
                                "unit": item.switching_condition.unit,
                                "condition_text": item.switching_condition.condition_text,
                            }} if item.switching_condition else {}
                        ),
                    }
                    for item in o.step3a.overrides
                ],
            }
        if o.step3c:
            entry["step3c_constraints"] = [
                {
                    "id": c.id,
                    "description": c.description,
                    **({"branch": c.branch} if c.branch else {}),
                }
                for c in o.step3c.constraints
            ]
        if o.step3:
            entry["step3_instructions"] = [
                {
                    "id": inst.id,
                    "instruction": inst.instruction,
                    "milestone": inst.milestone,
                    "type": inst.revision_type,
                }
                for inst in o.step3.instructions
            ]
        if o.attempt_records:
            entry["attempt_records"] = o.attempt_records
        if o.step4:
            entry["step4_passed"] = o.step4.passed
            entry["step4_worth_retry"] = o.step4.worth_retry
            entry["step4_retry_from_step"] = o.step4.retry_from_step
            entry["step4_reason"] = o.step4.reason
        if o.step5:
            entry["step5_passed"] = o.step5.passed
            entry["step5_issues"] = o.step5.issues
            entry["step5_mapping"] = [
                {
                    "constraint_id": e.constraint_id,
                    "instruction_id": e.instruction_id,
                    "confidence": e.confidence,
                    "reason": e.reason,
                }
                for e in o.step5.mapping
            ]
        report.append(entry)

    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    Path(path + ".tmp").replace(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dyna Task Generation Pipeline")

    # Single task mode
    p.add_argument("--task", default=None, help="Single task instruction to process")
    p.add_argument("--website", default=None, help="Website URL for single task mode")
    p.add_argument("--task-id", default="manual_001", help="Task ID for single task mode")
    p.add_argument("--scene", default="", help="Scene label for single task mode")

    # Batch mode
    p.add_argument("--input-csv", default=None, help="CSV file with tasks to process")
    p.add_argument("--limit", type=int, default=0, help="Max tasks to process (0 = all)")
    p.add_argument("--offset", type=int, default=0, help="Skip first N tasks")

    # Output
    p.add_argument("--run-name", default=None, help="Run name (default: pipeline_YYYYMMDD_HHMMSS)")
    p.add_argument("--log-dir", default="logs/pipeline", help="Base log directory")

    # Model
    p.add_argument("--model", default="gpt-4o", help="LLM model for all steps")
    p.add_argument("--openai-base-url", default=None)

    # Step 3 routing
    p.add_argument(
        "--instruction-track-mode",
        choices=["base", "override"],
        default="base",
        help="Step 3 generation track",
    )
    p.add_argument(
        "--override-planning-strategy",
        choices=["rule_based", "hybrid", "llm_only"],
        default="rule_based",
        help="Unused (kept for old scripts). Step 3a only builds cosine candidates; Step 3b selects.",
    )
    p.add_argument(
        "--override-insertion-timing",
        choices=["early", "late"],
        default="early",
        help="Prompt-level override timing guidance for Step 3b",
    )
    p.add_argument(
        "--replacement-pairs-path",
        default="data/pipeline/replacement_pairs.json",
        help="Replacement-pairs JSON path used in override mode",
    )
    p.add_argument(
        "--override-mode",
        choices=["unconditional", "conditional"],
        default="unconditional",
        help="Override sub-mode (only used when --instruction-track-mode=override)",
    )
    p.add_argument(
        "--switching-conditions-path",
        default="data/pipeline/switching_conditions.yaml",
        help="Unused (kept for old scripts). Conditional Step 3b picks count/price/rating/time.",
    )

    # Options
    p.add_argument("--dry-run", action="store_true", help="Run step 1 only, no browser")
    p.add_argument("--skip-done", action="store_true",
                   help="Skip tasks already accepted in previous runs (scans log-dir)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Run directory & logging ──
    run_name = args.run_name or datetime.now().strftime("pipeline_%Y%m%d_%H%M%S")
    run_dir = Path(args.log_dir) / run_name
    logger = _configure_pipeline_logging(run_dir)

    # Build input list
    if args.task:
        if not args.website:
            print("Error: --website is required in single task mode", file=sys.stderr)
            sys.exit(1)
        inputs = [PipelineInput(
            task_id=args.task_id,
            website=args.website,
            original_instruction=args.task,
            scene=args.scene,
        )]
    elif args.input_csv:
        inputs = load_inputs_from_csv(args.input_csv)
    else:
        print("Error: provide --task or --input-csv", file=sys.stderr)
        sys.exit(1)

    # Apply offset/limit
    if args.offset > 0:
        inputs = inputs[args.offset:]
    if args.limit > 0:
        inputs = inputs[:args.limit]

    # Skip already-done tasks
    if args.skip_done:
        done_ids = collect_done_task_ids(args.log_dir)
        before = len(inputs)
        inputs = [t for t in inputs if t.task_id not in done_ids]
        logger.info("SKIP_DONE | %d already passed, %d → %d tasks remaining",
                     len(done_ids), before, len(inputs))

    logger.info("PIPELINE_START | tasks=%d | model=%s", len(inputs), args.model)

    replacement_pairs: Dict[str, Any] | None = None
    switching_conditions: Dict[str, Any] | None = None
    if args.instruction_track_mode == "override":
        replacement_pairs = load_replacement_pairs(args.replacement_pairs_path, logger)
        logger.info(
            "PIPELINE_OVERRIDE | timing=%s | mode=%s | planning_strategy_ignored=%s",
            args.override_insertion_timing,
            args.override_mode,
            args.override_planning_strategy,
        )
        if args.override_mode == "conditional":
            logger.info(
                "PIPELINE_CONDITIONAL | third_dim=count|price|rating|time | yaml_unused=%s",
                args.switching_conditions_path,
            )

    # OpenAI client
    client = get_client(args.model)

    # Output paths (persisted incrementally so a crash doesn't lose finished tasks)
    report_path = str(run_dir / "report.json")
    jsonl_path = str(run_dir / "tasks.jsonl")
    if not args.skip_done:
        # Fresh run — clear any stale partial outputs from a prior crash
        for p in (report_path, jsonl_path):
            if Path(p).exists():
                Path(p).unlink()

    # Run pipeline
    outputs: List[PipelineOutput] = []
    for task_input in inputs:
        result = run_pipeline(
            task=task_input,
            client=client,
            model=args.model,
            instruction_track_mode=args.instruction_track_mode,
            override_planning_strategy=args.override_planning_strategy,
            override_insertion_timing=args.override_insertion_timing,
            replacement_pairs=replacement_pairs,
            dry_run=args.dry_run,
            logger=logger,
            override_mode=args.override_mode,
            switching_conditions=switching_conditions,
        )
        outputs.append(result)

        # Incremental persistence — runs after every task so a mid-run crash
        # leaves a valid report.json + tasks.jsonl for `--skip-done` resume.
        if not args.dry_run:
            save_pipeline_report(outputs, report_path)
            if not result.filtered_at and result.step3:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(to_dyna_task_json(result), ensure_ascii=False) + "\n")

    # Summary
    accepted = [o for o in outputs if not o.filtered_at]
    filtered = [o for o in outputs if o.filtered_at]
    logger.info(
        "PIPELINE_END | total=%d | passed=%d | filtered=%d",
        len(outputs), len(accepted), len(filtered),
    )

    # Final report (already up-to-date from incremental writes; keep an explicit
    # final write so dry-run path still produces a report).
    if args.dry_run:
        save_pipeline_report(outputs, report_path)
    logger.info("REPORT | Saved to %s", report_path)
    if not args.dry_run:
        n_jsonl = sum(1 for _ in open(jsonl_path)) if Path(jsonl_path).exists() else 0
        logger.info("EXPORT | %d tasks in %s", n_jsonl, jsonl_path)

    print(f"\n>>> Run directory: {run_dir}")


if __name__ == "__main__":
    main()
