#!/usr/bin/env python3
"""Build CTS/CCS curves on a paired multi-model cohort.

Supports action-step budgets and token budgets (requires runner
summary.token_usage from the token meter).

A task enters the cohort only when all models ran and evaluated the same
(subset, instance_id) pair, and none of those runs ended in a runtime/system
failure.

Run inventory comes from a JSON config (``--runs-config``) mapping model name
to subset -> run directory name under ``--runs-root``::

    {
      "Qwen3-14B": {
        "om2w": "batch0_om2w_qwen3-14b_20260920",
        "override_om2w": "batch0_override_om2w_qwen3-14b_20260920"
      },
      "Claude-Sonnet-4-6": { ... same subsets ... }
    }

Outputs (``--label`` prefix, suffix depends on --cost steps|tokens):
  - {label}_task_records_{cost}.csv
  - {label}_cost_curves_{cost}.csv
  - {label}_cost_curves_{cost}.{svg,pdf,png}
  - {label}_cost_curves_{cost}.metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ModelRuns = dict[str, dict[str, str]]

COLOR_PALETTE = [
    "#3366CC",
    "#D94841",
    "#2E8B57",
    "#8E44AD",
    "#E67E22",
    "#16A085",
    "#7F8C8D",
]

# These indicate infrastructure/runtime failure rather than a completed agent
# attempt. Old "stuck" results remain in the cohort because they were an
# intentional runner termination rule in the original experiments.
INVALID_STOP_REASONS = {
    "env_error",
    "env_setup_error",
    "task_exception",
    "system_error",
    "watchdog_timeout",
}


@dataclass(frozen=True)
class Observation:
    model: str
    subset: str
    task_id: str
    run_name: str
    steps: int
    tokens: int | None
    score: float
    passed: int
    stopped_reason: str

    def cost(self, cost_kind: str) -> int:
        if cost_kind == "steps":
            return self.steps
        if cost_kind == "tokens":
            if self.tokens is None:
                raise ValueError(
                    f"missing token_usage for {self.subset}/{self.task_id}/{self.model}"
                )
            return self.tokens
        raise ValueError(f"unsupported cost kind: {cost_kind}")


def load_model_runs(config_path: Path) -> ModelRuns:
    """Read and validate the model -> subset -> run_name mapping."""
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"runs config must be a non-empty JSON object: {config_path}")

    model_runs: ModelRuns = {}
    reference_subsets: set[str] | None = None
    for model, subset_runs in value.items():
        if not isinstance(subset_runs, dict) or not subset_runs:
            raise ValueError(f"runs config entry for {model!r} must map subset -> run name")
        for subset, run_name in subset_runs.items():
            if not isinstance(run_name, str) or not run_name:
                raise ValueError(f"invalid run name for {model!r}/{subset!r}")
        subsets = set(subset_runs)
        if reference_subsets is None:
            reference_subsets = subsets
        elif subsets != reference_subsets:
            raise ValueError(
                "all models must cover the same subsets; "
                f"{model!r} has {sorted(subsets)}, expected {sorted(reference_subsets)}"
            )
        model_runs[str(model)] = {str(k): str(v) for k, v in subset_runs.items()}
    return model_runs


def assign_colors(model_runs: ModelRuns) -> dict[str, str]:
    return {
        model: COLOR_PALETTE[index % len(COLOR_PALETTE)]
        for index, model in enumerate(model_runs)
    }


def read_token_total(summary: dict[str, Any]) -> int | None:
    usage = summary.get("token_usage")
    if not isinstance(usage, dict):
        return None
    if usage.get("total_tokens") is None:
        return None
    return int(usage["total_tokens"])


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_run(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    summaries: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*/summary.json")):
        summary = read_json(path)
        task_id = str(summary.get("instance_id") or path.parent.name)
        summaries[task_id] = summary

    evaluations: dict[str, dict[str, Any]] = {}
    eval_dir = run_dir / "evaluations"
    for path in sorted(eval_dir.glob("*.evaluation.json")):
        evaluation = read_json(path)
        task_id = str(evaluation.get("task_id") or path.name.removesuffix(".evaluation.json"))
        result = evaluation.get("result")
        if isinstance(result, dict) and result.get("correctness_score") is not None:
            evaluations[task_id] = result

    return summaries, evaluations


def build_paired_cohort(
    runs_root: Path,
    model_runs: ModelRuns,
) -> tuple[list[Observation], dict[str, Any]]:
    loaded: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    run_inventory: dict[str, dict[str, dict[str, int | str]]] = {}

    for model, subset_runs in model_runs.items():
        run_inventory[model] = {}
        for subset, run_name in subset_runs.items():
            summaries, evaluations = load_run(runs_root / run_name)
            loaded[(model, subset)] = (summaries, evaluations)
            run_inventory[model][subset] = {
                "run_name": run_name,
                "summaries": len(summaries),
                "evaluations": len(evaluations),
            }

    observations: list[Observation] = []
    exclusions: list[dict[str, Any]] = []
    paired_before_filter: Counter[str] = Counter()
    paired_after_filter: Counter[str] = Counter()
    pass_flag_mismatches: list[dict[str, Any]] = []

    subsets = tuple(next(iter(model_runs.values())).keys())
    models = tuple(model_runs.keys())
    for subset in subsets:
        available_by_model: list[set[str]] = []
        for model in models:
            summaries, evaluations = loaded[(model, subset)]
            available_by_model.append(set(summaries) & set(evaluations))
        common_ids = set.intersection(*available_by_model)
        paired_before_filter[subset] = len(common_ids)

        for task_id in sorted(common_ids):
            invalid_runs: list[dict[str, str]] = []
            for model in models:
                summary = loaded[(model, subset)][0][task_id]
                reason = str(summary.get("stopped_reason") or "")
                if reason in INVALID_STOP_REASONS:
                    invalid_runs.append({"model": model, "stopped_reason": reason})
            if invalid_runs:
                exclusions.append(
                    {
                        "subset": subset,
                        "task_id": task_id,
                        "reason": "runtime_or_system_failure_in_at_least_one_model",
                        "invalid_runs": invalid_runs,
                    }
                )
                continue

            paired_after_filter[subset] += 1
            for model in models:
                summaries, evaluations = loaded[(model, subset)]
                summary = summaries[task_id]
                result = evaluations[task_id]
                steps = int(summary.get("steps", 0) or 0)
                tokens = read_token_total(summary)
                score = float(result["correctness_score"])
                if steps < 0:
                    raise ValueError(f"negative step count for {subset}/{task_id}/{model}")
                if tokens is not None and tokens < 0:
                    raise ValueError(f"negative token count for {subset}/{task_id}/{model}")
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(f"invalid score for {subset}/{task_id}/{model}: {score}")
                passed = int(math.isclose(score, 1.0, abs_tol=1e-12))
                if "is_correct" in result and bool(result["is_correct"]) != bool(passed):
                    pass_flag_mismatches.append(
                        {
                            "subset": subset,
                            "task_id": task_id,
                            "model": model,
                            "correctness_score": score,
                            "is_correct": bool(result["is_correct"]),
                        }
                    )
                observations.append(
                    Observation(
                        model=model,
                        subset=subset,
                        task_id=task_id,
                        run_name=model_runs[model][subset],
                        steps=steps,
                        tokens=tokens,
                        score=score,
                        passed=passed,
                        stopped_reason=str(summary.get("stopped_reason") or ""),
                    )
                )

    cohort_size = sum(paired_after_filter.values())
    expected_observations = cohort_size * len(models)
    if len(observations) != expected_observations:
        raise AssertionError(
            f"unbalanced cohort: {len(observations)} observations, "
            f"expected {expected_observations}"
        )

    audit = {
        "models": list(models),
        "subsets": list(subsets),
        "run_inventory": run_inventory,
        "paired_tasks_before_runtime_filter": {
            key: paired_before_filter[key] for key in subsets
        },
        "paired_tasks_after_runtime_filter": {
            key: paired_after_filter[key] for key in subsets
        },
        "cohort_size_before_runtime_filter": sum(paired_before_filter.values()),
        "cohort_size": cohort_size,
        "invalid_stop_reasons": sorted(INVALID_STOP_REASONS),
        "excluded_task_count": len(exclusions),
        "exclusions": exclusions,
        "score_is_correct_mismatches": pass_flag_mismatches,
    }
    return observations, audit


def compute_curves(
    observations: list[Observation],
    max_budget: int,
    cost_kind: str,
    models: tuple[str, ...],
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    budget_key = "action_budget" if cost_kind == "steps" else "token_budget"
    if cost_kind == "steps":
        budgets = list(range(0, max_budget + 1))
    else:
        observed = sorted(
            {
                item.cost(cost_kind)
                for item in observations
                if 0 < item.cost(cost_kind) <= max_budget
            }
        )
        budgets = [0]
        budgets.extend(observed)
        if max_budget not in budgets:
            budgets.append(max_budget)
    for model in models:
        model_records = [item for item in observations if item.model == model]
        for budget in budgets:
            eligible = [item for item in model_records if item.cost(cost_kind) <= budget]
            rows.append(
                {
                    "model": model,
                    budget_key: budget,
                    "cts": sum(item.passed for item in eligible),
                    "ccs": sum(item.score for item in eligible),
                    "eligible_tasks": len(eligible),
                    "cohort_size": len(model_records),
                }
            )
    return rows


def filter_observations_for_cost(
    observations: list[Observation],
    cost_kind: str,
) -> tuple[list[Observation], list[dict[str, Any]]]:
    if cost_kind == "steps":
        return observations, []
    kept: list[Observation] = []
    dropped: list[dict[str, Any]] = []
    for item in observations:
        if item.tokens is None:
            dropped.append(
                {
                    "subset": item.subset,
                    "task_id": item.task_id,
                    "model": item.model,
                    "reason": "missing_summary_token_usage",
                }
            )
            continue
        kept.append(item)
    return kept, dropped


def write_records(path: Path, observations: list[Observation]) -> None:
    fields = [
        "subset",
        "task_id",
        "model",
        "run_name",
        "steps",
        "tokens",
        "correctness_score",
        "task_pass",
        "stopped_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in sorted(observations, key=lambda x: (x.subset, x.task_id, x.model)):
            writer.writerow(
                {
                    "subset": item.subset,
                    "task_id": item.task_id,
                    "model": item.model,
                    "run_name": item.run_name,
                    "steps": item.steps,
                    "tokens": "" if item.tokens is None else item.tokens,
                    "correctness_score": f"{item.score:.10g}",
                    "task_pass": item.passed,
                    "stopped_reason": item.stopped_reason,
                }
            )


def write_curves(
    path: Path,
    rows: list[dict[str, int | float | str]],
    cost_kind: str,
) -> None:
    budget_key = "action_budget" if cost_kind == "steps" else "token_budget"
    fields = ["model", budget_key, "cts", "ccs", "eligible_tasks", "cohort_size"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_curves(
    output_stem: Path,
    rows: list[dict[str, int | float | str]],
    cohort_size: int,
    max_budget: int,
    cost_kind: str,
    model_colors: dict[str, str],
) -> list[Path]:
    # Import locally so non-plotting helpers remain usable without Matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    budget_key = "action_budget" if cost_kind == "steps" else "token_budget"
    xlabel = "Action-step budget" if cost_kind == "steps" else "Token budget"
    cost_label = "action-step" if cost_kind == "steps" else "token"

    grouped = {
        model: [row for row in rows if row["model"] == model] for model in model_colors
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.25), sharey=False)
    panels = [
        (axes[0], "cts", "Cumulative Task Success (CTS)"),
        (axes[1], "ccs", "Cumulative Constraint Satisfaction (CCS)"),
    ]
    marker_every = max(1, max_budget // 10)
    for axis, metric, title in panels:
        observed_max = max(float(row[metric]) for row in rows)
        if metric == "cts":
            y_max = max(1.0, math.ceil(observed_max + 1.0))
        else:
            y_max = max(5.0, math.ceil((observed_max * 1.1) / 5.0) * 5.0)
        for model, model_rows in grouped.items():
            budgets = [int(row[budget_key]) for row in model_rows]
            values = [float(row[metric]) for row in model_rows]
            axis.plot(
                budgets,
                values,
                label=model,
                color=model_colors[model],
                linewidth=2.1,
                marker="o",
                markersize=3.3,
                markevery=marker_every,
            )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_xlim(0, max_budget)
        axis.set_ylim(0, y_max)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=metric == "cts"))
        if cost_kind == "steps":
            axis.set_xticks(range(0, max_budget + 1, 10))
        else:
            tick = max(1, max_budget // 10)
            axis.set_xticks(range(0, max_budget + 1, tick))
        axis.grid(axis="both", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_ylabel("Raw cumulative value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(model_colors),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        f"Cost–quality curves ({cost_label}) on common valid tasks (N={cohort_size})",
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94), w_pad=2.5)

    outputs: list[Path] = []
    for suffix, options in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 300}),
    ):
        output_path = output_stem.with_suffix(suffix)
        fig.savefig(output_path, bbox_inches="tight", **options)
        outputs.append(output_path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot raw CTS/CCS curves under action-step or token budgets."
    )
    parser.add_argument(
        "--runs-config",
        type=Path,
        required=True,
        help="JSON file mapping model name -> subset -> run directory name",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("logs/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--label",
        default="common",
        help="prefix for output filenames (default: common)",
    )
    parser.add_argument(
        "--cost",
        choices=("steps", "tokens"),
        default="steps",
        help="cost axis for CTS/CCS (default: steps)",
    )
    parser.add_argument(
        "--max-budget",
        type=int,
        default=None,
        help="largest budget to report; default 50 for steps, max observed tokens for tokens",
    )
    args = parser.parse_args()

    model_runs = load_model_runs(args.runs_config)
    model_colors = assign_colors(model_runs)
    models = tuple(model_runs.keys())

    observations, audit = build_paired_cohort(args.runs_root, model_runs)
    observations, token_drop = filter_observations_for_cost(observations, args.cost)
    if args.cost == "tokens" and not observations:
        raise SystemExit(
            "no observations have summary.token_usage; re-run agents with the "
            "updated token meter before plotting token CTS/CCS"
        )

    if args.max_budget is None:
        if args.cost == "steps":
            max_budget = 50
        else:
            max_budget = max(item.tokens or 0 for item in observations)
    else:
        max_budget = args.max_budget
    if max_budget <= 0:
        parser.error("--max-budget must be positive")

    # Keep cohort balanced across models after token filtering.
    if args.cost == "tokens":
        by_key: dict[tuple[str, str], list[Observation]] = {}
        for item in observations:
            by_key.setdefault((item.subset, item.task_id), []).append(item)
        balanced: list[Observation] = []
        dropped_unbalanced: list[dict[str, Any]] = []
        for (subset, task_id), items in sorted(by_key.items()):
            present = {item.model for item in items}
            if present != set(models):
                dropped_unbalanced.append(
                    {
                        "subset": subset,
                        "task_id": task_id,
                        "reason": "token_usage_missing_for_some_models",
                        "present_models": sorted(present),
                    }
                )
                continue
            balanced.extend(items)
        observations = balanced
        token_drop.extend(dropped_unbalanced)
        if not observations:
            raise SystemExit("token-cost cohort is empty after requiring all models")

    curves = compute_curves(observations, max_budget, args.cost, models)
    cohort_size = len({(item.subset, item.task_id) for item in observations})

    aggregate_dir = args.output_dir / "aggregates"
    figure_dir = args.output_dir / "figures"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{args.label}_cost_curves_{args.cost}"
    records_path = aggregate_dir / f"{args.label}_task_records_{args.cost}.csv"
    curves_path = aggregate_dir / f"{stem}.csv"
    metadata_path = aggregate_dir / f"{stem}.metadata.json"
    figure_stem = figure_dir / stem

    write_records(records_path, observations)
    write_curves(curves_path, curves, args.cost)
    cost_field = (
        "runner summary field 'steps'"
        if args.cost == "steps"
        else "runner summary field 'token_usage.total_tokens'"
    )
    audit.update(
        {
            "runs_config": str(args.runs_config),
            "cost": cost_field,
            "cost_kind": args.cost,
            "max_budget": max_budget,
            "cohort_size": cohort_size,
            "cts_definition": f"sum(task_pass * I[{args.cost} <= budget])",
            "ccs_definition": f"sum(correctness_score * I[{args.cost} <= budget])",
            "normalization": "none",
            "token_filter_exclusions": token_drop,
            "outputs": {
                "task_records": str(records_path),
                "curves": str(curves_path),
                "figures": [
                    str(figure_stem.with_suffix(ext)) for ext in (".svg", ".pdf", ".png")
                ],
            },
        }
    )
    metadata_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    figure_paths = plot_curves(
        figure_stem, curves, cohort_size, max_budget, args.cost, model_colors
    )

    print(f"paired before runtime filter: {audit['cohort_size_before_runtime_filter']}")
    print(f"paired valid cohort: {cohort_size}")
    print(f"excluded runtime/system failures: {audit['excluded_task_count']}")
    if token_drop:
        print(f"excluded missing/unbalanced token usage: {len(token_drop)}")
    print(f"task records: {records_path}")
    print(f"curve data: {curves_path}")
    print(f"metadata: {metadata_path}")
    print("figures: " + ", ".join(str(path) for path in figure_paths))


if __name__ == "__main__":
    main()
