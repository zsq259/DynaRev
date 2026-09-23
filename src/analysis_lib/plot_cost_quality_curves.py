#!/usr/bin/env python3
"""CTS/CCS curves and CE on each model's official tasks.

Official = correctness_score present and not exclude_from_official.
no_final_submit stays at score 0. Environment errors are skipped.
Models are not intersected; env failures are expected to be rerun later.

--runs-config maps model -> subset -> run directory name under --runs-root.

Outputs (--label prefix, suffix depends on --cost steps|tokens):
  - {label}_task_records_{cost}.csv
  - {label}_cost_curves_{cost}.csv
  - {label}_ce_{cost}.json
  - {label}_cost_curves_{cost}.{svg,pdf,png}
  - {label}_cost_curves_{cost}.metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"runs config must be a non-empty JSON object: {config_path}")
    model_runs: ModelRuns = {}
    for model, subset_runs in value.items():
        if not isinstance(subset_runs, dict) or not subset_runs:
            raise ValueError(f"runs config entry for {model!r} must map subset -> run name")
        parsed: dict[str, str] = {}
        for subset, run_name in subset_runs.items():
            if not isinstance(run_name, str) or not run_name:
                raise ValueError(f"invalid run name for {model!r}/{subset!r}")
            parsed[str(subset)] = run_name
        model_runs[str(model)] = parsed
    return model_runs


def assign_colors(model_runs: ModelRuns) -> dict[str, str]:
    return {
        model: COLOR_PALETTE[index % len(COLOR_PALETTE)]
        for index, model in enumerate(model_runs)
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_token_total(summary: dict[str, Any]) -> int | None:
    usage = summary.get("token_usage")
    if not isinstance(usage, dict) or usage.get("total_tokens") is None:
        return None
    return int(usage["total_tokens"])


def load_run(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    summaries: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*/summary.json")):
        summary = read_json(path)
        summaries[str(summary.get("instance_id") or path.parent.name)] = summary

    evaluations: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "evaluations").glob("*.evaluation.json")):
        evaluation = read_json(path)
        result = evaluation.get("result")
        if (
            isinstance(result, dict)
            and result.get("correctness_score") is not None
            and not result.get("exclude_from_official")
        ):
            task_id = str(
                evaluation.get("task_id") or path.name.removesuffix(".evaluation.json")
            )
            evaluations[task_id] = result
    return summaries, evaluations


def build_observation(
    model: str,
    subset: str,
    task_id: str,
    run_name: str,
    summary: dict[str, Any],
    result: dict[str, Any],
) -> Observation:
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
        raise ValueError(
            f"is_correct mismatch for {subset}/{task_id}/{model}: "
            f"score={score} is_correct={result['is_correct']}"
        )
    return Observation(
        model=model,
        subset=subset,
        task_id=task_id,
        run_name=run_name,
        steps=steps,
        tokens=tokens,
        score=score,
        passed=passed,
        stopped_reason=str(summary.get("stopped_reason") or ""),
    )


def build_model_cohort(
    runs_root: Path,
    model_runs: ModelRuns,
) -> tuple[list[Observation], dict[str, Any]]:
    observations: list[Observation] = []
    inventory: dict[str, dict[str, dict[str, int | str]]] = {}
    for model, subset_runs in model_runs.items():
        inventory[model] = {}
        for subset, run_name in subset_runs.items():
            summaries, evaluations = load_run(runs_root / run_name)
            inventory[model][subset] = {
                "run_name": run_name,
                "summaries": len(summaries),
                "evaluations": len(evaluations),
            }
            for task_id, result in evaluations.items():
                summary = summaries.get(task_id)
                if summary is None:
                    raise FileNotFoundError(
                        f"missing summary.json for {model}/{subset}/{task_id}"
                    )
                observations.append(
                    build_observation(model, subset, task_id, run_name, summary, result)
                )
    return observations, {"alignment": "per_model", "run_inventory": inventory}


def count_by_model(observations: list[Observation], models: tuple[str, ...]) -> dict[str, int]:
    return {model: sum(item.model == model for item in observations) for model in models}


def compute_ce(
    observations: list[Observation],
    max_budget: int,
    cost_kind: str,
    models: tuple[str, ...],
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for model in models:
        records = [item for item in observations if item.model == model]
        n = len(records)
        ce = (
            sum(item.score * (1.0 - item.cost(cost_kind) / max_budget) for item in records) / n
            if n
            else 0.0
        )
        rows.append({"model": model, "n": n, "B": max_budget, "cost": cost_kind, "ce": ce})
    return rows


def compute_curves(
    observations: list[Observation],
    max_budget: int,
    cost_kind: str,
    models: tuple[str, ...],
) -> list[dict[str, int | float | str]]:
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
        budgets = [0, *observed]
        if max_budget not in budgets:
            budgets.append(max_budget)
    rows: list[dict[str, int | float | str]] = []
    for model in models:
        records = [item for item in observations if item.model == model]
        for budget in budgets:
            eligible = [item for item in records if item.cost(cost_kind) <= budget]
            rows.append(
                {
                    "model": model,
                    budget_key: budget,
                    "cts": sum(item.passed for item in eligible),
                    "ccs": sum(item.score for item in eligible),
                    "eligible_tasks": len(eligible),
                    "cohort_size": len(records),
                }
            )
    return rows


def filter_observations_for_cost(
    observations: list[Observation],
    cost_kind: str,
) -> tuple[list[Observation], list[dict[str, Any]]]:
    if cost_kind == "steps":
        return observations, []
    kept = [item for item in observations if item.tokens is not None]
    dropped = [
        {
            "subset": item.subset,
            "task_id": item.task_id,
            "model": item.model,
            "reason": "missing_summary_token_usage",
        }
        for item in observations
        if item.tokens is None
    ]
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
        writer.writerows(rows)


def plot_curves(
    output_stem: Path,
    rows: list[dict[str, int | float | str]],
    max_budget: int,
    cost_kind: str,
    model_colors: dict[str, str],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    budget_key = "action_budget" if cost_kind == "steps" else "token_budget"
    xlabel = "Action-step budget" if cost_kind == "steps" else "Token budget"
    cost_label = "action-step" if cost_kind == "steps" else "token"
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
        y_max = (
            max(1.0, math.ceil(observed_max + 1.0))
            if metric == "cts"
            else max(5.0, math.ceil((observed_max * 1.1) / 5.0) * 5.0)
        )
        for model in model_colors:
            model_rows = [row for row in rows if row["model"] == model]
            axis.plot(
                [int(row[budget_key]) for row in model_rows],
                [float(row[metric]) for row in model_rows],
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
    fig.suptitle(f"Cost–quality curves ({cost_label})", fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94), w_pad=2.5)

    outputs: list[Path] = []
    for suffix, options in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 300})):
        output_path = output_stem.with_suffix(suffix)
        fig.savefig(output_path, bbox_inches="tight", **options)
        outputs.append(output_path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CTS/CCS curves and CE on each model's official tasks."
    )
    parser.add_argument(
        "--runs-config",
        type=Path,
        required=True,
        help="JSON file mapping model name -> subset -> run directory name",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("logs/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--label", default="common", help="prefix for output filenames")
    parser.add_argument(
        "--cost",
        choices=("steps", "tokens"),
        default="steps",
        help="cost axis (default: steps)",
    )
    parser.add_argument(
        "--max-budget",
        type=int,
        default=None,
        help="curve/CE budget B; default 50 for steps, max observed tokens for tokens",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="write records/curves/CE without regenerating figures",
    )
    args = parser.parse_args()

    model_runs = load_model_runs(args.runs_config)
    model_colors = assign_colors(model_runs)
    models = tuple(model_runs.keys())

    observations, audit = build_model_cohort(args.runs_root, model_runs)
    observations, token_drop = filter_observations_for_cost(observations, args.cost)
    if args.cost == "tokens" and not observations:
        raise SystemExit(
            "no observations have summary.token_usage; re-run agents with the "
            "updated token meter before plotting token CTS/CCS"
        )

    if args.max_budget is None:
        max_budget = 50 if args.cost == "steps" else max(item.tokens or 0 for item in observations)
    else:
        max_budget = args.max_budget
    if max_budget <= 0:
        parser.error("--max-budget must be positive")

    n_by_model = count_by_model(observations, models)
    curves = compute_curves(observations, max_budget, args.cost, models)
    ce_rows = compute_ce(observations, max_budget, args.cost, models)

    aggregate_dir = args.output_dir / "aggregates"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.label}_cost_curves_{args.cost}"
    records_path = aggregate_dir / f"{args.label}_task_records_{args.cost}.csv"
    curves_path = aggregate_dir / f"{stem}.csv"
    ce_path = aggregate_dir / f"{args.label}_ce_{args.cost}.json"
    metadata_path = aggregate_dir / f"{stem}.metadata.json"

    write_records(records_path, observations)
    write_curves(curves_path, curves, args.cost)
    ce_path.write_text(json.dumps(ce_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    outputs = {
        "task_records": str(records_path),
        "curves": str(curves_path),
        "ce": str(ce_path),
        "figures": [],
    }
    figure_paths: list[Path] = []
    if not args.no_plot:
        figure_dir = args.output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        figure_paths = plot_curves(
            figure_dir / stem, curves, max_budget, args.cost, model_colors
        )
        outputs["figures"] = [str(path) for path in figure_paths]

    audit.update(
        {
            "models": list(models),
            "runs_config": str(args.runs_config),
            "cost_kind": args.cost,
            "max_budget": max_budget,
            "cohort_size_by_model": n_by_model,
            "ce": ce_rows,
            "ce_definition": "mean(correctness_score * (1 - cost / B))",
            "token_filter_exclusions": token_drop,
            "outputs": outputs,
        }
    )
    metadata_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for row in ce_rows:
        print(f"CE {row['model']}: n={row['n']} B={row['B']} ce={float(row['ce']):.4f}")
    if token_drop:
        print(f"excluded missing token usage: {len(token_drop)}")
    print(f"task records: {records_path}")
    print(f"curve data: {curves_path}")
    print(f"ce: {ce_path}")
    print(f"metadata: {metadata_path}")
    if figure_paths:
        print("figures: " + ", ".join(str(path) for path in figure_paths))


if __name__ == "__main__":
    main()
