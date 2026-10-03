#!/usr/bin/env python3
"""Share of each model's official tasks with score >= t, t = 0.1 ... 1.0."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.analysis_lib.plot_cost_quality_curves import (
    assign_colors,
    build_model_cohort,
    load_model_runs,
)

THRESHOLDS = [i / 10 for i in range(1, 11)]


def threshold_rows(observations, models: tuple[str, ...]) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for model in models:
        scores = [item.score for item in observations if item.model == model]
        n = len(scores)
        for threshold in THRESHOLDS:
            hit = sum(score + 1e-12 >= threshold for score in scores)
            rows.append(
                {
                    "model": model,
                    "threshold": threshold,
                    "rate": hit / n if n else 0.0,
                    "n": n,
                }
            )
    return rows


def plot_thresholds(
    output_stem: Path,
    rows: list[dict[str, str | float | int]],
    model_colors: dict[str, str],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, axis = plt.subplots(figsize=(5.2, 4.25))
    for model, color in model_colors.items():
        model_rows = [row for row in rows if row["model"] == model]
        axis.plot(
            [float(row["threshold"]) for row in model_rows],
            [float(row["rate"]) for row in model_rows],
            label=model,
            color=color,
            linewidth=2.1,
            marker="o",
            markersize=3.3,
        )
    axis.set_xlim(0.1, 1.0)
    axis.set_ylim(0, 1)
    axis.set_xticks(THRESHOLDS)
    axis.set_yticks([i / 5 for i in range(6)])
    axis.set_xlabel("Score threshold")
    axis.set_ylabel("Proportion of tasks")
    axis.grid(axis="both", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    fig.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.28)
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.55, 0.12),
        handlelength=2.0,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    # Avoid tight_layout resetting the legend room below the axes.

    outputs: list[Path] = []
    for suffix, options in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 300})):
        path = output_stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **options)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the share of official tasks with score >= t."
    )
    parser.add_argument("--runs-config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("logs/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--label", default="common")
    args = parser.parse_args()

    model_runs = load_model_runs(args.runs_config)
    models = tuple(model_runs)
    observations, _audit = build_model_cohort(args.runs_root, model_runs)
    rows = threshold_rows(observations, models)

    aggregate_dir = args.output_dir / "aggregates"
    figure_dir = args.output_dir / "figures"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = aggregate_dir / f"{args.label}_score_thresholds.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "threshold", "rate", "n"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "threshold": f"{float(row['threshold']):.1f}",
                    "rate": f"{float(row['rate']):.6f}",
                    "n": row["n"],
                }
            )
    figure_paths = plot_thresholds(
        figure_dir / f"{args.label}_score_thresholds",
        rows,
        assign_colors(model_runs),
    )
    print(f"table: {csv_path}")
    print("figures: " + ", ".join(str(path) for path in figure_paths))


if __name__ == "__main__":
    main()
