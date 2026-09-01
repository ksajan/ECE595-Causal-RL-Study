#!/usr/bin/env python3
"""Plot publication-ready paired effects after enforcing seed-count gates."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

EXPECTED_CELLS = {
    "mujoco_sac": {
        "tasks": ("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4"),
        "variants": ("duplicate", "oracle_cf"),
        "metric": "return",
    },
    "d4rl_cql": {
        "tasks": ("halfcheetah-medium-v2", "hopper-medium-v2"),
        "variants": ("simulator_mean", "fresh_residual", "factual_residual"),
        "metric": "normalized_d4rl_score",
    },
}

DISPLAY = {
    "HalfCheetah-v4": "HalfCheetah",
    "Hopper-v4": "Hopper",
    "Walker2d-v4": "Walker2d",
    "Ant-v4": "Ant",
    "halfcheetah-medium-v2": "HalfCheetah-medium",
    "hopper-medium-v2": "Hopper-medium",
    "duplicate": "Duplicate control",
    "oracle_cf": "Oracle counterfactual",
    "simulator_mean": "Simulator mean",
    "fresh_residual": "Fresh residual",
    "factual_residual": "Factual residual",
}

COLORS = {
    "duplicate": "#6B7280",
    "oracle_cf": "#0072B2",
    "simulator_mean": "#6B7280",
    "fresh_residual": "#D55E00",
    "factual_residual": "#009E73",
}


def load_paired_summary(path: Path) -> list[dict[str, str]]:
    """Load non-empty paired-summary rows from CSV."""

    if not path.is_file():
        raise FileNotFoundError(f"Paired summary does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Paired summary is empty: {path}")
    return rows


def validate_publication_cells(
    rows: Sequence[Mapping[str, str]], min_seeds: int
) -> dict[str, list[dict[str, float | str]]]:
    """Return expected cells or reject missing, duplicate, and underpowered rows."""

    if min_seeds < 2:
        raise ValueError("min_seeds must be at least 2")
    validated: dict[str, list[dict[str, float | str]]] = {}
    problems: list[str] = []
    for domain, spec in EXPECTED_CELLS.items():
        domain_rows: list[dict[str, float | str]] = []
        for task in spec["tasks"]:
            for variant in spec["variants"]:
                matches = [
                    row
                    for row in rows
                    if row.get("domain") == domain
                    and row.get("task") == task
                    and row.get("variant") == variant
                    and row.get("metric") == spec["metric"]
                ]
                label = f"{domain}/{task}/{variant}/{spec['metric']}"
                if len(matches) != 1:
                    problems.append(f"{label}: expected one row, found {len(matches)}")
                    continue
                row = matches[0]
                n = int(row["n"])
                if n < min_seeds:
                    problems.append(f"{label}: n={n}, requires n>={min_seeds}")
                    continue
                try:
                    paired_seeds = [int(seed) for seed in json.loads(row["paired_seeds"])]
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    problems.append(f"{label}: invalid paired_seeds field")
                    continue
                if len(paired_seeds) != n or len(set(paired_seeds)) != n:
                    problems.append(f"{label}: paired seed identities do not match n={n}")
                    continue
                expected_seeds = set(range(min_seeds))
                if not expected_seeds.issubset(paired_seeds):
                    problems.append(
                        f"{label}: missing required seeds 0--{min_seeds - 1}"
                    )
                    continue
                low = float(row["bootstrap_ci95_low"])
                mean = float(row["mean"])
                high = float(row["bootstrap_ci95_high"])
                if not low <= mean <= high:
                    problems.append(f"{label}: confidence interval does not contain mean")
                    continue
                domain_rows.append(
                    {
                        "task": task,
                        "variant": variant,
                        "n": n,
                        "paired_seeds": paired_seeds,
                        "mean": mean,
                        "low": low,
                        "high": high,
                    }
                )
        validated[domain] = domain_rows
    if problems:
        detail = "\n".join(f"- {problem}" for problem in problems)
        raise RuntimeError(f"Publication figure gate failed:\n{detail}")
    return validated


def plot_paired_effects(
    validated: Mapping[str, Sequence[Mapping[str, float | str]]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Render paired mean deltas and 95% bootstrap intervals as a two-panel plot."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    panels = (
        ("mujoco_sac", "Online SAC", "Return difference vs. real replay"),
        ("d4rl_cql", "Offline CQL", "Normalized D4RL score difference vs. real data"),
    )
    for axis, (domain, title, xlabel) in zip(axes, panels, strict=True):
        spec = EXPECTED_CELLS[domain]
        rows = validated[domain]
        offsets = {
            variant: (index - (len(spec["variants"]) - 1) / 2) * 0.18
            for index, variant in enumerate(spec["variants"])
        }
        for variant in spec["variants"]:
            selected = {str(row["task"]): row for row in rows if row["variant"] == variant}
            ys = [index + offsets[variant] for index in range(len(spec["tasks"]))]
            means = [float(selected[task]["mean"]) for task in spec["tasks"]]
            lower = [
                mean - float(selected[task]["low"])
                for task, mean in zip(spec["tasks"], means, strict=True)
            ]
            upper = [
                float(selected[task]["high"]) - mean
                for task, mean in zip(spec["tasks"], means, strict=True)
            ]
            axis.errorbar(
                means,
                ys,
                xerr=[lower, upper],
                fmt="o",
                capsize=3,
                color=COLORS[variant],
                label=DISPLAY[variant],
            )
        axis.axvline(0.0, color="#111827", linewidth=1, linestyle="--")
        axis.set_yticks(range(len(spec["tasks"])), [DISPLAY[task] for task in spec["tasks"]])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color="#D1D5DB", linewidth=0.6)
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle("Paired continuous-control effects with 95% bootstrap intervals")
    png_path = output_dir / "continuous_control_paired_effects.png"
    pdf_path = output_dir / "continuous_control_paired_effects.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse summary input, output directory, and publication seed gate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-seeds", type=int, default=10)
    return parser.parse_args(argv)


def main() -> None:
    """Validate complete paired evidence before writing plots."""

    args = parse_args()
    rows = load_paired_summary(args.summary_dir / "paired_summary.csv")
    try:
        validated = validate_publication_cells(rows, args.min_seeds)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    png_path, pdf_path = plot_paired_effects(validated, args.output_dir)
    print(f"[plot] wrote {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
