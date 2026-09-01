"""Render continuous-control Markdown and LaTeX tables after evidence gates pass."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.workshop.plot_continuous_control_results import (
    DISPLAY,
    EXPECTED_CELLS,
    load_paired_summary,
    validate_publication_cells,
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    """Load a required non-empty CSV file."""

    if not path.is_file():
        raise FileNotFoundError(f"Summary does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Summary is empty: {path}")
    return rows


def validate_aggregate_cells(
    rows: Sequence[Mapping[str, str]], min_seeds: int
) -> dict[tuple[str, str, str], Mapping[str, str]]:
    """Require one sufficiently replicated absolute-score row per planned arm."""

    validated: dict[tuple[str, str, str], Mapping[str, str]] = {}
    problems: list[str] = []
    for domain, spec in EXPECTED_CELLS.items():
        metric = str(spec["metric"])
        variants = ("real", *spec["variants"])
        for task in spec["tasks"]:
            for variant in variants:
                matches = [
                    row
                    for row in rows
                    if row.get("domain") == domain
                    and row.get("task") == task
                    and row.get("variant") == variant
                    and row.get("metric") == metric
                ]
                label = f"{domain}/{task}/{variant}/{metric}"
                if len(matches) != 1:
                    problems.append(f"{label}: expected one row, found {len(matches)}")
                    continue
                n = int(matches[0]["n"])
                if n < min_seeds:
                    problems.append(f"{label}: n={n}, requires n>={min_seeds}")
                    continue
                try:
                    seeds = [int(seed) for seed in json.loads(matches[0]["seeds"])]
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    problems.append(f"{label}: invalid seeds field")
                    continue
                if len(seeds) != n or len(set(seeds)) != n:
                    problems.append(f"{label}: seed identities do not match n={n}")
                    continue
                if not set(range(min_seeds)).issubset(seeds):
                    problems.append(f"{label}: missing required seeds 0--{min_seeds - 1}")
                    continue
                validated[(domain, task, variant)] = matches[0]
    if problems:
        detail = "\n".join(f"- {problem}" for problem in problems)
        raise RuntimeError(f"Absolute-score report gate failed:\n{detail}")
    return validated


def _fmt(value: str | float) -> str:
    """Format a finite result compactly without hiding its sign."""

    return f"{float(value):.2f}"


def _effect_label(row: Mapping[str, str]) -> str:
    """Classify direction from the paired bootstrap interval only."""

    low = float(row["bootstrap_ci95_low"])
    high = float(row["bootstrap_ci95_high"])
    if low > 0.0:
        return "positive interval"
    if high < 0.0:
        return "negative interval"
    return "interval includes zero"


def _paired_lookup(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str, str], Mapping[str, str]]:
    """Index validated paired rows by domain, task, and variant."""

    return {
        (row["domain"], row["task"], row["variant"]): row
        for row in rows
        if row["domain"] in EXPECTED_CELLS
        and row["task"] in EXPECTED_CELLS[row["domain"]]["tasks"]
        and row["variant"] in EXPECTED_CELLS[row["domain"]]["variants"]
        and row["metric"] == EXPECTED_CELLS[row["domain"]]["metric"]
    }


def render_markdown(
    aggregate: Mapping[tuple[str, str, str], Mapping[str, str]],
    paired: Mapping[tuple[str, str, str], Mapping[str, str]],
) -> str:
    """Render an auditable report without automatic superiority claims."""

    lines = [
        "# Continuous-Control Validation Results",
        "",
        (
            "All rows use ten paired learner seeds and fixed evaluation seed banks. "
            "Intervals are paired 95% bootstrap intervals. Holm values adjust each "
            "test across the planned task-by-variant family within a domain."
        ),
        "",
    ]
    for domain, heading in (
        ("mujoco_sac", "Online MuJoCo SAC"),
        ("d4rl_cql", "Offline D4RL CQL"),
    ):
        spec = EXPECTED_CELLS[domain]
        score_name = "return" if domain == "mujoco_sac" else "normalized score"
        lines.extend(
            [
                f"## {heading}",
                "",
                (
                    f"| Task | Arm | n | Absolute {score_name} (mean +/- SD) | "
                    "Paired delta vs real [95% CI] | +/-/ties | Holm p (t/W/R) |"
                ),
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for task in spec["tasks"]:
            real = aggregate[(domain, task, "real")]
            lines.append(
                f"| {DISPLAY[task]} | Real | {real['n']} | "
                f"{_fmt(real['mean'])} +/- {_fmt(real['std'])} | -- | -- | -- |"
            )
            for variant in spec["variants"]:
                arm = aggregate[(domain, task, variant)]
                effect = paired[(domain, task, variant)]
                interval = (
                    f"{_fmt(effect['mean'])} "
                    f"[{_fmt(effect['bootstrap_ci95_low'])}, "
                    f"{_fmt(effect['bootstrap_ci95_high'])}]"
                )
                signs = (
                    f"{effect['positive_seeds']}/{effect['negative_seeds']}/"
                    f"{effect['ties']}"
                )
                pvalues = "/".join(
                    _fmt(effect[field])
                    for field in (
                        "paired_t_p_holm",
                        "wilcoxon_p_holm",
                        "sign_randomization_p_holm",
                    )
                )
                lines.append(
                    f"| {DISPLAY[task]} | {DISPLAY[variant]} | {arm['n']} | "
                    f"{_fmt(arm['mean'])} +/- {_fmt(arm['std'])} | {interval} | "
                    f"{signs} | {pvalues} |"
                )
        labels = [
            f"{DISPLAY[task]} {DISPLAY[variant]}: {_effect_label(paired[(domain, task, variant)])}"
            for task in spec["tasks"]
            for variant in spec["variants"]
        ]
        lines.extend(["", "Interval diagnostics: " + "; ".join(labels) + ".", ""])
    lines.extend(
        [
            (
                "These experiments isolate simulator-based one-step augmentation. "
                "They are not a learned-BiCoGAN CTRL reproduction, and a confidence "
                "interval that excludes zero is not by itself evidence of practical "
                "importance or generality."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_latex(
    aggregate: Mapping[tuple[str, str, str], Mapping[str, str]],
    paired: Mapping[tuple[str, str, str], Mapping[str, str]],
) -> str:
    """Render compact LaTeX tables for optional appendix inclusion."""

    lines = ["% Generated only after the ten-paired-seed publication gate passed."]
    for domain, caption, label in (
        ("mujoco_sac", "Online SAC paired effects", "tab:sac_paired"),
        ("d4rl_cql", "Offline CQL paired effects", "tab:d4rl_paired"),
    ):
        spec = EXPECTED_CELLS[domain]
        lines.extend(
            [
                "\\begin{table}[t]",
                "\\centering",
                "\\small",
                (
                    f"\\caption{{{caption}; mean difference from real-only with "
                    "95\\% paired bootstrap interval.}"
                ),
                f"\\label{{{label}}}",
                "\\begin{tabular}{llrr}",
                "\\toprule",
                "Task & Arm & Absolute mean & $\\Delta$ [95\\% CI] \\\\",
                "\\midrule",
            ]
        )
        for task in spec["tasks"]:
            real = aggregate[(domain, task, "real")]
            task_name = DISPLAY[task].replace("_", "\\_")
            lines.append(f"{task_name} & Real & {_fmt(real['mean'])} & -- \\\\")
            for variant in spec["variants"]:
                arm = aggregate[(domain, task, variant)]
                effect = paired[(domain, task, variant)]
                lines.append(
                    f"{task_name} & {DISPLAY[variant]} & {_fmt(arm['mean'])} & "
                    f"{_fmt(effect['mean'])} "
                    f"[{_fmt(effect['bootstrap_ci95_low'])}, "
                    f"{_fmt(effect['bootstrap_ci95_high'])}] \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_report(summary_dir: Path, output_dir: Path, min_seeds: int = 10) -> None:
    """Validate and write both reviewer-readable and manuscript-ready reports."""

    paired_rows = load_paired_summary(summary_dir / "paired_summary.csv")
    validate_publication_cells(paired_rows, min_seeds)
    aggregate = validate_aggregate_cells(
        _load_csv(summary_dir / "aggregate_results.csv"), min_seeds
    )
    paired = _paired_lookup(paired_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "continuous_control_results.md").write_text(
        render_markdown(aggregate, paired), encoding="utf-8"
    )
    (output_dir / "continuous_control_tables.tex").write_text(
        render_latex(aggregate, paired), encoding="utf-8"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse summary input, output directory, and minimum paired seed count."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-seeds", type=int, default=10)
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point with a concise failed-gate message."""

    args = parse_args()
    try:
        render_report(args.summary_dir, args.output_dir, args.min_seeds)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    print(f"[report] wrote publication tables to {args.output_dir}")


if __name__ == "__main__":
    main()
