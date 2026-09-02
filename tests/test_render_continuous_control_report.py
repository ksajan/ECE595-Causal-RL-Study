from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.workshop.plot_continuous_control_results import (
    EXPECTED_CELLS,
    EXPECTED_MATCHED_CONTROLS,
)
from scripts.workshop.render_continuous_control_report import render_report


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summaries(
    n: int = 10,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    aggregate: list[dict[str, str]] = []
    paired: list[dict[str, str]] = []
    matched: list[dict[str, str]] = []
    for domain, spec in EXPECTED_CELLS.items():
        for task in spec["tasks"]:
            for index, variant in enumerate(("real", *spec["variants"])):
                aggregate.append(
                    {
                        "domain": domain,
                        "task": task,
                        "variant": variant,
                        "metric": str(spec["metric"]),
                        "n": str(n),
                        "seeds": json.dumps(list(range(n))),
                        "mean": str(100 + index),
                        "std": "5.0",
                    }
                )
            for index, variant in enumerate(spec["variants"]):
                paired.append(
                    {
                        "domain": domain,
                        "task": task,
                        "variant": variant,
                        "metric": str(spec["metric"]),
                        "n": str(n),
                        "paired_seeds": json.dumps(list(range(n))),
                        "mean": str(index + 1),
                        "bootstrap_ci95_low": str(index + 0.5),
                        "bootstrap_ci95_high": str(index + 1.5),
                        "positive_seeds": str(n),
                        "negative_seeds": "0",
                        "ties": "0",
                        "paired_t_p_holm": "0.04",
                        "wilcoxon_p_holm": "0.05",
                        "sign_randomization_p_holm": "0.06",
                    }
                )
    for domain, spec in EXPECTED_MATCHED_CONTROLS.items():
        for task in spec["tasks"]:
            matched.append(
                {
                    "domain": domain,
                    "task": task,
                    "contrast": str(spec["contrast"]),
                    "metric": str(spec["metric"]),
                    "n": str(n),
                    "paired_seeds": json.dumps(list(range(n))),
                    "mean": "2.0",
                    "bootstrap_ci95_low": "1.0",
                    "bootstrap_ci95_high": "3.0",
                    "positive_seeds": str(n),
                    "negative_seeds": "0",
                    "ties": "0",
                    "paired_t_p_holm": "0.04",
                    "wilcoxon_p_holm": "0.05",
                    "sign_randomization_p_holm": "0.06",
                }
            )
    return aggregate, paired, matched


def test_report_writes_markdown_and_latex_after_complete_gate(tmp_path: Path) -> None:
    summary = tmp_path / "summary"
    output = tmp_path / "report"
    aggregate, paired, matched = _summaries()
    _write_csv(summary / "aggregate_results.csv", aggregate)
    _write_csv(summary / "paired_summary.csv", paired)
    _write_csv(summary / "matched_control_summary.csv", matched)

    render_report(summary, output)

    markdown = (output / "continuous_control_results.md").read_text()
    latex = (output / "continuous_control_tables.tex").read_text()
    assert "Online MuJoCo SAC" in markdown
    assert "Offline D4RL CQL" in markdown
    assert "10/0/0" in markdown
    assert "Matched augmentation controls" in markdown
    assert "intervention-based augmentation" in markdown
    assert "do not by themselves prove causal identification" in markdown
    assert "\\label{tab:sac_paired}" in latex
    assert "\\label{tab:matched_control_paired}" in latex
    assert "\\label{tab:d4rl_paired}" in latex
    assert "interval.}}" not in latex


def test_report_rejects_underpowered_absolute_scores(tmp_path: Path) -> None:
    summary = tmp_path / "summary"
    aggregate, paired, matched = _summaries()
    aggregate[0]["n"] = "9"
    _write_csv(summary / "aggregate_results.csv", aggregate)
    _write_csv(summary / "paired_summary.csv", paired)
    _write_csv(summary / "matched_control_summary.csv", matched)

    with pytest.raises(RuntimeError, match=r"requires n>=10"):
        render_report(summary, tmp_path / "report")

    assert not (tmp_path / "report").exists()
