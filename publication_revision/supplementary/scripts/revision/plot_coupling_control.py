"""Plot the frozen five-arm CartPole coupling-control summary.

Only training-seed aggregates are plotted.  Evaluation episodes remain nested
observations and are never expanded into additional plotting observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SCHEMA = "ctrl-cartpole-coupling-control-summary-v1"
EXPECTED_SEEDS = tuple(range(1030, 1060))
ARMS = ("random", "real", "fresh_independent", "fresh_shared", "oracle_cf")
ARM_LABELS = {
    "random": "Random",
    "real": "Real-only",
    "fresh_independent": "Fresh-independent\nsynthetic",
    "fresh_shared": "Fresh-shared\nsynthetic",
    "oracle_cf": "Oracle CF",
}
ARM_COLORS = {
    "random": "#6B7280",
    "real": "#0072B2",
    "fresh_independent": "#D55E00",
    "fresh_shared": "#CC79A7",
    "oracle_cf": "#009E73",
}
METRICS = ("clean", "noisy")
METRIC_LABELS = {"clean": "Clean evaluation", "noisy": "Process-noise evaluation"}
METRIC_THRESHOLDS = {"clean": 25.0, "noisy": 5.0}
CONTRASTS = (
    (
        "oracle_cf_minus_fresh_shared_clean",
        "Oracle CF - fresh-shared synthetic",
        "clean",
    ),
    (
        "oracle_cf_minus_fresh_shared_noisy",
        "Oracle CF - fresh-shared synthetic",
        "noisy",
    ),
    (
        "fresh_shared_minus_fresh_independent_clean",
        "Fresh-shared - fresh-independent synthetic",
        "clean",
    ),
    (
        "fresh_shared_minus_fresh_independent_noisy",
        "Fresh-shared - fresh-independent synthetic",
        "noisy",
    ),
)
EXPECTED_CONTRAST_KEYS = tuple(item[0] for item in CONTRASTS)


def read_summary(path: Path) -> dict[str, Any]:
    """Read one JSON summary and require a top-level object."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read summary {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: summary must be a JSON object")
    return payload


def _finite_seed_values(value: Any, label: str) -> np.ndarray:
    """Convert exactly one finite value per registered training seed."""
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if values.shape != (len(EXPECTED_SEEDS),):
        raise ValueError(f"{label} must contain exactly 30 seed values")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    return values


def _finite_scalar(value: Any, label: str) -> float:
    """Convert one finite scalar used by an interval or effect summary."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def validate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Validate the frozen summary schema, arms, and exact 30-seed contract."""
    if summary.get("artifact_schema") != SCHEMA:
        raise ValueError(f"Expected summary schema {SCHEMA}")
    if tuple(summary.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("Summary must contain exactly seeds 1030 through 1059")
    if tuple(summary.get("registered_contrasts", ())) != EXPECTED_CONTRAST_KEYS:
        raise ValueError("Summary contrasts do not match the four registered effects")
    if summary.get("registered_primary") != "oracle_cf_minus_fresh_shared_clean":
        raise ValueError("Summary primary contrast is not the registered clean effect")
    if summary.get("interpretation", {}).get("unit_of_replication") != (
        "training seed; evaluation episodes are nested observations"
    ):
        raise ValueError(
            "Summary does not declare training seeds as the replication unit"
        )

    arms = summary.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("Summary must contain exactly the five registered arms")
    for arm in ARMS:
        for metric in METRICS:
            result = arms[arm].get(metric)
            if not isinstance(result, dict):
                raise TypeError(f"Missing arm/metric summary: {arm}/{metric}")
            _finite_seed_values(
                result.get("seed_means"), f"arms.{arm}.{metric}.seed_means"
            )
            for bound in (
                "mean",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
            ):
                _finite_scalar(result.get(bound), f"arms.{arm}.{metric}.{bound}")

    contrasts = summary.get("contrasts")
    if not isinstance(contrasts, dict) or set(contrasts) != set(EXPECTED_CONTRAST_KEYS):
        raise ValueError("Summary must contain exactly the four registered contrasts")
    for key in EXPECTED_CONTRAST_KEYS:
        effect = contrasts[key]
        if not isinstance(effect, dict):
            raise TypeError(f"Contrast {key} must be an object")
        _finite_seed_values(effect.get("seed_deltas"), f"contrasts.{key}.seed_deltas")
        for bound in (
            "mean_delta",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
        ):
            _finite_scalar(effect.get(bound), f"contrasts.{key}.{bound}")
    return summary


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save one vector PDF and one 300-dpi PNG, then close the figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _style() -> None:
    """Set compact, color-vision-readable publication defaults."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_seed_distributions(summary: dict[str, Any], output_dir: Path) -> None:
    """Plot the five seed-level arm distributions in clean/noisy panels."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=False)
    rng = np.random.default_rng(7)
    for axis, metric in zip(axes, METRICS):
        for index, arm in enumerate(ARMS):
            result = summary["arms"][arm][metric]
            values = np.asarray(result["seed_means"], dtype=np.float64)
            jitter = rng.uniform(-0.075, 0.075, len(values))
            axis.scatter(
                index + jitter,
                values,
                s=17,
                alpha=0.62,
                color=ARM_COLORS[arm],
                edgecolors="white",
                linewidths=0.3,
                zorder=2,
            )
            mean = float(result["mean"])
            low = float(result["bootstrap_ci95_low"])
            high = float(result["bootstrap_ci95_high"])
            axis.errorbar(
                index,
                mean,
                yerr=[[mean - low], [high - mean]],
                color="#111827",
                marker="D",
                markersize=4.5,
                capsize=3,
                linewidth=1.1,
                zorder=3,
            )
        axis.set_xticks(
            range(len(ARMS)),
            [ARM_LABELS[arm] for arm in ARMS],
            rotation=28,
            ha="right",
        )
        axis.set_xlabel("Training condition")
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlim(-0.5, len(ARMS) - 0.5)
    axes[0].set_ylabel("Return (seed-level evaluation mean)")
    fig.suptitle("Five-arm return distributions", y=1.01)
    fig.text(
        0.5,
        -0.02,
        "Dots: one training seed; diamonds and bars: mean with 95% bootstrap interval",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, output_dir, "five_arm_seed_distributions_clean_process_noise")


def plot_primary_delta(summary: dict[str, Any], output_dir: Path) -> None:
    """Plot paired oracle-minus-fresh-shared clean seed-level differences."""
    effect = summary["contrasts"]["oracle_cf_minus_fresh_shared_clean"]
    deltas = np.asarray(effect["seed_deltas"], dtype=np.float64)
    x = np.arange(len(deltas))
    mean = float(effect["mean_delta"])
    low = float(effect["bootstrap_ci95_low"])
    high = float(effect["bootstrap_ci95_high"])
    fig, axis = plt.subplots(figsize=(7.8, 4.0))
    axis.axhspan(-25, 25, color="#9CA3AF", alpha=0.13, zorder=0)
    axis.axhline(0, color="#111827", linewidth=1.0, label="No difference")
    axis.axhline(25, color="#6B7280", linestyle="--", linewidth=0.9)
    axis.axhline(-25, color="#6B7280", linestyle="--", linewidth=0.9)
    axis.scatter(x, deltas, color=ARM_COLORS["oracle_cf"], s=23, alpha=0.82, zorder=2)
    axis.axhline(
        mean,
        color=ARM_COLORS["fresh_independent"],
        linewidth=1.8,
        label="Mean difference",
    )
    axis.axhspan(
        low,
        high,
        color=ARM_COLORS["fresh_independent"],
        alpha=0.17,
        label="95% bootstrap CI",
    )
    axis.set_xlabel("Training seed (30 paired runs)")
    axis.set_ylabel("Oracle CF - fresh-shared synthetic return")
    axis.set_title("Paired clean-return effect of factual noise reuse")
    axis.set_xticks(
        x[::2], [str(seed) for seed in EXPECTED_SEEDS[::2]], rotation=45, ha="right"
    )
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    axis.text(
        0.995,
        0.02,
        "Dashed lines: +/-25 practical-effect thresholds",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#4B5563",
    )
    fig.tight_layout()
    _save(fig, output_dir, "oracle_minus_fresh_shared_clean_deltas")


def plot_effect_forest(summary: dict[str, Any], output_dir: Path) -> None:
    """Plot all four registered paired effects in separate clean/noisy panels."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=False)
    rows_by_metric = {
        metric: [item for item in CONTRASTS if item[2] == metric] for metric in METRICS
    }
    for axis, metric in zip(axes, METRICS):
        rows = rows_by_metric[metric]
        y = np.arange(len(rows))
        for row, (key, label, _) in enumerate(rows):
            effect = summary["contrasts"][key]
            mean = float(effect["mean_delta"])
            low = float(effect["bootstrap_ci95_low"])
            high = float(effect["bootstrap_ci95_high"])
            color = (
                ARM_COLORS["oracle_cf"]
                if "oracle_cf" in key
                else ARM_COLORS["fresh_shared"]
            )
            axis.plot(
                [low, high],
                [row, row],
                color=color,
                linewidth=2.5,
                solid_capstyle="round",
            )
            axis.scatter(
                mean,
                row,
                color=color,
                edgecolor="#111827",
                linewidth=0.5,
                s=45,
                zorder=3,
            )
        threshold = METRIC_THRESHOLDS[metric]
        axis.axvspan(-threshold, threshold, color="#9CA3AF", alpha=0.12, zorder=0)
        axis.axvline(0, color="#111827", linewidth=1.0)
        axis.axvline(threshold, color="#6B7280", linestyle="--", linewidth=0.8)
        axis.axvline(-threshold, color="#6B7280", linestyle="--", linewidth=0.8)
        axis.set_yticks(y, [label for _, label, _ in rows])
        axis.set_xlabel(f"Difference (+/-{threshold:g} threshold)")
        axis.set_title(METRIC_LABELS[metric])
        axis.invert_yaxis()
    fig.suptitle("Registered paired effects with 95% bootstrap intervals", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Intervals and points summarize 30 paired training seeds",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, output_dir, "registered_paired_effects_forest_clean_process_noise")


def main() -> None:
    """Validate the frozen summary and write three PDF/PNG figure pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/revision/coupling_control_final/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/coupling_control_final/figures"),
    )
    args = parser.parse_args()
    summary = validate_summary(read_summary(args.summary))
    _style()
    plot_seed_distributions(summary, args.output_dir)
    plot_primary_delta(summary, args.output_dir)
    plot_effect_forest(summary, args.output_dir)
    print(f"Wrote six files to {args.output_dir}")


if __name__ == "__main__":
    main()
