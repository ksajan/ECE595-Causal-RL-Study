#!/usr/bin/env python3
"""Plot seed-level CTRL revision results from reproduction JSON artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.revision.result_validation import load_and_validate
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from result_validation import load_and_validate

CONDITIONS = ("real", "fresh_noise", "oracle_cf", "learned_cf")
CONDITION_LABELS = {
    "real": "Real",
    "fresh_noise": "Fresh-noise",
    "oracle_cf": "Oracle-CF",
    "learned_cf": "Learned-CF",
}
METRICS = ("clean", "noisy")
REQUIRED_PATH = "bicogan.counterfactual_diagnostics.external_validation.normalized_mse"


def _number(value: Any, description: str) -> float:
    """Convert a JSON scalar to a finite float with an informative error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be numeric") from error
    if not np.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _read_path(payload: dict[str, Any], path: str, source: Path) -> Any:
    """Read a dotted path from an artifact, reporting the source on failure."""
    current: Any = payload
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"{source}: missing field {path}")
        current = current[key]
    return current


def load_artifacts(
    input_dir: Path,
    *,
    expected_seeds: list[int] | None = None,
    expected_count: int | None = None,
    development: bool = False,
) -> list[dict[str, Any]]:
    """Load only artifacts accepted by the shared integrity validator."""
    runs, _ = load_and_validate(
        input_dir,
        expected_seeds=expected_seeds,
        expected_count=expected_count,
        development=development,
    )
    return runs


def _values(runs: list[dict[str, Any]], condition: str, metric: str) -> np.ndarray:
    """Return seed-level evaluation means for one condition and metric."""
    return np.asarray(
        [float(run[condition][metric]["mean"]) for run in runs], dtype=np.float64
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save a figure as tightly bounded PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_dir / f"{stem}{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_returns(runs: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot paired clean and noisy returns for all four evaluation conditions."""
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for axis, metric in zip(axes, METRICS):
        for seed_index, run in enumerate(runs):
            axis.plot(
                range(len(CONDITIONS)),
                [_values([run], condition, metric)[0] for condition in CONDITIONS],
                color="#777777",
                alpha=0.35,
                linewidth=0.8,
                marker="o",
                markersize=3,
            )
        means = [_values(runs, condition, metric) for condition in CONDITIONS]
        for index, values in enumerate(means):
            axis.scatter(
                np.full(len(values), index),
                values,
                color=colors[index],
                s=22,
                alpha=0.85,
                label=CONDITION_LABELS[CONDITIONS[index]],
            )
        axis.set_xticks(range(4), [CONDITION_LABELS[c] for c in CONDITIONS])
        axis.set_title(f"{metric.capitalize()} evaluation")
        axis.set_xlabel("Training condition")
        axis.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Return")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Seed-level CTRL revision returns", y=1.02)
    fig.tight_layout()
    _save(fig, output_dir, "ctrl_revision_seed_returns")


def plot_deltas(runs: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot paired learned-CF minus real return deltas by seed."""
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(runs))
    for offset, metric in enumerate(METRICS):
        delta = _values(runs, "learned_cf", metric) - _values(runs, "real", metric)
        axis.plot(x + (offset - 0.5) * 0.12, delta, "o-", label=metric.capitalize())
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_xlabel("Seed")
    axis.set_ylabel("Learned-CF minus real return")
    axis.set_xticks(x, [str(run["config"]["seed"]) for run in runs])
    axis.legend(frameon=False)
    axis.set_title("Paired learned-CF return deltas")
    fig.tight_layout()
    _save(fig, output_dir, "ctrl_revision_learned_cf_deltas")


def plot_mse_relationship(runs: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot held-out learned-CF normalized MSE against noisy return delta."""
    mse = np.asarray(
        [
            _number(_read_path(run, REQUIRED_PATH, Path(run["_path"])), REQUIRED_PATH)
            for run in runs
        ]
    )
    delta = _values(runs, "learned_cf", "noisy") - _values(runs, "real", "noisy")
    fig, axis = plt.subplots(figsize=(6.2, 4.6))
    axis.scatter(mse, delta, s=42, color="#0072B2", edgecolor="white", linewidth=0.6)
    for x_value, y_value, run in zip(mse, delta, runs):
        axis.annotate(
            str(run["config"]["seed"]),
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_xlabel("Held-out learned-CF normalized MSE")
    axis.set_ylabel("Noisy learned-CF minus real return")
    axis.set_title("Counterfactual fidelity and return improvement")
    fig.tight_layout()
    _save(fig, output_dir, "ctrl_revision_mse_vs_return_delta")


def main() -> None:
    """Parse command-line arguments and generate all requested figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir", type=Path, help="Directory containing JSON artifacts"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/revision/figures")
    )
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-seeds", type=int, nargs="+")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    if (
        not args.development
        and args.expected_count is None
        and args.expected_seeds is None
    ):
        parser.error("confirmatory plots require --expected-count or --expected-seeds")
    runs = load_artifacts(
        args.input_dir,
        expected_seeds=args.expected_seeds,
        expected_count=args.expected_count,
        development=args.development,
    )
    plot_returns(runs, args.output_dir)
    plot_deltas(runs, args.output_dir)
    plot_mse_relationship(runs, args.output_dir)
    print(f"Wrote three figures in PNG and PDF formats to {args.output_dir}")


if __name__ == "__main__":
    main()
