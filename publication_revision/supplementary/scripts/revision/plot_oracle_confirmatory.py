"""Create publication figures from the audited oracle-CF summary artifacts.

The plots operate on training-seed aggregates.  They do not expand returns into
episode-level observations or imply episode-level replication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ORACLE_SCHEMA = "ctrl-cartpole-oracle-confirmatory-v1"
MODEL_SCHEMA = "ctrl-cartpole-model-gate-summary-v1"
ORACLE_SEEDS = tuple(range(1000, 1030))
MODEL_SEEDS = tuple(range(960, 965))
ARMS = ("random", "real", "fresh_noise", "oracle_cf")
ARM_LABELS = {
    "random": "Random",
    "real": "Real-only",
    "fresh_noise": "Fresh-noise synthetic",
    "oracle_cf": "Oracle CF",
}
ARM_COLORS = {
    "random": "#6B7280",
    "real": "#0072B2",
    "fresh_noise": "#D55E00",
    "oracle_cf": "#009E73",
}


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and include its path in malformed-file errors."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON summary {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: summary must contain a JSON object")
    return payload


def _finite_array(values: Any, *, name: str, length: int) -> np.ndarray:
    """Convert a summary array to a finite one-dimensional float array."""
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if result.shape != (length,):
        raise ValueError(f"{name} must contain exactly {length} seed values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def validate_summaries(
    oracle: dict[str, Any], model_gate: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate schemas, registered seeds, arms, and plotting data contracts."""
    if oracle.get("artifact_schema") != ORACLE_SCHEMA:
        raise ValueError("Oracle summary has an unexpected artifact schema")
    if tuple(oracle.get("seeds", ())) != ORACLE_SEEDS:
        raise ValueError("Oracle summary must contain exactly seeds 1000 through 1029")
    if oracle.get("registered_primary") != "oracle_cf_minus_fresh_noise_clean":
        raise ValueError(
            "Oracle summary primary contrast is not the registered clean contrast"
        )
    for arm in ARMS:
        arm_payload = oracle.get("arms", {}).get(arm)
        if not isinstance(arm_payload, dict):
            raise TypeError(f"Oracle summary is missing arm {arm}")
        for metric in ("clean", "noisy"):
            if metric not in arm_payload:
                raise ValueError(f"Oracle arm {arm} is missing {metric} evaluation")
            _finite_array(
                arm_payload[metric].get("seed_means"),
                name=f"arms.{arm}.{metric}.seed_means",
                length=len(ORACLE_SEEDS),
            )
            for bound in ("t_ci95_low", "t_ci95_high"):
                if not np.isfinite(float(arm_payload[metric].get(bound))):
                    raise ValueError(f"arms.{arm}.{metric}.{bound} is not finite")

    primary = oracle.get("contrasts", {}).get("oracle_cf_minus_fresh_noise_clean")
    if not isinstance(primary, dict):
        raise TypeError("Oracle summary is missing the registered primary contrast")
    _finite_array(primary.get("seed_deltas"), name="primary.seed_deltas", length=30)
    for bound in ("bootstrap_ci95_low", "bootstrap_ci95_high"):
        if not np.isfinite(float(primary.get(bound))):
            raise ValueError(f"primary.{bound} is not finite")

    if model_gate.get("artifact_schema") != MODEL_SCHEMA:
        raise ValueError("Model-gate summary has an unexpected artifact schema")
    if tuple(model_gate.get("seeds", ())) != MODEL_SEEDS:
        raise ValueError(
            "Model-gate summary must contain exactly seeds 960 through 964"
        )
    metrics = model_gate.get("per_seed_metrics")
    if not isinstance(metrics, dict):
        raise TypeError("Model-gate summary is missing per-seed metrics")
    for seed in MODEL_SEEDS:
        record = metrics.get(str(seed))
        if not isinstance(record, dict):
            raise TypeError(f"Model-gate summary is missing seed {seed}")
        for key in ("learned_to_fresh_mse_ratio", "terminal_disagreement"):
            if not np.isfinite(float(record.get(key))):
                raise ValueError(f"model-gate {seed}.{key} is not finite")
    return oracle, model_gate


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save vector PDF and 300-dpi raster PNG, then release the figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _style() -> None:
    """Apply restrained, colorblind-readable plot defaults."""
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


def plot_primary_delta(oracle: dict[str, Any], output_dir: Path) -> None:
    """Plot paired seed-level oracle-minus-fresh clean-return differences."""
    contrast = oracle["contrasts"]["oracle_cf_minus_fresh_noise_clean"]
    deltas = np.asarray(contrast["seed_deltas"], dtype=np.float64)
    x = np.arange(len(deltas))
    mean = float(contrast["mean_delta"])
    low = float(contrast["bootstrap_ci95_low"])
    high = float(contrast["bootstrap_ci95_high"])
    fig, axis = plt.subplots(figsize=(7.4, 3.9))
    axis.axhspan(-25, 25, color="#9CA3AF", alpha=0.12, zorder=0)
    axis.axhline(0, color="#111827", linewidth=1.0, label="No difference")
    axis.axhline(25, color="#6B7280", linestyle="--", linewidth=0.9)
    axis.axhline(-25, color="#6B7280", linestyle="--", linewidth=0.9)
    axis.plot(x, deltas, "o", color="#009E73", markersize=4.5, alpha=0.82)
    axis.axhline(mean, color="#D55E00", linewidth=1.8, label="Mean difference")
    axis.axhspan(low, high, color="#D55E00", alpha=0.16, label="95% bootstrap CI")
    axis.set_xlabel("Training seed (30 paired runs)")
    axis.set_ylabel("Oracle CF minus fresh-noise clean return")
    axis.set_title("Paired clean-return differences across training seeds")
    axis.set_xticks(
        x[::2], [str(seed) for seed in ORACLE_SEEDS[::2]], rotation=45, ha="right"
    )
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    axis.text(
        0.995,
        0.02,
        "Dashed lines: +/-25 practical thresholds",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#4B5563",
    )
    fig.tight_layout()
    _save(fig, output_dir, "oracle_minus_fresh_clean_seed_deltas")


def plot_arm_distributions(oracle: dict[str, Any], output_dir: Path) -> None:
    """Plot independent clean and process-noise seed distributions in panels."""
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.9), sharey=False)
    rng = np.random.default_rng(7)
    for axis, metric in zip(axes, ("clean", "noisy")):
        for index, arm in enumerate(ARMS):
            payload = oracle["arms"][arm][metric]
            values = np.asarray(payload["seed_means"], dtype=np.float64)
            jitter = rng.uniform(-0.075, 0.075, len(values))
            axis.scatter(
                index + jitter,
                values,
                s=14,
                alpha=0.58,
                color=ARM_COLORS[arm],
                edgecolors="white",
                linewidths=0.25,
                zorder=2,
            )
            mean = float(payload["mean"])
            low = float(payload["t_ci95_low"])
            high = float(payload["t_ci95_high"])
            axis.errorbar(
                index,
                mean,
                yerr=[[mean - low], [high - mean]],
                color="#111827",
                marker="D",
                markersize=4,
                capsize=3,
                linewidth=1.1,
                zorder=3,
            )
        axis.set_xticks(
            range(len(ARMS)), [ARM_LABELS[arm] for arm in ARMS], rotation=28, ha="right"
        )
        axis.set_xlabel("Training condition")
        axis.set_title(
            "Clean evaluation" if metric == "clean" else "Process-noise evaluation"
        )
        axis.set_xlim(-0.5, len(ARMS) - 0.5)
    axes[0].set_ylabel("Return (seed-level evaluation mean)")
    fig.suptitle("Four-arm return distributions by evaluation protocol", y=1.01)
    fig.text(
        0.5,
        -0.02,
        "Points: individual training seeds; diamonds: mean with 95% t interval",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, output_dir, "four_arm_seed_returns_clean_process_noise")


def plot_model_gate(model_gate: dict[str, Any], output_dir: Path) -> None:
    """Plot learned-CF fidelity diagnostics against registered model thresholds."""
    metrics = model_gate["per_seed_metrics"]
    ratios = np.asarray(
        [metrics[str(seed)]["learned_to_fresh_mse_ratio"] for seed in MODEL_SEEDS]
    )
    terminal = np.asarray(
        [metrics[str(seed)]["terminal_disagreement"] for seed in MODEL_SEEDS]
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    x = np.arange(len(MODEL_SEEDS))
    axes[0].bar(x, ratios, color="#D55E00", alpha=0.86, width=0.65)
    axes[0].axhline(0.8, color="#111827", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Learned / fresh normalized MSE")
    axes[0].set_title("Held-out transition fidelity")
    axes[0].text(
        0.98,
        0.91,
        "Pass if ratio < 0.8",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=7.5,
    )
    axes[1].bar(x, terminal, color="#0072B2", alpha=0.86, width=0.65)
    axes[1].axhline(0.05, color="#111827", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Terminal disagreement")
    axes[1].set_title("Termination consistency")
    axes[1].text(
        0.98,
        0.91,
        "Pass if disagreement < 0.05",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=7.5,
    )
    for axis in axes:
        axis.set_xlabel("Development seed")
        axis.set_xticks(x, [str(seed) for seed in MODEL_SEEDS])
    fig.suptitle("Development-only BiCoGAN model-gate diagnostics", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Registered thresholds are shown as dashed lines; this is not confirmatory downstream evidence.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout()
    _save(fig, output_dir, "model_gate_fidelity_and_terminal_disagreement")


def main() -> None:
    """Validate summaries and write all publication figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-summary",
        type=Path,
        default=Path("results/revision/oracle_confirmatory_final/summary.json"),
    )
    parser.add_argument(
        "--model-gate-summary",
        type=Path,
        default=Path("results/revision/model_gate_final/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/oracle_confirmatory_final/figures"),
    )
    args = parser.parse_args()
    oracle, model_gate = validate_summaries(
        _read_json(args.oracle_summary), _read_json(args.model_gate_summary)
    )
    _style()
    plot_primary_delta(oracle, args.output_dir)
    plot_arm_distributions(oracle, args.output_dir)
    plot_model_gate(model_gate, args.output_dir)
    print(f"Wrote six files (three PDF/PNG pairs) to {args.output_dir}")


if __name__ == "__main__":
    main()
