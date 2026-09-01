#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _bar_with_error(
    labels: list[str],
    means: list[float],
    errs: list[float],
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    colors = ["#5B8DEF", "#28A745", "#F2A93B", "#D95F59"][: len(labels)]
    ax.bar(labels, means, yerr=errs, capsize=5, color=colors, edgecolor="#222222")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0, top=max(520, max(means) + max(errs) + 40))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot revision experiment summaries.")
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("results/revision/remote_collect"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/figures"),
    )
    args = parser.parse_args()

    sanity = json.loads((args.summary_dir / "cartpole_sanity_summary.json").read_text())
    oracle = json.loads((args.summary_dir / "cartpole_oracle_5k_summary.json").read_text())

    m = sanity["metrics"]
    _bar_with_error(
        labels=[
            "Random clean",
            "DQN clean",
            "Random noisy",
            "DQN noisy",
        ],
        means=[
            m["random_clean"]["mean"],
            m["best_clean"]["mean"],
            m["random_ctrl_noisy"]["mean"],
            m["best_ctrl_noisy_at_best"]["mean"],
        ],
        errs=[
            m["random_clean"]["ci95"],
            m["best_clean"]["ci95"],
            m["random_ctrl_noisy"]["ci95"],
            m["best_ctrl_noisy_at_best"]["ci95"],
        ],
        ylabel="Mean return over 100 rollouts",
        title="Corrected 11-action CartPole sanity check",
        output=args.output_dir / "cartpole_corrected_sanity.png",
    )

    _bar_with_error(
        labels=["Real clean", "Oracle-CF clean", "Real noisy", "Oracle-CF noisy"],
        means=[
            oracle["real"]["clean"]["mean"],
            oracle["oracle_cf"]["clean"]["mean"],
            oracle["real"]["ctrl_noisy"]["mean"],
            oracle["oracle_cf"]["ctrl_noisy"]["mean"],
        ],
        errs=[
            oracle["real"]["clean"]["ci95"],
            oracle["oracle_cf"]["clean"]["ci95"],
            oracle["real"]["ctrl_noisy"]["ci95"],
            oracle["oracle_cf"]["ctrl_noisy"]["ci95"],
        ],
        ylabel="Mean return over 100 rollouts",
        title="Offline DQN with exact physics counterfactuals",
        output=args.output_dir / "cartpole_oracle_cf_5k.png",
    )

    real_by_seed = {}
    cf_by_seed = {}
    for path in args.summary_dir.rglob("*cartpole_oracle_cf_5k*/*_seed_*.json"):
        data = json.loads(path.read_text())
        if data.get("mode") == "real":
            real_by_seed[data["config"]["seed"]] = data
        elif data.get("mode") == "oracle_cf":
            cf_by_seed[data["config"]["seed"]] = data

    seeds = sorted(set(real_by_seed) & set(cf_by_seed))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    for seed in seeds:
        real = real_by_seed[seed]["clean"]["mean"]
        cf = cf_by_seed[seed]["clean"]["mean"]
        ax.plot([0, 1], [real, cf], marker="o", color="#555555", alpha=0.75)
        ax.text(1.03, cf, str(seed), fontsize=8, va="center")
    ax.set_xticks([0, 1], ["Real", "Oracle-CF"])
    ax.set_ylabel("Clean mean return")
    ax.set_title("Per-seed effect of exact counterfactual augmentation")
    ax.set_ylim(0, 520)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "cartpole_oracle_cf_seed_deltas.png")
    plt.close(fig)

    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
