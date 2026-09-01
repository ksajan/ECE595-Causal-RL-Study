#!/usr/bin/env python3
"""Run a development-only CartPole offline-D3QN baseline diagnostic."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from scripts.revision.cartpole_ctrl_reproduction import (
    ReproductionConfig,
    evaluate_policy,
    generate_paired_dataset,
    run_condition,
    source_digest,
    summarize_episodes,
)
from scripts.revision.cartpole_sanity import set_seed


def parse_args() -> ReproductionConfig:
    """Parse one baseline configuration from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--target-update-interval", type=int, default=0)
    parser.add_argument("--cql-alpha", type=float, default=0.0)
    parser.add_argument("--q-width", type=int, default=512)
    parser.add_argument("--q-depth", type=int, default=4)
    parser.add_argument(
        "--q-batch-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed-base", type=int, default=500_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return ReproductionConfig(
        seed=args.seed,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        target_tau=args.target_tau,
        target_update_interval=args.target_update_interval,
        cql_alpha=args.cql_alpha,
        q_width=args.q_width,
        q_depth=args.q_depth,
        q_batch_norm=args.q_batch_norm,
        eval_episodes=args.eval_episodes,
        eval_seed_base=args.eval_seed_base,
        output_dir=args.output_dir,
    )


def main() -> None:
    """Generate one dataset, train the real-only arm, and save an audit artifact."""
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real, _, _, dataset = generate_paired_dataset(config)
    real_result = run_condition("real", real, None, config, device)
    random_clean = evaluate_policy(
        None,
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        config,
        device,
        noisy=False,
    )
    random_noisy = evaluate_policy(
        None,
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        config,
        device,
        noisy=True,
    )
    payload = {
        "artifact_schema": "ctrl-cartpole-baseline-tuning-v1",
        "development_only": True,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_digest(),
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "dataset": dataset,
        "conditions": {
            "random": {
                "clean": summarize_episodes(random_clean),
                "noisy": summarize_episodes(random_noisy),
            },
            "real": real_result,
        },
    }
    output = config.output_dir / f"baseline_seed_{config.seed}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
