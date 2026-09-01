"""Development-only diagnostic for the CartPole protocol interpretation gap.

This harness trains the published-size real-only D3QN for every combination of
state-noise semantics and trial termination rule.  It is intentionally a
protocol-sensitivity diagnostic, not a confirmatory reproduction: all four
arms use the same seed, optimizer budget, evaluation seed bank, and learner
architecture within a seed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.revision.cartpole_ctrl_reproduction import (
    ReproductionConfig,
    evaluate_policy,
    generate_paired_dataset,
    normalization,
    train_offline_d3qn,
)
from scripts.revision.cartpole_sanity import set_seed

EXPECTED_SEEDS = tuple(range(970, 980))
ARTIFACT_SCHEMA = "ctrl-cartpole-protocol-sensitivity-v1"
EXPERIMENT_TIER = "development_protocol_sensitivity"
ARM_NAMES = (
    "process_stop",
    "process_continue",
    "observation_stop",
    "observation_continue",
)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file used by the harness."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    """Capture hashes for the runner and every shared executable dependency."""
    root = Path(__file__).resolve().parents[2]
    paths = {
        "cartpole_protocol_sensitivity.py": Path(__file__),
        "cartpole_ctrl_reproduction.py": Path(__file__).with_name(
            "cartpole_ctrl_reproduction.py"
        ),
        "cartpole_sanity.py": Path(__file__).with_name("cartpole_sanity.py"),
        "pyproject.toml": root / "pyproject.toml",
        "uv.lock": root / "uv.lock",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def git_provenance() -> dict[str, object]:
    """Return the current Git commit and working-tree state when available."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def software_metadata(device: torch.device) -> dict[str, object]:
    """Return versions and accelerator details needed to interpret a run."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gymnasium": importlib.metadata.version("gymnasium"),
        "scipy": importlib.metadata.version("scipy"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def config_dict(config: ReproductionConfig) -> dict[str, Any]:
    """Serialize a reproduction configuration without a non-JSON Path value."""
    values = asdict(config)
    values["output_dir"] = str(config.output_dir)
    return values


def make_config(seed: int, output_dir: Path) -> ReproductionConfig:
    """Construct the registered published-size real-only learner config."""
    return ReproductionConfig(
        seed=seed,
        experiment_tier=EXPERIMENT_TIER,
        dataset_trials=250,
        validation_dataset_trials=50,
        trial_horizon=20,
        train_steps=10_000,
        batch_size=256,
        eval_episodes=100,
        eval_horizon=500,
        learning_rate=1e-4,
        target_tau=0.005,
        cql_alpha=0.0,
        q_width=512,
        q_depth=4,
        q_batch_norm=True,
        state_noise_std=0.05,
        action_noise_std=0.05,
        eval_seed_base=600_000,
        output_dir=output_dir,
    )


def run_arm(
    name: str,
    config: ReproductionConfig,
    device: torch.device,
) -> dict[str, object]:
    """Train and evaluate one protocol arm using the fixed real-only learner."""
    set_seed(config.seed)
    real, _, _, dataset = generate_paired_dataset(config)
    state_mean, state_std = normalization(real)
    policy, training_logs, _, _ = train_offline_d3qn(
        real, None, config, device
    )
    clean_returns = evaluate_policy(
        policy, state_mean, state_std, config, device, noisy=False
    )
    noisy_returns = evaluate_policy(
        policy, state_mean, state_std, config, device, noisy=True
    )
    return {
        "arm": name,
        "config": config_dict(config),
        "dataset": dataset,
        "training": training_logs,
        "clean": {"returns": clean_returns},
        "noisy": {"returns": noisy_returns},
    }


def run_seed(seed: int, output_dir: Path) -> dict[str, object]:
    """Run all four protocol interpretations for one paired training seed."""
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"seed must be one of {EXPECTED_SEEDS}, got {seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arms: dict[str, dict[str, object]] = {}
    for noise_semantics, stop_on_termination, name in (
        ("process", True, "process_stop"),
        ("process", False, "process_continue"),
        ("observation", True, "observation_stop"),
        ("observation", False, "observation_continue"),
    ):
        config = replace(
            make_config(seed, output_dir),
            noise_semantics=noise_semantics,
            stop_on_termination=stop_on_termination,
        )
        arms[name] = run_arm(name, config, device)
        print(
            f"[seed {seed}] {name}: "
            f"clean={np.mean(arms[name]['clean']['returns']):.2f} "
            f"noisy={np.mean(arms[name]['noisy']['returns']):.2f}",
            flush=True,
        )
    payload: dict[str, object] = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "development_only": True,
        "confirmatory": False,
        "experiment_tier": EXPERIMENT_TIER,
        "interpretation": (
            "Protocol-sensitivity diagnostic for explaining reproduction gaps; "
            "not confirmatory evidence and not a claim that any arm is CTRL."
        ),
        "config": config_dict(make_config(seed, output_dir)),
        "matrix": {
            "arms": list(ARM_NAMES),
            "same_seed_initialization_and_update_budget": True,
            "state_noise_std": 0.05,
            "action_noise_std": 0.05,
            "dataset_trials": 250,
            "trial_horizon": 20,
            "evaluation_episodes": 100,
            "evaluation_seed_range": [600_000, 600_099],
            "evaluation_clean_and_noisy_use_same_seed_bank": True,
        },
        "command": [sys.executable, *sys.argv],
        "git": git_provenance(),
        "software": software_metadata(device),
        "device": str(device),
        "source_hashes": source_hashes(),
        "arms": arms,
    }
    path = output_dir / f"protocol_sensitivity_seed_{seed}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return payload


def main() -> None:
    """Run one registered development protocol-sensitivity seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/protocol_sensitivity"),
    )
    args = parser.parse_args()
    run_seed(args.seed, args.output_dir)


if __name__ == "__main__":
    main()
