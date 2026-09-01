#!/usr/bin/env python3
"""Run the development-only learned CTRL SCM model-quality gate.

This script deliberately stops after BiCoGAN validation.  It does not train an
offline learner and its artifacts must never be used as downstream evidence.
The training set contains 250 process-noise trials capped at 20 transitions;
the model checks are computed on an independent 50-trial seed bank.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from scripts.revision.bicogan_ctrl import BiCoGANConfig, CTRLBiCoGAN
from scripts.revision.cartpole_ctrl_reproduction import (
    ReproductionConfig,
    counterfactual_pool_diagnostics,
    generate_learned_counterfactuals,
    generate_paired_dataset,
    normalization,
)
from scripts.revision.cartpole_sanity import set_seed

ARTIFACT_SCHEMA = "ctrl-cartpole-model-gate-v1"
EXPERIMENT_TIER = "ctrl_bicogan_reproduction"
GENERATOR_KIND = "monotonic_bicogan"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source or configuration file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    """Hash every shared source/configuration file used by this diagnostic."""
    root = Path(__file__).resolve().parents[2]
    paths = {
        "cartpole_model_gate.py": Path(__file__),
        "cartpole_ctrl_reproduction.py": Path(__file__).with_name(
            "cartpole_ctrl_reproduction.py"
        ),
        "bicogan_ctrl.py": Path(__file__).with_name("bicogan_ctrl.py"),
        "cartpole_sanity.py": Path(__file__).with_name("cartpole_sanity.py"),
        "REVISION_EXPERIMENT_PROTOCOL.md": root / "REVISION_EXPERIMENT_PROTOCOL.md",
        "pyproject.toml": root / "pyproject.toml",
        "uv.lock": root / "uv.lock",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def git_provenance() -> dict[str, object]:
    """Capture the current commit and dirty state when Git is available."""
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
    """Return versions and accelerator information needed to reproduce a run."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gymnasium": importlib.metadata.version("gymnasium"),
        "scipy": importlib.metadata.version("scipy"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def parse_args() -> argparse.Namespace:
    """Parse the fixed model-gate protocol with explicit development overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/revision/model_gate"))
    parser.add_argument("--bicogan-pretrain-steps", type=int, default=2_000)
    parser.add_argument("--bicogan-steps", type=int, default=5_000)
    parser.add_argument("--state-noise-std", type=float, default=0.05)
    parser.add_argument("--action-noise-std", type=float, default=0.05)
    return parser.parse_args()


def run_model_gate(args: argparse.Namespace) -> dict[str, object]:
    """Train one registered monotonic SCM and emit independent diagnostics."""
    if args.bicogan_pretrain_steps <= 0 or args.bicogan_steps <= 0:
        raise ValueError("BiCoGAN training steps must be positive.")
    config = ReproductionConfig(
        seed=args.seed,
        experiment_tier=EXPERIMENT_TIER,
        dataset_trials=250,
        validation_dataset_trials=50,
        trial_horizon=20,
        stop_on_termination=True,
        noise_semantics="process",
        state_noise_std=args.state_noise_std,
        action_noise_std=args.action_noise_std,
        bicogan_pretrain_steps=args.bicogan_pretrain_steps,
        bicogan_steps=args.bicogan_steps,
        bicogan_generator=GENERATOR_KIND,
        output_dir=args.output_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real, _, oracle, dataset = generate_paired_dataset(config)
    validation_config = ReproductionConfig(
        **{
            **asdict(config),
            "seed": config.seed + config.validation_seed_offset,
            "dataset_trials": config.validation_dataset_trials,
        }
    )
    validation_real, validation_fresh, validation_oracle, validation_metadata = (
        generate_paired_dataset(validation_config)
    )
    state_mean, state_std = normalization(real)
    model = CTRLBiCoGAN(
        BiCoGANConfig(
            pretrain_steps=config.bicogan_pretrain_steps,
            adversarial_steps=config.bicogan_steps,
            generator_kind=GENERATOR_KIND,
        ),
        device,
    )
    diagnostics = model.fit(
        real.states,
        real.actions,
        real.next_states,
        real.trial_ids,
        state_mean,
        state_std,
        config.seed,
        validation_states=validation_real.states,
        validation_actions=validation_real.actions,
        validation_next_states=validation_real.next_states,
        validation_trial_ids=validation_real.trial_ids + config.dataset_trials,
    )
    learned_training = generate_learned_counterfactuals(
        real, model, state_mean, state_std
    )
    learned_validation = generate_learned_counterfactuals(
        validation_real, model, state_mean, state_std
    )
    external = counterfactual_pool_diagnostics(
        learned_validation, validation_oracle, state_std
    )
    fresh_external = counterfactual_pool_diagnostics(
        validation_fresh, validation_oracle, state_std
    )
    training_error = counterfactual_pool_diagnostics(learned_training, oracle, state_std)
    checkpoint = config.output_dir / f"model_gate_seed_{config.seed}.pt"
    model.save(checkpoint)
    dataset.update(
        {
            "validation_seed": validation_config.seed,
            "validation_real_transitions": validation_metadata["real_transitions"],
            "validation_post_failure_transitions": validation_metadata[
                "post_failure_transitions"
            ],
        }
    )
    payload: dict[str, object] = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "development_only": True,
        "experiment_tier": EXPERIMENT_TIER,
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "command": [sys.executable, *sys.argv],
        "git": git_provenance(),
        "software": software_metadata(device),
        "device": str(device),
        "source_hashes": source_hashes(),
        "dataset": dataset,
        "bicogan": {
            "diagnostics": diagnostics,
            "counterfactual_diagnostics": {
                "training": training_error,
                "external_validation": external,
                "fresh_noise_external_validation": fresh_external,
            },
            "checkpoint": str(checkpoint),
            "learned_cf_transitions": len(learned_training),
        },
        "registered_gates": {
            "learned_mse_lt_0.8_fresh_mse": external["normalized_mse"]
            < 0.8 * fresh_external["normalized_mse"],
            "terminal_disagreement_lt_0.05": external["terminal_disagreement"] < 0.05,
            "latent_std_each_between_0.5_and_2.0": bool(
                np.all(
                    (np.asarray(diagnostics["latent_std_by_dimension"]) >= 0.5)
                    & (np.asarray(diagnostics["latent_std_by_dimension"]) <= 2.0)
                )
            ),
            "action_mse_lt_0.9_central_baseline": diagnostics[
                "action_reconstruction_mse"
            ]
            < 0.9 * diagnostics["central_action_baseline_mse"],
        },
    }
    output = config.output_dir / f"model_gate_seed_{config.seed}.json"
    output.write_text(json.dumps(payload, indent=2))
    print(f"[seed {config.seed}] wrote {output}")
    return payload


def main() -> None:
    """Run one development gate seed."""
    run_model_gate(parse_args())


if __name__ == "__main__":
    main()
