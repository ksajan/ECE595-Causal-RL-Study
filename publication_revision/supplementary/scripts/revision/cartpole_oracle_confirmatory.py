#!/usr/bin/env python3
"""Run one frozen, paired CartPole oracle-CF confirmatory replicate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.revision.cartpole_ctrl_reproduction import (
    ReproductionConfig,
    counterfactual_pool_diagnostics,
    evaluate_policy,
    generate_paired_dataset,
    git_provenance,
    normalization,
    run_condition,
    set_seed,
    source_digest,
)

SCHEMA = "ctrl-cartpole-oracle-confirmatory-v1"
DEFAULT_MANIFEST = Path(__file__).with_name("oracle_confirmatory_manifest.json")
FROZEN_MANIFEST_SHA256 = (
    "40650403535193e8897be4fdd1b63041c6b1a767acc06c5d90a33e483e750b39"
)


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically for provenance hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest_digest(manifest: Path) -> str:
    """Hash the reused implementation sources together with the manifest."""
    digest = hashlib.sha256()
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("cartpole_ctrl_reproduction.py"),
        Path(__file__).with_name("cartpole_sanity.py"),
        manifest,
    )
    for path in source_paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and structurally validate the frozen manifest."""
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load manifest {path}") from error
    if sha256_file(path) != FROZEN_MANIFEST_SHA256:
        raise ValueError("Manifest hash does not match the frozen confirmatory design")
    if manifest.get("artifact_schema") != SCHEMA:
        raise ValueError(f"Manifest must declare schema {SCHEMA}")
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or seeds != list(range(1000, 1030)):
        raise ValueError("Manifest must contain exactly seeds 1000 through 1029")
    if manifest.get("arms") != ["random", "real", "fresh_noise", "oracle_cf"]:
        raise ValueError("Manifest arms do not match the frozen oracle design")
    return manifest


def config_from_manifest(
    manifest: dict[str, Any], seed: int, output_dir: Path
) -> ReproductionConfig:
    """Construct the reused learner configuration from frozen manifest fields."""
    values = dict(manifest["config"])
    values.update(
        {
            "seed": seed,
            "experiment_tier": "oracle_confirmatory",
            "validation_dataset_trials": 50,
            "validation_seed_offset": 2_000_000,
            "bicogan_pretrain_steps": 0,
            "bicogan_steps": 0,
            "bicogan_reconstruction_weight": 0.0,
            "bicogan_extrinsic_weight": 0.0,
            "bicogan_latent_cycle_weight": 0.0,
            "bicogan_generator": "unconstrained",
            "output_dir": output_dir,
        }
    )
    return ReproductionConfig(**values)


def validate_config(config: ReproductionConfig, manifest: dict[str, Any]) -> None:
    """Reject any run configuration that differs from the frozen manifest."""
    expected = dict(manifest["config"])
    actual = asdict(config)
    for field, expected_value in expected.items():
        if actual.get(field) != expected_value:
            raise ValueError(
                f"Configuration drift for {field}: expected {expected_value!r}, "
                f"got {actual.get(field)!r}"
            )
    if config.seed not in manifest["seeds"]:
        raise ValueError(f"Seed {config.seed} is outside the manifest seed set")


def software_metadata(device: torch.device) -> dict[str, Any]:
    """Record the runtime needed to interpret the result."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "gymnasium": importlib.metadata.version("gymnasium"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def random_condition(
    config: ReproductionConfig,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate the uniformly random policy using the shared evaluator."""
    clean = evaluate_policy(None, state_mean, state_std, config, device, noisy=False)
    noisy = evaluate_policy(None, state_mean, state_std, config, device, noisy=True)
    return {
        "clean": {
            "episodes": len(clean),
            "mean": float(np.mean(clean)),
            "returns": clean,
        },
        "noisy": {
            "episodes": len(noisy),
            "mean": float(np.mean(noisy)),
            "returns": noisy,
        },
    }


def run_seed(seed: int, manifest_path: Path, output_dir: Path) -> Path:
    """Run all four paired arms for one manifest-approved training seed."""
    manifest = load_manifest(manifest_path)
    config = config_from_manifest(manifest, seed, output_dir)
    validate_config(config, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real, fresh, oracle, dataset = generate_paired_dataset(config)
    state_mean, state_std = normalization(real)

    arms: dict[str, dict[str, object]] = {
        "random": random_condition(config, state_mean, state_std, device),
        "real": run_condition("real", real, None, config, device),
        "fresh_noise": run_condition("fresh_noise", real, fresh, config, device),
        "oracle_cf": run_condition("oracle_cf", real, oracle, config, device),
    }
    # The reused diagnostic is retained as provenance, not as a fifth learner arm.
    fresh_diag = counterfactual_pool_diagnostics(fresh, oracle, state_std)
    source_hash = source_digest()
    combined_hash = source_manifest_digest(manifest_path)
    payload: dict[str, Any] = {
        "artifact_schema": SCHEMA,
        "experiment_name": manifest["experiment_name"],
        "config": {**asdict(config), "output_dir": str(output_dir)},
        "command": [sys.executable, *sys.argv],
        "git": git_provenance(),
        "software": software_metadata(device),
        "source_sha256": source_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "source_manifest_sha256": combined_hash,
        "dataset": {
            **dataset,
            "postfailure_transitions": int(dataset["post_failure_transitions"]),
            "trial_horizon": config.trial_horizon,
            "dataset_trials": config.dataset_trials,
            "stop_on_termination": config.stop_on_termination,
        },
        "diagnostics": {"fresh_noise_vs_oracle": fresh_diag},
        "arms": arms,
    }
    result_path = output_dir / f"oracle_seed_{seed}.json"
    result_path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str))
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/revision/oracle_confirmatory")
    )
    args = parser.parse_args()
    result = run_seed(args.seed, args.manifest, args.output_dir)
    print(f"wrote {result}")


if __name__ == "__main__":
    main()
