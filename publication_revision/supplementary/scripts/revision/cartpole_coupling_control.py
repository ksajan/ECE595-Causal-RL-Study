"""Run a frozen CartPole coupling-control experiment.

The learner, dataset size, process-noise semantics, optimization budget, and
evaluation bank are copied from ``oracle_confirmatory_manifest.json``.  This
experiment isolates how exogenous-noise coupling affects augmentation:
``fresh_independent`` draws new noise for every alternative action,
``fresh_shared`` draws one fresh noise pair per factual transition, and
``oracle_cf`` reuses the factual pair.  The runner never trains a learned SCM.
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
from typing import Any

import numpy as np
import torch

from scripts.revision.cartpole_ctrl_reproduction import (
    ReproductionConfig,
    TransitionBuilder,
    TransitionPool,
    counterfactual_pool_diagnostics,
    evaluate_policy,
    generate_paired_dataset,
    noisy_observation,
    normalization,
    physics_step,
    run_condition,
)
from scripts.revision.cartpole_sanity import set_seed

SCHEMA = "ctrl-cartpole-coupling-control-v1"
DEFAULT_MANIFEST = Path(__file__).with_name("coupling_control_manifest.json")
FROZEN_MANIFEST_SHA256 = (
    "0dbb740ba4045badc105be161631e60cc8fa392e6e25d100bcd8accd0458a257"
)
EXPECTED_SEEDS = tuple(range(1030, 1060))
ARMS = ("random", "real", "fresh_independent", "fresh_shared", "oracle_cf")
ALTERNATIVES_PER_TRANSITION = 10


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    """Hash the new runner and every executable source it imports directly."""
    root = Path(__file__).resolve().parents[2]
    paths = {
        "cartpole_coupling_control.py": Path(__file__),
        "cartpole_ctrl_reproduction.py": Path(__file__).with_name(
            "cartpole_ctrl_reproduction.py"
        ),
        "bicogan_ctrl.py": Path(__file__).with_name("bicogan_ctrl.py"),
        "cartpole_sanity.py": Path(__file__).with_name("cartpole_sanity.py"),
        "pyproject.toml": root / "pyproject.toml",
        "uv.lock": root / "uv.lock",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def source_digest() -> str:
    """Return a deterministic digest of all executable source files."""
    digest = hashlib.sha256()
    for name, value in sorted(source_hashes().items()):
        digest.update(name.encode())
        digest.update(value.encode())
    return digest.hexdigest()


def source_manifest_digest(manifest_path: Path) -> str:
    """Return a digest binding the runner, shared sources, and manifest."""
    digest = hashlib.sha256()
    for name, value in sorted(source_hashes().items()):
        digest.update(name.encode())
        digest.update(value.encode())
    digest.update(manifest_path.name.encode())
    digest.update(manifest_path.read_bytes())
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and reject any change to the prospectively frozen design."""
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load manifest {path}") from error
    if sha256_file(path) != FROZEN_MANIFEST_SHA256:
        raise ValueError("Manifest hash does not match the frozen coupling design")
    if manifest.get("artifact_schema") != SCHEMA:
        raise ValueError(f"Manifest must declare schema {SCHEMA}")
    if manifest.get("seeds") != list(EXPECTED_SEEDS):
        raise ValueError("Manifest must contain exactly seeds 1030 through 1059")
    if manifest.get("arms") != list(ARMS):
        raise ValueError("Manifest arms do not match the frozen coupling design")
    if manifest["config"].get("noise_semantics") != "process":
        raise ValueError("The frozen coupling design requires process noise")
    return manifest


def config_from_manifest(
    manifest: dict[str, Any], seed: int, output_dir: Path
) -> ReproductionConfig:
    """Construct the exact stabilized learner configuration for one seed."""
    values = dict(manifest["config"])
    values.update(
        {
            "seed": seed,
            "experiment_tier": "oracle_coupling_control",
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
    """Reject learner or protocol drift from the manifest."""
    for field, expected in manifest["config"].items():
        if getattr(config, field) != expected:
            raise ValueError(
                f"Configuration drift for {field}: expected {expected!r}, "
                f"got {getattr(config, field)!r}"
            )
    if config.seed not in manifest["seeds"]:
        raise ValueError(f"Seed {config.seed} is outside the manifest seed set")


def software_metadata(device: torch.device) -> dict[str, Any]:
    """Record runtime versions and accelerator information."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "gymnasium": importlib.metadata.version("gymnasium"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def git_provenance() -> dict[str, object]:
    """Record the repository commit and dirty state used by the run."""
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


def _assert_pool_equal(left: TransitionPool, right: TransitionPool, label: str) -> None:
    """Require exact equality for every factual transition field."""
    for field in ("states", "actions", "rewards", "next_states", "dones", "trial_ids"):
        if not np.array_equal(getattr(left, field), getattr(right, field)):
            raise RuntimeError(f"Factual {label} mismatch in field {field}")


def _shared_noise_diagnostics(
    shared_action_noise: np.ndarray,
    shared_state_noise: np.ndarray,
    shared_pool: TransitionPool,
    independent_pool: TransitionPool,
) -> dict[str, object]:
    """Summarize within-transition reuse and across-transition diversity."""
    transition_count = len(shared_action_noise)
    if transition_count == 0:
        raise RuntimeError("Shared-noise diagnostics require factual transitions")
    action_by_sibling = np.repeat(shared_action_noise, ALTERNATIVES_PER_TRANSITION)
    state_by_sibling = np.repeat(
        shared_state_noise, ALTERNATIVES_PER_TRANSITION, axis=0
    )
    action_groups = action_by_sibling.reshape(transition_count, -1)
    state_groups = state_by_sibling.reshape(transition_count, -1, 4)
    shared_unique_states = len(np.unique(shared_state_noise, axis=0))
    independent_equal_fraction = float(
        np.mean(
            np.all(
                np.asarray(shared_pool.next_states)
                == np.asarray(independent_pool.next_states),
                axis=1,
            )
        )
    )
    return {
        "factual_alignment": True,
        "shared_noise_draws": transition_count,
        "fresh_independent_noise_draws": len(independent_pool),
        "alternative_actions_per_transition": ALTERNATIVES_PER_TRANSITION,
        "within_transition_action_noise_reuse_max_abs": float(
            np.max(np.ptp(action_groups, axis=1))
        ),
        "within_transition_state_noise_reuse_max_abs": float(
            np.max(np.ptp(state_groups, axis=1))
        ),
        "shared_action_noise_unique_count": len(np.unique(shared_action_noise)),
        "shared_state_noise_unique_count": int(shared_unique_states),
        "shared_action_noise_diversity_fraction": float(
            len(np.unique(shared_action_noise)) / transition_count
        ),
        "shared_state_noise_diversity_fraction": float(
            shared_unique_states / transition_count
        ),
        "fresh_shared_vs_independent_exact_next_state_fraction": independent_equal_fraction,
        "fresh_shared_next_state_sibling_std_mean": float(
            np.mean(
                np.std(
                    shared_pool.next_states.reshape(
                        transition_count, ALTERNATIVES_PER_TRANSITION, 4
                    ),
                    axis=1,
                )
            )
        ),
    }


def generate_coupling_datasets(
    config: ReproductionConfig,
) -> tuple[
    TransitionPool,
    TransitionPool,
    TransitionPool,
    TransitionPool,
    dict[str, object],
    dict[str, object],
]:
    """Build shared-noise CFs and prove factual alignment with the public dataset."""
    if config.noise_semantics != "process":
        raise ValueError("Coupling control is frozen to process-noise semantics")
    reference_real, independent, oracle, reference_metadata = generate_paired_dataset(
        config
    )
    factual_rng = np.random.default_rng(config.seed)
    shared_rng = np.random.default_rng(config.seed + 1_000_000)
    real_builder = TransitionBuilder()
    shared_builder = TransitionBuilder()
    shared_action_noise: list[float] = []
    shared_state_noise: list[np.ndarray] = []

    for trial_id in range(config.dataset_trials):
        dynamics_state = factual_rng.uniform(-0.05, 0.05, size=4)
        state_observation = dynamics_state.astype(np.float32)
        trial_failed = False
        for _step in range(config.trial_horizon):
            factual_action = int(factual_rng.integers(0, 11))
            action_noise = float(factual_rng.normal(0.0, config.action_noise_std))
            next_observation_noise = factual_rng.normal(size=4)
            next_clean_state, factual_terminated = physics_step(
                dynamics_state, factual_action, action_noise
            )
            next_observation = noisy_observation(
                next_clean_state,
                next_observation_noise,
                config.state_noise_std,
            )
            real_builder.add(
                state_observation,
                factual_action,
                1.0,
                next_observation,
                factual_terminated,
                trial_id,
            )

            one_shared_action_noise = float(
                shared_rng.normal(0.0, config.action_noise_std)
            )
            one_shared_state_noise = shared_rng.normal(size=4)
            shared_action_noise.append(one_shared_action_noise)
            shared_state_noise.append(one_shared_state_noise.copy())
            for counterfactual_action in range(11):
                if counterfactual_action == factual_action:
                    continue
                cf_clean_state, cf_terminated = physics_step(
                    dynamics_state,
                    counterfactual_action,
                    one_shared_action_noise,
                )
                cf_observation = noisy_observation(
                    cf_clean_state,
                    one_shared_state_noise,
                    config.state_noise_std,
                )
                shared_builder.add(
                    state_observation,
                    counterfactual_action,
                    1.0,
                    cf_observation,
                    cf_terminated,
                    trial_id,
                )

            trial_failed = trial_failed or factual_terminated
            dynamics_state = next_observation
            state_observation = next_observation
            if factual_terminated and config.stop_on_termination:
                break

    factual = real_builder.finish()
    shared = shared_builder.finish()
    _assert_pool_equal(factual, reference_real, "dataset")
    if len(shared) != len(factual) * ALTERNATIVES_PER_TRANSITION:
        raise RuntimeError(
            "Shared CF pool does not contain ten alternatives per transition"
        )
    shared_action_array = np.asarray(shared_action_noise, dtype=np.float64)
    shared_state_array = np.asarray(shared_state_noise, dtype=np.float64)
    expanded_action = np.repeat(shared_action_array, ALTERNATIVES_PER_TRANSITION)
    expanded_state = np.repeat(shared_state_array, ALTERNATIVES_PER_TRANSITION, axis=0)
    diagnostics = _shared_noise_diagnostics(
        shared_action_array, shared_state_array, shared, independent
    )
    debug = {
        "shared_action_noise_per_transition": shared_action_array,
        "shared_state_noise_per_transition": shared_state_array,
        "shared_action_noise_per_alternative": expanded_action,
        "shared_state_noise_per_alternative": expanded_state,
    }
    metadata: dict[str, object] = {
        **reference_metadata,
        "real_transitions": len(factual),
        "fresh_independent_transitions": len(independent),
        "fresh_shared_transitions": len(shared),
        "oracle_cf_transitions": len(oracle),
        "factual_alignment_with_generate_paired_dataset": True,
        "stop_on_termination": config.stop_on_termination,
    }
    return factual, independent, shared, oracle, metadata, {**debug, **diagnostics}


def random_condition(
    config: ReproductionConfig,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate a uniformly random policy on the shared evaluation bank."""
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
    """Run every registered arm for one seed and write its raw artifact."""
    manifest = load_manifest(manifest_path)
    config = config_from_manifest(manifest, seed, output_dir)
    validate_config(config, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real, independent, shared, oracle, dataset, coupling = generate_coupling_datasets(
        config
    )
    state_mean, state_std = normalization(real)
    arms: dict[str, dict[str, object]] = {
        "random": random_condition(config, state_mean, state_std, device),
        "real": run_condition("real", real, None, config, device),
        "fresh_independent": run_condition(
            "fresh_independent", real, independent, config, device
        ),
        "fresh_shared": run_condition("fresh_shared", real, shared, config, device),
        "oracle_cf": run_condition("oracle_cf", real, oracle, config, device),
    }
    coupling_public = {
        key: value
        for key, value in coupling.items()
        if not isinstance(value, np.ndarray)
    }
    payload: dict[str, Any] = {
        "artifact_schema": SCHEMA,
        "experiment_name": manifest["experiment_name"],
        "experiment_tier": "oracle_coupling_control",
        "confirmatory": True,
        "config": {**asdict(config), "output_dir": str(output_dir)},
        "command": [sys.executable, *sys.argv],
        "git": git_provenance(),
        "software": software_metadata(device),
        "source_hashes": source_hashes(),
        "source_sha256": source_digest(),
        "manifest_sha256": sha256_file(manifest_path),
        "source_manifest_sha256": source_manifest_digest(manifest_path),
        "dataset": {
            **dataset,
            "dataset_trials": config.dataset_trials,
            "trial_horizon": config.trial_horizon,
            "stop_on_termination": config.stop_on_termination,
            "terminal_label_rule": "pre_noise_next_state",
            "noise_semantics": config.noise_semantics,
        },
        "diagnostics": {
            "coupling": coupling_public,
            "fresh_independent_vs_oracle": counterfactual_pool_diagnostics(
                independent, oracle, state_std
            ),
            "fresh_shared_vs_oracle": counterfactual_pool_diagnostics(
                shared, oracle, state_std
            ),
        },
        "arms": arms,
    }
    result_path = output_dir / f"coupling_seed_{seed}.json"
    result_path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str))
    return result_path


def main() -> None:
    """Parse one manifest-approved seed and run it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/revision/coupling_control")
    )
    args = parser.parse_args()
    result = run_seed(args.seed, args.manifest, args.output_dir)
    print(f"wrote {result}")


if __name__ == "__main__":
    main()
