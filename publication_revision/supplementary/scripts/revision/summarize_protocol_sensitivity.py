"""Validate and summarize the development-only protocol sensitivity matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from scripts.revision.cartpole_protocol_sensitivity import (
    ARM_NAMES,
    ARTIFACT_SCHEMA,
    EXPECTED_SEEDS,
    source_hashes,
)


def finite(value: Any, label: str) -> float:
    """Convert a scalar to a finite float or reject it."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def reject_nonfinite(value: Any, label: str) -> None:
    """Reject nonfinite numeric values recursively in an artifact."""
    if isinstance(value, dict):
        for key, child in value.items():
            reject_nonfinite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_nonfinite(child, f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        finite(value, label)


def validate_artifact(payload: dict[str, Any], path: Path) -> None:
    """Apply schema, protocol, provenance, arm, and finite-value checks."""
    reject_nonfinite(payload, str(path))
    if payload.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ValueError(f"{path}: unexpected artifact schema")
    if payload.get("development_only") is not True:
        raise ValueError(f"{path}: artifact is not development-only")
    if payload.get("confirmatory") is not False:
        raise ValueError(f"{path}: protocol matrix cannot be confirmatory")
    if payload.get("experiment_tier") != "development_protocol_sensitivity":
        raise ValueError(f"{path}: incorrect experiment tier")
    if payload.get("source_hashes") != source_hashes():
        raise ValueError(f"{path}: source hash drift")
    matrix = payload.get("matrix")
    if not isinstance(matrix, dict) or matrix.get("arms") != list(ARM_NAMES):
        raise ValueError(f"{path}: matrix arm drift")
    if matrix.get("evaluation_seed_range") != [600_000, 600_099]:
        raise ValueError(f"{path}: evaluation seed bank drift")
    if matrix.get("evaluation_episodes") != 100:
        raise ValueError(f"{path}: evaluation episode count drift")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARM_NAMES):
        raise ValueError(f"{path}: missing or unexpected arms")
    for name in ARM_NAMES:
        arm = arms[name]
        if arm.get("arm") != name:
            raise ValueError(f"{path}: arm label drift for {name}")
        config = arm.get("config", {})
        expected = {
            "dataset_trials": 250,
            "trial_horizon": 20,
            "train_steps": 10_000,
            "learning_rate": 1e-4,
            "target_tau": 0.005,
            "cql_alpha": 0.0,
            "q_width": 512,
            "q_depth": 4,
            "q_batch_norm": True,
            "state_noise_std": 0.05,
            "action_noise_std": 0.05,
            "eval_episodes": 100,
            "eval_seed_base": 600_000,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"{path}: {name} config drift in {key}")
        expected_semantics = "process" if name.startswith("process") else "observation"
        if config.get("noise_semantics") != expected_semantics:
            raise ValueError(f"{path}: {name} noise semantics drift")
        if config.get("stop_on_termination") != name.endswith("stop"):
            raise ValueError(f"{path}: {name} termination rule drift")
        clean_returns = arm.get("clean", {}).get("returns")
        noisy_returns = arm.get("noisy", {}).get("returns")
        if not isinstance(clean_returns, list) or len(clean_returns) != 100:
            raise ValueError(f"{path}: {name} clean return count drift")
        if not isinstance(noisy_returns, list) or len(noisy_returns) != 100:
            raise ValueError(f"{path}: {name} noisy return count drift")
        for index, value in enumerate(clean_returns):
            finite(value, f"{path}: {name} clean return {index}")
        for index, value in enumerate(noisy_returns):
            finite(value, f"{path}: {name} noisy return {index}")


def mean_sd_interval(values: list[float]) -> dict[str, float]:
    """Return mean, sample SD, and a two-sided 95% t interval."""
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    margin = float(stats.t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def paired_delta(values: list[float]) -> dict[str, float | int]:
    """Summarize ten paired seed-level differences with a 95% t interval."""
    result = mean_sd_interval(values)
    result["n"] = len(values)
    return result


def summarize(root: Path) -> dict[str, Any]:
    """Load exactly seeds 970--979 and summarize protocol-factor deltas."""
    paths = sorted(root.glob("protocol_sensitivity_seed_*.json"))
    if len(paths) != len(EXPECTED_SEEDS):
        raise ValueError(f"Expected exactly 10 artifacts, found {len(paths)}")
    payloads: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read {path}") from error
        validate_artifact(payload, path)
        payloads.append(payload)
    seeds = [payload["config"].get("seed") for payload in payloads]
    if sorted(seeds) != list(EXPECTED_SEEDS):
        raise ValueError(f"Expected seeds {EXPECTED_SEEDS}, found {sorted(seeds)}")
    reference = payloads[0]
    reference_config = {
        key: value
        for key, value in reference["config"].items()
        if key not in {"seed", "output_dir"}
    }
    reference_hashes = reference["source_hashes"]
    for payload in payloads[1:]:
        config = {
            key: value
            for key, value in payload["config"].items()
            if key not in {"seed", "output_dir"}
        }
        if config != reference_config or payload["source_hashes"] != reference_hashes:
            raise ValueError("Artifacts contain configuration or source drift")

    seed_means: dict[str, dict[str, dict[str, float]]] = {}
    for seed, payload in zip(seeds, payloads):
        seed_means[str(seed)] = {
            name: {
                "clean": mean_sd_interval(payload["arms"][name]["clean"]["returns"]),
                "noisy": mean_sd_interval(payload["arms"][name]["noisy"]["returns"]),
            }
            for name in ARM_NAMES
        }

    def deltas(left: str, right: str, condition: str) -> list[float]:
        return [
            seed_means[str(seed)][left][condition]["mean"]
            - seed_means[str(seed)][right][condition]["mean"]
            for seed in EXPECTED_SEEDS
        ]

    arm_summary = {
        name: {
            condition: mean_sd_interval(
                [
                    seed_means[str(seed)][name][condition]["mean"]
                    for seed in EXPECTED_SEEDS
                ]
            )
            for condition in ("clean", "noisy")
        }
        for name in ARM_NAMES
    }
    dataset_summary = {
        name: {
            "real_transitions": mean_sd_interval(
                [
                    float(payload["arms"][name]["dataset"]["real_transitions"])
                    for payload in payloads
                ]
            ),
            "post_failure_transitions": mean_sd_interval(
                [
                    float(payload["arms"][name]["dataset"]["post_failure_transitions"])
                    for payload in payloads
                ]
            ),
        }
        for name in ARM_NAMES
    }

    contrasts = {
        "process_vs_observation_stop": {
            condition: paired_delta(
                deltas("process_stop", "observation_stop", condition)
            )
            for condition in ("clean", "noisy")
        },
        "process_vs_observation_continue": {
            condition: paired_delta(
                deltas("process_continue", "observation_continue", condition)
            )
            for condition in ("clean", "noisy")
        },
        "stop_vs_continue_process": {
            condition: paired_delta(
                deltas("process_stop", "process_continue", condition)
            )
            for condition in ("clean", "noisy")
        },
        "stop_vs_continue_observation": {
            condition: paired_delta(
                deltas("observation_stop", "observation_continue", condition)
            )
            for condition in ("clean", "noisy")
        },
    }
    report = {
        "artifact_schema": "ctrl-cartpole-protocol-sensitivity-summary-v1",
        "development_only": True,
        "confirmatory": False,
        "seeds": list(EXPECTED_SEEDS),
        "per_seed_arm_means": seed_means,
        "arm_summary": arm_summary,
        "dataset_summary": dataset_summary,
        "contrasts": contrasts,
        "interpretation": (
            "Paired protocol-factor diagnostics only. These results explain how "
            "dataset semantics can change scores; they do not establish a CTRL "
            "algorithm improvement or confirm the original paper's result."
        ),
    }
    return report


def main() -> None:
    """Validate artifacts and write the development summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    output = args.output or args.root / "protocol_sensitivity_summary.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
