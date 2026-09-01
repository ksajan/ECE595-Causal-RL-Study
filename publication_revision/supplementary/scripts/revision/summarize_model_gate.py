#!/usr/bin/env python3
"""Validate exactly five development model-gate artifacts and summarize gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.revision.cartpole_model_gate import ARTIFACT_SCHEMA, source_hashes

EXPECTED_SEEDS = (960, 961, 962, 963, 964)


def finite(value: Any, label: str) -> float:
    """Convert a scalar to finite float or reject the artifact."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def reject_nonfinite(value: Any, label: str) -> None:
    """Reject nonfinite numeric values recursively anywhere in an artifact."""
    if isinstance(value, dict):
        for key, child in value.items():
            reject_nonfinite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_nonfinite(child, f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        finite(value, label)


def validate_artifact(payload: dict[str, Any], path: Path) -> dict[str, bool]:
    """Apply every registered model-quality gate to one artifact."""
    reject_nonfinite(payload, str(path))
    if payload.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ValueError(f"{path}: unexpected artifact schema")
    if payload.get("development_only") is not True:
        raise ValueError(f"{path}: model-gate artifacts must be development-only")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise TypeError(f"{path}: missing config")
    if config.get("experiment_tier") != "ctrl_bicogan_reproduction":
        raise ValueError(f"{path}: incorrect experiment tier")
    if config.get("bicogan_generator") != "monotonic_bicogan":
        raise ValueError(f"{path}: incorrect generator")
    if config.get("noise_semantics") != "process":
        raise ValueError(f"{path}: model gate requires process noise")
    if (
        config.get("dataset_trials") != 250
        or config.get("validation_dataset_trials") != 50
    ):
        raise ValueError(f"{path}: dataset protocol drift")
    if (
        config.get("trial_horizon") != 20
        or config.get("stop_on_termination") is not True
    ):
        raise ValueError(f"{path}: trial protocol drift")
    hashes = payload.get("source_hashes")
    if hashes != source_hashes():
        raise ValueError(f"{path}: source/configuration hash drift")
    dataset = payload.get("dataset", {})
    if dataset.get("terminal_label_rule") != "pre_noise_next_state":
        raise ValueError(f"{path}: terminal rule drift")
    if dataset.get("noise_semantics") != "process":
        raise ValueError(f"{path}: dataset noise semantics drift")
    if (
        finite(dataset.get("post_failure_transitions"), f"{path}: post-failure count")
        != 0
    ):
        raise ValueError(f"{path}: post-failure transitions present")
    if (
        finite(
            dataset.get("validation_post_failure_transitions"),
            f"{path}: validation post-failure count",
        )
        != 0
    ):
        raise ValueError(f"{path}: validation post-failure transitions present")
    bicogan = payload.get("bicogan", {})
    diagnostics = bicogan.get("diagnostics", {})
    cf = bicogan.get("counterfactual_diagnostics", {})
    external = cf.get("external_validation", {})
    fresh = cf.get("fresh_noise_external_validation", {})
    learned_mse = finite(external.get("normalized_mse"), f"{path}: learned MSE")
    fresh_mse = finite(fresh.get("normalized_mse"), f"{path}: fresh MSE")
    terminal = finite(
        external.get("terminal_disagreement"), f"{path}: terminal disagreement"
    )
    latent = np.asarray(diagnostics.get("latent_std_by_dimension"), dtype=float)
    if latent.shape != (4,) or not np.isfinite(latent).all():
        raise ValueError(f"{path}: latent standard deviations are invalid")
    action_mse = finite(
        diagnostics.get("action_reconstruction_mse"), f"{path}: action MSE"
    )
    central_mse = finite(
        diagnostics.get("central_action_baseline_mse"), f"{path}: central action MSE"
    )
    return {
        "learned_mse_at_least_20pct_below_fresh_mse": learned_mse < 0.8 * fresh_mse,
        "terminal_disagreement_below_0.05": terminal < 0.05,
        "latent_std_each_in_[0.5,2.0]": bool(np.all((latent >= 0.5) & (latent <= 2.0))),
        "action_mse_at_least_10pct_below_central_baseline": action_mse
        < 0.9 * central_mse,
    }


def extract_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract publication-facing held-out diagnostics from one artifact."""
    bicogan = payload["bicogan"]
    diagnostics = bicogan["diagnostics"]
    counterfactuals = bicogan["counterfactual_diagnostics"]
    external = counterfactuals["external_validation"]
    fresh = counterfactuals["fresh_noise_external_validation"]
    learned_mse = finite(external["normalized_mse"], "learned MSE")
    fresh_mse = finite(fresh["normalized_mse"], "fresh MSE")
    action_mse = finite(diagnostics["action_reconstruction_mse"], "action MSE")
    central_mse = finite(
        diagnostics["central_action_baseline_mse"], "central-action MSE"
    )
    return {
        "learned_cf_normalized_mse": learned_mse,
        "fresh_noise_normalized_mse": fresh_mse,
        "learned_to_fresh_mse_ratio": learned_mse / fresh_mse,
        "terminal_disagreement": finite(
            external["terminal_disagreement"], "terminal disagreement"
        ),
        "latent_std_by_dimension": [
            float(value) for value in diagnostics["latent_std_by_dimension"]
        ],
        "action_reconstruction_mse": action_mse,
        "central_action_baseline_mse": central_mse,
        "action_to_baseline_mse_ratio": action_mse / central_mse,
    }


def describe(values: list[float]) -> dict[str, float]:
    """Describe a metric across the five development seeds."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize(root: Path) -> dict[str, Any]:
    """Load exactly the registered five seeds and return a mechanical report."""
    paths = sorted(root.glob("model_gate_seed_*.json"))
    if len(paths) != len(EXPECTED_SEEDS):
        raise ValueError(
            f"Expected exactly five model-gate artifacts, found {len(paths)}"
        )
    payloads: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read {path}") from error
        payloads.append(payload)
    seeds = [payload.get("config", {}).get("seed") for payload in payloads]
    if sorted(seeds) != list(EXPECTED_SEEDS):
        raise ValueError(f"Expected seeds {EXPECTED_SEEDS}, found {sorted(seeds)}")
    reference = payloads[0]
    reference_config = {
        k: v for k, v in reference["config"].items() if k not in {"seed", "output_dir"}
    }
    reference_hashes = reference["source_hashes"]
    gates = {
        str(seed): validate_artifact(payload, path)
        for seed, payload, path in zip(seeds, payloads, paths)
    }
    metrics = {
        str(seed): extract_metrics(payload) for seed, payload in zip(seeds, payloads)
    }
    for payload in payloads[1:]:
        config = {
            k: v
            for k, v in payload["config"].items()
            if k not in {"seed", "output_dir"}
        }
        if config != reference_config or payload["source_hashes"] != reference_hashes:
            raise ValueError("Artifacts contain configuration or source drift")
    all_passed = all(all(checks.values()) for checks in gates.values())
    scalar_metric_names = (
        "learned_cf_normalized_mse",
        "fresh_noise_normalized_mse",
        "learned_to_fresh_mse_ratio",
        "terminal_disagreement",
        "action_reconstruction_mse",
        "central_action_baseline_mse",
        "action_to_baseline_mse_ratio",
    )
    report = {
        "artifact_schema": "ctrl-cartpole-model-gate-summary-v1",
        "development_only": True,
        "confirmatory": False,
        "seeds": list(EXPECTED_SEEDS),
        "all_registered_gates_passed": all_passed,
        "per_seed_gates": gates,
        "per_seed_metrics": metrics,
        "metric_summary": {
            name: describe([seed_metrics[name] for seed_metrics in metrics.values()])
            for name in scalar_metric_names
        },
        "interpretation": (
            "Eligible for a learned-CF downstream pilot only if every seed passes; "
            "this report is never confirmatory evidence."
        ),
    }
    return report


def main() -> None:
    """Validate and write a model-gate summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    output = args.output or args.root / "model_gate_summary.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
