"""Integrity and eligibility checks for CartPole revision artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_SCHEMA = "ctrl-cartpole-revision-v4"
TIER_GENERATOR = {
    "ctrl_bicogan_reproduction": "monotonic_bicogan",
    "unconstrained_bicogan_ablation": "unconstrained",
    "triangular_flow_extension": "triangular",
}
CONDITIONS = ("random", "real", "fresh_noise", "oracle_cf", "learned_cf")
METRICS = ("clean", "noisy")


def _path(payload: dict[str, Any], dotted: str, source: Path) -> Any:
    """Read a required dotted field and include its artifact in errors."""
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"{source}: missing field {dotted}")
        value = value[part]
    return value


def _finite(value: Any, description: str) -> float:
    """Return a finite numeric scalar."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"{description} must be finite")
    return number


def _finite_vector(value: Any, description: str) -> np.ndarray:
    """Return a non-empty finite numeric vector from a scalar or JSON list."""
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{description} must be a non-empty finite vector")
    return array.reshape(-1)


def _quality_gate(run: dict[str, Any], source: Path) -> dict[str, Any]:
    """Evaluate the model-quality criteria without hiding failed development runs."""
    external = _path(
        run,
        "bicogan.counterfactual_diagnostics.external_validation",
        source,
    )
    fresh = _path(
        run,
        "bicogan.counterfactual_diagnostics.fresh_noise_external_validation",
        source,
    )
    learned_mse = _finite(external.get("normalized_mse"), f"{source}: learned MSE")
    fresh_mse = _finite(fresh.get("normalized_mse"), f"{source}: fresh MSE")
    terminal = _finite(
        external.get("terminal_disagreement"),
        f"{source}: terminal disagreement",
    )
    diagnostics = _path(run, "bicogan.diagnostics", source)
    latent_value = diagnostics.get(
        "latent_std_by_dimension", diagnostics.get("latent_std")
    )
    latent = _finite_vector(latent_value, f"{source}: latent standard deviation")
    checks = {
        "learned_mse_at_least_20pct_below_fresh_mse": learned_mse < 0.8 * fresh_mse,
        "terminal_disagreement_below_0.05": terminal < 0.05,
        "latent_std_each_in_[0.5,2.0]": bool(np.all((latent >= 0.5) & (latent <= 2.0))),
        "latent_std_aggregate_in_[0.5,2.0]": bool(0.5 <= float(np.mean(latent)) <= 2.0),
    }
    tier = _path(run, "config.experiment_tier", source)
    if tier != "triangular_flow_extension":
        action_mse = _finite(
            diagnostics.get("action_reconstruction_mse"),
            f"{source}: action reconstruction MSE",
        )
        central_mse = _finite(
            diagnostics.get("central_action_baseline_mse"),
            f"{source}: central-action baseline MSE",
        )
        checks["action_mse_at_least_10pct_below_central_baseline"] = (
            action_mse < 0.9 * central_mse
        )
    return {
        "gate_passed": bool(all(checks.values())),
        "checks": checks,
        "learned_external_normalized_mse": learned_mse,
        "fresh_external_normalized_mse": fresh_mse,
        "terminal_disagreement": terminal,
        "latent_std": latent.tolist(),
        "latent_std_aggregate": float(np.mean(latent)),
    }


def validate_run(run: dict[str, Any], source: Path) -> dict[str, Any]:
    """Validate one v4 artifact and return its quality-gate report."""
    if run.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ValueError(
            f"{source}: unsupported artifact schema; expected {ARTIFACT_SCHEMA}"
        )
    config = run.get("config")
    if not isinstance(config, dict):
        raise TypeError(f"{source}: config must be an object")
    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"{source}: config.seed must be an integer")
    tier = config.get("experiment_tier")
    generator = config.get("bicogan_generator")
    if tier not in TIER_GENERATOR:
        raise ValueError(f"{source}: unsupported experiment_tier {tier!r}")
    if generator != TIER_GENERATOR[tier]:
        raise ValueError(
            f"{source}: tier {tier!r} requires generator {TIER_GENERATOR[tier]!r}, "
            f"got {generator!r}"
        )
    if run.get("experiment_tier") != tier:
        raise ValueError(f"{source}: top-level experiment tier does not match config")
    command = run.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"{source}: command provenance is missing")
    git = run.get("git")
    if not isinstance(git, dict) or "commit" not in git or "dirty" not in git:
        raise ValueError(f"{source}: Git provenance is missing")
    source_hash = run.get("source_sha256")
    if not isinstance(source_hash, str) or not source_hash:
        raise ValueError(f"{source}: source_sha256 is missing")
    for condition in CONDITIONS:
        for metric in METRICS:
            _finite(
                _path(run, f"{condition}.{metric}.mean", source), f"{source}: return"
            )
            _finite_vector(
                _path(run, f"{condition}.{metric}.returns", source),
                f"{source}: episode returns",
            )

    dataset = run.get("dataset")
    if not isinstance(dataset, dict):
        raise TypeError(f"{source}: dataset metadata is missing")
    noise_semantics = config.get("noise_semantics")
    if noise_semantics not in {"process", "observation"}:
        raise ValueError(f"{source}: unsupported noise semantics {noise_semantics!r}")
    if dataset.get("noise_semantics") != noise_semantics:
        raise ValueError(f"{source}: dataset and config noise semantics differ")
    if dataset.get("terminal_label_rule") != "pre_noise_next_state":
        raise ValueError(f"{source}: unsupported terminal-label rule")
    for field in ("post_failure_transitions", "validation_post_failure_transitions"):
        if _finite(_path(run, f"dataset.{field}", source), f"{source}: {field}") != 0:
            raise ValueError(f"{source}: {field} must be zero")
    validation_seed = _finite(
        dataset.get("validation_seed"), f"{source}: validation_seed"
    )
    if int(validation_seed) == seed:
        raise ValueError(f"{source}: validation seed must differ from training seed")
    if (
        _finite(
            dataset.get("validation_real_transitions"),
            f"{source}: validation transitions",
        )
        <= 0
    ):
        raise ValueError(f"{source}: independent validation dataset is empty")
    if (
        _path(run, "bicogan.diagnostics.validation_source", source)
        != "independent_dataset"
    ):
        raise ValueError(f"{source}: validation_source is not independent_dataset")
    validation_ids = _path(run, "bicogan.diagnostics.validation_trial_ids", source)
    if not isinstance(validation_ids, list) or not validation_ids:
        raise ValueError(f"{source}: independent validation trial IDs are missing")
    train_ids = dataset.get("trial_ids")
    if isinstance(train_ids, list) and set(train_ids).intersection(validation_ids):
        raise ValueError(f"{source}: training and validation trial IDs overlap")
    quality = _quality_gate(run, source)
    run["_quality_gate"] = quality
    run["_path"] = str(source)
    return quality


def load_and_validate(
    root: Path,
    *,
    expected_seeds: Iterable[int] | None = None,
    expected_count: int | None = None,
    development: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load all artifacts and enforce integrity, seed, and quality contracts.

    Development mode still enforces structural integrity but records failed
    model-quality gates and never labels the returned report confirmatory.
    """
    paths = sorted(root.rglob("reproduction_seed_*.json"))
    if not paths:
        raise FileNotFoundError(f"No reproduction_seed_*.json files under {root}")
    by_seed: dict[int, dict[str, Any]] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read artifact {path}") from error
        seed_value = payload.get("config", {}).get("seed")
        if not isinstance(seed_value, int) or isinstance(seed_value, bool):
            raise TypeError(f"{path}: missing integer config.seed")
        if seed_value in by_seed:
            raise ValueError(f"Duplicate artifact for seed {seed_value}: {path}")
        payload["_path"] = str(path)
        by_seed[seed_value] = payload
    runs = [by_seed[seed] for seed in sorted(by_seed)]
    for run in runs:
        validate_run(run, Path(run["_path"]))
    seeds = set(by_seed)
    if expected_count is not None and len(runs) != expected_count:
        raise ValueError(f"Expected {expected_count} seeds, found {len(runs)}")
    if expected_seeds is not None:
        expected_list = [int(seed) for seed in expected_seeds]
        if len(expected_list) != len(set(expected_list)):
            raise ValueError("Expected seed set contains duplicates")
        required = set(expected_list)
        if seeds != required:
            raise ValueError(
                f"Expected seed set {sorted(required)}, found {sorted(seeds)}"
            )
    reference = runs[0]
    for run in runs:
        software = run.get("software")
        if not isinstance(software, dict) or not software:
            raise ValueError(f"{run['_path']}: software metadata is missing")
    reference_config = {
        k: v for k, v in reference["config"].items() if k not in {"seed", "output_dir"}
    }
    for run in runs[1:]:
        config = {
            k: v for k, v in run["config"].items() if k not in {"seed", "output_dir"}
        }
        if config != reference_config:
            raise ValueError("Artifacts contain mismatched configs (non-seed fields)")
        if run.get("source_sha256") != reference.get("source_sha256"):
            raise ValueError("Artifacts contain multiple source hashes")
        if run.get("software") != reference.get("software"):
            raise ValueError("Artifacts contain multiple software environments")
    gate_passed = all(run["_quality_gate"]["gate_passed"] for run in runs)
    if not development and reference_config.get("noise_semantics") != "process":
        raise ValueError("Confirmatory artifacts require process-noise semantics")
    if not development and not gate_passed:
        failed = [
            run["config"]["seed"]
            for run in runs
            if not run["_quality_gate"]["gate_passed"]
        ]
        raise ValueError(f"Model-quality gates failed for seeds: {failed}")
    report = {
        "confirmatory": bool(not development and gate_passed),
        "development": bool(development),
        "gate_passed": bool(gate_passed and not development),
        "seeds": sorted(seeds),
        "artifact_schema": ARTIFACT_SCHEMA,
    }
    return runs, report
