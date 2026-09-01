"""Validate and summarize the frozen 30-seed coupling-control experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from scripts.revision.cartpole_coupling_control import (
    ARMS,
    EXPECTED_SEEDS,
    SCHEMA,
    load_manifest,
    sha256_file,
    source_digest,
    source_hashes,
    source_manifest_digest,
)

METRICS = ("clean", "noisy")
EXPECTED_CONTRASTS = (
    ("oracle_cf", "fresh_shared", "clean"),
    ("oracle_cf", "fresh_shared", "noisy"),
    ("fresh_shared", "fresh_independent", "clean"),
    ("fresh_shared", "fresh_independent", "noisy"),
)


def require_finite_json(value: Any, label: str) -> None:
    """Reject non-finite numeric values anywhere in a JSON artifact."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} contains a non-finite value")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            require_finite_json(child, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite_json(child, f"{label}.{key}")


def finite_array(value: Any, label: str) -> np.ndarray:
    """Convert a non-empty sequence to a finite one-dimensional array."""
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be non-empty and finite")
    return array


def t_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Return a two-sided Student-t interval for a seed-level mean."""
    if len(values) < 2:
        raise ValueError("At least two seed values are required for an interval")
    alpha = 1.0 - confidence
    half = stats.t.ppf(1.0 - alpha / 2.0, len(values) - 1)
    half *= values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - half), float(values.mean() + half)


def bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int,
    confidence: float,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval over training-seed means."""
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    tail = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(low), float(high)


def _finite_p(value: float) -> float:
    """Convert a numerical-test p-value to a JSON-safe value."""
    return float(value) if np.isfinite(value) else 1.0


def sign_randomization_p(
    delta: np.ndarray, rng: np.random.Generator, samples: int
) -> float:
    """Return a two-sided paired sign-randomization p-value."""
    observed = abs(float(delta.mean()))
    signs = rng.choice((-1.0, 1.0), size=(samples, len(delta)))
    randomized = np.abs((signs * delta).mean(axis=1))
    return float((np.count_nonzero(randomized >= observed) + 1) / (samples + 1))


def classification(
    ci95_low: float,
    ci95_high: float,
    ci90_low: float,
    ci90_high: float,
    threshold: float,
) -> tuple[str, bool]:
    """Classify a signed effect symmetrically around zero and threshold."""
    equivalent = ci90_low >= -threshold and ci90_high <= threshold
    if equivalent:
        return "practically_equivalent", True
    if ci95_low > threshold:
        return "practically_meaningful_benefit", False
    if ci95_high < -threshold:
        return "practically_meaningful_harm", False
    if ci95_low > 0.0:
        return "positive_but_below_practical_threshold", False
    if ci95_high < 0.0:
        return "negative_but_below_practical_threshold", False
    return "inconclusive", False


def paired_statistics(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    bootstrap_samples: int,
    randomization_samples: int,
    practical_threshold: float,
) -> dict[str, Any]:
    """Compute paired seed-level uncertainty, tests, and practical classification."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Paired statistics require equally sized seed arrays")
    delta = left - right
    t_low, t_high = t_interval(delta)
    boot_low, boot_high = bootstrap_interval(
        delta, rng, bootstrap_samples, confidence=0.95
    )
    boot90_low, boot90_high = bootstrap_interval(
        delta, rng, bootstrap_samples, confidence=0.90
    )
    try:
        wilcoxon = _finite_p(float(stats.wilcoxon(delta).pvalue))
    except ValueError:
        wilcoxon = 1.0
    effect, equivalent = classification(
        boot_low, boot_high, boot90_low, boot90_high, practical_threshold
    )
    standard_deviation = float(delta.std(ddof=1))
    return {
        "n_training_seeds": len(delta),
        "mean_delta": float(delta.mean()),
        "sd_delta": standard_deviation,
        "median_delta": float(np.median(delta)),
        "iqr_delta": float(np.percentile(delta, 75) - np.percentile(delta, 25)),
        "cohens_d_paired": (
            float(delta.mean() / standard_deviation) if standard_deviation > 0 else 0.0
        ),
        "t_ci95_low": t_low,
        "t_ci95_high": t_high,
        "bootstrap_ci95_low": boot_low,
        "bootstrap_ci95_high": boot_high,
        "bootstrap_ci90_low": boot90_low,
        "bootstrap_ci90_high": boot90_high,
        "practical_effect_threshold": practical_threshold,
        "effect_classification": effect,
        "practical_equivalence": equivalent,
        "paired_t_p": _finite_p(float(stats.ttest_rel(left, right).pvalue)),
        "wilcoxon_p": wilcoxon,
        "sign_randomization_p": sign_randomization_p(delta, rng, randomization_samples),
        "positive_seeds": int(np.count_nonzero(delta > 0)),
        "negative_seeds": int(np.count_nonzero(delta < 0)),
        "ties": int(np.count_nonzero(delta == 0)),
        "seed_deltas": [float(value) for value in delta],
    }


def describe(
    values: np.ndarray,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Summarize an arm's seed-level evaluation means and uncertainty."""
    low, high = t_interval(values)
    boot_low, boot_high = bootstrap_interval(
        values, rng, bootstrap_samples, confidence=0.95
    )
    boot90_low, boot90_high = bootstrap_interval(
        values, rng, bootstrap_samples, confidence=0.90
    )
    return {
        "n_training_seeds": len(values),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "t_ci95_low": low,
        "t_ci95_high": high,
        "bootstrap_ci95_low": boot_low,
        "bootstrap_ci95_high": boot_high,
        "bootstrap_ci90_low": boot90_low,
        "bootstrap_ci90_high": boot90_high,
        "seed_means": [float(value) for value in values],
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm step-down family-wise error correction."""
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.ones(len(values), dtype=np.float64)
    order = np.argsort(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (len(values) - rank)))
        adjusted[index] = running
    return [float(value) for value in adjusted]


def _contrast_key(left: str, right: str, metric: str) -> str:
    """Return the stable JSON key for one registered contrast."""
    return f"{left}_minus_{right}_{metric}"


def validate_artifacts(
    root: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Require exactly the frozen seed set and all artifact provenance fields."""
    manifest = load_manifest(manifest_path)
    paths = sorted(root.glob("coupling_seed_*.json"))
    if len(paths) != len(EXPECTED_SEEDS):
        raise ValueError(f"Expected exactly 30 artifacts, found {len(paths)}")
    expected_manifest = sha256_file(manifest_path)
    expected_source = source_digest()
    expected_sources = source_hashes()
    expected_combined = source_manifest_digest(manifest_path)
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    reference_config: dict[str, Any] | None = None
    reference_software: dict[str, Any] | None = None
    reference_git: dict[str, Any] | None = None
    for path in paths:
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read {path}") from error
        require_finite_json(record, str(path))
        if record.get("artifact_schema") != SCHEMA:
            raise ValueError(f"{path}: wrong artifact schema")
        config = record.get("config")
        if not isinstance(config, dict) or not isinstance(config.get("seed"), int):
            raise TypeError(f"{path}: missing integer config.seed")
        seed = config["seed"]
        if seed in seen:
            raise ValueError(f"Duplicate seed {seed}")
        seen.add(seed)
        if seed not in EXPECTED_SEEDS:
            raise ValueError(f"Unexpected seed {seed}")
        for field, expected in manifest["config"].items():
            if config.get(field) != expected:
                raise ValueError(f"{path}: configuration drift for {field}")
        if record.get("manifest_sha256") != expected_manifest:
            raise ValueError(f"{path}: manifest hash mismatch")
        if record.get("source_sha256") != expected_source:
            raise ValueError(f"{path}: source digest mismatch")
        if record.get("source_hashes") != expected_sources:
            raise ValueError(f"{path}: source hash mismatch")
        if record.get("source_manifest_sha256") != expected_combined:
            raise ValueError(f"{path}: combined source hash mismatch")
        if not isinstance(record.get("command"), list) or not record["command"]:
            raise ValueError(f"{path}: command provenance missing")
        git = record.get("git")
        software = record.get("software")
        if not isinstance(git, dict) or set(git) != {"commit", "dirty"}:
            raise ValueError(f"{path}: Git provenance missing")
        if not isinstance(software, dict) or not software:
            raise ValueError(f"{path}: software provenance missing")
        if reference_git is None:
            reference_git = git
        elif git != reference_git:
            raise ValueError("Artifacts contain Git provenance drift")
        if reference_software is None:
            reference_software = software
        elif software != reference_software:
            raise ValueError("Artifacts contain software drift")
        config_without_seed = {
            key: value
            for key, value in config.items()
            if key not in {"seed", "output_dir"}
        }
        if reference_config is None:
            reference_config = config_without_seed
        elif config_without_seed != reference_config:
            raise ValueError("Artifacts contain configuration drift")
        dataset = record.get("dataset")
        if not isinstance(dataset, dict):
            raise TypeError(f"{path}: dataset metadata missing")
        if (
            dataset.get("noise_semantics") != "process"
            or dataset.get("terminal_label_rule") != "pre_noise_next_state"
            or dataset.get("stop_on_termination") is not True
            or dataset.get("post_failure_transitions") != 0
            or dataset.get("factual_alignment_with_generate_paired_dataset") is not True
        ):
            raise ValueError(f"{path}: dataset protocol drift")
        real_count = dataset.get("real_transitions")
        expected_cf_count = real_count * 10 if isinstance(real_count, int) else None
        if expected_cf_count is None or any(
            dataset.get(field) != expected_cf_count
            for field in (
                "fresh_independent_transitions",
                "fresh_shared_transitions",
                "oracle_cf_transitions",
            )
        ):
            raise ValueError(f"{path}: counterfactual count drift")
        diagnostics = record.get("diagnostics", {}).get("coupling", {})
        required_diagnostics = {
            "factual_alignment": True,
            "alternative_actions_per_transition": 10,
            "within_transition_action_noise_reuse_max_abs": 0.0,
            "within_transition_state_noise_reuse_max_abs": 0.0,
        }
        for key, expected in required_diagnostics.items():
            if diagnostics.get(key) != expected:
                raise ValueError(f"{path}: coupling diagnostic {key} failed")
        if diagnostics.get("shared_noise_draws") != dataset.get("real_transitions"):
            raise ValueError(f"{path}: shared noise count does not match factual count")
        if diagnostics.get("fresh_independent_noise_draws") != expected_cf_count:
            raise ValueError(f"{path}: independent noise count does not match CF count")
        if (
            diagnostics.get("shared_action_noise_unique_count", 0) < 2
            or diagnostics.get("shared_state_noise_unique_count", 0) < 2
            or diagnostics.get(
                "fresh_shared_vs_independent_exact_next_state_fraction", 1.0
            )
            >= 1.0
        ):
            raise ValueError(f"{path}: coupling noise diversity diagnostic failed")
        arms = record.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(ARMS):
            raise ValueError(f"{path}: arm set mismatch")
        for arm in ARMS:
            for metric in METRICS:
                result = arms[arm].get(metric)
                if not isinstance(result, dict):
                    raise TypeError(f"{path}: missing {arm}/{metric}")
                returns = finite_array(result.get("returns"), f"{path}: {arm}/{metric}")
                if len(returns) != manifest["config"]["eval_episodes"]:
                    raise ValueError(f"{path}: wrong evaluation episode count")
                mean = float(result.get("mean"))
                if not np.isfinite(mean) or not np.isclose(mean, returns.mean()):
                    raise ValueError(f"{path}: invalid {arm}/{metric} mean")
        records.append(record)
    if seen != set(EXPECTED_SEEDS):
        raise ValueError(f"Missing seeds: {sorted(set(EXPECTED_SEEDS) - seen)}")
    return sorted(records, key=lambda item: item["config"]["seed"]), manifest


def summarize(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Validate artifacts and produce the complete publication-facing summary."""
    records, manifest = validate_artifacts(root, manifest_path)
    settings = manifest["statistics"]
    expected_contrast_names = [
        _contrast_key(*contrast) for contrast in EXPECTED_CONTRASTS
    ]
    if settings["contrasts"] != expected_contrast_names:
        raise ValueError("Manifest contrasts do not match the frozen analysis")
    expected_primary = expected_contrast_names[0]
    if settings["primary"] != expected_primary:
        raise ValueError("Manifest primary does not match the frozen analysis")
    values = {
        arm: {
            metric: np.asarray(
                [record["arms"][arm][metric]["mean"] for record in records],
                dtype=np.float64,
            )
            for metric in METRICS
        }
        for arm in ARMS
    }
    rng = np.random.default_rng(20260901)
    arm_summary = {
        arm: {
            metric: describe(values[arm][metric], rng, settings["bootstrap_samples"])
            for metric in METRICS
        }
        for arm in ARMS
    }
    contrasts: dict[str, dict[str, Any]] = {}
    for left, right, metric in EXPECTED_CONTRASTS:
        key = _contrast_key(left, right, metric)
        contrasts[key] = paired_statistics(
            values[left][metric],
            values[right][metric],
            rng,
            settings["bootstrap_samples"],
            settings["randomization_samples"],
            float(settings["practical_effect_thresholds"][metric]),
        )
    for test_name in ("paired_t_p", "wilcoxon_p", "sign_randomization_p"):
        adjusted = holm_adjust(
            [contrasts[key][test_name] for key in expected_contrast_names]
        )
        for key, value in zip(expected_contrast_names, adjusted):
            contrasts[key][f"{test_name}_holm"] = value
    return {
        "artifact_schema": "ctrl-cartpole-coupling-control-summary-v1",
        "seeds": list(EXPECTED_SEEDS),
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": source_manifest_digest(manifest_path),
        "config": manifest["config"],
        "registered_primary": settings["primary"],
        "registered_contrasts": settings["contrasts"],
        "arms": arm_summary,
        "contrasts": contrasts,
        "interpretation": {
            "unit_of_replication": "training seed; evaluation episodes are nested observations",
            "primary_metric": "clean return",
            "practical_thresholds": settings["practical_effect_thresholds"],
            "classification_rule": "bootstrap 90% equivalence is checked before signed 95% benefit/harm",
        },
    }


def main() -> None:
    """Validate a completed result directory and print/write its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("coupling_control_manifest.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.root, args.manifest)
    output = args.output or args.root / "coupling_control_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
