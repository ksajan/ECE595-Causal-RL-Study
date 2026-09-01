#!/usr/bin/env python3
"""Validate and statistically summarize the frozen oracle-CF experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from scripts.revision.cartpole_oracle_confirmatory import (
    SCHEMA,
    load_manifest,
    sha256_file,
    source_digest,
    source_manifest_digest,
)

SEEDS = tuple(range(1000, 1030))
ARMS = ("random", "real", "fresh_noise", "oracle_cf")
METRICS = ("clean", "noisy")
CONTRASTS = (
    ("oracle_cf", "fresh_noise", "clean"),
    ("oracle_cf", "fresh_noise", "noisy"),
    ("oracle_cf", "real", "clean"),
    ("fresh_noise", "real", "clean"),
)


def finite_array(value: Any, label: str) -> np.ndarray:
    """Convert a non-empty JSON sequence to a finite one-dimensional array."""
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be non-empty and finite")
    return array


def require_finite_json(value: Any, label: str) -> None:
    """Reject non-finite numeric values anywhere in an artifact payload."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} contains a non-finite value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_json(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, f"{label}.{key}")


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    """Return the two-sided 95 percent Student-t interval for a mean."""
    if len(values) < 2:
        return float("nan"), float("nan")
    half = (
        stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    return float(values.mean() - half), float(values.mean() + half)


def bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval over training-seed deltas."""
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    tail = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(low), float(high)


def sign_randomization_p(
    delta: np.ndarray, rng: np.random.Generator, samples: int
) -> float:
    """Return a two-sided paired sign-randomization p-value."""
    observed = abs(float(delta.mean()))
    signs = rng.choice((-1.0, 1.0), size=(samples, len(delta)))
    randomized = np.abs((signs * delta).mean(axis=1))
    return float((np.count_nonzero(randomized >= observed) + 1) / (samples + 1))


def paired_statistics(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    bootstrap_samples: int,
    randomization_samples: int,
    practical_threshold: float,
) -> dict[str, Any]:
    """Compute paired effect statistics without treating episodes as replicates."""
    delta = left - right
    t_low, t_high = confidence_interval(delta)
    boot_low, boot_high = bootstrap_interval(delta, rng, bootstrap_samples)
    boot90_low, boot90_high = bootstrap_interval(
        delta, rng, bootstrap_samples, confidence=0.90
    )
    try:
        wilcoxon_p = float(stats.wilcoxon(delta).pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    practically_equivalent = (
        boot90_low >= -practical_threshold and boot90_high <= practical_threshold
    )
    if boot_low > practical_threshold:
        directional = "practically_meaningful_benefit"
    elif boot_high < -practical_threshold:
        directional = "practically_meaningful_harm"
    elif boot_low > 0.0:
        directional = "positive_but_below_practical_threshold"
    elif boot_high < 0.0:
        directional = "negative_but_below_practical_threshold"
    else:
        directional = "no_detected_direction"
    classification = "practically_equivalent" if practically_equivalent else directional
    if classification == "no_detected_direction":
        classification = "inconclusive"
    return {
        "n_training_seeds": len(delta),
        "mean_delta": float(delta.mean()),
        "sd_delta": float(delta.std(ddof=1)),
        "median_delta": float(np.median(delta)),
        "iqr_delta": float(np.percentile(delta, 75) - np.percentile(delta, 25)),
        "cohens_d_paired": float(delta.mean() / delta.std(ddof=1))
        if delta.std(ddof=1) > 0
        else None,
        "t_ci95_low": t_low,
        "t_ci95_high": t_high,
        "bootstrap_ci95_low": boot_low,
        "bootstrap_ci95_high": boot_high,
        "bootstrap_ci90_low": boot90_low,
        "bootstrap_ci90_high": boot90_high,
        "practical_effect_threshold": practical_threshold,
        "directional_effect_classification": directional,
        "practical_equivalence": practically_equivalent,
        "effect_classification": classification,
        "paired_t_p": float(stats.ttest_rel(left, right).pvalue),
        "wilcoxon_p": wilcoxon_p,
        "sign_randomization_p": sign_randomization_p(delta, rng, randomization_samples),
        "positive_seeds": int(np.count_nonzero(delta > 0)),
        "negative_seeds": int(np.count_nonzero(delta < 0)),
        "ties": int(np.count_nonzero(delta == 0)),
        "seed_deltas": [float(value) for value in delta],
    }


def describe(values: np.ndarray) -> dict[str, Any]:
    """Summarize the distribution of per-seed evaluation means."""
    low, high = confidence_interval(values)
    return {
        "n_training_seeds": len(values),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "t_ci95_low": low,
        "t_ci95_high": high,
        "seed_means": [float(value) for value in values],
    }


def validate_artifacts(
    root: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Require the exact manifest seed set and immutable artifact metadata."""
    manifest = load_manifest(manifest_path)
    paths = sorted(root.glob("oracle_seed_*.json"))
    if len(paths) != len(SEEDS):
        raise ValueError(f"Expected exactly 30 artifacts, found {len(paths)}")
    expected_source = source_digest()
    expected_combined = source_manifest_digest(manifest_path)
    expected_manifest = sha256_file(manifest_path)
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    reference_config: dict[str, Any] | None = None
    reference_software: dict[str, Any] | None = None
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
        if seed not in SEEDS:
            raise ValueError(f"Unexpected seed {seed}")
        expected_config = dict(manifest["config"])
        for field, value in expected_config.items():
            if config.get(field) != value:
                raise ValueError(f"{path}: configuration drift for {field}")
        if record.get("manifest_sha256") != expected_manifest:
            raise ValueError(f"{path}: manifest hash mismatch")
        if record.get("source_sha256") != expected_source:
            raise ValueError(f"{path}: source hash mismatch")
        if record.get("source_manifest_sha256") != expected_combined:
            raise ValueError(f"{path}: combined source hash mismatch")
        command = record.get("command")
        git = record.get("git")
        software = record.get("software")
        if not isinstance(command, list) or not command:
            raise ValueError(f"{path}: command provenance missing")
        if not isinstance(git, dict) or "commit" not in git or "dirty" not in git:
            raise ValueError(f"{path}: Git provenance missing")
        if not isinstance(software, dict) or not software:
            raise ValueError(f"{path}: software provenance missing")
        if reference_software is None:
            reference_software = software
        elif software != reference_software:
            raise ValueError("Artifacts contain software drift")
        dataset = record.get("dataset")
        if not isinstance(dataset, dict):
            raise TypeError(f"{path}: dataset metadata missing")
        if dataset.get("postfailure_transitions") != 0:
            raise ValueError(f"{path}: postfailure transitions must be zero")
        if dataset.get("post_failure_transitions") != 0:
            raise ValueError(f"{path}: post-failure transitions must be zero")
        if reference_config is None:
            reference_config = {k: v for k, v in config.items() if k != "seed"}
        elif {k: v for k, v in config.items() if k != "seed"} != reference_config:
            raise ValueError("Artifacts contain configuration drift")
        arms = record.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(ARMS):
            raise ValueError(f"{path}: arm set mismatch")
        for arm in ARMS:
            for metric in METRICS:
                result = arms[arm].get(metric)
                if not isinstance(result, dict):
                    raise TypeError(f"{path}: missing {arm}/{metric}")
                returns = finite_array(result.get("returns"), f"{path}: returns")
                if len(returns) != manifest["config"]["eval_episodes"]:
                    raise ValueError(f"{path}: wrong evaluation episode count")
                mean = float(result.get("mean"))
                if not np.isfinite(mean) or not np.isclose(mean, returns.mean()):
                    raise ValueError(f"{path}: invalid or inconsistent mean")
        records.append(record)
    if seen != set(SEEDS):
        raise ValueError(f"Missing seeds: {sorted(set(SEEDS) - seen)}")
    return sorted(records, key=lambda item: item["config"]["seed"]), manifest


def holm_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm step-down family-wise error correction."""
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(values[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (len(order) - rank)))
        adjusted[index] = running
    return [float(value) for value in adjusted]


def summarize(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Validate artifacts and produce the publication-facing summary object."""
    records, manifest = validate_artifacts(root, manifest_path)
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
    settings = manifest["statistics"]
    thresholds = settings["practical_effect_thresholds"]
    rng = np.random.default_rng(20260831)
    contrasts: dict[str, dict[str, Any]] = {}
    for left_name, right_name in (
        ("oracle_cf", "fresh_noise"),
        ("oracle_cf", "real"),
        ("fresh_noise", "real"),
    ):
        for metric in METRICS:
            key = f"{left_name}_minus_{right_name}_{metric}"
            contrasts[key] = paired_statistics(
                values[left_name][metric],
                values[right_name][metric],
                rng,
                settings["bootstrap_samples"],
                settings["randomization_samples"],
                float(thresholds[metric]),
            )
    summary: dict[str, Any] = {
        "artifact_schema": SCHEMA,
        "seeds": list(SEEDS),
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": source_manifest_digest(manifest_path),
        "config": manifest["config"],
        "registered_primary": settings["primary"],
        "registered_contrasts": settings["contrasts"],
        "arms": {
            arm: {metric: describe(values[arm][metric]) for metric in METRICS}
            for arm in ARMS
        },
        "contrasts": contrasts,
    }
    for test_name in ("paired_t_p", "wilcoxon_p", "sign_randomization_p"):
        keys = [f"{left}_minus_{right}_{metric}" for left, right, metric in CONTRASTS]
        adjusted = holm_adjust([contrasts[key][test_name] for key in keys])
        for key, value in zip(keys, adjusted):
            contrasts[key][f"{test_name}_holm"] = value
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("oracle_confirmatory_manifest.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.root, args.manifest)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
