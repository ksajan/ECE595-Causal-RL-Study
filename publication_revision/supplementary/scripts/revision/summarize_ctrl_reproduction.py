#!/usr/bin/env python3
"""Aggregate paired CTRL reproduction seeds using seed-level inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

try:
    from scripts.revision.result_validation import load_and_validate
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from result_validation import load_and_validate

CONDITIONS = ("random", "real", "fresh_noise", "oracle_cf", "learned_cf")
METRICS = ("clean", "noisy")
HOLM_TESTS = (
    ("oracle_cf_minus_fresh_noise", "noisy"),
    ("oracle_cf_minus_real", "noisy"),
    ("fresh_noise_minus_real", "noisy"),
    ("oracle_cf_minus_fresh_noise", "clean"),
)


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    """Return a two-sided 95% Student-t confidence interval."""
    if len(values) < 2:
        return float("nan"), float("nan")
    mean = float(values.mean())
    half_width = float(
        stats.t.ppf(0.975, df=len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return mean - half_width, mean + half_width


def describe(values: np.ndarray) -> dict[str, object]:
    """Describe seed-level means without conflating episodes and seeds."""
    low, high = confidence_interval(values)
    return {
        "n_seeds": len(values),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "median": float(np.median(values)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "ci95_low": low,
        "ci95_high": high,
        "seed_means": [float(value) for value in values],
    }


def bootstrap_mean_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int = 20_000,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval for a seed-level mean."""
    if len(values) < 2:
        return float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_randomization_p(
    delta: np.ndarray,
    rng: np.random.Generator,
    samples: int = 100_000,
) -> float:
    """Return a two-sided paired sign-randomization p-value."""
    if len(delta) < 2:
        return float("nan")
    observed = abs(float(delta.mean()))
    signs = rng.choice((-1.0, 1.0), size=(samples, len(delta)))
    randomized = np.abs((signs * delta).mean(axis=1))
    return float((np.sum(randomized >= observed) + 1) / (samples + 1))


def paired_test(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Return paired effect size, t-test, Wilcoxon test, and sign count."""
    delta = left - right
    low, high = confidence_interval(delta)
    bootstrap_low, bootstrap_high = bootstrap_mean_interval(delta, rng)
    if len(delta) < 2:
        paired_t_p = float("nan")
        wilcoxon_p = float("nan")
    else:
        paired_t_p = float(stats.ttest_rel(left, right).pvalue)
        try:
            wilcoxon_p = float(stats.wilcoxon(delta).pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
    return {
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std(ddof=1)) if len(delta) > 1 else None,
        "paired_cohens_d": (
            float(delta.mean() / delta.std(ddof=1))
            if len(delta) > 1 and delta.std(ddof=1) > 0
            else None
        ),
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_ci95_low": bootstrap_low,
        "bootstrap_ci95_high": bootstrap_high,
        "paired_t_p": paired_t_p,
        "wilcoxon_p": wilcoxon_p,
        "randomization_p": paired_randomization_p(delta, rng),
        "positive_seeds": int(np.sum(delta > 0)),
        "negative_seeds": int(np.sum(delta < 0)),
        "ties": int(np.sum(delta == 0)),
        "seed_deltas": [float(value) for value in delta],
    }


def load_runs(
    root: Path,
    *,
    expected_seeds: list[int] | None = None,
    expected_count: int | None = None,
    development: bool = False,
) -> list[dict[str, object]]:
    """Load artifacts through the shared v4 validator."""
    runs, _ = load_and_validate(
        root,
        expected_seeds=expected_seeds,
        expected_count=expected_count,
        development=development,
    )
    return runs


def comparable_config(run: dict[str, object]) -> dict[str, object]:
    """Return configuration fields that must match across confirmatory seeds."""
    config = dict(run["config"])
    config.pop("seed", None)
    config.pop("output_dir", None)
    return config


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values."""
    values = np.asarray(p_values, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(values))
    adjusted = np.full_like(values, np.nan)
    if not len(finite_indices):
        return [float(value) for value in adjusted]
    order = finite_indices[np.argsort(values[finite_indices])]
    running_maximum = 0.0
    count = len(finite_indices)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (count - rank))
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return [float(value) for value in adjusted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        help="Exact seed set required for a confirmatory summary.",
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Allow failed model-quality gates, but mark output non-confirmatory.",
    )
    args = parser.parse_args()
    if (
        not args.development
        and args.expected_count is None
        and args.expected_seeds is None
    ):
        parser.error(
            "confirmatory summaries require --expected-count or --expected-seeds"
        )
    runs, validation = load_and_validate(
        args.root,
        expected_seeds=args.expected_seeds,
        expected_count=args.expected_count,
        development=args.development,
    )
    seeds = [int(run["config"]["seed"]) for run in runs]
    source_hashes = [str(runs[0]["source_sha256"])]
    reference_config = comparable_config(runs[0])
    software_records = [run["software"] for run in runs]

    values: dict[str, dict[str, np.ndarray]] = {
        condition: {
            metric: np.asarray([float(run[condition][metric]["mean"]) for run in runs])
            for metric in METRICS
        }
        for condition in CONDITIONS
    }
    inference_rng = np.random.default_rng(20260831)
    contrasts = {
        "learned_cf_minus_real": {
            metric: paired_test(
                values["learned_cf"][metric],
                values["real"][metric],
                inference_rng,
            )
            for metric in METRICS
        },
        "oracle_cf_minus_fresh_noise": {
            metric: paired_test(
                values["oracle_cf"][metric],
                values["fresh_noise"][metric],
                inference_rng,
            )
            for metric in METRICS
        },
        "oracle_cf_minus_real": {
            metric: paired_test(
                values["oracle_cf"][metric],
                values["real"][metric],
                inference_rng,
            )
            for metric in METRICS
        },
        "fresh_noise_minus_real": {
            metric: paired_test(
                values["fresh_noise"][metric],
                values["real"][metric],
                inference_rng,
            )
            for metric in METRICS
        },
        "learned_cf_minus_oracle_cf": {
            metric: paired_test(
                values["learned_cf"][metric],
                values["oracle_cf"][metric],
                inference_rng,
            )
            for metric in METRICS
        },
    }
    for field in ("paired_t_p", "wilcoxon_p", "randomization_p"):
        adjusted = holm_adjust(
            [
                float(contrasts[contrast][metric][field])
                for contrast, metric in HOLM_TESTS
            ]
        )
        for (contrast, metric), adjusted_value in zip(HOLM_TESTS, adjusted):
            contrasts[contrast][metric][f"{field}_holm"] = adjusted_value

    summary: dict[str, object] = {
        "seeds": seeds,
        "validation": validation,
        "source_sha256": source_hashes[0],
        "config": reference_config,
        "software": software_records[0],
        "conditions": {
            condition: {
                metric: describe(values[condition][metric]) for metric in METRICS
            }
            for condition in CONDITIONS
        },
        "contrasts": contrasts,
        "bicogan": {
            "normalized_next_state_mse": describe(
                np.asarray(
                    [
                        float(
                            run["bicogan"]["diagnostics"]["normalized_next_state_mse"]
                        )
                        for run in runs
                    ]
                )
            ),
            "held_out_counterfactual_normalized_mse": describe(
                np.asarray(
                    [
                        float(
                            run["bicogan"]["counterfactual_diagnostics"][
                                "external_validation"
                            ]["normalized_mse"]
                        )
                        for run in runs
                    ]
                )
            ),
            "held_out_counterfactual_terminal_disagreement": describe(
                np.asarray(
                    [
                        float(
                            run["bicogan"]["counterfactual_diagnostics"][
                                "external_validation"
                            ]["terminal_disagreement"]
                        )
                        for run in runs
                    ]
                )
            ),
        },
        "dataset": {
            field: describe(np.asarray([float(run["dataset"][field]) for run in runs]))
            for field in (
                "real_transitions",
                "mean_trial_length",
                "physics_failure_trial_fraction",
                "post_failure_transitions",
                "noisy_vs_pre_noise_terminal_disagreement",
            )
        },
        "files": [str(run["_path"]) for run in runs],
    }
    output = args.output or args.root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
