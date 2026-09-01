"""Focused tests for the prospectively frozen coupling-control harness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.revision.cartpole_coupling_control import (
    ALTERNATIVES_PER_TRANSITION,
    ARMS,
    EXPECTED_SEEDS,
    SCHEMA,
    config_from_manifest,
    generate_coupling_datasets,
    load_manifest,
)
from scripts.revision.cartpole_ctrl_reproduction import (
    ReproductionConfig,
    generate_paired_dataset,
)
from scripts.revision.summarize_coupling_control import (
    classification,
    holm_adjust,
    paired_statistics,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "scripts" / "revision" / "coupling_control_manifest.json"


def tiny_config(tmp_path: Path, seed: int = 1030) -> ReproductionConfig:
    """Return a small process-noise config suitable for dataset-only tests."""
    return ReproductionConfig(
        seed=seed,
        experiment_tier="oracle_coupling_control",
        dataset_trials=4,
        trial_horizon=5,
        stop_on_termination=True,
        batch_size=8,
        noise_semantics="process",
        output_dir=tmp_path,
    )


def test_manifest_is_frozen_and_has_registered_arms() -> None:
    """The manifest declares the exact seed bank and five required arms."""
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest["artifact_schema"] == SCHEMA
    assert manifest["seeds"] == list(EXPECTED_SEEDS)
    assert manifest["arms"] == list(ARMS)
    assert manifest["statistics"]["primary"] == ("oracle_cf_minus_fresh_shared_clean")
    config = config_from_manifest(manifest, 1030, Path("results/test"))
    assert config.q_width == 256
    assert config.q_depth == 2
    assert config.q_batch_norm is False
    assert config.cql_alpha == pytest.approx(0.05)
    assert config.train_steps == 5000
    assert config.cf_batch_fraction == pytest.approx(0.5)
    assert config.noise_semantics == "process"


def test_factual_pool_matches_public_generator_exactly(tmp_path: Path) -> None:
    """The duplicated shared-noise loop must preserve every factual field."""
    config = tiny_config(tmp_path)
    factual, independent, _shared, oracle, metadata, diagnostics = (
        generate_coupling_datasets(config)
    )
    reference, reference_independent, reference_oracle, _ = generate_paired_dataset(
        config
    )
    for field in ("states", "actions", "rewards", "next_states", "dones", "trial_ids"):
        np.testing.assert_array_equal(
            getattr(factual, field), getattr(reference, field)
        )
        np.testing.assert_array_equal(
            getattr(independent, field), getattr(reference_independent, field)
        )
        np.testing.assert_array_equal(
            getattr(oracle, field), getattr(reference_oracle, field)
        )
    assert metadata["factual_alignment_with_generate_paired_dataset"] is True
    assert diagnostics["factual_alignment"] is True


def test_shared_siblings_reuse_one_pair_and_differ_from_independent(
    tmp_path: Path,
) -> None:
    """Shared siblings reuse one pair, while independent siblings are distinct."""
    _, independent, shared, _, _, diagnostics = generate_coupling_datasets(
        tiny_config(tmp_path)
    )
    transition_count = len(shared) // ALTERNATIVES_PER_TRANSITION
    action_noise = diagnostics["shared_action_noise_per_alternative"]
    state_noise = diagnostics["shared_state_noise_per_alternative"]
    action_groups = action_noise.reshape(transition_count, ALTERNATIVES_PER_TRANSITION)
    state_groups = state_noise.reshape(transition_count, ALTERNATIVES_PER_TRANSITION, 4)
    assert np.all(np.ptp(action_groups, axis=1) == 0.0)
    assert np.all(np.ptp(state_groups, axis=1) == 0.0)
    assert diagnostics["within_transition_action_noise_reuse_max_abs"] == 0.0
    assert diagnostics["within_transition_state_noise_reuse_max_abs"] == 0.0
    assert not np.array_equal(shared.next_states, independent.next_states)
    assert diagnostics["fresh_shared_vs_independent_exact_next_state_fraction"] < 1.0
    assert diagnostics["shared_action_noise_unique_count"] > 1
    assert diagnostics["shared_state_noise_unique_count"] > 1


def test_practical_classification_is_symmetric() -> None:
    """Benefit and harm use the same threshold around zero."""
    assert classification(30.0, 40.0, 31.0, 39.0, 25.0) == (
        "practically_meaningful_benefit",
        False,
    )
    assert classification(-40.0, -30.0, -39.0, -31.0, 25.0) == (
        "practically_meaningful_harm",
        False,
    )
    assert classification(-10.0, 10.0, -9.0, 9.0, 25.0) == (
        "practically_equivalent",
        True,
    )


def test_paired_summary_contains_all_requested_statistics() -> None:
    """Paired output exposes CIs, three tests, and win/loss/tie counts."""
    left = np.asarray([40.0, 42.0, 41.0, 43.0, 44.0])
    right = np.asarray([10.0, 12.0, 11.0, 13.0, 14.0])
    summary = paired_statistics(left, right, np.random.default_rng(9), 200, 500, 25.0)
    for key in (
        "mean_delta",
        "sd_delta",
        "median_delta",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "bootstrap_ci90_low",
        "bootstrap_ci90_high",
        "paired_t_p",
        "wilcoxon_p",
        "sign_randomization_p",
    ):
        assert key in summary
        assert np.isfinite(summary[key])
    assert summary["positive_seeds"] == 5
    assert summary["negative_seeds"] == 0
    assert summary["ties"] == 0
    assert holm_adjust([0.01, 0.02, 0.5, 1.0]) == pytest.approx([0.04, 0.06, 1.0, 1.0])
