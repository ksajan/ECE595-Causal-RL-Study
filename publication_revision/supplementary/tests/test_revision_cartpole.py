"""Focused regression tests for the corrected CartPole revision harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.revision.cartpole_ctrl_reproduction as reproduction
import scripts.revision.cartpole_sanity as sanity
from scripts.revision.bicogan_ctrl import (
    MonotonicBiCoGANGenerator,
    PositiveLinear,
    TriangularMonotoneGenerator,
)
from scripts.revision.cartpole_ctrl_reproduction import (
    ReproductionConfig,
    generate_paired_dataset,
    noisy_observation,
    physics_step,
    sample_training_batch,
)
from scripts.revision.plot_ctrl_revision import load_artifacts
from scripts.revision.result_validation import load_and_validate
from scripts.revision.summarize_ctrl_reproduction import main as summarize_main


def test_physics_step_is_deterministic_for_fixed_exogenous_noise() -> None:
    state = np.asarray([0.01, -0.02, 0.03, 0.04], dtype=np.float64)

    first_state, first_done = physics_step(state, action=7, action_noise=0.0125)
    second_state, second_done = physics_step(state, action=7, action_noise=0.0125)

    np.testing.assert_array_equal(first_state, second_state)
    assert first_done == second_done


def test_noisy_observation_does_not_mutate_clean_state() -> None:
    clean_state = np.asarray([1.0, 2.0, 0.1, 4.0], dtype=np.float64)
    original = clean_state.copy()
    observation_noise = np.asarray([0.5, -1.0, 2.0, -3.0])

    observed = noisy_observation(clean_state, observation_noise, noise_std=0.05)

    np.testing.assert_array_equal(clean_state, original)
    np.testing.assert_allclose(observed, original + 0.05 * observation_noise)
    assert observed.dtype == np.float32


def test_sanity_rejects_overlapping_validation_and_test_seed_banks() -> None:
    config = sanity.SanityConfig(
        validation_episodes=20,
        validation_seed_base=100,
        eval_episodes=100,
        test_seed_base=110,
    )

    with pytest.raises(ValueError, match="seed banks overlap"):
        sanity.validate_seed_banks(config)


def test_process_noise_is_part_of_the_next_dynamics_state() -> None:
    env = sanity.CartPole11Env(
        seed=7,
        max_steps=20,
        state_noise_std=0.05,
        noise_semantics="process",
    )
    env.reset(seed=7)
    next_observation, *_ = env.step(5)

    np.testing.assert_allclose(env.state, next_observation)


def test_observation_noise_does_not_change_the_hidden_dynamics_state() -> None:
    env = sanity.CartPole11Env(
        seed=7,
        max_steps=20,
        state_noise_std=0.05,
        noise_semantics="observation",
    )
    env.reset(seed=7)
    next_observation, *_ = env.step(5)

    assert not np.allclose(env.state, next_observation)


def test_triangular_generator_is_strictly_increasing_on_its_diagonal() -> None:
    torch.manual_seed(7)
    generator = TriangularMonotoneGenerator(latent_dim=4).eval()
    state = torch.zeros(1, 4)
    action = torch.full((1, 1), 0.5)
    baseline_latent = torch.zeros(1, 4)
    baseline = generator(state, action, baseline_latent)

    for dimension in range(4):
        changed_latent = baseline_latent.clone()
        changed_latent[0, dimension] = 1.0
        changed = generator(state, action, changed_latent)
        assert torch.all(changed >= baseline)
        assert changed[0, dimension] > baseline[0, dimension]

    latent = torch.tensor([[0.2, -0.3, 0.5, 1.2]])
    generated = generator(state, action, latent)
    recovered = generator.inverse(state, action, generated)
    torch.testing.assert_close(recovered, latent)


def test_monotonic_bicogan_has_positive_latent_paths() -> None:
    torch.manual_seed(11)
    generator = MonotonicBiCoGANGenerator(latent_dim=4).eval()
    state = torch.zeros(8, 4)
    action = torch.full((8, 1), 0.5)
    lower = generator(state, action, torch.full((8, 4), -1.0))
    upper = generator(state, action, torch.full((8, 4), 1.0))

    assert torch.all(upper >= lower)
    assert torch.any(upper > lower)
    lower_random = torch.randn(8, 4)
    upper_random = lower_random + torch.rand(8, 4)
    lower_output = generator(state, action, lower_random)
    upper_output = generator(state, action, upper_random)
    assert torch.all(upper_output >= lower_output)
    for module in generator.modules():
        if isinstance(module, PositiveLinear):
            assert torch.all(module.raw_weight.exp() > 0)


def test_tiny_dataset_has_factual_fresh_and_oracle_pool_semantics() -> None:
    config = ReproductionConfig(
        seed=17,
        dataset_trials=2,
        trial_horizon=3,
        state_noise_std=0.0,
        action_noise_std=0.0,
    )

    real, fresh, oracle, metadata = generate_paired_dataset(config)

    assert len(real) == int(metadata["real_transitions"])
    assert len(fresh) == 10 * len(real)
    assert len(oracle) == 10 * len(real)
    assert len(fresh) == int(metadata["fresh_noise_transitions"])
    assert len(oracle) == int(metadata["oracle_cf_transitions"])
    assert set(real.trial_ids) == {0, 1}

    factual_actions = np.repeat(real.actions, 10)
    np.testing.assert_array_equal(np.repeat(real.states, 10, axis=0), oracle.states)
    assert np.all(oracle.actions != factual_actions)
    assert np.all(fresh.actions != factual_actions)

    # With zero noise, the oracle transition for a given state/action is the
    # simulator transition with the same action and the reused factual draw.
    for index in range(min(5, len(oracle))):
        expected, expected_done = physics_step(
            oracle.states[index], int(oracle.actions[index]), action_noise=0.0
        )
        np.testing.assert_allclose(oracle.next_states[index], expected, rtol=1e-6)
        assert oracle.dones[index] == float(expected_done)


def test_primary_dataset_stops_at_hidden_physics_termination(monkeypatch) -> None:
    def always_terminates(state, action, action_noise):
        del action, action_noise
        return np.asarray(state, dtype=np.float64) + 0.01, True

    monkeypatch.setattr(reproduction, "physics_step", always_terminates)
    config = ReproductionConfig(
        seed=23,
        dataset_trials=3,
        trial_horizon=20,
        state_noise_std=0.0,
        action_noise_std=0.0,
    )

    real, fresh, oracle, metadata = generate_paired_dataset(config)

    assert config.stop_on_termination is True
    assert len(real) == 3
    assert len(fresh) == len(oracle) == 30
    assert np.all(real.dones == 1.0)
    assert metadata["post_failure_transitions"] == 0
    assert metadata["terminal_label_rule"] == "pre_noise_next_state"


def test_continuation_sensitivity_records_post_failure_transitions(
    monkeypatch,
) -> None:
    def always_terminates(state, action, action_noise):
        del action, action_noise
        return np.asarray(state, dtype=np.float64) + 0.01, True

    monkeypatch.setattr(reproduction, "physics_step", always_terminates)
    config = ReproductionConfig(
        seed=24,
        dataset_trials=2,
        trial_horizon=4,
        stop_on_termination=False,
        state_noise_std=0.0,
        action_noise_std=0.0,
    )

    real, _, _, metadata = generate_paired_dataset(config)

    assert len(real) == 8
    assert metadata["post_failure_transitions"] == 6


def test_sanity_environment_uses_configured_noise_levels() -> None:
    config = sanity.SanityConfig(state_noise_std=0.123, action_noise_std=0.234)
    clean = sanity.make_env(seed=1, cfg=config, noisy=False)
    noisy = sanity.make_env(seed=1, cfg=config, noisy=True)

    assert clean.state_noise_std == 0.0
    assert noisy.state_noise_std == pytest.approx(0.123)


def test_training_batch_uses_exact_requested_cf_fraction(monkeypatch) -> None:
    config = ReproductionConfig(
        seed=3,
        dataset_trials=1,
        trial_horizon=2,
        state_noise_std=0.0,
        action_noise_std=0.0,
    )
    real, _, oracle, _ = generate_paired_dataset(config)

    requested_counts: list[int] = []
    original_sample_pool = reproduction.sample_pool

    def recording_sample_pool(pool, count, rng):
        requested_counts.append(count)
        return original_sample_pool(pool, count, rng)

    monkeypatch.setattr(reproduction, "sample_pool", recording_sample_pool)
    states, actions, rewards, next_states, dones = sample_training_batch(
        real,
        oracle,
        batch_size=20,
        cf_fraction=0.25,
        rng=np.random.default_rng(99),
    )

    assert states.shape[0] == actions.shape[0] == 20
    assert rewards.shape[0] == next_states.shape[0] == dones.shape[0] == 20
    assert requested_counts == [15, 5]
    assert np.isfinite(states).all()
    assert np.isfinite(next_states).all()


def _complete_summary_record(seed: int, train_steps: int) -> dict[str, object]:
    condition = {
        "clean": {"mean": 1.0, "returns": [1.0]},
        "noisy": {"mean": 1.0, "returns": [1.0]},
    }
    return {
        "artifact_schema": "ctrl-cartpole-revision-v4",
        "experiment_tier": "ctrl_bicogan_reproduction",
        "config": {
            "seed": seed,
            "train_steps": train_steps,
            "experiment_tier": "ctrl_bicogan_reproduction",
            "bicogan_generator": "monotonic_bicogan",
            "noise_semantics": "process",
        },
        "command": ["python", "-m", "scripts.revision.cartpole_ctrl_reproduction"],
        "git": {"commit": "test", "dirty": False},
        "source_sha256": "same-source",
        "software": {"python": "test", "torch": "test", "numpy": "test"},
        "dataset": {
            "real_transitions": 1.0,
            "mean_trial_length": 1.0,
            "physics_failure_trial_fraction": 0.0,
            "post_failure_transitions": 0,
            "terminal_label_rule": "pre_noise_next_state",
            "noise_semantics": "process",
            "noisy_vs_pre_noise_terminal_disagreement": 0.0,
            "validation_seed": seed + 2_000_000,
            "validation_real_transitions": 1.0,
            "validation_post_failure_transitions": 0,
        },
        "bicogan": {
            "diagnostics": {
                "normalized_next_state_mse": 0.1,
                "validation_source": "independent_dataset",
                "validation_trial_ids": [1000 + seed],
                "latent_std_by_dimension": [1.0, 1.1, 0.9, 1.2],
                "action_reconstruction_mse": 0.05,
                "central_action_baseline_mse": 0.1,
            },
            "counterfactual_diagnostics": {
                "external_validation": {
                    "normalized_mse": 0.1,
                    "terminal_disagreement": 0.0,
                },
                "fresh_noise_external_validation": {"normalized_mse": 0.2},
            },
        },
        "random": condition,
        "real": condition,
        "fresh_noise": condition,
        "oracle_cf": condition,
        "learned_cf": condition,
    }


def test_summarizer_rejects_mismatched_configs(tmp_path: Path) -> None:
    (tmp_path / "reproduction_seed_1.json").write_text(
        json.dumps(_complete_summary_record(seed=1, train_steps=20))
    )
    (tmp_path / "reproduction_seed_2.json").write_text(
        json.dumps(_complete_summary_record(seed=2, train_steps=30))
    )

    old_argv = sys.argv
    sys.argv = [
        "summarize_ctrl_reproduction",
        str(tmp_path),
        "--expected-seeds",
        "1",
        "2",
    ]
    try:
        with pytest.raises(ValueError, match="mismatched configs"):
            summarize_main()
    finally:
        sys.argv = old_argv


def test_validator_requires_explicit_confirmatory_seed_contract(tmp_path: Path) -> None:
    (tmp_path / "reproduction_seed_1.json").write_text(
        json.dumps(_complete_summary_record(seed=1, train_steps=20))
    )
    with pytest.raises(ValueError, match="Expected seed set|Expected 2 seeds"):
        load_and_validate(tmp_path, expected_count=2)


def test_validator_rejects_duplicate_seed_even_when_one_is_incomplete(
    tmp_path: Path,
) -> None:
    (tmp_path / "reproduction_seed_a.json").write_text(
        json.dumps(_complete_summary_record(seed=1, train_steps=20))
    )
    incomplete = {"artifact_schema": "ctrl-cartpole-revision-v4", "config": {"seed": 1}}
    (tmp_path / "reproduction_seed_b.json").write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="Duplicate artifact"):
        load_and_validate(tmp_path, development=True)


def test_validator_marks_development_gate_failure_without_confirmatory_status(
    tmp_path: Path,
) -> None:
    record = _complete_summary_record(seed=1, train_steps=20)
    record["bicogan"]["counterfactual_diagnostics"]["external_validation"][
        "normalized_mse"
    ] = 0.3
    (tmp_path / "reproduction_seed_1.json").write_text(json.dumps(record))
    runs, report = load_and_validate(tmp_path, expected_seeds=[1], development=True)
    assert runs[0]["_quality_gate"]["gate_passed"] is False
    assert report == {
        "confirmatory": False,
        "development": True,
        "gate_passed": False,
        "seeds": [1],
        "artifact_schema": "ctrl-cartpole-revision-v4",
    }


def test_validator_rejects_tier_generator_mismatch(tmp_path: Path) -> None:
    record = _complete_summary_record(seed=1, train_steps=20)
    record["config"]["experiment_tier"] = "triangular_flow_extension"
    record["experiment_tier"] = "triangular_flow_extension"
    (tmp_path / "reproduction_seed_1.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="requires generator"):
        load_and_validate(tmp_path, expected_seeds=[1], development=True)


def test_validator_rejects_latent_dimension_outside_gate(tmp_path: Path) -> None:
    record = _complete_summary_record(seed=1, train_steps=20)
    record["bicogan"]["diagnostics"]["latent_std_by_dimension"][2] = 2.1
    (tmp_path / "reproduction_seed_1.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Model-quality gates failed"):
        load_and_validate(tmp_path, expected_seeds=[1])


def test_plot_loader_rejects_invalid_artifact_without_development_flag(
    tmp_path: Path,
) -> None:
    record = _complete_summary_record(seed=1, train_steps=20)
    record["dataset"]["post_failure_transitions"] = 1
    (tmp_path / "reproduction_seed_1.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="post_failure_transitions"):
        load_artifacts(tmp_path, expected_seeds=[1])
