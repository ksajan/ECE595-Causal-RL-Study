from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from scripts.workshop.run_d4rl_simulator_cf_cql import (
    D4RL_REFERENCE_RETURNS,
    DirectArrayReplayBuffer,
    SyntheticArrays,
    TrainingArrays,
    TransitionArrays,
    _cache_path,
    compose_residual_variants,
    deranged_permutation,
    load_d4rl_hdf5,
    make_fixed_size_training_arm,
    normalized_d4rl_score,
    parse_args,
    sample_alternative_actions,
    validate_cache_metadata,
)


def _training_arrays(size: int = 12) -> TrainingArrays:
    observations = np.arange(size * 3, dtype=np.float32).reshape(size, 3)
    # Deliberately unrelated to the next row of observations.
    next_observations = observations + 1_000.0
    return TrainingArrays(
        observations=observations,
        actions=np.linspace(-0.8, 0.8, size, dtype=np.float32)[:, None],
        rewards=np.linspace(-1.0, 1.0, size, dtype=np.float32),
        next_observations=next_observations,
        terminals=np.zeros(size, dtype=np.bool_),
        timeouts=np.zeros(size, dtype=np.bool_),
    )


def _dataset_arrays(size: int = 12) -> TransitionArrays:
    training = _training_arrays(size)
    return TransitionArrays(
        observations=training.observations,
        actions=training.actions,
        rewards=training.rewards,
        next_observations=training.next_observations,
        terminals=training.terminals,
        timeouts=training.timeouts,
        qpos=np.zeros((size, 2), dtype=np.float64),
        qvel=np.zeros((size, 2), dtype=np.float64),
    )


def _synthetic_arrays(size: int = 12) -> SyntheticArrays:
    return SyntheticArrays(
        actions=np.full((size, 1), 0.25, dtype=np.float32),
        next_observations=np.full((size, 3), -10.0, dtype=np.float32),
        rewards=np.full(size, 3.0, dtype=np.float32),
        terminals=np.zeros(size, dtype=np.bool_),
        timeouts=np.zeros(size, dtype=np.bool_),
    )


def test_direct_replay_uses_explicit_next_observations() -> None:
    pytest.importorskip("d3rlpy")
    arrays = _training_arrays()
    replay = DirectArrayReplayBuffer(arrays, seed=7)

    transition = replay.transition_at(4)

    np.testing.assert_array_equal(transition.observation, arrays.observations[4])
    np.testing.assert_array_equal(
        transition.next_observation, arrays.next_observations[4]
    )
    assert not np.array_equal(transition.next_observation, arrays.observations[5]), (
        "Replay must not infer next state by shifting observation rows"
    )


def test_hdf5_loader_requires_and_preserves_explicit_next_states(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "tiny.hdf5"
    size = 4
    observations = np.arange(size * 2, dtype=np.float32).reshape(size, 2)
    explicit_next = observations + 50.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observations", data=observations)
        handle.create_dataset("actions", data=np.zeros((size, 1), dtype=np.float32))
        handle.create_dataset("rewards", data=np.arange(size, dtype=np.float32))
        handle.create_dataset("next_observations", data=explicit_next)
        handle.create_dataset("terminals", data=np.zeros(size, dtype=np.bool_))
        handle.create_dataset("timeouts", data=np.zeros(size, dtype=np.bool_))
        infos = handle.create_group("infos")
        infos.create_dataset("qpos", data=np.zeros((size, 2), dtype=np.float64))
        infos.create_dataset("qvel", data=np.zeros((size, 2), dtype=np.float64))

    loaded = load_d4rl_hdf5(path)

    np.testing.assert_array_equal(loaded.next_observations, explicit_next)
    assert not np.array_equal(loaded.next_observations[:-1], observations[1:])


def test_factual_and_fresh_residual_coupling() -> None:
    simulator_next = np.asarray(
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32
    )
    simulator_rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    state_residuals = np.asarray([[0.1, 0.2], [1.1, 1.2], [2.1, 2.2]], dtype=np.float32)
    reward_residuals = np.asarray([0.5, 1.5, 2.5], dtype=np.float32)
    fresh_indices = np.asarray([2, 0, 1], dtype=np.int64)

    factual_next, factual_reward, fresh_next, fresh_reward = compose_residual_variants(
        simulator_next,
        simulator_rewards,
        state_residuals,
        reward_residuals,
        fresh_indices,
    )

    np.testing.assert_allclose(factual_next, simulator_next + state_residuals)
    np.testing.assert_allclose(factual_reward, simulator_rewards + reward_residuals)
    np.testing.assert_allclose(
        fresh_next, simulator_next + state_residuals[fresh_indices]
    )
    np.testing.assert_allclose(
        fresh_reward, simulator_rewards + reward_residuals[fresh_indices]
    )


def test_residual_permutation_has_no_factual_self_matches() -> None:
    indices = deranged_permutation(100, np.random.default_rng(13))
    np.testing.assert_array_equal(np.sort(indices), np.arange(100))
    assert not np.any(indices == np.arange(100))


def test_continuous_interventions_are_bounded_and_changed() -> None:
    action_space = gym.spaces.Box(
        low=np.asarray([-2.0, -1.0], dtype=np.float32),
        high=np.asarray([2.0, 3.0], dtype=np.float32),
    )
    factual = np.asarray(
        [[-2.0, -1.0], [2.0, 3.0], [0.0, 1.0], [1.9, -0.9]],
        dtype=np.float32,
    )

    alternatives = sample_alternative_actions(
        factual,
        action_space,
        np.random.default_rng(21),
        scale=0.2,
    )

    assert alternatives.shape == factual.shape
    assert np.all(alternatives >= action_space.low)
    assert np.all(alternatives <= action_space.high)
    assert np.all(np.any(np.abs(alternatives - factual) > 1e-7, axis=1))


@pytest.mark.parametrize("dataset_name", sorted(D4RL_REFERENCE_RETURNS))
def test_official_d4rl_normalized_score_endpoints(dataset_name: str) -> None:
    random_return, expert_return = D4RL_REFERENCE_RETURNS[dataset_name]
    assert normalized_d4rl_score(dataset_name, random_return) == pytest.approx(0.0)
    assert normalized_d4rl_score(dataset_name, expert_return) == pytest.approx(100.0)
    midpoint = 0.5 * (random_return + expert_return)
    assert normalized_d4rl_score(dataset_name, midpoint) == pytest.approx(50.0)


def test_cql_defaults_match_official_d3rlpy_reproduction() -> None:
    args = parse_args(
        ["--dataset", "hopper-medium-v2", "--seed", "3", "--mode", "train"]
    )

    assert args.steps == 500_000
    assert args.encoder_hidden_units == (256, 256, 256)
    assert args.alpha_learning_rate == 0.0
    assert args.conservative_weight == 10.0
    assert args.temp_learning_rate == 1e-4


def test_learner_seeds_share_identical_arm_with_fixed_augmentation_seed() -> None:
    pytest.importorskip("d3rlpy")
    dataset = _dataset_arrays()
    synthetic = _synthetic_arrays()
    arm_a, mask_a = make_fixed_size_training_arm(
        dataset,
        "factual_residual",
        cf_fraction=0.5,
        augmentation_seed=17,
        synthetic=synthetic,
    )
    arm_b, mask_b = make_fixed_size_training_arm(
        dataset,
        "factual_residual",
        cf_fraction=0.5,
        augmentation_seed=17,
        synthetic=synthetic,
    )

    # Learner seeds alter minibatch RNG only, not the fixed data arm.
    replay_seed_3 = DirectArrayReplayBuffer(arm_a, seed=3)
    replay_seed_91 = DirectArrayReplayBuffer(arm_b, seed=91)
    np.testing.assert_array_equal(mask_a, mask_b)
    for index in range(dataset.size):
        transition_a = replay_seed_3.transition_at(index)
        transition_b = replay_seed_91.transition_at(index)
        np.testing.assert_array_equal(
            transition_a.next_observation, transition_b.next_observation
        )
        np.testing.assert_array_equal(transition_a.action, transition_b.action)
        np.testing.assert_array_equal(transition_a.reward, transition_b.reward)


def test_cache_path_uses_augmentation_seed_not_learner_seed() -> None:
    args_a = parse_args(
        [
            "--dataset",
            "hopper-medium-v2",
            "--seed",
            "3",
            "--augmentation-seed",
            "7",
        ]
    )
    args_b = parse_args(
        [
            "--dataset",
            "hopper-medium-v2",
            "--seed",
            "91",
            "--augmentation-seed",
            "7",
        ]
    )
    args_c = parse_args(
        [
            "--dataset",
            "hopper-medium-v2",
            "--seed",
            "91",
            "--augmentation-seed",
            "8",
        ]
    )

    assert _cache_path(args_a) == _cache_path(args_b)
    assert _cache_path(args_a) != _cache_path(args_c)


def test_cache_validation_uses_augmentation_seed(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    cache_path = tmp_path / "augmentation.hdf5"
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["schema_version"] = 2
        handle.attrs["dataset_name"] = "hopper-medium-v2"
        handle.attrs["dataset_sha256"] = "dataset-hash"
        handle.attrs["transition_count"] = 12
        handle.attrs["augmentation_seed"] = 5
        handle.attrs["intervention_scale"] = 0.1

    validate_cache_metadata(
        cache_path,
        "hopper-medium-v2",
        "dataset-hash",
        12,
        augmentation_seed=5,
        intervention_scale=0.1,
    )
    with pytest.raises(RuntimeError, match="augmentation_seed"):
        validate_cache_metadata(
            cache_path,
            "hopper-medium-v2",
            "dataset-hash",
            12,
            augmentation_seed=6,
            intervention_scale=0.1,
        )


def test_tiny_d3rlpy_cql_fit_uses_direct_array_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d3rlpy = pytest.importorskip("d3rlpy")
    arrays = _training_arrays(size=16)
    replay = DirectArrayReplayBuffer(arrays, seed=9)
    monkeypatch.chdir(tmp_path)
    d3rlpy.seed(9)
    algorithm = d3rlpy.algos.CQLConfig(
        batch_size=4,
        n_action_samples=2,
        n_critics=1,
    ).create(device=False)

    metrics = algorithm.fit(
        replay,
        n_steps=2,
        n_steps_per_epoch=2,
        experiment_name="tiny_direct_replay",
        with_timestamp=False,
        show_progress=False,
        save_interval=1,
    )

    assert len(metrics) == 1
    assert metrics[0][0] == 1
    action = algorithm.predict(arrays.observations[:1])
    assert action.shape == (1, 1)
    assert np.all(np.isfinite(action))
