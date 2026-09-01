from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from scripts.workshop.run_mujoco_oracle_cf_sac import (
    ExperimentConfig,
    add_replay_transition,
    branch_one_step,
    capture_env_snapshot,
    config_from_args,
    parse_args,
    restore_env_snapshot,
    sample_alternative_action,
)


class RecordingReplayBuffer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def add(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        env_id="HalfCheetah-v4",
        seed=0,
        variant="oracle_cf",
        total_timesteps=100,
        eval_episodes=2,
        eval_seed_base=10_000,
        intervention_scale=0.2,
        learning_starts=10,
        buffer_size=1_000,
        batch_size=32,
        train_freq=1,
        gradient_steps=1,
        output_root=Path("results/test"),
        job_id="unit-test",
    )


def test_snapshot_restores_state_and_time_limit() -> None:
    env = gym.make("Pendulum-v1", max_episode_steps=3)
    try:
        env.reset(seed=7)
        snapshot = capture_env_snapshot(env)
        action = np.asarray([0.4], dtype=np.float32)
        expected = env.step(action)
        assert env._elapsed_steps == 1

        restore_env_snapshot(env, snapshot)
        assert env._elapsed_steps == 0
        actual = env.step(action)

        np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=1e-7)
        assert actual[1:] == expected[1:]
    finally:
        env.close()


def test_branch_matches_direct_step_and_restores_factual_state() -> None:
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    reference = gym.make("Pendulum-v1", max_episode_steps=5)
    try:
        env.reset(seed=19)
        reference.reset(seed=19)
        pre_step = capture_env_snapshot(env)

        factual_action = np.asarray([0.75], dtype=np.float32)
        factual = env.step(factual_action)
        post_factual = capture_env_snapshot(env)

        alternative = np.asarray([-0.5], dtype=np.float32)
        branch = branch_one_step(env, pre_step, post_factual, alternative)
        expected = reference.step(alternative)

        np.testing.assert_allclose(branch.next_obs, expected[0], atol=1e-7)
        assert branch.reward == pytest.approx(expected[1])
        assert branch.terminated is expected[2]
        assert branch.truncated is expected[3]

        restored = capture_env_snapshot(env)
        np.testing.assert_allclose(restored.physics_state, post_factual.physics_state)
        assert restored.time_limit_steps == post_factual.time_limit_steps

        next_action = np.asarray([0.1], dtype=np.float32)
        continued = env.step(next_action)
        restore_env_snapshot(env, post_factual)
        repeated = env.step(next_action)
        np.testing.assert_allclose(continued[0], repeated[0], atol=1e-7)
        assert continued[1:] == repeated[1:]
        assert factual[0].shape == branch.next_obs.shape
    finally:
        env.close()
        reference.close()


def test_replay_insertion_has_single_env_shapes_and_exact_count() -> None:
    replay = RecordingReplayBuffer()
    for index in range(2):
        add_replay_transition(
            replay,
            obs=np.asarray([index, 1.0, 2.0], dtype=np.float32),
            next_obs=np.asarray([index + 1, 1.5, 2.5], dtype=np.float32),
            action=np.asarray([0.25, -0.5], dtype=np.float32),
            reward=1.25,
            done=index == 1,
            info={"TimeLimit.truncated": index == 1},
        )

    assert len(replay.calls) == 2
    for call in replay.calls:
        assert call["obs"].shape == (1, 3)
        assert call["next_obs"].shape == (1, 3)
        assert call["action"].shape == (1, 2)
        assert call["reward"].shape == (1,)
        assert call["done"].shape == (1,)
        assert len(call["infos"]) == 1


def test_alternative_action_is_bounded_and_changed() -> None:
    action_space = gym.spaces.Box(
        low=np.asarray([-2.0, -1.0], dtype=np.float32),
        high=np.asarray([2.0, 3.0], dtype=np.float32),
    )
    factual = np.asarray([2.0, 0.25], dtype=np.float32)
    alternative = sample_alternative_action(
        factual,
        action_space,
        np.random.default_rng(3),
        scale=0.2,
    )
    assert action_space.contains(alternative)
    assert not np.allclose(alternative, factual, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("total_timesteps", 0, "total_timesteps"),
        ("intervention_scale", 0.0, "intervention_scale"),
        ("eval_episodes", 0, "eval_episodes"),
        ("buffer_size", 199, "buffer_size"),
        ("batch_size", 1_001, "batch_size"),
        ("gradient_steps", 0, "gradient_steps"),
        ("job_id", "", "job_id"),
    ],
)
def test_config_rejects_invalid_arguments(field: str, value: Any, message: str) -> None:
    config = replace(_config(), **{field: value})
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_cli_parses_exactly_one_job() -> None:
    args = parse_args(
        [
            "--env-id",
            "Hopper-v4",
            "--seed",
            "4",
            "--variant",
            "duplicate",
            "--job-id",
            "reviewer-r1",
        ]
    )
    config = config_from_args(args)
    assert config.env_id == "Hopper-v4"
    assert config.seed == 4
    assert config.variant == "duplicate"
    assert config.job_id == "reviewer-r1"


def test_config_rejects_train_frequency_overshoot() -> None:
    with pytest.raises(ValueError, match="divisible"):
        replace(_config(), total_timesteps=101, train_freq=2).validate()


def test_cli_rejects_unknown_variant() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--env-id",
                "Hopper-v4",
                "--seed",
                "0",
                "--variant",
                "not-a-variant",
            ]
        )
