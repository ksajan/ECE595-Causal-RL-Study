#!/usr/bin/env python3
"""Compute-matched CartPole-SD reproduction and oracle-CF diagnostic.

The primary environment treats additive state noise as transition/process
noise and feeds the resulting next state into the next physics step. Oracle
counterfactuals reuse the factual action- and state-noise draws, implementing
SCM abduction-action-prediction for the known simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from scripts.revision.bicogan_ctrl import BiCoGANConfig, CTRLBiCoGAN
from scripts.revision.cartpole_sanity import (
    GRAVITY,
    LENGTH,
    MASS_POLE,
    MAX_FORCE,
    POLEMASS_LENGTH,
    TAU,
    THETA_THRESHOLD,
    TOTAL_MASS,
    X_THRESHOLD,
    CartPole11Env,
    set_seed,
)


@dataclass(frozen=True)
class ReproductionConfig:
    """Configuration for one paired offline-RL seed."""

    seed: int = 0
    experiment_tier: str = "development"
    dataset_trials: int = 250
    validation_dataset_trials: int = 50
    validation_seed_offset: int = 2_000_000
    trial_horizon: int = 20
    stop_on_termination: bool = True
    train_steps: int = 10_000
    batch_size: int = 256
    eval_episodes: int = 100
    eval_horizon: int = 500
    gamma: float = 0.99
    learning_rate: float = 1e-4
    target_tau: float = 0.005
    target_update_interval: int = 0
    cql_alpha: float = 0.0
    cf_batch_fraction: float = 0.5
    q_width: int = 512
    q_depth: int = 4
    q_batch_norm: bool = True
    bicogan_pretrain_steps: int = 2_000
    bicogan_steps: int = 5_000
    bicogan_reconstruction_weight: float = 10.0
    bicogan_extrinsic_weight: float = 1.0
    bicogan_latent_cycle_weight: float = 1.0
    bicogan_generator: str = "monotonic_bicogan"
    state_noise_std: float = 0.05
    action_noise_std: float = 0.05
    noise_semantics: str = "process"
    eval_seed_base: int = 300_000
    output_dir: Path = Path("results/revision/cartpole_ctrl_reproduction")


@dataclass
class TransitionPool:
    """Array-backed transition pool with states shaped ``(N, 4)``."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    trial_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.actions.shape[0])


class TransitionBuilder:
    """Mutable helper used only while constructing a transition pool."""

    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.next_states: list[np.ndarray] = []
        self.dones: list[float] = []
        self.trial_ids: list[int] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        trial_id: int,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.dones.append(float(done))
        self.trial_ids.append(int(trial_id))

    def finish(self) -> TransitionPool:
        if not self.states:
            raise RuntimeError("Cannot construct an empty transition pool.")
        return TransitionPool(
            states=np.stack(self.states),
            actions=np.asarray(self.actions, dtype=np.int64),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            next_states=np.stack(self.next_states),
            dones=np.asarray(self.dones, dtype=np.float32),
            trial_ids=np.asarray(self.trial_ids, dtype=np.int64),
        )


class DuelingQNetwork(nn.Module):
    """Dueling Q-network mapping normalized states ``(B, 4)`` to ``(B, 11)``."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 11,
        width: int = 512,
        depth: int = 4,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = state_dim
        for _ in range(depth):
            layers.append(nn.Linear(input_dim, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            input_dim = width
        self.features = nn.Sequential(*layers)
        self.value = nn.Linear(width, 1)
        self.advantage = nn.Linear(width, action_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        features = self.features(states)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def physics_step(
    state: np.ndarray,
    action: int,
    action_noise: float,
) -> tuple[np.ndarray, bool]:
    """Advance clean CartPole physics using a fixed exogenous action-noise draw."""
    action_continuous = float(action) / 10.0 + float(action_noise)
    force = (2.0 * action_continuous - 1.0) * MAX_FORCE
    x, x_dot, theta, theta_dot = np.asarray(state, dtype=np.float64)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sin_theta) / TOTAL_MASS
    theta_acceleration = (GRAVITY * sin_theta - cos_theta * temp) / (
        LENGTH * (4.0 / 3.0 - MASS_POLE * cos_theta**2 / TOTAL_MASS)
    )
    x_acceleration = (
        temp - POLEMASS_LENGTH * theta_acceleration * cos_theta / TOTAL_MASS
    )
    next_state = np.asarray(
        [
            x + TAU * x_dot,
            x_dot + TAU * x_acceleration,
            theta + TAU * theta_dot,
            theta_dot + TAU * theta_acceleration,
        ],
        dtype=np.float64,
    )
    terminated = bool(
        abs(next_state[0]) > X_THRESHOLD or abs(next_state[2]) > THETA_THRESHOLD
    )
    return next_state, terminated


def noisy_observation(
    clean_state: np.ndarray,
    observation_noise: np.ndarray,
    noise_std: float,
) -> np.ndarray:
    """Return a noisy observation without mutating the hidden physics state."""
    observation = np.asarray(clean_state, dtype=np.float64).copy()
    observation += noise_std * np.asarray(observation_noise, dtype=np.float64)
    observation[0] = np.clip(observation[0], -4.8, 4.8)
    observation[2] = np.clip(observation[2], -0.418, 0.418)
    return observation.astype(np.float32)


def termination_from_observation(observation: np.ndarray) -> bool:
    """Apply one shared offline terminal-label rule to an observed next state."""
    return bool(
        abs(float(observation[0])) > X_THRESHOLD
        or abs(float(observation[2])) > THETA_THRESHOLD
    )


def generate_paired_dataset(
    config: ReproductionConfig,
) -> tuple[TransitionPool, TransitionPool, TransitionPool, dict[str, float]]:
    """Generate factual and exact noise-reused counterfactual transition pools."""
    factual_rng = np.random.default_rng(config.seed)
    fresh_rng = np.random.default_rng(config.seed + 1_000_000)
    real = TransitionBuilder()
    fresh_noise = TransitionBuilder()
    oracle_cf = TransitionBuilder()
    completed_steps: list[int] = []
    failed_trials: list[bool] = []
    terminal_label_disagreements = 0
    post_failure_transitions = 0
    transition_count = 0

    for trial_id in range(config.dataset_trials):
        dynamics_state = factual_rng.uniform(-0.05, 0.05, size=4)
        if config.noise_semantics == "process":
            state_observation = dynamics_state.astype(np.float32)
        elif config.noise_semantics == "observation":
            state_observation = noisy_observation(
                dynamics_state,
                factual_rng.normal(size=4),
                config.state_noise_std,
            )
        else:
            raise ValueError("noise_semantics must be 'process' or 'observation'.")
        steps = 0
        trial_failed = False
        for step in range(config.trial_horizon):
            if trial_failed:
                post_failure_transitions += 1
            factual_action = int(factual_rng.integers(0, 11))
            action_noise = float(factual_rng.normal(0.0, config.action_noise_std))
            next_observation_noise = factual_rng.normal(size=4)
            next_clean_state, factual_terminated = physics_step(
                dynamics_state,
                factual_action,
                action_noise,
            )
            next_observation = noisy_observation(
                next_clean_state,
                next_observation_noise,
                config.state_noise_std,
            )
            observed_terminated = termination_from_observation(next_observation)
            factual_done = factual_terminated
            terminal_label_disagreements += int(
                observed_terminated != factual_terminated
            )
            transition_count += 1
            real.add(
                state_observation,
                factual_action,
                1.0,
                next_observation,
                factual_done,
                trial_id,
            )

            for counterfactual_action in range(11):
                if counterfactual_action == factual_action:
                    continue
                cf_clean_state, cf_terminated = physics_step(
                    dynamics_state,
                    counterfactual_action,
                    action_noise,
                )
                cf_observation = noisy_observation(
                    cf_clean_state,
                    next_observation_noise,
                    config.state_noise_std,
                )
                cf_done = cf_terminated
                oracle_cf.add(
                    state_observation,
                    counterfactual_action,
                    1.0,
                    cf_observation,
                    cf_done,
                    trial_id,
                )
                fresh_action_noise = float(
                    fresh_rng.normal(0.0, config.action_noise_std)
                )
                fresh_observation_noise = fresh_rng.normal(size=4)
                fresh_clean_state, fresh_terminated = physics_step(
                    dynamics_state,
                    counterfactual_action,
                    fresh_action_noise,
                )
                fresh_observation = noisy_observation(
                    fresh_clean_state,
                    fresh_observation_noise,
                    config.state_noise_std,
                )
                fresh_done = fresh_terminated
                fresh_noise.add(
                    state_observation,
                    counterfactual_action,
                    1.0,
                    fresh_observation,
                    fresh_done,
                    trial_id,
                )

            steps += 1
            trial_failed = trial_failed or factual_terminated
            dynamics_state = (
                next_observation
                if config.noise_semantics == "process"
                else next_clean_state
            )
            state_observation = next_observation
            if factual_terminated and config.stop_on_termination:
                break
        completed_steps.append(steps)
        failed_trials.append(trial_failed)

    metadata = {
        "real_transitions": float(len(real.states)),
        "fresh_noise_transitions": float(len(fresh_noise.states)),
        "oracle_cf_transitions": float(len(oracle_cf.states)),
        "mean_trial_length": float(np.mean(completed_steps)),
        "physics_failure_trial_fraction": float(np.mean(failed_trials)),
        "stop_on_termination": config.stop_on_termination,
        "post_failure_transitions": post_failure_transitions,
        "terminal_label_rule": "pre_noise_next_state",
        "noise_semantics": config.noise_semantics,
        "noisy_vs_pre_noise_terminal_disagreement": (
            float(terminal_label_disagreements / transition_count)
        ),
    }
    return real.finish(), fresh_noise.finish(), oracle_cf.finish(), metadata


def normalization(real: TransitionPool) -> tuple[np.ndarray, np.ndarray]:
    """Compute state normalization only from factual data."""
    values = np.concatenate([real.states, real.next_states], axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    return mean, np.maximum(std, 1e-5)


def sample_pool(
    pool: TransitionPool,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = rng.integers(0, len(pool), size=count)
    return (
        pool.states[indices],
        pool.actions[indices],
        pool.rewards[indices],
        pool.next_states[indices],
        pool.dones[indices],
    )


def sample_training_batch(
    real: TransitionPool,
    counterfactual: TransitionPool | None,
    batch_size: int,
    cf_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a fixed-size batch with an exact real/CF composition."""
    if counterfactual is None:
        return sample_pool(real, batch_size, rng)
    counterfactual_count = round(batch_size * cf_fraction)
    real_count = batch_size - counterfactual_count
    real_batch = sample_pool(real, real_count, rng)
    cf_batch = sample_pool(counterfactual, counterfactual_count, rng)
    return tuple(
        np.concatenate([real_values, cf_values], axis=0)
        for real_values, cf_values in zip(real_batch, cf_batch)
    )


def train_offline_d3qn(
    real: TransitionPool,
    counterfactual: TransitionPool | None,
    config: ReproductionConfig,
    device: torch.device,
) -> tuple[DuelingQNetwork, dict[str, list[float]], np.ndarray, np.ndarray]:
    """Train compute-matched offline D3QN with an optional CQL regularizer."""
    if config.target_update_interval < 0:
        raise ValueError("target_update_interval must be non-negative.")
    state_mean, state_std = normalization(real)
    mean_tensor = torch.as_tensor(state_mean, device=device)
    std_tensor = torch.as_tensor(state_std, device=device)
    online = DuelingQNetwork(
        width=config.q_width,
        depth=config.q_depth,
        batch_norm=config.q_batch_norm,
    ).to(device)
    target = DuelingQNetwork(
        width=config.q_width,
        depth=config.q_depth,
        batch_norm=config.q_batch_norm,
    ).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed + 50_000)
    logs: dict[str, list[float]] = {"td_loss": [], "cql_loss": [], "q_mean": []}

    for train_step in range(config.train_steps):
        states, actions, rewards, next_states, dones = sample_training_batch(
            real,
            counterfactual,
            config.batch_size,
            config.cf_batch_fraction,
            rng,
        )
        states_tensor = (
            torch.as_tensor(states, device=device) - mean_tensor
        ) / std_tensor
        next_states_tensor = (
            torch.as_tensor(next_states, device=device) - mean_tensor
        ) / std_tensor
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
        rewards_tensor = torch.as_tensor(rewards, device=device)
        dones_tensor = torch.as_tensor(dones, device=device)

        q_values = online(states_tensor)
        selected_q = q_values.gather(1, actions_tensor[:, None]).squeeze(1)
        with torch.no_grad():
            online.eval()
            next_actions = online(next_states_tensor).argmax(dim=1, keepdim=True)
            online.train()
            next_q = target(next_states_tensor).gather(1, next_actions).squeeze(1)
            targets = rewards_tensor + config.gamma * (1.0 - dones_tensor) * next_q

        td_loss = F.smooth_l1_loss(selected_q, targets)
        cql_loss = (
            config.cql_alpha * (torch.logsumexp(q_values, dim=1) - selected_q).mean()
        )
        loss = td_loss + cql_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
        optimizer.step()

        with torch.no_grad():
            if config.target_update_interval:
                if (train_step + 1) % config.target_update_interval == 0:
                    target.load_state_dict(online.state_dict())
                    target.eval()
            else:
                for target_parameter, online_parameter in zip(
                    target.parameters(), online.parameters()
                ):
                    target_parameter.mul_(1.0 - config.target_tau)
                    target_parameter.add_(config.target_tau * online_parameter)
                for target_buffer, online_buffer in zip(
                    target.buffers(), online.buffers()
                ):
                    if target_buffer.is_floating_point():
                        target_buffer.mul_(1.0 - config.target_tau)
                        target_buffer.add_(config.target_tau * online_buffer)
                    else:
                        target_buffer.copy_(online_buffer)

        if train_step % 100 == 0 or train_step + 1 == config.train_steps:
            logs["td_loss"].append(float(td_loss.item()))
            logs["cql_loss"].append(float(cql_loss.item()))
            logs["q_mean"].append(float(q_values.mean().item()))

    return online, logs, state_mean, state_std


def evaluate_policy(
    policy: DuelingQNetwork | None,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    config: ReproductionConfig,
    device: torch.device,
    noisy: bool,
) -> list[float]:
    """Evaluate a learned policy or a uniformly random policy."""
    returns: list[float] = []
    rng = np.random.default_rng(config.seed + 90_000 + int(noisy))
    if policy is not None:
        policy.eval()
    for episode in range(config.eval_episodes):
        evaluation_seed = config.eval_seed_base + episode
        env = CartPole11Env(
            seed=evaluation_seed,
            max_steps=config.eval_horizon,
            state_noise_std=config.state_noise_std if noisy else 0.0,
            noise_semantics=config.noise_semantics,
        )
        state, _ = env.reset(seed=evaluation_seed)
        episode_return = 0.0
        for _ in range(config.eval_horizon):
            if policy is None:
                action = int(rng.integers(0, 11))
            else:
                normalized = (state - state_mean) / state_std
                with torch.no_grad():
                    values = policy(
                        torch.as_tensor(normalized, device=device).unsqueeze(0)
                    )
                action = int(values.argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(
                action,
                action_noise_std=config.action_noise_std if noisy else 0.0,
            )
            episode_return += float(reward)
            if terminated or truncated:
                break
        returns.append(episode_return)
    return returns


def summarize_episodes(values: list[float]) -> dict[str, object]:
    """Summarize evaluation episodes without treating them as training replicates."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "episodes": len(array),
        "mean": float(array.mean()),
        "episode_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "returns": [float(value) for value in array],
    }


def generate_learned_counterfactuals(
    real: TransitionPool,
    model: CTRLBiCoGAN,
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> TransitionPool:
    """Generate all alternative-action CFs by reusing each inferred latent."""
    normalized_states = (real.states - state_mean) / state_std
    normalized_next_states = (real.next_states - state_mean) / state_std
    factual_actions = (real.actions.astype(np.float32) / 10.0)[:, None]
    factual_latent = model.infer_latent(
        normalized_states,
        factual_actions,
        normalized_next_states,
    )
    alternative_actions = np.asarray(
        [
            [action for action in range(11) if action != factual]
            for factual in real.actions
        ],
        dtype=np.int64,
    )
    repeated_states = np.repeat(normalized_states, 10, axis=0).astype(np.float32)
    repeated_latent = np.repeat(factual_latent, 10, axis=0).astype(np.float32)
    flattened_actions = alternative_actions.reshape(-1)
    action_values = (flattened_actions.astype(np.float32) / 10.0)[:, None]
    predicted_normalized = model.predict(
        repeated_states,
        action_values,
        repeated_latent,
    )
    predicted_states = predicted_normalized * state_std + state_mean
    predicted_dones = np.logical_or(
        np.abs(predicted_states[:, 0]) > X_THRESHOLD,
        np.abs(predicted_states[:, 2]) > THETA_THRESHOLD,
    ).astype(np.float32)
    predicted_states[:, 0] = np.clip(predicted_states[:, 0], -4.8, 4.8)
    predicted_states[:, 2] = np.clip(predicted_states[:, 2], -0.418, 0.418)
    return TransitionPool(
        states=np.repeat(real.states, 10, axis=0),
        actions=flattened_actions,
        rewards=np.ones(len(flattened_actions), dtype=np.float32),
        next_states=predicted_states.astype(np.float32),
        dones=predicted_dones,
        trial_ids=np.repeat(real.trial_ids, 10),
    )


def counterfactual_pool_diagnostics(
    learned: TransitionPool,
    oracle: TransitionPool,
    state_std: np.ndarray,
) -> dict[str, object]:
    """Compare aligned learned alternate-action outcomes with simulator CFs."""
    if len(learned) != len(oracle):
        raise ValueError("Learned and oracle counterfactual pools must align.")
    error = learned.next_states - oracle.next_states
    normalized_error = error / state_std
    return {
        "count": len(learned),
        "normalized_mse": float(np.mean(normalized_error**2)),
        "rmse_by_dimension": [
            float(value) for value in np.sqrt(np.mean(error**2, axis=0))
        ],
        "terminal_disagreement": float(np.mean(learned.dones != oracle.dones)),
    }


def source_digest() -> str:
    """Hash the executable sources stored with each result artifact."""
    digest = hashlib.sha256()
    repository_root = Path(__file__).resolve().parents[2]
    for path in (
        Path(__file__),
        Path(__file__).with_name("bicogan_ctrl.py"),
        Path(__file__).with_name("cartpole_sanity.py"),
        repository_root / "REVISION_EXPERIMENT_PROTOCOL.md",
        repository_root / "pyproject.toml",
        repository_root / "uv.lock",
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_provenance() -> dict[str, object]:
    """Return the current commit and dirty status without requiring Git."""
    repository_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def validate_experiment_tier(config: ReproductionConfig) -> None:
    """Reject ambiguous labels for publication-facing SCM experiments."""
    expected_generators = {
        "ctrl_bicogan_reproduction": "monotonic_bicogan",
        "unconstrained_bicogan_ablation": "unconstrained",
        "triangular_flow_extension": "triangular",
    }
    if config.experiment_tier == "development":
        return
    expected = expected_generators.get(config.experiment_tier)
    if expected is None:
        raise ValueError(f"Unknown experiment tier: {config.experiment_tier}")
    if config.bicogan_generator != expected:
        raise ValueError(
            f"{config.experiment_tier} requires generator {expected}, got "
            f"{config.bicogan_generator}."
        )


def run_condition(
    name: str,
    real: TransitionPool,
    counterfactual: TransitionPool | None,
    config: ReproductionConfig,
    device: torch.device,
) -> dict[str, object]:
    """Train and evaluate one paired experimental condition."""
    set_seed(config.seed)
    policy, training_logs, state_mean, state_std = train_offline_d3qn(
        real,
        counterfactual,
        config,
        device,
    )
    clean_returns = evaluate_policy(
        policy,
        state_mean,
        state_std,
        config,
        device,
        noisy=False,
    )
    noisy_returns = evaluate_policy(
        policy,
        state_mean,
        state_std,
        config,
        device,
        noisy=True,
    )
    checkpoint_path = config.output_dir / f"{name}_seed_{config.seed}.pt"
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "state_mean": state_mean,
            "state_std": state_std,
            "config": {**asdict(config), "output_dir": str(config.output_dir)},
        },
        checkpoint_path,
    )
    return {
        "clean": summarize_episodes(clean_returns),
        "noisy": summarize_episodes(noisy_returns),
        "training": training_logs,
        "checkpoint": str(checkpoint_path),
    }


def parse_args() -> ReproductionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--experiment-tier",
        choices=(
            "ctrl_bicogan_reproduction",
            "unconstrained_bicogan_ablation",
            "triangular_flow_extension",
        ),
        required=True,
    )
    parser.add_argument("--dataset-trials", type=int, default=250)
    parser.add_argument("--validation-dataset-trials", type=int, default=50)
    parser.add_argument("--trial-horizon", type=int, default=20)
    parser.add_argument(
        "--stop-on-termination",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--target-update-interval", type=int, default=0)
    parser.add_argument("--cql-alpha", type=float, default=0.0)
    parser.add_argument("--cf-batch-fraction", type=float, default=0.5)
    parser.add_argument("--q-width", type=int, default=512)
    parser.add_argument("--q-depth", type=int, default=4)
    parser.add_argument(
        "--q-batch-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bicogan-pretrain-steps", type=int, default=2_000)
    parser.add_argument("--bicogan-steps", type=int, default=5_000)
    parser.add_argument("--bicogan-reconstruction-weight", type=float, default=10.0)
    parser.add_argument("--bicogan-extrinsic-weight", type=float, default=1.0)
    parser.add_argument("--bicogan-latent-cycle-weight", type=float, default=1.0)
    parser.add_argument(
        "--bicogan-generator",
        choices=("triangular", "monotonic_bicogan", "unconstrained"),
        required=True,
    )
    parser.add_argument("--eval-seed-base", type=int, default=300_000)
    parser.add_argument(
        "--noise-semantics",
        choices=("process", "observation"),
        default="process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/cartpole_ctrl_reproduction"),
    )
    args = parser.parse_args()
    return ReproductionConfig(
        seed=args.seed,
        experiment_tier=args.experiment_tier,
        dataset_trials=args.dataset_trials,
        validation_dataset_trials=args.validation_dataset_trials,
        trial_horizon=args.trial_horizon,
        stop_on_termination=args.stop_on_termination,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        learning_rate=args.learning_rate,
        target_tau=args.target_tau,
        target_update_interval=args.target_update_interval,
        cql_alpha=args.cql_alpha,
        cf_batch_fraction=args.cf_batch_fraction,
        q_width=args.q_width,
        q_depth=args.q_depth,
        q_batch_norm=args.q_batch_norm,
        bicogan_pretrain_steps=args.bicogan_pretrain_steps,
        bicogan_steps=args.bicogan_steps,
        bicogan_reconstruction_weight=args.bicogan_reconstruction_weight,
        bicogan_extrinsic_weight=args.bicogan_extrinsic_weight,
        bicogan_latent_cycle_weight=args.bicogan_latent_cycle_weight,
        bicogan_generator=args.bicogan_generator,
        eval_seed_base=args.eval_seed_base,
        noise_semantics=args.noise_semantics,
        output_dir=args.output_dir,
    )


def main() -> None:
    config = parse_args()
    validate_experiment_tier(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    random.seed(config.seed)
    source_sha256 = source_digest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real, fresh_noise, oracle_cf, dataset_metadata = generate_paired_dataset(config)
    validation_config = replace(
        config,
        seed=config.seed + config.validation_seed_offset,
        dataset_trials=config.validation_dataset_trials,
    )
    (
        validation_real,
        validation_fresh_noise,
        validation_oracle_cf,
        validation_metadata,
    ) = generate_paired_dataset(validation_config)
    dataset_metadata.update(
        {
            "validation_seed": validation_config.seed,
            "validation_real_transitions": validation_metadata["real_transitions"],
            "validation_post_failure_transitions": validation_metadata[
                "post_failure_transitions"
            ],
        }
    )
    state_mean, state_std = normalization(real)
    bicogan = CTRLBiCoGAN(
        BiCoGANConfig(
            pretrain_steps=config.bicogan_pretrain_steps,
            adversarial_steps=config.bicogan_steps,
            reconstruction_weight=config.bicogan_reconstruction_weight,
            extrinsic_weight=config.bicogan_extrinsic_weight,
            latent_cycle_weight=config.bicogan_latent_cycle_weight,
            generator_kind=config.bicogan_generator,
        ),
        device,
    )
    bicogan_diagnostics = bicogan.fit(
        real.states,
        real.actions,
        real.next_states,
        real.trial_ids,
        state_mean,
        state_std,
        config.seed,
        validation_states=validation_real.states,
        validation_actions=validation_real.actions,
        validation_next_states=validation_real.next_states,
        validation_trial_ids=validation_real.trial_ids + config.dataset_trials,
    )
    learned_cf = generate_learned_counterfactuals(
        real,
        bicogan,
        state_mean,
        state_std,
    )
    validation_learned_cf = generate_learned_counterfactuals(
        validation_real,
        bicogan,
        state_mean,
        state_std,
    )
    bicogan_path = config.output_dir / f"bicogan_seed_{config.seed}.pt"
    bicogan.save(bicogan_path)

    payload: dict[str, object] = {
        "artifact_schema": "ctrl-cartpole-revision-v4",
        "experiment_tier": config.experiment_tier,
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "command": [sys.executable, *sys.argv],
        "git": git_provenance(),
        "device": str(device),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "gymnasium": importlib.metadata.version("gymnasium"),
            "scipy": importlib.metadata.version("scipy"),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "source_sha256": source_sha256,
        "dataset": dataset_metadata,
        "bicogan": {
            "diagnostics": bicogan_diagnostics,
            "counterfactual_diagnostics": {
                "training": counterfactual_pool_diagnostics(
                    learned_cf,
                    oracle_cf,
                    state_std,
                ),
                "external_validation": counterfactual_pool_diagnostics(
                    validation_learned_cf,
                    validation_oracle_cf,
                    state_std,
                ),
                "fresh_noise_external_validation": counterfactual_pool_diagnostics(
                    validation_fresh_noise,
                    validation_oracle_cf,
                    state_std,
                ),
            },
            "checkpoint": str(bicogan_path),
            "learned_cf_transitions": len(learned_cf),
        },
        "random": {
            "clean": summarize_episodes(
                evaluate_policy(
                    None,
                    state_mean,
                    state_std,
                    config,
                    device,
                    noisy=False,
                )
            ),
            "noisy": summarize_episodes(
                evaluate_policy(
                    None,
                    state_mean,
                    state_std,
                    config,
                    device,
                    noisy=True,
                )
            ),
        },
    }
    payload["real"] = run_condition(
        "real",
        real,
        None,
        config,
        device,
    )
    payload["fresh_noise"] = run_condition(
        "fresh_noise",
        real,
        fresh_noise,
        config,
        device,
    )
    payload["oracle_cf"] = run_condition(
        "oracle_cf",
        real,
        oracle_cf,
        config,
        device,
    )
    payload["learned_cf"] = run_condition(
        "learned_cf",
        real,
        learned_cf,
        config,
        device,
    )
    output_path = config.output_dir / f"reproduction_seed_{config.seed}.json"
    output_path.write_text(json.dumps(payload, indent=2))
    print(
        f"[seed {config.seed}] real clean={payload['real']['clean']['mean']:.2f} "
        f"learned clean={payload['learned_cf']['clean']['mean']:.2f} "
        f"oracle clean={payload['oracle_cf']['clean']['mean']:.2f}"
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
