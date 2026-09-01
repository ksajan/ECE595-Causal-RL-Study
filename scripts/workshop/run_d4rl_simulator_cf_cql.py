#!/usr/bin/env python3
"""Run a compute-matched D4RL simulator-residual CQL experiment.

This experiment restores MuJoCo state from D4RL ``infos/qpos`` and
``infos/qvel`` arrays, simulates factual and intervened one-step transitions,
and transfers additive factual residuals to alternate actions.  It is a
simulator-residual SCM approximation, not an exact oracle or a faithful CTRL
implementation.

Unlike ``MDPDataset``, :class:`DirectArrayReplayBuffer` constructs d3rlpy
transitions directly from explicit HDF5 ``next_observations``.  Every arm has
the same transition count, batch size, update count, and evaluation seed bank.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Literal, cast
from urllib.request import urlretrieve

import gymnasium as gym
import numpy as np

try:
    from d3rlpy.dataset import ReplayBufferBase as _ReplayBufferBase
except ImportError:

    class _ReplayBufferBase:  # type: ignore[no-redef]
        """Import-time fallback so non-training utilities remain usable."""


DatasetName = Literal[
    "halfcheetah-medium-v2",
    "hopper-medium-v2",
    "walker2d-medium-v2",
]
Variant = Literal["real", "simulator_mean", "fresh_residual", "factual_residual"]
Mode = Literal["validate", "augment", "train", "run"]

VARIANTS: tuple[Variant, ...] = (
    "real",
    "simulator_mean",
    "fresh_residual",
    "factual_residual",
)
SYNTHETIC_VARIANTS: tuple[Variant, ...] = VARIANTS[1:]

D4RL_URLS: dict[str, str] = {
    "halfcheetah-medium-v2": (
        "https://rail.eecs.berkeley.edu/datasets/offline_rl/"
        "gym_mujoco_v2/halfcheetah_medium-v2.hdf5"
    ),
    "hopper-medium-v2": (
        "https://rail.eecs.berkeley.edu/datasets/offline_rl/"
        "gym_mujoco_v2/hopper_medium-v2.hdf5"
    ),
    "walker2d-medium-v2": (
        "https://rail.eecs.berkeley.edu/datasets/offline_rl/"
        "gym_mujoco_v2/walker2d_medium-v2.hdf5"
    ),
}

EVAL_ENVS: dict[str, str] = {
    "halfcheetah-medium-v2": "HalfCheetah-v4",
    "hopper-medium-v2": "Hopper-v4",
    "walker2d-medium-v2": "Walker2d-v4",
}

# Official D4RL v2 reference returns from d4rl/infos.py.
D4RL_REFERENCE_RETURNS: dict[str, tuple[float, float]] = {
    "halfcheetah-medium-v2": (-280.178953, 12135.0),
    "hopper-medium-v2": (-20.272305, 3234.3),
    "walker2d-medium-v2": (1.629008, 4592.3),
}


@dataclass(frozen=True)
class TransitionArrays:
    """Explicit D4RL transitions and simulator state, with leading size ``N``."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray

    @property
    def size(self) -> int:
        """Return the number of transitions."""

        return int(self.observations.shape[0])

    def validate(self) -> None:
        """Reject malformed, non-finite, or shifted-transition input arrays."""

        arrays = {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "terminals": self.terminals,
            "timeouts": self.timeouts,
            "qpos": self.qpos,
            "qvel": self.qvel,
        }
        sizes = {name: int(value.shape[0]) for name, value in arrays.items()}
        if not sizes or len(set(sizes.values())) != 1:
            raise ValueError(f"All D4RL arrays must share one leading size: {sizes}")
        if self.size < 2:
            raise ValueError("At least two transitions are required")
        expected_2d = {
            "observations": self.observations,
            "actions": self.actions,
            "next_observations": self.next_observations,
            "qpos": self.qpos,
            "qvel": self.qvel,
        }
        for name, value in expected_2d.items():
            if value.ndim != 2:
                raise ValueError(f"{name} must have shape (N, D), got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        for name, value in {
            "rewards": self.rewards,
            "terminals": self.terminals,
            "timeouts": self.timeouts,
        }.items():
            if value.shape != (self.size,):
                raise ValueError(f"{name} must have shape (N,), got {value.shape}")
        if self.next_observations.shape != self.observations.shape:
            raise ValueError("observations and next_observations must match in shape")
        if not np.all(np.isfinite(self.rewards)):
            raise ValueError("rewards contains non-finite values")
        if np.any(self.terminals & self.timeouts):
            raise ValueError("A transition cannot be terminal and timeout together")


@dataclass(frozen=True)
class TrainingArrays:
    """Arrays consumed directly by offline CQL, each with leading size ``N``."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray

    @property
    def size(self) -> int:
        """Return the number of transitions."""

        return int(self.observations.shape[0])


@dataclass(frozen=True)
class GateConfig:
    """Maximum normalized replay errors accepted before augmentation."""

    restored_observation_nrmse: float = 0.05
    factual_next_nrmse: float = 0.10
    factual_reward_nmae: float = 0.10
    terminal_disagreement: float = 0.01
    action_clip_fraction: float = 0.001

    def validate(self) -> None:
        """Reject negative or non-finite thresholds."""

        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Gate threshold {name} must be finite and >= 0")


@dataclass(frozen=True)
class ReplayDiagnostics:
    """One-step factual simulator replay metrics and gate decision."""

    sample_count: int
    restored_observation_rmse: float
    restored_observation_nrmse: float
    factual_next_rmse: float
    factual_next_nrmse: float
    factual_reward_mae: float
    factual_reward_nmae: float
    terminal_disagreement: float
    action_clip_fraction: float
    failed_checks: tuple[str, ...]
    passed: bool


@dataclass(frozen=True)
class SimulatorTransition:
    """One deterministic MuJoCo branch from restored D4RL simulator state."""

    restored_observation: np.ndarray
    next_observation: np.ndarray
    reward: float
    terminated: bool


@dataclass(frozen=True)
class SyntheticArrays:
    """One synthetic candidate per factual transition."""

    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray


def _require_d3rlpy() -> Any:
    """Import d3rlpy 2.8.x or raise an actionable dependency error."""

    try:
        import d3rlpy
    except ImportError as exc:
        raise RuntimeError(
            "d3rlpy is required for replay construction and training; install "
            "d3rlpy==2.8.0"
        ) from exc
    if not str(d3rlpy.__version__).startswith("2.8."):
        raise RuntimeError(
            f"This runner is validated for d3rlpy 2.8.x, got {d3rlpy.__version__}"
        )
    return d3rlpy


class DirectArrayReplayBuffer(_ReplayBufferBase):  # type: ignore[misc]
    """Read-only d3rlpy replay over explicit transition arrays.

    This class intentionally does not infer ``next_observation`` from row
    order.  A sampled transition at index ``i`` always contains the exact
    ``next_observations[i]`` supplied by the D4RL HDF5 file or synthetic arm.
    Rewards have shape ``(1,)`` and observations/actions retain their feature
    dimensions, matching d3rlpy 2.8's :class:`Transition` contract.
    """

    def __init__(self, arrays: TrainingArrays, seed: int) -> None:
        d3rlpy = _require_d3rlpy()
        arrays = _as_contiguous_training_arrays(arrays)
        _validate_training_arrays(arrays)
        self._arrays = arrays
        self._rng = np.random.default_rng(seed)
        self._dataset_info = d3rlpy.dataset.DatasetInfo(
            observation_signature=d3rlpy.dataset.Signature(
                dtype=[np.dtype(np.float32)],
                shape=[arrays.observations.shape[1:]],
            ),
            action_signature=d3rlpy.dataset.Signature(
                dtype=[np.dtype(np.float32)],
                shape=[arrays.actions.shape[1:]],
            ),
            reward_signature=d3rlpy.dataset.Signature(
                dtype=[np.dtype(np.float32)], shape=[(1,)]
            ),
            action_space=d3rlpy.constants.ActionSpace.CONTINUOUS,
            action_size=int(arrays.actions.shape[1]),
        )

    def transition_at(self, index: int) -> Any:
        """Construct the exact d3rlpy transition stored at ``index``."""

        if index < 0 or index >= self.transition_count:
            raise IndexError(f"Transition index {index} is out of range")
        d3rlpy = _require_d3rlpy()
        reward = np.asarray([self._arrays.rewards[index]], dtype=np.float32)
        action = self._arrays.actions[index].copy()
        return d3rlpy.dataset.Transition(
            observation=self._arrays.observations[index].copy(),
            action=action,
            reward=reward,
            next_observation=self._arrays.next_observations[index].copy(),
            next_action=np.zeros_like(action, dtype=np.float32),
            terminal=float(self._arrays.terminals[index]),
            interval=1,
            rewards_to_go=reward.reshape(1, 1),
        )

    def sample_transition(self) -> Any:
        """Sample one exact transition uniformly with the buffer RNG."""

        index = int(self._rng.integers(0, self.transition_count))
        return self.transition_at(index)

    def sample_transition_batch(self, batch_size: int) -> Any:
        """Sample a d3rlpy minibatch uniformly with replacement."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        d3rlpy = _require_d3rlpy()
        indices = self._rng.integers(0, self.transition_count, size=batch_size)
        transitions = [self.transition_at(int(index)) for index in indices]
        return d3rlpy.dataset.TransitionMiniBatch.from_transitions(transitions)

    def append(self, observation: Any, action: Any, reward: Any) -> None:
        """Reject writes because publication datasets are immutable."""

        del observation, action, reward
        raise RuntimeError("DirectArrayReplayBuffer is read-only")

    def append_episode(self, episode: Any) -> None:
        """Reject episode writes because publication datasets are immutable."""

        del episode
        raise RuntimeError("DirectArrayReplayBuffer is read-only")

    def clip_episode(self, terminated: bool) -> None:
        """Reject online episode operations."""

        del terminated
        raise RuntimeError("DirectArrayReplayBuffer is offline and read-only")

    def sample_trajectory(self, length: int) -> Any:
        """Reject trajectory sampling, which CQL does not use."""

        del length
        raise NotImplementedError("Trajectory sampling is not supported")

    def sample_trajectory_batch(self, batch_size: int, length: int) -> Any:
        """Reject trajectory sampling, which CQL does not use."""

        del batch_size, length
        raise NotImplementedError("Trajectory sampling is not supported")

    def dump(self, file: BinaryIO) -> None:
        """Serialize explicit arrays to an open binary file."""

        np.savez_compressed(
            file,
            observations=self._arrays.observations,
            actions=self._arrays.actions,
            rewards=self._arrays.rewards,
            next_observations=self._arrays.next_observations,
            terminals=self._arrays.terminals,
            timeouts=self._arrays.timeouts,
        )

    @classmethod
    def from_episode_generator(cls, *args: Any, **kwargs: Any) -> Any:
        """Reject implicit episode conversion to preserve explicit next states."""

        del args, kwargs
        raise NotImplementedError("Use DirectArrayReplayBuffer(arrays, seed)")

    @classmethod
    def load(cls, *args: Any, **kwargs: Any) -> Any:
        """Reject d3rlpy episode loading to preserve explicit next states."""

        del args, kwargs
        raise NotImplementedError("Load arrays explicitly, then construct the buffer")

    @property
    def episodes(self) -> Sequence[Any]:
        """Return no inferred episodes; default CQL scalers do not require them."""

        return ()

    def size(self) -> int:
        """Return zero inferred episodes."""

        return 0

    @property
    def buffer(self) -> Any:
        """Reject access to an episode backing store."""

        raise NotImplementedError("This replay is backed directly by arrays")

    @property
    def transition_count(self) -> int:
        """Return the exact number of explicit transitions."""

        return self._arrays.size

    @property
    def transition_picker(self) -> Any:
        """Return no picker because transitions are constructed directly."""

        return None

    @property
    def trajectory_slicer(self) -> Any:
        """Return no trajectory slicer because CQL samples transitions."""

        return None

    @property
    def dataset_info(self) -> Any:
        """Return d3rlpy signatures for continuous-control CQL."""

        return self._dataset_info


class MujocoStateReplayer:
    """Restore D4RL MuJoCo states and execute deterministic one-step branches."""

    def __init__(self, env_id: str, seed: int) -> None:
        self.env = gym.make(env_id)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        if not isinstance(self.env.action_space, gym.spaces.Box):
            self.env.close()
            raise TypeError(f"{env_id} does not expose a continuous Box action")
        base = self.env.unwrapped
        if not hasattr(base, "set_state") or not hasattr(base, "_get_obs"):
            self.env.close()
            raise RuntimeError(f"{env_id} does not expose MuJoCo state restoration")

    @property
    def action_space(self) -> gym.spaces.Box:
        """Return finite environment action bounds."""

        return cast(gym.spaces.Box, self.env.action_space)

    def close(self) -> None:
        """Release MuJoCo resources."""

        self.env.close()

    def simulate(
        self, qpos: np.ndarray, qvel: np.ndarray, action: np.ndarray
    ) -> SimulatorTransition:
        """Restore one state and step one clipped action in environment units."""

        base = self.env.unwrapped
        model = getattr(base, "model", None)
        if model is None:
            raise RuntimeError("MuJoCo model is unavailable after environment creation")
        if qpos.shape != (int(model.nq),) or qvel.shape != (int(model.nv),):
            raise ValueError(
                "D4RL qpos/qvel shapes do not match evaluation model: "
                f"{qpos.shape}/{qvel.shape} vs {(model.nq,)}/{(model.nv,)}"
            )
        base.set_state(
            np.asarray(qpos, dtype=np.float64),
            np.asarray(qvel, dtype=np.float64),
        )
        _reset_time_limit_counters(self.env)
        restored = np.asarray(base._get_obs(), dtype=np.float32).copy()
        bounded_action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        next_observation, reward, terminated, _truncated, _info = self.env.step(
            bounded_action
        )
        return SimulatorTransition(
            restored_observation=restored,
            next_observation=np.asarray(next_observation, dtype=np.float32).copy(),
            reward=float(reward),
            terminated=bool(terminated),
        )


def _reset_time_limit_counters(env: gym.Env[Any, Any]) -> None:
    """Set all Gymnasium TimeLimit counters to zero before a one-step branch."""

    current: Any = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, gym.wrappers.TimeLimit):
            current._elapsed_steps = 0
        current = getattr(current, "env", None)
        if current is None:
            break


def _as_contiguous_training_arrays(arrays: TrainingArrays) -> TrainingArrays:
    """Cast CQL inputs to stable contiguous dtypes."""

    return TrainingArrays(
        observations=np.ascontiguousarray(arrays.observations, dtype=np.float32),
        actions=np.ascontiguousarray(arrays.actions, dtype=np.float32),
        rewards=np.ascontiguousarray(arrays.rewards, dtype=np.float32),
        next_observations=np.ascontiguousarray(
            arrays.next_observations, dtype=np.float32
        ),
        terminals=np.ascontiguousarray(arrays.terminals, dtype=np.bool_),
        timeouts=np.ascontiguousarray(arrays.timeouts, dtype=np.bool_),
    )


def _validate_training_arrays(arrays: TrainingArrays) -> None:
    """Validate direct replay shapes and finite values."""

    size = arrays.size
    if size < 2:
        raise ValueError("At least two training transitions are required")
    if arrays.observations.ndim != 2 or arrays.next_observations.ndim != 2:
        raise ValueError("Observations must be two-dimensional")
    if arrays.observations.shape != arrays.next_observations.shape:
        raise ValueError("Explicit current and next observations must match in shape")
    if arrays.actions.ndim != 2 or arrays.actions.shape[0] != size:
        raise ValueError("Actions must have shape (N, action_dim)")
    for name, value in {
        "rewards": arrays.rewards,
        "terminals": arrays.terminals,
        "timeouts": arrays.timeouts,
    }.items():
        if value.shape != (size,):
            raise ValueError(f"{name} must have shape (N,), got {value.shape}")
    if np.any(arrays.terminals & arrays.timeouts):
        raise ValueError("A transition cannot be terminal and timeout together")
    for name, value in {
        "observations": arrays.observations,
        "actions": arrays.actions,
        "rewards": arrays.rewards,
        "next_observations": arrays.next_observations,
    }.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")


def normalized_d4rl_score(dataset_name: str, episode_return: float) -> float:
    """Convert an undiscounted raw return to the official D4RL v2 score."""

    try:
        random_return, expert_return = D4RL_REFERENCE_RETURNS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"No official D4RL reference for {dataset_name!r}") from exc
    return (
        100.0
        * (float(episode_return) - random_return)
        / (expert_return - random_return)
    )


def sample_alternative_actions(
    factual_actions: np.ndarray,
    action_space: gym.spaces.Box,
    rng: np.random.Generator,
    scale: float,
) -> np.ndarray:
    """Perturb continuous actions in normalized coordinates and clip to bounds.

    Args:
        factual_actions: Actions of shape ``(N, action_dim)`` in environment units.
        action_space: Finite Box bounds for the target MuJoCo environment.
        rng: Seeded NumPy random generator.
        scale: Gaussian standard deviation in normalized ``[-1, 1]`` units.
    """

    actions = np.asarray(factual_actions, dtype=np.float64)
    low = np.asarray(action_space.low, dtype=np.float64)
    high = np.asarray(action_space.high, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1:] != action_space.shape:
        raise ValueError(
            f"Actions {actions.shape} do not match Box shape {action_space.shape}"
        )
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Intervention scale must be finite and positive")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("Action bounds must be finite")
    width = high - low
    if np.any(width <= 0):
        raise ValueError("Action bounds must have positive width")
    normalized = 2.0 * (np.clip(actions, low, high) - low) / width - 1.0
    noise = rng.normal(0.0, scale, size=actions.shape)
    alternatives = low + 0.5 * (np.clip(normalized + noise, -1.0, 1.0) + 1.0) * width
    unchanged = np.all(np.isclose(alternatives, actions, atol=1e-7, rtol=0.0), axis=1)
    if np.any(unchanged):
        dimensions = rng.integers(0, actions.shape[1], size=int(unchanged.sum()))
        for row, dimension in zip(np.flatnonzero(unchanged), dimensions, strict=True):
            direction = -1.0 if normalized[row, dimension] >= 0.0 else 1.0
            delta = max(scale, 1e-3) * direction
            changed = np.clip(normalized[row, dimension] + delta, -1.0, 1.0)
            alternatives[row, dimension] = (
                low[dimension] + 0.5 * (changed + 1.0) * width[dimension]
            )
    return np.asarray(np.clip(alternatives, low, high), dtype=np.float32)


def deranged_permutation(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a deterministic random permutation with no fixed points."""

    if size < 2:
        raise ValueError("A fresh-residual permutation needs at least two rows")
    order = rng.permutation(size)
    source = np.empty(size, dtype=np.int64)
    source[order] = np.roll(order, 1)
    if np.any(source == np.arange(size)):
        raise RuntimeError("Internal error: residual permutation has fixed points")
    return source


def compose_residual_variants(
    simulator_next: np.ndarray,
    simulator_rewards: np.ndarray,
    state_residuals: np.ndarray,
    reward_residuals: np.ndarray,
    fresh_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compose factual and independently permuted additive residual branches."""

    simulator_next = np.asarray(simulator_next, dtype=np.float32)
    simulator_rewards = np.asarray(simulator_rewards, dtype=np.float32)
    state_residuals = np.asarray(state_residuals, dtype=np.float32)
    reward_residuals = np.asarray(reward_residuals, dtype=np.float32)
    fresh_indices = np.asarray(fresh_indices, dtype=np.int64)
    size = simulator_next.shape[0]
    if simulator_next.shape != state_residuals.shape:
        raise ValueError("Simulator next states and state residuals must match")
    if simulator_rewards.shape != (size,) or reward_residuals.shape != (size,):
        raise ValueError("Reward and reward residual arrays must have shape (N,)")
    if fresh_indices.shape != (size,) or np.any(
        (fresh_indices < 0) | (fresh_indices >= size)
    ):
        raise ValueError("fresh_indices must contain valid source rows")
    factual_next = simulator_next + state_residuals
    factual_rewards = simulator_rewards + reward_residuals
    fresh_next = simulator_next + state_residuals[fresh_indices]
    fresh_rewards = simulator_rewards + reward_residuals[fresh_indices]
    return factual_next, factual_rewards, fresh_next, fresh_rewards


def load_d4rl_hdf5(path: Path, max_transitions: int | None = None) -> TransitionArrays:
    """Load strict D4RL arrays, including explicit next states and MuJoCo state."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load D4RL datasets") from exc
    if not path.is_file():
        raise FileNotFoundError(f"D4RL HDF5 file does not exist: {path}")
    required = (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "terminals",
        "timeouts",
        "infos/qpos",
        "infos/qvel",
    )
    with h5py.File(path, "r") as handle:
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(
                "D4RL file is missing arrays required for explicit replay: "
                + ", ".join(missing)
            )
        total = int(handle["observations"].shape[0])
        if max_transitions is not None:
            if max_transitions < 2:
                raise ValueError("max_transitions must be at least 2")
            total = min(total, max_transitions)
        rows = slice(0, total)
        arrays = TransitionArrays(
            observations=np.asarray(handle["observations"][rows], dtype=np.float32),
            actions=np.asarray(handle["actions"][rows], dtype=np.float32),
            rewards=np.asarray(handle["rewards"][rows], dtype=np.float32).reshape(-1),
            next_observations=np.asarray(
                handle["next_observations"][rows], dtype=np.float32
            ),
            terminals=np.asarray(handle["terminals"][rows], dtype=np.bool_).reshape(-1),
            timeouts=np.asarray(handle["timeouts"][rows], dtype=np.bool_).reshape(-1),
            qpos=np.asarray(handle["infos/qpos"][rows], dtype=np.float64),
            qvel=np.asarray(handle["infos/qvel"][rows], dtype=np.float64),
        )
    if arrays.actions.ndim == 1:
        arrays = replace(arrays, actions=arrays.actions[:, None])
    arrays.validate()
    return arrays


def dataset_to_training_arrays(dataset: TransitionArrays) -> TrainingArrays:
    """Drop simulator-only state arrays while preserving explicit transitions."""

    return _as_contiguous_training_arrays(
        TrainingArrays(
            observations=dataset.observations,
            actions=dataset.actions,
            rewards=dataset.rewards,
            next_observations=dataset.next_observations,
            terminals=dataset.terminals,
            timeouts=dataset.timeouts,
        )
    )


def compute_replay_diagnostics(
    dataset: TransitionArrays,
    replayer: MujocoStateReplayer,
    sample_count: int,
    seed: int,
    gate: GateConfig,
) -> ReplayDiagnostics:
    """Replay seeded factual transitions and evaluate version-compatibility gates."""

    gate.validate()
    if sample_count <= 0:
        raise ValueError("diagnostic sample_count must be positive")
    count = min(sample_count, dataset.size)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(dataset.size, size=count, replace=False))
    restored = np.empty((count, dataset.observations.shape[1]), dtype=np.float32)
    simulated_next = np.empty_like(restored)
    simulated_rewards = np.empty(count, dtype=np.float32)
    simulated_terminals = np.empty(count, dtype=np.bool_)
    clipped = 0
    low = replayer.action_space.low
    high = replayer.action_space.high
    for output_row, source_row in enumerate(indices):
        action = dataset.actions[source_row]
        clipped += int(np.any((action < low) | (action > high)))
        transition = replayer.simulate(
            dataset.qpos[source_row], dataset.qvel[source_row], action
        )
        restored[output_row] = transition.restored_observation
        simulated_next[output_row] = transition.next_observation
        simulated_rewards[output_row] = transition.reward
        simulated_terminals[output_row] = transition.terminated

    obs_scale = np.maximum(np.std(dataset.observations, axis=0), 1e-6)
    next_scale = np.maximum(np.std(dataset.next_observations, axis=0), 1e-6)
    reward_scale = max(float(np.std(dataset.rewards)), 1e-6)
    restored_error = restored - dataset.observations[indices]
    next_error = simulated_next - dataset.next_observations[indices]
    reward_error = simulated_rewards - dataset.rewards[indices]
    metrics = {
        "restored_observation_nrmse": float(
            np.sqrt(np.mean(np.square(restored_error / obs_scale)))
        ),
        "factual_next_nrmse": float(
            np.sqrt(np.mean(np.square(next_error / next_scale)))
        ),
        "factual_reward_nmae": float(np.mean(np.abs(reward_error)) / reward_scale),
        "terminal_disagreement": float(
            np.mean(simulated_terminals != dataset.terminals[indices])
        ),
        "action_clip_fraction": float(clipped / count),
    }
    thresholds = asdict(gate)
    failed = tuple(
        name for name, value in metrics.items() if value > float(thresholds[name])
    )
    return ReplayDiagnostics(
        sample_count=count,
        restored_observation_rmse=float(np.sqrt(np.mean(np.square(restored_error)))),
        restored_observation_nrmse=metrics["restored_observation_nrmse"],
        factual_next_rmse=float(np.sqrt(np.mean(np.square(next_error)))),
        factual_next_nrmse=metrics["factual_next_nrmse"],
        factual_reward_mae=float(np.mean(np.abs(reward_error))),
        factual_reward_nmae=metrics["factual_reward_nmae"],
        terminal_disagreement=metrics["terminal_disagreement"],
        action_clip_fraction=metrics["action_clip_fraction"],
        failed_checks=failed,
        passed=not failed,
    )


def build_augmentation_cache(
    dataset: TransitionArrays,
    replayer: MujocoStateReplayer,
    dataset_name: str,
    dataset_sha256: str,
    cache_path: Path,
    augmentation_seed: int,
    intervention_scale: float,
    overwrite: bool,
) -> None:
    """Simulate all factual/alternate branches and atomically cache residuals."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to write augmentation caches") from exc
    if cache_path.exists() and not overwrite:
        validate_cache_metadata(
            cache_path,
            dataset_name,
            dataset_sha256,
            dataset.size,
            augmentation_seed,
            intervention_scale,
        )
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(augmentation_seed + 17_171)
    alternatives = sample_alternative_actions(
        dataset.actions, replayer.action_space, rng, intervention_scale
    )
    size = dataset.size
    observation_dim = dataset.observations.shape[1]
    simulator_next = np.empty((size, observation_dim), dtype=np.float32)
    simulator_rewards = np.empty(size, dtype=np.float32)
    alternate_terminals = np.empty(size, dtype=np.bool_)
    state_residuals = np.empty_like(simulator_next)
    reward_residuals = np.empty(size, dtype=np.float32)
    for index in range(size):
        factual = replayer.simulate(
            dataset.qpos[index], dataset.qvel[index], dataset.actions[index]
        )
        alternate = replayer.simulate(
            dataset.qpos[index], dataset.qvel[index], alternatives[index]
        )
        simulator_next[index] = alternate.next_observation
        simulator_rewards[index] = alternate.reward
        alternate_terminals[index] = alternate.terminated
        state_residuals[index] = (
            dataset.next_observations[index] - factual.next_observation
        )
        reward_residuals[index] = dataset.rewards[index] - factual.reward
        if (index + 1) % 25_000 == 0 or index + 1 == size:
            print(f"[augment] simulated {index + 1:,}/{size:,} transitions")
    fresh_indices = deranged_permutation(size, rng)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    compression = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = 2
            handle.attrs["method"] = "simulator_residual_scm_approximation"
            handle.attrs["dataset_name"] = dataset_name
            handle.attrs["dataset_sha256"] = dataset_sha256
            handle.attrs["transition_count"] = size
            handle.attrs["augmentation_seed"] = augmentation_seed
            handle.attrs["intervention_scale"] = intervention_scale
            handle.attrs["created_at_utc"] = _utc_now()
            handle.create_dataset(
                "alternative_actions", data=alternatives, **compression
            )
            handle.create_dataset(
                "simulator_next_observations", data=simulator_next, **compression
            )
            handle.create_dataset(
                "simulator_rewards", data=simulator_rewards, **compression
            )
            handle.create_dataset(
                "alternate_terminals", data=alternate_terminals, **compression
            )
            handle.create_dataset(
                "state_residuals", data=state_residuals, **compression
            )
            handle.create_dataset(
                "reward_residuals", data=reward_residuals, **compression
            )
            handle.create_dataset("fresh_indices", data=fresh_indices, **compression)
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_cache_metadata(
    cache_path: Path,
    dataset_name: str,
    dataset_sha256: str,
    transition_count: int,
    augmentation_seed: int,
    intervention_scale: float,
) -> None:
    """Reject a stale cache whose source or intervention configuration changed."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to validate augmentation caches") from exc
    with h5py.File(cache_path, "r") as handle:
        expected: dict[str, Any] = {
            "dataset_name": dataset_name,
            "dataset_sha256": dataset_sha256,
            "transition_count": transition_count,
            "augmentation_seed": augmentation_seed,
        }
        for name, value in expected.items():
            if handle.attrs.get(name) != value:
                raise RuntimeError(
                    f"Stale augmentation cache: {name}={handle.attrs.get(name)!r}, "
                    f"expected {value!r}; rebuild with --overwrite-cache"
                )
        cached_scale = float(handle.attrs.get("intervention_scale", np.nan))
        if not np.isclose(cached_scale, intervention_scale, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "Stale augmentation cache intervention scale; rebuild with "
                "--overwrite-cache"
            )


def load_synthetic_arrays(cache_path: Path, variant: Variant) -> SyntheticArrays:
    """Materialize one synthetic candidate arm from the shared residual cache."""

    if variant not in SYNTHETIC_VARIANTS:
        raise ValueError(f"Synthetic cache cannot construct variant {variant!r}")
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load augmentation caches") from exc
    with h5py.File(cache_path, "r") as handle:
        actions = np.asarray(handle["alternative_actions"], dtype=np.float32)
        simulator_next = np.asarray(
            handle["simulator_next_observations"], dtype=np.float32
        )
        simulator_rewards = np.asarray(handle["simulator_rewards"], dtype=np.float32)
        terminals = np.asarray(handle["alternate_terminals"], dtype=np.bool_)
        state_residuals = np.asarray(handle["state_residuals"], dtype=np.float32)
        reward_residuals = np.asarray(handle["reward_residuals"], dtype=np.float32)
        fresh_indices = np.asarray(handle["fresh_indices"], dtype=np.int64)
    if variant == "simulator_mean":
        next_observations = simulator_next
        rewards = simulator_rewards
    else:
        factual_next, factual_rewards, fresh_next, fresh_rewards = (
            compose_residual_variants(
                simulator_next,
                simulator_rewards,
                state_residuals,
                reward_residuals,
                fresh_indices,
            )
        )
        if variant == "factual_residual":
            next_observations, rewards = factual_next, factual_rewards
        else:
            next_observations, rewards = fresh_next, fresh_rewards
    return SyntheticArrays(
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        terminals=terminals,
        timeouts=np.zeros_like(terminals),
    )


def make_fixed_size_training_arm(
    dataset: TransitionArrays,
    variant: Variant,
    cf_fraction: float,
    augmentation_seed: int,
    synthetic: SyntheticArrays | None = None,
) -> tuple[TrainingArrays, np.ndarray]:
    """Replace a shared seeded subset with synthetic candidates, preserving ``N``."""

    if not 0.0 <= cf_fraction <= 1.0:
        raise ValueError("cf_fraction must be in [0, 1]")
    real = dataset_to_training_arrays(dataset)
    if variant == "real":
        return real, np.zeros(dataset.size, dtype=np.bool_)
    if synthetic is None:
        raise ValueError(f"Variant {variant} requires synthetic arrays")
    if synthetic.actions.shape != dataset.actions.shape:
        raise ValueError("Synthetic actions do not match factual action shape")
    if synthetic.next_observations.shape != dataset.next_observations.shape:
        raise ValueError("Synthetic next observations do not match factual shape")
    count = round(cf_fraction * dataset.size)
    rng = np.random.default_rng(augmentation_seed + 91_019)
    mask = np.zeros(dataset.size, dtype=np.bool_)
    if count:
        mask[rng.choice(dataset.size, size=count, replace=False)] = True
    actions = real.actions.copy()
    rewards = real.rewards.copy()
    next_observations = real.next_observations.copy()
    terminals = real.terminals.copy()
    timeouts = real.timeouts.copy()
    actions[mask] = synthetic.actions[mask]
    rewards[mask] = synthetic.rewards[mask]
    next_observations[mask] = synthetic.next_observations[mask]
    terminals[mask] = synthetic.terminals[mask]
    # The alternate action occurs at the same horizon as its factual source.
    timeouts[mask] = dataset.timeouts[mask]
    terminals[mask & timeouts] = False
    arrays = _as_contiguous_training_arrays(
        TrainingArrays(
            observations=real.observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            timeouts=timeouts,
        )
    )
    _validate_training_arrays(arrays)
    return arrays, mask


def evaluate_policy(
    algorithm: Any,
    env_id: str,
    dataset_name: str,
    episodes: int,
    seed_base: int,
) -> dict[str, Any]:
    """Evaluate deterministic actions on one fixed seed bank and both score scales."""

    if episodes <= 0:
        raise ValueError("eval episodes must be positive")
    env = gym.make(env_id)
    env.action_space.seed(seed_base)
    returns: list[float] = []
    lengths: list[int] = []
    seeds = [seed_base + episode for episode in range(episodes)]
    try:
        for eval_seed in seeds:
            observation, _ = env.reset(seed=eval_seed)
            terminated = truncated = False
            total = 0.0
            length = 0
            while not (terminated or truncated):
                batch = np.asarray(observation, dtype=np.float32)[None, ...]
                action = np.asarray(algorithm.predict(batch)[0], dtype=np.float32)
                observation, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                length += 1
            returns.append(total)
            lengths.append(length)
    finally:
        env.close()
    normalized = [normalized_d4rl_score(dataset_name, value) for value in returns]
    return {
        "environment_id": env_id,
        "evaluation_seeds": seeds,
        "episode_returns_raw": returns,
        "episode_returns_normalized_d4rl": normalized,
        "episode_lengths": lengths,
        "raw_mean": float(np.mean(returns)),
        "raw_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "normalized_mean": float(np.mean(normalized)),
        "normalized_std": (
            float(np.std(normalized, ddof=1)) if len(normalized) > 1 else 0.0
        ),
        "normalization_reference": {
            "random_return": D4RL_REFERENCE_RETURNS[dataset_name][0],
            "expert_return": D4RL_REFERENCE_RETURNS[dataset_name][1],
            "formula": "100 * (raw - random) / (expert - random)",
        },
    }


def train_cql(
    arrays: TrainingArrays,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Any, list[tuple[int, dict[str, float]]]]:
    """Train d3rlpy 2.8 CQL for the exact configured update count."""

    d3rlpy = _require_d3rlpy()
    if args.steps <= 0 or args.steps_per_epoch <= 0:
        raise ValueError("steps and steps_per_epoch must be positive")
    if args.steps % args.steps_per_epoch:
        raise ValueError("steps must be divisible by steps_per_epoch")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _seed_everything(args.seed, d3rlpy)
    replay = DirectArrayReplayBuffer(arrays, seed=args.seed)
    device: str | bool
    if args.device == "auto":
        try:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else False
        except ImportError:
            device = False
    elif args.device == "cpu":
        device = False
    else:
        device = args.device
    config = d3rlpy.algos.CQLConfig(
        batch_size=args.batch_size,
        gamma=args.gamma,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        temp_learning_rate=args.temp_learning_rate,
        alpha_learning_rate=args.alpha_learning_rate,
        actor_encoder_factory=d3rlpy.models.encoders.VectorEncoderFactory(
            list(args.encoder_hidden_units)
        ),
        critic_encoder_factory=d3rlpy.models.encoders.VectorEncoderFactory(
            list(args.encoder_hidden_units)
        ),
        tau=args.tau,
        n_critics=args.n_critics,
        conservative_weight=args.conservative_weight,
        n_action_samples=args.n_action_samples,
        compile_graph=args.compile_graph,
    )
    algorithm = config.create(device=device)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger_factory = d3rlpy.logging.FileAdapterFactory(
        root_dir=str(run_dir / "training_logs")
    )
    logs = algorithm.fit(
        replay,
        n_steps=args.steps,
        n_steps_per_epoch=args.steps_per_epoch,
        experiment_name="cql_training",
        with_timestamp=False,
        logger_adapter=logger_factory,
        show_progress=not args.quiet,
        save_interval=max(1, args.steps // args.steps_per_epoch),
    )
    algorithm.save(str(run_dir / "model.d3"))
    return algorithm, logs


def _seed_everything(seed: int, d3rlpy: Any) -> None:
    """Seed Python, NumPy, d3rlpy, Torch, and CUDA deterministically."""

    random.seed(seed)
    np.random.seed(seed)
    d3rlpy.seed(seed)


def dependency_metadata() -> dict[str, Any]:
    """Record interpreter, OS, package, CUDA, and GPU versions."""

    packages: dict[str, str | None] = {}
    for name in ("d3rlpy", "gymnasium", "h5py", "mujoco", "numpy", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version() if torch.cuda.is_available() else None
            ),
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        torch_info = {"cuda_available": False}
    return {
        "generated_at_utc": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_runtime": torch_info,
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of one artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Atomically write a JSON artifact with NumPy-aware conversion."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    """Convert common experiment objects to JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat()


def download_dataset(dataset_name: str, cache_dir: Path) -> Path:
    """Download a supported official D4RL v2 HDF5 file when absent."""

    try:
        url = D4RL_URLS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported D4RL dataset: {dataset_name}") from exc
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{dataset_name}.hdf5"
    if not output.exists():
        print(f"[download] {url} -> {output}")
        urlretrieve(url, output)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one deterministic D4RL job."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("validate", "augment", "train", "run"), default="run"
    )
    parser.add_argument("--dataset", choices=tuple(D4RL_URLS), required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="factual_residual")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--augmentation-seed",
        type=int,
        default=0,
        help="Fixed seed for data generation, cache identity, and CF replacement.",
    )
    parser.add_argument("--hdf5-path", type=Path)
    parser.add_argument(
        "--dataset-cache-dir", type=Path, default=Path("data/d4rl_hdf5")
    )
    parser.add_argument("--augmentation-cache", type=Path)
    parser.add_argument(
        "--augmentation-cache-dir", type=Path, default=Path("data/d4rl_simulator_cf")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/d4rl_simulator_cf_cql")
    )
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--diagnostic-samples", type=int, default=1_000)
    parser.add_argument("--intervention-scale", type=float, default=0.10)
    parser.add_argument("--cf-fraction", type=float, default=0.50)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--steps-per-epoch", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-base", type=int, default=100_000)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, or d3rlpy device such as cuda:0"
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--temp-learning-rate", type=float, default=1e-4)
    parser.add_argument("--alpha-learning-rate", type=float, default=0.0)
    parser.add_argument(
        "--encoder-hidden-units", type=int, nargs="+", default=(256, 256, 256)
    )
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--n-critics", type=int, default=2)
    parser.add_argument("--conservative-weight", type=float, default=10.0)
    parser.add_argument("--n-action-samples", type=int, default=10)
    parser.add_argument("--compile-graph", action="store_true")
    parser.add_argument("--gate-restored-observation-nrmse", type=float, default=0.05)
    parser.add_argument("--gate-factual-next-nrmse", type=float, default=0.10)
    parser.add_argument("--gate-factual-reward-nmae", type=float, default=0.10)
    parser.add_argument("--gate-terminal-disagreement", type=float, default=0.01)
    parser.add_argument("--gate-action-clip-fraction", type=float, default=0.001)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _validate_cli(args: argparse.Namespace) -> None:
    """Reject configurations that break matching or reproducibility."""

    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.augmentation_seed < 0:
        raise ValueError("augmentation_seed must be non-negative")
    if not 0.0 <= args.cf_fraction <= 1.0:
        raise ValueError("cf_fraction must be in [0, 1]")
    if args.intervention_scale <= 0 or not np.isfinite(args.intervention_scale):
        raise ValueError("intervention_scale must be finite and positive")
    if args.eval_episodes <= 0 or args.eval_seed_base < 0:
        raise ValueError("evaluation settings must be positive/non-negative")
    if not args.encoder_hidden_units or any(
        width <= 0 for width in args.encoder_hidden_units
    ):
        raise ValueError("encoder_hidden_units must contain positive widths")
    if args.mode in ("train", "run"):
        if args.steps <= 0 or args.steps_per_epoch <= 0:
            raise ValueError("training step counts must be positive")
        if args.steps % args.steps_per_epoch:
            raise ValueError("steps must be divisible by steps_per_epoch")


def _gate_from_args(args: argparse.Namespace) -> GateConfig:
    """Construct replay-gate thresholds from CLI flags."""

    return GateConfig(
        restored_observation_nrmse=args.gate_restored_observation_nrmse,
        factual_next_nrmse=args.gate_factual_next_nrmse,
        factual_reward_nmae=args.gate_factual_reward_nmae,
        terminal_disagreement=args.gate_terminal_disagreement,
        action_clip_fraction=args.gate_action_clip_fraction,
    )


def _cache_path(args: argparse.Namespace) -> Path:
    """Resolve a deterministic cache path for one dataset and seed."""

    if args.augmentation_cache is not None:
        return args.augmentation_cache
    suffix = "all" if args.max_transitions is None else str(args.max_transitions)
    scale = str(args.intervention_scale).replace(".", "p")
    return (
        args.augmentation_cache_dir
        / args.dataset
        / (f"augmentation_seed_{args.augmentation_seed}_scale_{scale}_n_{suffix}.hdf5")
    )


def run_job(args: argparse.Namespace) -> None:
    """Execute one validate, augment, train, or end-to-end publication job."""

    _validate_cli(args)
    dataset_path = args.hdf5_path or download_dataset(
        args.dataset, args.dataset_cache_dir
    )
    dataset_sha256 = sha256_file(dataset_path)
    dataset = load_d4rl_hdf5(dataset_path, args.max_transitions)
    env_id = EVAL_ENVS[args.dataset]
    cache_path = _cache_path(args)
    run_dir = args.output_root / args.dataset / args.variant / f"seed_{args.seed}"
    gate = _gate_from_args(args)

    diagnostics: ReplayDiagnostics | None = None
    if args.mode in ("validate", "augment", "run"):
        replayer = MujocoStateReplayer(env_id, args.augmentation_seed)
        try:
            diagnostics = compute_replay_diagnostics(
                dataset,
                replayer,
                args.diagnostic_samples,
                args.augmentation_seed + 7_007,
                gate,
            )
            write_json(
                run_dir / "factual_replay_diagnostics.json",
                {
                    "dataset": args.dataset,
                    "environment_id": env_id,
                    "dataset_sha256": dataset_sha256,
                    "thresholds": asdict(gate),
                    "diagnostics": asdict(diagnostics),
                    "method_label": "simulator-residual SCM approximation",
                },
            )
            if not diagnostics.passed:
                raise RuntimeError(
                    "Factual simulator replay gate failed: "
                    + ", ".join(diagnostics.failed_checks)
                )
            if args.mode in ("augment", "run") and args.variant != "real":
                build_augmentation_cache(
                    dataset,
                    replayer,
                    args.dataset,
                    dataset_sha256,
                    cache_path,
                    args.augmentation_seed,
                    args.intervention_scale,
                    args.overwrite_cache,
                )
        finally:
            replayer.close()

    if args.mode in ("validate", "augment"):
        return

    if (run_dir / "model.d3").exists() and not args.overwrite_run:
        raise FileExistsError(
            f"Run already exists at {run_dir}; pass --overwrite-run to replace it"
        )
    synthetic: SyntheticArrays | None = None
    if args.variant != "real":
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Augmentation cache is missing: {cache_path}; run --mode augment first"
            )
        validate_cache_metadata(
            cache_path,
            args.dataset,
            dataset_sha256,
            dataset.size,
            args.augmentation_seed,
            args.intervention_scale,
        )
        synthetic = load_synthetic_arrays(cache_path, args.variant)
    training, synthetic_mask = make_fixed_size_training_arm(
        dataset,
        args.variant,
        args.cf_fraction,
        args.augmentation_seed,
        synthetic,
    )
    config_payload = {
        "created_at_utc": _utc_now(),
        "method_label": "simulator-residual SCM approximation; not exact oracle",
        "arguments": vars(args),
        "random_seeds": {
            "learner_seed": args.seed,
            "augmentation_seed": args.augmentation_seed,
        },
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "augmentation_cache": str(cache_path) if args.variant != "real" else None,
        "transition_count": training.size,
        "synthetic_transition_count": int(synthetic_mask.sum()),
        "explicit_next_observations": True,
        "timeout_bootstrap": True,
        "evaluation_seed_bank": [
            args.eval_seed_base + episode for episode in range(args.eval_episodes)
        ],
    }
    write_json(run_dir / "config.json", config_payload)
    write_json(run_dir / "dependencies.json", dependency_metadata())
    algorithm, logs = train_cql(training, args, run_dir)
    write_json(
        run_dir / "training_metrics.json",
        [{"epoch": epoch, "metrics": metrics} for epoch, metrics in logs],
    )
    evaluation = evaluate_policy(
        algorithm,
        env_id,
        args.dataset,
        args.eval_episodes,
        args.eval_seed_base,
    )
    evaluation.update(
        {
            "dataset": args.dataset,
            "variant": args.variant,
            "seed": args.seed,
            "learner_seed": args.seed,
            "augmentation_seed": args.augmentation_seed,
            "training_updates": args.steps,
            "batch_size": args.batch_size,
            "transition_count": training.size,
            "synthetic_transition_count": int(synthetic_mask.sum()),
        }
    )
    write_json(run_dir / "evaluation.json", evaluation)
    print(
        f"[{args.dataset}:{args.variant}:seed={args.seed}] "
        f"raw={evaluation['raw_mean']:.2f}, "
        f"normalized={evaluation['normalized_mean']:.2f}"
    )


def main() -> None:
    """CLI entry point."""

    run_job(parse_args())


if __name__ == "__main__":
    main()
