#!/usr/bin/env python3
"""Run one compute-matched online SAC replay-augmentation experiment.

The ``oracle_cf`` arm branches the MuJoCo simulator for exactly one step from
the factual pre-action physics state. It then restores the factual trajectory
before training continues. This is an oracle simulator control, not a learned
SCM counterfactual.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

Variant = Literal["real", "duplicate", "oracle_cf"]
VARIANTS: tuple[Variant, ...] = ("real", "duplicate", "oracle_cf")


class ReplayBufferLike(Protocol):
    """Minimal Stable-Baselines3 replay-buffer interface used by this script."""

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Add one vectorized transition."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one environment, seed, and replay variant."""

    env_id: str
    seed: int
    variant: Variant
    total_timesteps: int
    eval_episodes: int
    eval_seed_base: int
    intervention_scale: float
    learning_starts: int
    buffer_size: int
    batch_size: int
    train_freq: int
    gradient_steps: int
    output_root: Path
    job_id: str

    def validate(self) -> None:
        """Raise ``ValueError`` when the run cannot support a fair comparison."""

        if not self.env_id.strip():
            raise ValueError("env_id must not be empty")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive")
        if self.eval_seed_base < 0:
            raise ValueError("eval_seed_base must be non-negative")
        if not np.isfinite(self.intervention_scale):
            raise ValueError("intervention_scale must be finite")
        if self.intervention_scale <= 0:
            raise ValueError("intervention_scale must be positive")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be non-negative")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        required_capacity = 2 * self.total_timesteps
        if self.buffer_size < required_capacity:
            raise ValueError(
                "buffer_size must be at least 2 * total_timesteps so every "
                "variant uses the same non-wrapping replay capacity"
            )
        if self.batch_size <= 0 or self.batch_size > self.buffer_size:
            raise ValueError("batch_size must be in [1, buffer_size]")
        if self.train_freq <= 0:
            raise ValueError("train_freq must be positive")
        if self.total_timesteps % self.train_freq != 0:
            raise ValueError("total_timesteps must be divisible by train_freq")
        if self.gradient_steps <= 0:
            raise ValueError("gradient_steps must be positive")
        if not self.job_id.strip():
            raise ValueError("job_id must not be empty")


@dataclass(frozen=True)
class EnvSnapshot:
    """Restorable physics and TimeLimit state for one Gymnasium environment."""

    backend: Literal["mujoco", "classic"]
    physics_state: np.ndarray
    mujoco_state_spec: int | None
    time_limit_steps: tuple[int | None, ...]
    classic_last_u: Any = None


@dataclass(frozen=True)
class BranchTransition:
    """One exact simulator branch transition in environment action units."""

    next_obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


def _wrapper_chain(env: gym.Env[Any, Any]) -> list[gym.Env[Any, Any]]:
    """Return the wrapper chain, including the unwrapped environment."""

    chain: list[gym.Env[Any, Any]] = []
    current: gym.Env[Any, Any] = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        child = getattr(current, "env", None)
        if child is None:
            break
        current = cast(gym.Env[Any, Any], child)
    return chain


def find_time_limit(env: gym.Env[Any, Any]) -> gym.Env[Any, Any]:
    """Find the TimeLimit branch point while bypassing SB3's Monitor wrapper."""

    for wrapper in _wrapper_chain(env):
        if isinstance(wrapper, gym.wrappers.TimeLimit):
            return wrapper
    raise RuntimeError(
        "A Gymnasium TimeLimit wrapper is required for exact branch truncation"
    )


def _capture_mujoco_state(base_env: gym.Env[Any, Any]) -> tuple[np.ndarray, int]:
    """Capture MuJoCo's integration state, or FULLPHYSICS on older versions."""

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo state capture requires the 'mujoco' package; install "
            "gymnasium[mujoco]"
        ) from exc

    model = getattr(base_env, "model", None)
    data = getattr(base_env, "data", None)
    if model is None or data is None:
        raise RuntimeError("Environment exposes neither MuJoCo model nor data")

    state_enum = mujoco.mjtState
    state_spec = int(
        getattr(state_enum, "mjSTATE_INTEGRATION", state_enum.mjSTATE_FULLPHYSICS)
    )
    size = int(mujoco.mj_stateSize(model, state_spec))
    state = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(model, data, state, state_spec)
    return state, state_spec


def capture_env_snapshot(env: gym.Env[Any, Any]) -> EnvSnapshot:
    """Capture exact MuJoCo physics and every nested TimeLimit counter.

    Pendulum-like ``state`` environments are supported only as a lightweight
    test backend; publication runs are required to use MuJoCo.
    """

    chain = _wrapper_chain(env)
    time_steps = tuple(
        copy.deepcopy(wrapper._elapsed_steps)
        for wrapper in chain
        if isinstance(wrapper, gym.wrappers.TimeLimit)
    )
    base_env = env.unwrapped
    if hasattr(base_env, "model") and hasattr(base_env, "data"):
        state, state_spec = _capture_mujoco_state(base_env)
        return EnvSnapshot("mujoco", state, state_spec, time_steps)

    classic_state = getattr(base_env, "state", None)
    if classic_state is None:
        raise RuntimeError(
            "Unsupported environment: expected MuJoCo model/data attributes"
        )
    return EnvSnapshot(
        backend="classic",
        physics_state=np.asarray(classic_state, dtype=np.float64).copy(),
        mujoco_state_spec=None,
        time_limit_steps=time_steps,
        classic_last_u=copy.deepcopy(getattr(base_env, "last_u", None)),
    )


def restore_env_snapshot(env: gym.Env[Any, Any], snapshot: EnvSnapshot) -> None:
    """Restore a snapshot without resetting or advancing the environment."""

    base_env = env.unwrapped
    if snapshot.backend == "mujoco":
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("Cannot restore MuJoCo state without mujoco") from exc
        model = getattr(base_env, "model", None)
        data = getattr(base_env, "data", None)
        if model is None or data is None or snapshot.mujoco_state_spec is None:
            raise RuntimeError("MuJoCo environment does not match its snapshot")
        expected = int(mujoco.mj_stateSize(model, snapshot.mujoco_state_spec))
        if snapshot.physics_state.shape != (expected,):
            raise ValueError(
                "MuJoCo snapshot size does not match the destination model"
            )
        mujoco.mj_setState(
            model,
            data,
            snapshot.physics_state,
            snapshot.mujoco_state_spec,
        )
        mujoco.mj_forward(model, data)
    elif snapshot.backend == "classic":
        if not hasattr(base_env, "state"):
            raise RuntimeError("Classic-control environment has no state field")
        base_env.state = snapshot.physics_state.copy()
        if hasattr(base_env, "last_u"):
            base_env.last_u = copy.deepcopy(snapshot.classic_last_u)
    else:
        raise ValueError(f"Unknown snapshot backend: {snapshot.backend}")

    time_limits = [
        wrapper
        for wrapper in _wrapper_chain(env)
        if isinstance(wrapper, gym.wrappers.TimeLimit)
    ]
    if len(time_limits) != len(snapshot.time_limit_steps):
        raise RuntimeError("TimeLimit wrapper structure changed after snapshot")
    for wrapper, elapsed_steps in zip(
        time_limits, snapshot.time_limit_steps, strict=True
    ):
        wrapper._elapsed_steps = copy.deepcopy(elapsed_steps)


def branch_one_step(
    env: gym.Env[Any, Any],
    pre_step: EnvSnapshot,
    post_factual: EnvSnapshot,
    alternative_action: np.ndarray,
) -> BranchTransition:
    """Step once from ``pre_step`` and always restore ``post_factual``."""

    try:
        restore_env_snapshot(env, pre_step)
        next_obs, reward, terminated, truncated, info = env.step(alternative_action)
        transition = BranchTransition(
            next_obs=np.asarray(next_obs, dtype=np.float32).copy(),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )
    finally:
        restore_env_snapshot(env, post_factual)
    return transition


def sample_alternative_action(
    factual_action: np.ndarray,
    action_space: gym.spaces.Box,
    rng: np.random.Generator,
    scale: float,
    max_attempts: int = 128,
) -> np.ndarray:
    """Sample a changed clipped-Gaussian action in normalized action units.

    ``scale`` is the standard deviation after mapping each finite action bound
    to ``[-1, 1]``. Samples that clip back to the factual action are rejected.
    """

    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    factual = np.asarray(factual_action, dtype=np.float64)
    low = np.asarray(action_space.low, dtype=np.float64)
    high = np.asarray(action_space.high, dtype=np.float64)
    if factual.shape != action_space.shape:
        raise ValueError(
            f"factual action shape {factual.shape} != {action_space.shape}"
        )
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("Oracle intervention requires finite Box action bounds")
    width = high - low
    if np.any(width <= 0):
        raise ValueError("Action-space bounds must have positive width")

    normalized = 2.0 * (np.clip(factual, low, high) - low) / width - 1.0
    for _ in range(max_attempts):
        candidate_norm = np.clip(
            normalized + rng.normal(0.0, scale, size=factual.shape), -1.0, 1.0
        )
        candidate = low + 0.5 * (candidate_norm + 1.0) * width
        if not np.allclose(candidate, factual, rtol=0.0, atol=1e-7):
            return candidate.astype(action_space.dtype, copy=False)
    raise RuntimeError("Unable to sample an alternative action after clipping")


def add_replay_transition(
    replay_buffer: ReplayBufferLike,
    obs: np.ndarray,
    next_obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    done: bool,
    info: dict[str, Any],
) -> None:
    """Insert one transition using SB3's one-environment vectorized shapes."""

    obs_array = np.asarray(obs, dtype=np.float32)
    next_array = np.asarray(next_obs, dtype=np.float32)
    action_array = np.asarray(action, dtype=np.float32)
    if obs_array.shape != next_array.shape:
        raise ValueError("obs and next_obs must have identical shapes")
    if obs_array.ndim != 1 or action_array.ndim != 1:
        raise ValueError("obs, next_obs, and action must be one-dimensional")
    replay_buffer.add(
        obs=obs_array[np.newaxis, ...],
        next_obs=next_array[np.newaxis, ...],
        action=action_array[np.newaxis, ...],
        reward=np.asarray([reward], dtype=np.float32),
        done=np.asarray([done], dtype=np.float32),
        infos=[dict(info)],
    )


class ReplayAugmentationCallback(BaseCallback):
    """Insert duplicate or exact one-step oracle transitions into SAC replay."""

    def __init__(
        self,
        variant: Variant,
        intervention_scale: float,
        seed: int,
    ) -> None:
        super().__init__(verbose=0)
        if variant not in VARIANTS:
            raise ValueError(f"Unknown replay variant: {variant}")
        self.variant = variant
        self.intervention_scale = intervention_scale
        self.rng = np.random.default_rng(seed)
        self.extra_insertions = 0
        self.branch_terminated = 0
        self.branch_truncated = 0
        self.training_episodes: list[dict[str, Any]] = []
        self._branch_env: gym.Env[Any, Any] | None = None
        self._pre_snapshot: EnvSnapshot | None = None
        self._pre_observation: np.ndarray | None = None

    def _on_training_start(self) -> None:
        if self.training_env.num_envs != 1:
            raise RuntimeError("Replay augmentation supports exactly one environment")
        if self.variant == "oracle_cf":
            envs = getattr(self.training_env, "envs", None)
            if not envs or len(envs) != 1:
                raise RuntimeError("Expected a DummyVecEnv with one base environment")
            self._branch_env = find_time_limit(envs[0])
            if not isinstance(self._branch_env.action_space, gym.spaces.Box):
                raise RuntimeError("Oracle SAC requires a continuous Box action space")
            self._pre_snapshot = capture_env_snapshot(self._branch_env)
            if self.model._last_obs is None:
                raise RuntimeError("SAC has no initial observation at training start")
            self._pre_observation = np.asarray(
                self.model._last_obs[0], dtype=np.float32
            ).copy()

    def _required_local(self, name: str) -> Any:
        if name not in self.locals:
            raise RuntimeError(f"SB3 callback locals are missing {name!r}")
        return self.locals[name]

    def _factual_next_obs(
        self,
        new_obs: np.ndarray,
        done: bool,
        info: dict[str, Any],
    ) -> np.ndarray:
        if done and "terminal_observation" in info:
            return np.asarray(info["terminal_observation"], dtype=np.float32)
        return np.asarray(new_obs, dtype=np.float32)

    def _insert_duplicate(self) -> None:
        replay_buffer = self.model.replay_buffer
        if replay_buffer is None:
            raise RuntimeError("SAC replay buffer is not initialized")
        last_obs = np.asarray(self._required_local("self")._last_obs[0])
        new_obs = np.asarray(self._required_local("new_obs"))[0]
        buffer_action = np.asarray(self._required_local("buffer_actions"))[0]
        reward = float(np.asarray(self._required_local("rewards"))[0])
        done = bool(np.asarray(self._required_local("dones"))[0])
        info = dict(self._required_local("infos")[0])
        next_obs = self._factual_next_obs(new_obs, done, info)
        add_replay_transition(
            replay_buffer,
            last_obs,
            next_obs,
            buffer_action,
            reward,
            done,
            info,
        )
        self.extra_insertions += 1

    def _insert_oracle(self) -> None:
        if (
            self._branch_env is None
            or self._pre_snapshot is None
            or self._pre_observation is None
        ):
            raise RuntimeError("Oracle callback was not initialized")
        replay_buffer = self.model.replay_buffer
        if replay_buffer is None:
            raise RuntimeError("SAC replay buffer is not initialized")

        factual_action = np.asarray(self._required_local("actions"))[0]
        action_space = cast(gym.spaces.Box, self._branch_env.action_space)
        alternative = sample_alternative_action(
            factual_action,
            action_space,
            self.rng,
            self.intervention_scale,
        )
        post_factual = capture_env_snapshot(self._branch_env)
        branch = branch_one_step(
            self._branch_env,
            self._pre_snapshot,
            post_factual,
            alternative,
        )
        scaled_action = np.asarray(
            self.model.policy.scale_action(alternative[np.newaxis, ...])[0],
            dtype=np.float32,
        )
        done = branch.terminated or branch.truncated
        info = dict(branch.info)
        info["TimeLimit.truncated"] = branch.truncated and not branch.terminated
        info["oracle_counterfactual"] = True
        add_replay_transition(
            replay_buffer,
            self._pre_observation,
            branch.next_obs,
            scaled_action,
            branch.reward,
            done,
            info,
        )
        self.extra_insertions += 1
        self.branch_terminated += int(branch.terminated)
        self.branch_truncated += int(branch.truncated)

        new_obs = np.asarray(self._required_local("new_obs"))[0]
        self._pre_observation = np.asarray(new_obs, dtype=np.float32).copy()
        self._pre_snapshot = post_factual

    def _on_step(self) -> bool:
        if self.variant == "duplicate":
            self._insert_duplicate()
        elif self.variant == "oracle_cf":
            self._insert_oracle()
        info = dict(self._required_local("infos")[0])
        episode = info.get("episode")
        if episode is not None:
            self.training_episodes.append(
                {
                    "return": float(episode["r"]),
                    "length": int(episode["l"]),
                    "real_timestep": int(self.num_timesteps),
                }
            )
        return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse exactly one environment/seed/variant experiment job."""

    parser = argparse.ArgumentParser(
        description="Online MuJoCo SAC with compute-matched replay controls."
    )
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-base", type=int, default=100_000)
    parser.add_argument(
        "--intervention-scale",
        type=float,
        default=0.20,
        help="Gaussian std in normalized [-1,1] action coordinates.",
    )
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--buffer-size", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/mujoco_oracle_cf_sac"),
    )
    parser.add_argument(
        "--job-id",
        default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build and validate a serializable experiment configuration."""

    config = ExperimentConfig(
        env_id=args.env_id,
        seed=args.seed,
        variant=cast(Variant, args.variant),
        total_timesteps=args.total_timesteps,
        eval_episodes=args.eval_episodes,
        eval_seed_base=args.eval_seed_base,
        intervention_scale=args.intervention_scale,
        learning_starts=args.learning_starts,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        output_root=args.output_root,
        job_id=args.job_id,
    )
    config.validate()
    return config


def _jsonable_config(config: ExperimentConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_root"] = str(config.output_root)
    payload["replay_semantics"] = {
        "real": "one factual replay transition per real interaction",
        "duplicate": "factual transition inserted twice",
        "oracle_cf": (
            "one factual plus one exact simulator branch transition; factual "
            "physics and TimeLimit state are restored"
        ),
    }[config.variant]
    payload["intervention_scale_units"] = "normalized action coordinates [-1,1]"
    payload["fixed_compute_statement"] = (
        "All variants use identical real timesteps, train_freq, gradient_steps, "
        "learning_starts, and evaluation seeds."
    )
    return payload


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic, human-readable JSON after creating its directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def dependency_metadata() -> dict[str, Any]:
    """Record software and accelerator versions needed to interpret a run."""

    packages = {}
    for name in (
        "gymnasium",
        "mujoco",
        "numpy",
        "stable-baselines3",
        "torch",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }


def evaluate_seed_bank(
    model: SAC,
    env_id: str,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Evaluate deterministic actions once for each fixed environment seed."""

    env = gym.make(env_id)
    episodes: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            total_return = 0.0
            length = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_return += float(reward)
                length += 1
            episodes.append(
                {
                    "seed": int(seed),
                    "return": total_return,
                    "length": length,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
    finally:
        env.close()
    return episodes


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for an artifact file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_experiment(config: ExperimentConfig) -> Path:
    """Train, evaluate, and save one reviewer-auditable experiment job."""

    config.validate()
    run_dir = (
        config.output_root
        / config.env_id
        / config.variant
        / f"seed_{config.seed}"
        / config.job_id
    )
    if run_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment directory: {run_dir}"
        )
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", _jsonable_config(config))
    write_json(run_dir / "dependencies.json", dependency_metadata())

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    try:
        env = gym.make(config.env_id)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to create {config.env_id!r}; install gymnasium[mujoco]"
        ) from exc
    if not isinstance(env.action_space, gym.spaces.Box):
        env.close()
        raise TypeError("SAC oracle experiment requires a Box action space")
    if not hasattr(env.unwrapped, "model") or not hasattr(env.unwrapped, "data"):
        env.close()
        raise ValueError(
            f"{config.env_id!r} is not a MuJoCo environment; publication runs "
            "require MuJoCo"
        )

    callback = ReplayAugmentationCallback(
        variant=config.variant,
        intervention_scale=config.intervention_scale,
        seed=config.seed + 1_000_003,
    )
    try:
        model = SAC(
            "MlpPolicy",
            env,
            seed=config.seed,
            device="auto",
            verbose=0,
            learning_starts=config.learning_starts,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
        )
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback,
            progress_bar=False,
        )
        if model.num_timesteps != config.total_timesteps:
            raise RuntimeError(
                "Real interaction count differs from total_timesteps: "
                f"{model.num_timesteps} != {config.total_timesteps}"
            )
        model_path = run_dir / "model"
        model.save(model_path)
        model_zip = model_path.with_suffix(".zip")

        seeds = [config.eval_seed_base + index for index in range(config.eval_episodes)]
        episodes = evaluate_seed_bank(model, config.env_id, seeds)
        returns = np.asarray(
            [episode["return"] for episode in episodes], dtype=np.float64
        )
        replay_size = int(model.replay_buffer.size())
        expected_extra = config.total_timesteps if config.variant != "real" else 0
        if callback.extra_insertions != expected_extra:
            raise RuntimeError(
                "Replay insertion count mismatch: expected "
                f"{expected_extra}, got {callback.extra_insertions}"
            )
        write_json(
            run_dir / "eval.json",
            {
                "episodes": episodes,
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns, ddof=1))
                if len(returns) > 1
                else 0.0,
                "median_return": float(np.median(returns)),
                "evaluation_seed_bank": seeds,
                "deterministic_policy": True,
            },
        )
        write_json(
            run_dir / "training_summary.json",
            {
                "actual_device": str(model.device),
                "real_environment_interactions": int(model.num_timesteps),
                "actual_gradient_updates": int(model._n_updates),
                "gradient_updates_per_train_call": config.gradient_steps,
                "train_frequency_real_steps": config.train_freq,
                "extra_replay_insertions": callback.extra_insertions,
                "expected_extra_replay_insertions": expected_extra,
                "final_replay_size": replay_size,
                "branch_terminated": callback.branch_terminated,
                "branch_truncated": callback.branch_truncated,
            },
        )
        write_json(run_dir / "training_episodes.json", callback.training_episodes)
        write_json(
            run_dir / "artifacts.json",
            {
                "model.zip": {
                    "sha256": sha256_file(model_zip),
                    "bytes": model_zip.stat().st_size,
                }
            },
        )
    finally:
        env.close()

    print(
        f"[{config.env_id}:{config.variant}:seed={config.seed}] "
        f"mean={float(np.mean(returns)):.2f}, output={run_dir}"
    )
    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for one scheduler-friendly experiment job."""

    config = config_from_args(parse_args(argv))
    run_experiment(config)


if __name__ == "__main__":
    main()
