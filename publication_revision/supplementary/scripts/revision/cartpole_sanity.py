#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]

GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
LENGTH = 0.5
POLEMASS_LENGTH = MASS_POLE * LENGTH
TAU = 0.02
MAX_FORCE = 10.0
X_THRESHOLD = 2.4
THETA_THRESHOLD = 12 * np.pi / 180


class CartPole11Env:
    """11-action CartPole with explicit process- or observation-noise semantics."""

    def __init__(
        self,
        seed: int,
        max_steps: int,
        state_noise_std: float = 0.0,
        noise_semantics: str = "process",
    ) -> None:
        if noise_semantics not in {"process", "observation"}:
            raise ValueError("noise_semantics must be 'process' or 'observation'.")
        self.np_random = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.state_noise_std = state_noise_std
        self.noise_semantics = noise_semantics
        self.state: np.ndarray | None = None
        self.steps = 0

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0
        return self._observe(), {}

    def _observe(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Environment reset() must be called before observe.")
        obs = self.state.copy()
        if self.state_noise_std > 0 and self.noise_semantics == "observation":
            obs = obs + self.state_noise_std * self.np_random.normal(size=4)
            obs[0] = np.clip(obs[0], -4.8, 4.8)
            obs[2] = np.clip(obs[2], -0.418, 0.418)
        return obs.astype(np.float32)

    def step(self, action_idx: int, action_noise_std: float = 0.0):
        if self.state is None:
            raise RuntimeError("Environment reset() must be called before step().")
        a_cont = float(action_idx) / 10.0
        if action_noise_std > 0:
            a_cont += float(self.np_random.normal(0.0, action_noise_std))
        force = (2.0 * a_cont - 1.0) * MAX_FORCE
        x, x_dot, theta, theta_dot = self.state
        costheta, sintheta = np.cos(theta), np.sin(theta)
        temp = (force + POLEMASS_LENGTH * theta_dot**2 * sintheta) / TOTAL_MASS
        thetaacc = (GRAVITY * sintheta - costheta * temp) / (
            LENGTH * (4.0 / 3.0 - MASS_POLE * costheta**2 / TOTAL_MASS)
        )
        xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS
        sp_clean = np.array(
            [
                x + TAU * x_dot,
                x_dot + TAU * xacc,
                theta + TAU * theta_dot,
                theta_dot + TAU * thetaacc,
            ],
            dtype=np.float64,
        )
        self.steps += 1
        terminated = bool(
            sp_clean[0] < -X_THRESHOLD
            or sp_clean[0] > X_THRESHOLD
            or sp_clean[2] < -THETA_THRESHOLD
            or sp_clean[2] > THETA_THRESHOLD
        )
        if self.state_noise_std > 0 and self.noise_semantics == "process":
            next_state = sp_clean + self.state_noise_std * self.np_random.normal(size=4)
            next_state[0] = np.clip(next_state[0], -4.8, 4.8)
            next_state[2] = np.clip(next_state[2], -0.418, 0.418)
            self.state = next_state
        else:
            self.state = sp_clean
        truncated = self.steps >= self.max_steps
        return self._observe(), 1.0, terminated, truncated, {}

    def close(self) -> None:
        return None


@dataclass
class SanityConfig:
    seed: int = 0
    train_episodes: int = 800
    eval_episodes: int = 100
    validation_episodes: int = 20
    validation_seed_base: int = 400_000
    test_seed_base: int = 500_000
    max_steps: int = 500
    batch_size: int = 256
    replay_size: int = 50000
    warmup_steps: int = 1000
    gamma: float = 0.99
    lr: float = 1e-4
    target_update: int = 500
    target_tau: float = 0.005
    eval_every: int = 100
    state_noise_std: float = 0.05
    action_noise_std: float = 0.05
    noise_semantics: str = "process"
    output_dir: Path = Path("results/revision/cartpole_sanity")


class QNet(nn.Module):
    def __init__(self, state_dim: int = 4, n_actions: int = 11) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Replay:
    def __init__(self, capacity: int) -> None:
        self.buf: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(
            maxlen=capacity
        )

    def add(self, s: np.ndarray, a: int, r: float, sp: np.ndarray, done: bool) -> None:
        self.buf.append(
            (s.astype(np.float32), a, r, sp.astype(np.float32), float(done))
        )

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buf, batch_size)
        s, a, r, sp, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=device),
            torch.tensor(a, dtype=torch.long, device=device),
            torch.tensor(r, dtype=torch.float32, device=device),
            torch.tensor(np.array(sp), dtype=torch.float32, device=device),
            torch.tensor(d, dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.buf)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_digest() -> str:
    """Hash the sanity runner and its shared CartPole implementation."""
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("cartpole_ctrl_reproduction.py"),
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_env(seed: int, cfg: SanityConfig, noisy: bool) -> CartPole11Env:
    return CartPole11Env(
        seed=seed,
        max_steps=cfg.max_steps,
        state_noise_std=cfg.state_noise_std if noisy else 0.0,
        noise_semantics=cfg.noise_semantics,
    )


def rollout_random(
    cfg: SanityConfig,
    noisy: bool,
    seed_base: int,
) -> list[float]:
    returns: list[float] = []
    action_rng = np.random.default_rng(cfg.seed + seed_base + int(noisy))
    for ep in range(cfg.eval_episodes):
        evaluation_seed = seed_base + ep
        env = make_env(evaluation_seed, cfg=cfg, noisy=noisy)
        env.reset(seed=evaluation_seed)
        total = 0.0
        for _ in range(cfg.max_steps):
            a = int(action_rng.integers(0, 11))
            _, r, done, trunc, _ = env.step(
                a,
                action_noise_std=cfg.action_noise_std if noisy else 0.0,
            )
            total += float(r)
            if done or trunc:
                break
        returns.append(total)
        env.close()
    return returns


def evaluate(
    q: QNet,
    cfg: SanityConfig,
    device: torch.device,
    noisy: bool,
    seed_base: int,
    episodes: int,
) -> list[float]:
    q.eval()
    returns: list[float] = []
    for ep in range(episodes):
        evaluation_seed = seed_base + ep
        env = make_env(evaluation_seed, cfg=cfg, noisy=noisy)
        s, _ = env.reset(seed=evaluation_seed)
        total = 0.0
        for _ in range(cfg.max_steps):
            with torch.no_grad():
                st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                a = int(q(st).argmax(dim=1).item())
            s, r, done, trunc, _ = env.step(
                a,
                action_noise_std=cfg.action_noise_std if noisy else 0.0,
            )
            total += float(r)
            if done or trunc:
                break
        returns.append(total)
        env.close()
    return returns


def train_online_dqn(
    cfg: SanityConfig, device: torch.device
) -> tuple[QNet, list[dict]]:
    q = QNet().to(device)
    target = QNet().to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    replay = Replay(cfg.replay_size)
    logs: list[dict] = []
    global_step = 0
    best_noisy = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for ep in range(cfg.train_episodes):
        env = make_env(cfg.seed + ep, cfg=cfg, noisy=True)
        s, _ = env.reset(seed=cfg.seed + ep)
        ep_ret = 0.0
        eps = max(0.05, 1.0 - ep / (0.65 * cfg.train_episodes))
        for _ in range(cfg.max_steps):
            if random.random() < eps:
                a = int(np.random.randint(0, 11))
            else:
                with torch.no_grad():
                    st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(
                        0
                    )
                    a = int(q(st).argmax(dim=1).item())
            sp, r, done, trunc, _ = env.step(
                a,
                action_noise_std=cfg.action_noise_std,
            )
            replay.add(s, a, float(r), sp, bool(done or trunc))
            s = sp
            ep_ret += float(r)
            global_step += 1

            if len(replay) >= max(cfg.batch_size, cfg.warmup_steps):
                sb, ab, rb, spb, db = replay.sample(cfg.batch_size, device)
                q_sa = q(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_a = q(spb).argmax(dim=1)
                    next_q = target(spb).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb + cfg.gamma * (1.0 - db) * next_q
                loss = F.smooth_l1_loss(q_sa, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(q.parameters(), 100.0)
                opt.step()

            if cfg.target_tau > 0:
                with torch.no_grad():
                    for target_param, param in zip(target.parameters(), q.parameters()):
                        target_param.mul_(1.0 - cfg.target_tau)
                        target_param.add_(cfg.target_tau * param)
            elif global_step % cfg.target_update == 0:
                target.load_state_dict(q.state_dict())

            if done or trunc:
                break
        env.close()

        if (ep + 1) % cfg.eval_every == 0 or ep == cfg.train_episodes - 1:
            clean = evaluate(
                q,
                cfg,
                device,
                noisy=False,
                seed_base=cfg.validation_seed_base,
                episodes=cfg.validation_episodes,
            )
            ctrl = evaluate(
                q,
                cfg,
                device,
                noisy=True,
                seed_base=cfg.validation_seed_base,
                episodes=cfg.validation_episodes,
            )
            logs.append(
                {
                    "episode": ep + 1,
                    "train_return": ep_ret,
                    "epsilon": eps,
                    "clean_mean": float(np.mean(clean)),
                    "clean_std": float(np.std(clean, ddof=1)),
                    "ctrl_mean": float(np.mean(ctrl)),
                    "ctrl_std": float(np.std(ctrl, ddof=1)),
                }
            )
            if float(np.mean(ctrl)) > best_noisy:
                best_noisy = float(np.mean(ctrl))
                best_state = {
                    k: v.detach().cpu().clone() for k, v in q.state_dict().items()
                }
            print(
                f"[sanity] ep={ep + 1} clean={np.mean(clean):.2f} "
                f"ctrl={np.mean(ctrl):.2f} eps={eps:.3f}",
                flush=True,
            )

    if best_state is not None:
        q.load_state_dict(best_state)
    return q, logs


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "std": std,
        "ci95": float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "returns": [float(x) for x in values],
    }


def validate_seed_banks(config: SanityConfig) -> None:
    """Reject overlapping checkpoint-selection and final-test episode seeds."""
    validation = range(
        config.validation_seed_base,
        config.validation_seed_base + config.validation_episodes,
    )
    test = range(
        config.test_seed_base,
        config.test_seed_base + config.eval_episodes,
    )
    if validation.start < test.stop and test.start < validation.stop:
        raise ValueError("Validation and test evaluation seed banks overlap.")


def parse_args() -> SanityConfig:
    p = argparse.ArgumentParser(description="CartPole CTRL sanity baselines.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-episodes", type=int, default=800)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--validation-episodes", type=int, default=20)
    p.add_argument("--validation-seed-base", type=int, default=400_000)
    p.add_argument("--test-seed-base", type=int, default=500_000)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--state-noise-std", type=float, default=0.05)
    p.add_argument("--action-noise-std", type=float, default=0.05)
    p.add_argument(
        "--noise-semantics",
        choices=("process", "observation"),
        default="process",
    )
    p.add_argument("--lr", type=float, default=SanityConfig.lr)
    p.add_argument("--batch-size", type=int, default=SanityConfig.batch_size)
    p.add_argument("--target-tau", type=float, default=SanityConfig.target_tau)
    p.add_argument("--target-update", type=int, default=SanityConfig.target_update)
    p.add_argument(
        "--output-dir", type=Path, default=Path("results/revision/cartpole_sanity")
    )
    args = p.parse_args()
    return SanityConfig(
        seed=args.seed,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        validation_episodes=args.validation_episodes,
        validation_seed_base=args.validation_seed_base,
        test_seed_base=args.test_seed_base,
        max_steps=args.max_steps,
        state_noise_std=args.state_noise_std,
        action_noise_std=args.action_noise_std,
        noise_semantics=args.noise_semantics,
        lr=args.lr,
        batch_size=args.batch_size,
        target_tau=args.target_tau,
        target_update=args.target_update,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    validate_seed_banks(cfg)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = source_digest()

    random_clean = rollout_random(cfg, noisy=False, seed_base=cfg.test_seed_base)
    random_ctrl = rollout_random(cfg, noisy=True, seed_base=cfg.test_seed_base)
    q, logs = train_online_dqn(cfg, device)
    online_clean = evaluate(
        q,
        cfg,
        device,
        noisy=False,
        seed_base=cfg.test_seed_base,
        episodes=cfg.eval_episodes,
    )
    online_ctrl = evaluate(
        q,
        cfg,
        device,
        noisy=True,
        seed_base=cfg.test_seed_base,
        episodes=cfg.eval_episodes,
    )

    torch.save(q.state_dict(), cfg.output_dir / f"online_dqn_seed_{cfg.seed}.pt")
    payload = {
        "artifact_schema": "cartpole-sanity-revision-v2",
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "device": str(device),
        "source_sha256": source_sha256,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "checkpoint_selection": {
            "metric": "noisy_validation_mean_return",
            "validation_seed_base": cfg.validation_seed_base,
            "validation_episodes": cfg.validation_episodes,
            "test_seed_base": cfg.test_seed_base,
            "test_episodes": cfg.eval_episodes,
        },
        "random_clean": summarize(random_clean),
        "random_ctrl": summarize(random_ctrl),
        "online_clean": summarize(online_clean),
        "online_ctrl": summarize(online_ctrl),
        "training_logs": logs,
    }
    out = cfg.output_dir / f"sanity_seed_{cfg.seed}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[sanity] wrote {out}")


if __name__ == "__main__":
    main()
