#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent))
from cartpole_sanity import CartPole11Env, QNet, set_seed, summarize  # noqa: E402


@dataclass
class OracleConfig:
    seed: int = 0
    dataset_episodes: int = 250
    dataset_horizon: int = 20
    train_steps: int = 20000
    eval_episodes: int = 100
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 1e-4
    target_tau: float = 0.005
    alpha_cql: float = 0.02
    cf_frac: float = 0.0
    output_dir: Path = Path("results/revision/cartpole_oracle_cf")


class Replay:
    def __init__(self) -> None:
        self.s: list[np.ndarray] = []
        self.a: list[int] = []
        self.r: list[float] = []
        self.sp: list[np.ndarray] = []
        self.d: list[float] = []

    def add(self, s: np.ndarray, a: int, r: float, sp: np.ndarray, done: bool) -> None:
        self.s.append(s.astype(np.float32))
        self.a.append(int(a))
        self.r.append(float(r))
        self.sp.append(sp.astype(np.float32))
        self.d.append(float(done))

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, len(self.s), size=batch_size)
        return (
            torch.tensor(np.asarray([self.s[i] for i in idx]), device=device),
            torch.tensor([self.a[i] for i in idx], dtype=torch.long, device=device),
            torch.tensor([self.r[i] for i in idx], dtype=torch.float32, device=device),
            torch.tensor(np.asarray([self.sp[i] for i in idx]), device=device),
            torch.tensor([self.d[i] for i in idx], dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.s)


def make_transition(
    state: np.ndarray,
    action_idx: int,
    seed: int,
    state_noise_std: float,
    action_noise_std: float,
) -> tuple[np.ndarray, float, bool]:
    env = CartPole11Env(seed=seed, max_steps=500, state_noise_std=state_noise_std)
    env.state = state.copy()
    sp, reward, done, _, _ = env.step(action_idx, action_noise_std=action_noise_std)
    return sp, float(reward), bool(done)


def generate_dataset(cfg: OracleConfig, mode: str) -> Replay:
    rng = np.random.default_rng(cfg.seed)
    replay = Replay()
    for ep in range(cfg.dataset_episodes):
        env = CartPole11Env(
            seed=cfg.seed + ep,
            max_steps=cfg.dataset_horizon,
            state_noise_std=0.05,
        )
        s_obs, _ = env.reset(seed=cfg.seed + ep)
        for t in range(cfg.dataset_horizon):
            clean_state = env.state.copy()
            action = int(rng.integers(0, 11))
            sp_obs, reward, done, trunc, _ = env.step(action, action_noise_std=0.05)
            replay.add(s_obs, action, reward, sp_obs, done or trunc)

            if mode == "oracle_cf" and cfg.cf_frac > 0:
                n_cf = max(1, int(round(cfg.cf_frac * 10)))
                alternatives = [a for a in range(11) if a != action]
                rng.shuffle(alternatives)
                for cf_action in alternatives[:n_cf]:
                    sp_cf, r_cf, d_cf = make_transition(
                        clean_state,
                        cf_action,
                        seed=cfg.seed * 1_000_000 + ep * 100 + t * 11 + cf_action,
                        state_noise_std=0.05,
                        action_noise_std=0.05,
                    )
                    replay.add(s_obs, cf_action, r_cf, sp_cf, d_cf)

            s_obs = sp_obs
            if done or trunc:
                break
    return replay


def train_offline(replay: Replay, cfg: OracleConfig, device: torch.device) -> QNet:
    q = QNet().to(device)
    target = QNet().to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.AdamW(q.parameters(), lr=cfg.lr, amsgrad=True)

    for step in range(1, cfg.train_steps + 1):
        s, a, r, sp, done = replay.sample(cfg.batch_size, device)
        q_values = q(s)
        q_sa = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_a = q(sp).argmax(dim=1)
            next_q = target(sp).gather(1, next_a.unsqueeze(1)).squeeze(1)
            y = r + cfg.gamma * (1.0 - done) * next_q
        bellman = F.smooth_l1_loss(q_sa, y)
        cql = torch.logsumexp(q_values, dim=1).mean() - q_sa.mean()
        loss = bellman + cfg.alpha_cql * cql
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(q.parameters(), 100.0)
        opt.step()

        with torch.no_grad():
            for target_param, param in zip(target.parameters(), q.parameters()):
                target_param.mul_(1.0 - cfg.target_tau)
                target_param.add_(cfg.target_tau * param)

        if step % 5000 == 0:
            print(
                f"[oracle-cf] step={step} loss={loss.item():.4f} "
                f"bellman={bellman.item():.4f} cql={cql.item():.4f}",
                flush=True,
            )
    return q


def evaluate(q: QNet, cfg: OracleConfig, device: torch.device, noisy: bool) -> list[float]:
    q.eval()
    returns: list[float] = []
    for ep in range(cfg.eval_episodes):
        env = CartPole11Env(
            seed=cfg.seed + 50_000 + ep,
            max_steps=500,
            state_noise_std=0.05 if noisy else 0.0,
        )
        s, _ = env.reset(seed=cfg.seed + 50_000 + ep)
        total = 0.0
        for _ in range(500):
            with torch.no_grad():
                st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(q(st).argmax(dim=1).item())
            s, reward, done, trunc, _ = env.step(
                action,
                action_noise_std=0.05 if noisy else 0.0,
            )
            total += reward
            if done or trunc:
                break
        returns.append(total)
    return returns


def parse_args() -> tuple[OracleConfig, str]:
    parser = argparse.ArgumentParser(description="Offline DQN with oracle CartPole CF.")
    parser.add_argument("--seed", type=int, default=OracleConfig.seed)
    parser.add_argument("--mode", choices=["real", "oracle_cf"], default="real")
    parser.add_argument("--cf-frac", type=float, default=OracleConfig.cf_frac)
    parser.add_argument("--train-steps", type=int, default=OracleConfig.train_steps)
    parser.add_argument("--eval-episodes", type=int, default=OracleConfig.eval_episodes)
    parser.add_argument("--output-dir", type=Path, default=OracleConfig.output_dir)
    args = parser.parse_args()
    return (
        OracleConfig(
            seed=args.seed,
            train_steps=args.train_steps,
            eval_episodes=args.eval_episodes,
            cf_frac=args.cf_frac,
            output_dir=args.output_dir,
        ),
        args.mode,
    )


def main() -> None:
    cfg, mode = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    replay = generate_dataset(cfg, mode)
    q = train_offline(replay, cfg, device)
    clean = evaluate(q, cfg, device, noisy=False)
    ctrl = evaluate(q, cfg, device, noisy=True)
    payload = {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "mode": mode,
        "device": str(device),
        "dataset_transitions": len(replay),
        "clean": summarize(clean),
        "ctrl_noisy": summarize(ctrl),
    }
    out = cfg.output_dir / f"{mode}_seed_{cfg.seed}.json"
    torch.save(q.state_dict(), cfg.output_dir / f"{mode}_seed_{cfg.seed}.pt")
    out.write_text(json.dumps(payload, indent=2))
    print(f"[oracle-cf] wrote {out}")


if __name__ == "__main__":
    main()
