#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import datetime as dt
from pathlib import Path
import sys
from typing import Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "CTRL") not in sys.path:
    sys.path.append(str(REPO_ROOT / "CTRL"))

from ctrl_algorithms.rainbow import RainbowNet, evaluate_rainbow_policy
from ctrl_algorithms.sac import Actor, evaluate_sac_policy
from ctrl_trainer import QNetCTRL
from ctrl_utilities import evaluate_policy


def _timestamped_dir(
    base: Path | None, algo: str, use_timestamp: bool = True
) -> Path:
    root = base if base is not None else REPO_ROOT / "results" / "eval" / algo
    return (
        root / dt.datetime.now().strftime("%Y%m%d-%H%M%S") if use_timestamp else root
    )


def _load_stats(metrics_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    meta = json.loads(metrics_path.read_text())
    mean = torch.tensor(meta["state_mean"], dtype=torch.float32)
    std = torch.tensor(meta["state_std"], dtype=torch.float32)
    return mean, std


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained policies (no rendering).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algo", choices=["d3qn", "sac", "rainbow"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-ctrl-env",
        action="store_true",
        help="Use CTRL noisy env; otherwise clean CartPole-v1.",
    )
    parser.add_argument(
        "--action-noise-std",
        type=float,
        default=0.0,
        help="Std dev of Gaussian action noise for eval.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where to dump eval metrics; defaults to results/eval/<algo>/<ts>.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="If set, do not append a timestamp to the eval output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    use_timestamp = not args.no_timestamp

    run_dir = args.run_dir
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json under {run_dir}")

    state_mean, state_std = _load_stats(metrics_path)
    output_dir = _timestamped_dir(args.output_dir, args.algo, use_timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.algo == "d3qn":
        q_net_path = run_dir / "q_net.pt"
        if not q_net_path.exists():
            raise FileNotFoundError(f"Missing q_net.pt under {run_dir}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q_net = QNetCTRL().to(device)
        q_net.load_state_dict(torch.load(q_net_path, map_location="cpu"))
        returns = evaluate_policy(
            q_net,
            state_mean,
            state_std,
            episodes=args.episodes,
            use_ctrl_env=args.use_ctrl_env,
            action_noise_std=args.action_noise_std,
        )
    elif args.algo == "sac":
        actor_path = run_dir / "actor.pt"
        if not actor_path.exists():
            raise FileNotFoundError(f"Missing actor.pt under {run_dir}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = Actor().to(device)
        actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
        returns = evaluate_sac_policy(
            actor,
            state_mean,
            state_std,
            episodes=args.episodes,
            seed=args.seed,
            use_ctrl_env=args.use_ctrl_env,
            action_noise_std=args.action_noise_std,
        )
    else:
        q_net_path = run_dir / "q_net.pt"
        if not q_net_path.exists():
            raise FileNotFoundError(f"Missing q_net.pt under {run_dir}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q_net = RainbowNet().to(device)
        q_net.load_state_dict(torch.load(q_net_path, map_location="cpu"))
        returns = evaluate_rainbow_policy(
            q_net,
            state_mean,
            state_std,
            episodes=args.episodes,
            seed=args.seed,
            use_ctrl_env=args.use_ctrl_env,
            action_noise_std=args.action_noise_std,
        )

    payload = {
        "algo": args.algo,
        "run_dir": str(run_dir),
        "episodes": args.episodes,
        "seed": args.seed,
        "use_ctrl_env": args.use_ctrl_env,
        "action_noise_std": args.action_noise_std,
        "mean_return": float(torch.tensor(returns).mean().item()),
        "std_return": float(torch.tensor(returns).std().item()),
        "returns": [float(r) for r in returns],
    }
    out_path = output_dir / f"{args.algo}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"[eval:{args.algo}] mean={payload['mean_return']:.2f} "
        f"std={payload['std_return']:.2f} episodes={args.episodes} "
        f"saved to {out_path}"
    )


if __name__ == "__main__":
    main()
