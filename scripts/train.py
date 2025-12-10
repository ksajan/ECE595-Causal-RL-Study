#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "CTRL") not in sys.path:
    sys.path.append(str(REPO_ROOT / "CTRL"))

import run_ctrl
from ctrl_algorithms.rainbow import RainbowConfig, train_rainbow_offline
from ctrl_algorithms.sac import SACConfig, train_sac_offline


def _timestamped_dir(
    base: Path | None, exp_name: str, use_timestamp: bool = True
) -> Path:
    root = base if base is not None else REPO_ROOT / "results" / exp_name
    return (
        root / dt.datetime.now().strftime("%Y%m%d-%H%M%S") if use_timestamp else root
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CTRL components and baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="If set, do not append a timestamp to output directories.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ds = sub.add_parser("dataset", help="Generate SD dataset.")
    ds.add_argument("--episodes", type=int, default=run_ctrl.DatasetConfig.episodes)
    ds.add_argument("--horizon", type=int, default=run_ctrl.DatasetConfig.horizon)
    ds.add_argument("--seed", type=int, default=run_ctrl.DatasetConfig.seed)
    ds.add_argument("--output", type=Path, default=run_ctrl.DatasetConfig.output)

    bicogan = sub.add_parser("bicogan", help="Train BiCoGAN.")
    bicogan.add_argument("--dataset-path", type=Path, required=True)
    bicogan.add_argument("--output-dir", type=Path)
    bicogan.add_argument(
        "--batch-size",
        type=int,
        default=run_ctrl.BiCoGANConfig.batch_size,
    )
    bicogan.add_argument("--seed", type=int, default=run_ctrl.BiCoGANConfig.seed)
    bicogan.add_argument("--udim", type=int, default=run_ctrl.BiCoGANConfig.udim)
    bicogan.add_argument(
        "--pre-train-lr", type=float, default=run_ctrl.BiCoGANConfig.pre_train_lr
    )
    bicogan.add_argument(
        "--pre-train-epochs",
        type=int,
        default=run_ctrl.BiCoGANConfig.pre_train_epochs,
    )
    bicogan.add_argument("--lr", type=float, default=run_ctrl.BiCoGANConfig.lr)
    bicogan.add_argument(
        "--disc-lr", type=float, default=run_ctrl.BiCoGANConfig.disc_lr
    )
    bicogan.add_argument(
        "--beta1", type=float, default=run_ctrl.BiCoGANConfig.beta1
    )
    bicogan.add_argument(
        "--beta2", type=float, default=run_ctrl.BiCoGANConfig.beta2
    )
    bicogan.add_argument(
        "--num-epochs", type=int, default=run_ctrl.BiCoGANConfig.num_epochs
    )
    bicogan.add_argument(
        "--alpha", type=float, default=run_ctrl.BiCoGANConfig.alpha
    )
    bicogan.add_argument("--rho", type=float, default=run_ctrl.BiCoGANConfig.rho)
    bicogan.add_argument("--phi", type=float, default=run_ctrl.BiCoGANConfig.phi)

    d3qn = sub.add_parser("d3qn", help="Train offline D3QN+CQL.")
    d3qn.add_argument("--dataset-path", type=Path, required=True)
    d3qn.add_argument("--output-dir", type=Path)
    d3qn.add_argument("--seed", type=int, default=run_ctrl.D3QNCliConfig.seed)
    d3qn.add_argument("--use-cf", action="store_true")
    d3qn.add_argument("--cf-k", type=int, default=run_ctrl.D3QNCliConfig.cf_k)
    d3qn.add_argument("--bicogan-dir", type=Path, help="Directory with G.pt/E.pt.")
    d3qn.add_argument(
        "--cf-filter-done",
        action="store_true",
        help="Filter CF transitions with reward <= 0.5.",
    )
    d3qn.add_argument(
        "--cf-sample-frac",
        type=float,
        help="Optional fraction of CF transitions to keep (0-1).",
    )
    d3qn.add_argument(
        "--cf-quality-thresh-x",
        type=float,
        help="Optional abs(x) threshold to keep CF samples.",
    )
    d3qn.add_argument(
        "--cf-quality-thresh-theta",
        type=float,
        help="Optional abs(theta) threshold to keep CF samples.",
    )
    d3qn.add_argument(
        "--cf-action-noise-std",
        type=float,
        default=run_ctrl.D3QNCliConfig.cf_action_noise_std,
        help="Gaussian noise std added to CF continuous actions.",
    )
    d3qn.add_argument(
        "--cf-use-env-step",
        action="store_true",
        help="Use CartPole dynamics to rescore CF reward/done (ignores generator for reward/done).",
    )
    d3qn.add_argument("--udim", type=int, default=run_ctrl.D3QNCliConfig.udim)
    d3qn.add_argument("--epochs", type=int, default=run_ctrl.D3QNCliConfig.epochs)
    d3qn.add_argument(
        "--batch-size", type=int, default=run_ctrl.D3QNCliConfig.batch_size
    )
    d3qn.add_argument("--gamma", type=float, default=run_ctrl.D3QNCliConfig.gamma)
    d3qn.add_argument("--lr", type=float, default=run_ctrl.D3QNCliConfig.lr)
    d3qn.add_argument("--tau", type=float, default=run_ctrl.D3QNCliConfig.tau)
    d3qn.add_argument(
        "--alpha-cql", type=float, default=run_ctrl.D3QNCliConfig.alpha_cql
    )
    d3qn.add_argument(
        "--reward-clip", type=float, default=run_ctrl.D3QNCliConfig.reward_clip
    )
    d3qn.add_argument(
        "--target-clip", type=float, default=run_ctrl.D3QNCliConfig.target_clip
    )
    d3qn.add_argument(
        "--max-grad-norm", type=float, default=run_ctrl.D3QNCliConfig.max_grad_norm
    )

    sac = sub.add_parser("sac", help="Train SAC baseline offline.")
    sac.add_argument("--dataset-path", type=Path, required=True)
    sac.add_argument("--output-dir", type=Path)
    sac.add_argument("--seed", type=int, default=SACConfig.seed)
    sac.add_argument("--epochs", type=int, default=SACConfig.epochs)
    sac.add_argument("--batch-size", type=int, default=SACConfig.batch_size)
    sac.add_argument("--gamma", type=float, default=SACConfig.gamma)
    sac.add_argument("--tau", type=float, default=SACConfig.tau)
    sac.add_argument("--lr", type=float, default=SACConfig.lr)
    sac.add_argument("--alpha", type=float, default=SACConfig.alpha)
    sac.add_argument(
        "--auto-alpha", action="store_true", default=SACConfig.auto_alpha
    )
    sac.add_argument(
        "--no-auto-alpha",
        dest="auto_alpha",
        action="store_false",
        help="Disable entropy auto-tuning.",
    )
    sac.add_argument(
        "--target-entropy", type=float, default=SACConfig.target_entropy
    )
    sac.add_argument("--eval-every", type=int, default=SACConfig.eval_every)

    rb = sub.add_parser("rainbow", help="Train Rainbow DQN baseline offline.")
    rb.add_argument("--dataset-path", type=Path, required=True)
    rb.add_argument("--output-dir", type=Path)
    rb.add_argument("--seed", type=int, default=RainbowConfig.seed)
    rb.add_argument("--epochs", type=int, default=RainbowConfig.epochs)
    rb.add_argument("--batch-size", type=int, default=RainbowConfig.batch_size)
    rb.add_argument("--gamma", type=float, default=RainbowConfig.gamma)
    rb.add_argument("--lr", type=float, default=RainbowConfig.lr)
    rb.add_argument("--tau", type=float, default=RainbowConfig.tau)
    rb.add_argument("--n-atoms", type=int, default=RainbowConfig.n_atoms)
    rb.add_argument("--v-min", type=float, default=RainbowConfig.v_min)
    rb.add_argument("--v-max", type=float, default=RainbowConfig.v_max)
    rb.add_argument("--eval-every", type=int, default=RainbowConfig.eval_every)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    use_timestamp = not args.no_timestamp

    if args.command == "dataset":
        cfg = run_ctrl.DatasetConfig(
            episodes=args.episodes,
            horizon=args.horizon,
            seed=args.seed,
            output=args.output,
        )
        output = run_ctrl.generate_dataset(cfg)
        print(f"[dataset] saved dataset to {output}")
        return

    if args.command == "bicogan":
        output_dir = _timestamped_dir(args.output_dir, "cartpole/bicogan", use_timestamp)
        cfg = run_ctrl.BiCoGANConfig(
            dataset_path=args.dataset_path,
            output_dir=output_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            udim=args.udim,
            pre_train_lr=args.pre_train_lr,
            pre_train_epochs=args.pre_train_epochs,
            lr=args.lr,
            disc_lr=args.disc_lr,
            beta1=args.beta1,
            beta2=args.beta2,
            num_epochs=args.num_epochs,
            alpha=args.alpha,
            rho=args.rho,
            phi=args.phi,
        )
        run_ctrl.train_bicogan(cfg)
        print(f"[bicogan] artifacts saved to {cfg.output_dir}")
        return

    if args.command == "d3qn":
        base_name = "cartpole/d3qn_cf" if args.use_cf else "cartpole/d3qn_real"
        output_dir = _timestamped_dir(args.output_dir, base_name, use_timestamp)
        cfg = run_ctrl.D3QNCliConfig(
            dataset_path=args.dataset_path,
            output_dir=output_dir,
            seed=args.seed,
            use_cf=args.use_cf,
            cf_k=args.cf_k,
            bicogan_dir=args.bicogan_dir,
            cf_filter_done=args.cf_filter_done,
            cf_sample_frac=args.cf_sample_frac,
            cf_quality_thresh_x=args.cf_quality_thresh_x,
            cf_quality_thresh_theta=args.cf_quality_thresh_theta,
            cf_action_noise_std=args.cf_action_noise_std,
            cf_use_env_step=args.cf_use_env_step,
            udim=args.udim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            lr=args.lr,
            tau=args.tau,
            alpha_cql=args.alpha_cql,
            reward_clip=args.reward_clip,
            target_clip=args.target_clip,
            max_grad_norm=args.max_grad_norm,
        )
        run_ctrl.train_d3qn(cfg)
        print(f"[d3qn] artifacts saved to {cfg.output_dir}")
        return

    if args.command == "sac":
        output_dir = _timestamped_dir(args.output_dir, "cartpole/sac", use_timestamp)
        cfg = SACConfig(
            dataset_path=args.dataset_path,
            output_dir=output_dir,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            lr=args.lr,
            alpha=args.alpha,
            auto_alpha=args.auto_alpha,
            target_entropy=args.target_entropy,
            eval_every=args.eval_every,
        )
        train_sac_offline(cfg)
        print(f"[sac] artifacts saved to {cfg.output_dir}")
        return

    if args.command == "rainbow":
        output_dir = _timestamped_dir(args.output_dir, "cartpole/rainbow", use_timestamp)
        cfg = RainbowConfig(
            dataset_path=args.dataset_path,
            output_dir=output_dir,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            lr=args.lr,
            tau=args.tau,
            n_atoms=args.n_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            eval_every=args.eval_every,
        )
        train_rainbow_offline(cfg)
        print(f"[rainbow] artifacts saved to {cfg.output_dir}")
        return


if __name__ == "__main__":
    main()
