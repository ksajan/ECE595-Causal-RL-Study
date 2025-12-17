#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.demo_policy import run_demo


def _timestamped_dir(algo: str) -> Path:
    return REPO_ROOT / "results" / "infer" / algo / dt.datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference/rendering for trained policies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo", choices=["sac", "rainbow", "d3qn"], required=True, help="Policy type."
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Checkpoint file.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Dataset to compute normalization stats.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record videos to results/infer/<algo>/<timestamp>/ by default.",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        help="Override record directory. Ignored unless --record is set.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    record_dir = None
    if args.record:
        record_dir = args.record_dir if args.record_dir else _timestamped_dir(args.algo)

    run_demo(
        algo=args.algo,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        episodes=args.episodes,
        seed=args.seed,
        record_dir=record_dir,
    )
    if record_dir:
        print(f"[infer:{args.algo}] recorded to {record_dir}")


if __name__ == "__main__":
    main()
