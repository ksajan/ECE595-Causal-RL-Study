#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": std,
        "ci95": float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize oracle-CF outputs.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--path-contains", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(args.input_dir.rglob("*_seed_*.json")):
        if args.path_contains and args.path_contains not in str(path):
            continue
        data = json.loads(path.read_text())
        if "mode" not in data:
            continue
        grouped[data["mode"]].append(data)

    if not grouped:
        raise FileNotFoundError(f"No oracle-CF JSON files found under {args.input_dir}")

    summary = {}
    for mode, rows in grouped.items():
        summary[mode] = {
            "seeds": [row["config"]["seed"] for row in rows],
            "dataset_transitions": summarize(
                [float(row["dataset_transitions"]) for row in rows]
            ),
            "clean": summarize([float(row["clean"]["mean"]) for row in rows]),
            "ctrl_noisy": summarize(
                [float(row["ctrl_noisy"]["mean"]) for row in rows]
            ),
        }

    print("Offline CartPole oracle-CF summary")
    for mode, row in sorted(summary.items()):
        print(
            f"{mode}: n={row['clean']['n']} "
            f"clean={row['clean']['mean']:.2f} ± {row['clean']['std']:.2f} "
            f"(ci95 {row['clean']['ci95']:.2f}), "
            f"noisy={row['ctrl_noisy']['mean']:.2f} ± "
            f"{row['ctrl_noisy']['std']:.2f} "
            f"(ci95 {row['ctrl_noisy']['ci95']:.2f})"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
