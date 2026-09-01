#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ci95(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": len(arr),
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ci95": ci95(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CartPole sanity JSONs.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("sanity_seed_*.json"))
    if not files:
        raise FileNotFoundError(f"No sanity JSON files found under {args.input_dir}")

    rows = []
    for path in files:
        data = json.loads(path.read_text())
        logs = data.get("training_logs", [])
        best_log = max(logs, key=lambda item: item["clean_mean"]) if logs else {}
        rows.append(
            {
                "seed": data["config"]["seed"],
                "random_clean": data["random_clean"]["mean"],
                "random_ctrl_noisy": data["random_ctrl"]["mean"],
                "best_ep": best_log.get("episode"),
                "best_clean": best_log.get("clean_mean"),
                "best_ctrl_noisy_at_best": best_log.get("ctrl_mean"),
                "final_selected_clean": data["online_clean"]["mean"],
                "final_selected_ctrl_noisy": data["online_ctrl"]["mean"],
            }
        )

    metrics = {
        key: summarize([float(row[key]) for row in rows if row[key] is not None])
        for key in [
            "random_clean",
            "random_ctrl_noisy",
            "best_clean",
            "best_ctrl_noisy_at_best",
            "final_selected_clean",
            "final_selected_ctrl_noisy",
        ]
    }
    payload = {"source_dir": str(args.input_dir), "seeds": rows, "metrics": metrics}

    print("CartPole 11-action sanity summary")
    print(f"files: {len(files)}")
    for key, val in metrics.items():
        print(
            f"{key}: n={val['n']} mean={val['mean']:.2f} "
            f"std={val['std']:.2f} ci95={val['ci95']:.2f}"
        )
    print(f"best checkpoint episodes: {[row['best_ep'] for row in rows]}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
