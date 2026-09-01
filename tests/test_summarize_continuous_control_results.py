from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.workshop.summarize_continuous_control_results import summarize


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mujoco_artifact(
    root: Path,
    task: str,
    variant: str,
    seed: int,
    job_id: str,
    value: float,
    timesteps: int = 20_000,
) -> None:
    run_dir = root / task / variant / f"seed_{seed}" / job_id
    eval_seeds = [900_000, 900_001, 900_002]
    _write_json(
        run_dir / "config.json",
        {
            "env_id": task,
            "variant": variant,
            "seed": seed,
            "job_id": job_id,
            "total_timesteps": timesteps,
            "eval_episodes": len(eval_seeds),
            "eval_seed_base": eval_seeds[0],
            "intervention_scale": 0.2,
            "learning_starts": 1_000,
            "buffer_size": 40_000,
            "batch_size": 256,
            "train_freq": 1,
            "gradient_steps": 1,
        },
    )
    _write_json(
        run_dir / "eval.json",
        {
            "episodes": [
                {"seed": eval_seed, "return": value, "length": 10}
                for eval_seed in eval_seeds
            ],
            "mean_return": value,
            "std_return": 2.0,
            "median_return": value,
            "evaluation_seed_bank": eval_seeds,
            "deterministic_policy": True,
        },
    )


def _d4rl_artifact(
    root: Path,
    variant: str,
    seed: int,
    job_id: str,
    raw_value: float,
    normalized_value: float,
) -> None:
    dataset = "halfcheetah-medium-v2"
    run_dir = root / dataset / variant / f"seed_{seed}"
    eval_seeds = [700_000, 700_001]
    arguments = {
        "steps": 50_000,
        "steps_per_epoch": 10_000,
        "batch_size": 256,
        "eval_episodes": len(eval_seeds),
        "eval_seed_base": eval_seeds[0],
        "augmentation_seed": 0,
        "intervention_scale": 0.1,
        "cf_fraction": 0.5,
        "max_transitions": None,
        "gamma": 0.99,
        "actor_learning_rate": 0.0001,
        "critic_learning_rate": 0.0003,
        "temp_learning_rate": 0.0001,
        "alpha_learning_rate": 0.0,
        "encoder_hidden_units": [256, 256, 256],
        "tau": 0.005,
        "n_critics": 2,
        "conservative_weight": 10.0,
        "n_action_samples": 10,
        "compile_graph": False,
    }
    _write_json(
        run_dir / "config.json",
        {
            "created_at_utc": job_id,
            "arguments": arguments,
            "transition_count": 100_000,
            "synthetic_transition_count": 50_000 if variant != "real" else 0,
        },
    )
    _write_json(
        run_dir / "evaluation.json",
        {
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "environment_id": "HalfCheetah-v4",
            "training_updates": 50_000,
            "batch_size": 256,
            "transition_count": 100_000,
            "synthetic_transition_count": 50_000 if variant != "real" else 0,
            "evaluation_seeds": eval_seeds,
            "episode_returns_raw": [raw_value - 1.0, raw_value + 1.0],
            "episode_returns_normalized_d4rl": [
                normalized_value - 0.5,
                normalized_value + 0.5,
            ],
            "episode_lengths": [1_000, 1_000],
            "raw_mean": raw_value,
            "raw_std": 1.414213562,
            "normalized_mean": normalized_value,
            "normalized_std": 0.707106781,
        },
    )


def _row(rows: list[dict[str, Any]], **matches: Any) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in matches.items())
    ]
    assert len(selected) == 1
    return selected[0]


def test_summarize_emits_tidy_statistics_and_paired_seed_deltas(
    tmp_path: Path,
) -> None:
    mujoco = tmp_path / "mujoco"
    d4rl = tmp_path / "d4rl"
    output = tmp_path / "summary"
    for seed, real, oracle in ((0, 10.0, 13.0), (1, 20.0, 24.0)):
        _mujoco_artifact(mujoco, "Hopper-v4", "real", seed, "pilot-v1", real)
        _mujoco_artifact(mujoco, "Hopper-v4", "oracle_cf", seed, "pilot-v1", oracle)
    _d4rl_artifact(d4rl, "real", 0, "2026-09-01T00:00:00Z", 1_000.0, 10.0)
    _d4rl_artifact(
        d4rl,
        "factual_residual",
        0,
        "2026-09-01T00:01:00Z",
        1_100.0,
        11.0,
    )

    outputs = summarize(
        [mujoco], [d4rl], output, bootstrap_samples=2_000, bootstrap_seed=17
    )

    assert len(outputs["run_results"]) == 8
    aggregate = _row(
        outputs["aggregate_results"],
        domain="mujoco_sac",
        task="Hopper-v4",
        variant="real",
        metric="return",
    )
    assert aggregate["n"] == 2
    assert aggregate["mean"] == pytest.approx(15.0)
    assert aggregate["std"] == pytest.approx(7.071067812)
    assert aggregate["median"] == pytest.approx(15.0)
    assert aggregate["bootstrap_ci95_low"] == pytest.approx(10.0)
    assert aggregate["bootstrap_ci95_high"] == pytest.approx(20.0)
    assert aggregate["budget_unit"] == "real_environment_interactions"
    assert aggregate["budget_value"] == 20_000
    assert aggregate["total_timesteps"] == 20_000

    paired = [
        row
        for row in outputs["paired_deltas"]
        if row["domain"] == "mujoco_sac" and row["variant"] == "oracle_cf"
    ]
    assert [row["seed"] for row in paired] == [0, 1]
    assert [row["delta_vs_real"] for row in paired] == pytest.approx([3.0, 4.0])
    paired_summary = _row(
        outputs["paired_summary"],
        domain="mujoco_sac",
        variant="oracle_cf",
        metric="return",
    )
    assert paired_summary["n"] == 2
    assert paired_summary["mean"] == pytest.approx(3.5)
    assert paired_summary["paired_seeds"] == "[0, 1]"

    d4rl_run = _row(
        outputs["run_results"],
        domain="d4rl_cql",
        variant="factual_residual",
        metric="normalized_d4rl_score",
    )
    assert d4rl_run["training_updates"] == 50_000
    assert d4rl_run["transition_count"] == 100_000
    assert d4rl_run["synthetic_transition_count"] == 50_000
    assert d4rl_run["cf_fraction"] == pytest.approx(0.5)
    assert json.loads(d4rl_run["protocol_json"])["steps"] == 50_000

    for stem in (
        "run_results",
        "aggregate_results",
        "paired_deltas",
        "paired_summary",
    ):
        assert (output / f"{stem}.json").is_file()
        csv_path = output / f"{stem}.csv"
        assert csv_path.is_file()
        with csv_path.open(encoding="utf-8", newline="") as handle:
            assert list(csv.DictReader(handle))


def test_pairing_uses_only_matching_task_seed_and_protocol(tmp_path: Path) -> None:
    mujoco = tmp_path / "mujoco"
    output = tmp_path / "summary"
    _mujoco_artifact(mujoco, "Hopper-v4", "real", 0, "pilot-v1", 10.0)
    _mujoco_artifact(mujoco, "Hopper-v4", "oracle_cf", 1, "pilot-v1", 50.0)
    _mujoco_artifact(mujoco, "Walker2d-v4", "real", 1, "pilot-v1", 20.0)
    _mujoco_artifact(
        mujoco,
        "Walker2d-v4",
        "oracle_cf",
        1,
        "full-v1",
        80.0,
        timesteps=40_000,
    )

    outputs = summarize([mujoco], [], output, bootstrap_samples=500, bootstrap_seed=3)

    assert outputs["paired_deltas"] == []
    assert outputs["paired_summary"] == []
    assert json.loads((output / "paired_deltas.json").read_text()) == []


def test_duplicate_protocol_job_id_fails_clearly(tmp_path: Path) -> None:
    first = tmp_path / "copy_a"
    second = tmp_path / "copy_b"
    _mujoco_artifact(first, "Ant-v4", "real", 2, "full-v2", 100.0)
    _mujoco_artifact(second, "Ant-v4", "real", 2, "full-v2", 100.0)

    with pytest.raises(ValueError, match="Duplicate run.*task/variant/seed"):
        summarize(
            [first, second],
            [],
            tmp_path / "summary",
            bootstrap_samples=500,
        )

    assert not (tmp_path / "summary").exists()


def test_bootstrap_outputs_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "mujoco"
    for seed, value in enumerate((5.0, 11.0, 19.0, 31.0)):
        _mujoco_artifact(root, "Ant-v4", "real", seed, "full-v1", value)

    first = summarize(
        [root], [], tmp_path / "one", bootstrap_samples=3_000, bootstrap_seed=91
    )
    second = summarize(
        [root], [], tmp_path / "two", bootstrap_samples=3_000, bootstrap_seed=91
    )

    assert first == second
    assert (tmp_path / "one" / "aggregate_results.json").read_text() == (
        tmp_path / "two" / "aggregate_results.json"
    ).read_text()
