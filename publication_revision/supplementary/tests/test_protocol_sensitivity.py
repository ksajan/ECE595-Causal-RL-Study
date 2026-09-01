"""Tests for the isolated CartPole protocol-sensitivity diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.revision.cartpole_protocol_sensitivity import (
    ARM_NAMES,
    ARTIFACT_SCHEMA,
    EXPECTED_SEEDS,
    config_dict,
    make_config,
    source_hashes,
)
from scripts.revision.summarize_protocol_sensitivity import summarize


def _payload(seed: int, root: Path) -> dict[str, object]:
    """Build a compact valid fixture using the current provenance hashes."""
    config = make_config(seed, root)
    arms: dict[str, object] = {}
    for name in ARM_NAMES:
        noise_semantics = "process" if name.startswith("process") else "observation"
        stop = name.endswith("stop")
        arm_config = {
            **config_dict(config),
            "noise_semantics": noise_semantics,
            "stop_on_termination": stop,
        }
        arms[name] = {
            "arm": name,
            "config": arm_config,
            "dataset": {
                "real_transitions": 100.0,
                "post_failure_transitions": 0,
            },
            "training": {},
            "clean": {"returns": [1.0] * 100},
            "noisy": {"returns": [2.0] * 100},
        }
    return {
        "artifact_schema": ARTIFACT_SCHEMA,
        "development_only": True,
        "confirmatory": False,
        "experiment_tier": "development_protocol_sensitivity",
        "config": config_dict(config),
        "matrix": {
            "arms": list(ARM_NAMES),
            "evaluation_seed_range": [600_000, 600_099],
            "evaluation_episodes": 100,
        },
        "source_hashes": source_hashes(),
        "arms": arms,
    }


def test_summary_accepts_exact_registered_seed_set(tmp_path: Path) -> None:
    for seed in EXPECTED_SEEDS:
        (tmp_path / f"protocol_sensitivity_seed_{seed}.json").write_text(
            json.dumps(_payload(seed, tmp_path))
        )

    report = summarize(tmp_path)

    assert report["development_only"] is True
    assert report["confirmatory"] is False
    assert report["seeds"] == list(EXPECTED_SEEDS)
    assert report["arm_summary"]["process_stop"]["clean"]["mean"] == 1.0
    assert report["contrasts"]["process_vs_observation_stop"]["clean"]["mean"] == 0.0


def test_summary_rejects_missing_seed(tmp_path: Path) -> None:
    for seed in EXPECTED_SEEDS[:-1]:
        (tmp_path / f"protocol_sensitivity_seed_{seed}.json").write_text(
            json.dumps(_payload(seed, tmp_path))
        )

    with pytest.raises(ValueError, match="Expected exactly 10 artifacts"):
        summarize(tmp_path)


def test_summary_rejects_source_drift(tmp_path: Path) -> None:
    for seed in EXPECTED_SEEDS:
        payload = _payload(seed, tmp_path)
        if seed == EXPECTED_SEEDS[0]:
            payload["source_hashes"]["cartpole_sanity.py"] = "changed"  # type: ignore[index]
        (tmp_path / f"protocol_sensitivity_seed_{seed}.json").write_text(
            json.dumps(payload)
        )

    with pytest.raises(ValueError, match="source hash drift"):
        summarize(tmp_path)


def test_summary_rejects_nonfinite_returns(tmp_path: Path) -> None:
    for seed in EXPECTED_SEEDS:
        payload = _payload(seed, tmp_path)
        if seed == EXPECTED_SEEDS[0]:
            payload["arms"]["process_stop"]["clean"]["returns"][0] = "nan"  # type: ignore[index]
        (tmp_path / f"protocol_sensitivity_seed_{seed}.json").write_text(
            json.dumps(payload)
        )

    with pytest.raises(ValueError, match="must be finite"):
        summarize(tmp_path)
