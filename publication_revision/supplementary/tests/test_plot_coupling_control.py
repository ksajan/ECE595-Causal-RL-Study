"""Contract tests for frozen coupling-control publication figures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.revision.plot_coupling_control import (
    EXPECTED_SEEDS,
    _finite_seed_values,
    read_summary,
    validate_summary,
)

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/revision/coupling_control_final/summary.json"


def test_frozen_summary_has_exact_seed_and_arm_contract() -> None:
    summary = validate_summary(read_summary(SUMMARY))
    assert tuple(summary["seeds"]) == EXPECTED_SEEDS
    assert set(summary["arms"]) == {
        "random",
        "real",
        "fresh_independent",
        "fresh_shared",
        "oracle_cf",
    }


def test_validation_rejects_missing_seed_value() -> None:
    summary = read_summary(SUMMARY)
    summary["arms"]["real"]["clean"]["seed_means"] = summary["arms"]["real"]["clean"][
        "seed_means"
    ][:-1]
    with pytest.raises(ValueError, match="exactly 30 seed values"):
        validate_summary(summary)


def test_validation_rejects_episode_level_shape() -> None:
    summary = read_summary(SUMMARY)
    summary["contrasts"]["oracle_cf_minus_fresh_shared_clean"]["seed_deltas"] = [
        0.0
    ] * 100
    with pytest.raises(ValueError, match="exactly 30 seed values"):
        validate_summary(summary)


def test_validation_rejects_unregistered_contrast() -> None:
    summary = read_summary(SUMMARY)
    summary["registered_contrasts"] = summary["registered_contrasts"][:-1]
    with pytest.raises(ValueError, match="four registered effects"):
        validate_summary(summary)


def test_validation_rejects_nonfinite_seed_value() -> None:
    summary = read_summary(SUMMARY)
    summary["arms"]["oracle_cf"]["noisy"]["seed_means"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_summary(summary)


def test_summary_read_does_not_mutate_json_source() -> None:
    before = json.dumps(read_summary(SUMMARY), sort_keys=True)
    validate_summary(read_summary(SUMMARY))
    assert json.dumps(read_summary(SUMMARY), sort_keys=True) == before


def test_seed_helper_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="exactly 30 seed values"):
        _finite_seed_values([1.0, 2.0], "test")
