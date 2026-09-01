"""Focused contract tests for audited revision figures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.revision.plot_oracle_confirmatory import (
    MODEL_SEEDS,
    ORACLE_SEEDS,
    _read_json,
    validate_summaries,
)

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "results/revision/oracle_confirmatory_final/summary.json"
MODEL = ROOT / "results/revision/model_gate_final/summary.json"


def test_validated_summaries_have_registered_seed_contract() -> None:
    oracle, model = validate_summaries(_read_json(ORACLE), _read_json(MODEL))
    assert tuple(oracle["seeds"]) == ORACLE_SEEDS
    assert tuple(model["seeds"]) == MODEL_SEEDS


def test_validation_rejects_incomplete_oracle_seed_array() -> None:
    oracle = _read_json(ORACLE)
    model = _read_json(MODEL)
    oracle["arms"]["real"]["clean"]["seed_means"] = oracle["arms"]["real"]["clean"][
        "seed_means"
    ][:-1]
    with pytest.raises(ValueError, match="exactly 30 seed values"):
        validate_summaries(oracle, model)


def test_validation_rejects_nonfinite_model_metric() -> None:
    oracle = _read_json(ORACLE)
    model = _read_json(MODEL)
    model["per_seed_metrics"]["960"]["terminal_disagreement"] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        validate_summaries(oracle, model)


def test_validation_rejects_wrong_primary_contrast() -> None:
    oracle = _read_json(ORACLE)
    model = _read_json(MODEL)
    oracle["registered_primary"] = "oracle_cf_minus_fresh_noise_noisy"
    with pytest.raises(ValueError, match="primary contrast"):
        validate_summaries(oracle, model)


def test_json_summary_is_not_mutated_by_validation() -> None:
    oracle = _read_json(ORACLE)
    model = _read_json(MODEL)
    before = json.dumps(oracle, sort_keys=True)
    validate_summaries(oracle, model)
    assert json.dumps(oracle, sort_keys=True) == before
