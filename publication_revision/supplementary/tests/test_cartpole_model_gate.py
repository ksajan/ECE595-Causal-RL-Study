"""Focused tests for the development-only learned SCM model gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.revision.cartpole_model_gate as model_gate
from scripts.revision.summarize_model_gate import EXPECTED_SEEDS, summarize


def _payload(seed: int, root: Path) -> dict:
    """Build a minimal valid model-gate fixture from the current source hashes."""
    config = {
        "seed": seed,
        "experiment_tier": "ctrl_bicogan_reproduction",
        "bicogan_generator": "monotonic_bicogan",
        "noise_semantics": "process",
        "dataset_trials": 250,
        "validation_dataset_trials": 50,
        "trial_horizon": 20,
        "stop_on_termination": True,
        "output_dir": str(root),
    }
    return {
        "artifact_schema": model_gate.ARTIFACT_SCHEMA,
        "development_only": True,
        "config": config,
        "source_hashes": model_gate.source_hashes(),
        "dataset": {
            "noise_semantics": "process",
            "terminal_label_rule": "pre_noise_next_state",
            "post_failure_transitions": 0,
            "validation_post_failure_transitions": 0,
        },
        "bicogan": {
            "diagnostics": {
                "latent_std_by_dimension": [1.0, 1.0, 1.0, 1.0],
                "action_reconstruction_mse": 0.01,
                "central_action_baseline_mse": 0.1,
            },
            "counterfactual_diagnostics": {
                "external_validation": {
                    "normalized_mse": 0.5,
                    "terminal_disagreement": 0.01,
                },
                "fresh_noise_external_validation": {"normalized_mse": 1.0},
            },
        },
    }


def test_summary_accepts_exact_registered_seed_set(tmp_path: Path) -> None:
    for seed in EXPECTED_SEEDS:
        (tmp_path / f"model_gate_seed_{seed}.json").write_text(
            json.dumps(_payload(seed, tmp_path))
        )

    report = summarize(tmp_path)

    assert report["development_only"] is True
    assert report["confirmatory"] is False
    assert report["all_registered_gates_passed"] is True
    assert report["metric_summary"]["learned_to_fresh_mse_ratio"]["mean"] == 0.5


def test_summary_rejects_seed_drift(tmp_path: Path) -> None:
    for seed in EXPECTED_SEEDS[:-1]:
        (tmp_path / f"model_gate_seed_{seed}.json").write_text(
            json.dumps(_payload(seed, tmp_path))
        )
    (tmp_path / "model_gate_seed_999.json").write_text(
        json.dumps(_payload(999, tmp_path))
    )

    with pytest.raises(ValueError, match="Expected seeds"):
        summarize(tmp_path)


def test_summary_rejects_nonfinite_diagnostic(tmp_path: Path) -> None:
    for seed in EXPECTED_SEEDS:
        payload = _payload(seed, tmp_path)
        if seed == EXPECTED_SEEDS[0]:
            payload["bicogan"]["diagnostics"]["action_reconstruction_mse"] = "nan"
        (tmp_path / f"model_gate_seed_{seed}.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="must be finite"):
        summarize(tmp_path)


def test_summary_rejects_source_drift(tmp_path: Path) -> None:
    for seed in EXPECTED_SEEDS:
        payload = _payload(seed, tmp_path)
        if seed == EXPECTED_SEEDS[0]:
            payload["source_hashes"]["bicogan_ctrl.py"] = "changed"
        (tmp_path / f"model_gate_seed_{seed}.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="source/configuration hash drift"):
        summarize(tmp_path)
