"""Tests for the frozen oracle-CF confirmatory harness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.revision.cartpole_oracle_confirmatory import (
    SCHEMA,
    config_from_manifest,
    load_manifest,
    source_digest,
    source_manifest_digest,
    validate_config,
)
from scripts.revision.summarize_oracle_confirmatory import (
    describe,
    holm_adjust,
    paired_statistics,
    validate_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "scripts/revision/oracle_confirmatory_manifest.json"


def test_manifest_freezes_seed_set_and_configuration() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest["artifact_schema"] == SCHEMA
    assert manifest["seeds"] == list(range(1000, 1030))
    config = config_from_manifest(manifest, 1000, Path("results/test"))
    validate_config(config, manifest)


def test_runner_rejects_seed_or_configuration_drift() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    with pytest.raises(ValueError, match="outside the manifest"):
        validate_config(
            config_from_manifest(manifest, 999, Path("results/test")), manifest
        )
    config = config_from_manifest(manifest, 1000, Path("results/test"))
    drifted = replace(config, train_steps=5001)
    with pytest.raises(ValueError, match="Configuration drift for train_steps"):
        validate_config(drifted, manifest)


def _artifact(seed: int, *, manifest: dict[str, object]) -> dict[str, object]:
    returns = {
        "clean": [float(seed - 1000 + 10)] * 100,
        "noisy": [float(seed - 1000 + 5)] * 100,
    }
    arms = {
        arm: {
            metric: {
                "episodes": 100,
                "mean": values[0],
                "returns": values,
            }
            for metric, values in returns.items()
        }
        for arm in ("random", "real", "fresh_noise", "oracle_cf")
    }
    config = dict(manifest["config"])
    config["seed"] = seed
    return {
        "artifact_schema": SCHEMA,
        "experiment_name": "cartpole_oracle_confirmatory",
        "config": config,
        "command": ["python", "runner.py"],
        "git": {"commit": "test", "dirty": False},
        "software": {"python": "3.13"},
        "source_sha256": "source",
        "manifest_sha256": "manifest",
        "source_manifest_sha256": "combined",
        "dataset": {
            "postfailure_transitions": 0,
            "post_failure_transitions": 0,
        },
        "arms": arms,
    }


def test_summary_validator_requires_exact_complete_seed_set(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    for seed in range(1000, 1030):
        (tmp_path / f"oracle_seed_{seed}.json").write_text(
            json.dumps(_artifact(seed, manifest=manifest))
        )
    # Replace provenance values with the actual current values so the fixture
    # tests the seed and artifact contracts rather than hash checking.
    for path in tmp_path.glob("oracle_seed_*.json"):
        record = json.loads(path.read_text())
        record["source_sha256"] = source_digest()
        record["manifest_sha256"] = hashlib.sha256(
            MANIFEST_PATH.read_bytes()
        ).hexdigest()
        record["source_manifest_sha256"] = source_manifest_digest(MANIFEST_PATH)
        path.write_text(json.dumps(record))
    records, _ = validate_artifacts(tmp_path, MANIFEST_PATH)
    assert [record["config"]["seed"] for record in records] == list(range(1000, 1030))
    (tmp_path / "oracle_seed_1000.json").unlink()
    with pytest.raises(ValueError, match="Expected exactly 30 artifacts"):
        validate_artifacts(tmp_path, MANIFEST_PATH)


def test_summary_validator_rejects_nonfinite_artifact_data(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    for seed in range(1000, 1030):
        record = _artifact(seed, manifest=manifest)
        if seed == 1000:
            record["arms"]["real"]["clean"]["returns"][0] = float("nan")
        (tmp_path / f"oracle_seed_{seed}.json").write_text(
            json.dumps(record, allow_nan=True)
        )
    with pytest.raises(ValueError, match="non-finite"):
        validate_artifacts(tmp_path, MANIFEST_PATH)


def test_summary_statistics_are_seed_level_and_holm_adjusted() -> None:
    left = np.asarray([4.0, 5.0, 6.0, 7.0])
    right = np.asarray([1.0, 2.0, 3.0, 4.0])
    result = paired_statistics(left, right, np.random.default_rng(4), 200, 500, 1.0)
    assert result["n_training_seeds"] == 4
    assert result["mean_delta"] == pytest.approx(3.0)
    assert len(result["seed_deltas"]) == 4
    assert result["effect_classification"] == "practically_meaningful_benefit"
    assert describe(left)["n_training_seeds"] == 4
    assert holm_adjust([0.01, 0.02, 0.5, 1.0]) == pytest.approx([0.04, 0.06, 1.0, 1.0])


def test_effect_classification_is_symmetric() -> None:
    left = np.asarray([1.0, 2.0, 3.0, 4.0])
    right = left + 3.0
    result = paired_statistics(left, right, np.random.default_rng(7), 200, 500, 1.0)
    assert result["effect_classification"] == "practically_meaningful_harm"
