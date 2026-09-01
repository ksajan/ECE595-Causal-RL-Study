#!/usr/bin/env python3
"""Aggregate reviewer-facing MuJoCo SAC and D4RL CQL result artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

MUJOCO_PROTOCOL_FIELDS = (
    "total_timesteps",
    "eval_episodes",
    "eval_seed_base",
    "intervention_scale",
    "learning_starts",
    "buffer_size",
    "batch_size",
    "train_freq",
    "gradient_steps",
)
D4RL_PROTOCOL_FIELDS = (
    "steps",
    "steps_per_epoch",
    "batch_size",
    "eval_episodes",
    "eval_seed_base",
    "augmentation_seed",
    "intervention_scale",
    "cf_fraction",
    "max_transitions",
    "gamma",
    "actor_learning_rate",
    "critic_learning_rate",
    "temp_learning_rate",
    "alpha_learning_rate",
    "encoder_hidden_units",
    "tau",
    "n_critics",
    "conservative_weight",
    "n_action_samples",
    "compile_graph",
)
OUTPUT_STEMS = (
    "run_results",
    "aggregate_results",
    "paired_deltas",
    "paired_summary",
)


@dataclass(frozen=True)
class ArtifactRun:
    """One completed training run and its seed-level evaluation metrics."""

    domain: str
    task: str
    variant: str
    seed: int
    protocol_job_id: str
    protocol_id: str
    protocol_json: str
    budget_unit: str
    budget_value: int
    eval_episodes: int
    eval_seed_base: int
    batch_size: int
    total_timesteps: int | None
    training_updates: int | None
    transition_count: int | None
    synthetic_transition_count: int | None
    cf_fraction: float | None
    intervention_scale: float
    artifact_path: str
    metrics: Mapping[str, float]
    within_run_std: Mapping[str, float]


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and report its path when decoding fails."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _required(payload: Mapping[str, Any], key: str, path: Path) -> Any:
    """Return a required field or raise an artifact-specific error."""

    if key not in payload or payload[key] is None:
        raise ValueError(f"Missing required field {key!r} in {path}")
    return payload[key]


def _finite_float(value: Any, field: str, path: Path) -> float:
    """Convert a metric to a finite float."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {field!r} in {path} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"Field {field!r} in {path} must be finite")
    return result


def _integer(value: Any, field: str, path: Path) -> int:
    """Convert an integral JSON value without accepting fractional values."""

    if isinstance(value, bool):
        raise TypeError(f"Field {field!r} in {path} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {field!r} in {path} must be an integer") from exc
    if float(value) != result:
        raise ValueError(f"Field {field!r} in {path} must be an integer")
    return result


def _canonical_protocol(fields: Mapping[str, Any]) -> tuple[str, str]:
    """Return canonical protocol JSON and its short stable identifier."""

    protocol_json = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    protocol_id = hashlib.sha256(protocol_json.encode("utf-8")).hexdigest()[:16]
    return protocol_json, protocol_id


def _protocol_fields(
    arguments: Mapping[str, Any], names: Sequence[str], path: Path
) -> dict[str, Any]:
    """Extract a bounded set of comparison-defining protocol fields."""

    fields: dict[str, Any] = {}
    for name in names:
        if name not in arguments:
            raise ValueError(f"Missing required protocol field {name!r} in {path}")
        fields[name] = arguments[name]
    return fields


def _evaluation_seed_base(
    evaluation: Mapping[str, Any], config_value: Any, path: Path
) -> int:
    """Validate the fixed evaluation seed bank and return its first seed."""

    seeds_value = evaluation.get("evaluation_seed_bank")
    if seeds_value is None:
        seeds_value = evaluation.get("evaluation_seeds")
    if not isinstance(seeds_value, list) or not seeds_value:
        raise ValueError(f"Missing non-empty evaluation seed bank in {path}")
    seeds = [_integer(value, "evaluation seed", path) for value in seeds_value]
    expected = list(range(seeds[0], seeds[0] + len(seeds)))
    if seeds != expected:
        raise ValueError(f"Evaluation seeds must be consecutive in {path}: {seeds}")
    configured = _integer(config_value, "eval_seed_base", path)
    if seeds[0] != configured:
        raise ValueError(
            f"Evaluation seed base mismatch in {path}: {seeds[0]} != {configured}"
        )
    return seeds[0]


def _load_mujoco_run(eval_path: Path) -> ArtifactRun:
    """Load one MuJoCo ``eval.json`` and its adjacent run configuration."""

    evaluation = _read_json(eval_path)
    config_path = eval_path.with_name("config.json")
    config = _read_json(config_path)
    fields = _protocol_fields(config, MUJOCO_PROTOCOL_FIELDS, config_path)
    fields.update({"algorithm": "SAC", "domain": "online_mujoco"})
    protocol_json, protocol_id = _canonical_protocol(fields)
    job_id = str(_required(config, "job_id", config_path)).strip()
    if not job_id:
        raise ValueError(f"Empty protocol job id in {config_path}")
    episodes = _integer(fields["eval_episodes"], "eval_episodes", config_path)
    seed_base = _evaluation_seed_base(evaluation, fields["eval_seed_base"], eval_path)
    episode_rows = _required(evaluation, "episodes", eval_path)
    if not isinstance(episode_rows, list) or len(episode_rows) != episodes:
        raise ValueError(
            f"Evaluation episode count mismatch in {eval_path}: expected {episodes}"
        )
    mean_return = _finite_float(
        _required(evaluation, "mean_return", eval_path), "mean_return", eval_path
    )
    std_return = _finite_float(
        _required(evaluation, "std_return", eval_path), "std_return", eval_path
    )
    total_timesteps = _integer(
        fields["total_timesteps"], "total_timesteps", config_path
    )
    return ArtifactRun(
        domain="mujoco_sac",
        task=str(_required(config, "env_id", config_path)),
        variant=str(_required(config, "variant", config_path)),
        seed=_integer(_required(config, "seed", config_path), "seed", config_path),
        protocol_job_id=job_id,
        protocol_id=protocol_id,
        protocol_json=protocol_json,
        budget_unit="real_environment_interactions",
        budget_value=total_timesteps,
        eval_episodes=episodes,
        eval_seed_base=seed_base,
        batch_size=_integer(fields["batch_size"], "batch_size", config_path),
        total_timesteps=total_timesteps,
        training_updates=None,
        transition_count=None,
        synthetic_transition_count=None,
        cf_fraction=None,
        intervention_scale=_finite_float(
            fields["intervention_scale"], "intervention_scale", config_path
        ),
        artifact_path=str(eval_path.resolve()),
        metrics={"return": mean_return},
        within_run_std={"return": std_return},
    )


def _d4rl_job_id(
    evaluation: Mapping[str, Any], config: Mapping[str, Any], path: Path
) -> str:
    """Resolve the explicit job id or the runner's immutable creation timestamp."""

    arguments = config.get("arguments", {})
    if not isinstance(arguments, dict):
        raise TypeError(f"Field 'arguments' in {path} must be an object")
    candidates = (
        evaluation.get("protocol_job_id"),
        config.get("protocol_job_id"),
        arguments.get("protocol_job_id"),
        arguments.get("job_id"),
        config.get("created_at_utc"),
    )
    for candidate in candidates:
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    raise ValueError(
        f"Missing protocol job id in {path}; provide protocol_job_id, job_id, "
        "or created_at_utc"
    )


def _load_d4rl_run(evaluation_path: Path) -> ArtifactRun:
    """Load one D4RL ``evaluation.json`` and its adjacent run configuration."""

    evaluation = _read_json(evaluation_path)
    config_path = evaluation_path.with_name("config.json")
    config = _read_json(config_path)
    arguments = _required(config, "arguments", config_path)
    if not isinstance(arguments, dict):
        raise TypeError(f"Field 'arguments' in {config_path} must be an object")
    fields = _protocol_fields(arguments, D4RL_PROTOCOL_FIELDS, config_path)
    fields.update({"algorithm": "CQL", "domain": "d4rl_offline"})
    protocol_json, protocol_id = _canonical_protocol(fields)
    episodes = _integer(fields["eval_episodes"], "eval_episodes", config_path)
    seed_base = _evaluation_seed_base(
        evaluation, fields["eval_seed_base"], evaluation_path
    )
    raw_returns = _required(evaluation, "episode_returns_raw", evaluation_path)
    normalized_returns = _required(
        evaluation, "episode_returns_normalized_d4rl", evaluation_path
    )
    if not isinstance(raw_returns, list) or len(raw_returns) != episodes:
        raise ValueError(f"Raw evaluation episode count mismatch in {evaluation_path}")
    if not isinstance(normalized_returns, list) or len(normalized_returns) != episodes:
        raise ValueError(
            f"Normalized evaluation episode count mismatch in {evaluation_path}"
        )
    updates = _integer(
        _required(evaluation, "training_updates", evaluation_path),
        "training_updates",
        evaluation_path,
    )
    if updates != _integer(fields["steps"], "steps", config_path):
        raise ValueError(f"Training update budget mismatch in {evaluation_path}")
    transition_count = _integer(
        _required(evaluation, "transition_count", evaluation_path),
        "transition_count",
        evaluation_path,
    )
    synthetic_count = _integer(
        _required(evaluation, "synthetic_transition_count", evaluation_path),
        "synthetic_transition_count",
        evaluation_path,
    )
    return ArtifactRun(
        domain="d4rl_cql",
        task=str(_required(evaluation, "dataset", evaluation_path)),
        variant=str(_required(evaluation, "variant", evaluation_path)),
        seed=_integer(
            _required(evaluation, "seed", evaluation_path), "seed", evaluation_path
        ),
        protocol_job_id=_d4rl_job_id(evaluation, config, config_path),
        protocol_id=protocol_id,
        protocol_json=protocol_json,
        budget_unit="gradient_updates",
        budget_value=updates,
        eval_episodes=episodes,
        eval_seed_base=seed_base,
        batch_size=_integer(fields["batch_size"], "batch_size", config_path),
        total_timesteps=None,
        training_updates=updates,
        transition_count=transition_count,
        synthetic_transition_count=synthetic_count,
        cf_fraction=_finite_float(fields["cf_fraction"], "cf_fraction", config_path),
        intervention_scale=_finite_float(
            fields["intervention_scale"], "intervention_scale", config_path
        ),
        artifact_path=str(evaluation_path.resolve()),
        metrics={
            "raw_return": _finite_float(
                _required(evaluation, "raw_mean", evaluation_path),
                "raw_mean",
                evaluation_path,
            ),
            "normalized_d4rl_score": _finite_float(
                _required(evaluation, "normalized_mean", evaluation_path),
                "normalized_mean",
                evaluation_path,
            ),
        },
        within_run_std={
            "raw_return": _finite_float(
                _required(evaluation, "raw_std", evaluation_path),
                "raw_std",
                evaluation_path,
            ),
            "normalized_d4rl_score": _finite_float(
                _required(evaluation, "normalized_std", evaluation_path),
                "normalized_std",
                evaluation_path,
            ),
        },
    )


def scan_artifacts(
    mujoco_roots: Iterable[Path], d4rl_roots: Iterable[Path]
) -> list[ArtifactRun]:
    """Scan only the two reviewer runner artifact names under explicit roots."""

    runs: list[ArtifactRun] = []
    for root in mujoco_roots:
        if not root.is_dir():
            raise ValueError(f"MuJoCo artifact root is not a directory: {root}")
        runs.extend(_load_mujoco_run(path) for path in sorted(root.rglob("eval.json")))
    for root in d4rl_roots:
        if not root.is_dir():
            raise ValueError(f"D4RL artifact root is not a directory: {root}")
        runs.extend(
            _load_d4rl_run(path) for path in sorted(root.rglob("evaluation.json"))
        )
    if not runs:
        raise ValueError("No MuJoCo eval.json or D4RL evaluation.json artifacts found")

    seen: dict[tuple[str, str, str, int, str], ArtifactRun] = {}
    for run in runs:
        key = (
            run.domain,
            run.task,
            run.variant,
            run.seed,
            run.protocol_job_id,
        )
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                "Duplicate run for task/variant/seed/protocol job id "
                f"{key}: {previous.artifact_path} and {run.artifact_path}"
            )
        seen[key] = run
    return sorted(
        runs,
        key=lambda run: (
            run.domain,
            run.task,
            run.protocol_id,
            run.variant,
            run.seed,
            run.protocol_job_id,
        ),
    )


def _base_row(run: ArtifactRun) -> dict[str, Any]:
    """Return protocol and budget columns shared by all tidy outputs."""

    row = asdict(run)
    row.pop("metrics")
    row.pop("within_run_std")
    return row


def tidy_run_rows(runs: Sequence[ArtifactRun]) -> list[dict[str, Any]]:
    """Expand one artifact run into one row per reported metric."""

    rows: list[dict[str, Any]] = []
    for run in runs:
        for metric, value in sorted(run.metrics.items()):
            rows.append(
                {
                    **_base_row(run),
                    "metric": metric,
                    "value": value,
                    "within_run_std": run.within_run_std[metric],
                }
            )
    return rows


def bootstrap_mean_ci(
    values: Sequence[float], samples: int, seed: int
) -> tuple[float, float]:
    """Compute a deterministic percentile 95% bootstrap CI for the mean."""

    if not values:
        raise ValueError("Cannot bootstrap an empty sample")
    if not 100 <= samples <= 1_000_000:
        raise ValueError("bootstrap samples must be in [100, 1000000]")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk_size = min(samples, max(1, 1_000_000 // array.size))
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        means[start:stop] = np.mean(array[indices], axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _summary_statistics(
    values: Sequence[float], bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, Any]:
    """Summarize independent training-seed values."""

    array = np.asarray(values, dtype=np.float64)
    low, high = bootstrap_mean_ci(values, bootstrap_samples, bootstrap_seed)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
    }


def _finite_pvalue(value: float) -> float:
    """Convert degenerate numerical-test output to a conservative p-value."""

    return float(value) if np.isfinite(value) else 1.0


def _sign_randomization_pvalue(values: np.ndarray, seed: int) -> float:
    """Compute a deterministic two-sided paired sign-randomization p-value."""

    observed = abs(float(values.mean()))
    count = 1 << values.size
    if values.size <= 16:
        masks = np.arange(count, dtype=np.uint32)[:, None]
        bits = (masks >> np.arange(values.size, dtype=np.uint32)) & 1
        signs = bits.astype(np.float64) * 2.0 - 1.0
        randomized = np.abs((signs * values).mean(axis=1))
        return float(np.count_nonzero(randomized >= observed - 1e-12) / count)
    samples = 100_000
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(samples, values.size))
    randomized = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(randomized >= observed - 1e-12) + 1) / (samples + 1))


def _paired_inference(values: Sequence[float], seed: int) -> dict[str, Any]:
    """Return paired tests and sign counts over independent training seeds."""

    array = np.asarray(values, dtype=np.float64)
    result: dict[str, Any] = {
        "positive_seeds": int(np.count_nonzero(array > 0.0)),
        "negative_seeds": int(np.count_nonzero(array < 0.0)),
        "ties": int(np.count_nonzero(array == 0.0)),
        "paired_t_p": None,
        "wilcoxon_p": None,
        "sign_randomization_p": None,
        "paired_t_p_holm": None,
        "wilcoxon_p_holm": None,
        "sign_randomization_p_holm": None,
    }
    if array.size < 2:
        return result
    try:
        wilcoxon = _finite_pvalue(float(stats.wilcoxon(array).pvalue))
    except ValueError:
        wilcoxon = 1.0
    result.update(
        {
            "paired_t_p": _finite_pvalue(float(stats.ttest_1samp(array, 0.0).pvalue)),
            "wilcoxon_p": wilcoxon,
            "sign_randomization_p": _sign_randomization_pvalue(array, seed),
        }
    )
    return result


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Apply Holm step-down family-wise error correction."""

    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.ones(values.size, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (values.size - rank)))
        adjusted[index] = running
    return [float(value) for value in adjusted]


def _add_holm_adjustments(rows: list[dict[str, Any]]) -> None:
    """Adjust each test across task/variant contrasts in one protocol family."""

    families: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[(row["domain"], row["protocol_id"], row["metric"])].append(row)
    for family_rows in families.values():
        for field in ("paired_t_p", "wilcoxon_p", "sign_randomization_p"):
            eligible = [row for row in family_rows if row[field] is not None]
            adjusted = _holm_adjust([float(row[field]) for row in eligible])
            for row, value in zip(eligible, adjusted, strict=True):
                row[f"{field}_holm"] = value


def aggregate_rows(
    run_rows: Sequence[Mapping[str, Any]], bootstrap_samples: int, bootstrap_seed: int
) -> list[dict[str, Any]]:
    """Aggregate seed-level metrics without mixing protocols or budgets."""

    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        key = (
            str(row["domain"]),
            str(row["task"]),
            str(row["protocol_id"]),
            str(row["variant"]),
            str(row["metric"]),
        )
        grouped[key].append(row)
    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        first = rows[0]
        values = [float(row["value"]) for row in rows]
        seed = bootstrap_seed + int(
            hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()[:8], 16
        )
        summaries.append(
            {
                "domain": key[0],
                "task": key[1],
                "protocol_id": key[2],
                "variant": key[3],
                "metric": key[4],
                "protocol_json": first["protocol_json"],
                "protocol_job_ids": json.dumps(
                    sorted({str(row["protocol_job_id"]) for row in rows})
                ),
                "budget_unit": first["budget_unit"],
                "budget_value": first["budget_value"],
                "eval_episodes": first["eval_episodes"],
                "eval_seed_base": first["eval_seed_base"],
                "batch_size": first["batch_size"],
                "total_timesteps": first["total_timesteps"],
                "training_updates": first["training_updates"],
                "transition_count": first["transition_count"],
                "cf_fraction": first["cf_fraction"],
                "intervention_scale": first["intervention_scale"],
                "seeds": json.dumps(sorted(int(row["seed"]) for row in rows)),
                **_summary_statistics(values, bootstrap_samples, seed),
            }
        )
    return summaries


def paired_delta_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pair each non-real result only with real from the same task and seed."""

    by_pair: dict[tuple[str, str, str, str, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in run_rows:
        key = (
            str(row["domain"]),
            str(row["task"]),
            str(row["protocol_id"]),
            str(row["metric"]),
            int(row["seed"]),
        )
        by_pair[key].append(row)

    paired: list[dict[str, Any]] = []
    for key, rows in sorted(by_pair.items()):
        real_rows = [row for row in rows if row["variant"] == "real"]
        variant_rows = [row for row in rows if row["variant"] != "real"]
        if not variant_rows or not real_rows:
            continue
        if len(real_rows) != 1:
            paths = [str(row["artifact_path"]) for row in real_rows]
            raise ValueError(
                "Ambiguous paired real run for domain/task/protocol/metric/seed "
                f"{key}: {paths}"
            )
        real = real_rows[0]
        variants: dict[str, Mapping[str, Any]] = {}
        for variant in variant_rows:
            name = str(variant["variant"])
            if name in variants:
                raise ValueError(
                    "Ambiguous paired variant run for "
                    f"domain/task/protocol/metric/seed/variant {key + (name,)}"
                )
            variants[name] = variant
        for name, variant in sorted(variants.items()):
            paired.append(
                {
                    "domain": key[0],
                    "task": key[1],
                    "protocol_id": key[2],
                    "metric": key[3],
                    "seed": key[4],
                    "variant": name,
                    "variant_value": variant["value"],
                    "real_value": real["value"],
                    "delta_vs_real": float(variant["value"]) - float(real["value"]),
                    "variant_protocol_job_id": variant["protocol_job_id"],
                    "real_protocol_job_id": real["protocol_job_id"],
                    "protocol_json": variant["protocol_json"],
                    "budget_unit": variant["budget_unit"],
                    "budget_value": variant["budget_value"],
                    "eval_episodes": variant["eval_episodes"],
                    "eval_seed_base": variant["eval_seed_base"],
                    "batch_size": variant["batch_size"],
                    "total_timesteps": variant["total_timesteps"],
                    "training_updates": variant["training_updates"],
                    "transition_count": variant["transition_count"],
                    "cf_fraction": variant["cf_fraction"],
                    "intervention_scale": variant["intervention_scale"],
                    "variant_artifact_path": variant["artifact_path"],
                    "real_artifact_path": real["artifact_path"],
                }
            )
    return paired


def paired_summary_rows(
    paired_rows: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Aggregate paired per-seed deltas for every non-real variant."""

    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        key = (
            str(row["domain"]),
            str(row["task"]),
            str(row["protocol_id"]),
            str(row["variant"]),
            str(row["metric"]),
        )
        grouped[key].append(row)
    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        first = rows[0]
        values = [float(row["delta_vs_real"]) for row in rows]
        seed = bootstrap_seed + int(
            hashlib.sha256(("paired|" + "|".join(key)).encode("utf-8")).hexdigest()[:8],
            16,
        )
        summaries.append(
            {
                "domain": key[0],
                "task": key[1],
                "protocol_id": key[2],
                "variant": key[3],
                "metric": key[4],
                "protocol_json": first["protocol_json"],
                "budget_unit": first["budget_unit"],
                "budget_value": first["budget_value"],
                "eval_episodes": first["eval_episodes"],
                "eval_seed_base": first["eval_seed_base"],
                "batch_size": first["batch_size"],
                "total_timesteps": first["total_timesteps"],
                "training_updates": first["training_updates"],
                "transition_count": first["transition_count"],
                "cf_fraction": first["cf_fraction"],
                "intervention_scale": first["intervention_scale"],
                "paired_seeds": json.dumps(sorted(int(row["seed"]) for row in rows)),
                **_summary_statistics(values, bootstrap_samples, seed),
                **_paired_inference(values, seed),
            }
        )
    _add_holm_adjustments(summaries)
    return summaries


def _write_rows(output_dir: Path, stem: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the same tidy rows as deterministic CSV and JSON files."""

    json_path = output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(list(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_path = output_dir / f"{stem}.csv"
    if not rows:
        csv_path.write_text("\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    mujoco_roots: Sequence[Path],
    d4rl_roots: Sequence[Path],
    output_dir: Path,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Scan artifacts, calculate summaries, and emit CSV/JSON output pairs."""

    if bootstrap_seed < 0:
        raise ValueError("bootstrap seed must be non-negative")
    runs = scan_artifacts(mujoco_roots, d4rl_roots)
    run_results = tidy_run_rows(runs)
    aggregate_results = aggregate_rows(run_results, bootstrap_samples, bootstrap_seed)
    paired_deltas = paired_delta_rows(run_results)
    paired_summary = paired_summary_rows(
        paired_deltas, bootstrap_samples, bootstrap_seed
    )
    outputs = {
        "run_results": run_results,
        "aggregate_results": aggregate_results,
        "paired_deltas": paired_deltas,
        "paired_summary": paired_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem in OUTPUT_STEMS:
        _write_rows(output_dir, stem, outputs[stem])
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit local artifact roots and bounded bootstrap settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco-root", type=Path, action="append", default=[])
    parser.add_argument("--d4rl-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.mujoco_root and not args.d4rl_root:
        parser.error("at least one --mujoco-root or --d4rl-root is required")
    return args


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    outputs = summarize(
        args.mujoco_root,
        args.d4rl_root,
        args.output_dir,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    print(
        f"[summary] runs={len(outputs['run_results'])}, "
        f"paired={len(outputs['paired_deltas'])}, output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
