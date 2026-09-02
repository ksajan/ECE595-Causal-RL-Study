from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.workshop.plot_continuous_control_results import (
    EXPECTED_CELLS,
    EXPECTED_MATCHED_CONTROLS,
    plot_matched_control_effects,
    plot_paired_effects,
    validate_matched_control_cells,
    validate_publication_cells,
)


def _rows(n: int = 10) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for domain, spec in EXPECTED_CELLS.items():
        for task in spec["tasks"]:
            for index, variant in enumerate(spec["variants"]):
                mean = float(index + 1)
                rows.append(
                    {
                        "domain": domain,
                        "task": task,
                        "variant": variant,
                        "metric": str(spec["metric"]),
                        "n": str(n),
                        "paired_seeds": json.dumps(list(range(n))),
                        "mean": str(mean),
                        "bootstrap_ci95_low": str(mean - 0.5),
                        "bootstrap_ci95_high": str(mean + 0.5),
                    }
                )
    return rows


def _matched_rows(n: int = 10) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for domain, spec in EXPECTED_MATCHED_CONTROLS.items():
        for index, task in enumerate(spec["tasks"]):
            mean = float(index + 1)
            rows.append(
                {
                    "domain": domain,
                    "task": task,
                    "contrast": str(spec["contrast"]),
                    "metric": str(spec["metric"]),
                    "n": str(n),
                    "paired_seeds": json.dumps(list(range(n))),
                    "mean": str(mean),
                    "bootstrap_ci95_low": str(mean - 0.5),
                    "bootstrap_ci95_high": str(mean + 0.5),
                }
            )
    return rows


def test_gate_accepts_complete_ten_seed_matrix() -> None:
    validated = validate_publication_cells(_rows(), min_seeds=10)
    assert len(validated["mujoco_sac"]) == 8
    assert len(validated["d4rl_cql"]) == 6


def test_gate_rejects_missing_and_underpowered_cells() -> None:
    rows = _rows(n=9)
    rows.pop()
    with pytest.raises(RuntimeError, match=r"requires n>=10"):
        validate_publication_cells(rows, min_seeds=10)


def test_gate_rejects_substituted_seed_identity() -> None:
    rows = _rows()
    rows[0]["paired_seeds"] = json.dumps(list(range(1, 11)))
    with pytest.raises(RuntimeError, match="missing required seeds 0--9"):
        validate_publication_cells(rows, min_seeds=10)


def test_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    validated = validate_publication_cells(_rows(), min_seeds=10)
    png_path, pdf_path = plot_paired_effects(validated, tmp_path)
    assert png_path.stat().st_size > 1_000
    assert pdf_path.stat().st_size > 1_000


def test_matched_control_gate_and_plot_require_complete_seed_matrix(
    tmp_path: Path,
) -> None:
    validated = validate_matched_control_cells(_matched_rows(), min_seeds=10)
    assert len(validated["mujoco_sac"]) == 4
    assert len(validated["d4rl_cql"]) == 2
    png_path, pdf_path = plot_matched_control_effects(validated, tmp_path)
    assert png_path.stat().st_size > 1_000
    assert pdf_path.stat().st_size > 1_000

    with pytest.raises(RuntimeError, match=r"requires n>=10"):
        validate_matched_control_cells(_matched_rows(n=9), min_seeds=10)


def test_current_csv_shape_can_be_read_by_dict_reader(tmp_path: Path) -> None:
    path = tmp_path / "paired_summary.csv"
    rows = _rows()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding="utf-8", newline="") as handle:
        loaded = list(csv.DictReader(handle))
    assert validate_publication_cells(loaded, 10)
