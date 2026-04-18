from __future__ import annotations

import csv
import sys
from pathlib import Path

from gymnasium_moon_lander.cli_benchmark import main as benchmark_main
from gymnasium_moon_lander.cli_simulate import main as simulate_main


def test_simulate_descent_cli_produces_full_artifact_bundle(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simulate_descent",
            "--controller",
            "tvlqr_tracking",
            "--scenario",
            "nominal",
            "--t-final",
            "5",
            "--output-root",
            str(tmp_path),
        ],
    )
    simulate_main()
    artifact_dir = tmp_path / "tvlqr_tracking" / "nominal"
    expected = {
        "metrics.json",
        "trajectory_local.png",
        "state_history.png",
        "control_history.png",
        "cost_history.png",
        "rollout.gif",
    }
    assert expected == {path.name for path in artifact_dir.iterdir() if path.is_file()}


def test_benchmark_cli_writes_reports_with_finite_scores(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_controllers",
            "--controllers",
            "scripted_hover_descent",
            "--t-final",
            "5",
            "--output-root",
            str(tmp_path),
            "--skip-gif",
        ],
    )
    benchmark_main()
    csv_path = tmp_path / "benchmark.csv"
    md_path = tmp_path / "benchmark.md"
    assert csv_path.exists()
    assert md_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(float(row["total_cost"]) == float(row["total_cost"]) for row in rows)
