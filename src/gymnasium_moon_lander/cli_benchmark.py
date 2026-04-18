from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .artifacts import save_rollout_artifacts
from .loader import builtin_controller_names, load_controller_factory
from .scenarios import (
    default_cost_spec,
    default_reference_trajectory,
    default_scenarios,
    default_terrain_model,
)
from .simulation import SimulationConfig, simulate_rollout
from .utils import ensure_directory, slugify_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one or more controllers across the paper-aligned lunar descent scenarios."
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=list(builtin_controller_names()),
        help="Controller specs to benchmark.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=sorted(default_scenarios(default_reference_trajectory()).keys()),
        help="Scenario names to benchmark.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory for benchmark outputs.",
    )
    parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="Skip GIF generation for benchmark runs.",
    )
    parser.add_argument(
        "--t-final",
        type=float,
        default=5_000.0,
        help="Maximum simulation time horizon in seconds for each rollout. Default: 5000.",
    )
    return parser.parse_args()


def _write_benchmark_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "controller",
        "scenario",
        "success",
        "terminal_status",
        "total_cost",
        "integrated_cost",
        "terminal_cost",
        "failure_penalty",
        "fuel_used",
        "final_mass",
        "max_peak_thrust_fraction",
        "max_constraint_violation",
        "artifact_dir",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_benchmark_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        "# Benchmark Results",
        "",
        "| Controller | Scenario | Success | Status | Total cost | Fuel used | Artifact directory |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['controller']} | "
            f"{row['scenario']} | "
            f"{row['success']} | "
            f"{row['terminal_status']} | "
            f"{float(row['total_cost']):.3f} | "
            f"{float(row['fuel_used']):.3f} | "
            f"{row['artifact_dir']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    reference = default_reference_trajectory()
    terrain = default_terrain_model()
    cost = default_cost_spec()
    simulation = SimulationConfig(t_final=args.t_final)
    scenarios = default_scenarios(reference)
    ensure_directory(args.output_root)

    rows: list[dict[str, object]] = []
    for controller_spec in args.controllers:
        controller_label, factory = load_controller_factory(
            controller_spec,
            moon=reference.moon,
            vehicle=reference.vehicle,
            cost=cost,
            simulation=simulation,
        )
        for scenario_name in args.scenarios:
            scenario = scenarios[scenario_name]
            controller = factory()
            result = simulate_rollout(
                controller=controller,
                scenario=scenario,
                reference=reference,
                terrain=terrain,
                cost=cost,
                config=simulation,
            )
            output_dir = args.output_root / slugify_name(controller_label) / scenario.name
            save_rollout_artifacts(
                result=result,
                terrain=terrain,
                vehicle=reference.vehicle,
                output_dir=output_dir,
                include_gif=not args.skip_gif,
            )
            report = result.score_report
            row = {
                "controller": report.controller_name,
                "scenario": report.scenario_name,
                "success": report.success,
                "terminal_status": report.terminal_status,
                "total_cost": report.total_cost,
                "integrated_cost": report.integrated_cost,
                "terminal_cost": report.terminal_cost,
                "failure_penalty": report.failure_penalty,
                "fuel_used": report.fuel_used,
                "final_mass": report.final_mass,
                "max_peak_thrust_fraction": report.max_peak_thrust_fraction,
                "max_constraint_violation": report.max_constraint_violation,
                "artifact_dir": str(output_dir),
            }
            rows.append(row)
            print(
                f"controller={report.controller_name} scenario={report.scenario_name} "
                f"success={report.success} total_cost={report.total_cost:.3f}"
            )

    _write_benchmark_csv(rows, args.output_root / "benchmark.csv")
    _write_benchmark_markdown(rows, args.output_root / "benchmark.md")
    print(f"benchmark_csv={args.output_root / 'benchmark.csv'}")
    print(f"benchmark_md={args.output_root / 'benchmark.md'}")


if __name__ == "__main__":
    main()
