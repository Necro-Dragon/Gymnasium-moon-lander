from __future__ import annotations

import argparse
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
from .utils import slugify_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paper-aligned lunar descent rollout and export metrics, plots, and a GIF."
    )
    parser.add_argument(
        "--controller",
        default="tvlqr_tracking",
        help=(
            "Controller spec. Use one of "
            f"{', '.join(builtin_controller_names())}, a Python factory path like pkg.module:factory, "
            "or an open-loop .npz bundle with t and u arrays."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="nominal",
        choices=sorted(default_scenarios(default_reference_trajectory()).keys()),
        help="Named scenario to simulate.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory for artifact output bundles.",
    )
    parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="Skip GIF generation and only save metrics plus static plots.",
    )
    parser.add_argument(
        "--t-final",
        type=float,
        default=5_000.0,
        help="Maximum simulation time horizon in seconds. Default: 5000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = default_reference_trajectory()
    terrain = default_terrain_model()
    cost = default_cost_spec()
    simulation = SimulationConfig(t_final=args.t_final)
    scenarios = default_scenarios(reference)
    scenario = scenarios[args.scenario]
    controller_label, factory = load_controller_factory(
        args.controller,
        moon=reference.moon,
        vehicle=reference.vehicle,
        cost=cost,
        simulation=simulation,
    )
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
    print(f"controller={report.controller_name}")
    print(f"scenario={report.scenario_name}")
    print(f"success={report.success}")
    print(f"terminal_status={report.terminal_status}")
    print(f"total_cost={report.total_cost:.3f}")
    print(f"fuel_used={report.fuel_used:.3f}")
    print(f"artifact_dir={output_dir}")


if __name__ == "__main__":
    main()
