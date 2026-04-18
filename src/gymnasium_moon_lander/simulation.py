from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import step_rk4
from .models import (
    ControllerProtocol,
    CostSpec,
    LanderState,
    MoonParams,
    ReferenceTrajectory,
    RolloutResult,
    ScenarioConfig,
    ScoreReport,
    TrackingState,
    VehicleParams,
)
from .scoring import stage_cost, terminal_cost
from .terrain import TerrainModel
from .tracking import tracking_state
from .utils import wrap_angle


@dataclass(slots=True)
class SimulationConfig:
    sim_dt: float = 0.02
    control_dt: float = 0.10
    t_final: float = 5_000.0


@dataclass(slots=True)
class TerminationStatus:
    success: bool
    terminal_status: str
    touchdown_errors: dict[str, float]
    max_constraint_violation: float


def touchdown_error_dict(tracking: TrackingState, state: LanderState) -> dict[str, float]:
    return {
        "radial_velocity": abs(tracking.radial_velocity),
        "along_track_velocity": abs(tracking.along_track_velocity),
        "attitude_error_deg": abs(np.rad2deg(tracking.attitude_error)),
        "attitude_rate_error_deg_s": abs(np.rad2deg(tracking.attitude_rate_error)),
        "remaining_mass_margin": state.m,
    }


def _constraint_violation(
    tracking: TrackingState,
    altitude_above_surface: float,
    state: LanderState,
    vehicle: VehicleParams,
) -> float:
    return max(
        0.0,
        abs(tracking.along_track) / 2_000.0 - 1.0,
        altitude_above_surface / 3_000.0 - 1.0,
        abs(tracking.attitude_error) / np.deg2rad(90.0) - 1.0,
        max(vehicle.m_dry - state.m, 0.0) / max(vehicle.m_dry, 1.0),
    )


def check_termination(
    state: LanderState,
    reference: ReferenceTrajectory,
    terrain: TerrainModel,
    t: float,
    vehicle: VehicleParams,
) -> TerminationStatus | None:
    tracking = tracking_state(state, reference, t)
    local_x = terrain.local_x(state.theta, reference, t)
    surface_height = terrain.surface_height_local(local_x)
    contact_radius = reference.moon.radius + surface_height + vehicle.h_cm
    altitude_above_surface = state.r - (reference.moon.radius + surface_height + vehicle.h_cm)
    violation = _constraint_violation(tracking, altitude_above_surface, state, vehicle)

    if state.r <= contact_radius:
        touchdown_errors = touchdown_error_dict(tracking, state)
        safe_touchdown = (
            terrain.is_on_pad_local(local_x)
            and abs(tracking.radial_velocity) <= 2.0
            and abs(tracking.along_track_velocity) <= 1.5
            and abs(tracking.attitude_error) <= np.deg2rad(5.0)
            and abs(tracking.attitude_rate_error) <= np.deg2rad(5.0)
            and state.m > vehicle.m_dry
        )
        if safe_touchdown:
            return TerminationStatus(
                success=True,
                terminal_status="touchdown_success",
                touchdown_errors=touchdown_errors,
                max_constraint_violation=violation,
            )
        return TerminationStatus(
            success=False,
            terminal_status="touchdown_hard" if terrain.is_on_pad_local(local_x) else "touchdown_off_pad",
            touchdown_errors=touchdown_errors,
            max_constraint_violation=max(
                violation,
                abs(tracking.radial_velocity) / 2.0 - 1.0,
                abs(tracking.along_track_velocity) / 1.5 - 1.0,
                abs(tracking.attitude_error) / np.deg2rad(5.0) - 1.0,
                abs(tracking.attitude_rate_error) / np.deg2rad(5.0) - 1.0,
            ),
        )

    if state.m <= vehicle.m_dry + 1e-9:
        return TerminationStatus(
            success=False,
            terminal_status="fuel_depletion",
            touchdown_errors={},
            max_constraint_violation=max(violation, (vehicle.m_dry - state.m) / max(vehicle.m_dry, 1.0)),
        )
    if abs(tracking.along_track) > 2_000.0:
        return TerminationStatus(
            success=False,
            terminal_status="along_track_escape",
            touchdown_errors={},
            max_constraint_violation=max(violation, abs(tracking.along_track) / 2_000.0 - 1.0),
        )
    if altitude_above_surface > 3_000.0:
        return TerminationStatus(
            success=False,
            terminal_status="altitude_escape",
            touchdown_errors={},
            max_constraint_violation=max(violation, altitude_above_surface / 3_000.0 - 1.0),
        )
    if abs(wrap_angle(state.phi - state.theta)) > np.deg2rad(90.0):
        return TerminationStatus(
            success=False,
            terminal_status="attitude_escape",
            touchdown_errors={},
            max_constraint_violation=max(
                violation,
                abs(wrap_angle(state.phi - state.theta)) / np.deg2rad(90.0) - 1.0,
            ),
        )
    return None


def _prepare_control(
    controller: ControllerProtocol,
    t: float,
    state: LanderState,
    vehicle: VehicleParams,
) -> np.ndarray:
    if state.m <= vehicle.m_dry + 1e-9:
        return np.zeros(7, dtype=np.float64)
    requested = np.asarray(controller.action(t, state), dtype=np.float64)
    return vehicle.clamp_control(requested)


def simulate_rollout(
    controller: ControllerProtocol,
    scenario: ScenarioConfig,
    reference: ReferenceTrajectory,
    terrain: TerrainModel,
    cost: CostSpec,
    config: SimulationConfig,
) -> RolloutResult:
    controller.reset(scenario, reference)
    moon = reference.moon
    vehicle = reference.vehicle
    state = scenario.initial_state
    t = 0.0
    next_control_time = 0.0
    control = np.zeros(7, dtype=np.float64)

    times = [t]
    states = [state.as_array()]
    tracking_history = [tracking_state(state, reference, t).as_array()]
    reference_history = [reference.sample(t).as_array()]
    surface_history = [terrain.height_at(state.theta, reference, t)]
    control_history: list[np.ndarray] = []
    control_times: list[float] = []
    stage_costs: list[float] = []
    cumulative_costs = [0.0]
    integrated_cost = 0.0
    max_peak_thrust_fraction = 0.0
    max_constraint_violation = 0.0
    terminal = None

    total_steps = int(np.ceil(config.t_final / config.sim_dt))
    for _ in range(total_steps):
        if t >= next_control_time - 1e-12:
            control = _prepare_control(controller, t, state, vehicle)
            next_control_time += config.control_dt

        track = tracking_state(state, reference, t)
        instantaneous_cost = stage_cost(track, control, cost, vehicle)
        integrated_cost += instantaneous_cost * config.sim_dt
        stage_costs.append(instantaneous_cost)
        cumulative_costs.append(integrated_cost)
        control_history.append(control.copy())
        control_times.append(t)
        max_peak_thrust_fraction = max(
            max_peak_thrust_fraction,
            float(np.max(control / vehicle.thrust_limits())),
        )

        next_state = step_rk4(
            t=t,
            state=state,
            control=control,
            sim_dt=config.sim_dt,
            moon=moon,
            vehicle=vehicle,
        )
        t = min(t + config.sim_dt, config.t_final)
        next_tracking = tracking_state(next_state, reference, t)
        next_surface = terrain.height_at(next_state.theta, reference, t)
        terminal = check_termination(next_state, reference, terrain, t, vehicle)
        if terminal is not None:
            max_constraint_violation = max(max_constraint_violation, terminal.max_constraint_violation)
        else:
            altitude_above_surface = next_state.r - (moon.radius + next_surface + vehicle.h_cm)
            max_constraint_violation = max(
                max_constraint_violation,
                _constraint_violation(next_tracking, altitude_above_surface, next_state, vehicle),
            )

        state = next_state
        times.append(t)
        states.append(state.as_array())
        tracking_history.append(next_tracking.as_array())
        reference_history.append(reference.sample(t).as_array())
        surface_history.append(next_surface)

        if terminal is not None:
            break

    if terminal is None:
        terminal = TerminationStatus(
            success=False,
            terminal_status="timeout",
            touchdown_errors={},
            max_constraint_violation=max_constraint_violation,
        )

    final_tracking = tracking_state(state, reference, times[-1])
    terminal_cost_value = terminal_cost(final_tracking, cost)
    failure_penalty = 0.0 if terminal.success else cost.penalty_for(terminal.terminal_status)
    total_cost = integrated_cost + terminal_cost_value + failure_penalty
    report = ScoreReport(
        controller_name=controller.name,
        scenario_name=scenario.name,
        success=terminal.success,
        terminal_status=terminal.terminal_status,
        total_cost=float(total_cost),
        integrated_cost=float(integrated_cost),
        terminal_cost=float(terminal_cost_value),
        failure_penalty=float(failure_penalty),
        fuel_used=float(scenario.initial_state.m - state.m),
        final_mass=float(state.m),
        max_peak_thrust_fraction=float(max_peak_thrust_fraction),
        max_constraint_violation=float(max(max_constraint_violation, terminal.max_constraint_violation)),
        touchdown_errors=terminal.touchdown_errors,
        final_tracking_state=final_tracking,
    )

    return RolloutResult(
        controller_name=controller.name,
        scenario_name=scenario.name,
        times=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        controls=np.asarray(control_history, dtype=np.float64),
        control_times=np.asarray(control_times, dtype=np.float64),
        tracking=np.asarray(tracking_history, dtype=np.float64),
        reference_states=np.asarray(reference_history, dtype=np.float64),
        surface_heights=np.asarray(surface_history, dtype=np.float64),
        stage_costs=np.asarray(stage_costs, dtype=np.float64),
        cumulative_costs=np.asarray(cumulative_costs, dtype=np.float64),
        score_report=report,
    )
