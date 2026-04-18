from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .allocation import allocate_body_wrench
from .dynamics import state_derivative_array
from .models import CostSpec, FloatArray, LanderState, MoonParams, ReferenceTrajectory, ScenarioConfig, VehicleParams
from .simulation import SimulationConfig
from .tracking import polar_force_to_body, state_from_tracking_vector, tracking_derivative, tracking_state


def _clip_wrench(wrench: FloatArray, vehicle: VehicleParams) -> FloatArray:
    limits = vehicle.max_body_wrench()
    return np.array(
        [
            np.clip(wrench[0], -limits[0], limits[0]),
            np.clip(wrench[1], 0.0, limits[1]),
            np.clip(wrench[2], -limits[2], limits[2]),
        ],
        dtype=np.float64,
    )


def _guidance_wrench(
    state: LanderState,
    track: FloatArray,
    moon: MoonParams,
    vehicle: VehicleParams,
) -> FloatArray:
    hover_force = state.m * (moon.mu / state.r**2 - state.r * moon.omega**2)
    force_r = hover_force - 16.0 * track[0] - 1_550.0 * track[1]
    force_theta = -7.5 * track[4] - 2_100.0 * track[5]
    force_body = polar_force_to_body(force_r, force_theta, track[2])
    moment = -9_500.0 * track[2] - 6_500.0 * track[3]
    return _clip_wrench(
        np.array([force_body[0], force_body[1], moment], dtype=np.float64),
        vehicle,
    )


@dataclass(slots=True)
class ScriptedHoverDescentController:
    moon: MoonParams
    vehicle: VehicleParams
    name: str = "scripted_hover_descent"
    reference: ReferenceTrajectory | None = field(init=False, default=None)

    def reset(self, scenario: ScenarioConfig, reference: ReferenceTrajectory) -> None:
        del scenario
        self.reference = reference

    def action(self, t: float, state: LanderState) -> FloatArray:
        if self.reference is None:
            raise RuntimeError("controller must be reset before use")

        track = tracking_state(state, self.reference, t).as_array()
        wrench = _guidance_wrench(state, track, self.moon, self.vehicle)
        return allocate_body_wrench(*wrench, vehicle=self.vehicle)


@dataclass(slots=True)
class TVLQRTrackingController:
    moon: MoonParams
    vehicle: VehicleParams
    cost: CostSpec
    simulation: SimulationConfig
    name: str = "tvlqr_tracking"
    reference: ReferenceTrajectory | None = field(init=False, default=None)
    gains: list[FloatArray] = field(init=False, default_factory=list)
    step_index: int = field(init=False, default=0)

    def reset(self, scenario: ScenarioConfig, reference: ReferenceTrajectory) -> None:
        del scenario
        self.reference = reference
        self.step_index = 0
        self.gains = self._build_gain_schedule()

    def _nominal_wrench(self, state: LanderState) -> FloatArray:
        hover_force = state.m * (self.moon.mu / state.r**2 - state.r * self.moon.omega**2)
        return np.array([0.0, max(0.0, hover_force), 0.0], dtype=np.float64)

    def _reduced_error_dynamics(self, error_state: FloatArray, delta_wrench: FloatArray) -> FloatArray:
        if self.reference is None:
            raise RuntimeError("controller must be reset before use")

        state = state_from_tracking_vector(self.reference, 0.0, np.asarray(error_state, dtype=np.float64))
        nominal = self._nominal_wrench(self.reference.sample(0.0))
        wrench = _clip_wrench(nominal + delta_wrench, self.vehicle)
        control = allocate_body_wrench(*wrench, vehicle=self.vehicle)
        state_dot = state_derivative_array(0.0, state.as_array(), control, self.moon, self.vehicle)
        return tracking_derivative(state, state_dot, self.reference, 0.0).as_array()

    def _linearize(self) -> tuple[FloatArray, FloatArray]:
        state_eps = np.array(
            [
                0.5,
                0.05,
                np.deg2rad(0.05),
                np.deg2rad(0.05),
                0.5,
                0.05,
                1.0,
            ],
            dtype=np.float64,
        )
        control_eps = np.array([10.0, 10.0, 5.0], dtype=np.float64)
        a_matrix = np.zeros((7, 7), dtype=np.float64)
        b_matrix = np.zeros((7, 3), dtype=np.float64)
        zero_state = np.zeros(7, dtype=np.float64)
        zero_wrench = np.zeros(3, dtype=np.float64)

        for index in range(7):
            perturb = np.zeros(7, dtype=np.float64)
            perturb[index] = state_eps[index]
            forward = self._reduced_error_dynamics(zero_state + perturb, zero_wrench)
            backward = self._reduced_error_dynamics(zero_state - perturb, zero_wrench)
            a_matrix[:, index] = (forward - backward) / (2.0 * state_eps[index])

        for index in range(3):
            perturb = np.zeros(3, dtype=np.float64)
            perturb[index] = control_eps[index]
            forward = self._reduced_error_dynamics(zero_state, zero_wrench + perturb)
            backward = self._reduced_error_dynamics(zero_state, zero_wrench - perturb)
            b_matrix[:, index] = (forward - backward) / (2.0 * control_eps[index])

        return a_matrix, b_matrix

    def _build_gain_schedule(self) -> list[FloatArray]:
        a_matrix, b_matrix = self._linearize()
        ad = np.eye(7, dtype=np.float64) + a_matrix * self.simulation.control_dt
        bd = b_matrix * self.simulation.control_dt

        q = np.diag(
            np.array(
                [
                    1.0 / self.cost.altitude_scale**2,
                    1.0 / self.cost.radial_velocity_scale**2,
                    1.0 / self.cost.attitude_scale**2,
                    1.0 / self.cost.attitude_rate_scale**2,
                    1.0 / self.cost.along_track_scale**2,
                    1.0 / self.cost.along_track_velocity_scale**2,
                    1.0 / self.cost.mass_scale**2,
                ],
                dtype=np.float64,
            )
        )
        wrench_limits = self.vehicle.max_body_wrench()
        r = self.cost.control_weight * np.diag(1.0 / np.maximum(wrench_limits**2, 1.0))
        q_final = self.cost.terminal_weight * q

        horizon_steps = int(np.ceil(self.simulation.t_final / self.simulation.control_dt))
        gains = [np.zeros((3, 7), dtype=np.float64) for _ in range(horizon_steps)]
        p_matrix = q_final.copy()
        for index in range(horizon_steps - 1, -1, -1):
            gram = r + bd.T @ p_matrix @ bd
            gain = np.linalg.solve(gram, bd.T @ p_matrix @ ad)
            gains[index] = gain
            p_matrix = q + ad.T @ p_matrix @ (ad - bd @ gain)
            p_matrix = 0.5 * (p_matrix + p_matrix.T)
        return gains

    def action(self, t: float, state: LanderState) -> FloatArray:
        if self.reference is None or not self.gains:
            raise RuntimeError("controller must be reset before use")

        track = tracking_state(state, self.reference, t).as_array()
        gain = self.gains[min(self.step_index, len(self.gains) - 1)]
        delta_wrench = -gain @ track
        nominal = _guidance_wrench(state, track, self.moon, self.vehicle)
        wrench = _clip_wrench(nominal + 0.10 * delta_wrench, self.vehicle)
        self.step_index += 1
        return allocate_body_wrench(*wrench, vehicle=self.vehicle)


@dataclass(slots=True)
class OpenLoopController:
    control_times: FloatArray
    controls: FloatArray
    name: str

    def reset(self, scenario: ScenarioConfig, reference: ReferenceTrajectory) -> None:
        del scenario, reference

    def action(self, t: float, state: LanderState) -> FloatArray:
        del state
        index = int(np.searchsorted(self.control_times, t, side="right") - 1)
        index = min(max(index, 0), self.controls.shape[0] - 1)
        return np.asarray(self.controls[index], dtype=np.float64)
