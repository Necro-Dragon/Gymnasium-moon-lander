from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
STATE_SIZE = 7
CONTROL_SIZE = 7


@dataclass(slots=True)
class LanderState:
    r: float
    r_dot: float
    phi: float
    phi_dot: float
    theta: float
    theta_dot: float
    m: float

    def as_array(self) -> FloatArray:
        return np.array(
            [
                self.r,
                self.r_dot,
                self.phi,
                self.phi_dot,
                self.theta,
                self.theta_dot,
                self.m,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, values: FloatArray) -> "LanderState":
        if values.shape != (STATE_SIZE,):
            raise ValueError(f"lander state must have shape {(STATE_SIZE,)}, got {values.shape}")
        return cls(*[float(value) for value in values])


@dataclass(slots=True)
class TrackingState:
    altitude: float
    radial_velocity: float
    attitude_error: float
    attitude_rate_error: float
    along_track: float
    along_track_velocity: float
    mass_error: float

    def as_array(self) -> FloatArray:
        return np.array(
            [
                self.altitude,
                self.radial_velocity,
                self.attitude_error,
                self.attitude_rate_error,
                self.along_track,
                self.along_track_velocity,
                self.mass_error,
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "altitude": self.altitude,
            "radial_velocity": self.radial_velocity,
            "attitude_error": self.attitude_error,
            "attitude_rate_error": self.attitude_rate_error,
            "along_track": self.along_track,
            "along_track_velocity": self.along_track_velocity,
            "mass_error": self.mass_error,
        }


@dataclass(slots=True)
class MoonParams:
    radius: float
    mu: float
    omega: float
    g0: float = 9.80665


@dataclass(slots=True)
class VehicleParams:
    m0: float
    m_dry: float
    h_cm: float
    l_cm: float
    j_dry: float
    k_fuel: float
    u_rcs_max: float
    u_dp_max: float
    isp_rcs: float
    isp_dp: float

    def inertia(self, mass: float) -> float:
        clamped_mass = max(float(mass), self.m_dry)
        return self.j_dry + (clamped_mass - self.m_dry) * self.k_fuel**2

    def inertia_dot(self, m_dot: float) -> float:
        return self.k_fuel**2 * float(m_dot)

    def thrust_limits(self) -> FloatArray:
        return np.array(
            [self.u_rcs_max] * 6 + [self.u_dp_max],
            dtype=np.float64,
        )

    def clamp_control(self, control: FloatArray) -> FloatArray:
        clipped = np.clip(np.asarray(control, dtype=np.float64), 0.0, self.thrust_limits())
        if clipped.shape != (CONTROL_SIZE,):
            raise ValueError(f"control must have shape {(CONTROL_SIZE,)}, got {clipped.shape}")
        return clipped

    def wrench_matrix(self) -> FloatArray:
        return np.array(
            [
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                [
                    -self.l_cm,
                    self.h_cm,
                    self.l_cm,
                    self.l_cm,
                    -self.h_cm,
                    -self.l_cm,
                    0.0,
                ],
            ],
            dtype=np.float64,
        )

    def max_body_wrench(self) -> FloatArray:
        return np.array(
            [
                2.0 * self.u_rcs_max,
                4.0 * self.u_rcs_max + self.u_dp_max,
                2.0 * self.l_cm * self.u_rcs_max + self.h_cm * self.u_rcs_max,
            ],
            dtype=np.float64,
        )


@dataclass(slots=True)
class ScenarioConfig:
    name: str
    description: str
    initial_state: LanderState


@dataclass(slots=True)
class ReferenceTrajectory:
    moon: MoonParams
    vehicle: VehicleParams
    theta0: float = 0.0
    beta: float = 0.20

    @property
    def r_ref(self) -> float:
        return self.moon.radius + self.vehicle.h_cm

    @property
    def m_ref(self) -> float:
        return self.beta * self.vehicle.m0 + self.vehicle.m_dry

    def sample(self, t: float) -> LanderState:
        theta = self.theta0 + self.moon.omega * float(t)
        return LanderState(
            r=self.r_ref,
            r_dot=0.0,
            phi=theta,
            phi_dot=self.moon.omega,
            theta=theta,
            theta_dot=self.moon.omega,
            m=self.m_ref,
        )


@dataclass(slots=True)
class CostSpec:
    altitude_scale: float = 100.0
    radial_velocity_scale: float = 5.0
    attitude_scale: float = np.deg2rad(10.0)
    attitude_rate_scale: float = np.deg2rad(5.0)
    along_track_scale: float = 100.0
    along_track_velocity_scale: float = 2.0
    mass_scale: float = 250.0
    control_weight: float = 0.05
    terminal_weight: float = 10.0
    failure_penalties: dict[str, float] = field(
        default_factory=lambda: {
            "touchdown_hard": 350.0,
            "touchdown_off_pad": 500.0,
            "fuel_depletion": 400.0,
            "along_track_escape": 275.0,
            "altitude_escape": 250.0,
            "attitude_escape": 300.0,
            "timeout": 225.0,
        }
    )

    def penalty_for(self, terminal_status: str) -> float:
        return float(self.failure_penalties.get(terminal_status, 0.0))


@dataclass(slots=True)
class ScoreReport:
    controller_name: str
    scenario_name: str
    success: bool
    terminal_status: str
    total_cost: float
    integrated_cost: float
    terminal_cost: float
    failure_penalty: float
    fuel_used: float
    final_mass: float
    max_peak_thrust_fraction: float
    max_constraint_violation: float
    touchdown_errors: dict[str, float]
    final_tracking_state: TrackingState

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_name": self.controller_name,
            "scenario_name": self.scenario_name,
            "success": self.success,
            "terminal_status": self.terminal_status,
            "total_cost": self.total_cost,
            "integrated_cost": self.integrated_cost,
            "terminal_cost": self.terminal_cost,
            "failure_penalty": self.failure_penalty,
            "fuel_used": self.fuel_used,
            "final_mass": self.final_mass,
            "max_peak_thrust_fraction": self.max_peak_thrust_fraction,
            "max_constraint_violation": self.max_constraint_violation,
            "touchdown_errors": self.touchdown_errors,
            "final_tracking_state": self.final_tracking_state.to_dict(),
        }


@dataclass(slots=True)
class RolloutResult:
    controller_name: str
    scenario_name: str
    times: FloatArray
    states: FloatArray
    controls: FloatArray
    control_times: FloatArray
    tracking: FloatArray
    reference_states: FloatArray
    surface_heights: FloatArray
    stage_costs: FloatArray
    cumulative_costs: FloatArray
    score_report: ScoreReport


class ControllerProtocol(Protocol):
    name: str

    def reset(self, scenario: ScenarioConfig, reference: ReferenceTrajectory) -> None:
        ...

    def action(self, t: float, state: LanderState) -> FloatArray:
        ...
