from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FloatArray, LanderState, MoonParams, VehicleParams


@dataclass(slots=True)
class ForceBreakdown:
    fx_body: float
    fy_body: float
    moment_body: float
    q_r: float
    q_theta: float
    q_phi: float
    mass_flow_rate: float


def control_to_body_wrench(control: FloatArray, vehicle: VehicleParams) -> tuple[float, float, float]:
    u = vehicle.clamp_control(control)
    fx_body = float(u[1] - u[4])
    fy_body = float(u[0] + u[2] + u[3] + u[5] + u[6])
    moment_body = float(
        (u[3] + u[2] - u[0] - u[5]) * vehicle.l_cm + (u[1] - u[4]) * vehicle.h_cm
    )
    return fx_body, fy_body, moment_body


def mass_flow_rate(control: FloatArray, state: LanderState, moon: MoonParams, vehicle: VehicleParams) -> float:
    if state.m <= vehicle.m_dry + 1e-9:
        return 0.0
    u = vehicle.clamp_control(control)
    rcs_flow = float(np.sum(u[:6])) / (vehicle.isp_rcs * moon.g0)
    dp_flow = float(u[6]) / (vehicle.isp_dp * moon.g0)
    return -(rcs_flow + dp_flow)


def force_breakdown(
    state: LanderState,
    control: FloatArray,
    moon: MoonParams,
    vehicle: VehicleParams,
) -> ForceBreakdown:
    fx_body, fy_body, moment_body = control_to_body_wrench(control, vehicle)
    alpha = state.phi - state.theta
    sine = np.sin(alpha)
    cosine = np.cos(alpha)
    q_r = -fx_body * sine + fy_body * cosine
    q_theta = state.r * (fx_body * cosine + fy_body * sine)
    q_phi = moment_body
    return ForceBreakdown(
        fx_body=fx_body,
        fy_body=fy_body,
        moment_body=moment_body,
        q_r=float(q_r),
        q_theta=float(q_theta),
        q_phi=float(q_phi),
        mass_flow_rate=mass_flow_rate(control, state, moon, vehicle),
    )


def state_derivative_array(
    t: float,
    state_values: FloatArray,
    control: FloatArray,
    moon: MoonParams,
    vehicle: VehicleParams,
) -> FloatArray:
    del t
    state = LanderState.from_array(np.asarray(state_values, dtype=np.float64))
    if state.m <= vehicle.m_dry + 1e-9:
        control = np.zeros(7, dtype=np.float64)

    forces = force_breakdown(state, control, moon, vehicle)
    mass = max(state.m, vehicle.m_dry)
    inertia = vehicle.inertia(mass)
    inertia_dot = vehicle.inertia_dot(forces.mass_flow_rate)

    r_dot = state.r_dot
    r_ddot = (
        state.r * state.theta_dot**2
        - moon.mu / state.r**2
        + (forces.q_r - forces.mass_flow_rate * state.r_dot) / mass
    )
    phi_dot = state.phi_dot
    phi_ddot = (forces.q_phi - inertia_dot * state.phi_dot) / inertia
    theta_dot = state.theta_dot
    theta_ddot = (
        forces.q_theta
        - 2.0 * mass * state.r * state.r_dot * state.theta_dot
        - forces.mass_flow_rate * state.r**2 * state.theta_dot
    ) / (mass * state.r**2)
    m_dot = forces.mass_flow_rate
    if state.m <= vehicle.m_dry + 1e-9 and m_dot < 0.0:
        m_dot = 0.0

    return np.array(
        [r_dot, r_ddot, phi_dot, phi_ddot, theta_dot, theta_ddot, m_dot],
        dtype=np.float64,
    )


def step_rk4(
    t: float,
    state: LanderState,
    control: FloatArray,
    sim_dt: float,
    moon: MoonParams,
    vehicle: VehicleParams,
) -> LanderState:
    x0 = state.as_array()
    k1 = state_derivative_array(t, x0, control, moon, vehicle)
    k2 = state_derivative_array(t + 0.5 * sim_dt, x0 + 0.5 * sim_dt * k1, control, moon, vehicle)
    k3 = state_derivative_array(t + 0.5 * sim_dt, x0 + 0.5 * sim_dt * k2, control, moon, vehicle)
    k4 = state_derivative_array(t + sim_dt, x0 + sim_dt * k3, control, moon, vehicle)
    next_state = x0 + (sim_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state[6] = max(next_state[6], vehicle.m_dry)
    return LanderState.from_array(next_state)
