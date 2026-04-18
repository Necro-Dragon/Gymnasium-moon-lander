from __future__ import annotations

import numpy as np

from .models import LanderState, ReferenceTrajectory, TrackingState
from .utils import wrap_angle


def tracking_state(state: LanderState, reference: ReferenceTrajectory, t: float) -> TrackingState:
    target = reference.sample(t)
    along_track = reference.r_ref * wrap_angle(state.theta - target.theta)
    along_track_velocity = state.r * state.theta_dot - target.r * target.theta_dot
    attitude_error = wrap_angle((state.phi - state.theta) - (target.phi - target.theta))
    attitude_rate_error = (state.phi_dot - state.theta_dot) - (target.phi_dot - target.theta_dot)
    return TrackingState(
        altitude=state.r - target.r,
        radial_velocity=state.r_dot - target.r_dot,
        attitude_error=attitude_error,
        attitude_rate_error=attitude_rate_error,
        along_track=along_track,
        along_track_velocity=along_track_velocity,
        mass_error=state.m - target.m,
    )


def tracking_state_array(state: LanderState, reference: ReferenceTrajectory, t: float) -> np.ndarray:
    return tracking_state(state, reference, t).as_array()


def state_from_tracking_offsets(
    reference: ReferenceTrajectory,
    t: float,
    altitude: float,
    radial_velocity: float,
    attitude_error: float,
    attitude_rate_error: float,
    along_track: float,
    along_track_velocity: float,
    mass: float,
) -> LanderState:
    target = reference.sample(t)
    theta = target.theta + along_track / reference.r_ref
    theta_dot = target.theta_dot + along_track_velocity / reference.r_ref
    phi = theta + attitude_error
    phi_dot = theta_dot + attitude_rate_error
    return LanderState(
        r=target.r + altitude,
        r_dot=target.r_dot + radial_velocity,
        phi=phi,
        phi_dot=phi_dot,
        theta=theta,
        theta_dot=theta_dot,
        m=float(mass),
    )


def state_from_tracking_vector(
    reference: ReferenceTrajectory,
    t: float,
    values: np.ndarray,
) -> LanderState:
    return state_from_tracking_offsets(
        reference=reference,
        t=t,
        altitude=float(values[0]),
        radial_velocity=float(values[1]),
        attitude_error=float(values[2]),
        attitude_rate_error=float(values[3]),
        along_track=float(values[4]),
        along_track_velocity=float(values[5]),
        mass=float(reference.sample(t).m + values[6]),
    )


def tracking_derivative(
    state: LanderState,
    state_dot: np.ndarray,
    reference: ReferenceTrajectory,
    t: float,
) -> TrackingState:
    target = reference.sample(t)
    altitude_dot = float(state_dot[0] - target.r_dot)
    radial_velocity_dot = float(state_dot[1])
    attitude_dot = float(state_dot[2] - state_dot[4])
    attitude_rate_dot = float(state_dot[3] - state_dot[5])
    along_track_dot = reference.r_ref * float(state_dot[4] - target.theta_dot)
    along_track_velocity_dot = float(state.r_dot * state.theta_dot + state.r * state_dot[5])
    mass_dot = float(state_dot[6])
    return TrackingState(
        altitude=altitude_dot,
        radial_velocity=radial_velocity_dot,
        attitude_error=attitude_dot,
        attitude_rate_error=attitude_rate_dot,
        along_track=along_track_dot,
        along_track_velocity=along_track_velocity_dot,
        mass_error=mass_dot,
    )


def body_force_to_polar(fx_body: float, fy_body: float, alpha: float) -> np.ndarray:
    sine = np.sin(alpha)
    cosine = np.cos(alpha)
    return np.array(
        [
            -fx_body * sine + fy_body * cosine,
            fx_body * cosine + fy_body * sine,
        ],
        dtype=np.float64,
    )


def polar_force_to_body(force_r: float, force_theta: float, alpha: float) -> np.ndarray:
    sine = np.sin(alpha)
    cosine = np.cos(alpha)
    return np.array(
        [
            -force_r * sine + force_theta * cosine,
            force_r * cosine + force_theta * sine,
        ],
        dtype=np.float64,
    )
