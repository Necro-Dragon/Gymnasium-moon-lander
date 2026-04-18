from __future__ import annotations

import numpy as np

from gymnasium_moon_lander.allocation import allocate_body_wrench
from gymnasium_moon_lander.dynamics import control_to_body_wrench, force_breakdown
from gymnasium_moon_lander.scenarios import default_moon_params, default_reference_trajectory, default_vehicle_params


def test_symmetric_vertical_thrusters_produce_zero_lateral_force() -> None:
    vehicle = default_vehicle_params()
    moon = default_moon_params()
    state = default_reference_trajectory(moon=moon, vehicle=vehicle).sample(0.0)
    control = np.array([1_500.0, 0.0, 1_500.0, 1_500.0, 0.0, 1_500.0, 0.0], dtype=np.float64)
    forces = force_breakdown(state, control, moon, vehicle)
    assert np.isclose(forces.fx_body, 0.0)


def test_matched_side_thrusters_cancel_moment() -> None:
    vehicle = default_vehicle_params()
    control = np.array([0.0, 1_750.0, 0.0, 0.0, 1_750.0, 0.0, 0.0], dtype=np.float64)
    fx_body, fy_body, moment = control_to_body_wrench(control, vehicle)
    assert np.isclose(fx_body, 0.0)
    assert np.isclose(fy_body, 0.0)
    assert np.isclose(moment, 0.0)


def test_allocate_body_wrench_respects_caps_and_matches_feasible_target() -> None:
    vehicle = default_vehicle_params()
    target = np.array([1_200.0, 16_000.0, 2_500.0], dtype=np.float64)
    control = allocate_body_wrench(*target, vehicle=vehicle)
    achieved = vehicle.wrench_matrix() @ control

    assert control.shape == (7,)
    assert np.all(control >= 0.0)
    assert np.all(control <= vehicle.thrust_limits() + 1e-9)
    assert np.allclose(achieved, target, atol=15.0)
