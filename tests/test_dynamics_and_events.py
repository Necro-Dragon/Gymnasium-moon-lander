from __future__ import annotations

import numpy as np

from gymnasium_moon_lander.dynamics import step_rk4
from gymnasium_moon_lander.scenarios import default_reference_trajectory, default_terrain_model, default_vehicle_params
from gymnasium_moon_lander.simulation import check_termination
from gymnasium_moon_lander.tracking import state_from_tracking_offsets


def test_zero_thrust_descent_loses_altitude() -> None:
    reference = default_reference_trajectory()
    vehicle = reference.vehicle
    initial = state_from_tracking_offsets(
        reference=reference,
        t=0.0,
        altitude=200.0,
        radial_velocity=-5.0,
        attitude_error=0.0,
        attitude_rate_error=0.0,
        along_track=0.0,
        along_track_velocity=0.0,
        mass=vehicle.m0,
    )
    next_state = step_rk4(
        t=0.0,
        state=initial,
        control=np.zeros(7, dtype=np.float64),
        sim_dt=0.1,
        moon=reference.moon,
        vehicle=vehicle,
    )
    assert next_state.r < initial.r


def test_mass_decreases_under_thrust_and_stops_at_dry_mass() -> None:
    reference = default_reference_trajectory()
    vehicle = reference.vehicle
    state = state_from_tracking_offsets(
        reference=reference,
        t=0.0,
        altitude=20.0,
        radial_velocity=0.0,
        attitude_error=0.0,
        attitude_rate_error=0.0,
        along_track=0.0,
        along_track_velocity=0.0,
        mass=vehicle.m_dry + 5.0,
    )
    control = np.array([vehicle.u_rcs_max] * 6 + [vehicle.u_dp_max], dtype=np.float64)
    masses = [state.m]
    for _ in range(50):
        state = step_rk4(
            t=0.0,
            state=state,
            control=control,
            sim_dt=0.1,
            moon=reference.moon,
            vehicle=vehicle,
        )
        masses.append(state.m)
    assert masses[-1] >= vehicle.m_dry
    assert all(left >= right for left, right in zip(masses, masses[1:]))


def test_contact_detection_reports_success_on_reference_touchdown() -> None:
    reference = default_reference_trajectory()
    terrain = default_terrain_model()
    state = reference.sample(0.0)
    status = check_termination(
        state=state,
        reference=reference,
        terrain=terrain,
        t=0.0,
        vehicle=reference.vehicle,
    )
    assert status is not None
    assert status.success
    assert status.terminal_status == "touchdown_success"


def test_escape_and_fuel_termination_conditions_trigger() -> None:
    reference = default_reference_trajectory()
    terrain = default_terrain_model()
    vehicle = default_vehicle_params()
    along_track_escape = state_from_tracking_offsets(
        reference=reference,
        t=0.0,
        altitude=100.0,
        radial_velocity=0.0,
        attitude_error=0.0,
        attitude_rate_error=0.0,
        along_track=2_100.0,
        along_track_velocity=0.0,
        mass=vehicle.m0,
    )
    fuel_depleted = state_from_tracking_offsets(
        reference=reference,
        t=0.0,
        altitude=100.0,
        radial_velocity=0.0,
        attitude_error=0.0,
        attitude_rate_error=0.0,
        along_track=0.0,
        along_track_velocity=0.0,
        mass=vehicle.m_dry,
    )
    assert check_termination(along_track_escape, reference, terrain, 0.0, vehicle).terminal_status == "along_track_escape"
    assert check_termination(fuel_depleted, reference, terrain, 0.0, vehicle).terminal_status == "fuel_depletion"
