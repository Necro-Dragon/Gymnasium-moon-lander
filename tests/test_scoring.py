from __future__ import annotations

import numpy as np

from gymnasium_moon_lander.models import TrackingState
from gymnasium_moon_lander.scenarios import default_cost_spec, default_reference_trajectory, default_vehicle_params
from gymnasium_moon_lander.scoring import normalized_tracking_error, stage_cost, terminal_cost
from gymnasium_moon_lander.tracking import tracking_state


def test_reference_tracking_scores_better_than_perturbed_state() -> None:
    reference = default_reference_trajectory()
    cost = default_cost_spec()
    control = np.zeros(7, dtype=np.float64)
    perfect = tracking_state(reference.sample(0.0), reference, 0.0)
    perturbed = TrackingState(
        altitude=250.0,
        radial_velocity=-7.5,
        attitude_error=np.deg2rad(12.0),
        attitude_rate_error=np.deg2rad(3.0),
        along_track=300.0,
        along_track_velocity=-4.0,
        mass_error=500.0,
    )
    assert stage_cost(perfect, control, cost, reference.vehicle) < stage_cost(perturbed, control, cost, reference.vehicle)
    assert terminal_cost(perfect, cost) < terminal_cost(perturbed, cost)


def test_unsafe_touchdown_scores_worse_than_safe_touchdown() -> None:
    cost = default_cost_spec()
    vehicle = default_vehicle_params()
    safe = TrackingState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    unsafe = TrackingState(
        altitude=0.0,
        radial_velocity=-3.5,
        attitude_error=np.deg2rad(9.0),
        attitude_rate_error=np.deg2rad(7.0),
        along_track=40.0,
        along_track_velocity=-2.5,
        mass_error=0.0,
    )
    safe_total = terminal_cost(safe, cost)
    unsafe_total = terminal_cost(unsafe, cost) + cost.penalty_for("touchdown_hard")
    assert unsafe_total > safe_total
    assert np.linalg.norm(normalized_tracking_error(unsafe, cost)) > np.linalg.norm(normalized_tracking_error(safe, cost))
