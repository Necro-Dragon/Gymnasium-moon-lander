from __future__ import annotations

import numpy as np

from .models import CostSpec, FloatArray, TrackingState, VehicleParams


def normalized_tracking_error(tracking: TrackingState, cost: CostSpec) -> FloatArray:
    return np.array(
        [
            tracking.altitude / cost.altitude_scale,
            tracking.radial_velocity / cost.radial_velocity_scale,
            tracking.attitude_error / cost.attitude_scale,
            tracking.attitude_rate_error / cost.attitude_rate_scale,
            tracking.along_track / cost.along_track_scale,
            tracking.along_track_velocity / cost.along_track_velocity_scale,
            tracking.mass_error / cost.mass_scale,
        ],
        dtype=np.float64,
    )


def normalized_control(control: FloatArray, vehicle: VehicleParams) -> FloatArray:
    return np.asarray(control, dtype=np.float64) / vehicle.thrust_limits()


def stage_cost(
    tracking: TrackingState,
    control: FloatArray,
    cost: CostSpec,
    vehicle: VehicleParams,
) -> float:
    error = normalized_tracking_error(tracking, cost)
    control_fraction = normalized_control(control, vehicle)
    return float(error @ error + cost.control_weight * (control_fraction @ control_fraction))


def terminal_cost(tracking: TrackingState, cost: CostSpec) -> float:
    error = normalized_tracking_error(tracking, cost)
    return float(cost.terminal_weight * (error @ error))
