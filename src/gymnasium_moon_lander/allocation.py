from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from .models import FloatArray, VehicleParams


def allocate_body_wrench(
    fx_body: float,
    fy_body: float,
    moment_body: float,
    vehicle: VehicleParams,
) -> FloatArray:
    target = np.array([fx_body, fy_body, moment_body], dtype=np.float64)
    limits = vehicle.thrust_limits()
    wrench_scale = np.maximum(vehicle.max_body_wrench(), 1.0)
    weighted_matrix = vehicle.wrench_matrix() / wrench_scale[:, None]
    weighted_target = target / wrench_scale

    preferred = np.zeros(7, dtype=np.float64)
    preferred[-1] = np.clip(fy_body, 0.0, vehicle.u_dp_max)
    regularization = 1e-3
    control_regularizer = np.diag(1.0 / limits)

    augmented_matrix = np.vstack(
        [
            weighted_matrix,
            np.sqrt(regularization) * control_regularizer,
        ]
    )
    augmented_target = np.concatenate(
        [
            weighted_target,
            np.sqrt(regularization) * (preferred / limits),
        ]
    )

    result = lsq_linear(
        augmented_matrix,
        augmented_target,
        bounds=(np.zeros(7, dtype=np.float64), limits),
        method="trf",
        lsmr_tol="auto",
    )
    return vehicle.clamp_control(result.x)
