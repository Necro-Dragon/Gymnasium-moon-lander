from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import ReferenceTrajectory
from .utils import wrap_angle


@dataclass(slots=True)
class TerrainModel:
    pad_width: float = 120.0
    crater_center: float = 220.0
    crater_depth: float = 25.0
    crater_width: float = 80.0
    roughness_primary_amplitude: float = 3.0
    roughness_secondary_amplitude: float = 1.5

    def is_on_pad_local(self, local_x: float) -> bool:
        return abs(float(local_x)) <= 0.5 * self.pad_width

    def surface_height_local(self, local_x: float) -> float:
        x = float(local_x)
        height = 0.0
        if not self.is_on_pad_local(x):
            height += self.roughness_primary_amplitude * np.sin(2.0 * np.pi * x / 180.0)
            height += self.roughness_secondary_amplitude * np.sin(2.0 * np.pi * x / 67.0)

        crater_half_width = 0.5 * self.crater_width
        crater_offset = (x - self.crater_center) / crater_half_width
        if abs(crater_offset) <= 1.0:
            height -= self.crater_depth * (1.0 - crater_offset**2)
        return float(height)

    def local_x(self, theta: float, reference: ReferenceTrajectory, t: float) -> float:
        theta_ref = reference.sample(t).theta
        return reference.r_ref * wrap_angle(theta - theta_ref)

    def height_at(self, theta: float, reference: ReferenceTrajectory, t: float) -> float:
        return self.surface_height_local(self.local_x(theta, reference, t))

    def is_on_pad(self, theta: float, reference: ReferenceTrajectory, t: float) -> bool:
        return self.is_on_pad_local(self.local_x(theta, reference, t))
