"""Paper-aligned lunar descent simulator and legacy Gymnasium demos."""

from .allocation import allocate_body_wrench
from .models import (
    CONTROL_SIZE,
    STATE_SIZE,
    ControllerProtocol,
    CostSpec,
    LanderState,
    MoonParams,
    ReferenceTrajectory,
    RolloutResult,
    ScenarioConfig,
    ScoreReport,
    TrackingState,
    VehicleParams,
)
from .scenarios import (
    default_cost_spec,
    default_moon_params,
    default_reference_trajectory,
    default_scenarios,
    default_terrain_model,
    default_vehicle_params,
)
from .simulation import SimulationConfig, simulate_rollout

__all__ = [
    "CONTROL_SIZE",
    "STATE_SIZE",
    "ControllerProtocol",
    "CostSpec",
    "LanderState",
    "MoonParams",
    "ReferenceTrajectory",
    "RolloutResult",
    "ScenarioConfig",
    "ScoreReport",
    "SimulationConfig",
    "TrackingState",
    "VehicleParams",
    "allocate_body_wrench",
    "default_cost_spec",
    "default_moon_params",
    "default_reference_trajectory",
    "default_scenarios",
    "default_terrain_model",
    "default_vehicle_params",
    "simulate_rollout",
]
