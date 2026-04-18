from __future__ import annotations

from math import pi

from .models import CostSpec, MoonParams, ReferenceTrajectory, ScenarioConfig, VehicleParams
from .terrain import TerrainModel
from .tracking import state_from_tracking_offsets


def default_moon_params() -> MoonParams:
    return MoonParams(
        radius=1_737_400.0,
        mu=4.9048695e12,
        omega=2.0 * pi / (27.321661 * 24.0 * 60.0 * 60.0),
    )


def default_vehicle_params() -> VehicleParams:
    return VehicleParams(
        m0=15_000.0,
        m_dry=6_000.0,
        h_cm=2.0,
        l_cm=1.8,
        j_dry=8.0e4,
        k_fuel=2.5,
        u_rcs_max=4_500.0,
        u_dp_max=45_000.0,
        isp_rcs=290.0,
        isp_dp=311.0,
    )


def default_reference_trajectory(
    moon: MoonParams | None = None,
    vehicle: VehicleParams | None = None,
) -> ReferenceTrajectory:
    moon = default_moon_params() if moon is None else moon
    vehicle = default_vehicle_params() if vehicle is None else vehicle
    return ReferenceTrajectory(moon=moon, vehicle=vehicle, theta0=0.0, beta=0.20)


def default_cost_spec() -> CostSpec:
    return CostSpec()


def default_terrain_model() -> TerrainModel:
    return TerrainModel()


def default_scenarios(reference: ReferenceTrajectory) -> dict[str, ScenarioConfig]:
    vehicle = reference.vehicle
    return {
        "nominal": ScenarioConfig(
            name="nominal",
            description="Moderate downrange offset with healthy fuel reserve.",
            initial_state=state_from_tracking_offsets(
                reference=reference,
                t=0.0,
                altitude=1_200.0,
                radial_velocity=-15.0,
                attitude_error=pi * 8.0 / 180.0,
                attitude_rate_error=-pi / 180.0,
                along_track=250.0,
                along_track_velocity=-1.5,
                mass=vehicle.m0,
            ),
        ),
        "crossrange": ScenarioConfig(
            name="crossrange",
            description="Aggressive along-track recovery requirement.",
            initial_state=state_from_tracking_offsets(
                reference=reference,
                t=0.0,
                altitude=1_200.0,
                radial_velocity=-12.0,
                attitude_error=pi * 5.0 / 180.0,
                attitude_rate_error=0.0,
                along_track=600.0,
                along_track_velocity=-6.0,
                mass=vehicle.m0,
            ),
        ),
        "attitude_recovery": ScenarioConfig(
            name="attitude_recovery",
            description="High initial attitude error and angular-rate recovery case.",
            initial_state=state_from_tracking_offsets(
                reference=reference,
                t=0.0,
                altitude=900.0,
                radial_velocity=-8.0,
                attitude_error=pi * 25.0 / 180.0,
                attitude_rate_error=-8.0 * pi / 180.0,
                along_track=200.0,
                along_track_velocity=0.0,
                mass=vehicle.m0,
            ),
        ),
        "low_fuel": ScenarioConfig(
            name="low_fuel",
            description="Reduced propellant reserve with moderate positional error.",
            initial_state=state_from_tracking_offsets(
                reference=reference,
                t=0.0,
                altitude=1_000.0,
                radial_velocity=-10.0,
                attitude_error=pi * 6.0 / 180.0,
                attitude_rate_error=-pi / 180.0,
                along_track=350.0,
                along_track_velocity=-2.0,
                mass=vehicle.m_dry + 0.25 * (vehicle.m0 - vehicle.m_dry),
            ),
        ),
    }
