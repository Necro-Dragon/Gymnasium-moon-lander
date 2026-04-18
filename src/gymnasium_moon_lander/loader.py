from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

import numpy as np

from .controllers import OpenLoopController, ScriptedHoverDescentController, TVLQRTrackingController
from .models import ControllerProtocol, CostSpec, MoonParams, VehicleParams
from .simulation import SimulationConfig


ControllerFactory = Callable[[], ControllerProtocol]


def builtin_controller_names() -> tuple[str, ...]:
    return ("scripted_hover_descent", "tvlqr_tracking")


def load_controller_factory(
    spec: str,
    moon: MoonParams,
    vehicle: VehicleParams,
    cost: CostSpec,
    simulation: SimulationConfig,
) -> tuple[str, ControllerFactory]:
    if spec == "scripted_hover_descent":
        return spec, lambda: ScriptedHoverDescentController(moon=moon, vehicle=vehicle)
    if spec == "tvlqr_tracking":
        return spec, lambda: TVLQRTrackingController(
            moon=moon,
            vehicle=vehicle,
            cost=cost,
            simulation=simulation,
        )
    if spec.endswith(".npz"):
        path = Path(spec)
        bundle = np.load(path)
        times = np.asarray(bundle["t"], dtype=np.float64)
        controls = np.asarray(bundle["u"], dtype=np.float64)
        if times.ndim != 1 or controls.ndim != 2 or controls.shape[1] != 7:
            raise ValueError("open-loop bundle must contain t with shape (N,) and u with shape (N, 7)")
        if controls.shape[0] != times.shape[0]:
            raise ValueError("open-loop bundle arrays t and u must have the same first dimension")
        return path.stem, lambda: OpenLoopController(control_times=times, controls=controls, name=path.stem)
    if ":" in spec:
        module_name, factory_name = spec.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        if not callable(factory):
            raise TypeError(f"{spec} must resolve to a callable factory")

        def wrapped_factory() -> ControllerProtocol:
            controller = factory()
            if not hasattr(controller, "reset") or not hasattr(controller, "action"):
                raise TypeError(f"{spec} did not return a compatible controller")
            return controller

        return factory_name, wrapped_factory
    raise ValueError(f"unknown controller spec: {spec}")
