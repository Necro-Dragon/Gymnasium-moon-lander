from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import RolloutResult, VehicleParams
from .terrain import TerrainModel
from .utils import ensure_directory


matplotlib.use("Agg")
import matplotlib.pyplot as plt


THRUSTER_LABELS = ("u_l1", "u_l2", "u_l3", "u_r1", "u_r2", "u_r3", "u_dp")


def _save_metrics_json(result: RolloutResult, output_path: Path) -> None:
    output_path.write_text(json.dumps(result.score_report.to_dict(), indent=2), encoding="utf-8")


def save_trajectory_plot(result: RolloutResult, terrain: TerrainModel, output_path: Path) -> None:
    along_track = result.tracking[:, 4]
    altitude = result.tracking[:, 0]
    x_min = min(-800.0, float(np.min(along_track)) - 100.0)
    x_max = max(800.0, float(np.max(along_track)) + 100.0)
    terrain_x = np.linspace(x_min, x_max, 800)
    terrain_y = np.array([terrain.surface_height_local(x) for x in terrain_x], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(terrain_x, terrain_y, terrain_y.min() - 100.0, color="#5a5148", alpha=0.85)
    ax.plot(along_track, altitude, color="#1f77b4", linewidth=2.0, label="Trajectory")
    ax.scatter([along_track[0]], [altitude[0]], color="#111111", s=55, label="Start", zorder=4)
    ax.scatter([along_track[-1]], [altitude[-1]], color="#d62728", s=55, label="End", zorder=4)
    ax.axvspan(-0.5 * terrain.pad_width, 0.5 * terrain.pad_width, color="#7eb26d", alpha=0.2, label="Landing pad")
    ax.set_title(f"Local trajectory: {result.controller_name} on {result.scenario_name}")
    ax.set_xlabel("Along-track position (m)")
    ax.set_ylabel("Altitude above pad reference (m)")
    ax.set_ylim(min(-50.0, terrain_y.min() - 20.0), max(float(np.max(altitude)) + 100.0, 300.0))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_state_history_plot(result: RolloutResult, output_path: Path) -> None:
    labels = (
        "r (m)",
        "r_dot (m/s)",
        "phi (rad)",
        "phi_dot (rad/s)",
        "theta (rad)",
        "theta_dot (rad/s)",
        "m (kg)",
        "along-track error (m)",
    )

    values = [
        result.states[:, 0],
        result.states[:, 1],
        result.states[:, 2],
        result.states[:, 3],
        result.states[:, 4],
        result.states[:, 5],
        result.states[:, 6],
        result.tracking[:, 4],
    ]
    references = [
        result.reference_states[:, 0],
        result.reference_states[:, 1],
        result.reference_states[:, 2],
        result.reference_states[:, 3],
        result.reference_states[:, 4],
        result.reference_states[:, 5],
        result.reference_states[:, 6],
        np.zeros_like(result.times),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    for axis, label, value, reference in zip(axes.flatten(), labels, values, references, strict=True):
        axis.plot(result.times, value, color="#1f77b4", linewidth=1.5)
        axis.plot(result.times, reference, color="#d62728", linewidth=1.0, linestyle="--", alpha=0.7)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    fig.suptitle(f"State history: {result.controller_name} on {result.scenario_name}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_control_history_plot(result: RolloutResult, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    flat_axes = axes.flatten()
    for index, label in enumerate(THRUSTER_LABELS):
        flat_axes[index].plot(result.control_times, result.controls[:, index], color="#ff7f0e", linewidth=1.4)
        flat_axes[index].set_ylabel(f"{label} (N)")
        flat_axes[index].grid(True, alpha=0.25)
    flat_axes[-1].plot(
        result.control_times,
        np.sum(result.controls, axis=1),
        color="#2ca02c",
        linewidth=1.4,
    )
    flat_axes[-1].set_ylabel("Total thrust (N)")
    flat_axes[-1].grid(True, alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    fig.suptitle(f"Control history: {result.controller_name} on {result.scenario_name}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_cost_history_plot(result: RolloutResult, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(result.control_times, result.stage_costs, color="#9467bd", linewidth=1.5)
    axes[0].set_ylabel("Instantaneous cost")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(result.times, result.cumulative_costs, color="#8c564b", linewidth=1.5)
    axes[1].set_ylabel("Integrated cost")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(f"Cost history: {result.controller_name} on {result.scenario_name}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _rotate_points(points: list[tuple[float, float]], angle: float) -> list[tuple[float, float]]:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return [
        (cosine * x - sine * y, sine * x + cosine * y)
        for x, y in points
    ]


def _render_rollout_frame(
    result: RolloutResult,
    index: int,
    terrain: TerrainModel,
    vehicle: VehicleParams,
    width: int = 960,
    height: int = 540,
) -> Image.Image:
    image = Image.new("RGB", (width, height), color="#0d1725")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, width, height), fill="#09121d")
    draw.rectangle((0, int(height * 0.60), width, height), fill="#101821")

    x_window = (-800.0, 800.0)
    y_window = (-100.0, max(1_500.0, float(result.tracking[index, 0]) + 250.0))
    x_scale = width / (x_window[1] - x_window[0])
    y_scale = (height - 80) / (y_window[1] - y_window[0])
    ground_px = height - 60

    def to_screen(local_x: float, local_y: float) -> tuple[float, float]:
        px = (local_x - x_window[0]) * x_scale
        py = ground_px - (local_y - y_window[0]) * y_scale
        return px, py

    terrain_points: list[tuple[float, float]] = []
    for sample in np.linspace(x_window[0], x_window[1], 600):
        terrain_points.append(to_screen(float(sample), terrain.surface_height_local(float(sample))))
    polygon = [(0.0, float(height)), *terrain_points, (float(width), float(height))]
    draw.polygon(polygon, fill="#5d5248")
    pad_left = to_screen(-0.5 * terrain.pad_width, 0.0)
    pad_right = to_screen(0.5 * terrain.pad_width, 0.0)
    draw.rectangle((pad_left[0], pad_left[1] - 3, pad_right[0], pad_right[1] + 3), fill="#7ac96d")

    trail_points = [
        to_screen(float(x), float(y))
        for x, y in zip(result.tracking[: index + 1, 4], result.tracking[: index + 1, 0], strict=True)
    ]
    if len(trail_points) >= 2:
        draw.line(trail_points, fill="#67b7ff", width=3)

    local_x = float(result.tracking[index, 4])
    local_y = float(result.tracking[index, 0])
    body_angle = -float(result.tracking[index, 2])
    center_x, center_y = to_screen(local_x, local_y)

    base_shape = [(-12.0, 14.0), (12.0, 14.0), (18.0, -12.0), (-18.0, -12.0)]
    rotated_shape = _rotate_points(base_shape, body_angle)
    body_points = [(center_x + x, center_y + y) for x, y in rotated_shape]
    draw.polygon(body_points, fill="#e2e2d5", outline="#111111")
    draw.line(
        [
            (center_x, center_y - 18.0),
            (
                center_x + 24.0 * np.sin(body_angle),
                center_y - 24.0 * np.cos(body_angle),
            ),
        ],
        fill="#ffdf5d",
        width=3,
    )

    control_index = min(index, result.controls.shape[0] - 1)
    control = result.controls[control_index]
    control_fraction = control / vehicle.thrust_limits()
    nozzle_map = [
        (-10.0, 14.0, 0.0, 1.0),
        (-18.0, 0.0, -1.0, 0.0),
        (10.0, 14.0, 0.0, 1.0),
        (8.0, 14.0, 0.0, 1.0),
        (18.0, 0.0, 1.0, 0.0),
        (-8.0, 14.0, 0.0, 1.0),
        (0.0, 16.0, 0.0, 1.0),
    ]
    cosine = float(np.cos(body_angle))
    sine = float(np.sin(body_angle))
    for thruster_fraction, (x0, y0, dx, dy) in zip(control_fraction, nozzle_map, strict=True):
        if thruster_fraction <= 0.02:
            continue
        nozzle_x = center_x + cosine * x0 - sine * y0
        nozzle_y = center_y + sine * x0 + cosine * y0
        direction_x = cosine * dx - sine * dy
        direction_y = sine * dx + cosine * dy
        plume_length = 20.0 + 40.0 * float(thruster_fraction)
        plume_tip = (
            nozzle_x + direction_x * plume_length,
            nozzle_y + direction_y * plume_length,
        )
        draw.line((nozzle_x, nozzle_y, plume_tip[0], plume_tip[1]), fill="#ff8c3b", width=4)

    hud_lines = [
        f"controller: {result.controller_name}",
        f"scenario: {result.scenario_name}",
        f"time: {result.times[index]:6.2f} s",
        f"mass: {result.states[index, 6]:8.1f} kg",
        f"score: {result.cumulative_costs[index]:8.2f}",
        f"status: {result.score_report.terminal_status}",
    ]
    for line_index, line in enumerate(hud_lines):
        draw.text((20, 18 + 18 * line_index), line, fill="#e8eef8", font=font)
    return image


def save_rollout_gif(
    result: RolloutResult,
    terrain: TerrainModel,
    vehicle: VehicleParams,
    output_path: Path,
    fps: int = 20,
    max_frames: int = 240,
) -> None:
    if result.times.size == 0:
        raise ValueError("cannot render an empty rollout")
    frame_count = min(max_frames, result.times.size)
    indices = np.unique(np.linspace(0, result.times.size - 1, frame_count, dtype=np.int32))
    frames = [_render_rollout_frame(result, int(index), terrain, vehicle) for index in indices]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1_000 / fps),
        loop=0,
    )


def save_rollout_artifacts(
    result: RolloutResult,
    terrain: TerrainModel,
    vehicle: VehicleParams,
    output_dir: Path,
    include_gif: bool = True,
) -> Path:
    ensure_directory(output_dir)
    _save_metrics_json(result, output_dir / "metrics.json")
    save_trajectory_plot(result, terrain, output_dir / "trajectory_local.png")
    save_state_history_plot(result, output_dir / "state_history.png")
    save_control_history_plot(result, output_dir / "control_history.png")
    save_cost_history_plot(result, output_dir / "cost_history.png")
    if include_gif:
        save_rollout_gif(result, terrain, vehicle, output_dir / "rollout.gif")
    return output_dir
