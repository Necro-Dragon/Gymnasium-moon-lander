from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
from scipy.optimize import minimize

from gymnasium.envs.box2d.lunar_lander import (
    FPS,
    MAIN_ENGINE_POWER,
    MAIN_ENGINE_Y_LOCATION,
    SCALE,
    SIDE_ENGINE_AWAY,
    SIDE_ENGINE_HEIGHT,
    SIDE_ENGINE_POWER,
    VIEWPORT_H,
    VIEWPORT_W,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DT = 1.0 / FPS
X_SCALE = VIEWPORT_W / SCALE / 2.0
Y_SCALE = VIEWPORT_H / SCALE / 2.0
VX_SCALE = X_SCALE / FPS
VY_SCALE = Y_SCALE / FPS
OMEGA_SCALE = 20.0 / FPS
MAIN_ACCEL_COEFF = MAIN_ENGINE_POWER * (MAIN_ENGINE_Y_LOCATION / SCALE) / DT
SIDE_TORQUE_IMPULSE = (
    (SIDE_ENGINE_HEIGHT / SCALE) * (SIDE_ENGINE_AWAY / SCALE) * SIDE_ENGINE_POWER
)


@dataclass(slots=True)
class ModelParams:
    mass: float
    inertia: float
    gravity: float = 10.0
    main_accel: float = MAIN_ACCEL_COEFF
    side_torque_accel: float = 0.0

    @classmethod
    def from_env(cls, env: gym.Env) -> "ModelParams":
        base = env.unwrapped
        side_torque_accel = SIDE_TORQUE_IMPULSE / (base.lander.inertia * DT)
        return cls(
            mass=float(base.lander.mass),
            inertia=float(base.lander.inertia),
            gravity=abs(float(base.gravity)),
            main_accel=MAIN_ACCEL_COEFF / float(base.lander.mass),
            side_torque_accel=side_torque_accel,
        )


@dataclass(slots=True)
class ControllerConfig:
    horizon: int = 14
    maxiter: int = 24
    replanning_interval: int = 4
    mode: str = "baseline"


@dataclass(slots=True)
class EpisodeResult:
    seed: int
    mode: str
    total_reward: float
    steps: int
    observations: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    target: np.ndarray


def observation_to_canonical(observation: np.ndarray, params: ModelParams) -> np.ndarray:
    vx = float(observation[2]) / VX_SCALE
    vy = float(observation[3]) / VY_SCALE
    omega = float(observation[5]) / OMEGA_SCALE
    return np.array(
        [
            float(observation[0]) * X_SCALE,
            float(observation[1]) * Y_SCALE,
            params.mass * vx,
            params.mass * vy,
            float(observation[4]),
            params.inertia * omega,
            float(observation[6]),
            float(observation[7]),
        ],
        dtype=np.float64,
    )


def canonical_to_features(state: np.ndarray, params: ModelParams) -> np.ndarray:
    vx = state[2] / params.mass
    vy = state[3] / params.mass
    omega = state[5] / params.inertia
    return np.array(
        [
            state[0] / X_SCALE,
            state[1] / Y_SCALE,
            vx * VX_SCALE,
            vy * VY_SCALE,
            state[4],
            omega * OMEGA_SCALE,
            state[6],
            state[7],
        ],
        dtype=np.float64,
    )


def main_engine_power(command: float) -> float:
    if command <= 0.0:
        return 0.0
    return 0.5 + 0.5 * np.clip(command, 0.0, 1.0)


def side_engine_command(command: float) -> float:
    if abs(command) <= 0.5:
        return 0.0
    return float(np.clip(command, -1.0, 1.0))


def hamiltonian_step(state: np.ndarray, control: np.ndarray, params: ModelParams) -> np.ndarray:
    x, y, px, py, theta, ptheta, left_contact, right_contact = state
    vx = px / params.mass
    vy = py / params.mass
    omega = ptheta / params.inertia

    main_power = main_engine_power(float(control[0]))
    side_cmd = side_engine_command(float(control[1]))

    ax = -params.main_accel * main_power * np.sin(theta)
    ay = params.main_accel * main_power * np.cos(theta) - params.gravity
    alpha = -params.side_torque_accel * side_cmd

    next_state = state.copy()
    next_state[0] = x + DT * vx
    next_state[1] = y + DT * vy
    next_state[2] = px + DT * params.mass * ax
    next_state[3] = py + DT * params.mass * ay
    next_state[4] = theta + DT * omega
    next_state[5] = ptheta + DT * params.inertia * alpha

    if next_state[1] <= 0.0:
        next_state[1] = 0.0
        if abs(vy) < 0.35 and abs(theta) < 0.2 and abs(vx) < 0.25:
            next_state[2] *= 0.4
            next_state[3] = 0.0
            next_state[5] *= 0.4
            next_state[6] = 1.0
            next_state[7] = 1.0
        else:
            next_state[2] *= 0.25
            next_state[3] = -0.2 * next_state[3]
            next_state[5] *= 0.25
            next_state[6] = 0.0
            next_state[7] = 0.0
    else:
        next_state[6] = 0.0
        next_state[7] = 0.0
    return next_state


def gym_like_stage_cost(state: np.ndarray, control: np.ndarray, params: ModelParams) -> float:
    obs = canonical_to_features(state, params)
    cost = (
        100.0 * np.hypot(obs[0], obs[1])
        + 100.0 * np.hypot(obs[2], obs[3])
        + 100.0 * abs(obs[4])
        - 10.0 * obs[6]
        - 10.0 * obs[7]
        + 0.30 * main_engine_power(float(control[0]))
        + 0.03 * abs(side_engine_command(float(control[1])))
    )
    if obs[1] < -0.02:
        cost += 600.0 * abs(obs[1])
    if abs(obs[0]) > 1.0:
        cost += 800.0 * (abs(obs[0]) - 1.0)
    return cost


def guidance_stage_cost(state: np.ndarray, control: np.ndarray, params: ModelParams) -> float:
    obs = canonical_to_features(state, params)
    angle_ref = np.clip(0.5 * obs[0] + 1.0 * obs[2], -0.4, 0.4)
    hover_ref = 0.55 * abs(obs[0])
    main_power = main_engine_power(float(control[0]))
    side_cmd = side_engine_command(float(control[1]))

    cost = (
        18.0 * (obs[4] - angle_ref) ** 2
        + 10.0 * obs[5] ** 2
        + 18.0 * (obs[1] - hover_ref) ** 2
        + 12.0 * obs[3] ** 2
        + 8.0 * obs[0] ** 2
        + 6.0 * obs[2] ** 2
        + 0.45 * main_power**2
        + 0.08 * side_cmd**2
        - 4.0 * (obs[6] + obs[7])
    )
    if abs(obs[0]) < 0.18:
        cost += 8.0 * obs[1] ** 2
    if obs[1] < -0.02:
        cost += 800.0 * abs(obs[1])
    return cost


def terminal_cost(state: np.ndarray, params: ModelParams, mode: str) -> float:
    obs = canonical_to_features(state, params)
    speed = np.hypot(obs[2], obs[3])
    cost = 120.0 * obs[0] ** 2 + 160.0 * obs[1] ** 2 + 140.0 * speed**2
    cost += 120.0 * obs[4] ** 2 + 60.0 * obs[5] ** 2
    if obs[6] > 0.5 and obs[7] > 0.5:
        cost -= 60.0
    if mode == "improved" and abs(obs[0]) < 0.12 and obs[1] < 0.12:
        if speed < 0.18 and abs(obs[4]) < 0.15 and abs(obs[5]) < 0.2:
            cost -= 120.0
    return cost


def rollout_cost(
    flat_controls: np.ndarray,
    initial_state: np.ndarray,
    params: ModelParams,
    mode: str,
    nominal_plan: np.ndarray | None = None,
) -> float:
    controls = flat_controls.reshape(-1, 2)
    if nominal_plan is not None:
        controls = np.clip(nominal_plan + controls, -1.0, 1.0)
    state = initial_state.copy()
    total_cost = 0.0
    for control in controls:
        if mode == "baseline":
            total_cost += gym_like_stage_cost(state, control, params)
        else:
            total_cost += guidance_stage_cost(state, control, params)
        state = hamiltonian_step(state, control, params)
    if nominal_plan is not None:
        total_cost += 0.75 * float(np.sum(flat_controls**2))
    total_cost += terminal_cost(state, params, mode)
    return float(total_cost)


def heuristic_guess(observation: np.ndarray) -> np.ndarray:
    angle_target = np.clip(observation[0] * 0.5 + observation[2], -0.4, 0.4)
    hover_target = 0.55 * abs(observation[0])

    angle_todo = (angle_target - observation[4]) * 0.5 - observation[5]
    hover_todo = (hover_target - observation[1]) * 0.5 - observation[3] * 0.5

    if observation[6] > 0.5 or observation[7] > 0.5:
        angle_todo = 0.0
        hover_todo = -0.5 * observation[3]

    action = np.array([hover_todo * 20.0 - 1.0, -angle_todo * 20.0], dtype=np.float64)
    return np.clip(action, -1.0, 1.0)


class RecedingHorizonController:
    def __init__(self, params: ModelParams, config: ControllerConfig) -> None:
        self.params = params
        self.config = config
        self._plan = np.zeros((config.horizon, 2), dtype=np.float64)
        self._steps_until_replan = 0

    def _initial_plan(self, observation: np.ndarray) -> np.ndarray:
        base_action = heuristic_guess(observation)
        return np.tile(base_action, (self.config.horizon, 1))

    def _nominal_plan(self, observation: np.ndarray) -> np.ndarray:
        nominal = np.zeros((self.config.horizon, 2), dtype=np.float64)
        predicted_obs = observation.astype(np.float64)
        predicted_state = observation_to_canonical(predicted_obs, self.params)
        for index in range(self.config.horizon):
            action = heuristic_guess(predicted_obs)
            nominal[index] = action
            predicted_state = hamiltonian_step(predicted_state, action, self.params)
            predicted_obs = canonical_to_features(predicted_state, self.params)
        return nominal

    def _solve(self, observation: np.ndarray) -> None:
        initial_state = observation_to_canonical(observation, self.params)
        bounds = [(-1.0, 1.0)] * (2 * self.config.horizon)
        if self.config.mode == "baseline":
            guess = self._plan.copy()
            if not np.any(guess):
                guess = self._initial_plan(observation)
            result = minimize(
                rollout_cost,
                guess.reshape(-1),
                args=(initial_state, self.params, self.config.mode),
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": self.config.maxiter, "ftol": 1e-3, "disp": False},
            )
            if result.success:
                self._plan = result.x.reshape(self.config.horizon, 2)
            else:
                self._plan = guess
        else:
            nominal = self._nominal_plan(observation)
            delta_guess = np.zeros_like(nominal)
            result = minimize(
                rollout_cost,
                delta_guess.reshape(-1),
                args=(initial_state, self.params, self.config.mode, nominal),
                method="SLSQP",
                bounds=[(-0.4, 0.4)] * (2 * self.config.horizon),
                options={"maxiter": self.config.maxiter, "ftol": 1e-3, "disp": False},
            )
            if result.success:
                self._plan = np.clip(
                    nominal + result.x.reshape(self.config.horizon, 2),
                    -1.0,
                    1.0,
                )
            else:
                self._plan = nominal
        self._steps_until_replan = self.config.replanning_interval

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self._steps_until_replan <= 0:
            self._solve(observation)
        action = self._plan[0].copy()
        self._plan[:-1] = self._plan[1:]
        self._plan[-1] = self._plan[-2]
        self._steps_until_replan -= 1
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def target_position_from_env(env: gym.Env) -> np.ndarray:
    base = env.unwrapped
    target_x = ((base.helipad_x1 + base.helipad_x2) * 0.5 - X_SCALE) / X_SCALE
    return np.array([target_x, 0.0], dtype=np.float64)


def save_parametric_trajectory_plot(result: EpisodeResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_positions = result.positions[:, 0]
    y_positions = result.positions[:, 1]
    timesteps = np.arange(result.positions.shape[0], dtype=np.int32)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_positions, y_positions, color="tab:blue", linewidth=1.5, alpha=0.7)
    trajectory = ax.scatter(
        x_positions,
        y_positions,
        c=timesteps,
        cmap="viridis",
        s=22,
        zorder=3,
    )
    ax.scatter(
        [result.target[0]],
        [result.target[1]],
        color="red",
        marker="x",
        s=160,
        linewidths=2.5,
        label="Target between flags",
        zorder=4,
    )
    ax.scatter(
        [x_positions[0]],
        [y_positions[0]],
        facecolors="none",
        edgecolors="black",
        s=70,
        label="Start",
        zorder=4,
    )
    ax.scatter(
        [x_positions[-1]],
        [y_positions[-1]],
        color="black",
        marker="s",
        s=40,
        label="End",
        zorder=4,
    )
    ax.set_title(f"LunarLander trajectory ({result.mode}, seed={result.seed})")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")

    colorbar = fig.colorbar(trajectory, ax=ax)
    colorbar.set_label("Timestep")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_state_timeseries_plot(result: EpisodeResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = (
        "x position",
        "y position",
        "x velocity",
        "y velocity",
        "angle",
        "angular velocity",
        "left leg contact",
        "right leg contact",
    )
    reference_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None)

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    flat_axes = axes.flatten()

    for index, (axis, label, reference) in enumerate(
        zip(flat_axes, labels, reference_values, strict=True)
    ):
        axis.plot(result.times, result.observations[:, index], color="tab:blue", linewidth=1.6)
        if reference is not None:
            axis.axhline(reference, color="tab:red", linewidth=1.0, linestyle="--", alpha=0.65)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.set_xlim(result.times[0], result.times[-1])

    for axis in flat_axes[-2:]:
        axis.set_xlabel("time (s)")

    fig.suptitle(f"LunarLander state history ({result.mode}, seed={result.seed})")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_episode(seed: int, mode: str) -> EpisodeResult:
    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        gravity=-10.0,
        enable_wind=False,
        render_mode=None,
    )
    try:
        observation, _ = env.reset(seed=seed)
        params = ModelParams.from_env(env)
        controller = RecedingHorizonController(
            params=params,
            config=ControllerConfig(mode=mode),
        )
        total_reward = 0.0
        observations = [observation.astype(np.float64)]
        target = target_position_from_env(env)
        steps = 0

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = controller.act(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            observations.append(observation.astype(np.float64))
            steps += 1
        observation_history = np.vstack(observations)
        times = np.arange(observation_history.shape[0], dtype=np.float64) * DT
        return EpisodeResult(
            seed=seed,
            mode=mode,
            total_reward=total_reward,
            steps=steps,
            observations=observation_history,
            times=times,
            positions=observation_history[:, :2],
            target=target,
        )
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Hamiltonian-style receding-horizon controller on LunarLander-v3 "
            "without wind or turbulence."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "improved"],
        default="baseline",
        help="Which cost functional to optimize.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 11, 19],
        help="Episode seeds to evaluate.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Optional PNG path for a parametric x/y trajectory plot. Requires exactly one seed.",
    )
    parser.add_argument(
        "--state-plot-output",
        type=Path,
        default=None,
        help="Optional PNG path for a time-series plot of all observation state variables. Requires exactly one seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.plot_output is not None or args.state_plot_output is not None) and len(args.seeds) != 1:
        raise SystemExit("--plot-output and --state-plot-output require exactly one seed.")

    results = [run_episode(seed=seed, mode=args.mode) for seed in args.seeds]
    scores = [result.total_reward for result in results]
    mean_score = sum(scores) / len(scores)
    for result in results:
        print(f"mode={result.mode} seed={result.seed} score={result.total_reward:.2f}")
    if args.plot_output is not None:
        save_parametric_trajectory_plot(results[0], args.plot_output)
        print(f"Saved trajectory plot to {args.plot_output}")
    if args.state_plot_output is not None:
        save_state_timeseries_plot(results[0], args.state_plot_output)
        print(f"Saved state history plot to {args.state_plot_output}")
    print(f"mode={args.mode} mean_score={mean_score:.2f}")


if __name__ == "__main__":
    main()
