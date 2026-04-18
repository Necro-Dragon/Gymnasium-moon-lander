from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    import gymnasium as gym


DEFAULT_OUTPUT = Path("outputs/random_lander.gif")


@dataclass(slots=True)
class RolloutResult:
    output_path: Path
    total_reward: float
    steps: int


class RandomActionModel:
    """Toy policy that samples a fresh random discrete action every step."""

    def __init__(self, action_space: Any) -> None:
        self._action_space = action_space

    def predict(self, observation: object) -> int:
        del observation
        return int(self._action_space.sample())


def _require_gymnasium():
    try:
        return importlib.import_module("gymnasium")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This legacy Gymnasium demo requires the optional legacy dependencies. "
            "Install them with `python -m pip install -e '.[legacy]'`."
        ) from exc


def save_gif(frames: list[Image.Image], output_path: Path, fps: int) -> None:
    if fps <= 0:
        raise ValueError("fps must be a positive integer")
    if not frames:
        raise ValueError("at least one frame is required to save a GIF")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )


def run_random_rollout(
    output_path: Path,
    seed: int,
    max_steps: int,
    fps: int,
) -> RolloutResult:
    if max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")

    gym = _require_gymnasium()
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    frames: list[Image.Image] = []
    total_reward = 0.0
    steps = 0

    try:
        observation, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        model = RandomActionModel(env.action_space)

        initial_frame = env.render()
        if initial_frame is not None:
            frames.append(Image.fromarray(initial_frame))

        for step_index in range(max_steps):
            action = model.predict(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps = step_index + 1

            frame = env.render()
            if frame is not None:
                frames.append(Image.fromarray(frame))

            if terminated or truncated:
                break
    finally:
        env.close()

    save_gif(frames, output_path=output_path, fps=fps)
    return RolloutResult(output_path=output_path, total_reward=total_reward, steps=steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a random-policy Gymnasium LunarLander episode and export it as a GIF."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for the generated GIF. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for the environment reset and action sampling.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum number of environment steps before ending the rollout.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the exported GIF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_random_rollout(
        output_path=args.output,
        seed=args.seed,
        max_steps=args.max_steps,
        fps=args.fps,
    )
    print(
        f"Saved random LunarLander rollout to {result.output_path} "
        f"after {result.steps} steps with total reward {result.total_reward:.2f}."
    )


if __name__ == "__main__":
    main()
