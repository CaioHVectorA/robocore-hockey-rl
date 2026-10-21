"""Headless PPO entry point; the desktop UI uses this same worker."""

from rl.cli import main
from rl.training import evaluate_config
from rl.training_config import TrainingConfig


def evaluate(trainer, profile="micro-2024", episodes=10, seed=10000, scenario="faceoff", max_steps=900):
    config = TrainingConfig(
        profile=profile,
        scenario=scenario,
        episode_seconds=max_steps / 30,
        opponent="heuristic",
        eval_episodes=episodes,
    )
    return evaluate_config(trainer, config, episodes, seed)


if __name__ == "__main__":
    main()
