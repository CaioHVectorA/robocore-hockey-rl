"""Configurable PPO training, also used by the desktop experiment interface."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import shutil
from uuid import uuid4
from core.constants import PROFILES, RobotConfig
from rl.env import SCENARIOS
from rl.ppo_trainer import HockeyPPOTrainer
from rl.training import TrainingSession, evaluate_config
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Experiment JSON saved by the interface")
    parser.add_argument("--run-dir")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--scenario", choices=(*SCENARIOS, "curriculum"))
    parser.add_argument("--opponent", choices=["selfplay", "mixed", "heuristic", "zero", "none"])
    parser.add_argument("--robots", type=int)
    parser.add_argument("--opponent-robots", type=int)
    parser.add_argument("--objective", choices=["reach", "score"])
    parser.add_argument("--save")
    parser.add_argument("--resume")
    parser.add_argument("--init-from", help="Transfer compatible actor weights to a new task")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rollout", type=int)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--log", help="Also copy the metrics JSONL here on exit")
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("Choose --resume or --init-from")
    try:
        config = TrainingConfig.load(args.config) if args.config else TrainingConfig()
        updates = {
            key: getattr(args, key)
            for key in (
                "steps",
                "profile",
                "scenario",
                "opponent",
                "seed",
                "rollout",
                "envs",
                "eval_episodes",
                "learning_rate",
                "speed",
                "opponent_robots",
                "objective",
            )
            if getattr(args, key) is not None
        }
        if args.robots is not None:
            updates["robots_per_team"] = args.robots
        if args.episode_steps is not None:
            updates["episode_seconds"] = args.episode_steps / 30
        if args.resume or args.init_from:
            updates.update(
                init_mode="resume" if args.resume else "transfer",
                checkpoint=str(Path(args.resume or args.init_from).resolve()),
            )
        config = replace(config, **updates)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    if args.eval_only:
        if not config.checkpoint:
            parser.error("--eval-only requires --resume, --init-from or a config with checkpoint")
        trainer = HockeyPPOTrainer()
        meta = trainer.load(config.checkpoint)
        if not args.config and "training_config" in meta:
            config = replace(TrainingConfig.from_dict(meta["training_config"]), **updates)
        if meta.get("configuration", {}).get("arena") != PROFILES[config.profile].to_dict():
            parser.error("Checkpoint and selected arena differ")
        if meta.get("configuration", {}).get("robot") != asdict(RobotConfig()):
            parser.error("Checkpoint and current HockeyBot model differ")
        print(json.dumps(evaluate_config(trainer, config), indent=2))
        return
    directory = (
        Path(args.run_dir)
        if args.run_dir
        else Path("runs/experiments") / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6])
    )
    save = Path(args.save).resolve() if args.save else None
    session = TrainingSession(config, directory, save)
    session.run()
    if args.log:
        target = Path(args.log)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / "metrics.jsonl", target)
    print(f"Session saved: {directory.resolve()}", flush=True)


if __name__ == "__main__":
    main()
