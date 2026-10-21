"""Evaluate frozen controllers on identical reserved initial seeds."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import torch
from core.constants import PROFILES, RobotConfig
from bots.hockey_bot import HeuristicHockeyBot
from rl.env import SCENARIOS
from rl.ppo_trainer import HockeyPPOTrainer
from rl.training_config import TrainingConfig
from rl.training import curriculum_stage


def benchmark(episodes=10, seed=10000, profile=None, policy=None, max_steps=None, scenario=None):
    torch.set_num_threads(1)
    controllers = ["zero", "heuristic"]
    trainer = None
    config = TrainingConfig()
    if policy:
        trainer = HockeyPPOTrainer()
        meta = trainer.load(policy)
        if "training_config" in meta:
            config = TrainingConfig.from_dict(meta["training_config"])
        if meta.get("configuration", {}).get("arena") != PROFILES[profile or config.profile].to_dict():
            raise ValueError("Policy/profile mismatch")
        if meta.get("configuration", {}).get("robot") != asdict(RobotConfig()):
            raise ValueError("Policy/current HockeyBot model mismatch")
        controllers.append("ppo")
    profile = profile or config.profile
    scenario = scenario or curriculum_stage(config, 1)
    max_steps = config.episode_steps if max_steps is None else max_steps
    result = dict(
        profile=profile,
        seed=seed,
        scenario=scenario,
        max_steps=max_steps,
        robots_per_team=config.robots_per_team,
        opponent_robots=config.opponent_robots,
        objective=config.objective,
        opponent="none"
        if config.opponent_robots == 0
        else "zero"
        if config.opponent == "zero"
        else "heuristic",
        controllers={},
    )
    for name in controllers:
        records = []
        for i in range(episodes):
            env = replace(config, profile=profile).make_env(scenario)
            env.max_steps = max_steps
            obs, _ = env.reset(seed=seed + i)
            bot = HeuristicHockeyBot(0, 0.92)
            reward = 0.0
            for tick in range(max_steps):
                actions = (
                    np.zeros((config.robots_per_team, 2), np.float32)
                    if name == "zero"
                    else (
                        bot.get_actions(env.engine, env.state)
                        if name == "heuristic"
                        else trainer.select_action(obs, True)[0]
                    )
                )
                red = np.zeros((config.opponent_robots, 2)) if config.opponent in ("none", "zero") else None
                obs, rew, term, trunc, info = env.step(actions, red)
                reward += rew
                if term or trunc:
                    break
            records.append(
                dict(
                    seed=seed + i,
                    goal=info["goal_scored"],
                    success=info["is_success"],
                    result=info["result"],
                    steps=tick + 1,
                    reward=reward,
                    shots=info["shots_blue"],
                    idle_fraction=info["idle_steps"] / (tick + 1),
                    tracking_lost_fraction=info["tracking_lost_steps"] / (tick + 1),
                )
            )
        result["controllers"][name] = dict(
            wins=sum(r["success"] for r in records),
            losses=sum(r["goal"] == -1 for r in records),
            timeouts=sum(r["result"] == "timeout" for r in records),
            success_rate=float(np.mean([r["success"] for r in records])),
            mean_reward=float(np.mean([r["reward"] for r in records])),
            records=records,
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--profile", choices=PROFILES)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--policy")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--scenario", choices=SCENARIOS)
    p.add_argument("--output", default="runs/benchmark.json")
    args = p.parse_args()
    if args.episodes <= 0 or (args.max_steps is not None and args.max_steps <= 0):
        p.error("Counts must be positive")
    result = benchmark(args.episodes, args.seed, args.profile, args.policy, args.max_steps, args.scenario)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    for name, data in result["controllers"].items():
        print(f"{name}: {data['wins']} wins / {data['losses']} losses / {data['timeouts']} timeouts")


if __name__ == "__main__":
    main()
