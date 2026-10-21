"""Evaluate frozen controllers on identical reserved initial seeds."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from core.constants import PROFILES, RobotConfig
from bots.hockey_bot import HeuristicHockeyBot
from rl.env import RoboCoreHockeyEnv, SCENARIOS
from rl.ppo_trainer import HockeyPPOTrainer


def benchmark(episodes=10, seed=10000, profile="micro-2024", policy=None, max_steps=900, scenario="faceoff"):
    torch.set_num_threads(1)
    controllers = ["zero", "heuristic"]
    trainer = None
    if policy:
        trainer = HockeyPPOTrainer()
        meta = trainer.load(policy)
        if meta.get("configuration", {}).get("arena") != PROFILES[profile].to_dict():
            raise ValueError("Policy/profile mismatch")
        if meta.get("configuration", {}).get("robot") != asdict(RobotConfig()):
            raise ValueError("Policy/current HockeyBot model mismatch")
        controllers.append("ppo")
    result = dict(profile=profile, seed=seed, scenario=scenario, max_steps=max_steps, controllers={})
    for name in controllers:
        records = []
        for i in range(episodes):
            env = RoboCoreHockeyEnv(profile=profile, max_steps=max_steps, scenario=scenario)
            obs, _ = env.reset(seed=seed + i)
            bot = HeuristicHockeyBot(0, 0.92)
            reward = 0.0
            for tick in range(max_steps):
                actions = (
                    np.zeros((3, 2), np.float32)
                    if name == "zero"
                    else (
                        bot.get_actions(env.engine, env.state)
                        if name == "heuristic"
                        else trainer.select_action(obs, True)[0]
                    )
                )
                obs, rew, term, trunc, info = env.step(actions)
                reward += rew
                if term or trunc:
                    break
            records.append(
                dict(
                    seed=seed + i,
                    goal=info["goal_scored"],
                    steps=tick + 1,
                    reward=reward,
                    shots=info["shots_blue"],
                    idle_fraction=info["idle_steps"] / (tick + 1),
                    tracking_lost_fraction=info["tracking_lost_steps"] / (tick + 1),
                )
            )
        result["controllers"][name] = dict(
            wins=sum(r["goal"] == 1 for r in records),
            losses=sum(r["goal"] == -1 for r in records),
            timeouts=sum(r["goal"] == 0 for r in records),
            mean_reward=float(np.mean([r["reward"] for r in records])),
            records=records,
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--profile", choices=PROFILES, default="micro-2024")
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--policy")
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--scenario", choices=SCENARIOS, default="faceoff")
    p.add_argument("--output", default="runs/benchmark.json")
    args = p.parse_args()
    if min(args.episodes, args.max_steps) <= 0:
        p.error("Counts must be positive")
    result = benchmark(args.episodes, args.seed, args.profile, args.policy, args.max_steps, args.scenario)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    for name, data in result["controllers"].items():
        print(f"{name}: {data['wins']} wins / {data['losses']} losses / {data['timeouts']} timeouts")


if __name__ == "__main__":
    main()
