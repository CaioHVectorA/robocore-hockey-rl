"""Seeded PPO training with curriculum, evaluation and checkpoint metadata."""

from __future__ import annotations
import argparse
from dataclasses import asdict
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import torch
from core.constants import PROFILES, RobotConfig
from rl.env import RoboCoreHockeyEnv, SCENARIOS
from rl.ppo_trainer import HockeyPPOTrainer


def evaluate(trainer, profile="micro-2024", episodes=10, seed=10000, scenario="faceoff", max_steps=900):
    records = []
    for i in range(episodes):
        env = RoboCoreHockeyEnv(
            profile=profile, scenario=scenario, max_steps=max_steps, shaping_gamma=trainer.gamma
        )
        obs, _ = env.reset(seed=seed + i)
        total = 0.0
        for step in range(max_steps):
            actions, _, _ = trainer.select_action(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(actions)
            total += reward
            if term or trunc:
                break
        records.append(
            dict(
                seed=seed + i,
                goal=info["goal_scored"],
                steps=step + 1,
                reward=total,
                **{k: info[k] for k in ("shots_blue", "shots_red", "idle_steps", "tracking_lost_steps")},
            )
        )
    wins = sum(r["goal"] == 1 for r in records)
    losses = sum(r["goal"] == -1 for r in records)
    return dict(
        episodes=episodes,
        wins=wins,
        losses=losses,
        timeouts=episodes - wins - losses,
        win_rate=wins / episodes,
        mean_reward=float(np.mean([r["reward"] for r in records])),
        records=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--profile", choices=PROFILES, default="micro-2024")
    parser.add_argument("--scenario", choices=(*SCENARIOS, "curriculum"), default="curriculum")
    parser.add_argument("--opponent", choices=["heuristic", "mixed", "selfplay"], default="mixed")
    parser.add_argument("--save", default="checkpoints/hockey_v2.pt")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout", type=int, default=256)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=900)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--log", default="runs/latest.jsonl")
    args = parser.parse_args()
    if min(args.steps, args.rollout, args.envs, args.episode_steps, args.eval_episodes) <= 0:
        parser.error("Counts must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    trainer = HockeyPPOTrainer(seed=args.seed)
    loaded_metadata = {}
    if args.resume:
        loaded_metadata = trainer.load(args.resume, resume=not args.eval_only)
        if loaded_metadata.get("configuration", {}).get("arena") != PROFILES[args.profile].to_dict():
            parser.error("Checkpoint and selected arena differ")
        if loaded_metadata.get("configuration", {}).get("robot") != asdict(RobotConfig()):
            parser.error("Checkpoint and current HockeyBot model differ; start a new training run")
    if args.eval_only:
        if not args.resume:
            parser.error("--eval-only requires --resume")
        print(
            json.dumps(
                evaluate(trainer, args.profile, args.eval_episodes, max_steps=args.episode_steps), indent=2
            )
        )
        return
    # Each environment/robot has a distinct time trajectory. Batched rows stay in
    # a fixed env-major order until GAE has been computed.
    initial_stage = "shooting" if args.scenario == "curriculum" else args.scenario
    envs = [
        RoboCoreHockeyEnv(
            profile=args.profile,
            max_steps=args.episode_steps,
            scenario=initial_stage,
            shaping_gamma=trainer.gamma,
        )
        for _ in range(args.envs)
    ]
    observations = [env.reset(seed=args.seed + i)[0] for i, env in enumerate(envs)]
    opponent_rng = np.random.default_rng(args.seed + 12345)
    pool = deque(maxlen=6)
    opponents = [None] * args.envs

    def choose_opponent(index):
        env = envs[index]
        env.opponent_bot.difficulty = float(opponent_rng.uniform(0.55, 0.98))
        roll = opponent_rng.random()
        if pool and (args.opponent == "selfplay" or (args.opponent == "mixed" and roll < 0.35)):
            opponents[index] = pool[int(opponent_rng.integers(len(pool)))]
        elif args.opponent == "mixed" and roll > 0.90:
            opponents[index] = "zero"
        else:
            opponents[index] = None

    for i in range(args.envs):
        choose_opponent(i)
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    best = -1.0
    finished = 0
    with log.open("w", encoding="utf8") as stream:
        while finished < args.steps:
            progress = finished / args.steps
            stage = (
                (
                    "shooting"
                    if progress < 0.20
                    else "defending"
                    if progress < 0.35
                    else "random"
                    if progress < 0.55
                    else "faceoff"
                )
                if args.scenario == "curriculum"
                else args.scenario
            )
            obs = np.concatenate(observations)
            actions, lp, values = trainer.select_action(obs)
            next_observations = []
            rewards = []
            terms = []
            truncs = []
            for i, env in enumerate(envs):
                opponent = opponents[i]
                red = None
                if isinstance(opponent, str):
                    red = np.zeros((3, 2))
                elif opponent is not None:
                    with torch.no_grad():
                        red = opponent(torch.from_numpy(env.get_team_obs(1)))[0].tanh().numpy()
                nxt, rew, term, trunc, info = env.step(actions[i * 3 : (i + 1) * 3], custom_red_actions=red)
                next_observations.append(nxt)
                rewards.extend(info["individual_rewards"])
                terms.extend([term] * 3)
                truncs.extend([trunc] * 3)
            trainer.store_transition(
                obs,
                actions,
                lp,
                rewards,
                values,
                terms,
                truncs,
                next_obs=np.concatenate(next_observations),
                env_count=args.envs,
            )
            for i, env in enumerate(envs):
                if terms[i * 3] or truncs[i * 3]:
                    choose_opponent(i)
                observations[i] = (
                    env.reset(options={"scenario": stage})[0]
                    if terms[i * 3] or truncs[i * 3]
                    else next_observations[i]
                )
            finished += args.envs
            if trainer.buffer_steps >= args.rollout or finished >= args.steps:
                trainer.update()
                if args.opponent != "heuristic" and (trainer.updates_count % 5 == 0 or not pool):
                    frozen = deepcopy(trainer.net).eval()
                    for param in frozen.parameters():
                        param.requires_grad_(False)
                    pool.append(frozen)
                metadata = dict(
                    configuration=envs[0].configuration(),
                    seed=args.seed,
                    stage=stage,
                    environment_transitions=finished,
                    wall_seconds=time.perf_counter() - started,
                    opponent=args.opponent,
                )
                record = dict(
                    environment_transitions=finished,
                    updates=trainer.updates_count,
                    stage=stage,
                    **trainer.metrics,
                )
                trainer.save(args.save, metadata)
                if trainer.updates_count % 10 == 0 or finished >= args.steps:
                    score = evaluate(trainer, args.profile, args.eval_episodes, max_steps=args.episode_steps)
                    record["evaluation"] = score
                    if score["win_rate"] > best:
                        best = score["win_rate"]
                        trainer.save(
                            str(Path(args.save).with_suffix(".best.pt")), dict(metadata, evaluation=score)
                        )
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(
                    f"{finished:>8}/{args.steps} env steps | update {trainer.updates_count} | {stage} | KL {trainer.metrics['approx_kl']:.4f}",
                    flush=True,
                )
    print(f"Saved {args.save}; {finished} transitions; {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
