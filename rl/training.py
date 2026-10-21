"""PPO worker with real training frames, cooperative controls and saved sessions."""

from collections import deque
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
from core.constants import RobotConfig
from rl.ppo_trainer import HockeyPPOTrainer


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf8")
    temp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf8"))
    except (OSError, ValueError):
        return default


def curriculum_stage(config, progress):
    if config.scenario != "curriculum":
        return config.scenario
    return (
        "shooting"
        if progress < 0.20
        else "defending"
        if progress < 0.35
        else "random"
        if progress < 0.55
        else "faceoff"
    )


def frame_from_env(env, **extra):
    e = env.engine
    return dict(
        robots=[(r.x, r.y, r.theta, r.team, r.role_id) for r in e.all_robots],
        puck=[e.puck.x, e.puck.y],
        time=e.episode_time,
        score_blue=e.score_blue,
        score_red=e.score_red,
        actions=env.previous_actions.tolist(),
        tracking=[(r.x, r.y, r.vx, r.vy) for r in env.state.robots.values() if r.valid(env.state.timestamp)],
        **extra,
    )


def evaluate_config(trainer, config, episodes=None, seed=10000, should_stop=lambda: False):
    """Reserved seeds, task-matched layout; adversary is a fixed tactical bot."""
    records = []
    for i in range(episodes or config.eval_episodes):
        if should_stop():
            break
        env = config.make_env(curriculum_stage(config, 1))
        obs, _ = env.reset(seed=seed + i)
        total = 0.0
        for step in range(config.episode_steps):
            if should_stop():
                break
            actions, _, _ = trainer.select_action(obs, deterministic=True)
            red = np.zeros((config.opponent_robots, 2)) if config.opponent in ("none", "zero") else None
            obs, reward, term, trunc, info = env.step(actions, red)
            total += reward
            if term or trunc:
                records.append(
                    dict(
                        seed=seed + i,
                        success=info["is_success"],
                        result=info["result"],
                        goal=info["goal_scored"],
                        steps=step + 1,
                        reward=total,
                    )
                )
                break
    wins = sum(r["success"] for r in records)
    losses = sum(r["goal"] == -1 for r in records)
    return dict(
        episodes=len(records),
        wins=wins,
        losses=losses,
        timeouts=sum(r["result"] == "timeout" for r in records),
        win_rate=wins / max(1, len(records)),
        success_rate=wins / max(1, len(records)),
        mean_reward=float(np.mean([r["reward"] for r in records])) if records else 0.0,
        opponent="none"
        if config.opponent_robots == 0
        else "zero"
        if config.opponent == "zero"
        else "heuristic",
        records=records,
    )


class TrainingSession:
    def __init__(self, config, directory, save_path=None):
        self.config = config
        self.directory = Path(directory)
        self.save_path = Path(save_path) if save_path else self.directory / "latest.pt"
        self.stop_requested = False
        self.paused = False
        self.speed = config.speed
        self.finished = 0
        self.started = time.monotonic()
        self.last_live = self.last_control = -math.inf
        self.phase = "starting"
        self.episode_records = deque(maxlen=100)
        self.episodes = 0
        self.episode_number = 1
        self.best = -math.inf
        self.last_evaluation = None

    def control(self):
        now = time.monotonic()
        if now - self.last_control >= 0.08:
            data = read_json(self.directory / "control.json", {})
            self.stop_requested |= bool(data.get("stop", False))
            self.paused = bool(data.get("paused", False))
            speed = data.get("speed", self.speed)
            if isinstance(speed, (int, float)) and math.isfinite(speed) and 0 <= speed <= 1000:
                self.speed = speed
            self.last_control = now
        return self.stop_requested

    def status(self, phase, **extra):
        self.phase = phase
        atomic_json(
            self.directory / "status.json",
            dict(
                state=phase,
                session_transitions=self.finished,
                target=self.config.effective_steps,
                wall_seconds=time.monotonic() - self.started,
                updated_at=time.time(),
                checkpoint=str(self.save_path),
                **extra,
            ),
        )

    def publish(self, env, trainer, force=False, **extra):
        now = time.monotonic()
        if force or now - self.last_live >= 0.10:
            atomic_json(
                self.directory / "live.json",
                frame_from_env(
                    env,
                    session_transitions=self.finished,
                    updates=trainer.updates_count,
                    state=self.phase,
                    episode=self.episode_number,
                    timestamp=time.time(),
                    **extra,
                ),
            )
            self.last_live = now

    def metadata(self, env, trainer):
        return dict(
            configuration=env.configuration(),
            training_config=self.config.to_dict(),
            seed=self.config.seed,
            stage=env.scenario,
            environment_transitions=trainer.environment_steps,
            session_transitions=self.finished,
            wall_seconds=time.monotonic() - self.started,
            opponent=self.config.opponent,
        )

    def _load(self, trainer, env):
        cfg = self.config
        if cfg.init_mode == "fresh":
            return
        meta = trainer.load(cfg.checkpoint, resume=cfg.init_mode == "resume")
        old = meta.get("configuration", {})
        if old.get("arena") != env.arena.to_dict() or old.get("robot") != asdict(RobotConfig()):
            raise ValueError("Checkpoint and current arena/HockeyBot model differ")
        if cfg.init_mode == "resume":
            previous_config = meta.get("training_config", {})
            for key in ("scenario", "opponent", "domain_randomization"):
                if previous_config.get(key) != getattr(cfg, key):
                    raise ValueError(
                        f"Retomada incompatível em {key}; use transferência para mudar o exercício"
                    )
            for key in (
                "robots_per_team",
                "opponent_robots",
                "objective",
                "reward",
                "simulation",
                "spawn_distance",
                "random_heading",
            ):
                if old.get(key) != env.configuration().get(key):
                    raise ValueError(
                        f"Retomada incompatível em {key}; use transferência para mudar o exercício"
                    )
            expected = dict(
                lr=cfg.learning_rate,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                clip_eps=cfg.clip_eps,
                target_kl=cfg.target_kl,
                entropy_coef=cfg.entropy_coef,
            )
            if any(getattr(trainer, key) != value for key, value in expected.items()):
                raise ValueError(
                    "Hiperparâmetros diferentes dos pesos; use transferência ou restaure a configuração"
                )
        else:
            trainer.total_steps = trainer.environment_steps = trainer.updates_count = 0

    def run(self):
        cfg = self.config
        self.directory.mkdir(parents=True, exist_ok=True)
        cfg.save(self.directory / "config.json")
        self.status("starting")
        old_handlers = {}

        # Also handle window close/OS termination at the next worker boundary.
        def stop(signum, frame):
            self.stop_requested = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, stop)
        try:
            return self._run()
        except Exception as exc:
            self.status("error", error=str(exc), detail=traceback.format_exc()[-4000:])
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)

    def _run(self):
        cfg = self.config
        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        trainer = HockeyPPOTrainer(
            lr=cfg.learning_rate,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_eps=cfg.clip_eps,
            target_kl=cfg.target_kl,
            entropy_coef=cfg.entropy_coef,
            seed=cfg.seed,
        )
        stage = curriculum_stage(cfg, 0)
        envs = [cfg.make_env(stage) for _ in range(cfg.envs)]
        observations = [env.reset(seed=cfg.seed + i)[0] for i, env in enumerate(envs)]
        self._load(trainer, envs[0])
        pool = deque(maxlen=6)
        opponents = [None] * cfg.envs
        rng = np.random.default_rng(cfg.seed + 12345)
        returns = np.zeros(cfg.envs)

        def snapshot():
            frozen = deepcopy(trainer.net).eval()
            frozen.requires_grad_(False)
            pool.append(frozen)

        def choose(index):
            envs[index].opponent_bot.difficulty = float(
                np.clip(cfg.opponent_difficulty + rng.uniform(-0.15, 0.1), 0, 1)
            )
            roll = rng.random()
            if cfg.opponent in ("none", "zero"):
                opponents[index] = "zero"
            elif cfg.opponent == "selfplay" or (cfg.opponent == "mixed" and roll < 0.35):
                opponents[index] = pool[int(rng.integers(len(pool)))]
            else:
                opponents[index] = "zero" if cfg.opponent == "mixed" and roll > 0.90 else None

        snapshot()  # Pure self-play starts against a frozen copy from episode 1.
        for i in range(cfg.envs):
            choose(i)
        trainer.save(self.save_path, self.metadata(envs[0], trainer))
        self.status("running")
        self.publish(envs[0], trainer, True, stage=stage, episode_return=0.0)
        with (
            (self.directory / "metrics.jsonl").open("w", encoding="utf8") as log,
            (self.directory / "episodes.jsonl").open("w", encoding="utf8") as episodes_log,
        ):
            while self.finished < cfg.effective_steps:
                tick = time.monotonic()
                if self.control():
                    break
                if self.paused:
                    self.status("paused", updates=trainer.updates_count)
                    self.publish(envs[0], trainer, stage=stage, episode_return=float(returns[0]))
                    time.sleep(0.04)
                    continue
                if self.phase == "paused":
                    self.status("running", updates=trainer.updates_count)
                stage = curriculum_stage(cfg, self.finished / cfg.effective_steps)
                obs = np.concatenate(observations)
                actions, lp, values = trainer.select_action(obs)
                next_obs, rewards, terms, truncs, ended = [], [], [], [], []
                for i, env in enumerate(envs):
                    opponent = opponents[i]
                    red = None
                    if isinstance(opponent, str):
                        red = np.zeros((cfg.opponent_robots, 2))
                    elif opponent is not None:
                        with torch.no_grad():
                            dist, _ = opponent.distribution(torch.from_numpy(env.get_team_obs(1)))
                            red = dist.sample().tanh().numpy()
                    n = cfg.robots_per_team
                    nxt, rew, term, trunc, info = env.step(actions[i * n : (i + 1) * n], red)
                    returns[i] += rew
                    next_obs.append(nxt)
                    rewards.extend(info["individual_rewards"])
                    terms.extend([term] * n)
                    truncs.extend([trunc] * n)
                    ended.append(term or trunc)
                    if term or trunc:
                        record = dict(
                            environment=i,
                            episode=self.episodes + 1,
                            reward=float(returns[i]),
                            success=info["is_success"],
                            result=info["result"],
                            seconds=info["episode_seconds"],
                            stage=env.scenario,
                        )
                        self.episodes += 1
                        self.episode_records.append(record)
                        episodes_log.write(json.dumps(record) + "\n")
                        episodes_log.flush()
                trainer.store_transition(
                    obs,
                    actions,
                    lp,
                    rewards,
                    values,
                    terms,
                    truncs,
                    next_obs=np.concatenate(next_obs),
                    env_count=cfg.envs,
                )
                self.finished += cfg.envs
                self.publish(
                    envs[0],
                    trainer,
                    ended[0],
                    stage=envs[0].scenario,
                    episode_return=float(returns[0]),
                    opponent="snapshot"
                    if not isinstance(opponents[0], (str, type(None)))
                    else "zero"
                    if isinstance(opponents[0], str)
                    else "heuristic",
                )
                for i, env in enumerate(envs):
                    if ended[i]:
                        if i == 0:
                            self.episode_number += 1
                        env.scenario = stage
                        observations[i] = env.reset()[0]
                        returns[i] = 0
                        choose(i)
                    else:
                        observations[i] = next_obs[i]
                if trainer.buffer_steps >= cfg.rollout or self.finished >= cfg.effective_steps:
                    self._update(trainer, envs[0], pool, snapshot, log)
                # Speed controls the environment shown, independent of number of workers.
                while self.speed > 0 and time.monotonic() - tick < envs[0].engine.dt / self.speed:
                    if self.control() or self.paused:
                        break
                    time.sleep(
                        max(0.0, min(0.02, envs[0].engine.dt / self.speed - (time.monotonic() - tick)))
                    )
            if trainer.buffer_steps:
                self._update(trainer, envs[0], pool, snapshot, log, evaluate=False)
        self.status("saving", updates=trainer.updates_count)
        trainer.save(self.save_path, self.metadata(envs[0], trainer))
        state = "stopped" if self.stop_requested else "completed"
        self.status(state, updates=trainer.updates_count, evaluation=self.last_evaluation)
        self.publish(envs[0], trainer, True, stage=envs[0].scenario, episode_return=float(returns[0]))
        return trainer

    def _update(self, trainer, env, pool, snapshot, log, evaluate=True):
        cfg = self.config
        trainer.update(epochs=cfg.epochs, batch_size=cfg.batch_size)
        if trainer.updates_count % cfg.snapshot_every == 0:
            snapshot()
        metadata = self.metadata(env, trainer)
        trainer.save(self.save_path, metadata)
        recent = list(self.episode_records)[-20:]
        record = dict(
            session_transitions=self.finished,
            environment_transitions=trainer.environment_steps,
            updates=trainer.updates_count,
            stage=env.scenario,
            episodes=self.episodes,
            mean_episode_reward=float(np.mean([r["reward"] for r in recent])) if recent else None,
            success_rate=float(np.mean([r["success"] for r in recent])) if recent else None,
            samples=trainer.total_steps,
            wall_seconds=time.monotonic() - self.started,
            selfplay_pool=len(pool),
            **trainer.metrics,
        )
        if (
            evaluate
            and not self.control()
            and (trainer.updates_count % cfg.eval_every == 0 or self.finished >= cfg.effective_steps)
        ):
            self.status("evaluating", updates=trainer.updates_count)
            score = evaluate_config(trainer, cfg, should_stop=self.control)
            self.last_evaluation = score
            record["evaluation"] = score
            if score["episodes"] and score["success_rate"] > self.best:
                self.best = score["success_rate"]
                trainer.save(self.save_path.with_suffix(".best.pt"), dict(metadata, evaluation=score))
        log.write(json.dumps(record, allow_nan=False) + "\n")
        log.flush()
        self.status("running", updates=trainer.updates_count, metrics=record)
        print(
            f"{self.finished}/{cfg.effective_steps} env steps | update {trainer.updates_count} | {env.scenario} | KL {trainer.metrics['approx_kl']:.4f}",
            flush=True,
        )
