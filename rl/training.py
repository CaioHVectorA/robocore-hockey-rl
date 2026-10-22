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
from core.file_io import atomic_destination, sharing_violation
from rl.ppo_trainer import HockeyPPOTrainer
from rl.curriculum import SoloCurriculum, SOLO_STAGES, ProgressGuard
from rl.tasks import RewardConfig


def atomic_json(path, data, best_effort=False):
    try:
        with atomic_destination(path, attempts=1 if best_effort else 7) as temp:
            temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf8")
        return True
    except OSError as exc:
        # Telemetry can drop a frame; it must never abort PPO because a reader holds it.
        if best_effort and sharing_violation(exc):
            return False
        raise


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf8"))
    except (OSError, ValueError):
        return default


def curriculum_stage(config, progress):
    if config.scenario == "solo_curriculum":
        # Evaluation always uses full search, even before the training advances.
        return SOLO_STAGES[-1] if progress >= 1 else SOLO_STAGES[0]
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
        robot_speeds=[math.hypot(r.vx, r.vy) for r in e.blue_robots],
        distance_to_puck=min(math.hypot(r.x - e.puck.x, r.y - e.puck.y) for r in e.blue_robots),
        reward_components=dict(getattr(env, "last_reward_components", {})),
        episode_reward_components=dict(getattr(env, "episode_reward_components", {})),
        tracking=[(r.x, r.y, r.vx, r.vy) for r in env.state.robots.values() if r.valid(env.state.timestamp)],
        **extra,
    )


def evaluate_config(
    trainer,
    config,
    episodes=None,
    seed=10000,
    should_stop=lambda: False,
    opponent_net=None,
    opponent_name=None,
    scenarios=None,
):
    """Reserved seeds, task-matched layout; adversary is a fixed tactical bot."""
    records = []
    for i in range(episodes or config.eval_episodes):
        if should_stop():
            break
        scenario = scenarios[i % len(scenarios)] if scenarios else curriculum_stage(config, 1)
        env = config.make_env(scenario)
        obs, _ = env.reset(seed=seed + i)
        total = 0.0
        for step in range(config.episode_steps):
            if should_stop():
                break
            actions, _, _ = trainer.select_action(obs, deterministic=True)
            red = np.zeros((config.opponent_robots, 2)) if config.opponent in ("none", "zero") else None
            if opponent_net is not None:
                with torch.no_grad():
                    distribution, _ = opponent_net.distribution(torch.from_numpy(env.get_team_obs(1)))
                    red = distribution.mean.tanh().numpy()
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
                        scenario=scenario,
                        **participation_record(info),
                    )
                )
                break
    wins = sum(r["success"] for r in records)
    losses = sum(r["goal"] == -1 for r in records)
    return dict(
        episodes=len(records),
        scenario="+".join(scenarios) if scenarios else curriculum_stage(config, 1),
        objective=config.objective,
        robots_per_team=config.robots_per_team,
        opponent_robots=config.opponent_robots,
        wins=wins,
        losses=losses,
        timeouts=sum(r["result"] == "timeout" for r in records),
        win_rate=wins / max(1, len(records)),
        success_rate=wins / max(1, len(records)),
        point_score=(wins + 0.5 * sum(r["result"] == "timeout" for r in records)) / max(1, len(records)),
        mean_reward=float(np.mean([r["reward"] for r in records])) if records else 0.0,
        opponent=opponent_name
        or ("none" if config.opponent_robots == 0 else "zero" if config.opponent == "zero" else "heuristic"),
        records=records,
        complete=len(records) == (episodes or config.eval_episodes),
        **participation_summary(records),
    )


def evaluate_opponent_suite(trainer, config, anchor, should_stop=lambda: False):
    """Fixed opponents and layouts provide a stable test outside the changing league."""
    results = {}
    for name, net in (("heuristic", None), ("initial_policy", anchor)):
        if should_stop():
            break
        results[name] = evaluate_config(
            trainer,
            config,
            should_stop=should_stop,
            opponent_net=net,
            opponent_name=name,
            scenarios=("faceoff", "random"),
        )
    episodes = sum(row["episodes"] for row in results.values())
    wins = sum(row["wins"] for row in results.values())
    losses = sum(row["losses"] for row in results.values())
    timeouts = sum(row["timeouts"] for row in results.values())
    complete = len(results) == 2 and all(row["episodes"] == config.eval_episodes for row in results.values())
    return dict(
        episodes=episodes,
        wins=wins,
        losses=losses,
        timeouts=timeouts,
        success_rate=wins / max(1, episodes),
        win_rate=wins / max(1, episodes),
        point_score=(wins + timeouts * 0.5) / max(1, episodes),
        complete=complete,
        opponent="fixed_suite",
        scenario="faceoff+random",
        by_opponent=results,
        **participation_summary([r for row in results.values() for r in row["records"]]),
    )


def participation_record(info):
    return dict(
        active_success=info["active_success"],
        blue_participated=info["blue_participated"],
        goal_last_touch_team=info["goal_last_touch_team"],
        touches_blue=info["touches_blue"],
        idle_fraction=info["idle_fraction"],
        tracking_lost_fraction=info["tracking_lost_fraction"],
        mean_distance_to_puck=info["mean_distance_to_puck"],
    )


def participation_summary(records):
    n = max(1, len(records))
    return dict(
        active_success_rate=sum(r.get("active_success", False) for r in records) / n,
        touch_episode_rate=sum(r.get("blue_participated", False) for r in records) / n,
        mean_idle_fraction=sum(r.get("idle_fraction", 0) for r in records) / n,
        mean_distance_to_puck=sum(r.get("mean_distance_to_puck", 0) for r in records) / n,
        wins_without_blue_touch=sum(
            r.get("success", False) and not r.get("blue_participated", False) for r in records
        ),
    )


class TrainingSession:
    def __init__(self, config, directory, save_path=None, reporter=None, publish_live=True, deadline=None):
        self.config = config
        self.directory = Path(directory)
        self.save_path = Path(save_path) if save_path else self.directory / "latest.pt"
        self.stop_requested = False
        self.paused = False
        self.speed = config.speed
        self.finished = 0
        self.started = time.perf_counter()
        self.last_live = self.last_control = -math.inf
        self.phase = "starting"
        self.episode_records = deque(maxlen=100)
        self.episodes = 0
        self.episode_number = 1
        self.best = -math.inf
        self.last_evaluation = None
        self.dropped_frames = 0
        self.reporter = reporter
        self.publish_live = publish_live
        self.last_progress = -math.inf
        self.updates = 0
        self.curriculum = SoloCurriculum(config) if config.scenario == "solo_curriculum" else None
        self.deadline = deadline
        self.stop_reason = None
        self.selfplay_pool = None
        self.selfplay_anchor = None
        self.progress_guard = ProgressGuard(config.stagnation_patience)

    def control(self):
        now = time.perf_counter()
        if now - self.last_control >= 0.08:
            data = read_json(self.directory / "control.json", {})
            if data.get("stop") and not self.stop_requested:
                self.stop_reason = "requested"
            self.stop_requested |= bool(data.get("stop", False))
            if self.deadline is not None and time.time() >= self.deadline and not self.stop_requested:
                self.stop_requested, self.stop_reason = True, "time_limit"
            self.paused = bool(data.get("paused", False))
            speed = data.get("speed", self.speed)
            if isinstance(speed, (int, float)) and math.isfinite(speed) and 0 <= speed <= 1000:
                self.speed = speed
            self.last_control = now
        return self.stop_requested

    def status(self, phase, **extra):
        self.phase = phase
        self.updates = extra.pop("updates", self.updates)
        data = dict(
            state=phase,
            session_transitions=self.finished,
            target=self.config.effective_steps,
            wall_seconds=time.perf_counter() - self.started,
            updated_at=time.time(),
            checkpoint=str(self.save_path),
            dropped_frames=self.dropped_frames,
            updates=self.updates,
            curriculum=self.curriculum.state() if self.curriculum else None,
            deadline=self.deadline,
            stop_reason=self.stop_reason,
            progress_guard=self.progress_guard.state(),
            **extra,
        )
        atomic_json(self.directory / "status.json", data)
        if self.reporter:
            self.reporter("progress", data)
            if phase == "error":
                self.reporter("error", data)

    def progress(self):
        if time.perf_counter() - self.last_progress >= 1:
            self.status(self.phase)
            self.last_progress = time.perf_counter()

    def publish(self, env, trainer, force=False, **extra):
        if not self.publish_live:
            return
        now = time.perf_counter()
        # At 1x, each physical control step is already paced to 30 Hz. Publish it
        # directly so sub-millisecond inference jitter cannot skip alternate steps.
        if force or self.speed == 1 or now - self.last_live >= 1 / 30:
            published = atomic_json(
                self.directory / "live.json",
                frame_from_env(
                    env,
                    session_transitions=self.finished,
                    updates=trainer.updates_count,
                    state=self.phase,
                    episode=self.episode_number,
                    timestamp=time.time(),
                    speed=self.speed,
                    transitions_per_second=self.finished / max(0.001, now - self.started),
                    curriculum=self.curriculum.state() if self.curriculum else None,
                    **extra,
                ),
                best_effort=True,
            )
            self.dropped_frames += int(not published)
            self.last_live = now

    def metadata(self, env, trainer):
        return dict(
            configuration=env.configuration(),
            training_config=self.config.to_dict(),
            seed=self.config.seed,
            stage=env.scenario,
            environment_transitions=trainer.environment_steps,
            session_transitions=self.finished,
            wall_seconds=time.perf_counter() - self.started,
            opponent=self.config.opponent,
            curriculum=self.curriculum.state() if self.curriculum else None,
            best_evaluation_score=self.best if math.isfinite(self.best) else None,
            evaluation_metric="active_success_rate-v1",
            progress_guard=self.progress_guard.state(),
            selfplay_state=dict(
                anchor=self.selfplay_anchor.state_dict(),
                pool=[net.state_dict() for net in self.selfplay_pool],
            )
            if self.selfplay_anchor is not None and self.config.opponent in ("selfplay", "mixed")
            else None,
        )

    def _load(self, trainer, env):
        cfg = self.config
        if cfg.init_mode == "fresh":
            return {}
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
            if self.curriculum:
                for key in (
                    "curriculum_success_threshold",
                    "curriculum_min_episodes",
                    "curriculum_window",
                    "curriculum_eval_gate",
                ):
                    if previous_config.get(key, False if key == "curriculum_eval_gate" else None) != getattr(
                        cfg, key
                    ):
                        raise ValueError("Critérios do currículo diferentes dos pesos; use transferência")
            for key in (
                "robots_per_team",
                "opponent_robots",
                "objective",
                "reward",
                "simulation",
                "spawn_distance",
                "random_heading",
            ):
                old_value = (
                    asdict(RewardConfig(**old[key])) if key == "reward" and key in old else old.get(key)
                )
                if old_value != env.configuration().get(key):
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
        return meta

    def run(self):
        cfg = self.config
        self.directory.mkdir(parents=True, exist_ok=True)
        cfg.save(self.directory / "config.json")
        self.status("starting")
        old_handlers = {}

        # Also handle window close/OS termination at the next worker boundary.
        def stop(signum, frame):
            self.stop_requested = True
            self.stop_reason = "interrupt"

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
        loaded_meta = self._load(trainer, envs[0])
        if self.curriculum and cfg.init_mode == "resume":
            self.curriculum.restore(loaded_meta.get("curriculum") or {})
            stage = self.curriculum.stage
            for i, env in enumerate(envs):
                env.scenario = stage
                observations[i] = env.reset(seed=cfg.seed + i)[0]
        pool = deque(maxlen=cfg.selfplay_pool_size)
        self.selfplay_pool = pool
        self.selfplay_anchor = deepcopy(trainer.net).eval()
        self.selfplay_anchor.requires_grad_(False)
        if cfg.init_mode == "resume":
            previous_best = loaded_meta.get("best_evaluation_score")
            if (
                loaded_meta.get("evaluation_metric") == "active_success_rate-v1"
                and isinstance(previous_best, (int, float))
                and math.isfinite(previous_best)
            ):
                self.best = previous_best
            self.progress_guard.restore(loaded_meta.get("progress_guard") or {})
            saved_league = loaded_meta.get("selfplay_state")
            if saved_league:
                self.selfplay_anchor.load_state_dict(saved_league["anchor"])
                for state in saved_league["pool"][-cfg.selfplay_pool_size :]:
                    net = deepcopy(self.selfplay_anchor)
                    net.load_state_dict(state)
                    pool.append(net)
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
                opponents[index] = (
                    self.selfplay_anchor
                    if rng.random() < cfg.selfplay_anchor_probability
                    else pool[int(rng.integers(len(pool)))]
                )
            else:
                opponents[index] = "zero" if cfg.opponent == "mixed" and roll > 0.90 else None

        if not pool:
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
                tick = time.perf_counter()
                if self.control():
                    break
                if self.paused:
                    self.status("paused", updates=trainer.updates_count)
                    self.publish(envs[0], trainer, stage=stage, episode_return=float(returns[0]))
                    time.sleep(0.04)
                    continue
                if self.phase == "paused":
                    self.status("running", updates=trainer.updates_count)
                stage = (
                    self.curriculum.stage
                    if self.curriculum
                    else curriculum_stage(cfg, self.finished / cfg.effective_steps)
                )
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
                            initialization=env.initialization,
                            reward_components=info["episode_reward_components"],
                            **participation_record(info),
                        )
                        self.episodes += 1
                        self.episode_records.append(record)
                        self.progress_guard.record(record)
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
                if self.curriculum:
                    self.curriculum.update(self.episode_records)
                    stage = self.curriculum.stage
                if self.progress_guard.exhausted(stage):
                    self.stop_requested, self.stop_reason = True, "stagnation"
                self.finished += cfg.envs
                self.progress()
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
                while self.speed > 0 and time.perf_counter() - tick < envs[0].engine.dt / self.speed:
                    if self.control() or self.paused:
                        break
                    time.sleep(
                        max(0.0, min(0.02, envs[0].engine.dt / self.speed - (time.perf_counter() - tick)))
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
        self.status("updating", updates=trainer.updates_count)
        trainer.update(epochs=cfg.epochs, batch_size=cfg.batch_size)
        if trainer.updates_count % cfg.snapshot_every == 0:
            snapshot()
        validation_record = None
        if (
            evaluate
            and not self.control()
            and self.curriculum
            and self.curriculum.eval_gate
            and self.curriculum.ready
            and (
                self.curriculum.validation is None
                or trainer.updates_count - self.curriculum.validation.get("updates", 0) >= cfg.eval_every
            )
        ):
            # Validate when mastery is observed, rather than overtraining an easy
            # stage until the next full-search evaluation interval.
            self.status("evaluating", updates=trainer.updates_count)
            current_stage = self.curriculum.stage
            validation = evaluate_config(
                trainer, cfg, seed=30000, should_stop=self.control, scenarios=(current_stage,)
            )
            promoted = self.curriculum.validate(
                current_stage, dict(validation, updates=trainer.updates_count)
            )
            validation_record = dict(stage=current_stage, promoted=promoted, **validation)
        metadata = self.metadata(env, trainer)
        self.status("saving", updates=trainer.updates_count)
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
            wall_seconds=time.perf_counter() - self.started,
            selfplay_pool=len(pool),
            curriculum=self.curriculum.state() if self.curriculum else None,
            mean_reward_components={
                key: float(np.mean([r["reward_components"][key] for r in recent]))
                for key in recent[0]["reward_components"]
            }
            if recent
            else {},
            **trainer.metrics,
            participation=participation_summary(recent),
            progress_guard=self.progress_guard.state(),
        )
        if validation_record is not None:
            record["curriculum_validation"] = validation_record
        if (
            evaluate
            and not self.control()
            and (trainer.updates_count % cfg.eval_every == 0 or self.finished >= cfg.effective_steps)
        ):
            self.status("evaluating", updates=trainer.updates_count)
            score = (
                evaluate_opponent_suite(trainer, cfg, self.selfplay_anchor, self.control)
                if cfg.evaluation_suite
                else evaluate_config(trainer, cfg, should_stop=self.control)
            )
            self.last_evaluation = score
            record["evaluation"] = score
            quality = score["active_success_rate"]
            if score["episodes"] and score.get("complete", True) and quality > max(0.0, self.best):
                self.best = quality
                trainer.save(
                    self.save_path.with_suffix(".best.pt"),
                    dict(self.metadata(env, trainer), evaluation=score, best_evaluation_score=self.best),
                )
            # Persist the selection score immediately, so a restart just after
            # evaluation cannot forget which quality threshold has been reached.
            trainer.save(self.save_path, dict(self.metadata(env, trainer), evaluation=score))
        log.write(json.dumps(record, allow_nan=False) + "\n")
        log.flush()
        self.status("running", updates=trainer.updates_count, metrics=record)
        if self.reporter:
            self.reporter("metrics", record)
        else:
            print(
                f"{self.finished}/{cfg.effective_steps} env steps | update {trainer.updates_count} | {env.scenario} | KL {trainer.metrics['approx_kl']:.4f}",
                flush=True,
            )
