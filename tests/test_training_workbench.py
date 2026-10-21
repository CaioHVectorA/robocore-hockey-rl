import json
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from rl.ppo_trainer import HockeyPPOTrainer
from rl.training import TrainingSession, read_json
from rl.training_config import TrainingConfig, preset_config, PRESET_LABELS
from rl.tasks import RewardConfig
from ui.training_session import TrainingProcess, list_sessions


@pytest.mark.parametrize("name", list(PRESET_LABELS))
def test_presets_create_actual_robot_counts_and_consistent_observations(name):
    cfg = preset_config(name)
    env = cfg.make_env()
    obs, _ = env.reset(seed=12)
    assert len(env.engine.blue_robots) == cfg.robots_per_team
    assert len(env.engine.red_robots) == cfg.opponent_robots
    assert obs.shape == (cfg.robots_per_team, 63)
    assert env.get_team_obs(1).shape == (cfg.opponent_robots, 63)
    obs, _, _, _, _ = env.step(np.zeros((cfg.robots_per_team, 2)))
    assert env.observation_space.contains(obs)
    assert env.get_team_obs(1).shape == (cfg.opponent_robots, 63)
    robot_bodies = {s.body for s in env.engine.space.shapes if s.collision_type == 2}
    assert len(robot_bodies) == cfg.robots_per_team + cfg.opponent_robots


def test_solo_reach_succeeds_without_goal_while_dribble_requires_goal():
    for name, expected in (("solo_reach", True), ("solo_goal", False)):
        env = preset_config(name).make_env()
        env.use_partial_vision = False
        env.reset(seed=2)
        env.engine.place_robot(env.engine.blue_robots[0], 0, 0, 0)
        env.engine.place_puck(9.5, 0)
        env.prev_potential, env.prev_approach = env._potential(), env._approach()
        _, reward, term, _, info = env.step([[0, 0]])
        assert info["goal_scored"] == 0
        assert term == expected and info["is_success"] == expected
        assert (reward > 0) == expected


def test_solo_goal_uses_real_goal_crossing_and_configured_rewards():
    cfg = replace(
        preset_config("solo_goal"),
        reward=RewardConfig(goal=7, progress=0, approach=0, time_penalty=0.1, energy_penalty=0),
    )
    env = cfg.make_env()
    env.reset(seed=3)
    _, reward, term, _, _ = env.step([[0, 0]])
    assert reward == pytest.approx(-0.1) and not term
    env.engine.place_puck(env.arena.half_length + env.arena.puck_radius + 0.2, 0)
    _, reward, term, _, info = env.step([[0, 0]])
    assert term and info["is_success"] and reward == pytest.approx(6.9)


def test_own_goal_cannot_count_as_reaching_success():
    cfg = replace(preset_config("solo_reach"), reward=RewardConfig(reach_distance=20))
    env = cfg.make_env()
    env.reset(seed=5)
    env.engine.place_robot(env.engine.blue_robots[0], -env.arena.half_length + 12, 0, 0)
    env.engine.place_puck(-env.arena.half_length - env.arena.puck_radius - 0.2, 0)
    _, _, term, _, info = env.step([[0, 0]])
    assert term and not info["is_success"] and info["result"] == "opponent_goal"


@pytest.mark.parametrize("blue,red", [(1, 3), (2, 1), (3, 0)])
def test_asymmetric_layouts_use_the_correct_actions_and_tracking_gate(blue, red):
    cfg = replace(
        TrainingConfig(),
        robots_per_team=blue,
        opponent_robots=red,
        opponent="none" if red == 0 else "selfplay",
    )
    env = cfg.make_env()
    env.reset(seed=4)
    env.step(np.zeros((blue, 2)), np.zeros((red, 2)))
    assert env.previous_red_actions.shape == (red, 2)
    with pytest.raises(ValueError, match="action count"):
        env.engine.step(np.zeros((blue, 2)), np.zeros((red + 1, 2)))


def test_experiment_round_trip_includes_rewards_and_ppo(tmp_path):
    cfg = replace(
        preset_config("solo_goal"),
        steps=1234,
        learning_rate=0.0001,
        reward=RewardConfig(goal=15, approach=1.7),
    )
    cfg.save(tmp_path / "exercise.json")
    assert TrainingConfig.load(tmp_path / "exercise.json") == cfg
    assert cfg.effective_steps == 1236
    for changes in ({"opponent_robots": 0}, {"steps": -1}, {"gamma": float("nan")}, {"envs": 100}):
        with pytest.raises(ValueError):
            replace(TrainingConfig(), **changes)


def test_pure_selfplay_never_uses_a_tactical_fallback_in_collection(tmp_path, monkeypatch):
    import rl.training
    from bots.hockey_bot import HeuristicHockeyBot

    def forbidden(*args, **kwargs):
        raise AssertionError("A heuristic was used in pure self-play")

    monkeypatch.setattr(HeuristicHockeyBot, "get_actions", forbidden)
    monkeypatch.setattr(rl.training, "evaluate_config", lambda *a, **k: dict(episodes=0, success_rate=0))
    cfg = replace(TrainingConfig(), steps=16, envs=2, rollout=4, epochs=1, eval_episodes=1)
    trainer = TrainingSession(cfg, tmp_path / "session").run()
    assert trainer.environment_steps == 16 and trainer.total_steps == 48
    assert trainer.updates_count == 2
    records = [json.loads(line) for line in (tmp_path / "session/metrics.jsonl").read_text().splitlines()]
    assert all(r["selfplay_pool"] >= 1 for r in records)
    assert len(read_json(tmp_path / "session/live.json")["robots"]) == 6


def test_retargeting_a_task_requires_transfer_and_keeps_a_new_optimizer(tmp_path):
    source = preset_config("solo_reach")
    trainer = HockeyPPOTrainer()
    trainer.environment_steps = 100
    path = tmp_path / "source.pt"
    trainer.save(path, dict(configuration=source.make_env().configuration()))
    target = replace(preset_config("solo_goal"), init_mode="resume", checkpoint=str(path))
    with pytest.raises(ValueError, match="transferência"):
        TrainingSession(target, tmp_path / "resume")._load(HockeyPPOTrainer(), target.make_env())
    target = replace(target, init_mode="transfer")
    fresh = HockeyPPOTrainer(lr=0.001)
    TrainingSession(target, tmp_path / "transfer")._load(fresh, target.make_env())
    assert fresh.environment_steps == fresh.total_steps == fresh.updates_count == 0
    assert fresh.optimizer.param_groups[0]["lr"] == 0.001
    obs, _ = source.make_env().reset(seed=9)
    np.testing.assert_array_equal(fresh.select_action(obs, True)[0], trainer.select_action(obs, True)[0])


def test_training_process_live_pause_resume_stop_and_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    process = TrainingProcess(root, tmp_path / "sessions")
    cfg = replace(
        preset_config("solo_reach"),
        steps=100000,
        envs=1,
        rollout=8,
        epochs=1,
        episode_seconds=1,
        speed=1,
        eval_episodes=1,
    )

    def until(predicate, seconds=12):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            process.poll(True)
            if process.status.get("state") == "error":
                pytest.fail(process.status.get("error", "Worker failed"))
            if predicate():
                return
            time.sleep(0.025)
        pytest.fail(f"Worker deadline: {process.status}")

    try:
        directory = process.start(cfg)
        until(lambda: process.frame is not None and process.frame["session_transitions"] >= 3)
        assert len(process.frame["robots"]) == 1
        process.send(paused=True)
        until(lambda: process.status.get("state") == "paused")
        stopped_at = process.status["session_transitions"]
        time.sleep(0.15)
        process.poll(True)
        assert process.status["session_transitions"] == stopped_at
        process.send(paused=False, speed=0)
        until(lambda: process.frame["session_transitions"] > stopped_at)
        process.send(stop=True, paused=False)
        until(lambda: process.status.get("state") == "stopped")
        process.process.wait(timeout=3)
        assert process.checkpoint.is_file()
        restored = HockeyPPOTrainer()
        meta = restored.load(process.checkpoint, resume=True)
        assert restored.updates_count >= 1 and restored.environment_steps > 0
        assert meta["configuration"]["opponent_robots"] == 0
        assert process.records and np.isfinite(process.records[-1]["loss"])
        from evaluate import benchmark

        result = benchmark(episodes=1, policy=process.checkpoint, max_steps=1)
        assert result["robots_per_team"] == 1 and result["opponent_robots"] == 0
        assert result["objective"] == "reach" and set(result["controllers"]) == {"zero", "heuristic", "ppo"}
        assert list_sessions(tmp_path / "sessions")[0]["path"] == directory
    finally:
        process.close()


def test_panel_mouse_preset_and_keyboard_edit_reach_worker_config(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame
    from ui.training_panel import TrainingPanel

    pygame.init()
    try:
        surface = pygame.Surface((500, 1000))
        fonts = {n: pygame.font.SysFont("DejaVu Sans", n) for n in (12, 14, 16, 18, 22)}
        previews, started = [], []
        panel = TrainingPanel(tmp_path, fonts, previews.append, lambda *args: None)
        panel.draw(surface, pygame.Rect(0, 0, 460, 960))
        # Second preset: genuinely one robot and no opponent.
        panel.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(330, 170), button=1))
        assert panel.config.opponent_robots == 0 and panel.config.robots_per_team == 1
        assert previews[-1] == panel.config
        panel.tab = "PPO"
        panel.draw(surface, pygame.Rect(0, 0, 460, 960))
        field = next(rect for rect, key, _ in panel.fields if key == "learning_rate")
        panel.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=field.center, button=1))
        panel.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_0, unicode="0.0001", mod=0))
        panel.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="", mod=0))
        monkeypatch.setattr(panel.worker, "start", started.append)
        panel.start()
        assert started[-1].learning_rate == 0.0001 and started[-1].opponent == "none"
    finally:
        pygame.quit()
