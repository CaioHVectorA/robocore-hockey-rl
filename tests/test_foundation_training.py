from dataclasses import asdict, replace
import io
import math
import numpy as np
import pytest
from rl.curriculum import SoloCurriculum, ProgressGuard
from rl.env import RoboCoreHockeyEnv
from rl.ppo_trainer import HockeyPPOTrainer
from rl.tasks import RewardConfig
from rl.training import TrainingSession, read_json, participation_record, participation_summary
from rl.training_config import preset_config


def test_alignment_potential_rewards_behind_and_facing_goal_not_ahead_or_backwards():
    env = preset_config("solo_foundation").make_env("solo_carry")
    env.reset(seed=0)
    r = env.engine.blue_robots[0]
    env.engine.place_robot(r, -15, 0, 0)
    env.engine.place_puck(0, 0)
    assert env._alignment() == pytest.approx(0.75)
    env.engine.place_robot(r, -15, 0, math.pi)
    assert env._alignment() == 0
    env.engine.place_robot(r, 15, 0, math.pi)
    assert env._alignment() == 0


def test_contact_bonus_cannot_be_farmed_and_resets_only_with_episode(monkeypatch):
    env = RoboCoreHockeyEnv(
        1, opponent_robots=0, reward_config=RewardConfig(first_touch=1), use_partial_vision=False
    )
    env.reset(seed=0)
    original = env.engine.step

    def contact(*args):
        info = original(*args)
        env.engine.blue_robots[0].touched_puck = True
        env.engine.puck.last_touch_team = 0
        return info

    monkeypatch.setattr(env.engine, "step", contact)
    for tick in range(20):
        _, reward, _, _, info = env.step([[0, 0]], np.zeros((0, 2)))
        assert info["reward_components"]["first_touch"] == (1 if tick == 0 else 0)
        assert sum(info["reward_components"].values()) == pytest.approx(reward)
    assert info["episode_reward_components"]["first_touch"] == 1
    env.reset(seed=0)
    assert env.step([[0, 0]], np.zeros((0, 2)))[4]["reward_components"]["first_touch"] == 1


def test_carry_pays_only_new_maximum_with_recent_blue_touch_and_never_red_progress():
    env = RoboCoreHockeyEnv(
        1, opponent_robots=0, reward_config=RewardConfig(carry=3), use_partial_vision=False
    )
    env.reset(seed=0)
    env.engine.place_puck(0, 0)
    env.best_potential = env._potential()
    env.last_blue_touch = 0
    env.engine.puck.last_touch_team = 0
    env.engine.place_puck(30, 0)
    first = env.step([[0, 0]], np.zeros((0, 2)))[4]
    assert first["reward_components"]["carry"] == pytest.approx(0.3)
    env.engine.place_puck(0, 0)
    assert env.step([[0, 0]], np.zeros((0, 2)))[4]["reward_components"]["carry"] == 0
    env.engine.place_puck(30, 0)
    assert env.step([[0, 0]], np.zeros((0, 2)))[4]["reward_components"]["carry"] == 0
    env.engine.puck.last_touch_team = 1
    env.engine.place_puck(60, 0)
    assert env.step([[0, 0]], np.zeros((0, 2)))[4]["reward_components"]["carry"] == 0
    assert env.episode_reward_components["carry"] == pytest.approx(0.3)


def test_idle_penalty_requires_sustained_inactivity_far_from_puck():
    env = RoboCoreHockeyEnv(
        1,
        opponent_robots=0,
        max_steps=200,
        reward_config=RewardConfig(idle_penalty=0.1),
        use_partial_vision=False,
    )
    env.reset(seed=0)
    for _ in range(89):
        assert env.step([[0, 0]], np.zeros((0, 2)))[4]["reward_components"]["idle_penalty"] == 0
    for _ in range(3):
        info = env.step([[0, 0]], np.zeros((0, 2)))[4]
    assert info["reward_components"]["idle_penalty"] == -0.1
    r = env.engine.blue_robots[0]
    env.engine.place_puck(r.x + 15, r.y)
    assert env.step([[0, 0]], np.zeros((0, 2)))[4]["reward_components"]["idle_penalty"] == 0


def test_opponent_own_goal_preserves_score_but_does_not_pass_skill_gate():
    env = preset_config("duel_selfplay").make_env()
    env.reset(seed=0)
    env.engine.place_puck(env.arena.half_length + env.arena.puck_radius + 0.2, 0)
    env.engine.puck.last_touch_team = 1
    _, _, term, _, info = env.step([[0, 0]], [[0, 0]])
    assert term and info["is_success"] and not info["active_success"]
    assert info["goal_last_touch_team"] == 1
    summary = participation_summary([dict(success=True, **participation_record(info))])
    assert summary["active_success_rate"] == 0 and summary["wins_without_blue_touch"] == 1


def test_curriculum_cannot_advance_without_complete_held_out_proof():
    cfg = preset_config("solo_foundation")
    c = SoloCurriculum(cfg)
    records = [dict(stage=c.stage, success=True, active_success=False)] * 30
    assert not c.update(records) and not c.ready
    records = [dict(stage=c.stage, success=True, active_success=True)] * 30
    assert not c.update(records) and c.ready
    assert not c.validate(c.stage, dict(episodes=10, complete=True, active_success_rate=0.6))
    assert not c.validate(c.stage, dict(episodes=10, complete=False, active_success_rate=1.0))
    assert not c.validate("solo_search", dict(episodes=10, complete=True, active_success_rate=1.0))
    assert c.validate(c.stage, dict(episodes=10, complete=True, active_success_rate=0.7))
    assert c.stage == "solo_carry" and not c.ready


def test_stagnation_is_per_stage_and_does_not_treat_passive_wins_as_progress():
    guard = ProgressGuard(2)
    guard.record(dict(stage="solo_finish", success=True, active_success=False))
    guard.record(dict(stage="solo_carry", success=False, active_success=False))
    assert not guard.exhausted("solo_finish")
    guard.record(dict(stage="solo_finish", success=True, active_success=False))
    assert guard.exhausted("solo_finish") and not guard.exhausted("solo_carry")
    restored = ProgressGuard(2)
    restored.restore(guard.state())
    assert restored.exhausted("solo_finish")
    restored.record(dict(stage="solo_finish", success=True, active_success=True))
    assert not restored.exhausted("solo_finish")


def test_stagnation_stops_actual_worker_cooperatively_and_saves(tmp_path):
    cfg = replace(
        preset_config("solo_goal"),
        stagnation_patience=2,
        steps=10000,
        episode_seconds=1,
        envs=1,
        rollout=8,
        epochs=1,
        eval_every=10000,
    )
    session = TrainingSession(cfg, tmp_path, publish_live=False)
    session.run()
    status = read_json(tmp_path / "status.json")
    assert status["state"] == "stopped" and status["stop_reason"] == "stagnation"
    assert status["session_transitions"] == 60
    HockeyPPOTrainer().load(tmp_path / "latest.pt")


def test_legacy_checkpoint_without_new_reward_fields_remains_resumable(tmp_path):
    cfg = replace(
        preset_config("solo_goal"), steps=8, envs=1, rollout=8, epochs=1, episode_seconds=1, eval_episodes=1
    )
    old = cfg.make_env().configuration()
    old["reward"] = {
        k: v
        for k, v in asdict(RewardConfig()).items()
        if k not in ("alignment", "first_touch", "carry", "idle_penalty")
    }
    checkpoint = tmp_path / "old.pt"
    trainer = HockeyPPOTrainer(gamma=cfg.gamma)
    trainer.save(checkpoint, dict(configuration=old, training_config=cfg.to_dict()))
    session = TrainingSession(
        replace(cfg, init_mode="resume", checkpoint=str(checkpoint)), tmp_path / "new", publish_live=False
    )
    session.run()
    assert read_json(tmp_path / "new/status.json")["state"] == "completed"


def test_stage_validation_does_not_wait_for_full_evaluation_interval(tmp_path, monkeypatch):
    cfg = replace(preset_config("solo_foundation"), envs=1, rollout=1, epochs=1, eval_every=100)
    session = TrainingSession(cfg, tmp_path, publish_live=False)
    env = cfg.make_env()
    obs, _ = env.reset(seed=1)
    trainer = HockeyPPOTrainer()
    action, logp, value = trainer.select_action(obs)
    nxt, reward, term, trunc, _ = env.step(action)
    trainer.store_transition(obs, action, logp, reward, value, term, trunc, next_obs=nxt)
    session.episode_records.extend(
        [
            dict(
                stage="solo_finish",
                success=True,
                active_success=True,
                reward=1.0,
                reward_components=dict(goal=1.0),
            )
        ]
        * 30
    )
    assert not session.curriculum.update(session.episode_records) and session.curriculum.ready
    calls = []

    def validation(*args, **kwargs):
        calls.append(kwargs)
        return dict(episodes=10, complete=True, success_rate=1, active_success_rate=1)

    monkeypatch.setattr("rl.training.evaluate_config", validation)
    session._update(trainer, env, [], lambda: None, io.StringIO())
    assert trainer.updates_count == 1
    assert session.curriculum.stage == "solo_carry"
    assert len(calls) == 1 and calls[0]["seed"] == 30000
    assert calls[0]["scenarios"] == ("solo_finish",)
    metadata = HockeyPPOTrainer().load(tmp_path / "latest.pt")
    assert metadata["curriculum"]["index"] == 1


@pytest.mark.parametrize("active,complete,promoted", [(0, True, False), (1, False, False), (1, True, True)])
def test_checkpoint_selection_requires_active_success_and_complete_evaluation(
    tmp_path, monkeypatch, active, complete, promoted
):
    cfg = preset_config("solo_goal")
    session = TrainingSession(cfg, tmp_path, publish_live=False, reporter=lambda *args: None)
    monkeypatch.setattr(
        "rl.training.evaluate_config",
        lambda *a, **k: dict(
            episodes=1, success_rate=1, point_score=1, active_success_rate=active, complete=complete
        ),
    )
    session._update(HockeyPPOTrainer(), cfg.make_env(), [], lambda: None, io.StringIO())
    assert (tmp_path / "latest.best.pt").exists() == promoted
    assert (tmp_path / "latest.pt").exists()
