from copy import deepcopy
from dataclasses import replace
import hashlib
import math
import time
import numpy as np
import pytest
import torch
from rl.env import RoboCoreHockeyEnv
from rl.modular import ModularRun, ModularChunk, load_plan, module_config, passes, single_worker
from rl.ppo_trainer import HockeyPPOTrainer
from rl.recording import record_episode
from rl.tasks import RewardConfig
from rl.training import TrainingSession, read_json
from rl.training_config import preset_config
from vision.state import wrap_angle


def minimal_plan():
    cfg = replace(
        preset_config("solo_reach"),
        profile="micro-practice",
        envs=1,
        rollout=4,
        epochs=1,
        steps=8,
        episode_seconds=1,
        eval_episodes=1,
        eval_every=10000,
        domain_randomization=False,
        reset_critic=True,
        reset_exploration=True,
    )
    return dict(
        version=1,
        threshold=0.8,
        evaluation_episodes=2,
        confirmation_episodes=2,
        regression_episodes=2,
        chunk_steps=8,
        max_stage_steps=8,
        plateau_checks=3,
        base=cfg.to_dict(),
        modules=[
            dict(id="first", label="First", retain=True, config={}),
            dict(id="second", label="Second", config={}),
        ],
    )


def initial_checkpoint(tmp_path, plan):
    path = tmp_path / "initial.pt"
    cfg = module_config(plan, plan["modules"][0])
    HockeyPPOTrainer().save(
        path, dict(configuration=cfg.make_env().configuration(), training_config=cfg.to_dict())
    )
    return path


def result(success):
    return dict(
        episodes=2,
        complete=True,
        active_success_rate=success,
        touch_episode_rate=success,
        mean_distance_to_puck=20,
        mean_idle_fraction=0,
    )


def test_plan_has_all_phases_with_compatible_observations_and_fresh_training_seeds():
    plan = load_plan("configs/training/modular.json")
    assert len(plan["modules"]) == 11
    assert plan["evaluation_episodes"] == plan["confirmation_episodes"] == 50
    assert plan["threshold"] == 0.8
    for module in plan["modules"]:
        cfg = module_config(plan, module)
        env = cfg.make_env()
        assert env.reset(seed=3)[0].shape == (1, 63)
    assert module_config(plan, plan["modules"][-1]).opponent == "selfplay"


def test_both_proof_sets_and_each_fixed_opponent_must_pass():
    assert passes(result(0.8), 0.8, 2)
    assert not passes(dict(result(1), complete=False), 0.8, 2)
    assert not passes(dict(result(1), episodes=1), 0.8, 2)
    assert not passes(dict(complete=True, by_opponent=dict(a=result(1), b=result(0.5))), 0.8, 2)


def test_os_lock_prevents_duplicate_workers_and_releases_on_exit(tmp_path):
    with single_worker(tmp_path):
        with pytest.raises(ValueError, match="worker ativo"):
            with single_worker(tmp_path):
                pytest.fail("Duplicate worker acquired the lock")
    with single_worker(tmp_path):
        pass


def test_confirmation_failure_never_advances_even_at_stage_budget(tmp_path, monkeypatch):
    plan = minimal_plan()
    source = initial_checkpoint(tmp_path, plan)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    seen = []
    monkeypatch.setattr(ModularRun, "record", lambda *a, **k: None)

    def evaluate(path, cfg, count, seed, stop):
        seen.append((cfg.name, seed))
        return result(0 if seed >= 90000 else 1)

    monkeypatch.setattr("rl.modular.evaluate_module", evaluate)
    run = ModularRun(plan, tmp_path / "run", source)
    run.run()
    assert run.state["index"] == 0 and not run.state["approved"]
    assert read_json(run.directory / "status.json")["state"] == "needs_attention"
    assert all(name == "First" for name, _ in seen)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert (run.directory / "01-first/latest.pt").exists()


def test_approved_phase_is_immutable_and_failed_next_phase_is_retained(tmp_path, monkeypatch):
    plan = minimal_plan()
    source = initial_checkpoint(tmp_path, plan)
    monkeypatch.setattr(ModularRun, "record", lambda *a, **k: None)
    monkeypatch.setattr(
        "rl.modular.evaluate_module", lambda path, cfg, *a: result(1 if cfg.name == "First" else 0)
    )
    run = ModularRun(plan, tmp_path / "run", source)
    run.run()
    assert run.state["index"] == 1 and len(run.state["approved"]) == 1
    approval = read_json(run.directory / "01-first/approval.json")
    assert (
        approval["validation"]["active_success_rate"] == approval["confirmation"]["active_success_rate"] == 1
    )
    assert (
        hashlib.sha256((run.directory / "01-first/approved.pt").read_bytes()).hexdigest()
        == approval["sha256"]
    )


def test_expired_pipeline_does_not_start_a_training_chunk(tmp_path, monkeypatch):
    plan = minimal_plan()
    source = initial_checkpoint(tmp_path, plan)
    monkeypatch.setattr(ModularRun, "record", lambda *a, **k: None)
    run = ModularRun(plan, tmp_path / "run", source, deadline=time.time() - 1)
    run.run()
    assert read_json(run.directory / "status.json")["stop_reason"] == "time_limit"
    assert run.state["chunk"] == 0


def test_regression_failure_prevents_promotion_even_if_current_module_passes(tmp_path, monkeypatch):
    plan = minimal_plan()
    source = initial_checkpoint(tmp_path, plan)
    monkeypatch.setattr(ModularRun, "record", lambda *a, **k: None)
    monkeypatch.setattr(
        "rl.modular.evaluate_module", lambda path, cfg, count, seed, stop: result(0 if seed >= 110000 else 1)
    )
    run = ModularRun(plan, tmp_path / "run", source)
    run.run()
    assert run.state["index"] == 1
    assert not (run.directory / "02-second/approved.pt").exists()
    regression = read_json(run.directory / "02-second/regressions-001.json")
    assert regression["first"]["active_success_rate"] == 0


def test_global_stop_reaches_child_and_saves_without_advancing(tmp_path):
    plan = minimal_plan()
    source = initial_checkpoint(tmp_path, plan)
    owner = ModularRun(plan, tmp_path / "run", source, deadline=time.time() - 1)
    cfg = module_config(plan, plan["modules"][0])
    child = ModularChunk(owner, cfg, tmp_path / "child", publish_live=False, reporter=lambda *a: None)
    child.run()
    status = read_json(tmp_path / "child/status.json")
    assert status["state"] == "stopped" and status["stop_reason"] == "time_limit"
    assert status["session_transitions"] == 0
    assert (tmp_path / "child/latest.pt").exists()


def test_pending_chunk_recovery_counts_saved_steps_once(tmp_path):
    plan = minimal_plan()
    source = initial_checkpoint(tmp_path, plan)
    run = ModularRun(plan, tmp_path / "run", source)
    chunk = run.directory / "pending"
    cfg = module_config(plan, plan["modules"][0])
    TrainingSession(cfg, chunk, publish_live=False, reporter=lambda *a: None).run()
    run.state.update(pending_chunk=str(chunk), chunk=1)
    run.recover_pending()
    assert run.state["trained_steps"] == 8
    assert run.state["checkpoint"] == str(chunk / "latest.pt")
    run.recover_pending()
    assert run.state["trained_steps"] == 8


def test_transfer_can_reset_value_and_noise_without_changing_actor():
    trainer = HockeyPPOTrainer()
    actor = deepcopy(trainer.net.actor.state_dict())
    with torch.no_grad():
        trainer.net.log_std.fill_(-3)
        for parameter in trainer.net.critic.parameters():
            parameter.fill_(2)
    trainer.reset_transfer_heads(critic=True, exploration=True)
    for key, value in actor.items():
        torch.testing.assert_close(trainer.net.actor.state_dict()[key], value)
    torch.testing.assert_close(trainer.net.log_std, torch.full((2,), -0.8))
    assert any(not torch.all(p == 2) for p in trainer.net.critic.parameters())


def test_near_puck_speed_is_required_in_controlled_approach():
    env = RoboCoreHockeyEnv(
        1,
        opponent_robots=0,
        objective="reach",
        use_partial_vision=False,
        domain_randomization=False,
        reach_speed_limit=30,
        reward_config=RewardConfig(goal=0, reach_distance=12, overspeed_penalty=0.01),
    )
    env.reset(seed=0)
    robot = env.engine.blue_robots[0]
    env.engine.place_robot(robot, 0, 0, 0)
    env.engine.place_puck(14, 0)
    robot.body.velocity = (80, 0)
    _, _, term, _, info = env.step([[0, 0]], np.zeros((0, 2)))
    assert info["distance_to_puck"] <= 12 and info["relative_puck_speed"] > 30
    assert not term and not info["reached_target"]
    assert info["reward_components"]["overspeed_penalty"] < 0
    robot.body.velocity = (0, 0)
    _, _, term, _, info = env.step([[0, 0]], np.zeros((0, 2)))
    assert term and info["reached_target"] and info["active_success"]


def test_limited_heading_is_a_real_curriculum_not_a_different_observation():
    cfg = replace(preset_config("solo_reach"), spawn_distance=30, heading_spread=math.pi / 4)
    env = cfg.make_env()
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        r, p = env.engine.blue_robots[0], env.engine.puck
        error = wrap_angle(r.theta - math.atan2(p.y - r.y, p.x - r.x))
        assert abs(error) <= math.pi / 4 + 1e-6
        assert obs.shape == (1, 63)


def test_real_gif_records_policy_and_provenance_without_modifying_checkpoint(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("pygame")
    from PIL import Image

    plan = minimal_plan()
    source = initial_checkpoint(tmp_path, plan)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cfg = module_config(plan, plan["modules"][0])
    path = tmp_path / "sample.gif"
    data = record_episode(source, cfg, path, seed=123, max_seconds=0.2)
    with Image.open(path) as gif:
        assert gif.n_frames >= 2 and gif.size == (760, 430)
    assert data["seed"] == 123 and data["deterministic"] and not data["selected_example"]
    assert data["result"] == "clip_limit" and data["simulated_seconds"] == pytest.approx(0.2)
    assert data["checkpoint_sha256"] == digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert read_json(path.with_suffix(".json"))["config"]["objective"] == "reach"


def test_legacy_config_defaults_match_old_checkpoint_identity(tmp_path):
    cfg = replace(
        preset_config("solo_reach"), steps=4, envs=1, rollout=4, epochs=1, episode_seconds=1, eval_episodes=1
    )
    old = cfg.make_env().configuration()
    old.pop("heading_spread")
    old.pop("reach_speed_limit")
    old["reward"].pop("overspeed_penalty")
    path = tmp_path / "legacy.pt"
    HockeyPPOTrainer(gamma=cfg.gamma).save(path, dict(configuration=old, training_config=cfg.to_dict()))
    TrainingSession(
        replace(cfg, init_mode="resume", checkpoint=str(path)), tmp_path / "resumed", publish_live=False
    ).run()
    assert read_json(tmp_path / "resumed/status.json")["state"] == "completed"
