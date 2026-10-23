"""Numerical stability, optimization and curriculum distribution regression tests."""

from dataclasses import replace
import json
import math
import numpy as np
import pytest
import torch
from rl.ppo_trainer import HockeyPPOTrainer
from rl.training import TrainingSession, evaluate_config, read_json
from rl.training_config import TrainingConfig, preset_config
from vision.state import OBS_DIM, wrap_angle


def rollout(trainer, *, count=64, reward=1.0):
    obs = np.zeros((count, OBS_DIM), np.float32)
    for _ in range(4):
        action, lp, value = trainer.select_action(obs)
        trainer.store_transition(obs, action, lp, reward, value, True, next_obs=obs)


def test_saturated_entropy_remains_finite_and_pushes_mean_out_of_saturation():
    trainer = HockeyPPOTrainer()
    with torch.no_grad():
        trainer.net.actor[-1].bias.fill_(15)
    entropy = trainer.net.entropy(torch.zeros(2048, OBS_DIM))
    entropy.backward()
    assert torch.isfinite(entropy)
    gradient = trainer.net.actor[-1].bias.grad
    assert torch.isfinite(gradient).all() and torch.all(gradient < -1.9)


def test_exact_rollout_kl_includes_last_optimizer_step_and_stops_epochs():
    torch.manual_seed(10)
    trainer = HockeyPPOTrainer(lr=0.01, target_kl=0.001, value_clip=0.5)
    rollout(trainer)
    trainer.update(epochs=20, batch_size=256)
    assert trainer.metrics["early_stop"]
    assert trainer.metrics["epochs_completed"] < 20
    assert trainer.metrics["full_kl"] > trainer.target_kl
    assert trainer.metrics["optimizer_steps"] == trainer.metrics["epochs_completed"]
    assert trainer.metrics["actor_grad_norm"] > 0
    assert trainer.metrics["critic_grad_norm"] > 0


def test_large_critic_loss_does_not_suppress_actor_update():
    # Terminal one-step tasks keep the actor advantages equal after normalization,
    # even though the shifted critic target has a much larger error.
    models = []
    for offset in (0.0, 10000.0):
        torch.manual_seed(5)
        trainer = HockeyPPOTrainer(entropy_coef=0)
        obs = np.zeros((256, OBS_DIM), np.float32)
        action, lp, value = trainer.select_action(obs)
        rewards = action[:, 0] + offset
        trainer.store_transition(obs, action, lp, rewards, value, True, next_obs=obs)
        trainer.update(epochs=1, batch_size=256)
        models.append(trainer)
    assert models[1].metrics["critic_grad_norm"] > 1000 * models[0].metrics["critic_grad_norm"]
    for first, second in zip(models[0].net.actor.parameters(), models[1].net.actor.parameters()):
        torch.testing.assert_close(first, second, atol=2e-6, rtol=1e-4)


def test_nonfinite_rollout_rejected_before_parameter_change():
    trainer = HockeyPPOTrainer()
    rollout(trainer, reward=np.inf)
    before = {key: val.clone() for key, val in trainer.net.state_dict().items()}
    with pytest.raises(ValueError, match="Non-finite PPO rollout"):
        trainer.update()
    for key, value in trainer.net.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_ppo_learns_a_known_continuous_action_target():
    torch.manual_seed(7)
    trainer = HockeyPPOTrainer(lr=0.001, entropy_coef=0.001, value_clip=0.5)
    obs = np.zeros((128, OBS_DIM), np.float32)
    target = np.array([0.6, -0.4])
    before = np.mean((trainer.select_action(obs, True)[0] - target) ** 2)
    for _ in range(35):
        action, lp, value = trainer.select_action(obs)
        rewards = -np.sum((action - target) ** 2, axis=1)
        trainer.store_transition(obs, action, lp, rewards, value, True, next_obs=obs)
        trainer.update(epochs=4, batch_size=128)
    after = np.mean((trainer.select_action(obs, True)[0] - target) ** 2)
    assert after < before * 0.1


def test_setup_progression_spans_behind_to_every_side_without_changing_observation():
    cfg = replace(preset_config("solo_goal"), scenario="solo_setup", setup_spread=0, heading_spread=0)
    env = cfg.make_env()
    for seed in range(30):
        obs, _ = env.reset(seed=seed)
        robot, puck = env.engine.blue_robots[0], env.engine.puck
        bearing = math.atan2(puck.y - robot.y, puck.x - robot.x)
        goal = math.atan2(-puck.y, env.arena.half_length - puck.x)
        assert abs(wrap_angle(bearing - goal)) < 0.001
        assert abs(wrap_angle(robot.theta - bearing)) < 0.001
        assert obs.shape == (1, OBS_DIM)
    broad = replace(cfg, setup_spread=math.pi, heading_spread=math.pi).make_env()
    angles = []
    for seed in range(100):
        broad.reset(seed=seed)
        robot, puck = broad.engine.blue_robots[0], broad.engine.puck
        angles.append(
            wrap_angle(
                math.atan2(puck.y - robot.y, puck.x - robot.x)
                - math.atan2(-puck.y, broad.arena.half_length - puck.x)
            )
        )
    assert min(angles) < -2.5 and max(angles) > 2.5


def test_rehearsal_uses_fresh_episodes_and_does_not_replace_evaluation(tmp_path):
    cfg = replace(
        preset_config("solo_goal"),
        scenario="solo_setup",
        envs=2,
        steps=150,
        episode_seconds=1,
        rollout=16,
        eval_episodes=2,
        epochs=1,
        rehearsal_scenarios=("solo_finish",),
        rehearsal_probability=0.5,
        value_clip=0.5,
    )
    cfg.save(tmp_path / "roundtrip.json")
    assert TrainingConfig.load(tmp_path / "roundtrip.json") == cfg
    trainer = TrainingSession(cfg, tmp_path / "train", publish_live=False, reporter=lambda *args: None).run()
    episodes = [json.loads(line) for line in (tmp_path / "train/episodes.jsonl").read_text().splitlines()]
    assert {row["stage"] for row in episodes} == {"solo_setup", "solo_finish"}
    result = evaluate_config(trainer, cfg)
    assert result["scenario"] == "solo_setup"
    assert all(row["scenario"] == "solo_setup" for row in result["records"])
    resumed = replace(cfg, init_mode="resume", checkpoint=str(tmp_path / "train/latest.pt"), steps=2)
    TrainingSession(resumed, tmp_path / "resumed", publish_live=False, reporter=lambda *args: None).run()
    assert read_json(tmp_path / "resumed/status.json")["state"] == "completed"


def test_legacy_checkpoint_can_resume_without_value_clipping(tmp_path):
    trainer = HockeyPPOTrainer()
    path = tmp_path / "legacy.pt"
    trainer.save(path)
    saved = torch.load(path, weights_only=True)
    del saved["hyperparameters"]["value_clip"]
    torch.save(saved, path)
    restored = HockeyPPOTrainer(value_clip=0.5)
    restored.load(path, resume=True)
    assert restored.value_clip == 0
    cfg = replace(preset_config("solo_goal"), steps=2, envs=1, rollout=2, epochs=1, eval_episodes=1)
    configuration = cfg.make_env().configuration()
    configuration.pop("setup_spread")
    old_config = cfg.to_dict()
    for name in ("setup_spread", "value_clip", "rehearsal_scenarios", "rehearsal_probability"):
        old_config.pop(name)
    saved["metadata"] = dict(configuration=configuration, training_config=old_config)
    torch.save(saved, path)
    cfg = replace(cfg, init_mode="resume", checkpoint=str(path))
    TrainingSession(cfg, tmp_path / "resumed", publish_live=False, reporter=lambda *args: None).run()
