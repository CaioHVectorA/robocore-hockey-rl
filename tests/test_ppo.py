import numpy as np
import pytest
import torch
from rl.ppo_trainer import HockeyPPOTrainer, compute_gae
from vision.state import OBS_DIM

torch.set_num_threads(1)


def test_gae_does_not_mix_robots_or_environments():
    rewards = np.zeros((64, 6))
    rewards[:, 0] = 1
    zeros = np.zeros_like(rewards)
    done = np.zeros_like(rewards, bool)
    done[-1] = True
    adv, ret = compute_gae(rewards, zeros, zeros, done, np.zeros_like(done))
    assert np.all(ret[:, 1:] == 0)
    assert ret[0, 0] > ret[-1, 0] == 1


def test_gae_bootstraps_truncation_but_stops_trace_across_reset():
    reward = np.array([[1.0], [100.0]])
    value = np.zeros_like(reward)
    nxt = np.array([[10.0], [0.0]])
    adv, _ = compute_gae(reward, value, nxt, [[False], [True]], [[True], [False]], gamma=0.9)
    np.testing.assert_allclose(adv[:, 0], [10.0, 100.0])


def test_gae_terminal_has_no_bootstrap():
    adv, _ = compute_gae([[1.0]], [[2.0]], [[100.0]], [[True]], [[False]], gamma=0.9)
    assert adv[0, 0] == -1


def test_gae_partial_rollout_bootstraps():
    adv, _ = compute_gae([[1.0]], [[0.0]], [[10.0]], [[False]], [[False]], gamma=0.9)
    assert adv[0, 0] == 10


def test_squashed_actions_retain_matching_probability_before_update():
    torch.manual_seed(123)
    t = HockeyPPOTrainer()
    obs = np.zeros((4096, OBS_DIM), np.float32)
    action, lp, _ = t.select_action(obs)
    with torch.no_grad():
        new, _ = t.net.evaluate(torch.from_numpy(obs), torch.from_numpy(action))
    np.testing.assert_allclose(np.exp(new.numpy() - lp), 1, atol=1e-5)
    assert np.all(np.abs(action) < 1)


def test_rollout_copies_inputs_and_updates_without_flattening_early():
    t = HockeyPPOTrainer()
    obs = np.zeros((6, OBS_DIM), np.float32)
    for i in range(8):
        act, lp, v = t.select_action(obs)
        t.store_transition(obs, act, lp, np.arange(6), v, False, next_obs=obs)
    obs[:] = 99
    assert t.buffer[0].obs.shape == (6, OBS_DIM)
    assert not np.any(t.buffer[0].obs)
    before = [p.clone() for p in t.net.parameters()]
    loss = t.update(epochs=2)
    assert np.isfinite(loss) and t.updates_count == 1 and not t.buffer
    assert any(not torch.equal(a, b) for a, b in zip(before, t.net.parameters()))


def test_checkpoint_schema_and_resume(tmp_path):
    t = HockeyPPOTrainer(gamma=0.98)
    path = tmp_path / "weights.pt"
    t.save(path, {"configuration": {"test": True}})
    restored = HockeyPPOTrainer()
    assert restored.load(path, resume=True)["configuration"]["test"]
    assert restored.gamma == 0.98
    obs = np.zeros((3, OBS_DIM), np.float32)
    np.testing.assert_array_equal(t.select_action(obs, True)[0], restored.select_action(obs, True)[0])
    torch.save({"model_state": {}, "updates_count": 0}, path)
    with pytest.raises(ValueError, match="Legacy"):
        restored.load(path)


@pytest.mark.parametrize("change", [{"mass": 454.0}, {"model": "previous-invented-chassis"}])
def test_evaluation_rejects_weights_trained_with_different_robot_physics(tmp_path, change):
    from evaluate import benchmark
    from rl.env import RoboCoreHockeyEnv

    cfg = RoboCoreHockeyEnv().configuration()
    cfg["robot"].update(change)
    path = tmp_path / "previous.pt"
    HockeyPPOTrainer().save(path, {"configuration": cfg})
    with pytest.raises(ValueError, match="HockeyBot model mismatch"):
        benchmark(episodes=1, policy=path, max_steps=1)
