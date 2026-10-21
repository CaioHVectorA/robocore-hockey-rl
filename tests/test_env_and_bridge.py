import asyncio
import time
import numpy as np
import pytest
from core.actuators import action_to_pwm, pwm_to_normalized
from rl.env import RoboCoreHockeyEnv
from bridge.vespa_ble_bridge import encode_command, decode_command, VespaBluetoothBridge, VespaTeamBridge
from run_robot import tracking_ready, validate_transport


def test_seed_reproduces_state_noise_and_trajectories():
    a = RoboCoreHockeyEnv()
    b = RoboCoreHockeyEnv()
    np.testing.assert_array_equal(a.reset(seed=42)[0], b.reset(seed=42)[0])
    for _ in range(20):
        oa, *_ = a.step(np.ones((3, 2)) * 0.2)
        ob, *_ = b.step(np.ones((3, 2)) * 0.2)
        np.testing.assert_array_equal(oa, ob)


def test_stationary_touch_has_no_positive_contact_reward():
    env = RoboCoreHockeyEnv(domain_randomization=False, use_partial_vision=False)
    env.reset(seed=1)
    r = env.engine.blue_robots[0]
    env.engine.place_robot(r, 0, 0, 0)
    env.engine.place_puck(9, 0)
    env.prev_potential = env._potential()
    env.prev_approach = env._approach()
    _, reward, _, _, _ = env.step(np.zeros((3, 2)), custom_red_actions=[(0, 0)] * 3)
    assert reward < 0


def test_gym_truncation_and_reset_contract():
    env = RoboCoreHockeyEnv(max_steps=1)
    o, _ = env.reset(seed=1)
    assert env.observation_space.contains(o)
    _, _, term, trunc, _ = env.step(np.zeros((3, 2)))
    assert trunc and not term
    with pytest.raises(RuntimeError):
        env.step(np.zeros((3, 2)))
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.full((3, 2), np.nan))


def test_shared_small_action_pwm_has_no_sim_real_deadband_mismatch():
    pwm = action_to_pwm([0.05, -0.05])
    wheel = pwm_to_normalized(pwm)
    assert wheel[0] > 0 and wheel[1] < 0
    np.testing.assert_allclose(wheel, [0.05, -0.05], atol=0.003)
    np.testing.assert_array_equal(action_to_pwm([0, 0]), [0, 0])


def test_protocol_crc_ttl_and_range():
    packet = encode_command(65535, -255, 255, 150)
    assert decode_command(packet) == (65535, -255, 255, 150)
    broken = bytearray(packet)
    broken[5] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        decode_command(broken)
    with pytest.raises(ValueError):
        encode_command(1, 500, 0)
    with pytest.raises(ValueError):
        encode_command(1, 0, 0, 500)
    with pytest.raises(ValueError):
        VespaTeamBridge(["same"] * 3)


def test_radio_rate_deadline_and_emergency_bypass_without_hardware():
    class Client:
        def __init__(self):
            self.packets = []

        async def write_gatt_char(self, uuid, data, **kwargs):
            self.packets.append((time.monotonic(), decode_command(data)))

    async def exercise():
        b = VespaBluetoothBridge("FAKE", control_rate_hz=30)
        b.client = Client()
        b.is_connected = True
        assert await b.send_wheel_speeds(0.2, 0.2)
        assert await b.send_wheel_speeds(0.2, 0.2)
        assert b.client.packets[1][0] - b.client.packets[0][0] >= 0.030
        assert not await b.send_wheel_speeds(1, 1, deadline=time.monotonic() - 1)
        assert b.client.packets[-1][1][1:3] == (0, 0)
        started = time.monotonic()
        await b.emergency_stop()
        assert time.monotonic() - started < 0.025

    asyncio.run(exercise())


def test_supervisor_requires_all_own_robots_and_puck():
    env = RoboCoreHockeyEnv(domain_randomization=False)
    env.reset(seed=1)
    state = env.state
    assert tracking_ready(state)
    del state.robots[1]
    assert not tracking_ready(state)


def test_simulator_uses_the_same_tracking_stop_as_hardware():
    env = RoboCoreHockeyEnv(domain_randomization=False)
    env.reset(seed=1)
    del env.vision_sensor.state.robots[1]
    env.step(np.ones((3, 2)))
    np.testing.assert_array_equal(env.previous_actions, np.zeros((3, 2)))


def test_recording_mode_never_connects_to_radio(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import run_robot

    class FakeCamera:
        running = True

        def __init__(self, source):
            self.t = time.monotonic()
            self.frame = np.full((600, 1200, 3), 240, np.uint8)

        def latest(self):
            return self.t, self.frame

        def close(self):
            pass

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run connected to hardware")

    monkeypatch.setattr(run_robot, "LatestCamera", FakeCamera)
    monkeypatch.setattr(run_robot, "VespaTeamBridge", forbidden)
    cfg = tmp_path / "camera.json"
    cfg.write_text(json.dumps(dict(floor_corners_px=[[0, 0], [1200, 0], [1200, 600], [0, 600]])))
    args = SimpleNamespace(
        config=str(cfg),
        track_only=True,
        policy=None,
        armed=False,
        seconds=0.05,
        log=str(tmp_path / "record.jsonl"),
        preview=False,
    )
    asyncio.run(run_robot.run(args))
    records = [json.loads(line) for line in (tmp_path / "record.jsonl").read_text().splitlines()]
    assert records and all(not rec["tracking_ready"] for rec in records)
    assert all(np.array(rec["actions"]).sum() == 0 for rec in records)


def test_original_bluetooth_is_never_mistaken_for_custom_adapter():
    with pytest.raises(ValueError, match="Stock HockeyBot"):
        validate_transport({"protocol_verified": True})
    with pytest.raises(ValueError, match="Stock HockeyBot"):
        validate_transport({"transport": "robocore-stock", "protocol_verified": True})
    with pytest.raises(ValueError, match="Verify custom"):
        validate_transport({"transport": "hockey-v2-custom"})
    validate_transport({"transport": "hockey-v2-custom", "protocol_verified": True})
