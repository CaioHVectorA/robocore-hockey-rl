"""Comprehensive unit and integration tests for Mini-Sumo 500g module."""

import math
import numpy as np
import pytest

from sumo.bots import BullRushBot, MatadorBot, SearchAndDestroyBot
from sumo.constants import (
    SumoRobotConfig,
    SUMO_ACT_DIM,
    SUMO_OBS_DIM,
)
from sumo.dohyo import Dohyo
from sumo.env import SumoEnv
from sumo.physics import SumoPhysicsEngine
from sumo.sensors import SumoSensorPack


def test_dohyo_geometry_and_white_border_line():
    dohyo = Dohyo()
    assert dohyo.radius == 38.5
    assert dohyo.inner_radius == 36.0
    assert dohyo.border_width == 2.5

    # Center is inside, not on white line
    assert dohyo.is_inside(0.0, 0.0)
    assert not dohyo.is_on_white_line(0.0, 0.0)
    assert dohyo.line_sensor_intensity(0.0, 0.0) == 0.0

    # Inside black zone at r = 25 cm
    assert dohyo.is_inside(25.0, 0.0)
    assert not dohyo.is_on_white_line(25.0, 0.0)
    assert dohyo.line_sensor_intensity(25.0, 0.0) == 0.0

    # Exactly on white line at r = 37.0 cm
    assert dohyo.is_inside(37.0, 0.0)
    assert dohyo.is_on_white_line(37.0, 0.0)
    assert dohyo.line_sensor_intensity(37.0, 0.0) > 0.0

    # Outside dohyo at r = 40.0 cm
    assert not dohyo.is_inside(40.0, 0.0)
    assert dohyo.line_sensor_intensity(40.0, 0.0) == 1.0


def test_dohyo_starting_positions():
    dohyo = Dohyo()
    (bx, by, bth), (rx, ry, rth) = dohyo.starting_positions("standard")
    assert bx == -5.0 and by == 0.0 and bth == 0.0
    assert rx == 5.0 and ry == 0.0 and rth == pytest.approx(math.pi)
    assert math.hypot(bx - rx, by - ry) == pytest.approx(10.0)


def test_robot_motor_step_and_kinematics():
    cfg = SumoRobotConfig()
    engine = SumoPhysicsEngine(robot_config=cfg)
    engine.reset()

    # Step full throttle forward for 10 ticks
    for _ in range(10):
        engine.step((1.0, 1.0), (0.0, 0.0))

    # Blue should accelerate forward (+x)
    assert engine.robot_blue.vx > 10.0
    assert engine.robot_blue.x > -5.0
    assert abs(engine.robot_blue.vy) < 1.0


def test_wedge_dynamics_transfers_traction():
    cfg = SumoRobotConfig()
    engine = SumoPhysicsEngine(robot_config=cfg)
    engine.reset()

    # Place Blue facing Red's flank (Blue at (0, 0) facing +x, Red at (6, 0) facing +y)
    engine.robot_blue.set_pose(0.0, 0.0, 0.0)
    engine.robot_red.set_pose(6.0, 0.0, math.pi / 2.0)  # Red turned 90 deg sideways

    # Step simulation
    engine.step((1.0, 1.0), (0.0, 0.0))

    # Blue blade should get under Red's flank!
    assert engine.robot_blue.wedge_advantage is True
    assert engine.robot_red.is_lifted is True
    # Blue should gain +40% traction, Red should drop to 40%
    assert engine.robot_blue.traction_multiplier == pytest.approx(1.4)
    assert engine.robot_red.traction_multiplier == pytest.approx(0.4)


def test_sensor_pack_line_and_distance_readings():
    cfg = SumoRobotConfig()
    dohyo = Dohyo()
    engine = SumoPhysicsEngine()
    sensors = SumoSensorPack(cfg, dohyo)

    # Place Blue on center, Red 20 cm ahead
    engine.robot_blue.set_pose(0.0, 0.0, 0.0)
    engine.robot_red.set_pose(20.0, 0.0, math.pi)

    # Center is black -> all 4 line sensors should be 0.0
    line = sensors.read_line_sensors(engine.robot_blue)
    assert len(line) == 4
    np.testing.assert_allclose(line, 0.0)

    # Distance sensors: center ray (index 2) should detect Red at ~15 cm
    dist = sensors.read_distance_sensors(engine.robot_blue, engine.robot_red)
    assert len(dist) == 5
    assert dist[2] < 0.40  # Center detected close
    assert dist[0] == 1.0  # Left 60 deg clear
    assert dist[4] == 1.0  # Right 60 deg clear

    obs = sensors.get_observation(engine.robot_blue, engine.robot_red, (0.5, 0.5))
    assert len(obs) == SUMO_OBS_DIM
    assert obs.dtype == np.float32


def test_sumo_environment_reset_step_and_ringout():
    env = SumoEnv(opponent=BullRushBot())
    obs, info = env.reset(seed=42)

    assert obs.shape == (SUMO_OBS_DIM,)
    assert env.action_space.shape == (SUMO_ACT_DIM,)

    # Step environment
    action = np.array([1.0, 1.0], dtype=np.float32)
    next_obs, reward, term, trunc, info = env.step(action)

    assert next_obs.shape == (SUMO_OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)


def test_ring_out_terminates_and_awards_winner():
    env = SumoEnv()
    env.reset()

    # Place Red outside dohyo (r = 45 cm)
    env.engine.robot_red.set_pose(45.0, 0.0, 0.0)
    obs, reward, term, trunc, info = env.step(np.array([0.0, 0.0]))

    assert term is True
    assert info["winner"] == 0  # Blue won!
    assert reward > 15.0  # Big terminal win reward


def test_heuristic_bots_generate_valid_actions():
    dohyo = Dohyo()
    cfg = SumoRobotConfig()
    engine = SumoPhysicsEngine()
    sensors = SumoSensorPack(cfg, dohyo)
    engine.reset()

    for BotCls in [BullRushBot, SearchAndDestroyBot, MatadorBot]:
        bot = BotCls()
        act = bot.get_action(engine.robot_red, engine.robot_blue, sensors)
        assert len(act) == 2
        assert -1.0 <= act[0] <= 1.0
        assert -1.0 <= act[1] <= 1.0
