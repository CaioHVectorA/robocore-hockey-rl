"""Comprehensive unit and integration tests for Mini-Sumo 500g physics, 7-ray sensors, and bots."""

import math
import numpy as np

from sumo.bots import (
    BullRushBot,
    MatadorBot,
    NeuralPolicyBot,
    SearchAndDestroyBot,
    TawaraGaeshiBot,
    WavePendulumBot,
)
from sumo.constants import CHASSIS_ARCHETYPES, SUMO_ACT_DIM, SUMO_OBS_DIM
from sumo.dohyo import Dohyo
from sumo.env import LAYOUT_CHOICES, SumoEnv
from sumo.physics import SumoPhysicsEngine, build_chassis_polygons, visual_hull_outline
from sumo.policy import SumoPolicyNet
from sumo.sensors import SumoSensorPack


def test_dohyo_geometry_and_white_border_line():
    dohyo = Dohyo()
    assert dohyo.radius == 38.5
    assert dohyo.inner_radius == 36.0
    assert dohyo.is_inside(0.0, 0.0)
    assert dohyo.is_inside(38.0, 0.0)
    assert not dohyo.is_inside(39.0, 0.0)

    assert dohyo.line_sensor_intensity(0.0, 0.0) == 0.0
    assert dohyo.line_sensor_intensity(35.0, 0.0) == 0.0
    assert dohyo.line_sensor_intensity(37.5, 0.0) > 0.5
    assert dohyo.line_sensor_intensity(40.0, 0.0) == 1.0


def test_all_starting_positions_never_overlap():
    dohyo = Dohyo()
    for layout in LAYOUT_CHOICES:
        p0, p1 = dohyo.starting_positions(layout)
        dist = math.hypot(p0[0] - p1[0], p0[1] - p1[1])
        # Minimum center distance for two 10cm robots must be > 14 cm so they never overlap at t=0
        assert dist >= 14.0, f"Layout {layout} overlaps at start (dist={dist:.2f} cm)"


def test_all_five_chassis_archetypes_build_and_collide():
    for key, cfg in CHASSIS_ARCHETYPES.items():
        polys = build_chassis_polygons(cfg)
        hull = visual_hull_outline(cfg)
        assert len(polys) >= 1
        assert len(hull) >= 4
        assert cfg.length <= 10.0
        assert cfg.width <= 10.0
        assert cfg.mass <= 500.0


def test_rigid_body_collision_never_tunnels_head_on():
    """Verify two full-speed robots collide solidly at ~9.8 cm and NEVER pass through each other."""
    engine = SumoPhysicsEngine(
        blue_config=CHASSIS_ARCHETYPES["muramasa_fork"],
        red_config=CHASSIS_ARCHETYPES["haryama"],
    )
    engine.reset("standard")

    min_center_dist = 999.0
    contact_occurred = False
    for _ in range(25):
        res = engine.step((1.0, 1.0), (1.0, 1.0))
        d = math.hypot(engine.robot_blue.x - engine.robot_red.x, engine.robot_blue.y - engine.robot_red.y)
        min_center_dist = min(min_center_dist, d)
        if res["in_contact"]:
            contact_occurred = True
        # Blue starts at x < 0 and Red at x > 0; Blue's x must ALWAYS remain < Red's x!
        assert engine.robot_blue.x < engine.robot_red.x - 8.0

    assert contact_occurred is True
    assert min_center_dist >= 8.8  # Solid polygon contact without tunneling


def test_wedge_mechanics_transfers_traction_on_flank_and_low_blade():
    engine = SumoPhysicsEngine(
        blue_config=CHASSIS_ARCHETYPES["muramasa_fork"],  # 18 deg fork
        red_config=CHASSIS_ARCHETYPES["haryama"],         # 26 deg tapered
    )
    engine.reset("standard")
    # Place Blue behind Red facing Red's rear (flank/rear attack)
    engine.robot_blue.set_pose(-5.0, 0.0, 0.0)
    engine.robot_red.set_pose(4.8, 0.0, 0.0)  # Red also facing +x (away from Blue)

    engine.step((1.0, 1.0), (0.0, 0.0))
    assert engine.robot_blue.in_contact is True
    assert engine.robot_blue.wedge_advantage is True
    assert engine.robot_red.is_lifted is True
    assert engine.robot_blue.traction_multiplier > 1.4
    assert engine.robot_red.traction_multiplier < 0.4


def test_pymunk_7_ray_distance_sensors_and_steering_convention():
    dohyo = Dohyo()
    engine = SumoPhysicsEngine()
    engine.reset("standard")
    sensors = SumoSensorPack(engine.blue_cfg, dohyo)

    # 1. Standard face-off: Center sensor C0 (index 3) must detect Red
    dist = sensors.read_distance_sensors(engine.robot_blue, engine.robot_red)
    assert len(dist) == 7
    assert dist[3] < 0.35
    assert sensors.ray_hits[3] is True

    # 2. Place Red on Blue's Left (+y) at 42 deg -> L45 (index 1) must trigger
    engine.robot_blue.set_pose(0.0, 0.0, 0.0)
    engine.robot_red.set_pose(14.0, 13.0, math.pi)
    dist_left = sensors.read_distance_sensors(engine.robot_blue, engine.robot_red)
    assert dist_left[1] < 0.50

    # 3. Verify SearchAndDestroyBot steers LEFT (right_pwm > left_pwm) when enemy is on Left
    bot = SearchAndDestroyBot(0)
    left_pwm, right_pwm = bot.get_action(engine.robot_blue, engine.robot_red, sensors)
    assert right_pwm > left_pwm


def test_sumo_environment_reset_step_and_ringout():
    env = SumoEnv(opponent=BullRushBot(), randomize_matchups=True)
    obs, info = env.reset(seed=42)

    assert obs.shape == (SUMO_OBS_DIM,)
    assert env.action_space.shape == (SUMO_ACT_DIM,)

    next_obs, reward, term, trunc, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert next_obs.shape == (SUMO_OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)


def test_ring_out_terminates_and_awards_winner():
    env = SumoEnv()
    env.reset()

    env.engine.robot_red.set_pose(42.0, 0.0, 0.0)
    obs, reward, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert term is True
    assert info["winner"] == 0
    assert reward > 15.0


def test_all_competitive_bots_and_neural_bot_generate_valid_actions():
    dohyo = Dohyo()
    engine = SumoPhysicsEngine()
    sensors = SumoSensorPack(engine.red_cfg, dohyo)
    engine.reset("standard")

    policy = SumoPolicyNet(SUMO_OBS_DIM, SUMO_ACT_DIM)
    bots = [
        BullRushBot(),
        SearchAndDestroyBot(),
        MatadorBot(),
        WavePendulumBot(),
        TawaraGaeshiBot(),
        NeuralPolicyBot(policy, deterministic=True),
    ]
    for bot in bots:
        act = bot.get_action(engine.robot_red, engine.robot_blue, sensors)
        assert len(act) == 2
        assert -1.0 <= act[0] <= 1.0
        assert -1.0 <= act[1] <= 1.0
