import math
import numpy as np
import pytest
from core.constants import ArenaConfig, RobotConfig, SimulationConfig
from core.hockey_physics import HockeyPhysicsEngine
from core.geometry import chassis_outline, convex_chassis_parts
import pymunk


def engine(**kwargs):
    return HockeyPhysicsEngine(
        robots_per_team=1, domain_randomization=False, simulation=SimulationConfig(radio_latency=0), **kwargs
    )


def test_goal_needs_entire_puck_inside_posts_and_past_line():
    e = engine()
    a = e.arena
    e.place_puck(a.half_length + a.puck_radius - 0.1, 0)
    assert e._resolve_arena_boundaries() == 0
    e.place_puck(a.half_length + a.puck_radius + 0.1, a.goal_width / 2 - a.puck_radius + 0.1)
    assert e._resolve_arena_boundaries() == 0
    e.place_puck(a.half_length + a.puck_radius + 0.1, 0)
    assert e._resolve_arena_boundaries() == 1
    assert e._resolve_arena_boundaries() == 0 and e.score_blue == 1


def test_spin_response_depends_on_inertia():
    e = engine()
    slow = engine(robot_config=RobotConfig(inertia=1e6))
    e.step([(-1, 1)], [(0, 0)])
    slow.step([(-1, 1)], [(0, 0)])
    assert abs(e.blue_robots[0].omega) > abs(slow.blue_robots[0].omega) * 10


def test_head_on_puck_collision_transfers_momentum_without_capture():
    e = engine()
    r = e.blue_robots[0]
    e.place_robot(r, 0, 0, 0)
    e.place_puck(12, 0, -150, 0)
    for _ in range(5):
        e.step([(0, 0)], [(0, 0)])
    assert e.puck.vx > 10  # Physical rebound, not assignment to stationary chassis.
    assert r.vx < 0


def test_puck_starting_inside_robot_is_ejected():
    e = engine()
    r = e.blue_robots[0]
    e.place_robot(r, 0, 0, 0)
    e.place_puck(0, 0)
    for _ in range(90):
        e.step([(0, 0)], [(0, 0)])
    assert math.hypot(e.puck.x - r.x, e.puck.y - r.y) > 7


def test_robot_wall_contact_uses_rotated_polygon():
    e = engine()
    r = e.blue_robots[0]
    a = e.arena
    e.place_robot(r, a.half_length - 8, 40, math.pi / 4)
    for _ in range(90):
        e.step([(1, 1)], [(0, 0)])
    pts = [r.body.local_to_world(v) for v in chassis_outline(0, r.config)]
    assert max(p.x for p in pts) < a.half_length + 0.5


def test_reset_faceoff_preserves_match_clock_and_score():
    e = engine()
    e.step([(0, 0)], [(0, 0)])
    t = e.match_time
    e.score_blue = 2
    e.reset_positions()
    assert e.match_time == t and e.score_blue == 2
    e.reset_match()
    assert e.match_time == 0 and e.score_blue == 0


def test_no_randomness_when_randomization_disabled_and_seeded():
    a = engine(seed=4)
    b = engine(seed=4)
    for _ in range(100):
        a.step([(0.2, 0.3)], [(0, 0)])
        b.step([(0.2, 0.3)], [(0, 0)])
    np.testing.assert_array_equal(a.blue_robots[0].body.position, b.blue_robots[0].body.position)


def test_shape_parts_preserve_outline_area():
    for role in range(3):
        assert sum(pymunk.area_for_poly(p) for p in convex_chassis_parts(role)) == pytest.approx(
            pymunk.area_for_poly(chassis_outline(role))
        )


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        RobotConfig(width=21)
    with pytest.raises(ValueError):
        ArenaConfig(corner_radius=100)
    with pytest.raises(ValueError):
        RobotConfig(inertia=-1)
    with pytest.raises(ValueError):
        SimulationConfig(physics_hz=-240)


def test_radio_watchdog_clears_a_stale_command_without_new_packets():
    cfg = SimulationConfig(packet_loss=0.999999999, radio_latency=0)
    e = HockeyPhysicsEngine(1, True, simulation=cfg, seed=12)
    r = e.blue_robots[0]
    r.command = np.ones(2)
    r.last_packet = -1
    e.step([(1, 1)], [(0, 0)])
    np.testing.assert_array_equal(r.command, [0, 0])


def test_configured_restitution_is_the_pair_coefficient():
    cfg = SimulationConfig(radio_latency=0, puck_friction=0, restitution=0.55)
    e = HockeyPhysicsEngine(1, False, simulation=cfg)
    e.place_puck(e.arena.half_length - 10, 40, 100, 0)
    for _ in range(6):
        e.step([(0, 0)], [(0, 0)])
    assert e.puck.vx == pytest.approx(-55, abs=1.0)
