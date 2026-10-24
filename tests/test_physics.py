import math
import numpy as np
import pytest
from core.constants import ArenaConfig, RobotConfig, SimulationConfig
from core.hockey_physics import HockeyPhysicsEngine
from core.geometry import chassis_outline, convex_chassis_parts, footprint_vertices, wheel_footprints
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


def test_manual_puck_placement_cannot_embed_disc_in_chassis():
    e = engine()
    r = e.blue_robots[0]
    e.place_robot(r, 0, 0, 0)
    e.place_puck(0, 0)
    shapes = [s for s in e.space.shapes if s.body is r.body]
    assert min(s.point_query(e.puck.body.position).distance for s in shapes) >= e.arena.puck_radius
    for _ in range(5):
        e.step([(0, 0)], [(0, 0)])
    # A deep bay can accommodate the disc closer to the origin. Check actual
    # separation from every solid piece rather than an old rectangular radius.
    assert min(s.point_query(e.puck.body.position).distance for s in shapes) >= e.arena.puck_radius - 0.05


def test_shallow_overlap_at_curved_bay_is_resolved_by_contacts():
    e = engine()
    r = e.blue_robots[0]
    e.place_robot(r, 0, 0, 0)
    e.puck.body.position = (6.0, 0)  # Deliberately bypass the bench placement guard.
    e.space.reindex_shapes_for_body(e.puck.body)
    shapes = [s for s in e.space.shapes if s.body is r.body]
    assert min(s.point_query(e.puck.body.position).distance for s in shapes) < e.arena.puck_radius
    for _ in range(90):
        e.step([(0, 0)], [(0, 0)])
    assert min(s.point_query(e.puck.body.position).distance for s in shapes) >= e.arena.puck_radius - 0.05


def test_robot_wall_contact_uses_rotated_polygon():
    e = engine()
    r = e.blue_robots[0]
    a = e.arena
    e.place_robot(r, a.half_length - 8, 40, math.pi / 4)
    for _ in range(90):
        e.step([(1, 1)], [(0, 0)])
    pts = [r.body.local_to_world(v) for v in footprint_vertices(r.config)]
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


def test_stock_roles_have_identical_hardware_and_assembled_envelope():
    cfg = RobotConfig()
    assert chassis_outline(0, cfg) == chassis_outline(1, cfg) == chassis_outline(2, cfg)
    points = np.array(footprint_vertices(cfg))
    np.testing.assert_allclose(np.ptp(points, axis=0), [cfg.length, cfg.width], atol=1e-9)
    assert cfg.mass == 360 and cfg.wheel_diameter == 6 and cfg.wheel_width == 2
    assert cfg.max_wheel_speed == pytest.approx(math.pi * 6 * cfg.motor_free_rpm / 60)


def test_real_concave_bay_is_open_and_forks_are_solid():
    e = engine()
    r = e.blue_robots[0]
    e.place_robot(r, 0, 0, 0)
    shapes = [s for s in e.space.shapes if s.body is r.body]
    for centre in ((6.3, 0), (8.2, 0), (10, 0)):
        assert min(s.point_query(centre).distance for s in shapes) >= e.arena.puck_radius
    for solid in ((7.5, 5.7), (7.5, -5.7), (3, 0)):
        assert any(s.point_query(solid).distance < 0 for s in shapes)


def test_exposed_tyres_are_colliders_and_transfer_impulse():
    e = engine()
    r = e.blue_robots[0]
    e.place_robot(r, 0, 0, 0)
    shapes = [s for s in e.space.shapes if s.body is r.body]
    for wheel in wheel_footprints(r.config):
        expected = {tuple(p) for p in wheel}
        assert any({tuple(p) for p in s.get_vertices()} == expected for s in shapes)
    e.place_puck(-12, r.config.track / 2, 120, 0)
    for _ in range(4):
        e.step([(0, 0)], [(0, 0)])
    assert e.puck.vx < 0 and r.vx > 0


@pytest.mark.parametrize("kwargs", [{"track": 14}, {"wheel_diameter": 8}, {"tire_mass": 200}])
def test_impossible_stock_assembly_rejected(kwargs):
    with pytest.raises(ValueError):
        RobotConfig(**kwargs)
