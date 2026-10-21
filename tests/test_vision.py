import math
import cv2
import numpy as np
import pytest
from core.constants import SimulationConfig
from rl.env import RoboCoreHockeyEnv
from vision.camera_tracker import ArenaHomography, HockeyVisionTracker, StateEstimator
from vision.state import EntityState, WorldState, build_team_observation, wrap_angle, OBS_DIM


def homography():
    return ArenaHomography(np.array([[0, 0], [1200, 0], [1200, 600], [0, 600]], np.float32))


def test_uncalibrated_and_degenerate_homographies_fail():
    with pytest.raises(ValueError):
        ArenaHomography().px_to_cm(10, 10)
    with pytest.raises(ValueError):
        ArenaHomography(np.zeros((4, 2)))
    h = homography()
    np.testing.assert_allclose(h.px_to_cm(0, 0), [-150, 75])
    assert h.cm_to_px(0, 0) == (600, 300)


def test_estimator_tracks_velocity_rotation_and_wraparound():
    e = StateEstimator()
    for i in range(61):
        s = e.update({0: dict(x=i, y=0, theta=wrap_angle(math.pi - 0.1 + i * 0.01))}, None, i / 60)
    r = s.robots[0]
    assert abs(r.vx - 60) < 2 and abs(r.omega - 0.6) < 0.1


def test_dropout_predicts_then_expires_without_fabricating_robots():
    e = StateEstimator()
    e.update({0: dict(x=30, y=20, theta=0)}, dict(x=0, y=0), 0)
    e.update({0: dict(x=31, y=20, theta=0.1)}, None, 0.03)
    assert e.snapshot(0.1).robots[0].x > e.robots[0].x
    assert not e.snapshot(1).robots[0].valid(1)
    assert 1 not in e.snapshot(1).robots
    with pytest.raises(ValueError):
        e.update({}, None, -1)


def test_same_state_same_actor_vector_in_simulation_and_real():
    env = RoboCoreHockeyEnv(domain_randomization=False)
    env.reset(seed=42)
    tracker = HockeyVisionTracker(ArenaHomography(arena=env.arena))
    tracker.state = env.state
    expected = tracker.build_team_obs_for_rl(previous_actions=env.previous_actions)
    np.testing.assert_array_equal(env.get_team_obs(), expected)


def test_spin_and_reverse_are_observable_without_wheel_encoders():
    s = WorldState(
        1.0, {0: EntityState(theta=0, vx=-20, omega=5, confidence=1, last_seen=1, timestamp=1)}, EntityState()
    )
    obs = build_team_observation(s)
    assert obs.shape == (3, OBS_DIM) and obs[0, 12] < 0 and obs[0, 13] > 0
    assert np.all(obs[1, :20] == 0)


def test_puck_detector_rejects_larger_square():
    h = homography()
    tracker = HockeyVisionTracker(h)
    frame = np.full((600, 1200, 3), 240, np.uint8)
    cv2.circle(frame, (400, 300), 6, (0, 0, 0), -1)
    cv2.rectangle(frame, (800, 200), (830, 230), (0, 0, 0), -1)
    _, p = tracker.detect_robots_and_puck(frame, 0)
    assert p.valid(0)
    np.testing.assert_allclose([p.x, p.y], h.px_to_cm(400, 300), atol=0.3)


def test_real_aruco_id_and_heading_convention():
    tracker = HockeyVisionTracker(homography())
    frame = np.full((600, 1200, 3), 240, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag = cv2.aruco.generateImageMarker(dictionary, 0, 80)
    frame[260:340, 560:640] = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
    robots, _ = tracker.detect_robots_and_puck(frame, 0)
    assert 0 in robots and abs(robots[0].theta - math.pi / 2) < 0.03


@pytest.mark.parametrize("pixel,visible", [((600, 275), True), ((574, 300), False), ((576, 328), False)])
def test_stock_bay_remains_visible_but_cover_and_tyres_cannot_be_puck(pixel, visible):
    h = homography()
    tracker = HockeyVisionTracker(h)
    frame = np.full((600, 1200, 3), 240, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag = cv2.aruco.generateImageMarker(dictionary, 0, 32)
    frame[284:316, 584:616] = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
    cv2.circle(frame, pixel, 6, (0, 0, 0), -1)
    robots, puck = tracker.detect_robots_and_puck(frame, 0)
    assert 0 in robots
    assert puck.valid(0) == visible
    if visible:
        np.testing.assert_allclose([puck.x, puck.y], h.px_to_cm(*pixel), atol=0.3)


def test_simulated_latency_is_applied():
    cfg = SimulationConfig(vision_latency=0.1, vision_noise=0, vision_dropout=0)
    env = RoboCoreHockeyEnv(simulation=cfg, domain_randomization=False)
    env.reset(seed=1)
    old = env.state.puck.x
    env.engine.place_puck(20, 0)
    for _ in range(3):
        env.step(np.zeros((3, 2)), custom_red_actions=[(0, 0)] * 3)
    assert env.state.puck.x == old
    env.step(np.zeros((3, 2)), custom_red_actions=[(0, 0)] * 3)
    assert env.state.puck.x > old


def test_team_rotation_preserves_observation_contract():
    from dataclasses import replace

    env = RoboCoreHockeyEnv(domain_randomization=False, use_partial_vision=False)
    env.reset(seed=5)
    state = env.state
    mirror = {}
    for idx, e in state.robots.items():
        new_id = (1 - idx // 3) * 3 + idx % 3
        mirror[new_id] = replace(e, x=-e.x, y=-e.y, theta=e.theta + math.pi, vx=-e.vx, vy=-e.vy)
    p = state.puck
    flipped = WorldState(state.timestamp, mirror, replace(p, x=-p.x, y=-p.y, vx=-p.vx, vy=-p.vy))
    np.testing.assert_allclose(
        build_team_observation(state, team=0), build_team_observation(flipped, team=1), atol=1e-6
    )
