"""Overhead ArUco + puck perception and a timestamp-aware shared estimator."""

from __future__ import annotations
from collections import deque
import math
import cv2
import numpy as np
from core.constants import ArenaConfig, SimulationConfig
from vision.state import EntityState, WorldState, wrap_angle, build_team_observation


class ArenaHomography:
    """Image corners TL,TR,BR,BL → metric coordinates with +y UP.

    Frame must first be undistorted. Homography is the floor plane; supply a
    second calibrated plane for raised robot tags to account for parallax.
    """

    def __init__(self, src_corners=None, arena=None):
        self.arena = arena or ArenaConfig()
        self.H = self.H_inv = None
        a = self.arena
        self.dst_corners = np.array(
            [
                [-a.half_length, a.half_width],
                [a.half_length, a.half_width],
                [a.half_length, -a.half_width],
                [-a.half_length, -a.half_width],
            ],
            np.float32,
        )
        if src_corners is not None:
            self.set_source_corners(src_corners)

    def set_source_corners(self, points):
        src = np.asarray(points, np.float32)
        if src.shape != (4, 2) or not np.isfinite(src).all() or not cv2.isContourConvex(src):
            raise ValueError("Provide four ordered, nondegenerate calibration corners")
        if abs(cv2.contourArea(src)) < 100:
            raise ValueError("Calibration polygon too small")
        self.H = cv2.getPerspectiveTransform(src, self.dst_corners)
        self.H_inv = np.linalg.inv(self.H)

    def px_to_cm(self, x, y):
        if self.H is None:
            raise ValueError("Arena has not been calibrated")
        p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self.H)[0, 0]
        return float(p[0]), float(p[1])

    def cm_to_px(self, x, y):
        if self.H_inv is None:
            raise ValueError("Arena has not been calibrated")
        p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self.H_inv)[0, 0]
        return tuple(int(round(v)) for v in p)


class StateEstimator:
    """Alpha-beta estimator with real capture timestamps and angular prediction.

    Identical estimator for simulated detections and camera detections; missing
    objects remain predicted states with age, never fictitious default positions.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.robots = {}
        self.puck = EntityState()
        self.last_capture = -math.inf

    def _correct(self, old, measurement, t):
        if not old.valid(t):
            return EntityState(**measurement, confidence=1.0, last_seen=t, timestamp=t)
        dt = t - old.last_seen
        if dt <= 1e-6:
            return old
        pred = old.predict(t)
        dx = measurement["x"] - pred.x
        dy = measurement["y"] - pred.y
        # Constant-velocity alpha-beta with bounded acceleration innovations.
        vx = pred.vx + 0.12 * dx / dt
        vy = pred.vy + 0.12 * dy / dt
        theta = measurement.get("theta", old.theta)
        dth = wrap_angle(theta - pred.theta)
        return EntityState(
            pred.x + 0.70 * dx,
            pred.y + 0.70 * dy,
            wrap_angle(pred.theta + 0.8 * dth),
            float(np.clip(vx, -400, 400)),
            float(np.clip(vy, -400, 400)),
            float(np.clip(pred.omega + 0.15 * dth / dt, -20, 20)),
            1.0,
            t,
            t,
        )

    def update(self, robots, puck, capture_time, now=None):
        if capture_time < self.last_capture:
            raise ValueError("Out-of-order camera frame")
        if capture_time > self.last_capture:
            for idx, m in robots.items():
                self.robots[idx] = self._correct(self.robots.get(idx, EntityState()), m, capture_time)
            if puck is not None:
                self.puck = self._correct(self.puck, puck, capture_time)
            self.last_capture = capture_time
        return self.snapshot(capture_time if now is None else now)

    def snapshot(self, now):
        return WorldState(now, {i: s.predict(now) for i, s in self.robots.items()}, self.puck.predict(now))


class EntityKalmanFilter:
    """Compatibility wrapper; new code uses StateEstimator for angles and validity."""

    def __init__(self, dt=1 / 60):
        self.dt = dt
        self.reset()

    def reset(self):
        self.est = StateEstimator()
        self.t = 0.0

    def update(self, x, y):
        self.t += self.dt
        s = self.est.update({}, dict(x=x, y=y), self.t).puck
        return s.x, s.y, s.vx, s.vy

    def predict_only(self):
        self.t += self.dt
        s = self.est.snapshot(self.t).puck
        return s.x, s.y, s.vx, s.vy


class SimulatedPartialVisionSensor:
    def __init__(self, enable_dropouts=True, simulation=None):
        self.simulation = simulation or SimulationConfig()
        self.enable_dropouts = enable_dropouts
        self.reset()

    def reset(self):
        self.estimator = StateEstimator()
        self.queue = deque()
        self.state = WorldState(0.0)

    @property
    def tracked_robots(self):
        return self.state.robots

    @property
    def tracked_puck(self):
        return self.state.puck

    def update(self, engine, prime=False):
        cfg = self.simulation
        now = engine.episode_time
        rng = engine.rng
        noise = cfg.vision_noise if engine.domain_randomization else 0.0
        dropout = cfg.vision_dropout if self.enable_dropouts and engine.domain_randomization else 0.0
        robots = {}
        for r in engine.all_robots:
            if rng.random() < dropout:
                continue
            nx, ny = rng.normal(0, noise, 2)
            robots[r.team * 3 + r.role_id] = dict(
                x=r.x + nx, y=r.y + ny, theta=wrap_angle(r.theta + rng.normal(0, noise / 9))
            )
        p = engine.puck
        puck = (
            None
            if rng.random() < dropout
            else dict(x=p.x + rng.normal(0, noise), y=p.y + rng.normal(0, noise))
        )
        self.queue.append((now + (0.0 if prime else cfg.vision_latency), now, robots, puck))
        while self.queue and self.queue[0][0] <= now + 1e-9:
            _, capture, robots, puck = self.queue.popleft()
            self.estimator.update(robots, puck, capture, now)
        self.state = self.estimator.snapshot(now)
        return self.state


class HockeyVisionTracker:
    def __init__(self, homography=None, robot_homography=None, camera_matrix=None, dist_coeffs=None):
        self.homography = homography or ArenaHomography()
        self.robot_homography = robot_homography or self.homography
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.estimator = StateEstimator()
        self.state = WorldState(0.0)
        self.puck_hsv_lower = np.array([0, 0, 0])
        self.puck_hsv_upper = np.array([179, 255, 65])

    def detect_robots_and_puck(self, frame, timestamp=None):
        import time

        timestamp = time.monotonic() if timestamp is None else timestamp
        if self.homography.H is None:
            raise ValueError("Calibrate the arena before detection")
        if self.camera_matrix is not None:
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        robots = {}
        # Convention: marker TOP edge points toward the robot's front.
        if ids is not None:
            for points, idx in zip(corners, ids.flatten()):
                if int(idx) not in range(6):
                    continue
                pts = points[0]
                h = self.robot_homography
                x, y = h.px_to_cm(*pts.mean(axis=0))
                fx, fy = h.px_to_cm(*((pts[0] + pts[1]) / 2))
                robots[int(idx)] = dict(x=x, y=y, theta=math.atan2(fy - y, fx - x))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.puck_hsv_lower, self.puck_hsv_upper)
        a = self.homography.arena
        roi = np.zeros(gray.shape, np.uint8)
        poly = np.array(
            [
                self.homography.cm_to_px(x, y)
                for x, y in [
                    (-a.half_length, -a.half_width),
                    (a.half_length, -a.half_width),
                    (a.half_length, a.half_width),
                    (-a.half_length, a.half_width),
                ]
            ]
        )
        cv2.fillConvexPoly(roi, poly, 255)
        mask = cv2.bitwise_and(mask, roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        prior = self.estimator.snapshot(timestamp).puck
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, True)
            if area < 8 or perim <= 0:
                continue
            circularity = 4 * math.pi * area / (perim * perim)
            if circularity < 0.80:
                continue
            pts = cv2.perspectiveTransform(cnt.astype(np.float32), self.homography.H)
            metric_area = cv2.contourArea(pts)
            expected = math.pi * a.puck_radius**2
            if not 0.45 * expected < metric_area < 1.8 * expected:
                continue
            m = cv2.moments(cnt)
            x, y = self.homography.px_to_cm(m["m10"] / m["m00"], m["m01"] / m["m00"])
            # Reject blobs on identified robot bodies; puck stays outside their centres.
            if any(math.hypot(x - r["x"], y - r["y"]) < 6 for r in robots.values()):
                continue
            dist = math.hypot(x - prior.x, y - prior.y)
            if prior.valid(timestamp) and dist > max(8.0, 350 * max(0.02, timestamp - prior.last_seen)):
                continue
            score = (
                circularity
                - abs(metric_area / expected - 1) * 0.3
                - (dist * 0.02 if prior.valid(timestamp) else 0)
            )
            candidates.append((score, dict(x=x, y=y)))
        puck = max(candidates, key=lambda item: item[0])[1] if candidates else None
        self.state = self.estimator.update(robots, puck, timestamp)
        return self.state.robots, self.state.puck

    def build_team_obs_for_rl(
        self, robots_data=None, puck_data=None, previous_actions=None, team=0, now=None
    ):
        state = self.state if now is None else self.estimator.snapshot(now)
        return build_team_observation(
            state, self.homography.arena, team=team, previous_actions=previous_actions
        )
