"""Authentic Pymunk raycast distance sensors (7x ToF/IR) and optical line sensors (4x QRE1113)."""

from __future__ import annotations
import math
from typing import List, Tuple
import numpy as np
import pymunk

from sumo.constants import SUMO_OBS_DIM, SumoRobotConfig
from sumo.dohyo import Dohyo
from sumo.physics import SumoRobotBody


class SumoSensorPack:
    """Computes onboard sensor readings using true Pymunk segment queries against chassis geometry."""

    def __init__(self, config: SumoRobotConfig, dohyo: Dohyo):
        self.config = config
        self.dohyo = dohyo

        # Onboard firmware memory (derived 100% from sensors, zero omniscient cheats)
        self.last_seen_dir: float = 1.0  # +1.0 = Left (+y), -1.0 = Right (-y), 0.0 = Center
        self.time_since_seen: float = 0.5  # [0.0 = locked on now .. 1.0 = lost > 1.5s]
        self.prev_min_front_dist: float = 1.0
        self.closing_speed_norm: float = 0.0
        self.front_border_timer: float = 0.0
        self.rear_border_timer: float = 0.0

        # Cached ray geometry for high-definition visualization
        self.last_line_readings = np.zeros(4, dtype=np.float32)
        self.last_dist_readings = np.ones(7, dtype=np.float32)
        self.ray_origins_world: List[Tuple[float, float]] = []
        self.ray_endpoints_world: List[Tuple[float, float]] = []
        self.ray_hits: List[bool] = []
        self._last_pose_key: Tuple[float, ...] | None = None

    def reset(self, initial_search_dir: float = 1.0):
        """Reset onboard sensor memory at the start of a round."""
        self.last_seen_dir = 1.0 if initial_search_dir >= 0.0 else -1.0
        self.time_since_seen = 0.5
        self.prev_min_front_dist = 1.0
        self.closing_speed_norm = 0.0
        self.front_border_timer = 0.0
        self.rear_border_timer = 0.0
        self.last_line_readings.fill(0.0)
        self.last_dist_readings.fill(1.0)
        self.ray_origins_world.clear()
        self.ray_endpoints_world.clear()
        self.ray_hits.clear()
        self._last_pose_key = None

    def sensor_local_mounts(self, robot: SumoRobotBody) -> List[Tuple[float, float]]:
        """Return local (x, y) mounting points on the chassis perimeter for the 7 ToF sensors.

        Order:
        0: L90 (+85° Left Flank)
        1: L45 (+42° Wide Left)
        2: L15 (+15° Front-Left)
        3: C0  (  0° Front-Center)
        4: R15 (-15° Front-Right)
        5: R45 (-42° Wide Right)
        6: R90 (-85° Right Flank)
        """
        hl = robot.length / 2.0
        hw = robot.width / 2.0
        return [
            (0.0, hw * 0.92),          # 0: L90
            (hl * 0.72, hw * 0.85),    # 1: L45
            (hl * 0.92, hw * 0.42),    # 2: L15
            (hl * 0.95, 0.0),          # 3: C0
            (hl * 0.92, -hw * 0.42),   # 4: R15
            (hl * 0.72, -hw * 0.85),   # 5: R45
            (0.0, -hw * 0.92),         # 6: R90
        ]

    def read_line_sensors(self, robot: SumoRobotBody, dt: float = 1.0 / 30.0) -> np.ndarray:
        """Return 4 normalized corner reflectance readings [0.0 (black) .. 1.0 (white border)]."""
        readings = np.zeros(4, dtype=np.float32)
        for i, (lx, ly) in enumerate(robot.config.line_sensor_offsets):
            wx, wy = robot.get_world_point(lx, ly)
            readings[i] = self.dohyo.line_sensor_intensity(wx, wy)

        # Update onboard border escape timers
        if readings[0] > 0.35 or readings[1] > 0.35:
            self.front_border_timer = 1.0
        else:
            self.front_border_timer = max(0.0, self.front_border_timer - dt * 2.5)

        if readings[2] > 0.35 or readings[3] > 0.35:
            self.rear_border_timer = 1.0
        else:
            self.rear_border_timer = max(0.0, self.rear_border_timer - dt * 2.5)

        self.last_line_readings = readings
        return readings

    def read_distance_sensors(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        dt: float = 1.0 / 30.0,
    ) -> np.ndarray:
        """Cast 7 ToF/IR beams against the opponent's Pymunk polygon shapes.

        Returns 7 normalized distances in [0.0 (touching bumper) .. 1.0 (clear / max range)].
        """
        pose_key = (
            round(robot.x, 4),
            round(robot.y, 4),
            round(robot.theta, 4),
            round(opponent.x, 4),
            round(opponent.y, 4),
            round(opponent.theta, 4),
        )
        if self._last_pose_key == pose_key and len(self.ray_origins_world) == 7:
            return self.last_dist_readings
        self._last_pose_key = pose_key

        cfg = robot.config
        max_rng = cfg.distance_sensor_range
        angles_rad = cfg.distance_sensor_angles_rad
        mounts = self.sensor_local_mounts(robot)

        # Filter so the raycast ONLY hits the opponent's shapes (ignores self)
        opp_filter = pymunk.ShapeFilter(group=robot.id + 10)

        readings = np.ones(7, dtype=np.float32)
        self.ray_origins_world.clear()
        self.ray_endpoints_world.clear()
        self.ray_hits.clear()

        # Sensor direction weights (+ is Left, - is Right)
        dir_weights = (+1.0, +0.75, +0.35, 0.0, -0.35, -0.75, -1.0)
        best_hit_idx = -1
        best_hit_dist = 1.0

        # Sub-ray spread (-4.5°, 0°, +4.5°) models realistic 9° optical cone of VL53L0X ToF sensor
        sub_offsets = (-0.08, 0.0, 0.08)

        for i, base_ang in enumerate(angles_rad):
            ox, oy = robot.get_world_point(*mounts[i])
            ray_world_ang = robot.theta + base_ang

            min_alpha = 1.0
            hit_pt = (
                ox + math.cos(ray_world_ang) * max_rng,
                oy + math.sin(ray_world_ang) * max_rng,
            )
            did_hit = False

            for d_ang in sub_offsets:
                ang = ray_world_ang + d_ang
                ex = ox + math.cos(ang) * max_rng
                ey = oy + math.sin(ang) * max_rng
                info = robot.space.segment_query_first((ox, oy), (ex, ey), 0.35, opp_filter)
                if info is not None and info.alpha < min_alpha:
                    min_alpha = max(0.0, float(info.alpha))
                    hit_pt = (
                        ox + math.cos(ang) * (min_alpha * max_rng),
                        oy + math.sin(ang) * (min_alpha * max_rng),
                    )
                    did_hit = True

            readings[i] = min_alpha
            self.ray_origins_world.append((ox, oy))
            self.ray_endpoints_world.append(hit_pt)
            self.ray_hits.append(did_hit)

            if did_hit and min_alpha < best_hit_dist:
                best_hit_dist = min_alpha
                best_hit_idx = i

        # Update onboard target tracking memory
        if best_hit_idx >= 0:
            self.time_since_seen = 0.0
            # Weighted average of all triggered sensors for smooth directional memory
            active_weights = []
            active_strengths = []
            for i in range(7):
                if readings[i] < 0.98:
                    strength = 1.0 - readings[i]
                    active_weights.append(dir_weights[i] * strength)
                    active_strengths.append(strength)
            if sum(active_strengths) > 1e-4:
                mean_dir = sum(active_weights) / sum(active_strengths)
                if abs(mean_dir) < 0.08 and readings[3] < 0.98:
                    self.last_seen_dir = 0.0
                else:
                    self.last_seen_dir = float(np.clip(mean_dir * 1.4, -1.0, 1.0))
        else:
            self.time_since_seen = min(1.0, self.time_since_seen + dt / 1.2)
            # Keep last_seen_dir latched to +/-1 so search spin continues in the direction enemy vanished
            if abs(self.last_seen_dir) < 0.2:
                self.last_seen_dir = 1.0 if self.last_seen_dir >= 0.0 else -1.0

        # Estimate closing speed on front 3 sensors (L15, C0, R15 -> indices 2, 3, 4)
        min_front = float(np.min(readings[2:5]))
        if min_front < 0.98 and self.prev_min_front_dist < 0.98:
            raw_closing = (self.prev_min_front_dist - min_front) * max_rng / max(1e-3, dt)
            self.closing_speed_norm = float(np.clip(raw_closing / 200.0, -1.0, 1.0))
        else:
            self.closing_speed_norm = 0.0
        self.prev_min_front_dist = min_front

        self.last_dist_readings = readings
        return readings

    def get_observation(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        prev_action: Tuple[float, float],
        dt: float = 1.0 / 30.0,
    ) -> np.ndarray:
        """Build the 24-dimensional sensor-driven observation vector (zero GPS cheating).

        Index map:
        [0..3]   : 4 Corner Line Sensors (FL, FR, RL, RR) in [0, 1]
        [4..10]  : 7 ToF Distance Sensors (L90, L45, L15, C0, R15, R45, R90) in [0, 1]
        [11..13] : Target Memory (last_seen_dir [-1..1], time_since_seen [0..1], closing_speed [-1..1])
        [14..16] : Onboard IMU / Encoders (v_forward / 150, v_lateral / 150, omega / 15)
        [17..19] : Contact & Wedge Telemetry (in_contact [0/1], wedge_state [-1/0/+1], motor_load [0..1])
        [20..21] : Border Escape Timers (front_border_timer, rear_border_timer)
        [22..23] : Previous Motor Command (left_pwm, right_pwm)
        """
        line = self.read_line_sensors(robot, dt)
        dist = self.read_distance_sensors(robot, opponent, dt)

        v_fwd = float(np.clip(robot.forward_speed / 150.0, -2.0, 2.0))
        v_lat = float(np.clip(robot.lateral_speed / 150.0, -2.0, 2.0))
        omega = float(np.clip(robot.omega / 15.0, -2.0, 2.0))

        in_contact = 1.0 if robot.in_contact else 0.0
        wedge_state = 1.0 if robot.wedge_advantage else (-1.0 if robot.is_lifted else 0.0)
        motor_load = float(np.clip(robot.motor_load, 0.0, 1.0))

        obs = np.array(
            [
                line[0], line[1], line[2], line[3],
                dist[0], dist[1], dist[2], dist[3], dist[4], dist[5], dist[6],
                self.last_seen_dir,
                self.time_since_seen,
                self.closing_speed_norm,
                v_fwd,
                v_lat,
                omega,
                in_contact,
                wedge_state,
                motor_load,
                self.front_border_timer,
                self.rear_border_timer,
                float(prev_action[0]),
                float(prev_action[1]),
            ],
            dtype=np.float32,
        )
        assert obs.shape == (SUMO_OBS_DIM,)
        return obs
