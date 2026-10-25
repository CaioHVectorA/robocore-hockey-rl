"""Simulated optical line sensors (QRE1113) and distance sensors (ToF/Sharp) for Mini-Sumo."""

import math
from typing import Tuple
import numpy as np

from sumo.constants import SumoRobotConfig
from sumo.dohyo import Dohyo
from sumo.physics import SumoRobotBody


class SumoSensorPack:
    """Computes all on-board sensor readings for a Mini-Sumo robot."""

    def __init__(self, config: SumoRobotConfig, dohyo: Dohyo):
        self.config = config
        self.dohyo = dohyo
        self.last_sighting_side = 0.0  # -1.0 (left), 0.0 (ahead), +1.0 (right)

    def read_line_sensors(self, robot: SumoRobotBody) -> np.ndarray:
        """Return 4 normalized readings [0.0 (black), 1.0 (white/void)] for corners."""
        readings = []
        for lx, ly in self.config.line_sensor_offsets:
            wx, wy = robot.get_world_point(lx, ly)
            intensity = self.dohyo.line_sensor_intensity(wx, wy)
            readings.append(intensity)
        return np.array(readings, dtype=np.float32)

    def read_distance_sensors(self, robot: SumoRobotBody, opponent: SumoRobotBody) -> np.ndarray:
        """Return 5 normalized distance readings [0.0 (touching) .. 1.0 (clear/max range)].

        Angles: -60°, -25°, 0°, +25°, +60° relative to robot heading.
        """
        readings = []
        max_rng = self.config.distance_sensor_range
        cone = self.config.distance_sensor_cone_rad

        # Vector to opponent center
        dx, dy = opponent.x - robot.x, opponent.y - robot.y
        dist_opp = math.hypot(dx, dy)
        bearing_to_opp = math.atan2(dy, dx) - robot.theta
        # Wrap bearing to [-pi, pi]
        bearing_to_opp = (bearing_to_opp + math.pi) % (2 * math.pi) - math.pi

        for sensor_ang in self.config.distance_sensor_angles_rad:
            ang_diff = abs(bearing_to_opp - sensor_ang)
            if ang_diff > math.pi:
                ang_diff = 2 * math.pi - ang_diff

            # If opponent is within sensor detection cone and range
            if ang_diff <= cone and dist_opp <= max_rng:
                # Approximate distance to chassis edge: dist - radius
                contact_d = max(0.0, dist_opp - opponent.width * 0.5)
                norm_d = min(1.0, contact_d / max_rng)
                readings.append(norm_d)

                # Update sighting side
                if sensor_ang < -0.1:
                    self.last_sighting_side = -1.0  # Enemy is on left
                elif sensor_ang > 0.1:
                    self.last_sighting_side = 1.0   # Enemy is on right
                else:
                    self.last_sighting_side = 0.0   # Enemy is straight ahead
            else:
                readings.append(1.0)  # Max range / no echo

        return np.array(readings, dtype=np.float32)

    def get_observation(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        prev_action: Tuple[float, float],
    ) -> np.ndarray:
        """Build the full 18-dimensional normalized observation vector for RL.

        Features:
        [0..3]  : 4 line sensors (FL, FR, RL, RR) in [0, 1]
        [4..8]  : 5 distance sensors (-60, -25, 0, +25, +60) in [0, 1]
        [9..11] : Self velocities: vx, vy (scaled / 150 cm/s), omega (scaled / 15 rad/s)
        [12..13]: Relative polar position of opponent (normalized distance, sin/cos bearing)
        [14]    : Last sighting memory [-1.0 (left), 0.0, 1.0 (right)]
        [15..16]: Previous action (left_pwm, right_pwm) in [-1, 1]
        [17]    : Wedge engagement flag (1.0 if under opponent, -1.0 if lifted, 0.0 neutral)
        """
        line = self.read_line_sensors(robot)
        dist = self.read_distance_sensors(robot, opponent)

        vx_norm = robot.vx / 150.0
        vy_norm = robot.vy / 150.0
        om_norm = robot.omega / 15.0

        # Opponent relative polar
        dx, dy = opponent.x - robot.x, opponent.y - robot.y
        rel_dist = min(2.0, math.hypot(dx, dy) / self.dohyo.radius)
        rel_bearing = (math.atan2(dy, dx) - robot.theta + math.pi) % (2 * math.pi) - math.pi
        sin_b = math.sin(rel_bearing)

        wedge_status = 1.0 if robot.wedge_advantage else (-1.0 if robot.is_lifted else 0.0)

        obs = np.array(
            [
                line[0], line[1], line[2], line[3],
                dist[0], dist[1], dist[2], dist[3], dist[4],
                vx_norm, vy_norm, om_norm,
                rel_dist, sin_b,
                self.last_sighting_side,
                prev_action[0], prev_action[1],
                wedge_status,
            ],
            dtype=np.float32,
        )
        return obs
