"""Circular Dohyo geometry, line detection and raycast math."""

import math
from typing import Optional, Tuple
from sumo.constants import DohyoConfig


class Dohyo:
    """Mathematical and geometric representation of the circular Mini-Sumo Dohyo."""

    def __init__(self, config: Optional[DohyoConfig] = None):
        self.config = config or DohyoConfig()
        self.radius = self.config.radius
        self.inner_radius = self.config.inner_radius
        self.border_width = self.config.border_width
        self.friction = self.config.friction

    def is_inside(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Return True if point (x, y) is within the Dohyo outer boundary."""
        return math.hypot(x, y) <= (self.radius - margin)

    def is_on_white_line(self, x: float, y: float) -> bool:
        """Return True if point (x, y) is on the white Tawaramono border line."""
        r = math.hypot(x, y)
        return self.inner_radius <= r <= self.radius

    def line_sensor_intensity(self, x: float, y: float) -> float:
        """Continuous [0, 1] reading of white line reflectivity under an IR sensor."""
        r = math.hypot(x, y)
        if r < self.inner_radius:
            return 0.0
        elif r <= self.radius:
            # Transition cleanly across the 2.5 cm white border
            frac = (r - self.inner_radius) / self.border_width
            return min(1.0, max(0.0, frac * 1.5))
        else:
            # Off the ring (sensor detects void/floor drop)
            return 1.0

    def radial_distance(self, x: float, y: float) -> float:
        """Distance from center of Dohyo."""
        return math.hypot(x, y)

    def normalized_radial_pos(self, x: float, y: float) -> float:
        """Radial position normalized to [0, 1] where 1.0 is the outer boundary."""
        return min(1.5, math.hypot(x, y) / self.radius)

    def starting_positions(self, layout: str = "standard") -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return initial (x, y, theta) poses for robot 0 (Blue) and robot 1 (Red).

        Layouts:
        - standard: facing each other on Shikiri lines (separated by 10 cm)
        - side_by_side: parallel facing opposite directions
        - back_to_back: touching backs, facing away
        - angle_45: angled approach
        """
        sep = self.config.shikiri_separation / 2.0  # 5.0 cm from center
        if layout == "standard":
            # Robot 0 at (-5, 0) facing right (0 rad), Robot 1 at (5, 0) facing left (pi rad)
            return ((-sep, 0.0, 0.0), (sep, 0.0, math.pi))
        elif layout == "back_to_back":
            return ((-sep, 0.0, math.pi), (sep, 0.0, 0.0))
        elif layout == "side_by_side":
            return ((0.0, -sep, 0.0), (0.0, sep, math.pi))
        elif layout == "angled":
            return ((-sep, -sep, math.pi / 4), (sep, sep, -3 * math.pi / 4))
        else:
            return ((-sep, 0.0, 0.0), (sep, 0.0, math.pi))
