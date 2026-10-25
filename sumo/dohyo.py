"""Dohyo circular arena geometry, Tawaramono white border line, and Shikiri start poses."""

import math
from typing import Tuple

from sumo.constants import DohyoConfig


class Dohyo:
    """Represents the 77 cm official Mini-Sumo circular Dohyo."""

    def __init__(self, config: DohyoConfig | None = None):
        self.config = config or DohyoConfig()
        self.radius = self.config.radius
        self.inner_radius = self.config.inner_radius
        self.border_width = self.config.border_width

    def is_inside(self, x: float, y: float) -> bool:
        """Return True if point (x, y) is within the outer edge of the Dohyo."""
        return math.hypot(x, y) <= self.radius

    def line_sensor_intensity(self, x: float, y: float) -> float:
        """Compute optical reflectivity [0.0 (black surface) .. 1.0 (white Tawaramono border)].

        Models a QRE1113 / TCRT5000 infrared reflectance sensor:
        - Inside r < 35.8 cm: 0.0 (matte black polyurethane/steel surface)
        - Across 35.8 <= r <= 38.5 cm: smooth high-contrast rise to 1.0 on the 2.5 cm white line
        - Outside r > 38.5 cm: 1.0 (ring drop-off / out of bounds)
        """
        r = math.hypot(x, y)
        if r < self.inner_radius - 0.2:
            return 0.0
        if r <= self.radius:
            frac = (r - (self.inner_radius - 0.2)) / (self.border_width + 0.2)
            return min(1.0, max(0.0, frac * 1.6))
        return 1.0

    def radial_distance(self, x: float, y: float) -> float:
        """Distance from center (0, 0) of the Dohyo in cm."""
        return math.hypot(x, y)

    def normalized_radial_pos(self, x: float, y: float) -> float:
        """Radial position normalized to [0, 1] where 1.0 is the outer edge."""
        return min(1.5, math.hypot(x, y) / self.radius)

    def starting_positions(
        self, layout: str = "standard"
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return official non-overlapping (x, y, theta) poses behind the Shikiri lines.

        The two Shikiri lines are at x = -5.0 cm and x = +5.0 cm.
        Since a 10 cm robot has half-length 5.0 cm, placing its center at x = +-13.0 cm
        puts its front blade at x = +-8.0 cm (3 cm safely behind the Shikiri line,
        with 16.0 cm clear distance between the two robots).
        """
        if layout == "standard":
            # Face-to-face behind Shikiri lines
            return ((-13.5, 0.0, 0.0), (13.5, 0.0, math.pi))
        if layout == "back_to_back":
            # Facing away from each other toward the borders
            return ((-12.5, 0.0, math.pi), (12.5, 0.0, 0.0))
        if layout == "side_by_side":
            # Offset laterally, facing opposite directions (classic pass/flank start)
            return ((-12.0, -8.5, 0.0), (12.0, 8.5, math.pi))
        if layout == "angled":
            # Diagonal approach
            return ((-14.0, -9.0, math.radians(28.0)), (14.0, 9.0, math.radians(-152.0)))
        if layout == "parallel_north":
            # Both robots facing +y (+90 deg) behind their Shikiri lines
            return ((-12.5, -2.0, math.pi / 2.0), (12.5, -2.0, math.pi / 2.0))
        return ((-13.5, 0.0, 0.0), (13.5, 0.0, math.pi))
