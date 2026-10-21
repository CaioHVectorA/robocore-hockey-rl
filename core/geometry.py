"""Stock HockeyBot reference shared by physics, rendering and CAD (cm).

Symmetrised trace of the manufacturer's 3388_4_H photograph, not measured CAD.
See docs/HOCKEYBOT_STOCK.md for the scaling assumptions and uncertainty.
Local +x is forward, +y left. Tactical roles have identical hardware.
"""

from functools import lru_cache
import pymunk
from pymunk.autogeometry import convex_decomposition
from core.constants import RobotConfig

# Right half, from rear centre along outside, then into the curved front bay.
# Pixel coordinates refer to the original 600 x 600 manufacturer image.
PHOTO_HALF_OUTLINE = (
    (299.5, 110),
    (460, 110),
    (461, 216),
    (529, 218),
    (536.5, 225),
    (536.5, 530),
    (531, 538),
    (413, 538),
    (407, 535),
    (405, 528),
    (405, 518),
    (410, 510),
    (425, 504),
    (437, 493),
    (446, 477),
    (450, 461),
    (447, 448),
    (440, 436),
    (429, 427),
    (415, 421),
    (396, 419),
    (354, 419),
    (341, 422),
    (326, 427),
    (311, 430),
    (299.5, 431),
)


def _photo_to_local(point, robot):
    px, py = point
    return ((py - 295) * robot.length / 486, (299.5 - px) * robot.width / 474)


def chassis_outline(role=0, robot=None):
    """Concave lower plate; role is kept for callers, never changes the body."""
    robot = robot or RobotConfig()
    half = PHOTO_HALF_OUTLINE
    pixels = half + tuple((599 - x, y) for x, y in reversed(half[1:-1]))
    return [_photo_to_local(p, robot) for p in pixels]


def convex_chassis_parts(role=0, robot=None):
    # Copy cached tuples so callers cannot mutate the geometry of another env.
    return [list(p) for p in _convex_parts(robot or RobotConfig())]


@lru_cache(maxsize=128)
def _convex_parts(robot):
    points = chassis_outline(robot=robot)
    parts = convex_decomposition(points + points[:1], 0.0)
    return tuple(tuple(tuple(v) for v in p[:-1]) for p in parts)


def wheel_footprints(robot=None):
    robot = robot or RobotConfig()
    rx, ry = robot.wheel_diameter / 2, robot.wheel_width / 2
    return [
        [
            (robot.axle_x - rx, cy - ry),
            (robot.axle_x + rx, cy - ry),
            (robot.axle_x + rx, cy + ry),
            (robot.axle_x - rx, cy + ry),
        ]
        for cy in (robot.track / 2, -robot.track / 2)
    ]


def collision_parts(robot=None):
    return convex_chassis_parts(robot=robot) + wheel_footprints(robot)


def footprint_vertices(robot=None):
    return chassis_outline(robot=robot) + [p for wheel in wheel_footprints(robot) for p in wheel]


def deck_outline(robot=None):
    """Raised rear cover for display only; height/ramp is not simulated in 2D."""
    robot = robot or RobotConfig()
    half = PHOTO_HALF_OUTLINE[:5] + ((536.5, 405), (299.5, 405))
    return [_photo_to_local(p, robot) for p in half + tuple((599 - x, y) for x, y in reversed(half[1:-1]))]


def estimated_inertia(robot=None):
    """Residual mass spread over plate + two tyres; moment about assumed COM."""
    robot = robot or RobotConfig()
    parts = convex_chassis_parts(robot=robot)
    areas = [pymunk.area_for_poly(p) for p in parts]
    residual_mass = robot.mass - 2 * robot.tire_mass
    return sum(pymunk.moment_for_poly(residual_mass * a / sum(areas), p) for a, p in zip(areas, parts)) + sum(
        pymunk.moment_for_poly(robot.tire_mass, p) for p in wheel_footprints(robot)
    )
