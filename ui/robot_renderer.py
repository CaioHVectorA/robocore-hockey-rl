"""Top view of the stock reference, using the physical outline and tyres."""

import math
from pathlib import Path
import pygame
from core.constants import RobotConfig
from core.geometry import chassis_outline, deck_outline, wheel_footprints


def draw_robot(surface, world, x, y, theta, color, robot):
    c, s = math.cos(theta), math.sin(theta)

    def point(lx, ly):
        return world(x + lx * c - ly * s, y + lx * s + ly * c)

    def polygon(points, fill, border=None):
        pixels = [point(*p) for p in points]
        pygame.draw.polygon(surface, fill, pixels)
        if border:
            pygame.draw.polygon(surface, border, pixels, 1)
        return pixels

    outline = chassis_outline(robot=robot)
    pixels = [point(*p) for p in outline]
    pygame.draw.polygon(surface, (111, 135, 146), [(px + 2, py + 3) for px, py in pixels])
    for wheel in wheel_footprints(robot):
        polygon(wheel, (31, 35, 32), (11, 18, 22))
        # Tread marks run across the tyre, with the same physical dimensions.
        cy = sum(p[1] for p in wheel) / 4
        for fraction in (-0.65, -0.3, 0, 0.3, 0.65):
            wx = robot.axle_x + fraction * robot.wheel_diameter / 2
            pygame.draw.line(
                surface,
                (54, 58, 51),
                point(wx, cy - robot.wheel_width * 0.4),
                point(wx, cy + robot.wheel_width * 0.4),
                1,
            )
    polygon(outline, tuple(round(v * 0.80) for v in color), (17, 40, 62))
    polygon(deck_outline(robot), color, (17, 40, 62))
    # Screws and the central cover opening are visual references, not sensors.
    pixel_scale = math.dist(point(0, 0), point(1, 0))
    for lx, ly in ((3.65, -5.4), (3.65, 5.4), (-2.1, -6.7), (-2.1, 6.7)):
        pygame.draw.circle(
            surface,
            (174, 192, 184),
            point(lx * robot.length / 16.4, ly * robot.width / 14.4),
            max(1, round(0.16 * pixel_scale)),
        )
    opening = [(-0.7, -1.15), (0.7, -1.15), (0.7, 1.15), (-0.7, 1.15)]
    polygon(opening, (30, 39, 48))


def save_reference_view(path):
    """Reproducible close-up, independent of the GUI camera zoom."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pygame.init()
    surface = pygame.Surface((760, 820))
    surface.fill((233, 240, 242))
    font = pygame.font.SysFont("DejaVu Sans", 23, bold=True)
    small = pygame.font.SysFont("DejaVu Sans", 17)
    surface.blit(font.render("HockeyBot comercial · referência 2D", True, (20, 40, 56)), (36, 25))
    surface.blit(
        small.render("Contorno estimado por foto; medidas do chassi a confirmar", True, (65, 83, 94)),
        (36, 62),
    )
    scale = 32

    def world(x, y):
        return round(380 + x * scale), round(390 - y * scale)

    cfg = RobotConfig()
    draw_robot(surface, world, 0, 0, -math.pi / 2, (16, 143, 196), cfg)
    puck = world(0, -6.3)
    pygame.draw.circle(surface, (26, 41, 48), puck, round(1.585 * scale))
    pygame.draw.circle(surface, (84, 110, 121), puck, round(1.585 * scale), 2)
    labels = (
        "Mesmo chassi para atacante, ala e goleiro",
        "Fabricante: ~360 g · pneus 60 × 20 mm · motor 6 V / 266 rpm",
        "Estimativa: envelope 164 × 144 mm · bitola 118 mm",
        "Rampa, centro de massa, atritos e resposta ainda sem medição",
    )
    for i, label in enumerate(labels):
        surface.blit(small.render(label, True, (43, 64, 77)), (36, 704 + i * 25))
    pygame.image.save(surface, str(path))
    pygame.quit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    save_reference_view(parser.parse_args().output)
