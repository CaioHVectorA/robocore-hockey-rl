"""Metric overhead viewport shared by training frames, previews and replay."""

import pygame
from core.arena_geometry import arena_outline, arena_wall_segments
from ui.robot_renderer import draw_robot

BLUE, RED = (34, 113, 217), (213, 66, 80)


def projection(rect, arena):
    scale = min((rect.width - 80) / (arena.length + 2 * arena.goal_depth), (rect.height - 58) / arena.width)
    cx, cy = rect.center

    def world(x, y):
        return round(cx + x * scale), round(cy - y * scale)

    def metric(pos):
        return (pos[0] - cx) / scale, (cy - pos[1]) / scale

    return world, metric, scale


def draw_arena(screen, rect, arena, robot, frame, font, selected=0, tracking=False):
    world, _, scale = projection(rect, arena)
    a = arena
    pygame.draw.rect(screen, (21, 32, 48), rect, border_radius=16)
    # The pockets are part of the floor, with the same boundary used by Pymunk.
    # No closed outline across the mouth and no decorative net hiding solid walls.
    floor = [world(x, y) for x, y in arena_outline(a)]
    pygame.draw.polygon(screen, (51, 59, 72), floor)
    for sign in (-1, 1):
        p, q = (
            world(sign * a.half_length, a.goal_width / 2),
            world(sign * (a.half_length + a.goal_depth), -a.goal_width / 2),
        )
        goal = pygame.Rect(min(p[0], q[0]), min(p[1], q[1]), abs(q[0] - p[0]), abs(q[1] - p[1]))
        color = BLUE if sign == -1 else RED
        pygame.draw.rect(screen, (34, 59, 91) if sign == -1 else (84, 44, 56), goal)
        # A dashed scoring line is a reference marking, never a collider.
        for y in range(round(-a.goal_width / 2), round(a.goal_width / 2), 5):
            pygame.draw.line(
                screen,
                color,
                world(sign * a.half_length, y),
                world(sign * a.half_length, min(y + 2, a.goal_width / 2)),
                2,
            )
        label = pygame.transform.rotate(font.render("GOL", True, (223, 231, 242)), 90)
        screen.blit(label, label.get_rect(center=world(sign * (a.half_length + a.goal_depth / 2), 0)))
    pygame.draw.line(screen, (91, 108, 128), world(0, a.half_width), world(0, -a.half_width), 1)
    pygame.draw.circle(screen, (91, 108, 128), world(0, 0), round(20 * scale), 1)
    pygame.draw.circle(screen, (180, 193, 209), world(0, 0), 3)
    for start, end in arena_wall_segments(a):
        corner = (
            start[0] * end[0] > 0
            and start[1] * end[1] > 0
            and all(
                abs(x) > a.half_length - a.corner_radius - 0.01
                and abs(y) > a.half_width - a.corner_radius - 0.01
                for x, y in (start, end)
            )
        )
        color = (187, 83, 88) if corner else (185, 195, 207)
        # A shadow and highlight suggest the modular physical barrier in the reference.
        p, q = world(*start), world(*end)
        pygame.draw.line(screen, (13, 20, 30), (p[0] + 2, p[1] + 3), (q[0] + 2, q[1] + 3), 5)
        pygame.draw.line(screen, color, p, q, 3)
    for x, y, theta, team, role in frame["robots"]:
        draw_robot(screen, world, x, y, theta, BLUE if team == 0 else RED, robot)
        centre = world(x, y)
        if team == 0 and role == selected:
            pygame.draw.circle(screen, (255, 155, 72), centre, round(12 * scale), 1)
        label = font.render(str(team * 3 + role), True, (255, 255, 255))
        screen.blit(label, label.get_rect(center=centre))
    if tracking:
        for x, y, vx, vy in frame.get("tracking", []):
            pygame.draw.circle(screen, (0, 151, 117), world(x, y), round(11 * scale), 1)
            pygame.draw.line(screen, (0, 151, 117), world(x, y), world(x + vx * 0.15, y + vy * 0.15), 1)
    puck = world(*frame["puck"])
    pygame.draw.circle(
        screen, (75, 95, 102), (puck[0] + 1, puck[1] + 2), max(2, round(a.puck_radius * scale))
    )
    pygame.draw.circle(screen, (21, 32, 39), puck, max(2, round(a.puck_radius * scale)))
    pygame.draw.circle(screen, (255, 183, 91), puck, max(2, round(a.puck_radius * scale)), 1)
