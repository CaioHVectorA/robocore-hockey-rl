"""Metric overhead viewport shared by training frames, previews and replay."""

import pygame
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
    rink = pygame.Rect(0, 0, round(a.length * scale), round(a.width * scale))
    rink.center = rect.center
    pygame.draw.rect(screen, (222, 233, 235), rink, border_radius=round(a.corner_radius * scale))
    pygame.draw.rect(screen, (187, 207, 214), rink, width=3, border_radius=round(a.corner_radius * scale))
    for sign in (-1, 1):
        p, q = (
            world(sign * a.half_length, a.goal_width / 2),
            world(sign * (a.half_length + a.goal_depth), -a.goal_width / 2),
        )
        goal = pygame.Rect(min(p[0], q[0]), min(p[1], q[1]), abs(q[0] - p[0]), abs(q[1] - p[1]))
        pygame.draw.rect(screen, BLUE if sign == -1 else RED, goal, border_radius=3)
        for yy in range(goal.top, goal.bottom, 7):
            pygame.draw.line(screen, (151, 176, 190), (goal.left, yy), (goal.right, yy), 1)
        pygame.draw.circle(screen, (123, 177, 200), world(sign * a.half_length, 0), round(22 * scale), 2)
    pygame.draw.line(screen, (195, 88, 103), world(0, a.half_width), world(0, -a.half_width), 2)
    pygame.draw.circle(screen, (115, 158, 182), world(0, 0), round(20 * scale), 2)
    pygame.draw.circle(screen, (195, 88, 103), world(0, 0), 3)
    for sign in (-1, 1):
        pygame.draw.line(
            screen,
            (149, 183, 201),
            world(sign * a.length * 0.25, -a.half_width),
            world(sign * a.length * 0.25, a.half_width),
            1,
        )
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
