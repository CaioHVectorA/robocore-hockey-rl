"""One closed rink boundary shared by collision walls and the overhead drawing."""

import math


def arena_outline(arena, corner_steps=12):
    """Rounded rectangle with open, three-wall goal recesses at both short ends."""
    hl, hw, radius = arena.half_length, arena.half_width, arena.corner_radius
    gw, depth = arena.goal_width / 2, arena.goal_depth
    points = [(hl, -gw), (hl, -hw + radius)]

    def arc(cx, cy, start):
        for i in range(1, corner_steps + 1):
            angle = math.radians(start - i * 90 / corner_steps)
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

    arc(hl - radius, -hw + radius, 0)
    points.append((-hl + radius, -hw))
    arc(-hl + radius, -hw + radius, -90)
    points.extend([(-hl, -gw), (-hl - depth, -gw), (-hl - depth, gw), (-hl, gw), (-hl, hw - radius)])
    arc(-hl + radius, hw - radius, 180)
    points.append((hl - radius, hw))
    arc(hl - radius, hw - radius, 90)
    points.extend([(hl, gw), (hl + depth, gw), (hl + depth, -gw)])
    return tuple(points)


def arena_wall_segments(arena):
    points = arena_outline(arena)
    return tuple(zip(points, points[1:] + points[:1]))
