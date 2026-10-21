"""Chassis outline shared by physics, rendering and CAD, in centimetres."""

from core.constants import RobotConfig


def chassis_outline(role=0, robot=None):
    robot = robot or RobotConfig()
    hl, hw = robot.length / 2, robot.width / 2
    depth = (1.2, 1.0, 0.6)[role % 3]
    mouth = min((8.0, 9.0, 11.0)[role % 3], robot.width - 2)
    return [
        (-hl, -hw),
        (hl, -hw),
        (hl, -mouth / 2),
        (hl - depth, -mouth * 0.35),
        (hl - depth, mouth * 0.35),
        (hl, mouth / 2),
        (hl, hw),
        (-hl, hw),
    ]


def convex_chassis_parts(role=0, robot=None):
    p = chassis_outline(role, robot)
    back = p[3][0]
    return [
        [p[0], (back, p[0][1]), (back, p[-1][1]), p[-1]],
        [(back, p[0][1]), p[1], p[2], p[3]],
        [p[4], p[5], p[6], (back, p[-1][1])],
    ]
