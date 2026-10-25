"""Classic competition heuristic bots for Mini-Sumo sparring and evaluation."""

from typing import Tuple

from sumo.physics import SumoRobotBody
from sumo.sensors import SumoSensorPack


class BaseSumoBot:
    """Interface for rule-based sparring opponents."""

    def __init__(self, bot_id: int = 1):
        self.bot_id = bot_id

    def get_action(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        sensors: SumoSensorPack,
    ) -> Tuple[float, float]:
        raise NotImplementedError


class BullRushBot(BaseSumoBot):
    """Aggressive direct charge straight ahead."""

    def get_action(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        sensors: SumoSensorPack,
    ) -> Tuple[float, float]:
        line = sensors.read_line_sensors(robot)
        # If front hits white line, emergency reverse
        if line[0] > 0.5 or line[1] > 0.5:
            return (-1.0, -1.0)
        # Otherwise full speed forward
        return (1.0, 1.0)


class SearchAndDestroyBot(BaseSumoBot):
    """Classic Japanese tournament bot: edge avoidance + search spin + tracking ram."""

    def __init__(self, bot_id: int = 1, search_direction: float = 1.0):
        super().__init__(bot_id)
        self.search_dir = search_direction
        self.backing_up = 0

    def get_action(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        sensors: SumoSensorPack,
    ) -> Tuple[float, float]:
        line = sensors.read_line_sensors(robot)

        # 1. Edge avoidance has highest priority
        if self.backing_up > 0:
            self.backing_up -= 1
            return (-1.0, -0.6 if self.search_dir > 0 else -1.0)

        if line[0] > 0.4 or line[1] > 0.4:  # Front hit white border
            self.backing_up = 10  # Back up for 10 frames (~0.33s)
            return (-1.0, -0.5)

        if line[2] > 0.4 or line[3] > 0.4:  # Rear hit white border
            return (1.0, 0.8)

        # 2. Distance sensor checks
        dist = sensors.read_distance_sensors(robot, opponent)
        # dist: [-60°, -25°, 0°, +25°, +60°]
        left_wide, left_mid, center, right_mid, right_wide = dist

        # Center locked on -> Full power charge!
        if center < 0.95:
            return (1.0, 1.0)

        # Enemy on left diagonal
        if left_mid < 0.95:
            return (0.3, 1.0)

        # Enemy on right diagonal
        if right_mid < 0.95:
            return (1.0, 0.3)

        # Enemy on wide left
        if left_wide < 0.95:
            self.search_dir = -1.0
            return (-0.8, 0.8)

        # Enemy on wide right
        if right_wide < 0.95:
            self.search_dir = 1.0
            return (0.8, -0.8)

        # 3. Not detected -> Spin search in current direction
        if self.search_dir > 0:
            return (0.7, -0.7)  # Clockwise spin
        else:
            return (-0.7, 0.7)  # Counter-clockwise spin


class MatadorBot(BaseSumoBot):
    """Tactical bot that dodges charging enemies then counter-attacks their flank."""

    def __init__(self, bot_id: int = 1):
        super().__init__(bot_id)
        self.cooldown = 0

    def get_action(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        sensors: SumoSensorPack,
    ) -> Tuple[float, float]:
        line = sensors.read_line_sensors(robot)
        if line[0] > 0.4 or line[1] > 0.4:
            return (-1.0, -0.7)

        dist = sensors.read_distance_sensors(robot, opponent)
        center = dist[2]

        # If enemy is rushing close straight ahead and cooldown expired -> Sidestep!
        if center < 0.40 and opponent.speed > 50.0 and self.cooldown <= 0:
            self.cooldown = 15
            return (-0.8, 1.0)  # Sharp dodge to the right

        if self.cooldown > 0:
            self.cooldown -= 1
            return (1.0, 0.6)  # Flank wrap-around

        # Default: Search and destroy behavior
        if center < 0.95:
            return (1.0, 1.0)
        return (0.6, -0.6)
