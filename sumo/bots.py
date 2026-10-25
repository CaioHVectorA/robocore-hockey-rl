"""Pure sensor-driven competitive Mini-Sumo bots and Self-Play policy wrapper.

All heuristic bots operate strictly on their onboard sensor pack:
- 4 optical border sensors: [FL, FR, RL, RR]
- 7 ToF distance sensors: [L90 (+85°), L45 (+42°), L15 (+15°), C0 (0°), R15 (-15°), R45 (-42°), R90 (-85°)]
- Onboard sensor memory: last_seen_dir, closing_speed_norm
Zero world-coordinate (GPS) cheating is used.
"""

from __future__ import annotations
import math
from typing import Dict, Tuple
import numpy as np

from sumo.physics import SumoRobotBody
from sumo.sensors import SumoSensorPack


class BaseSumoBot:
    """Interface for sensor-based sparring bots and self-play opponents."""

    name: str = "BaseBot"
    style: str = "Generic"

    def __init__(self, bot_id: int = 1):
        self.bot_id = bot_id
        self.prev_action: Tuple[float, float] = (0.0, 0.0)

    def reset(self):
        self.prev_action = (0.0, 0.0)

    def get_action(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        sensors: SumoSensorPack,
    ) -> Tuple[float, float]:
        raise NotImplementedError

    @staticmethod
    def _track_with_7_sensors(
        dist: np.ndarray,
        last_seen_dir: float,
        spin_speed: float = 0.72,
    ) -> Tuple[float, float]:
        """Standard 7-sensor competitive PD tracking law.

        Sensor indices:
        0: L90 (+85° Left)   -> turn LEFT:  left < right, e.g. (-0.85, +0.90)
        1: L45 (+42° Left)   -> turn LEFT:  (-0.45, +0.95)
        2: L15 (+15° Left)   -> steer LEFT: (+0.55, +1.00)
        3: C0  (  0° Center) -> RAM:        (+1.00, +1.00)
        4: R15 (-15° Right)  -> steer RIGHT:(+1.00, +0.55)
        5: R45 (-42° Right)  -> turn RIGHT: (+0.95, -0.45)
        6: R90 (-85° Right)  -> turn RIGHT: (+0.90, -0.85)
        """
        l90, l45, l15, c0, r15, r45, r90 = dist

        # 1. Center / Dual-Front Lock -> Full Power Charge with PD centering!
        if c0 < 0.96 or (l15 < 0.96 and r15 < 0.96):
            # Positive diff means enemy is slightly more on Left (+y) -> boost right wheel
            left_pull = (1.0 - l15) * 0.45 + (1.0 - l45) * 0.25
            right_pull = (1.0 - r15) * 0.45 + (1.0 - r45) * 0.25
            steer = float(np.clip(left_pull - right_pull, -0.6, 0.6))
            # If steer > 0 (target on left), right_pwm = 1.0, left_pwm = 1.0 - steer
            left_pwm = 1.0 - max(0.0, steer)
            right_pwm = 1.0 - max(0.0, -steer)
            return (left_pwm, right_pwm)

        # 2. Front-Left (L15) triggered
        if l15 < 0.96:
            return (0.25, 1.0)

        # 3. Front-Right (R15) triggered
        if r15 < 0.96:
            return (1.0, 0.25)

        # 4. Wide-Left (L45) triggered -> pivot left onto target (no wide orbiting)
        if l45 < 0.96:
            return (-0.75, 0.88)

        # 5. Wide-Right (R45) triggered -> pivot right onto target (no wide orbiting)
        if r45 < 0.96:
            return (0.88, -0.75)

        # 6. Flank-Left (L90) triggered -> snap spin left!
        if l90 < 0.96:
            return (-0.90, 0.92)

        # 7. Flank-Right (R90) triggered -> snap spin right!
        if r90 < 0.96:
            return (0.92, -0.90)

        # 8. No sensor active -> Spin toward last seen direction
        if last_seen_dir >= 0.0:
            return (-spin_speed, spin_speed)  # Spin Left (CCW)
        return (spin_speed, -spin_speed)      # Spin Right (CW)


class BullRushBot(BaseSumoBot):
    """Aggressive charger with 2-stage border recovery and 7-sensor lock."""

    name = "BullRush"
    style = "Arrancada Explosiva + Lock Frontal"

    def __init__(self, bot_id: int = 1):
        super().__init__(bot_id)
        self.escape_timer = 0
        self.escape_turn_dir = 1.0

    def reset(self):
        super().reset()
        self.escape_timer = 0

    def get_action(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        sensors: SumoSensorPack,
    ) -> Tuple[float, float]:
        line = sensors.read_line_sensors(robot)
        dist = sensors.read_distance_sensors(robot, opponent)

        # Active border escape sequence
        if self.escape_timer > 0:
            self.escape_timer -= 1
            if self.escape_timer >= 4:
                act = (-1.0, -1.0)  # Full reverse brake
            else:
                # Pivot away from border
                act = (-0.8, 0.8) if self.escape_turn_dir > 0 else (0.8, -0.8)
            self.prev_action = act
            return act

        # Trigger border escape if front hits white line (unless pushing enemy out right at edge)
        if (line[0] > 0.35 or line[1] > 0.35) and dist[3] > 0.12:
            self.escape_timer = 9
            # If Front-Left (line[0]) hit white first, turn RIGHT (-1.0)
            self.escape_turn_dir = -1.0 if line[0] >= line[1] else 1.0
            self.prev_action = (-1.0, -1.0)
            return self.prev_action

        # BullRush locks on with narrow front cone (L15, C0, R15) and commits to full-speed charge!
        l15, c0, r15 = dist[2], dist[3], dist[4]
        if c0 < 0.96:
            act = (1.0, 1.0)
        elif l15 < 0.55:
            act = (0.75, 1.0)
        elif r15 < 0.55:
            act = (1.0, 0.75)
        else:
            act = (1.0, 1.0)
        self.prev_action = act
        return act


class SearchAndDestroyBot(BaseSumoBot):
    """All-Japan / Karioka style tournament bot: 7-sensor PD lock + memory spin + arc sweep."""

    name = "Karioka S&D"
    style = "7-Sensor PD Lock + Giro de Memória"

    def __init__(self, bot_id: int = 1, search_direction: float = 1.0):
        super().__init__(bot_id)
        self.default_search_dir = search_direction
        self.escape_timer = 0
        self.escape_turn_dir = 1.0
        self.blind_steps = 0

    def reset(self):
        super().reset()
        self.escape_timer = 0
        self.blind_steps = 0

    def get_action(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        sensors: SumoSensorPack,
    ) -> Tuple[float, float]:
        line = sensors.read_line_sensors(robot)
        dist = sensors.read_distance_sensors(robot, opponent)

        # 1. Execute ongoing border recovery
        if self.escape_timer > 0:
            self.escape_timer -= 1
            # Cancel escape early if enemy locks straight ahead at close range!
            if dist[3] < 0.30 and line[0] < 0.2 and line[1] < 0.2:
                self.escape_timer = 0
            elif self.escape_timer >= 4:
                self.prev_action = (-1.0, -1.0)
                return self.prev_action
            else:
                act = (-0.85, 0.85) if self.escape_turn_dir > 0 else (0.85, -0.85)
                self.prev_action = act
                return act

        # 2. Check front white border (unless shoving enemy right on the edge)
        if (line[0] > 0.35 or line[1] > 0.35) and not (robot.in_contact and dist[3] < 0.14):
            self.escape_timer = 8
            self.escape_turn_dir = -1.0 if line[0] >= line[1] else 1.0
            self.prev_action = (-1.0, -1.0)
            return self.prev_action

        # 3. Check rear white border -> surge forward
        if line[2] > 0.45 or line[3] > 0.45:
            self.prev_action = (1.0, 1.0)
            return self.prev_action

        # 4. Target tracking via 7 sensors
        if np.min(dist) < 0.96:
            self.blind_steps = 0
            act = self._track_with_7_sensors(dist, sensors.last_seen_dir, spin_speed=0.75)
        else:
            self.blind_steps += 1
            # Alternate between fast spin in last_seen_dir and expanding spiral arc
            s_dir = sensors.last_seen_dir if abs(sensors.last_seen_dir) > 0.1 else self.default_search_dir
            if (self.blind_steps % 28) < 16:
                act = (-0.72, 0.72) if s_dir >= 0 else (0.72, -0.72)
            else:
                act = (0.55, 0.92) if s_dir >= 0 else (0.92, 0.55)

        self.prev_action = act
        return act


class MatadorBot(BaseSumoBot):
    """Tactical flanker that sidesteps oncoming charges and hooks into the opponent's side."""

    name = "Matador Flanker"
    style = "Esquiva Lateral + Contra-Ataque de Flanco"

    def __init__(self, bot_id: int = 1, dodge_side: float = 1.0):
        super().__init__(bot_id)
        self.dodge_side = dodge_side  # +1.0 dodges left, -1.0 dodges right
        self.phase = "idle"
        self.phase_timer = 0
        self.cooldown = 0
        self.escape_timer = 0

    def reset(self):
        super().reset()
        self.phase = "idle"
        self.phase_timer = 0
        self.cooldown = 0
        self.escape_timer = 0

    def get_action(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        sensors: SumoSensorPack,
    ) -> Tuple[float, float]:
        line = sensors.read_line_sensors(robot)
        dist = sensors.read_distance_sensors(robot, opponent)

        if self.cooldown > 0:
            self.cooldown -= 1

        # Emergency border check
        if line[0] > 0.35 or line[1] > 0.35:
            self.phase = "idle"
            self.escape_timer = 7
            self.prev_action = (-1.0, -0.7 if line[0] > line[1] else -1.0)
            return self.prev_action

        if self.escape_timer > 0:
            self.escape_timer -= 1
            self.prev_action = (-0.85, -0.45) if self.dodge_side > 0 else (-0.45, -0.85)
            return self.prev_action

        # State machine for Matador sidestep + hook maneuver
        if self.phase == "sidestep":
            self.phase_timer -= 1
            if self.phase_timer <= 0:
                self.phase = "hook"
                self.phase_timer = 7
            # First 3 frames: angle out at ~40 deg; next 4 frames: sprint diagonally forward!
            if self.phase_timer >= 4:
                act = (0.15, 1.0) if self.dodge_side > 0 else (1.0, 0.15)
            else:
                act = (1.0, 0.92) if self.dodge_side > 0 else (0.92, 1.0)
            self.prev_action = act
            return act

        if self.phase == "hook":
            self.phase_timer -= 1
            # If front sensors already locked onto opponent's flank, abort hook & RAM!
            if dist[2] < 0.40 or dist[3] < 0.40 or dist[4] < 0.40 or self.phase_timer <= 0:
                self.phase = "idle"
                self.cooldown = 35
            else:
                # Sharp 90-degree inward snap toward opponent's exposed flank
                act = (0.90, -0.88) if self.dodge_side > 0 else (-0.88, 0.90)
                self.prev_action = act
                return act

        # Trigger Matador sidestep when enemy is detected straight ahead at medium range
        c0 = dist[3]
        if (
            self.phase == "idle"
            and self.cooldown == 0
            and 0.16 < c0 < 0.48
        ):
            self.phase = "sidestep"
            self.phase_timer = 7
            act = (0.15, 1.0) if self.dodge_side > 0 else (1.0, 0.15)
            self.prev_action = act
            return act

        act = self._track_with_7_sensors(dist, sensors.last_seen_dir, spin_speed=0.70)
        self.prev_action = act
        return act


class WavePendulumBot(BaseSumoBot):
    """Japanese Snake/Pendulum bot: weaves in S-curves until locking onto the opponent."""

    name = "Cobra Pendulum"
    style = "Busca em Onda S + Investida Total"

    def __init__(self, bot_id: int = 1):
        super().__init__(bot_id)
        self.tick = 0
        self.escape_timer = 0
        self.escape_dir = 1.0

    def reset(self):
        super().reset()
        self.tick = 0
        self.escape_timer = 0

    def get_action(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        sensors: SumoSensorPack,
    ) -> Tuple[float, float]:
        self.tick += 1
        line = sensors.read_line_sensors(robot)
        dist = sensors.read_distance_sensors(robot, opponent)

        if self.escape_timer > 0:
            self.escape_timer -= 1
            if self.escape_timer >= 4:
                act = (-1.0, -1.0)
            else:
                act = (-0.8, 0.8) if self.escape_dir > 0 else (0.8, -0.8)
            self.prev_action = act
            return act

        if (line[0] > 0.35 or line[1] > 0.35) and dist[3] > 0.14:
            self.escape_timer = 9
            self.escape_dir = -1.0 if line[0] >= line[1] else 1.0
            self.prev_action = (-1.0, -1.0)
            return self.prev_action

        # If any sensor sees opponent within 50 cm, lock on with full PD control
        if np.min(dist) < 0.75:
            act = self._track_with_7_sensors(dist, sensors.last_seen_dir, spin_speed=0.82)
        else:
            # S-curve pendulum wave search
            wave = 0.45 * math.sin(self.tick * 0.28)
            act = (float(np.clip(0.80 - wave, -1.0, 1.0)), float(np.clip(0.80 + wave, -1.0, 1.0)))

        self.prev_action = act
        return act


class TawaraGaeshiBot(BaseSumoBot):
    """Counter-puncher that uses tactical tap-pulses and executes Tawara-gaeshi border throws."""

    name = "Tawara-Gaeshi"
    style = "Pulso Tático + Giro de Contra-Golpe na Borda"

    def __init__(self, bot_id: int = 1):
        super().__init__(bot_id)
        self.tick = 0
        self.escape_timer = 0

    def reset(self):
        super().reset()
        self.tick = 0
        self.escape_timer = 0

    def get_action(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        sensors: SumoSensorPack,
    ) -> Tuple[float, float]:
        self.tick += 1
        line = sensors.read_line_sensors(robot)
        dist = sensors.read_distance_sensors(robot, opponent)

        # Signature Move: Tawara-Gaeshi!
        # If pushed backward onto the rear white border while in contact, pivot violently
        # to sling the pushing attacker past us and over the edge!
        if (line[2] > 0.35 or line[3] > 0.35) and (robot.in_contact or dist[3] < 0.25):
            if line[2] >= line[3]:
                act = (1.0, -0.45)  # Pivot clockwise around right wheel
            else:
                act = (-0.45, 1.0)  # Pivot counter-clockwise around left wheel
            self.prev_action = act
            return act

        if self.escape_timer > 0:
            self.escape_timer -= 1
            self.prev_action = (-0.9, -0.5)
            return self.prev_action

        if (line[0] > 0.38 or line[1] > 0.38) and not robot.in_contact:
            self.escape_timer = 6
            self.prev_action = (-1.0, -1.0)
            return self.prev_action

        # Close range -> 100% power shove
        if np.min(dist[2:5]) < 0.42 or robot.in_contact:
            act = self._track_with_7_sensors(dist, sensors.last_seen_dir, spin_speed=0.78)
        elif np.min(dist) < 0.96:
            # Medium range -> Tactical pulsed approach (keeps front blade flat & low on dohyo)
            pulse = 0.95 if (self.tick % 6) < 4 else 0.30
            base_l, base_r = self._track_with_7_sensors(dist, sensors.last_seen_dir)
            act = (base_l * pulse, base_r * pulse)
        else:
            act = self._track_with_7_sensors(dist, sensors.last_seen_dir, spin_speed=0.65)

        self.prev_action = act
        return act


class NeuralPolicyBot(BaseSumoBot):
    """Wraps a PyTorch SumoPolicyNet snapshot to fight as a sensor-driven Self-Play opponent."""

    name = "PPO Self-Play"
    style = "Rede Neural (24D Sensores)"

    def __init__(self, policy_net, bot_id: int = 1, deterministic: bool = False):
        super().__init__(bot_id)
        self.policy_net = policy_net
        self.deterministic = deterministic

    def get_action(
        self,
        robot: SumoRobotBody,
        opponent: SumoRobotBody,
        sensors: SumoSensorPack,
    ) -> Tuple[float, float]:
        obs = sensors.get_observation(robot, opponent, self.prev_action)
        act_arr = self.policy_net.act(obs, deterministic=self.deterministic)
        self.prev_action = (float(act_arr[0]), float(act_arr[1]))
        return self.prev_action


COMPETITIVE_BOT_REGISTRY: Dict[str, type[BaseSumoBot]] = {
    "karioka": SearchAndDestroyBot,
    "bullrush": BullRushBot,
    "matador": MatadorBot,
    "pendulum": WavePendulumBot,
    "tawara_gaeshi": TawaraGaeshiBot,
}
