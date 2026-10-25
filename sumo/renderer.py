"""High-definition Pygame renderer for Mini-Sumo matches, sensor raycasts and HUD."""

import math
import os
from typing import Optional, Tuple
import cv2
import numpy as np
import pygame

from sumo.env import SumoEnv


class SumoRenderer:
    """Renders the circular Dohyo, robot models, line sensor dots, and distance rays."""

    def __init__(self, width: int = 1280, height: int = 720):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        pygame.init()
        pygame.font.init()

        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))

        self.font_title = pygame.font.SysFont("Segoe UI", 24, bold=True)
        self.font_sub = pygame.font.SysFont("Segoe UI", 16)
        self.font_telem = pygame.font.SysFont("Consolas", 15)
        self.font_big = pygame.font.SysFont("Segoe UI", 52, bold=True)

        self.center_x = width // 2
        self.center_y = height // 2 + 20
        # Dohyo diameter = 77 cm -> Scale so dohyo fits nicely in 580 px height
        self.scale = 540.0 / 77.0  # pixels per cm (~7.0 px/cm)

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert cm world coordinates (center origin) to screen pixel coordinates."""
        sx = round(self.center_x + x * self.scale)
        sy = round(self.center_y - y * self.scale)
        return sx, sy

    def draw_frame(
        self,
        env: SumoEnv,
        title: str = "Mini-Sumo 500g · Combate Autônomo",
        subtitle: str = "PPO Neural Controller vs Opponent",
        winner_text: Optional[str] = None,
    ) -> np.ndarray:
        """Render complete match frame and return BGR image for video recording."""
        surf = self.surface
        surf.fill((12, 18, 28))  # Dark tournament arena background

        dohyo = env.dohyo
        b_bot = env.engine.robot_blue
        r_bot = env.engine.robot_red

        # --- DOHYO RENDERING ---
        # Outer ring drop-off shadow
        outer_r_px = round(dohyo.radius * self.scale)
        pygame.draw.circle(surf, (6, 10, 16), (self.center_x + 6, self.center_y + 8), outer_r_px + 10)

        # White Border Line (Tawaramono: 2.5 cm outer ring)
        pygame.draw.circle(surf, (240, 245, 250), (self.center_x, self.center_y), outer_r_px)

        # Black Inner Ring (36 cm radius)
        inner_r_px = round(dohyo.inner_radius * self.scale)
        pygame.draw.circle(surf, (26, 32, 42), (self.center_x, self.center_y), inner_r_px)

        # Dohyo texture rings
        pygame.draw.circle(surf, (36, 44, 56), (self.center_x, self.center_y), round(20 * self.scale), 1)
        pygame.draw.circle(surf, (36, 44, 56), (self.center_x, self.center_y), round(10 * self.scale), 1)

        # Shikiri Starting Lines (two brown/white lines separated by 10 cm)
        sep_px = round((dohyo.config.shikiri_separation / 2.0) * self.scale)
        line_len_px = round((dohyo.config.shikiri_length / 2.0) * self.scale)
        for sign in (-1, 1):
            lx = self.center_x + sign * sep_px
            pygame.draw.line(surf, (150, 120, 90), (lx, self.center_y - line_len_px), (lx, self.center_y + line_len_px), 3)

        # --- DRAW SENSOR RAYS (BLUE ROBOT) ---
        dist_readings = env.sensors_blue.read_distance_sensors(b_bot, r_bot)
        max_dist_px = env.robot_cfg.distance_sensor_range * self.scale
        for i, ang in enumerate(env.robot_cfg.distance_sensor_angles_rad):
            ray_ang = b_bot.theta + ang
            d_norm = dist_readings[i]
            ray_len = d_norm * max_dist_px
            bx, by = self.world_to_screen(b_bot.x, b_bot.y)
            ex = round(bx + math.cos(ray_ang) * ray_len)
            ey = round(by - math.sin(ray_ang) * ray_len)
            ray_color = (255, 60, 60) if d_norm < 0.95 else (0, 180, 120)
            pygame.draw.line(surf, ray_color, (bx, by), (ex, ey), 1)

        # --- DRAW ROBOTS ---
        for bot, color, name in [(b_bot, (34, 113, 217), "Blue (PPO)"), (r_bot, (213, 66, 80), "Red (Bot)")]:
            # Chassis vertices in world
            hl, hw = bot.length / 2.0, bot.width / 2.0
            corners = [
                (-hl, -hw), (hl * 0.7, -hw), (hl, -hw * 0.9),
                (hl, hw * 0.9), (hl * 0.7, hw), (-hl, hw)
            ]
            poly_pts = [self.world_to_screen(*bot.get_world_point(cx, cy)) for cx, cy in corners]
            pygame.draw.polygon(surf, color, poly_pts)
            pygame.draw.polygon(surf, (240, 245, 255), poly_pts, 2)

            # Wedge blade highlight
            blade_p1 = self.world_to_screen(*bot.get_world_point(hl, -hw * 0.9))
            blade_p2 = self.world_to_screen(*bot.get_world_point(hl, hw * 0.9))
            pygame.draw.line(surf, (255, 215, 0), blade_p1, blade_p2, 4)

            # Heading arrow
            center_pt = self.world_to_screen(bot.x, bot.y)
            nose_pt = self.world_to_screen(*bot.get_world_point(hl + 2.5, 0.0))
            pygame.draw.line(surf, (255, 255, 255), center_pt, nose_pt, 3)

            # Wheels
            for wy_side in (-hw, hw):
                w_start = self.world_to_screen(*bot.get_world_point(-1.6, wy_side))
                w_end = self.world_to_screen(*bot.get_world_point(1.6, wy_side))
                pygame.draw.line(surf, (50, 50, 50), w_start, w_end, 5)

            # Line Sensor dots (4 corners)
            sensors_pack = env.sensors_blue if bot.id == 0 else env.sensors_red
            line_vals = sensors_pack.read_line_sensors(bot)
            for j, (lx, ly) in enumerate(bot.config.line_sensor_offsets):
                sx, sy = self.world_to_screen(*bot.get_world_point(lx, ly))
                dot_color = (255, 20, 20) if line_vals[j] > 0.4 else (0, 255, 100)
                pygame.draw.circle(surf, dot_color, (sx, sy), 4)

            # Wedge Advantage Glow
            if bot.wedge_advantage:
                pygame.draw.circle(surf, (255, 215, 0), center_pt, round(8 * self.scale), 2)

        # --- HEADER HUD ---
        # Title
        t_surf = self.font_title.render(title, True, (240, 246, 252))
        surf.blit(t_surf, (40, 18))
        s_surf = self.font_sub.render(subtitle, True, (139, 155, 175))
        surf.blit(s_surf, (40, 48))

        # Telemetry top right
        b_spd = b_bot.speed
        r_spd = r_bot.speed
        telem = f"Tempo: {env.engine.time:4.2f}s | Blue: {b_spd:5.1f} cm/s | Red: {r_spd:5.1f} cm/s"
        telem_surf = self.font_telem.render(telem, True, (88, 166, 255))
        surf.blit(telem_surf, (self.width - telem_surf.get_width() - 40, 25))

        # Footer
        footer = "Mini-Sumo 500g · Dohyo 77cm | Sensores de Borda (4x) | ToF Distância (5x) | PPO Self-Play"
        footer_surf = self.font_sub.render(footer, True, (100, 115, 135))
        surf.blit(footer_surf, (40, self.height - 35))

        # --- WINNER BANNER ---
        if winner_text:
            banner_box = pygame.Rect(self.width // 2 - 280, self.height // 2 - 60, 560, 120)
            pygame.draw.rect(surf, (15, 20, 30), banner_box, border_radius=16)
            pygame.draw.rect(surf, (255, 215, 0), banner_box, width=3, border_radius=16)

            w_surf = self.font_big.render(winner_text, True, (255, 215, 0))
            surf.blit(w_surf, w_surf.get_rect(center=(self.width // 2, self.height // 2 - 15)))

            time_sub = f"Vitória por Ring-Out aos {env.engine.time:.2f}s"
            ts_surf = self.font_sub.render(time_sub, True, (230, 240, 250))
            surf.blit(ts_surf, ts_surf.get_rect(center=(self.width // 2, self.height // 2 + 30)))

        # Convert to BGR array for OpenCV
        rgb = pygame.image.tobytes(surf, "RGB")
        bgr = cv2.cvtColor(np.frombuffer(rgb, dtype=np.uint8).reshape((self.height, self.width, 3)), cv2.COLOR_RGB2BGR)
        return bgr
