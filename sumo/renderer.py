"""Broadcast-grade Pygame HD renderer for Mini-Sumo 500g matches, multi-chassis hulls, and telemetry."""

from __future__ import annotations
import os
from typing import Optional, Tuple
import cv2
import numpy as np
import pygame

from sumo.env import SumoEnv
from sumo.physics import SumoRobotBody, visual_hull_outline
from sumo.sensors import SumoSensorPack


class SumoRenderer:
    """Renders the circular Dohyo, distinct robot chassis archetypes, 7-ray ToF arrays, and HUD."""

    def __init__(self, width: int = 1280, height: int = 720):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        pygame.init()
        pygame.font.init()

        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.overlay = pygame.Surface((width, height), pygame.SRCALPHA)

        self.font_title = pygame.font.SysFont("Segoe UI", 23, bold=True)
        self.font_sub = pygame.font.SysFont("Segoe UI", 15, bold=True)
        self.font_panel = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 12)
        self.font_big = pygame.font.SysFont("Segoe UI", 46, bold=True)

        self.center_x = width // 2
        self.center_y = height // 2 + 18
        self.scale = 548.0 / 77.0  # ~7.12 px per cm

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        sx = round(self.center_x + x * self.scale)
        sy = round(self.center_y - y * self.scale)
        return sx, sy

    def _draw_dohyo(self, surf: pygame.Surface, env: SumoEnv):
        dohyo = env.dohyo
        outer_r_px = round(dohyo.radius * self.scale)
        inner_r_px = round(dohyo.inner_radius * self.scale)

        # Arena drop shadow & outer steel rim
        pygame.draw.circle(surf, (5, 8, 14), (self.center_x + 7, self.center_y + 9), outer_r_px + 12)
        pygame.draw.circle(surf, (45, 52, 65), (self.center_x, self.center_y), outer_r_px + 4)

        # White Tawaramono border ring (2.5 cm)
        pygame.draw.circle(surf, (242, 245, 250), (self.center_x, self.center_y), outer_r_px)

        # Black polyurethane/steel inner fighting surface (r = 36.0 cm)
        pygame.draw.circle(surf, (22, 26, 34), (self.center_x, self.center_y), inner_r_px)

        # Subtle machined steel concentric rings & crosshair
        for r_cm in (12.0, 24.0):
            pygame.draw.circle(surf, (31, 37, 48), (self.center_x, self.center_y), round(r_cm * self.scale), 1)

        # Shikiri starting lines (10 cm separation, 10 cm length)
        sep_px = round((dohyo.config.shikiri_separation / 2.0) * self.scale)
        half_len_px = round((dohyo.config.shikiri_length / 2.0) * self.scale)
        line_w_px = max(3, round(dohyo.config.shikiri_width * self.scale))
        for sign in (-1, 1):
            lx = self.center_x + sign * sep_px
            pygame.draw.line(
                surf,
                (175, 130, 85),
                (lx, self.center_y - half_len_px),
                (lx, self.center_y + half_len_px),
                line_w_px,
            )

        # Draw tire skid marks from both robots
        for bot in (env.engine.robot_blue, env.engine.robot_red):
            if len(bot.skid_trail) >= 2:
                pts = [self.world_to_screen(px, py) for px, py in bot.skid_trail]
                pygame.draw.lines(surf, (14, 17, 22), False, pts, 4)

    def _draw_sensor_rays(
        self,
        overlay: pygame.Surface,
        sensors: SumoSensorPack,
        is_blue: bool,
    ):
        """Draw the 7 ToF beams from physical chassis mounts to exact hit points."""
        if not sensors.ray_origins_world:
            return

        clear_color = (40, 180, 255, 42) if is_blue else (255, 110, 90, 42)
        hit_color = (0, 255, 160, 210) if is_blue else (255, 220, 40, 210)

        for (ox, oy), (ex, ey), hit in zip(
            sensors.ray_origins_world,
            sensors.ray_endpoints_world,
            sensors.ray_hits,
        ):
            p1 = self.world_to_screen(ox, oy)
            p2 = self.world_to_screen(ex, ey)
            if hit:
                pygame.draw.line(overlay, hit_color, p1, p2, 2)
                pygame.draw.circle(overlay, (255, 60, 60, 245), p2, 5)
                pygame.draw.circle(overlay, (255, 255, 255, 255), p2, 2)
            else:
                pygame.draw.line(overlay, clear_color, p1, p2, 1)

    def _draw_robot(self, surf: pygame.Surface, bot: SumoRobotBody, sensors: SumoSensorPack):
        cfg = bot.config
        hl, hw = bot.length / 2.0, bot.width / 2.0

        # 1. Draw wheels (2WD or 4WD)
        w_rad = cfg.wheel_diameter / 2.0
        w_half_w = cfg.wheel_width / 2.0
        axle_positions = (
            [cfg.axle_x]
            if cfg.num_wheels == 2
            else [hl * 0.42, -hl * 0.48]
        )
        for ax in axle_positions:
            for side in (-1.0, 1.0):
                wy_center = side * (cfg.track / 2.0)
                w_corners = [
                    (ax - w_rad, wy_center - w_half_w),
                    (ax + w_rad, wy_center - w_half_w),
                    (ax + w_rad, wy_center + w_half_w),
                    (ax - w_rad, wy_center + w_half_w),
                ]
                w_pts = [self.world_to_screen(*bot.get_world_point(wx, wy)) for wx, wy in w_corners]
                # StickyMAX yellow/black silicone wheel styling
                pygame.draw.polygon(surf, (25, 28, 32), w_pts)
                pygame.draw.polygon(surf, (220, 185, 30), w_pts, 1)

        # 2. Main chassis hull polygon
        hull_local = visual_hull_outline(cfg)
        hull_pts = [self.world_to_screen(*bot.get_world_point(lx, ly)) for lx, ly in hull_local]
        pygame.draw.polygon(surf, cfg.primary_color, hull_pts)
        border_col = (255, 230, 80) if bot.wedge_advantage else (230, 238, 248)
        pygame.draw.polygon(surf, border_col, hull_pts, 2)

        # 3. Front Wedge / Blade bevel surface
        wtype = cfg.wedge_type
        if wtype == "twin_fork":
            # Draw two metallic fork blades
            for s_sign in (-1.0, 1.0):
                fork_poly = [
                    (hl * 0.45, s_sign * hw * 0.35),
                    (hl, s_sign * hw * 0.42),
                    (hl, s_sign * hw * 0.96),
                    (hl * 0.45, s_sign * hw * 0.96),
                ]
                f_pts = [self.world_to_screen(*bot.get_world_point(x, y)) for x, y in fork_poly]
                pygame.draw.polygon(surf, (210, 220, 230), f_pts)
                pygame.draw.polygon(surf, cfg.accent_color, f_pts, 2)
        elif wtype == "v_prow":
            prow_poly = [
                (hl * 0.35, -hw * 0.85),
                (hl * 0.60, -hw * 0.92),
                (hl, 0.0),
                (hl * 0.60, hw * 0.92),
                (hl * 0.35, hw * 0.85),
            ]
            p_pts = [self.world_to_screen(*bot.get_world_point(x, y)) for x, y in prow_poly]
            pygame.draw.polygon(surf, (205, 215, 228), p_pts)
            pygame.draw.polygon(surf, cfg.accent_color, p_pts, 2)
        else:
            # Flat / Tapered beveled steel/titanium ramp
            yb = hw * (0.98 if wtype == "wide_razor" else 0.85)
            wedge_poly = [
                (hl * 0.42, -yb),
                (hl, -yb),
                (hl, yb),
                (hl * 0.42, yb),
            ]
            w_pts = [self.world_to_screen(*bot.get_world_point(x, y)) for x, y in wedge_poly]
            pygame.draw.polygon(surf, (200, 210, 222), w_pts)
            pygame.draw.line(surf, cfg.accent_color, w_pts[1], w_pts[2], 4)

        # 4. Internal MCU / Battery core box & heading vector
        core_poly = [
            (-hl * 0.65, -hw * 0.45),
            (hl * 0.20, -hw * 0.45),
            (hl * 0.20, hw * 0.45),
            (-hl * 0.65, hw * 0.45),
        ]
        c_pts = [self.world_to_screen(*bot.get_world_point(x, y)) for x, y in core_poly]
        pygame.draw.polygon(surf, (18, 24, 34), c_pts)
        pygame.draw.polygon(surf, (90, 105, 125), c_pts, 1)

        center_pt = self.world_to_screen(bot.x, bot.y)
        nose_pt = self.world_to_screen(*bot.get_world_point(hl * 0.82, 0.0))
        pygame.draw.line(surf, cfg.accent_color, center_pt, nose_pt, 2)

        # 5. Corner Optical Line Sensor LEDs (4x)
        line_vals = sensors.last_line_readings
        for j, (lx, ly) in enumerate(cfg.line_sensor_offsets):
            sx, sy = self.world_to_screen(*bot.get_world_point(lx, ly))
            is_white = line_vals[j] > 0.35
            led_col = (255, 30, 30) if is_white else (0, 230, 90)
            pygame.draw.circle(surf, led_col, (sx, sy), 4 if is_white else 3)
            pygame.draw.circle(surf, (15, 20, 25), (sx, sy), 4 if is_white else 3, 1)

        # 6. Status badge above robot (Wedge Advantage or Lifted)
        if bot.wedge_advantage:
            pygame.draw.circle(surf, (255, 215, 0), center_pt, round(5.8 * self.scale), 2)
            lbl = self.font_small.render("▲ RAMPA SOB OPONENTE", True, (255, 225, 60))
            surf.blit(lbl, (center_pt[0] - lbl.get_width() // 2, center_pt[1] - 54))
        elif bot.is_lifted:
            lbl = self.font_small.render("▼ LEVANTADO (-TRAÇÃO)", True, (255, 95, 95))
            surf.blit(lbl, (center_pt[0] - lbl.get_width() // 2, center_pt[1] + 42))

    def _draw_telemetry_panel(
        self,
        surf: pygame.Surface,
        x: int,
        y: int,
        bot: SumoRobotBody,
        sensors: SumoSensorPack,
        role_title: str,
        header_color: Tuple[int, int, int],
    ):
        panel_w, panel_h = 270, 250
        rect = pygame.Rect(x, y, panel_w, panel_h)
        pygame.draw.rect(surf, (18, 25, 38), rect, border_radius=10)
        pygame.draw.rect(surf, header_color, rect, width=2, border_radius=10)

        # Robot & Strategy Title
        t1 = self.font_sub.render(role_title, True, header_color)
        t2 = self.font_panel.render(f"Chassi: {bot.config.name}", True, (235, 240, 248))
        t3 = self.font_small.render(
            f"{bot.config.num_wheels}WD | Rampa {bot.config.wedge_angle_deg:.0f}° | Downforce +{bot.config.downforce_mass_eq:.0f}g",
            True,
            (150, 168, 190),
        )
        surf.blit(t1, (x + 12, y + 10))
        surf.blit(t2, (x + 12, y + 32))
        surf.blit(t3, (x + 12, y + 50))

        # Kinematics & Grip
        spd_txt = f"Vel: {bot.speed:5.1f} cm/s | Giro: {bot.omega:5.1f} rad/s"
        grip_txt = f"Tração Efetiva: {bot.traction_multiplier * 100.0:3.0f}% {'[EMPURRANDO]' if bot.in_contact else ''}"
        surf.blit(self.font_small.render(spd_txt, True, (200, 215, 235)), (x + 12, y + 74))
        grip_col = (255, 220, 60) if bot.traction_multiplier > 1.0 else ((255, 90, 90) if bot.traction_multiplier < 1.0 else (130, 220, 160))
        surf.blit(self.font_small.render(grip_txt, True, grip_col), (x + 12, y + 92))

        # 7 ToF Distance Sensor Bars
        surf.blit(self.font_small.render("Sensores ToF (L90 L45 L15 C0 R15 R45 R90):", True, (160, 175, 195)), (x + 12, y + 116))
        labels = ("L90", "L45", "L15", "C0", "R15", "R45", "R90")
        bar_w = 28
        bar_max_h = 48
        for i, d_val in enumerate(sensors.last_dist_readings):
            bx = x + 16 + i * 35
            by = y + 136
            pygame.draw.rect(surf, (30, 38, 52), (bx, by, bar_w, bar_max_h), border_radius=3)
            proximity = 1.0 - float(d_val)
            fill_h = round(proximity * bar_max_h)
            if fill_h > 0:
                b_col = (255, 65, 65) if proximity > 0.6 else (0, 225, 150)
                pygame.draw.rect(surf, b_col, (bx, by + bar_max_h - fill_h, bar_w, fill_h), border_radius=3)
            lbl = self.font_small.render(labels[i], True, (180, 190, 205))
            surf.blit(lbl, (bx + 2, by + bar_max_h + 2))

        # Wheel PWM Command Indicators
        l_pwm, r_pwm = float(bot.command[0]), float(bot.command[1])
        pwm_str = f"PWM Rodas: Esq {l_pwm:+5.2f} | Dir {r_pwm:+5.2f}"
        surf.blit(self.font_panel.render(pwm_str, True, (120, 195, 255)), (x + 12, y + 212))

        # Line Sensors summary
        l_vals = sensors.last_line_readings
        line_hit = any(v > 0.35 for v in l_vals)
        l_str = f"Borda QRE: {'ALERTA LINHA BRANCA!' if line_hit else 'Preto (Seguro)'}"
        surf.blit(
            self.font_small.render(l_str, True, (255, 80, 80) if line_hit else (110, 200, 130)),
            (x + 12, y + 230),
        )

    def draw_frame(
        self,
        env: SumoEnv,
        title: str = "Mini-Sumo 500g · Simulador Oficial de Combate",
        subtitle: str = "Física Pymunk 240Hz + Raycasting 7 Sensores ToF",
        blue_role: str = "AZUL (Blue)",
        red_role: str = "VERMELHO (Red)",
        winner_text: Optional[str] = None,
    ) -> np.ndarray:
        """Render complete match frame and return BGR image for OpenCV video recording."""
        surf = self.surface
        surf.fill((10, 14, 22))
        self.overlay.fill((0, 0, 0, 0))

        b_bot = env.engine.robot_blue
        r_bot = env.engine.robot_red

        # 1. Dohyo + Skid Marks
        self._draw_dohyo(surf, env)

        # 2. Sensor Raycasts for BOTH robots on alpha overlay
        self._draw_sensor_rays(self.overlay, env.sensors_blue, is_blue=True)
        self._draw_sensor_rays(self.overlay, env.sensors_red, is_blue=False)
        surf.blit(self.overlay, (0, 0))

        # 3. Draw Robots (draw lifted robot first so dominant wedge renders on top!)
        if b_bot.wedge_advantage:
            self._draw_robot(surf, r_bot, env.sensors_red)
            self._draw_robot(surf, b_bot, env.sensors_blue)
        else:
            self._draw_robot(surf, b_bot, env.sensors_blue)
            self._draw_robot(surf, r_bot, env.sensors_red)

        # 4. Draw Collision Impact Sparks at Pymunk contact points
        for cx, cy in env.engine.contact_points_world:
            sx, sy = self.world_to_screen(cx, cy)
            pygame.draw.circle(surf, (255, 255, 220), (sx, sy), 6)
            pygame.draw.circle(surf, (255, 140, 20), (sx, sy), 10, 2)

        # 5. Header & Side Telemetry Panels
        t_surf = self.font_title.render(title, True, (242, 246, 252))
        s_surf = self.font_sub.render(subtitle, True, (150, 170, 195))
        surf.blit(t_surf, (28, 14))
        surf.blit(s_surf, (28, 44))

        clock_str = f"TEMPO: {env.engine.time:4.2f}s  |  Layout: {env.current_layout.upper()}"
        c_surf = self.font_panel.render(clock_str, True, (255, 215, 60))
        surf.blit(c_surf, (self.width - c_surf.get_width() - 28, 22))

        self._draw_telemetry_panel(
            surf, 24, 90, b_bot, env.sensors_blue, blue_role, b_bot.config.primary_color
        )
        self._draw_telemetry_panel(
            surf, self.width - 294, 90, r_bot, env.sensors_red, red_role, r_bot.config.primary_color
        )

        # Footer legend
        footer = (
            "Física Rígida 240Hz · 5 Arquétipos de Chassi · 7 Sensores ToF Raycast · "
            "4 Sensores de Borda QRE1113 · Liga Híbrida (Bots + PPO Self-Play)"
        )
        f_surf = self.font_small.render(footer, True, (115, 130, 150))
        surf.blit(f_surf, (self.width // 2 - f_surf.get_width() // 2, self.height - 24))

        # 6. Winner Banner
        if winner_text:
            box = pygame.Rect(self.width // 2 - 300, self.height // 2 - 58, 600, 116)
            pygame.draw.rect(surf, (12, 18, 28), box, border_radius=14)
            pygame.draw.rect(surf, (255, 215, 0), box, width=3, border_radius=14)

            w_surf = self.font_big.render(winner_text, True, (255, 215, 0))
            surf.blit(w_surf, w_surf.get_rect(center=(self.width // 2, self.height // 2 - 14)))

            sub_msg = f"Ring-Out confirmado aos {env.engine.time:.2f}s de luta"
            sm_surf = self.font_sub.render(sub_msg, True, (230, 238, 248))
            surf.blit(sm_surf, sm_surf.get_rect(center=(self.width // 2, self.height // 2 + 28)))

        rgb = pygame.image.tobytes(surf, "RGB")
        bgr = cv2.cvtColor(
            np.frombuffer(rgb, dtype=np.uint8).reshape((self.height, self.width, 3)),
            cv2.COLOR_RGB2BGR,
        )
        return bgr
