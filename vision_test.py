"""Ferramenta interativa para teste, visualização e calibração de visão computacional.

Suporta:
    - Foto: python vision_test.py --image caminho/da/foto.jpg
    - Vídeo: python vision_test.py --video caminho/do/video.mp4
    - Câmera ao vivo: python vision_test.py --camera 0
    - Gerar marcadores ArUco prontos para impressão: python vision_test.py --generate-markers
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np

from core.arena_geometry import arena_outline
from core.constants import ArenaConfig, RobotConfig, PROFILES
from vision.camera_tracker import ArenaHomography, HockeyVisionTracker


def generate_aruco_markers(output_dir: Path | str = "docs/assets/markers", marker_size_px: int = 400) -> list[Path]:
    """Gera marcadores ArUco DICT_4X4_50 com indicação clara de frente e time."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    generated = []
    # Folha combinada com os 6 marcadores (2 linhas x 3 colunas)
    cell_w, cell_h = marker_size_px + 120, marker_size_px + 180
    sheet = np.full((cell_h * 2 + 80, cell_w * 3 + 80, 3), 255, dtype=np.uint8)

    # Título da folha
    cv2.putText(
        sheet,
        "RoboCore Hockey - Marcadores ArUco DICT_4X4_50 (A aresta superior aponta para a FRENTE)",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    for idx in range(6):
        team = "Time Azul" if idx < 3 else "Time Vermelho"
        role = ["Atacante 1", "Atacante 2", "Goleiro"][idx % 3]
        color = (180, 50, 20) if idx < 3 else (20, 30, 180)  # BGR

        # Gera o marcador ArUco 4x4
        raw_marker = cv2.aruco.generateImageMarker(dictionary, idx, marker_size_px)
        marker_bgr = cv2.cvtColor(raw_marker, cv2.COLOR_GRAY2BGR)

        # Cria imagem individual com margem, indicador de frente e texto
        card = np.full((cell_h, cell_w, 3), 255, dtype=np.uint8)

        # Seta e texto de FRENTE (aresta superior)
        arrow_y = 55
        cv2.putText(
            card,
            "^ FRENTE DO ROBO ^",
            (cell_w // 2 - 130, arrow_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 120, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.arrowedLine(
            card,
            (cell_w // 2, arrow_y + 25),
            (cell_w // 2, arrow_y - 25),
            (0, 140, 0),
            3,
            tipLength=0.35,
        )

        # Cola o marcador centralizado
        x_offset = (cell_w - marker_size_px) // 2
        y_offset = 75
        card[y_offset : y_offset + marker_size_px, x_offset : x_offset + marker_size_px] = marker_bgr

        # Borda fina ao redor do marcador
        cv2.rectangle(
            card,
            (x_offset - 2, y_offset - 2),
            (x_offset + marker_size_px + 2, y_offset + marker_size_px + 2),
            (160, 160, 160),
            2,
        )

        # Legenda inferior: ID, Time e Posição
        label = f"ID: {idx} | {team} ({role})"
        cv2.putText(
            card,
            label,
            (20, y_offset + marker_size_px + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

        # Salva o arquivo individual
        file_path = out_path / f"marker_id_{idx}_{team.lower().replace(' ', '_')}.png"
        cv2.imwrite(str(file_path), card)
        generated.append(file_path)

        # Adiciona à folha combinada
        row = idx // 3
        col = idx % 3
        sx = 40 + col * cell_w
        sy = 70 + row * cell_h
        sheet[sy : sy + cell_h, sx : sx + cell_w] = card

    sheet_path = out_path / "all_markers_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    generated.append(sheet_path)
    return generated


class VisionInspector:
    """Inspeciona e anota imagens/vídeos com diagnósticos completos da visão do hockey."""

    def __init__(self, config_path: str | Path | None = None, arena_profile: str = "micro-2024"):
        self.config_path = Path(config_path) if config_path else None
        self.arena = PROFILES.get(arena_profile, ArenaConfig())
        self.robot_cfg = RobotConfig()

        # Configurações padrão
        self.floor_corners_px: list[list[int]] = []
        self.puck_hsv_lower = np.array([0, 0, 0], dtype=np.uint8)
        self.puck_hsv_upper = np.array([179, 255, 65], dtype=np.uint8)

        # Carrega config se existir
        if self.config_path and self.config_path.exists():
            self.load_config(self.config_path)

        # Tracker e Homografia
        self.homography: ArenaHomography | None = None
        self.tracker: HockeyVisionTracker | None = None
        self._setup_tracker()

        # Detector genérico de ArUco para capturar todos os IDs (diagnóstico)
        dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.all_aruco_detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())

        # Estado interativo
        self.calibrating_corners = False
        self.clicked_corners: list[list[int]] = []
        self.picking_puck_color = False
        self.show_trackbars = False
        self.view_mode = 0  # 0: Anotado, 1: Máscara HSV, 2: Lado a Lado
        self.last_diagnostics: dict = {}

    def load_config(self, path: Path):
        try:
            from dataclasses import replace

            data = json.loads(path.read_text(encoding="utf-8"))
            if "profile" in data and data["profile"] in PROFILES:
                self.arena = PROFILES[data["profile"]]
            if "puck_diameter" in data:
                self.arena = replace(self.arena, puck_diameter=float(data["puck_diameter"]))
            if "floor_corners_px" in data and len(data["floor_corners_px"]) == 4:
                self.floor_corners_px = data["floor_corners_px"]
            if "puck_hsv_lower" in data:
                self.puck_hsv_lower = np.array(data["puck_hsv_lower"], dtype=np.uint8)
            if "puck_hsv_upper" in data:
                self.puck_hsv_upper = np.array(data["puck_hsv_upper"], dtype=np.uint8)
        except Exception as e:
            print(f"[Aviso] Erro ao ler config {path}: {e}")

    def save_config(self, path: Path | None = None):
        target = Path(path or self.config_path or "configs/camera_test.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "profile": getattr(self.arena, "name", "micro-2024"),
            "camera": 0,
            "floor_corners_px": self.floor_corners_px,
            "puck_hsv_lower": self.puck_hsv_lower.tolist(),
            "puck_hsv_upper": self.puck_hsv_upper.tolist(),
            "puck_diameter": getattr(self.arena, "puck_diameter", 3.17),
            "note": "Gerado por vision_test.py",
        }
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[Sucesso] Configuração salva em: {target}")

    def _setup_tracker(self):
        if len(self.floor_corners_px) == 4:
            try:
                pts = np.array(self.floor_corners_px, np.float32)
                self.homography = ArenaHomography(pts, self.arena)
                self.tracker = HockeyVisionTracker(self.homography, robot_config=self.robot_cfg)
                self.tracker.puck_hsv_lower = self.puck_hsv_lower
                self.tracker.puck_hsv_upper = self.puck_hsv_upper
            except Exception as e:
                print(f"[Aviso] Falha ao criar homografia com cantos configurados: {e}")
                self.homography = None
                self.tracker = None
        else:
            self.homography = None
            self.tracker = None

    def set_corners(self, corners: list[list[int]]):
        if len(corners) == 4:
            self.floor_corners_px = corners
            self._setup_tracker()

    def set_puck_hsv(self, lower: list[int] | np.ndarray, upper: list[int] | np.ndarray):
        self.puck_hsv_lower = np.array(lower, dtype=np.uint8)
        self.puck_hsv_upper = np.array(upper, dtype=np.uint8)
        if self.tracker:
            self.tracker.puck_hsv_lower = self.puck_hsv_lower
            self.tracker.puck_hsv_upper = self.puck_hsv_upper

    def sample_puck_from_pixel(self, bgr_img: np.ndarray, px: int, py: int):
        """Ajusta o range HSV pegando a cor de uma janela 7x7 ao redor do ponto clicado."""
        h, w = bgr_img.shape[:2]
        x1, x2 = max(0, px - 3), min(w, px + 4)
        y1, y2 = max(0, py - 3), min(h, py + 4)
        patch = bgr_img[y1:y2, x1:x2]
        if patch.size == 0:
            return

        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        med_h = int(np.median(hsv_patch[:, :, 0]))
        med_s = int(np.median(hsv_patch[:, :, 1]))
        med_v = int(np.median(hsv_patch[:, :, 2]))

        # Se for um disco preto/escuro (V muito baixo)
        if med_v < 60:
            lower = [0, 0, 0]
            upper = [179, 255, min(255, med_v + 45)]
        else:
            # Disco colorido (laranja, vermelho, amarelo, verde, etc.)
            h_tol = 16
            s_tol = 60
            v_tol = 60
            lower = [max(0, med_h - h_tol), max(30, med_s - s_tol), max(30, med_v - v_tol)]
            upper = [min(179, med_h + h_tol), min(255, med_s + s_tol), min(255, med_v + v_tol)]

        self.set_puck_hsv(lower, upper)
        print(
            f"[Calibração Disco] HSV Amostrado em ({px}, {py}): "
            f"H={med_h}, S={med_s}, V={med_v} -> Lower={lower}, Upper={upper}"
        )

    def process_frame(
        self, frame: np.ndarray, timestamp: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Processa um frame e retorna (frame_anotado, frame_mascara, diagnosticos)."""
        ts = time.monotonic() if timestamp is None else timestamp
        vis = frame.copy()
        h, w = frame.shape[:2]

        diagnostics = {
            "calibrated": self.homography is not None,
            "robots_detected": {},
            "raw_markers": [],
            "puck_detected": None,
            "puck_candidates": [],
            "mask_coverage_px": 0,
        }

        # 1. Detectar TODOS os marcadores ArUco 4x4 no frame (para diagnosticar IDs errados)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_corners, raw_ids, _ = self.all_aruco_detector.detectMarkers(gray)

        if raw_ids is not None:
            for pts, marker_id in zip(raw_corners, raw_ids.flatten()):
                pts = pts[0]
                mid_x = float(pts[:, 0].mean())
                mid_y = float(pts[:, 1].mean())
                diagnostics["raw_markers"].append(
                    {"id": int(marker_id), "center_px": (int(mid_x), int(mid_y)), "corners": pts}
                )

        # 2. Desenhar Cantos da Arena (se configurados ou sendo calibrados)
        if self.calibrating_corners:
            for i, pt in enumerate(self.clicked_corners):
                cv2.circle(vis, tuple(pt), 7, (0, 255, 255), -1)
                cv2.putText(
                    vis,
                    f"Canto {i+1}",
                    (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            if len(self.clicked_corners) > 1:
                for i in range(len(self.clicked_corners) - 1):
                    cv2.line(vis, tuple(self.clicked_corners[i]), tuple(self.clicked_corners[i + 1]), (0, 255, 255), 2)
        elif self.homography is not None:
            try:
                outline_cm = arena_outline(self.arena, corner_steps=16)
                outline_px = np.array([self.homography.cm_to_px(x, y) for x, y in outline_cm], dtype=np.int32)
                cv2.polylines(vis, [outline_px], isClosed=True, color=(0, 255, 160), thickness=2, lineType=cv2.LINE_AA)

                gw = self.arena.goal_width / 2
                hl = self.arena.half_length
                cv2.line(vis, self.homography.cm_to_px(-hl, -gw), self.homography.cm_to_px(-hl, gw), (255, 130, 0), 2, cv2.LINE_AA)
                cv2.line(vis, self.homography.cm_to_px(hl, -gw), self.homography.cm_to_px(hl, gw), (40, 80, 255), 2, cv2.LINE_AA)

                top_mid = self.homography.cm_to_px(0, self.arena.half_width)
                bot_mid = self.homography.cm_to_px(0, -self.arena.half_width)
                cv2.line(vis, top_mid, bot_mid, (0, 200, 255), 1, cv2.LINE_AA)
            except Exception:
                arena_poly = np.array(self.floor_corners_px, dtype=np.int32)
                cv2.polylines(vis, [arena_poly], isClosed=True, color=(0, 220, 255), thickness=2)

            labels = ["TL (-150,+75)", "TR (+150,+75)", "BR (+150,-75)", "BL (-150,-75)"]
            for pt, lbl in zip(self.floor_corners_px, labels):
                cv2.circle(vis, tuple(pt), 4, (0, 220, 255), -1)
                cv2.putText(vis, lbl, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)

        # 3. Processamento de Máscara HSV do Disco
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.puck_hsv_lower, self.puck_hsv_upper)

        if self.homography is not None:
            roi = np.zeros(gray.shape, np.uint8)
            poly = np.array(
                [
                    self.homography.cm_to_px(x, y)
                    for x, y in [
                        (-self.arena.half_length, -self.arena.half_width),
                        (self.arena.half_length, -self.arena.half_width),
                        (self.arena.half_length, self.arena.half_width),
                        (-self.arena.half_length, self.arena.half_width),
                    ]
                ]
            )
            cv2.fillConvexPoly(roi, poly, 255)
            mask = cv2.bitwise_and(mask, roi)

        diagnostics["mask_coverage_px"] = int(cv2.countNonZero(mask))

        # 4. Detecção oficial via HockeyVisionTracker (se calibrado)
        if self.tracker is not None:
            try:
                robots, puck = self.tracker.detect_robots_and_puck(frame, ts)

                # Robôs válidos
                for rid, rstate in robots.items():
                    if rstate.valid(ts):
                        team_name = "Azul" if rid < 3 else "Vermelho"
                        team_color = (255, 140, 0) if rid < 3 else (40, 40, 255)  # BGR
                        px, py = self.homography.cm_to_px(rstate.x, rstate.y)
                        diagnostics["robots_detected"][rid] = {
                            "team": team_name,
                            "x_cm": rstate.x,
                            "y_cm": rstate.y,
                            "theta_rad": rstate.theta,
                            "theta_deg": math.degrees(rstate.theta),
                            "px": (px, py),
                        }

                        # Desenha marcador oficial e seta de frente física
                        # pts[0] e pts[1] definem a aresta frontal no padrão ArUco
                        cx, cy = px, py
                        arrow_len = 40
                        # Converte o vetor de orientação métrica (+y UP) para vetor de tela (+y DOWN)
                        hx = int(cx + arrow_len * math.cos(rstate.theta))
                        hy = int(cy - arrow_len * math.sin(rstate.theta))
                        cv2.circle(vis, (cx, cy), 8, team_color, -1)
                        cv2.arrowedLine(vis, (cx, cy), (hx, hy), (0, 255, 0), 3, tipLength=0.3)

                        txt = f"ID {rid} [{team_name}] ({rstate.x:.1f}, {rstate.y:.1f})cm {math.degrees(rstate.theta):.0f}deg"
                        cv2.putText(
                            vis, txt, (cx - 50, cy - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, team_color, 2
                        )

                # Disco válido
                if puck.valid(ts):
                    ppx, ppy = self.homography.cm_to_px(puck.x, puck.y)
                    diagnostics["puck_detected"] = {
                        "x_cm": puck.x,
                        "y_cm": puck.y,
                        "px": (ppx, ppy),
                    }
                    cv2.circle(vis, (ppx, ppy), 12, (0, 255, 255), 2)
                    cv2.drawMarker(
                        vis, (ppx, ppy), (0, 255, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA
                    )
                    txt_puck = f"DISCO ({puck.x:.1f}, {puck.y:.1f})cm"
                    cv2.putText(
                        vis,
                        txt_puck,
                        (ppx + 15, ppy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
            except Exception as e:
                diagnostics["error"] = str(e)

            # Detecção de apoio do robô azul por cor quando não há marcador ArUco colado
            if not diagnostics["robots_detected"]:
                car_mask = cv2.inRange(hsv, np.array([95, 100, 70]), np.array([130, 255, 255]))
                cnts_car, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts_car:
                    best_car = max(cnts_car, key=cv2.contourArea)
                    if cv2.contourArea(best_car) > 1200:
                        rect = cv2.minAreaRect(best_car)
                        cx, cy = int(rect[0][0]), int(rect[0][1])
                        cm_x, cm_y = self.homography.px_to_cm(cx, cy)
                        box = cv2.boxPoints(rect).astype(np.int32)
                        diagnostics["color_car_detected"] = {
                            "team": "Azul",
                            "px": (cx, cy),
                            "x_cm": cm_x,
                            "y_cm": cm_y,
                            "box": box,
                        }
                        cv2.polylines(vis, [box], True, (255, 140, 0), 2)
                        cv2.circle(vis, (cx, cy), 6, (255, 140, 0), -1)
                        cv2.putText(
                            vis,
                            f"ROBO AZUL ({cm_x:.1f}, {cm_y:.1f})cm [Sem ArUco]",
                            (cx - 100, cy - 22),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (255, 140, 0),
                            2,
                        )
                        cv2.putText(
                            vis,
                            "Cole a Tag ArUco ID 0 no teto",
                            (cx - 95, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 0),
                            1,
                        )

        # 5. Anotação dos Marcadores ArUco brutos (incluindo IDs inválidos)
        for m in diagnostics["raw_markers"]:
            mid = m["id"]
            pts = m["corners"].astype(np.int32)
            is_valid_id = mid in range(6)
            box_color = (0, 255, 0) if is_valid_id else (0, 69, 255)  # Verde se válido, Laranja se fora de 0..5
            cv2.polylines(vis, [pts], True, box_color, 2)

            # Marca a aresta superior (FRENTE)
            front_p1 = tuple(pts[0])
            front_p2 = tuple(pts[1])
            cv2.line(vis, front_p1, front_p2, (0, 255, 255), 3)

            cx, cy = m["center_px"]
            if not is_valid_id:
                cv2.putText(
                    vis,
                    f"TAG ID {mid} (Invalido: use 0..5)",
                    (cx - 70, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 69, 255),
                    2,
                )

        # 6. Painel Superior de Status (HUD)
        self._draw_hud(vis, diagnostics)

        self.last_diagnostics = diagnostics
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return vis, mask_bgr, diagnostics

    def _draw_hud(self, vis: np.ndarray, diag: dict):
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (vis.shape[1], 92), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.78, vis, 0.22, 0, vis)

        # Status da Arena
        if diag["calibrated"]:
            arena_str = "ARENA: CALIBRADA (300x150cm) | Coordenadas metricas ativas"
            arena_color = (0, 255, 120)
        else:
            arena_str = "ARENA: PENDENTE CALIBRACAO (Pressione 'C' para marcar os 4 cantos)"
            arena_color = (0, 165, 255)

        cv2.putText(vis, arena_str, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, arena_color, 2)

        # Status de Robôs e Disco
        num_robots = len(diag["robots_detected"])
        has_color_car = diag.get("color_car_detected") is not None
        puck_ok = diag["puck_detected"] is not None

        if num_robots > 0:
            rob_str = f"Robo: {num_robots} detectado(s) via Tag ArUco [OK]"
            rob_color = (0, 255, 0)
        elif has_color_car:
            c = diag["color_car_detected"]
            rob_str = f"Robo: Detectado por cor em ({c['x_cm']:.1f}, {c['y_cm']:.1f})cm [Cole a Tag ArUco ID 0 no teto]"
            rob_color = (255, 180, 0)
        else:
            rob_str = "Robo: Nenhum detectado"
            rob_color = (160, 160, 160)

        puck_str = (
            f"Disco: Detectado em ({diag['puck_detected']['x_cm']:.1f}, {diag['puck_detected']['y_cm']:.1f})cm [OK]"
            if puck_ok
            else f"Disco: Nao detectado (Pixels na mascara: {diag['mask_coverage_px']})"
        )

        cv2.putText(vis, rob_str, (15, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, rob_color, 1)
        cv2.putText(
            vis,
            puck_str,
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255) if puck_ok else (0, 120, 255),
            1,
        )

        # Dicas de Teclado
        keys_str = "[C] Calibrar Cantos | [P] Coletar Cor Disco | [T] Trackbars | [M] Modo Visao | [S] Salvar | [Q] Sair"
        cv2.putText(vis, keys_str, (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)


def run_interactive(
    source_type: str,
    source_path: str | int,
    inspector: VisionInspector,
    output_path: str | None = None,
):
    """Executa o loop interativo OpenCV com suporte a mouse e teclado."""
    window_name = "RoboCore Hockey - Teste de Visao Computacional"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Abre fonte
    if source_type == "image":
        img = cv2.imread(str(source_path))
        if img is None:
            print(f"[Erro] Nao foi possivel carregar a imagem: {source_path}")
            return
        cap = None
    elif source_type in ("video", "camera"):
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"[Erro] Nao foi possivel abrir video/camera: {source_path}")
            return
        img = None
    else:
        raise ValueError(f"Fonte desconhecida: {source_type}")

    # Gravador de vídeo se solicitado
    writer = None
    if output_path and source_type != "image":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Mouse Callback
    mouse_state = {"last_x": 0, "last_y": 0, "current_frame": None}

    def on_mouse(event, x, y, flags, param):
        mouse_state["last_x"] = x
        mouse_state["last_y"] = y

        if event == cv2.EVENT_LBUTTONDOWN:
            if inspector.calibrating_corners:
                inspector.clicked_corners.append([x, y])
                print(f"[Calibracao] Canto {len(inspector.clicked_corners)} marcado em ({x}, {y})")
                if len(inspector.clicked_corners) == 4:
                    inspector.set_corners(inspector.clicked_corners)
                    inspector.calibrating_corners = False
                    print("[Calibracao] Todos os 4 cantos definidos! Homografia ativada.")
            elif inspector.picking_puck_color and mouse_state["current_frame"] is not None:
                inspector.sample_puck_from_pixel(mouse_state["current_frame"], x, y)
                inspector.picking_puck_color = False
                print("[Calibracao] Cor do disco atualizada.")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if inspector.calibrating_corners and inspector.clicked_corners:
                removed = inspector.clicked_corners.pop()
                print(f"[Calibracao] Canto {len(inspector.clicked_corners) + 1} desfeito ({removed})")

    cv2.setMouseCallback(window_name, on_mouse)

    # Trackbars
    def create_trackbars():
        tb_win = "HSV Ajuste Fino - Disco"
        cv2.namedWindow(tb_win, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("H Min", tb_win, int(inspector.puck_hsv_lower[0]), 179, lambda v: None)
        cv2.createTrackbar("H Max", tb_win, int(inspector.puck_hsv_upper[0]), 179, lambda v: None)
        cv2.createTrackbar("S Min", tb_win, int(inspector.puck_hsv_lower[1]), 255, lambda v: None)
        cv2.createTrackbar("S Max", tb_win, int(inspector.puck_hsv_upper[1]), 255, lambda v: None)
        cv2.createTrackbar("V Min", tb_win, int(inspector.puck_hsv_lower[2]), 255, lambda v: None)
        cv2.createTrackbar("V Max", tb_win, int(inspector.puck_hsv_upper[2]), 255, lambda v: None)

    paused = False
    current_raw = None

    print("\n" + "=" * 65)
    print("  ROBOCORE HOCKEY - TESTE DE VISÃO COMPUTACIONAL INICIADO")
    print("=" * 65)
    print("Teclas de atalho:")
    print("  [C] - Iniciar calibração dos 4 cantos da pista (clique TL, TR, BR, BL)")
    print("  [P] - Selecionar cor do disco (clique sobre o disco na imagem)")
    print("  [T] - Abrir controles deslizantes (Trackbars HSV)")
    print("  [M] - Alternar modo de visão (Anotado / Máscara HSV / Lado a Lado)")
    print("  [S] - Salvar configuração atual em arquivo JSON")
    print("  [ESPAÇO] - Pausar/Continuar vídeo")
    print("  [Q] ou [ESC] - Sair")
    print("=" * 65 + "\n")

    try:
        while True:
            if not paused or current_raw is None:
                if source_type == "image":
                    current_raw = img.copy()
                else:
                    ret, frame_cap = cap.read()
                    if not ret:
                        # Se for vídeo, reinicia para loop
                        if source_type == "video":
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            break
                    current_raw = frame_cap

            mouse_state["current_frame"] = current_raw
            vis, mask_bgr, diag = inspector.process_frame(current_raw)

            # Atualiza sliders se a janela estiver aberta
            if inspector.show_trackbars:
                tb_win = "HSV Ajuste Fino - Disco"
                try:
                    h_min = cv2.getTrackbarPos("H Min", tb_win)
                    h_max = cv2.getTrackbarPos("H Max", tb_win)
                    s_min = cv2.getTrackbarPos("S Min", tb_win)
                    s_max = cv2.getTrackbarPos("S Max", tb_win)
                    v_min = cv2.getTrackbarPos("V Min", tb_win)
                    v_max = cv2.getTrackbarPos("V Max", tb_win)
                    inspector.set_puck_hsv([h_min, s_min, v_min], [h_max, s_max, v_max])
                except Exception:
                    pass

            # Exibição de acordo com o modo
            if inspector.view_mode == 0:
                display_img = vis
            elif inspector.view_mode == 1:
                display_img = mask_bgr
            else:
                # Lado a lado redimensionado se necessário
                display_img = np.hstack((vis, mask_bgr))

            cv2.imshow(window_name, display_img)

            if writer is not None:
                writer.write(vis)

            key = cv2.waitKey(20 if source_type != "image" else 50) & 0xFF

            if key in (ord("q"), 27):  # Q ou ESC
                break
            elif key == ord("c"):
                inspector.calibrating_corners = True
                inspector.clicked_corners = []
                print("[Calibração] Modo calibração ativado. Clique nos 4 cantos na ordem: TL -> TR -> BR -> BL")
            elif key == ord("p"):
                inspector.picking_puck_color = True
                print("[Calibração] Modo cor do disco ativado. Clique sobre o centro do disco na imagem.")
            elif key == ord("t"):
                inspector.show_trackbars = not inspector.show_trackbars
                if inspector.show_trackbars:
                    create_trackbars()
                else:
                    cv2.destroyWindow("HSV Ajuste Fino - Disco")
            elif key == ord("m"):
                inspector.view_mode = (inspector.view_mode + 1) % 3
                modes = ["Anotado Normal", "Mascara HSV", "Lado a Lado"]
                print(f"[Visão] Modo alterado para: {modes[inspector.view_mode]}")
            elif key == ord("s"):
                inspector.save_config()
            elif key == ord(" "):
                paused = not paused
                print("[Vídeo] Pausado" if paused else "[Vídeo] Reproduzindo")

    finally:
        if writer is not None:
            writer.release()
            print(f"[Arquivo] Vídeo anotado salvo em: {output_path}")
        if source_type == "image" and output_path:
            cv2.imwrite(output_path, vis)
            print(f"[Arquivo] Imagem anotada salva em: {output_path}")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Caminho para uma foto (jpg, png) para testar reconhecimento")
    parser.add_argument("--video", help="Caminho para um arquivo de vídeo (mp4, avi)")
    parser.add_argument("--camera", type=int, default=None, help="Índice da câmera USB/webcam (ex: 0, 1)")
    parser.add_argument("--config", default="configs/camera_test.json", help="Arquivo de configuração JSON")
    parser.add_argument("--output", help="Caminho para salvar a foto ou vídeo resultante anotado")
    parser.add_argument("--profile", default="micro-2024", help="Perfil da arena (ex: micro-2024)")
    parser.add_argument(
        "--generate-markers",
        action="store_true",
        help="Gera marcadores ArUco prontos para imprimir na pasta docs/assets/markers",
    )
    parser.add_argument("--headless", action="store_true", help="Executa sem interface gráfica (para testes)")

    args = parser.parse_args()

    # Gerar marcadores
    if args.generate_markers:
        print("[Aruco] Gerando marcadores ArUco DICT_4X4_50...")
        files = generate_aruco_markers()
        for f in files:
            print(f"  -> Gerado: {f}")
        print("[Aruco] Marcadores gerados com sucesso!")
        if not (args.image or args.video or args.camera is not None):
            return

    # Inicia Inspetor
    inspector = VisionInspector(config_path=args.config, arena_profile=args.profile)

    # Identifica fonte
    if args.image:
        source_type = "image"
        source_val = args.image
    elif args.video:
        source_type = "video"
        source_val = args.video
    elif args.camera is not None:
        source_type = "camera"
        source_val = args.camera
    else:
        # Padrão: se nada foi passado, tenta câmera 0 ou mostra instrução
        source_type = "camera"
        source_val = 0

    if args.headless:
        # Modo headless para testes automáticos
        if source_type == "image":
            frame = cv2.imread(source_val)
            if frame is None:
                raise RuntimeError(f"Imagem não encontrada: {source_val}")
            vis, mask, diag = inspector.process_frame(frame, time.monotonic())
            if args.output:
                cv2.imwrite(args.output, vis)
            print(json.dumps(diag, indent=2, default=str))
        return

    run_interactive(source_type, source_val, inspector, output_path=args.output)


if __name__ == "__main__":
    main()
