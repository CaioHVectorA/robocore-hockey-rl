"""
Módulo de Visão Computacional Zenital para Hockey de Robôs RoboCore (3v3).

Responsável pelo Pipeline Sim-to-Real:
1. Calibração de Perspectiva (Homografia): Mapeia os 4 cantos da mesa da imagem para o plano
   cartesiano oficial [-90, +90] x [-60, +60] cm.
2. Rastreamento das Marcas no Chassi (Atacante, Ala, Goleiro):
   - Marca Frontal de Proa: LED / Ponto de orientação direcional
   - Marca Traseira de Popa: Cor/Tag do papel tático (Atacante, Ala, Goleiro)
3. Rastreamento do Disco (Puck) via Filtro de Cor HSV e Contornos.
4. Filtro de Kalman Cinemático para Eliminação de Ruído e Artefatos de Imagem.
5. SimulatedPartialVisionSensor: Simula na física e no RL a visão computacional realista,
   com ruídos de centróide, oclusões em disputas mecânicas e decaimento de confiança.
6. Construtor Automático do Vetor de Observação 27D para a Política RL.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import cv2
import numpy as np

from core.constants import HockeyConstants as C


class ChassisMarkingsSpec:
    """
    Especificação física das marcas de teto para rastreamento por visão:
    - Marca frontal (Proa): Ponto de alta luminosidade (Branco/Amarelo) para vetor de orientação chassi.
    - Marca traseira (Popa): Cor específica do papel tático (Atacante, Ala, Goleiro).
    """

    FRONT_COLOR_BGR = (245, 245, 255)  # Branco / LED de Proa

    # Time Azul:
    BLUE_ROLES_BGR = {
        0: (230, 180, 50),   # Atacante (0): Ciano / Azul Claro
        1: (50, 215, 255),   # Ala (1): Amarelo / Dourado
        2: (80, 240, 120),   # Goleiro (2): Verde Lima
    }

    # Time Vermelho:
    RED_ROLES_BGR = {
        0: (50, 120, 255),   # Atacante (0): Laranja Intenso
        1: (200, 80, 240),   # Ala (1): Magenta / Rosa
        2: (220, 50, 150),   # Goleiro (2): Roxo
    }


class ArenaHomography:
    """Mapeia coordenadas de pixels da webcam para centímetros no ringue [-90, 90] x [-60, 60]."""

    def __init__(self, src_corners: Optional[np.ndarray] = None):
        # Destino métrico: Topo-Esq, Topo-Dir, Base-Dir, Base-Esq em centímetros
        self.dst_corners = np.array(
            [
                [-C.HALF_LENGTH, -C.HALF_WIDTH],  # (-90, -60)
                [C.HALF_LENGTH, -C.HALF_WIDTH],   # (+90, -60)
                [C.HALF_LENGTH, C.HALF_WIDTH],    # (+90, +60)
                [-C.HALF_LENGTH, C.HALF_WIDTH],   # (-90, +60)
            ],
            dtype=np.float32,
        )
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        if src_corners is not None:
            self.set_source_corners(src_corners)

    def set_source_corners(self, src_corners: np.ndarray) -> None:
        """src_corners: 4 pontos [4, 2] na imagem em pixels."""
        src = np.array(src_corners, dtype=np.float32)
        self.H, _ = cv2.findHomography(src, self.dst_corners)
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)

    def px_to_cm(self, px: float, py: float) -> Tuple[float, float]:
        """Converte ponto (pixel_x, pixel_y) para (x_cm, y_cm) da arena."""
        if self.H is None:
            return 0.0, 0.0
        pt = np.array([[[px, py]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self.H)
        return float(warped[0][0][0]), float(warped[0][0][1])

    def cm_to_px(self, cm_x: float, cm_y: float) -> Tuple[int, int]:
        """Converte ponto (cm_x, cm_y) para (pixel_x, pixel_y)."""
        if self.H_inv is None:
            return 0, 0
        pt = np.array([[[cm_x, cm_y]]], dtype=np.float32)
        unwarped = cv2.perspectiveTransform(pt, self.H_inv)
        return int(unwarped[0][0][0]), int(unwarped[0][0][1])


class EntityKalmanFilter:
    """Filtro de Kalman linear para suavizar jitter de imagem e estimar velocidades vx, vy."""

    def __init__(self, dt: float = 1.0 / 60.0):
        self.dt = dt
        # Estado: [x, y, vx, vy]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        # Covariâncias de ruído
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.15
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.40
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.last_update = time.time()
        self.is_initialized = False

    def reset(self) -> None:
        self.is_initialized = False

    def update(self, measured_x: float, measured_y: float) -> Tuple[float, float, float, float]:
        measurement = np.array([[measured_x], [measured_y]], dtype=np.float32)
        if not self.is_initialized:
            self.kf.statePost = np.array(
                [[measured_x], [measured_y], [0.0], [0.0]], dtype=np.float32
            )
            self.is_initialized = True
            return measured_x, measured_y, 0.0, 0.0

        self.kf.predict()
        est = self.kf.correct(measurement)
        x = float(est[0][0])
        y = float(est[1][0])
        vx = float(est[2][0])
        vy = float(est[3][0])
        return x, y, vx, vy

    def predict_only(self) -> Tuple[float, float, float, float]:
        """Predição quando o objeto é temporariamente ocluído por outro robô."""
        if not self.is_initialized:
            return 0.0, 0.0, 0.0, 0.0
        est = self.kf.predict()
        return float(est[0][0]), float(est[1][0]), float(est[2][0]), float(est[3][0])


@dataclass
class TrackedEntityState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    tracked: bool = True
    confidence: float = 1.0
    occlusion_ticks: int = 0
    raw_front: Tuple[float, float] = (0.0, 0.0)
    raw_rear: Tuple[float, float] = (0.0, 0.0)


class SimulatedPartialVisionSensor:
    """
    Sensor de Visão Computacional Parcial para o Simulador de Hockey 3v3.
    Simula o rastreamento das marcas físicas no teto do chassi de cada robô (Atacante, Ala, Goleiro):
    - Ponto Frontal: Marca de proa (LED branco / dot de orientação)
    - Ponto Traseiro: Marca de popa com a cor tática do papel
    Modela os artefatos de visão real:
    - Ruído gaussiano de quantização de pixel e calibração
    - Oclusões parciais momentâneas (dropouts de 1-4 quadros)
    - Oclusões decorrentes de choques mecânicos entre robôs
    - Oclusão do puck ao entrar na pá côncava
    - Filtro de Kalman cinemático em tempo real que reconstrói a pose contínua
    """

    def __init__(self, enable_dropouts: bool = True):
        self.enable_dropouts = enable_dropouts
        # 6 Robôs: 0..2 = Time Azul (Atacante, Ala, Goleiro); 3..5 = Time Vermelho
        self.tracked_robots: Dict[int, TrackedEntityState] = {
            i: TrackedEntityState() for i in range(6)
        }
        self.tracked_puck = TrackedEntityState()

        self.robot_kalmans = {i: EntityKalmanFilter() for i in range(6)}
        self.puck_kalman = EntityKalmanFilter()

    def reset(self) -> None:
        for kf in self.robot_kalmans.values():
            kf.reset()
        self.puck_kalman.reset()
        for s in self.tracked_robots.values():
            s.confidence = 1.0
            s.tracked = True
            s.occlusion_ticks = 0
        self.tracked_puck.confidence = 1.0
        self.tracked_puck.tracked = True
        self.tracked_puck.occlusion_ticks = 0

    def update(self, engine: Any) -> Tuple[Dict[int, TrackedEntityState], TrackedEntityState]:
        """
        Gera a leitura imperfeita da câmera a partir do estado físico real do simulador.
        """
        all_robs = engine.all_robots  # 3 azuis + 3 vermelhos
        puck = engine.puck

        # 1. Rastreamento das Marcas no Chassi dos Robôs
        for idx, r in enumerate(all_robs):
            state = self.tracked_robots[idx]
            cos_t = math.cos(r.theta)
            sin_t = math.sin(r.theta)

            # Posições físicas das marcas no teto do chassi
            front_true_x = r.x + C.MARKER_FRONT_OFFSET * cos_t
            front_true_y = r.y + C.MARKER_FRONT_OFFSET * sin_t
            rear_true_x = r.x + C.MARKER_REAR_OFFSET * cos_t
            rear_true_y = r.y + C.MARKER_REAR_OFFSET * sin_t

            # Cálculo de risco de oclusão parcial (choques mecânicos ou dropouts de visão)
            is_occluded = False
            if self.enable_dropouts:
                # Dropout aleatório básico de frame
                if float(np.random.rand()) < C.VISION_DROPOUT_PROB:
                    is_occluded = True

                # Proximidade com outros robôs aumenta risco de oclusão de visão
                for other_idx, other_r in enumerate(all_robs):
                    if other_idx != idx:
                        d = math.hypot(other_r.x - r.x, other_r.y - r.y)
                        if d < 18.0 and float(np.random.rand()) < C.VISION_COLLISION_OCCLUSION_PROB:
                            is_occluded = True
                            break

            if not is_occluded:
                # Marcas detectadas pela câmera com ruído de pixel/calibração
                n_fx = float(np.random.normal(0.0, C.VISION_POS_NOISE_STD))
                n_fy = float(np.random.normal(0.0, C.VISION_POS_NOISE_STD))
                n_rx = float(np.random.normal(0.0, C.VISION_POS_NOISE_STD))
                n_ry = float(np.random.normal(0.0, C.VISION_POS_NOISE_STD))

                meas_fx = front_true_x + n_fx
                meas_fy = front_true_y + n_fy
                meas_rx = rear_true_x + n_rx
                meas_ry = rear_true_y + n_ry

                # Centróide e orientação extraídos das 2 marcas
                meas_cx = (meas_fx + meas_rx) * 0.5
                meas_cy = (meas_fy + meas_ry) * 0.5
                meas_theta = math.atan2(meas_fy - meas_ry, meas_fx - meas_rx)

                # Atualiza Filtro de Kalman
                kf_x, kf_y, kf_vx, kf_vy = self.robot_kalmans[idx].update(meas_cx, meas_cy)

                state.x = kf_x
                state.y = kf_y
                state.theta = meas_theta
                state.vx = kf_vx
                state.vy = kf_vy
                state.tracked = True
                state.confidence = min(1.0, state.confidence + 0.12)
                state.occlusion_ticks = 0
                state.raw_front = (meas_fx, meas_fy)
                state.raw_rear = (meas_rx, meas_ry)
            else:
                # Oclusão Parcial: Câmera não viu as marcas neste frame -> Predição pura por Kalman
                pred_x, pred_y, pred_vx, pred_vy = self.robot_kalmans[idx].predict_only()
                state.x = pred_x
                state.y = pred_y
                # Mantém último theta com leve ruído de inércia
                state.vx = pred_vx
                state.vy = pred_vy
                state.tracked = False
                state.confidence = max(0.05, state.confidence * C.VISION_CONFIDENCE_DECAY)
                state.occlusion_ticks += 1

        # 2. Rastreamento do Puck (com possível oclusão ao entrar na pá frontal)
        puck_occluded = False
        if self.enable_dropouts:
            if float(np.random.rand()) < C.VISION_DROPOUT_PROB:
                puck_occluded = True
            # Se o disco está dentro da pá de algum robô, câmera pode perder o contorno por instantes
            for r in all_robs:
                if r.has_puck_in_scoop and float(np.random.rand()) < 0.22:
                    puck_occluded = True
                    break

        p_state = self.tracked_puck
        if not puck_occluded:
            p_nx = float(np.random.normal(0.0, C.VISION_POS_NOISE_STD))
            p_ny = float(np.random.normal(0.0, C.VISION_POS_NOISE_STD))
            pk_x, pk_y, pk_vx, pk_vy = self.puck_kalman.update(puck.x + p_nx, puck.y + p_ny)
            p_state.x = pk_x
            p_state.y = pk_y
            p_state.vx = pk_vx
            p_state.vy = pk_vy
            p_state.tracked = True
            p_state.confidence = min(1.0, p_state.confidence + 0.15)
            p_state.occlusion_ticks = 0
        else:
            pk_x, pk_y, pk_vx, pk_vy = self.puck_kalman.predict_only()
            p_state.x = pk_x
            p_state.y = pk_y
            p_state.vx = pk_vx
            p_state.vy = pk_vy
            p_state.tracked = False
            p_state.confidence = max(0.05, p_state.confidence * C.VISION_CONFIDENCE_DECAY)
            p_state.occlusion_ticks += 1

        return self.tracked_robots, self.tracked_puck


class HockeyVisionTracker:
    """
    Rastreador de Visão Computacional para o Hockey 3v3 real.
    Rastreia marcas físicas no teto do chassi e o puck na imagem da webcam.
    """

    def __init__(self, homography: Optional[ArenaHomography] = None):
        self.homography = homography or ArenaHomography()

        # Detector de ArUco Markers
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Filtros de Kalman para os 6 robôs e para o puck
        self.robot_kalmans = {i: EntityKalmanFilter() for i in range(6)}
        self.puck_kalman = EntityKalmanFilter()

        # Faixa HSV do mini puck
        self.puck_hsv_lower = np.array([0, 0, 0])
        self.puck_hsv_upper = np.array([180, 255, 65])

    def detect_robots_and_puck(
        self, frame: np.ndarray
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
        """
        Processa 1 quadro da câmera.
        Retorna dicionários de coordenadas métricas dos robôs e do puck.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        robots_data: Dict[int, Dict[str, float]] = {}

        if ids is not None and len(ids) > 0:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                pts = marker_corners[0]
                cx_px = float(np.mean(pts[:, 0]))
                cy_px = float(np.mean(pts[:, 1]))

                front_px = (pts[0] + pts[1]) * 0.5
                rear_px = (pts[2] + pts[3]) * 0.5

                cm_x, cm_y = self.homography.px_to_cm(cx_px, cy_px)

                f_cm_x, f_cm_y = self.homography.px_to_cm(front_px[0], front_px[1])
                r_cm_x, r_cm_y = self.homography.px_to_cm(rear_px[0], rear_px[1])
                theta = math.atan2(f_cm_y - r_cm_y, f_cm_x - r_cm_x)

                kf = self.robot_kalmans.get(marker_id)
                if kf:
                    kx, ky, kvx, kvy = kf.update(cm_x, cm_y)
                else:
                    kx, ky, kvx, kvy = cm_x, cm_y, 0.0, 0.0

                robots_data[int(marker_id)] = {
                    "x": kx,
                    "y": ky,
                    "theta": theta,
                    "vx": kvx,
                    "vy": kvy,
                }

        # Detecção do Puck por HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.puck_hsv_lower, self.puck_hsv_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        puck_data = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0}

        best_cnt = None
        best_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 80 < area < 4000 and area > best_area:
                best_area = area
                best_cnt = cnt

        if best_cnt is not None:
            M = cv2.moments(best_cnt)
            if M["m00"] > 1e-4:
                pcx_px = M["m10"] / M["m00"]
                pcy_px = M["m01"] / M["m00"]
                pcm_x, pcm_y = self.homography.px_to_cm(pcx_px, pcy_px)
                pkx, pky, pkvx, pkvy = self.puck_kalman.update(pcm_x, pcm_y)
                puck_data = {"x": pkx, "y": pky, "vx": pkvx, "vy": pkvy}
        else:
            pkx, pky, pkvx, pkvy = self.puck_kalman.predict_only()
            puck_data = {"x": pkx, "y": pky, "vx": pkvx, "vy": pkvy}

        return robots_data, puck_data

    def build_team_obs_for_rl(
        self,
        robots_data: Dict[int, Any],
        puck_data: Any,
    ) -> np.ndarray:
        """
        Gera a matriz [3, 27] das observações dos 3 robôs do Time Azul (IDs 0, 1, 2)
        pronta para ser passada na rede neural do PPO!
        Aceita tanto dicionários quanto instâncias de TrackedEntityState.
        """
        team_obs = np.zeros((3, 27), dtype=np.float32)

        p_x = puck_data.x if hasattr(puck_data, "x") else puck_data["x"]
        p_y = puck_data.y if hasattr(puck_data, "y") else puck_data["y"]
        p_vx = puck_data.vx if hasattr(puck_data, "vx") else puck_data["vx"]
        p_vy = puck_data.vy if hasattr(puck_data, "vy") else puck_data["vy"]

        # Azul: IDs 0, 1, 2
        def get_rob(idx: int, default_x: float, default_theta: float) -> Tuple[float, float, float, float, float]:
            if idx in robots_data:
                item = robots_data[idx]
                if hasattr(item, "x"):
                    return item.x, item.y, item.theta, item.vx, item.vy
                return item["x"], item["y"], item["theta"], item["vx"], item["vy"]
            return default_x, 0.0, default_theta, 0.0, 0.0

        blue_robs = [get_rob(i, -50.0 + i * 15.0, 0.0) for i in range(3)]
        red_robs = [get_rob(i, 50.0 - (i - 3) * 15.0, math.pi) for i in range(3, 6)]

        v_scale = 120.0

        for i in range(3):
            rx, ry, theta, r_vx, r_vy = blue_robs[i]
            cos_t, sin_t = math.cos(theta), math.sin(theta)

            px = p_x / C.HALF_LENGTH
            py = p_y / C.HALF_WIDTH
            pvx = p_vx / v_scale
            pvy = p_vy / v_scale

            rel_px = (p_x - rx) / C.ARENA_LENGTH
            rel_py = (p_y - ry) / C.ARENA_WIDTH

            goal_dx = (C.HALF_LENGTH - p_x) / C.ARENA_LENGTH
            goal_dy = (0.0 - p_y) / C.ARENA_WIDTH

            own_goal_d = math.hypot(rx - (-C.HALF_LENGTH), ry - 0.0) / C.ARENA_LENGTH
            dist_to_puck = math.hypot(p_x - rx, p_y - ry)
            scoop_flag = 1.0 if dist_to_puck < (C.HALF_ROBOT_L + C.PUCK_RADIUS + 1.5) else 0.0
            role_norm = float(i) / 2.0

            # Vetores para os 2 companheiros de equipe
            teammates = [m for idx, m in enumerate(blue_robs) if idx != i]
            mate_vecs = []
            for m in teammates:
                mate_vecs.extend([(m[0] - rx) / C.ARENA_LENGTH, (m[1] - ry) / C.ARENA_WIDTH])

            # Vetores para os 3 oponentes
            opp_vecs = []
            for o in red_robs:
                opp_vecs.extend([(o[0] - rx) / C.ARENA_LENGTH, (o[1] - ry) / C.ARENA_WIDTH])

            v_lin = math.hypot(r_vx, r_vy)
            v_norm = min(1.0, v_lin / C.MAX_WHEEL_SPEED)

            team_obs[i] = np.array(
                [
                    px, py, pvx, pvy,
                    rel_px, rel_py,
                    goal_dx, goal_dy,
                    rx / C.HALF_LENGTH, ry / C.HALF_WIDTH,
                    cos_t, sin_t,
                    v_norm, v_norm,
                    scoop_flag,
                    own_goal_d,
                    role_norm,
                    *mate_vecs,
                    *opp_vecs,
                ],
                dtype=np.float32,
            )

        return team_obs
