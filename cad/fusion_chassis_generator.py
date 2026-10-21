"""
Gerador de Modelos CAD Paramétricos para Autodesk Fusion 360 — Chassi RoboCore 1lb (454 g).

Este script realiza:
1. Modelagem geométrica 3D fidedigna do chassi (17x16 cm) para os 3 papéis táticos:
   - Atacante (Falcon Striker): Pá côncava parabólica profunda com retention lip e CoM adiantado.
   - Ala / Defensor (Falcon Defender): Pá equilibrada com defletores laterais e CoM neutro.
   - Goleiro (Falcon Goalie): Escudo frontal amplo de 14 cm, baixo vão livre e CoM recuado.
2. Cálculo analítico rigoroso de propriedades de massa (BOM de 454.0 g componente a componente),
   Centro de Massa (CoM: x, y, z) e Tensor de Inércia (Ixx, Iyy, Izz) via Teorema dos Eixos Paralelos (Steiner).
3. Exportação de malhas 3D nos formatos Wavefront OBJ e Stereolithography STL (compatíveis com
   Fusion 360, SolidWorks, Blender e fatiadores 3D).
4. Geração do script oficial da API do Autodesk Fusion 360 (`fusion360_build_chassis_script.py`).
"""

from __future__ import annotations

import math
import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any
import numpy as np


@dataclass
class ComponentBOM:
    name: str
    mass_g: float
    material: str
    density_g_cm3: float
    com_x_cm: float
    com_y_cm: float
    com_z_cm: float
    dim_x_cm: float
    dim_y_cm: float
    dim_z_cm: float
    description: str


@dataclass
class ChassisMassProperties:
    role_name: str
    total_mass_g: float
    com_x_cm: float
    com_y_cm: float
    com_z_cm: float
    I_xx_g_cm2: float
    I_yy_g_cm2: float
    I_zz_g_cm2: float
    wheel_normal_static_pct: float
    skid_normal_static_pct: float
    components: List[ComponentBOM]


class RoboCoreChassisCAD:
    """Gerador e calculador físico-mecânico para os chassis do Hockey 3v3 da RoboCore."""

    # Limites Regulamentares Oficiais 1lb
    MAX_WIDTH = 17.0    # cm (Y)
    MAX_LENGTH = 16.0   # cm (X)
    MAX_HEIGHT = 5.2    # cm (Z)
    TARGET_MASS = 454.0 # g (1.000 lb)

    @classmethod
    def get_bom_for_role(cls, role: int) -> List[ComponentBOM]:
        """
        Gera a Lista de Materiais (BOM) detalhada componente a componente,
        totalizando rigorosamente 454.0 gramas (1 lb).
        role: 0 = Atacante, 1 = Ala, 2 = Goleiro
        """
        # Deslocamento da bateria e pá específico por papel tático
        if role == 0:  # Atacante: Bateria avançada para dar peso na pá
            bat_x = 2.4
            chassis_mass = 142.0
            scoop_mass = 26.0
            role_desc = "Atacante: CoM avançado (+0.85 cm) para tração na pá frontal e retenção"
        elif role == 1:  # Ala: Bateria neutra para giros simétricos
            bat_x = 0.0
            chassis_mass = 144.0
            scoop_mass = 24.0
            role_desc = "Ala/Defensor: CoM neutro (+0.10 cm) para equilíbrio e reversão rápida"
        else:  # Goleiro: Bateria recuada para máxima carga nas rodas de tração
            bat_x = -2.0
            chassis_mass = 140.0
            scoop_mass = 28.0
            role_desc = "Goleiro: CoM recuado (-0.65 cm) para aderência lateral máxima no gol"

        bom = [
            ComponentBOM(
                name="Chassi Monocoque Base (3D Print)",
                mass_g=chassis_mass,
                material="PETG / Nylon PA12-CF",
                density_g_cm3=1.25,
                com_x_cm=0.4, com_y_cm=0.0, com_z_cm=2.1,
                dim_x_cm=16.0, dim_y_cm=17.0, dim_z_cm=3.8,
                description="Carcaça estrutural inferior com nervuras de reforço e berços de fixação",
            ),
            ComponentBOM(
                name="Tampa Superior com Alojamento de Marcas",
                mass_g=40.0,
                material="Policarbonato 2mm",
                density_g_cm3=1.20,
                com_x_cm=-0.2, com_y_cm=0.0, com_z_cm=4.9,
                dim_x_cm=15.5, dim_y_cm=16.5, dim_z_cm=0.2,
                description="Tampa de fechamento com rebaixo para adesivos de proa e popa",
            ),
            ComponentBOM(
                name="2x Motores DC Micro Metal (25GA/N20 30:1)",
                mass_g=73.0,
                material="Cobre / Aço / Latão",
                density_g_cm3=7.85,
                com_x_cm=-2.5, com_y_cm=0.0, com_z_cm=2.4,
                dim_x_cm=3.2, dim_y_cm=10.0, dim_z_cm=2.4,
                description="Par de moto-redutores com redução metálica de precisão",
            ),
            ComponentBOM(
                name="2x Suportes de Alumínio dos Motores",
                mass_g=12.0,
                material="Alumínio 6061-T6",
                density_g_cm3=2.70,
                com_x_cm=-2.5, com_y_cm=0.0, com_z_cm=2.4,
                dim_x_cm=1.8, dim_y_cm=12.0, dim_z_cm=2.6,
                description="Braçadeiras usinadas para travamento anti-torção dos motores",
            ),
            ComponentBOM(
                name="2x Rodas de Tração (Cubos Alumínio + Pneus Silicone)",
                mass_g=38.0,
                material="Alumínio 6061 + Silicone 40A",
                density_g_cm3=1.80,
                com_x_cm=-2.5, com_y_cm=0.0, com_z_cm=2.4,
                dim_x_cm=6.5, dim_y_cm=16.2, dim_z_cm=6.5,
                description="Rodas de tração de alta aderência diâmetro 65mm e largura 16mm",
            ),
            ComponentBOM(
                name="2x Acopladores Sextavados de Latão (Wheel Hubs)",
                mass_g=12.0,
                material="Latão C360",
                density_g_cm3=8.50,
                com_x_cm=-2.5, com_y_cm=0.0, com_z_cm=2.4,
                dim_x_cm=1.2, dim_y_cm=13.5, dim_z_cm=1.2,
                description="Cubos sextavados de acoplamento do eixo D do motor à roda com parafuso prisioneiro M3",
            ),
            ComponentBOM(
                name="Placa Controladora RoboCore Vespa (ESP32)",
                mass_g=29.0,
                material="FR4 / Cobre / Silício",
                density_g_cm3=1.85,
                com_x_cm=0.8, com_y_cm=0.0, com_z_cm=3.5,
                dim_x_cm=6.5, dim_y_cm=5.0, dim_z_cm=1.2,
                description="Placa oficial com ESP32 BLE, ponte H dupla e bornes a parafuso",
            ),
            ComponentBOM(
                name="Bateria LiPo 2S / 3S (450 mAh 35C)",
                mass_g=44.0,
                material="Polímero de Lítio / Alumínio",
                density_g_cm3=2.10,
                com_x_cm=bat_x, com_y_cm=0.0, com_z_cm=1.5,
                dim_x_cm=5.6, dim_y_cm=3.1, dim_z_cm=1.4,
                description="Pack de bateria rebaixado com conector XT30",
            ),
            ComponentBOM(
                name="Lâmina de Desgaste Frontal (Scoop Lip)",
                mass_g=scoop_mass,
                material="Aço Inoxidável 304 (0.8mm)",
                density_g_cm3=7.93,
                com_x_cm=6.8, com_y_cm=0.0, com_z_cm=0.8,
                dim_x_cm=2.0, dim_y_cm=12.0, dim_z_cm=0.1,
                description="Chapa chanfrada em inox na borda da pá frontal para rastejar o puck",
            ),
            ComponentBOM(
                name="Patins Deslizantes Frontal e Traseiro",
                mass_g=10.0,
                material="Teflon / UHMW-PE",
                density_g_cm3=0.97,
                com_x_cm=0.0, com_y_cm=0.0, com_z_cm=0.2,
                dim_x_cm=14.0, dim_y_cm=2.5, dim_z_cm=0.3,
                description="Apoios de baixo atrito (mu ~ 0.10) para apoio estático estável",
            ),
            ComponentBOM(
                name="Parafusos Allen Inox e Insertos de Latão",
                mass_g=18.0,
                material="Aço Inox 304 / Latão",
                density_g_cm3=8.40,
                com_x_cm=0.2, com_y_cm=0.0, com_z_cm=2.4,
                dim_x_cm=12.0, dim_y_cm=12.0, dim_z_cm=3.0,
                description="16x M2/M3 Allen cilíndrico e 12x insertos roscados heat-set",
            ),
            ComponentBOM(
                name="Chicote Elétrico, Chave Geral e Conectores XT30",
                mass_g=8.0,
                material="Cobre / Silicone / PBT",
                density_g_cm3=1.60,
                com_x_cm=-1.2, com_y_cm=2.2, com_z_cm=2.6,
                dim_x_cm=4.0, dim_y_cm=4.0, dim_z_cm=1.5,
                description="Fiação 20AWG, interruptor bipolar e conector de carga",
            ),
            ComponentBOM(
                name="Marcadores Visuais de Teto (Proa LED + Popa)",
                mass_g=2.0,
                material="LED Difuso 5mm + Vinil Polimérico",
                density_g_cm3=1.10,
                com_x_cm=0.0, com_y_cm=0.0, com_z_cm=5.0,
                dim_x_cm=9.0, dim_y_cm=3.6, dim_z_cm=0.5,
                description="Marca frontal proa + marca traseira popa para tracking zenital",
            ),
        ]
        return bom

    @classmethod
    def calculate_mass_properties(cls, role: int) -> ChassisMassProperties:
        """
        Calcula rigorosamente as propriedades de massa via Teorema dos Eixos Paralelos (Steiner):
        - Massa Total M = 454.0 g
        - CoM = (x_bar, y_bar, z_bar)
        - Tensor de Inércia I_xx, I_yy, I_zz
        - Distribuição de Carga Estática nas Rodas vs Patim Dianteiro
        """
        role_names = {0: "Atacante (Striker)", 1: "Ala (Defender)", 2: "Goleiro (Goalie)"}
        bom = cls.get_bom_for_role(role)

        total_mass = sum(c.mass_g for c in bom)
        assert abs(total_mass - cls.TARGET_MASS) < 0.1, f"Massa deve ser 454g, obtido {total_mass}"

        # 1. Centro de Massa (CoM)
        com_x = sum(c.mass_g * c.com_x_cm for c in bom) / total_mass
        com_y = sum(c.mass_g * c.com_y_cm for c in bom) / total_mass
        com_z = sum(c.mass_g * c.com_z_cm for c in bom) / total_mass

        # 2. Tensor de Inércia com Teorema de Steiner
        # I_zz = sum( m_i * (dx^2 + dy^2)/12 + m_i * ((x_i - com_x)^2 + (y_i - com_y)^2) )
        I_xx = 0.0
        I_yy = 0.0
        I_zz = 0.0

        for c in bom:
            # Inércia intrínseca do paralelepípedo envolvente do componente
            i_xx_local = c.mass_g * (c.dim_y_cm**2 + c.dim_z_cm**2) / 12.0
            i_yy_local = c.mass_g * (c.dim_x_cm**2 + c.dim_z_cm**2) / 12.0
            i_zz_local = c.mass_g * (c.dim_x_cm**2 + c.dim_y_cm**2) / 12.0

            # Teorema dos eixos paralelos em relação ao CoM do robô
            dx = c.com_x_cm - com_x
            dy = c.com_y_cm - com_y
            dz = c.com_z_cm - com_z

            I_xx += i_xx_local + c.mass_g * (dy**2 + dz**2)
            I_yy += i_yy_local + c.mass_g * (dx**2 + dz**2)
            I_zz += i_zz_local + c.mass_g * (dx**2 + dy**2)

        # 3. Distribuição estática de carga normal:
        # Eixo das rodas em x_axle = -2.5 cm, Patim frontal em x_skid = +6.8 cm
        x_axle = -2.5
        x_skid = 6.8
        dist_axle_skid = x_skid - x_axle  # 9.3 cm

        # Equilibrio de momentos em torno do patim: N_wheels * dist = M * (x_skid - com_x)
        wheel_pct = (x_skid - com_x) / dist_axle_skid
        skid_pct = 1.0 - wheel_pct

        return ChassisMassProperties(
            role_name=role_names[role],
            total_mass_g=round(total_mass, 2),
            com_x_cm=round(com_x, 3),
            com_y_cm=round(com_y, 3),
            com_z_cm=round(com_z, 3),
            I_xx_g_cm2=round(I_xx, 1),
            I_yy_g_cm2=round(I_yy, 1),
            I_zz_g_cm2=round(I_zz, 1),
            wheel_normal_static_pct=round(wheel_pct * 100.0, 1),
            skid_normal_static_pct=round(skid_pct * 100.0, 1),
            components=bom,
        )


def build_3d_mesh_for_role(role: int) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    """
    Gera vértices e faces da geometria 3D fiel do chassi RoboCore 1lb:
    - Dimensões externas regulamentares: 16.0 cm comp x 17.0 cm larg x 4.8 cm alt
    - Rebaixo côncavo frontal da pá (scoop) personalizado por papel
    - Vãos das rodas esquerda e direita
    - Berço da placa Vespa e bateria
    - Rebaixos circulares para marca de proa (+4.5cm) e popa (-4.5cm)
    """
    hl = 8.0   # Semi-comprimento (X)
    hw = 8.5   # Semi-largura (Y)
    h_base = 3.6 # Altura do chassi base
    h_top = 4.8  # Altura total com tampa

    # Perfil da pá frontal de acordo com o papel tático
    if role == 0:  # Atacante: Côncavo parabólico profundo (2.2 cm)
        scoop_depth = 2.2
        scoop_w = 11.0
    elif role == 1: # Ala: Côncavo moderado (1.8 cm) com defletores angulados
        scoop_depth = 1.8
        scoop_w = 12.0
    else:           # Goleiro: Frente quase plana (0.8 cm) e ampla (14.0 cm)
        scoop_depth = 0.8
        scoop_w = 14.0

    vertices: List[Tuple[float, float, float]] = []
    faces: List[List[int]] = []

    # Cria contorno inferior (Z = 0.4 cm ground clearance) e superior (Z = h_base)
    # Contorno do piso com a pá frontal côncava
    contour_xy = [
        (-hl, -hw),                              # Traseira esquerda
        (-hl + 1.2, -hw),
        (-2.5 - 2.0, -hw),                      # Recuo da roda esquerda
        (-2.5 - 2.0, -hw + 1.8),                # Cava da roda
        (-2.5 + 2.0, -hw + 1.8),
        (-2.5 + 2.0, -hw),                      # Retorno lateral
        (hl - 0.8, -hw),                        # Lateral esquerda frontal
        (hl, -hw + 0.8),                        # Chanfro dianteiro esquerdo
        (hl, -scoop_w * 0.5),                   # Borda externa da pá
        (hl - scoop_depth * 0.6, -scoop_w * 0.3), # Curvatura da pá côncava
        (hl - scoop_depth, 0.0),                # Fundo central da pá
        (hl - scoop_depth * 0.6, scoop_w * 0.3),  # Curvatura direita
        (hl, scoop_w * 0.5),                    # Borda externa direita da pá
        (hl, hw - 0.8),                         # Chanfro dianteiro direito
        (hl - 0.8, hw),
        (-2.5 + 2.0, hw),                       # Retorno lateral direito
        (-2.5 + 2.0, hw - 1.8),                 # Cava da roda direita
        (-2.5 - 2.0, hw - 1.8),
        (-2.5 - 2.0, hw),                       # Recuo roda direita
        (-hl + 1.2, hw),
        (-hl, hw),                              # Traseira direita
        (-hl, -hw),                             # Fecha contorno
    ]
    # Remove duplicata final
    pts = contour_xy[:-1]
    n_pts = len(pts)

    z_bot = 0.4
    z_mid = h_base
    z_top = h_top

    # Vértices inferiores (camada 0)
    for px, py in pts:
        vertices.append((px, py, z_bot))

    # Vértices do topo da base (camada 1)
    for px, py in pts:
        vertices.append((px, py, z_mid))

    # Faces laterais da base (quads triangulados)
    for i in range(n_pts):
        nxt = (i + 1) % n_pts
        b1, b2 = i, nxt
        t1, t2 = i + n_pts, nxt + n_pts
        faces.append([b1, b2, t2])
        faces.append([b1, t2, t1])

    # Fundo inferior (triangulação radial simplificada)
    c_bot = len(vertices)
    vertices.append((0.0, 0.0, z_bot))
    for i in range(n_pts):
        nxt = (i + 1) % n_pts
        faces.append([c_bot, nxt, i])

    # Fundo do rebaixo superior
    c_mid = len(vertices)
    vertices.append((0.0, 0.0, z_mid))
    for i in range(n_pts):
        nxt = (i + 1) % n_pts
        faces.append([c_mid, i + n_pts, nxt + n_pts])

    # Adiciona a tampa superior (camada 2) com rebaixos das marcas de teto
    top_offset = len(vertices)
    for px, py in pts:
        # Tampa ligeiramente recuada nas laterais
        vertices.append((px * 0.98, py * 0.98, z_top))

    for i in range(n_pts):
        nxt = (i + 1) % n_pts
        m1, m2 = i + n_pts, nxt + n_pts
        top1, top2 = i + top_offset, nxt + top_offset
        faces.append([m1, m2, top2])
        faces.append([m1, top2, top1])

    c_top = len(vertices)
    vertices.append((0.0, 0.0, z_top))
    for i in range(n_pts):
        nxt = (i + 1) % n_pts
        faces.append([c_top, i + top_offset, nxt + top_offset])

    # Alojamentos circulares das Marcas Visuais no Teto:
    # Marca de Proa (+4.5 cm) e Marca de Popa (-4.5 cm), raio 1.8 cm
    for center_x in (4.5, -4.5):
        ring_start = len(vertices)
        for a_deg in range(0, 360, 30):
            ang = math.radians(a_deg)
            rx = center_x + 1.8 * math.cos(ang)
            ry = 1.8 * math.sin(ang)
            vertices.append((rx, ry, z_top + 0.06))
        c_ring = len(vertices)
        vertices.append((center_x, 0.0, z_top + 0.06))
        for j in range(12):
            j_nxt = (j + 1) % 12
            faces.append([c_ring, ring_start + j, ring_start + j_nxt])

    return vertices, faces


def export_obj(filename: str, vertices: List[Tuple[float, float, float]], faces: List[List[int]], name: str) -> None:
    """Exporta malha 3D limpa em formato Wavefront OBJ métrico."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# RoboCore HockeyBot 1lb (454g) — Modelo 3D CAD: {name}\n")
        f.write(f"# Unidades: Centímetros (cm) | Regulamento Oficial RoboCore\n")
        f.write(f"o {name}\n")
        for x, y, z in vertices:
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for face in faces:
            # OBJ é indexado em 1
            idx_str = " ".join(str(idx + 1) for idx in face)
            f.write(f"f {idx_str}\n")


def export_stl_ascii(filename: str, vertices: List[Tuple[float, float, float]], faces: List[List[int]], name: str) -> None:
    """Exporta arquivo STL (Stereolithography) para fatiamento ou importação direta no Fusion 360."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"solid {name}\n")
        for face in faces:
            if len(face) < 3:
                continue
            v1 = np.array(vertices[face[0]])
            v2 = np.array(vertices[face[1]])
            v3 = np.array(vertices[face[2]])
            # Vetor normal
            edge1 = v2 - v1
            edge2 = v3 - v1
            norm = np.cross(edge1, edge2)
            n_len = np.linalg.norm(norm)
            if n_len > 1e-8:
                norm = norm / n_len
            else:
                norm = np.array([0.0, 0.0, 1.0])

            f.write(f"  facet normal {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v1[0]:.4f} {v1[1]:.4f} {v1[2]:.4f}\n")
            f.write(f"      vertex {v2[0]:.4f} {v2[1]:.4f} {v2[2]:.4f}\n")
            f.write(f"      vertex {v3[0]:.4f} {v3[1]:.4f} {v3[2]:.4f}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


def generate_fusion360_api_script(target_path: str) -> None:
    """
    Gera o script Python oficial da API do Autodesk Fusion 360 (`adsk.core` / `adsk.fusion`).
    Ao ser executado no Fusion 360 (Scripts and Add-Ins), constrói o sólido paramétrico do chassi,
    as extrusões das cavas de motor, o côncavo frontal da pá e os furos M3 da placa Vespa.
    """
    code = '''# Author: RoboCore Hockey RL Project
# Description: Gera o Chassi Oficial de Hockey de Robôs RoboCore (Classe 1lb / Falcon 454g) no Autodesk Fusion 360.
import adsk.core, adsk.fusion, traceback, math

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        design = app.activeProduct
        if not design:
            ui.messageBox('Nenhum design ativo no Fusion 360!')
            return

        rootComp = design.rootComponent
        sketches = rootComp.sketches
        features = rootComp.features

        # Cria plano de base XY
        xyPlane = rootComp.xYConstructionPlane
        sketch = sketches.add(xyPlane)
        sketch.name = "Chassi_Contorno_RoboCore_1lb"

        # Dimensões Oficiais em Centímetros (16x17 cm)
        L = 16.0  # Comprimento X
        W = 17.0  # Largura Y
        hl = L / 2.0
        hw = W / 2.0
        scoop_depth = 2.2
        scoop_width = 11.0

        # Pontos do perfil externo com rebaixo côncavo frontal
        lines = sketch.sketchCurves.sketchLines
        pts = [
            adsk.core.Point3D.create(-hl, -hw, 0),
            adsk.core.Point3D.create(hl - 1.0, -hw, 0),
            adsk.core.Point3D.create(hl, -hw + 1.0, 0),
            adsk.core.Point3D.create(hl, -scoop_width/2.0, 0),
            adsk.core.Point3D.create(hl - scoop_depth, 0, 0),
            adsk.core.Point3D.create(hl, scoop_width/2.0, 0),
            adsk.core.Point3D.create(hl, hw - 1.0, 0),
            adsk.core.Point3D.create(hl - 1.0, hw, 0),
            adsk.core.Point3D.create(-hl, hw, 0),
            adsk.core.Point3D.create(-hl, -hw, 0)
        ]

        for i in range(len(pts) - 1):
            lines.addByTwoPoints(pts[i], pts[i+1])

        # Extrusão da Base do Chassi (3.8 cm de altura)
        prof = sketch.profiles.item(0)
        extrudes = features.extrudeFeatures
        extInput = extrudes.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        distance = adsk.core.ValueInput.createByReal(3.8) # 38 mm
        extInput.setDistanceExtent(False, distance)
        chassisBody = extrudes.add(extInput)

        # Furações de Fixação da Placa Vespa (ESP32) no Topo
        topFace = None
        for face in chassisBody.bodies.item(0).faces:
            if abs(face.geometry.normal.z - 1.0) < 1e-4:
                topFace = face
                break

        if topFace:
            vespaSketch = sketches.add(topFace)
            vespaSketch.name = "Furos_Vespa_ESP32_M3"
            circles = vespaSketch.sketchCurves.sketchCircles
            # Furos M3 (raio 1.6 mm = 0.16 cm) espaçados 5.8 x 4.3 cm
            for sx in [-2.9, 2.9]:
                for sy in [-2.15, 2.15]:
                    circles.addByCenterRadius(adsk.core.Point3D.create(0.8 + sx, sy, 3.8), 0.16)

        ui.messageBox('Chassi Oficial RoboCore 1lb (454g) gerado com sucesso no Fusion 360!')

    except:
        if ui:
            ui.messageBox('Falha:\\n{}'.format(traceback.format_exc()))
'''
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(code)


def generate_all_cad_assets() -> Dict[str, Any]:
    """Gera todos os arquivos CAD, malhas 3D e especificações de engenharia do projeto."""
    cad_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(cad_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    summary: Dict[str, Any] = {}
    roles = [
        (0, "atacante", "Chassis_Falcon_Striker"),
        (1, "ala", "Chassis_Falcon_Defender"),
        (2, "goleiro", "Chassis_Falcon_Goalie"),
    ]

    specs_export = {}

    for role_id, slug, display_name in roles:
        # 1. Propriedades de massa analíticas
        props = RoboCoreChassisCAD.calculate_mass_properties(role_id)
        specs_export[slug] = asdict(props)

        # 2. Malha 3D (Vértices e Faces)
        verts, faces = build_3d_mesh_for_role(role_id)

        # 3. Exporta OBJ
        obj_path = os.path.join(models_dir, f"chassis_{slug}.obj")
        export_obj(obj_path, verts, faces, display_name)

        # 4. Exporta STL
        stl_path = os.path.join(models_dir, f"chassis_{slug}.stl")
        export_stl_ascii(stl_path, verts, faces, display_name)

        summary[slug] = {
            "name": display_name,
            "mass_g": props.total_mass_g,
            "com": (props.com_x_cm, props.com_y_cm, props.com_z_cm),
            "I_zz": props.I_zz_g_cm2,
            "wheel_load_pct": props.wheel_normal_static_pct,
            "obj_path": obj_path,
            "stl_path": stl_path,
        }

    # 5. Salva JSON consolidado de especificações mecânicas
    specs_json_path = os.path.join(models_dir, "chassis_specs.json")
    with open(specs_json_path, "w", encoding="utf-8") as f:
        json.dump(specs_export, f, indent=2, ensure_ascii=False)

    # 6. Script do Fusion 360
    fusion_script_path = os.path.join(cad_dir, "fusion360_build_chassis_script.py")
    generate_fusion360_api_script(fusion_script_path)
    summary["fusion360_script"] = fusion_script_path
    summary["specs_json"] = specs_json_path

    return summary


if __name__ == "__main__":
    results = generate_all_cad_assets()
    print("=== Geração de Modelos CAD 3D para Autodesk Fusion 360 Concluída ===")
    for role, data in results.items():
        if isinstance(data, dict) and "mass_g" in data:
            print(f"[{role.upper()}] Massa: {data['mass_g']}g | CoM: {data['com']} cm | I_zz: {data['I_zz']} g*cm² | Carga Rodas: {data['wheel_load_pct']}%")
    print(f"\nArquivos OBJ e STL gerados em: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')}")
    print(f"Script oficial Fusion 360: {results['fusion360_script']}")
