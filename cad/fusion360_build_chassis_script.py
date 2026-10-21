# Author: RoboCore Hockey RL Project
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
            ui.messageBox('Falha:\n{}'.format(traceback.format_exc()))
