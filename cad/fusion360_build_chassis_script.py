"""Fusion 360 script: import the shared outline and extrude a 3 mm design plate.

Does not create motors, holes, electronics or a validated 454 g assembly.
Run the generator first. Fusion internal length units are centimetres.
"""
import json
from pathlib import Path
import traceback
import adsk.core
import adsk.fusion


def run(context):
    app=adsk.core.Application.get();ui=app.userInterface
    try:
        design=adsk.fusion.Design.cast(app.activeProduct)
        if not design:raise ValueError('Open a Fusion design first')
        specs=json.loads((Path(__file__).parent/'models/chassis_specs.json').read_text())
        component=design.rootComponent
        for name,data in specs.items():
            occurrence=component.occurrences.addNewComponent(adsk.core.Matrix3D.create())
            part=occurrence.component;part.name='DesignPlate_'+name
            sketch=part.sketches.add(part.xYConstructionPlane)
            points=data['outline_cm'];lines=sketch.sketchCurves.sketchLines
            for a,b in zip(points,points[1:]+points[:1]):
                lines.addByTwoPoints(adsk.core.Point3D.create(*a,0),adsk.core.Point3D.create(*b,0))
            inputs=part.features.extrudeFeatures.createInput(sketch.profiles.item(0),adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            inputs.setDistanceExtent(False,adsk.core.ValueInput.createByReal(data['plate_thickness_cm']))
            part.features.extrudeFeatures.add(inputs)
        ui.messageBox('Three design base plates imported. Validate mounting, material and mass before fabrication.')
    except Exception:ui.messageBox(traceback.format_exc())
