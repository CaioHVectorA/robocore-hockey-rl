"""Export the photo-derived stock outline as watertight OBJ/STL base plates.

This is a design aid, not a reverse-engineered/certified HockeyBot assembly.
STL is unitless: import using centimetres. Plate thickness is 3 mm.
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core.geometry import chassis_outline, wheel_footprints, estimated_inertia
from core.constants import RobotConfig


def triangulate(points):
    """Ear clipping for the shared simple, counterclockwise concave outline."""
    remaining = list(range(len(points)))
    triangles = []

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    while len(remaining) > 3:
        for i, idx in enumerate(remaining):
            prev = remaining[i - 1]
            nxt = remaining[(i + 1) % len(remaining)]
            a, b, c = points[prev], points[idx], points[nxt]
            if cross(a, b, c) <= 1e-9:
                continue

            def inside(p):
                return cross(a, b, p) >= -1e-9 and cross(b, c, p) >= -1e-9 and cross(c, a, p) >= -1e-9

            if any(inside(points[j]) for j in remaining if j not in (prev, idx, nxt)):
                continue
            triangles.append([prev, idx, nxt])
            remaining.pop(i)
            break
        else:
            raise ValueError("Outline is not a simple counterclockwise polygon")
    triangles.append(remaining)
    return triangles


def build_3d_mesh_for_role(role, robot=None):
    outline = chassis_outline(role, robot)
    n = len(outline)
    vertices = [(x, y, z) for z in (0.0, 0.3) for x, y in outline]
    faces = []
    for a, b, c in triangulate(outline):
        faces.extend([[c, b, a], [a + n, b + n, c + n]])
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[i, j, j + n], [i, j + n, i + n]])
    return vertices, faces


def export_obj(path, vertices, faces, name):
    with Path(path).open("w", encoding="utf8") as f:
        f.write(f"# Design base plate; units cm; {name}\no {name}\n")
        for x, y, z in vertices:
            f.write(f"v {x:.5f} {y:.5f} {z:.5f}\n")
        for face in faces:
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")


def export_stl_ascii(path, vertices, faces, name):
    with Path(path).open("w", encoding="utf8") as f:
        f.write(f"solid {name}\n")
        for face in faces:
            v = np.array([vertices[i] for i in face])
            normal = np.cross(v[1] - v[0], v[2] - v[0])
            normal /= np.linalg.norm(normal)
            f.write("facet normal " + " ".join(f"{x:.6f}" for x in normal) + "\nouter loop\n")
            for p in v:
                f.write("vertex " + " ".join(f"{x:.5f}" for x in p) + "\n")
            f.write("endloop\nendfacet\n")
        f.write(f"endsolid {name}\n")


def main():
    directory = Path(__file__).parent / "models"
    directory.mkdir(exist_ok=True)
    cfg = RobotConfig()
    specs = {}
    for role, name in enumerate(["atacante", "ala", "goleiro"]):
        vertices, faces = build_3d_mesh_for_role(role, cfg)
        export_obj(directory / f"chassis_{name}.obj", vertices, faces, name)
        export_stl_ascii(directory / f"chassis_{name}.stl", vertices, faces, name)
        specs[name] = dict(
            model=cfg.model,
            outline_cm=chassis_outline(role, cfg),
            tyre_footprints_cm=wheel_footprints(cfg),
            plate_thickness_cm=0.3,
            assumed_assembled_mass_g=cfg.mass,
            estimated_assembled_inertia_g_cm2=estimated_inertia(cfg),
            provenance=cfg.provenance,
            notes="Same stock body for every role. Photo-derived flat plate; thickness assumed. Not fabrication CAD.",
        )
    (directory / "chassis_specs.json").write_text(json.dumps(specs, indent=2) + "\n", encoding="utf8")
    print("Exported identical stock-reference plates for all roles; photo estimates, not fabrication CAD.")


if __name__ == "__main__":
    main()
