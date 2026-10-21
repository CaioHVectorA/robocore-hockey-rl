from collections import Counter
import numpy as np
import pytest
from calibrate import fit_puck_trial
from cad.fusion_chassis_generator import build_3d_mesh_for_role


def test_sliding_fit_recovers_known_friction_from_independent_trial():
    t = np.linspace(0, 0.5, 50)
    x = 20 + 100 * t - 0.5 * 0.12 * 981 * t * t
    result = fit_puck_trial(t, x, np.zeros_like(x))
    assert result["friction_mu"] == pytest.approx(0.12, abs=1e-8)
    assert result["rmse_cm"] < 1e-8


def test_calibration_rejects_nonincreasing_timestamps():
    with pytest.raises(ValueError):
        fit_puck_trial(np.zeros(10), np.zeros(10), np.zeros(10))


@pytest.mark.parametrize("role", [0, 1, 2])
def test_concave_plate_is_watertight_and_has_no_degenerate_triangles(role):
    vertices, faces = build_3d_mesh_for_role(role)
    edges = Counter()
    for face in faces:
        points = np.array([vertices[i] for i in face])
        assert np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) > 1e-8
        for a, b in zip(face, face[1:] + face[:1]):
            edges[tuple(sorted([a, b]))] += 1
    assert set(edges.values()) == {2}


def test_cad_does_not_invent_different_stock_bodies_for_tactical_roles():
    assert build_3d_mesh_for_role(0) == build_3d_mesh_for_role(1) == build_3d_mesh_for_role(2)
