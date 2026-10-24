from pathlib import Path
import cv2
import numpy as np

from vision_test import VisionInspector, generate_aruco_markers


def test_generate_aruco_markers(tmp_path):
    files = generate_aruco_markers(output_dir=tmp_path, marker_size_px=100)
    assert len(files) == 7  # 6 individual markers + 1 sheet
    for f in files:
        assert Path(f).exists()
        img = cv2.imread(str(f))
        assert img is not None and img.shape[0] > 0 and img.shape[1] > 0


def test_vision_inspector_uncalibrated_and_calibrated():
    inspector = VisionInspector()
    assert not inspector.homography

    # Process uncalibrated frame with an ArUco tag
    frame = np.full((600, 800, 3), 200, np.uint8)
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag0 = cv2.aruco.generateImageMarker(dict_4x4, 0, 60)
    frame[100:160, 200:260] = cv2.cvtColor(tag0, cv2.COLOR_GRAY2BGR)

    vis, mask, diag = inspector.process_frame(frame)
    assert not diag["calibrated"]
    assert len(diag["raw_markers"]) == 1
    assert diag["raw_markers"][0]["id"] == 0

    # Calibrate corners
    corners = [[50, 50], [750, 50], [750, 550], [50, 550]]
    inspector.set_corners(corners)
    assert inspector.homography is not None
    assert inspector.tracker is not None

    vis2, mask2, diag2 = inspector.process_frame(frame)
    assert diag2["calibrated"]
    assert 0 in diag2["robots_detected"]
    assert diag2["robots_detected"][0]["team"] == "Azul"


def test_vision_inspector_puck_sampling():
    inspector = VisionInspector()
    sample = np.full((100, 100, 3), (10, 20, 25), np.uint8)  # Dark
    inspector.sample_puck_from_pixel(sample, 50, 50)
    assert inspector.puck_hsv_upper[2] < 100

    sample_orange = np.full((100, 100, 3), (20, 100, 240), np.uint8)  # Orange
    inspector.sample_puck_from_pixel(sample_orange, 50, 50)
    assert inspector.puck_hsv_lower[0] <= 15
    assert inspector.puck_hsv_upper[0] >= 10


def test_vision_inspector_save_and_load_config(tmp_path):
    cfg_file = tmp_path / "camera_test.json"
    inspector = VisionInspector()
    corners = [[10, 10], [500, 10], [500, 300], [10, 300]]
    inspector.set_corners(corners)
    inspector.save_config(cfg_file)

    assert cfg_file.exists()
    inspector2 = VisionInspector(config_path=cfg_file)
    assert inspector2.floor_corners_px == corners
    assert inspector2.homography is not None
