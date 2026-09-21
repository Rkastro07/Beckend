from __future__ import annotations

import cv2
import numpy as np

from plantatobim.raster_2d_import import (
    build_canonical_wall_hosts_2d,
    classify_structural_regions_2d,
    detect_doors_2d,
    detect_slab_contour_2d,
    detect_wall_regions_2d,
    detect_windows_2d,
    raster_2d_image_to_editor_model,
)


def _axis(orientation: str, fixed: float, start: float, end: float, thickness: float = 8.0):
    return {
        "orientation": orientation,
        "fixed": fixed,
        "start": start,
        "end": end,
        "thickness": thickness,
        "confidence": 0.95,
        "element_type": "wall",
    }


def test_2d_morphology_extracts_filled_horizontal_and_vertical_walls():
    image = np.full((220, 220, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (38, 20), (42, 200), (100, 100, 100), -1)
    cv2.rectangle(image, (38, 108), (195, 112), (100, 100, 100), -1)

    walls, _, diagnostic = detect_wall_regions_2d(image, canvas_width_m=5.5)

    assert diagnostic["wall_count"] >= 2
    assert any(item["orientation"] == "vertical" for item in walls)
    assert any(item["orientation"] == "horizontal" for item in walls)


def test_2d_chroma_finds_an_elongated_window_component():
    image = np.full((220, 220, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (98, 40), (107, 145), (253, 240, 241), -1)

    windows, _ = detect_windows_2d(image, canvas_width_m=5.5)

    assert len(windows) == 1
    assert windows[0]["orientation"] == "vertical"


def test_2d_arc_and_leaf_between_wall_parts_propose_a_door():
    image = np.full((220, 220, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (98, 15), (102, 80), (100, 100, 100), -1)
    cv2.rectangle(image, (98, 130), (102, 205), (100, 100, 100), -1)
    cv2.line(image, (100, 80), (50, 80), (200, 200, 200), 2, cv2.LINE_AA)
    cv2.ellipse(image, (100, 80), (50, 50), 0, 90, 180, (200, 200, 200), 2, cv2.LINE_AA)

    walls, thick, _ = detect_wall_regions_2d(image, canvas_width_m=5.5)
    doors, _, _ = detect_doors_2d(
        image,
        thick,
        walls,
        canvas_width_m=5.5,
    )

    assert doors
    assert doors[0]["type"] == "door"
    assert doors[0]["orientation"] == "vertical"


def test_2d_structural_classifier_separates_thick_wall_and_compact_column():
    image = np.full((260, 260, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (18, 38), (235, 44), (100, 100, 100), -1)
    cv2.rectangle(image, (25, 105), (225, 121), (100, 100, 100), -1)
    cv2.rectangle(image, (75, 175), (105, 210), (100, 100, 100), -1)

    axes, thick, _ = detect_wall_regions_2d(image, canvas_width_m=5.2)
    walls, columns, diagnostic = classify_structural_regions_2d(
        axes,
        thick,
        canvas_width_m=5.2,
    )

    assert any(item["element_type"] == "wall" for item in walls)
    assert any(item["element_type"] == "structural-wall" for item in walls)
    assert columns
    assert diagnostic["column_count"] >= 1
    assert diagnostic["structural_wall_count"] >= 1


def test_2d_editor_model_emits_ifc_column_metadata(tmp_path):
    image = np.full((260, 260, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (18, 38), (235, 44), (100, 100, 100), -1)
    cv2.rectangle(image, (75, 175), (105, 210), (100, 100, 100), -1)
    image_path = tmp_path / "planta_com_pilar.png"
    assert cv2.imwrite(str(image_path), image)

    model = raster_2d_image_to_editor_model(image_path, canvas_width_m=5.2)
    columns = [item for item in model["paredes"] if item.get("tipo") == "column"]

    assert columns
    assert all(item["ifc_class"] == "IfcColumn" for item in columns)
    assert all(item["origem"] == "raster-2d-column" for item in columns)


def test_2d_slab_envelope_preserves_l_shaped_reentrant_corner():
    axes = [
        _axis("horizontal", 20, 20, 200),
        _axis("vertical", 200, 20, 90),
        _axis("horizontal", 90, 110, 200),
        _axis("vertical", 110, 90, 200),
        _axis("horizontal", 200, 20, 110),
        _axis("vertical", 20, 20, 200),
    ]

    contour, diagnostic = detect_slab_contour_2d(
        axes,
        (220, 220, 3),
        canvas_width_m=5.5,
    )
    contour_cv = np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))

    assert diagnostic["slab_detected"] is True
    assert diagnostic["slab_method"] == "wall-envelope"
    assert len(contour) >= 6
    assert cv2.pointPolygonTest(contour_cv, (60, 160), False) > 0
    assert cv2.pointPolygonTest(contour_cv, (160, 160), False) < 0


def test_2d_editor_model_activates_editable_floor_slab(tmp_path):
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (25, 30), (210, 205), (90, 90, 90), 9)
    image_path = tmp_path / "planta_com_laje.png"
    assert cv2.imwrite(str(image_path), image)

    model = raster_2d_image_to_editor_model(image_path, canvas_width_m=6.0)

    assert model["laje"]["piso"]["ativo"] is True
    assert model["laje"]["teto"]["ativo"] is False
    assert len(model["laje"]["contorno"]) >= 4
    assert model["raster_2d"]["slab_area_m2"] > 10.0


def test_2d_wall_first_hosts_multiple_openings_on_one_canonical_wall():
    color = np.full((220, 240, 3), 255, dtype=np.uint8)
    axes = [
        _axis("horizontal", 100, 20, 60),
        _axis("horizontal", 100, 90, 130),
        _axis("horizontal", 100, 160, 215),
    ]
    openings = [
        {
            "type": "window",
            "orientation": "horizontal",
            "start_px": [60, 100],
            "end_px": [90, 100],
            "confidence": 0.96,
        },
        {
            "type": "window",
            "orientation": "horizontal",
            "start_px": [130, 100],
            "end_px": [160, 100],
            "confidence": 0.95,
        },
    ]

    walls, hosted, diagnostic = build_canonical_wall_hosts_2d(
        axes,
        openings,
        color,
        fixed_tolerance=5,
        minimum_gap=15,
        maximum_gap=60,
    )

    assert len(walls) == 1
    assert walls[0]["start"] == 20
    assert walls[0]["end"] == 215
    assert len(hosted) == 2
    assert {item["host_axis_index"] for item in hosted} == {0}
    assert all(item["type"] == "window" for item in hosted)
    assert diagnostic["wall_segments_absorbed"] == 2
    assert diagnostic["classified_wall_gaps"] == 2


def test_2d_wall_gap_parallel_lines_propose_window_without_splitting_wall():
    color = np.full((220, 240, 3), 255, dtype=np.uint8)
    cv2.line(color, (70, 96), (110, 96), (70, 70, 70), 1, cv2.LINE_8)
    cv2.line(color, (70, 104), (110, 104), (70, 70, 70), 1, cv2.LINE_8)
    axes = [
        _axis("horizontal", 100, 20, 70, 9),
        _axis("horizontal", 100, 110, 215, 9),
    ]

    walls, hosted, diagnostic = build_canonical_wall_hosts_2d(
        axes,
        [],
        color,
        fixed_tolerance=5,
        minimum_gap=15,
        maximum_gap=60,
    )

    assert len(walls) == 1
    assert len(hosted) == 1
    assert hosted[0]["type"] == "window"
    assert hosted[0]["host_axis_index"] == 0
    assert diagnostic["wall_gap_count"] == 1
    assert diagnostic["classified_wall_gaps"] == 1
