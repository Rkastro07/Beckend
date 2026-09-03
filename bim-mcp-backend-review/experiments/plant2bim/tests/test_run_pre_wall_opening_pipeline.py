from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_pre_wall_opening_pipeline import (  # noqa: E402
    collapse_parallel_wall_faces,
    draw_wall_mask,
    estimate_visual_detection_scale,
    filter_structural_wall_network,
    fuse_wall_sets,
    walls_from_segmentation_mask,
)


def test_structural_filter_rejects_detached_thin_information_boxes() -> None:
    building = [
        {"orientation": "horizontal", "fixed": 20, "start": 20, "end": 180, "thickness": 8},
        {"orientation": "horizontal", "fixed": 120, "start": 20, "end": 180, "thickness": 8},
        {"orientation": "vertical", "fixed": 20, "start": 20, "end": 120, "thickness": 8},
        {"orientation": "vertical", "fixed": 100, "start": 20, "end": 120, "thickness": 8},
        {"orientation": "vertical", "fixed": 180, "start": 20, "end": 120, "thickness": 8},
    ]
    information_box = [
        {"orientation": "horizontal", "fixed": 180, "start": 210, "end": 290, "thickness": 2},
        {"orientation": "horizontal", "fixed": 220, "start": 210, "end": 290, "thickness": 2},
        {"orientation": "vertical", "fixed": 210, "start": 180, "end": 220, "thickness": 2},
        {"orientation": "vertical", "fixed": 290, "start": 180, "end": 220, "thickness": 2},
    ]
    page_frame = [
        {"orientation": "horizontal", "fixed": 2, "start": 2, "end": 318, "thickness": 1},
        {"orientation": "horizontal", "fixed": 238, "start": 2, "end": 318, "thickness": 1},
        {"orientation": "vertical", "fixed": 2, "start": 2, "end": 238, "thickness": 1},
        {"orientation": "vertical", "fixed": 318, "start": 2, "end": 238, "thickness": 1},
    ]
    opening = {
        "orientation": "horizontal",
        "box_crop_px": [70, 14, 105, 26],
    }

    filtered, diagnostic = filter_structural_wall_network(
        [*building, *information_box, *page_frame],
        [opening],
        (240, 320, 3),
    )

    assert filtered == building
    assert diagnostic["component_count"] == 3
    assert diagnostic["kept_component_count"] == 1
    assert diagnostic["rejected_wall_count"] == 8


def test_parallel_outline_faces_become_one_wall_with_measured_thickness() -> None:
    walls = [
        {
            "orientation": "horizontal",
            "fixed": 20.0,
            "start": 10.0,
            "end": 110.0,
            "thickness": 2.0,
        },
        {
            "orientation": "horizontal",
            "fixed": 28.0,
            "start": 12.0,
            "end": 108.0,
            "thickness": 2.0,
        },
        {
            "orientation": "vertical",
            "fixed": 150.0,
            "start": 20.0,
            "end": 100.0,
            "thickness": 2.0,
        },
    ]

    collapsed, diagnostic = collapse_parallel_wall_faces(walls, (200, 240, 3))

    horizontal = [wall for wall in collapsed if wall["orientation"] == "horizontal"]
    vertical = [wall for wall in collapsed if wall["orientation"] == "vertical"]
    assert len(horizontal) == 1
    assert horizontal[0]["fixed"] == 24.0
    assert horizontal[0]["thickness"] == 10.0
    assert len(vertical) == 1
    assert diagnostic == {
        "input_count": 3,
        "paired_face_count": 1,
        "output_count": 2,
    }


def test_wall_segmentation_mask_becomes_editable_axes() -> None:
    mask = np.zeros((240, 360), dtype=np.uint8)
    mask[40:46, 30:250] = 255
    mask[90:220, 285:292] = 255

    walls, diagnostic = walls_from_segmentation_mask(mask, canvas_width_m=18.0)

    assert any(wall["orientation"] == "horizontal" for wall in walls)
    assert any(wall["orientation"] == "vertical" for wall in walls)
    assert diagnostic["wall_count"] >= 2


def test_draw_wall_mask_preserves_shape() -> None:
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[20:30, 10:100] = 255

    preview = draw_wall_mask(image, mask)

    assert preview.shape == image.shape
    assert not np.array_equal(preview[25, 50], image[25, 50])
    assert np.array_equal(preview[5, 5], image[5, 5])


def test_visual_scale_is_derived_from_wall_pixels() -> None:
    image = np.full((240, 360, 3), 255, dtype=np.uint8)
    mask = np.zeros((240, 360), dtype=np.uint8)
    mask[35:43, 20:330] = 255
    mask[70:220, 180:188] = 255

    scale = estimate_visual_detection_scale(image, mask)

    assert scale["mode"] == "visual-auto-wall-thickness"
    assert scale["typical_wall_thickness_px"] == 8.0
    assert scale["detection_canvas_extent_m"] == 9.0


def test_fusion_keeps_connected_2d_recovery_and_rejects_isolated_symbol() -> None:
    geometry = [
        {"orientation": "horizontal", "fixed": 10, "start": 0, "end": 100, "thickness": 7},
        {"orientation": "vertical", "fixed": 50, "start": 10, "end": 100, "thickness": 7},
        {"orientation": "vertical", "fixed": 200, "start": 200, "end": 222, "thickness": 7},
    ]
    yolo = [
        {"orientation": "horizontal", "fixed": 11, "start": 0, "end": 100, "thickness": 8},
    ]

    walls, diagnostic = fuse_wall_sets(geometry, yolo)

    assert len(walls) == 2
    assert diagnostic["supported_by_both"] == 1
    assert diagnostic["geometry_only"] == 1
    assert diagnostic["rejected_isolated_geometry_only"] == 1
    assert any(wall["source"] == "2d-wall-fusion-recovery" for wall in walls)


def test_fusion_absorbs_both_2d_faces_into_one_yolo_carrier() -> None:
    geometry = [
        {"orientation": "horizontal", "fixed": 6.5, "start": 147, "end": 1070, "thickness": 8},
        {"orientation": "horizontal", "fixed": 28.75, "start": 169, "end": 593, "thickness": 8.5},
        {"orientation": "horizontal", "fixed": 28.75, "start": 670, "end": 1048, "thickness": 8.5},
    ]
    yolo = [
        {"orientation": "horizontal", "fixed": 16, "start": 166, "end": 1070, "thickness": 27},
        {"orientation": "vertical", "fixed": 608.5, "start": 31, "end": 124, "thickness": 24},
        {"orientation": "vertical", "fixed": 663.5, "start": 31, "end": 125, "thickness": 16},
    ]

    walls, diagnostic = fuse_wall_sets(
        geometry,
        yolo,
        typical_wall_thickness_px=28,
    )

    horizontal = [wall for wall in walls if wall["orientation"] == "horizontal"]
    vertical = [wall for wall in walls if wall["orientation"] == "vertical"]
    assert len(horizontal) == 1
    assert horizontal[0]["fixed"] == 16
    assert horizontal[0]["start"] == 147
    assert horizontal[0]["end"] == 1070
    assert horizontal[0]["fusion_sources"] == ["geometry", "yolo"]
    assert [wall["fixed"] for wall in vertical] == [608.5, 663.5]
    assert diagnostic["absorbed_geometry_segments"] == 3


def test_fusion_pairs_unmatched_geometry_faces_without_merging_real_yolo_walls() -> None:
    geometry = [
        {"orientation": "horizontal", "fixed": 100, "start": 0, "end": 100, "thickness": 8},
        {"orientation": "horizontal", "fixed": 122, "start": 2, "end": 98, "thickness": 8},
    ]
    yolo = [
        {"orientation": "vertical", "fixed": 50, "start": 80, "end": 160, "thickness": 28},
    ]

    walls, diagnostic = fuse_wall_sets(
        geometry,
        yolo,
        typical_wall_thickness_px=28,
    )

    horizontal = [wall for wall in walls if wall["orientation"] == "horizontal"]
    assert len(horizontal) == 1
    assert horizontal[0]["fixed"] == 111
    assert horizontal[0]["thickness"] == 30
    assert diagnostic["paired_geometry_recovery_faces"] == 1
