from __future__ import annotations

import cv2
import numpy as np

from plantatobim.raster_slices_import import detect_slice_openings, detect_slice_walls


def _nearest(axes, orientation, fixed):
    matches = [axis for axis in axes if axis["orientation"] == orientation]
    return min(matches, key=lambda axis: abs(float(axis["fixed"]) - fixed))


def test_slice_profiles_reconstruct_double_line_walls():
    ink = np.zeros((220, 220), dtype=np.uint8)
    cv2.line(ink, (40, 20), (40, 195), 1, 1)
    cv2.line(ink, (52, 20), (52, 195), 1, 1)
    cv2.line(ink, (35, 100), (190, 100), 1, 1)
    cv2.line(ink, (35, 112), (190, 112), 1, 1)

    axes, faces, diagnostic = detect_slice_walls(ink, canvas_width_m=5.5)

    vertical = _nearest(axes, "vertical", 46)
    horizontal = _nearest(axes, "horizontal", 106)
    assert abs(float(vertical["fixed"]) - 46) <= 2
    assert abs(float(horizontal["fixed"]) - 106) <= 2
    assert 10 <= float(vertical["thickness"]) <= 15
    assert len(faces) >= 4
    assert diagnostic["wall_count"] >= 2


def test_short_text_like_strokes_do_not_become_walls():
    ink = np.zeros((220, 220), dtype=np.uint8)
    cv2.line(ink, (30, 30), (30, 38), 1, 2)
    cv2.line(ink, (40, 30), (40, 38), 1, 2)

    axes, _, diagnostic = detect_slice_walls(ink, canvas_width_m=11.0)

    assert axes == []
    assert diagnostic["wall_count"] == 0


def _split_horizontal_axes():
    base = {
        "orientation": "horizontal",
        "fixed": 100.0,
        "thickness": 12.0,
        "confidence": 0.9,
        "source": "paired-faces",
    }
    return [
        {**base, "start": 20.0, "end": 80.0},
        {**base, "start": 130.0, "end": 205.0},
    ]


def test_diagonal_leaf_and_swing_classify_a_door_and_join_the_wall():
    ink = np.zeros((220, 220), dtype=np.uint8)
    cv2.line(ink, (80, 100), (80, 50), 1, 2)
    cv2.ellipse(ink, (80, 100), (50, 50), 0, -90, 0, 1, 2)

    axes, openings, diagnostic = detect_slice_openings(
        _split_horizontal_axes(),
        ink,
        canvas_width_m=5.5,
    )

    assert len(axes) == 1
    assert len(openings) == 1
    assert openings[0]["kind"] == "door"
    assert diagnostic["door_count"] == 1


def test_parallel_lines_in_a_gap_classify_a_window_and_join_the_wall():
    ink = np.zeros((220, 220), dtype=np.uint8)
    cv2.line(ink, (80, 96), (130, 96), 1, 1)
    cv2.line(ink, (80, 104), (130, 104), 1, 1)

    axes, openings, diagnostic = detect_slice_openings(
        _split_horizontal_axes(),
        ink,
        canvas_width_m=5.5,
    )

    assert len(axes) == 1
    assert len(openings) == 1
    assert openings[0]["kind"] == "window"
    assert diagnostic["window_count"] == 1


def test_empty_gap_remains_two_walls_without_inventing_an_opening():
    axes, openings, diagnostic = detect_slice_openings(
        _split_horizontal_axes(),
        np.zeros((220, 220), dtype=np.uint8),
        canvas_width_m=5.5,
    )

    assert len(axes) == 2
    assert openings == []
    assert diagnostic["classified_openings"] == 0
