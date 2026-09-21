from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import plantatobim.pre_wall_opening_import as pre_wall_module
from plantatobim.pre_wall_opening_import import pre_wall_result_to_editor_model


def _result(canvas_width_m: float) -> dict:
    return {
        "canvas_width_m": canvas_width_m,
        "detection_scale": {
            "mode": "visual-auto-wall-thickness",
            "detection_pixel_m": 0.02,
            "detection_canvas_extent_m": 2.0,
        },
        "wall_geometry_source": "hybrid-2d-yolo-fusion",
        "walls": [
            {
                "orientation": "horizontal",
                "fixed": 30,
                "start": 10,
                "end": 90,
                "thickness": 10,
            }
        ],
        "openings": [
            {
                "type": "door",
                "orientation": "horizontal",
                "start_px": [35, 30],
                "end_px": [75, 30],
                "classification_width_m": 0.8,
                "wall_index": 0,
                "confidence": 0.9,
                "classification": "yolo-vote",
            }
        ],
        "raw_detection_count": 1,
        "consensus_candidate_count": 1,
        "raster_wall_count": 1,
        "scales": [640, 960, 1280],
    }


def _room_ocr_without_metric_evidence(fallback_pixel_m: float) -> dict:
    return {
        "rooms": [],
        "scale": {
            "applied": False,
            "pixel_m": fallback_pixel_m,
            "canvas_width_m": fallback_pixel_m * 100,
            "confidence": 0.0,
            "candidate_count": 0,
            "inlier_count": 0,
        },
        "room_detection": {"room_count": 0},
        "ocr": {
            "status": "ok",
            "line_count": 0,
            "matched_line_count": 0,
        },
    }


def test_visual_object_sizes_do_not_expand_with_user_canvas(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crop_path = tmp_path / "crop.png"
    assert cv2.imwrite(str(crop_path), np.full((100, 100, 3), 255, dtype=np.uint8))
    monkeypatch.setattr(
        pre_wall_module,
        "analyze_room_ocr",
        lambda _path, _axes, _shape, *, fallback_pixel_m: (
            _room_ocr_without_metric_evidence(fallback_pixel_m)
        ),
    )

    compact = pre_wall_result_to_editor_model(
        _result(12.0), crop_path, source_name="plan.png"
    )
    large = pre_wall_result_to_editor_model(
        _result(20.0), crop_path, source_name="plan.png"
    )

    assert compact["paredes"][0]["espessura"] == 0.2
    assert large["paredes"][0]["espessura"] == 0.2
    assert compact["aberturas"][0]["largura"] == 0.8
    assert large["aberturas"][0]["largura"] == 0.8
    assert compact["paredes"][0]["bx"] < large["paredes"][0]["bx"]
    assert compact["source"]["object_size_source"] == "visual-wall-thickness-calibration"


def test_confident_room_ocr_replaces_canvas_scale_and_populates_overlay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crop_path = tmp_path / "crop.png"
    assert cv2.imwrite(str(crop_path), np.full((100, 100, 3), 255, dtype=np.uint8))

    def confident_room_ocr(_path, _axes, _shape, *, fallback_pixel_m):
        return {
            "rooms": [{
                "id": "ROOM-OCR-001",
                "points_px": [[10, 10], [90, 10], [90, 90], [10, 90]],
                "area_px": 6400,
                "category_id": 3,
                "label": "Quarto",
                "ocr_lines": [{"text": "QUARTO 2,40 x 2,40 m"}],
                "dimension_pairs": [{
                    "values_m": [2.4, 2.4],
                    "text": "2,40 x 2,40 m",
                    "line_text": "QUARTO 2,40 x 2,40 m",
                    "bbox_px": {"xmin": 35, "ymin": 45, "xmax": 65, "ymax": 55},
                    "position_px": [50, 50],
                }],
                "areas": [{"value_m2": 5.76}],
            }],
            "scale": {
                "applied": True,
                "pixel_m": 0.03,
                "canvas_width_m": 3.0,
                "confidence": 0.91,
                "candidate_count": 2,
                "inlier_count": 2,
            },
            "room_detection": {"room_count": 1},
            "ocr": {
                "status": "ok",
                "line_count": 1,
                "matched_line_count": 1,
            },
        }

    monkeypatch.setattr(pre_wall_module, "analyze_room_ocr", confident_room_ocr)

    model = pre_wall_result_to_editor_model(
        _result(20.0), crop_path, source_name="plan.png"
    )

    assert model["bbox"]["xmax"] == 3.0
    assert model["source"]["scale_source"] == "room-ocr-consensus"
    assert model["paredes"][0]["espessura"] == 0.3
    assert model["aberturas"][0]["largura"] == 1.2
    assert model["reference"]["rooms"][0]["label"] == "Quarto"
    assert len(model["reference"]["dimensions"]) == 2
    assert model["raster_2d"]["room_scale_applied"] is True
