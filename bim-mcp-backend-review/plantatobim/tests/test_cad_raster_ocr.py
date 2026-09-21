import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_raster_ocr import (
    _ocr_cache_path,
    aligned_ocr_evidence,
    dimension_candidates_from_ocr,
    extract_raster_dimensions,
    pixel_to_cad,
    run_windows_ocr,
)


def _image_record():
    return {
        "resolved_path": "floor.png",
        "handle": "8C",
        "layer": "Linked image",
        "insert_raw": [100.0, 200.0],
        "u_pixel_raw": [2.0, 0.0],
        "v_pixel_raw": [0.0, 3.0],
        "image_size": [10.0, 20.0],
    }


def test_pixel_to_cad_uses_bottom_left_cad_origin():
    point = pixel_to_cad(
        50.0,
        100.0,
        _image_record(),
        raster_width=100.0,
        raster_height=200.0,
    )
    assert np.allclose(point, [110.0, 230.0])


def test_ocr_evidence_preserves_all_lines_and_emits_semantic_cue():
    def fake_runner(_):
        return {
            "engine": "fake-ocr",
            "language": "en-US",
            "width": 100,
            "height": 200,
            "text_angle": 0,
            "lines": [
                {
                    "text": "SECTIONAL DOOR 2180 x 4800",
                    "x": 10,
                    "y": 20,
                    "width": 40,
                    "height": 10,
                },
                {
                    "text": "GARAGE",
                    "x": 30,
                    "y": 80,
                    "width": 20,
                    "height": 10,
                },
            ],
        }

    semantic, diagnostic = aligned_ocr_evidence(
        _image_record(),
        runner=fake_runner,
    )

    assert diagnostic["status"] == "ok"
    assert diagnostic["line_count"] == 2
    assert diagnostic["semantic_cue_count"] == 1
    assert semantic[0]["source_kind"] == "raster-ocr"
    assert semantic[0]["subtype"] == "garage_door"
    assert semantic[0]["declared_width"] == 4.8
    assert semantic[0]["declared_height"] == 2.18


def test_dimension_candidates_are_aligned_to_raster2seq_square_canvas():
    result = {
        "engine": "fake-ocr",
        "language": "pt-BR",
        "width": 100,
        "height": 50,
        "lines": [
            {
                "text": "PAREDE 3,20 m",
                "x": 10,
                "y": 10,
                "width": 20,
                "height": 10,
                "words": [
                    {"text": "3,20m", "x": 10, "y": 10, "width": 20, "height": 10},
                ],
            },
            {
                "text": "320",
                "x": 40,
                "y": 10,
                "width": 10,
                "height": 10,
                "words": [
                    {"text": "320", "x": 40, "y": 10, "width": 10, "height": 10},
                ],
            },
            {
                "text": "ÁREA 12,50 m²",
                "x": 60,
                "y": 10,
                "width": 20,
                "height": 10,
                "words": [
                    {"text": "12,50", "x": 60, "y": 10, "width": 20, "height": 10},
                ],
            },
        ],
    }

    candidates = dimension_candidates_from_ocr(result, canvas_width_m=20)

    assert [item["value_m"] for item in candidates] == [3.2, 3.2]
    assert candidates[0]["bbox"] == {
        "xmin": 2.0,
        "ymin": 11.0,
        "xmax": 6.0,
        "ymax": 13.0,
    }
    assert candidates[0]["position"] == {"x": 4.0, "y": 12.0}
    assert candidates[0]["confidence"] == 0.98
    assert candidates[0]["kind"] == "linear"
    assert candidates[1]["assumption"] == "integer-cm"


def test_dimension_candidates_infer_millimeters_for_the_whole_drawing():
    result = {
        "width": 100,
        "height": 100,
        "lines": [
            {"text": "7760", "words": [{"text": "7760", "x": 5, "y": 5, "width": 10, "height": 5}]},
            {"text": "800", "words": [{"text": "800", "x": 20, "y": 5, "width": 10, "height": 5}]},
            {"text": "150", "words": [{"text": "150", "x": 35, "y": 5, "width": 10, "height": 5}]},
            {"text": "50", "words": [{"text": "50", "x": 45, "y": 5, "width": 10, "height": 5}]},
            {"text": "BED 1520 x 2030", "words": [{"text": "1520", "x": 50, "y": 5, "width": 10, "height": 5}]},
        ],
    }

    candidates = dimension_candidates_from_ocr(result, canvas_width_m=10)

    assert [item["value_m"] for item in candidates[:4]] == [7.76, 0.8, 0.15, 0.05]
    assert candidates[1]["assumption"] == "integer-mm-context"
    assert candidates[1]["kind"] == "linear"
    assert candidates[2]["kind"] == "thickness"
    assert candidates[3]["kind"] == "thickness"
    assert candidates[4]["kind"] == "object-size"


def test_extract_raster_dimensions_keeps_ocr_failure_non_fatal():
    def failing_runner(_):
        raise RuntimeError("engine unavailable")

    candidates, diagnostic = extract_raster_dimensions(
        "missing.png",
        canvas_width_m=20,
        runner=failing_runner,
    )

    assert candidates == []
    assert diagnostic["status"] == "failed"
    assert "engine unavailable" in diagnostic["error"]


def test_windows_ocr_reuses_content_addressed_cache(tmp_path, monkeypatch):
    image_path = tmp_path / "plan.png"
    image_path.write_bytes(b"fake-png")
    monkeypatch.setenv("RASTER_OCR_CACHE_ROOT", str(tmp_path / "cache"))
    cache_path = _ocr_cache_path(image_path)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        '{"engine":"fake","width":10,"height":10,"lines":[]}',
        encoding="utf-8",
    )

    result = run_windows_ocr(image_path)

    assert result["engine"] == "fake"
    assert result["cache_hit"] is True
