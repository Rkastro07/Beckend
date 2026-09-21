from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from plantatobim import local_area_estimator as estimator


def test_printed_pdf_scale_uses_render_resolution(monkeypatch, tmp_path):
    source = tmp_path / "plan.pdf"
    image = tmp_path / "page.png"
    source.write_bytes(b"pdf")
    image.write_bytes(b"png")
    (tmp_path / "pdf_render.json").write_text(
        json.dumps({"render_scale": 2.0833333333}), encoding="utf-8"
    )
    monkeypatch.setattr(estimator, "first_page_text", lambda path: "ESCALA 1/50")
    result = estimator.printed_scale_m_per_px(source, image)
    assert result is not None
    assert result["source"] == "printed-scale"
    assert result["meters_per_pixel"] == pytest.approx(0.00846667, rel=1e-4)


def test_multiple_printed_scales_are_ambiguous(monkeypatch, tmp_path):
    source = tmp_path / "plan.pdf"
    image = tmp_path / "page.png"
    source.write_bytes(b"pdf")
    image.write_bytes(b"png")
    (tmp_path / "pdf_render.json").write_text(
        json.dumps({"render_scale": 2}), encoding="utf-8"
    )
    monkeypatch.setattr(estimator, "first_page_text", lambda path: "1:50 detalhe 1:20")
    assert estimator.printed_scale_m_per_px(source, image) is None


def test_door_wall_consensus_stabilizes_scale():
    openings = []
    for index in range(12):
        width = 104 + (index % 3) - 1
        openings.append({"type": "door", "start_px": [0, 0], "end_px": [width, 0]})
    result = {
        "detection_scale": {"detection_pixel_m": 0.008},
        "openings": openings,
    }
    scale = estimator.door_wall_scale(result)
    assert scale is not None
    assert scale["source"] == "door-wall-consensus"
    assert scale["door_candidates"] == 12
    assert scale["meters_per_pixel"] == pytest.approx(0.00849, rel=0.01)


def test_footprint_closes_a_rectangular_plan(tmp_path):
    mask = np.zeros((220, 300), dtype=np.uint8)
    cv2.rectangle(mask, (40, 30), (260, 190), 255, thickness=9)
    path = tmp_path / "yolo_wall_mask.png"
    assert cv2.imwrite(str(path), mask)
    result = estimator.footprint_pixels(path, 9)
    assert result["area_px"] > 220 * 160
    assert result["fill_ratio"] > 0.50
