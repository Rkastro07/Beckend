from __future__ import annotations

from pathlib import Path

import pytest

from plantatobim.raster_room_ocr import analyze_room_ocr, extract_room_regions


def _two_square_rooms() -> list[dict]:
    return [
        {"orientation": "horizontal", "fixed": 10, "start": 10, "end": 210, "thickness": 4},
        {"orientation": "horizontal", "fixed": 110, "start": 10, "end": 210, "thickness": 4},
        {"orientation": "vertical", "fixed": 10, "start": 10, "end": 110, "thickness": 4},
        {"orientation": "vertical", "fixed": 110, "start": 10, "end": 110, "thickness": 4},
        {"orientation": "vertical", "fixed": 210, "start": 10, "end": 110, "thickness": 4},
    ]


def _line(text: str, x: float, y: float, width: float = 60, height: float = 9) -> dict:
    return {"text": text, "x": x, "y": y, "width": width, "height": height}


def test_wall_axes_create_two_enclosed_room_regions() -> None:
    rooms, diagnostic = extract_room_regions(_two_square_rooms(), (130, 230, 3))

    assert len(rooms) == 2
    assert diagnostic["room_count"] == 2
    assert all(room["area_px"] > 8000 for room in rooms)


def test_room_ocr_labels_rooms_and_calibrates_from_dimensions_and_area(
    tmp_path: Path,
) -> None:
    def fake_ocr(_path):
        return {
            "engine": "fake-room-ocr",
            "language": "pt-BR",
            "width": 230,
            "height": 130,
            "lines": [
                _line("SALA", 30, 35),
                _line("3,00 x 3,00 m", 30, 52),
                _line("9,00 m²", 30, 69),
                _line("QUARTO", 135, 35),
                _line("3,00 x 3,00 m", 135, 52),
                _line("9,00 m²", 135, 69),
            ],
        }

    analysis = analyze_room_ocr(
        tmp_path / "not-read-by-fake.png",
        _two_square_rooms(),
        (130, 230, 3),
        fallback_pixel_m=0.08,
        runner=fake_ocr,
    )

    assert [room["label"] for room in analysis["rooms"]] == ["Sala", "Quarto"]
    assert analysis["ocr"]["matched_line_count"] == 6
    assert analysis["scale"]["applied"] is True
    assert analysis["scale"]["confidence"] >= 0.8
    assert analysis["scale"]["pixel_m"] == pytest.approx(0.031, abs=0.003)
    assert analysis["scale"]["canvas_width_m"] == pytest.approx(7.1, abs=0.7)


def test_single_declared_area_is_not_enough_to_override_user_scale(tmp_path: Path) -> None:
    def fake_ocr(_path):
        return {
            "engine": "fake-room-ocr",
            "width": 230,
            "height": 130,
            "lines": [
                _line("SALA", 30, 35),
                _line("9,00 m²", 30, 55),
            ],
        }

    analysis = analyze_room_ocr(
        tmp_path / "not-read-by-fake.png",
        _two_square_rooms(),
        (130, 230, 3),
        fallback_pixel_m=0.08,
        runner=fake_ocr,
    )

    assert analysis["scale"]["candidate_count"] == 1
    assert analysis["scale"]["applied"] is False
    assert analysis["scale"]["pixel_m"] == 0.08


def test_furniture_dimensions_are_not_used_as_room_scale(tmp_path: Path) -> None:
    def fake_ocr(_path):
        return {
            "engine": "fake-room-ocr",
            "width": 230,
            "height": 130,
            "lines": [
                _line("QUARTO", 30, 35),
                _line("CAMA 1,38 x 1,88 m", 30, 55),
            ],
        }

    analysis = analyze_room_ocr(
        tmp_path / "not-read-by-fake.png",
        _two_square_rooms(),
        (130, 230, 3),
        fallback_pixel_m=0.08,
        runner=fake_ocr,
    )

    assert analysis["scale"]["candidate_count"] == 0
    assert analysis["scale"]["applied"] is False
