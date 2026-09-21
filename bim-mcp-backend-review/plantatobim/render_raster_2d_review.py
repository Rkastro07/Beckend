"""Renderiza a vetorização 2D automática e uma revisão humana opcional.

O JSON de revisão usa coordenadas de pixel da imagem original:
{"openings": [{"type": "door|window", "start_px": [x,y], "end_px": [x,y],
                "confidence": 0..1, "reason": "..."}]}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .raster_2d_import import vectorize_floorplan_2d
except ImportError:
    from raster_2d_import import vectorize_floorplan_2d


WALL_COLOR = (0, 140, 255)
STRUCTURAL_WALL_COLOR = (0, 92, 190)
COLUMN_COLOR = (246, 92, 139)
DOOR_COLOR = (70, 210, 70)
WINDOW_COLOR = (240, 175, 45)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Não foi possível abrir {path}")
    return image


def _axis_points(axis: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    if axis["orientation"] == "vertical":
        return (
            (int(round(axis["fixed"])), int(round(axis["start"]))),
            (int(round(axis["fixed"])), int(round(axis["end"]))),
        )
    return (
        (int(round(axis["start"])), int(round(axis["fixed"]))),
        (int(round(axis["end"])), int(round(axis["fixed"]))),
    )


def _opening_points(opening: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        tuple(int(round(value)) for value in opening["start_px"]),
        tuple(int(round(value)) for value in opening["end_px"]),
    )


def _draw_openings(image: np.ndarray, openings: list[dict[str, Any]]) -> None:
    door_index = 0
    window_index = 0
    scale = max(0.7, min(image.shape[:2]) / 900.0)
    for opening in openings:
        kind = str(opening["type"])
        if kind == "door":
            door_index += 1
            label = f"D{door_index}"
            color = DOOR_COLOR
        else:
            window_index += 1
            label = f"W{window_index}"
            color = WINDOW_COLOR
        start, end = _opening_points(opening)
        thickness = max(5, int(round(7 * scale)))
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)
        cv2.circle(image, start, max(4, thickness // 2), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, start, max(3, thickness // 3), color, -1, cv2.LINE_AA)
        cv2.circle(image, end, max(4, thickness // 2), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, end, max(3, thickness // 3), color, -1, cv2.LINE_AA)
        middle = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.putText(
            image,
            label,
            (middle[0] + 5, middle[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52 * scale,
            (255, 255, 255),
            max(3, int(round(4 * scale))),
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (middle[0] + 5, middle[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52 * scale,
            color,
            max(1, int(round(1.5 * scale))),
            cv2.LINE_AA,
        )


def render_review(
    image_path: Path,
    output_path: Path,
    *,
    canvas_width_m: float,
    review_path: Path | None = None,
) -> dict[str, Any]:
    image = _read_image(image_path)
    automatic = vectorize_floorplan_2d(image_path, canvas_width_m=canvas_width_m)
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path else {}
    openings = list(review.get("openings", automatic["openings"]))

    layer = image.copy()
    for wall in automatic["walls"]:
        color = STRUCTURAL_WALL_COLOR if wall.get("element_type") == "structural-wall" else WALL_COLOR
        start, end = _axis_points(wall)
        cv2.line(
            layer,
            start,
            end,
            color,
            max(3, int(round(float(wall["thickness"])))),
            cv2.LINE_AA,
        )
    for column in automatic.get("columns", []):
        start, end = _axis_points(column)
        cv2.line(
            layer,
            start,
            end,
            COLUMN_COLOR,
            max(5, int(round(float(column["thickness"])))),
            cv2.LINE_AA,
        )
    overlay = cv2.addWeighted(image, 0.58, layer, 0.72, 0.0)
    _draw_openings(overlay, openings)

    header_height = 74
    canvas = np.full((overlay.shape[0] + header_height, overlay.shape[1], 3), 255, dtype=np.uint8)
    canvas[header_height:] = overlay
    title = "REVISAO VISUAL 2D" if review_path else "2D AUTOMATICO"
    subtitle = (
        f"{len(automatic['walls'])} paredes | {len(automatic.get('columns', []))} pilares | "
        f"{sum(item['type'] == 'door' for item in openings)} portas | "
        f"{sum(item['type'] == 'window' for item in openings)} janelas"
    )
    cv2.putText(canvas, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (16, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (75, 75, 75), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Falha ao codificar PNG")
    encoded.tofile(str(output_path))
    result = {
        "output": str(output_path),
        "walls": len(automatic["walls"]),
        "structural_walls": automatic["diagnostics"].get("structural_wall_count", 0),
        "columns": len(automatic.get("columns", [])),
        "doors": sum(item["type"] == "door" for item in openings),
        "windows": sum(item["type"] == "window" for item in openings),
        "automatic_doors": automatic["diagnostics"]["door_count"],
        "automatic_windows": automatic["diagnostics"]["window_count"],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps({"summary": result, "automatic": automatic, "review": review}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--canvas-width", type=float, default=16.0)
    parser.add_argument("--review-json", type=Path)
    args = parser.parse_args()
    print(json.dumps(render_review(
        args.image,
        args.output,
        canvas_width_m=args.canvas_width,
        review_path=args.review_json,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
