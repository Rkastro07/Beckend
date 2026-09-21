"""Renderiza a saída do detector por fatias sobre a planta original."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    from .raster_slices_import import raster_slices_image_to_editor_model
except ImportError:
    from raster_slices_import import raster_slices_image_to_editor_model


WALL_COLOR = (0, 140, 255)      # laranja em BGR
DOOR_COLOR = (70, 210, 70)      # verde
WINDOW_COLOR = (240, 175, 45)   # azul


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Não foi possível abrir {path}")
    return image


def _world_to_image(
    x: float,
    y: float,
    *,
    canvas_width_m: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    canvas_size = max(image_width, image_height)
    pad_x = (canvas_size - image_width) / 2.0
    pad_y = (canvas_size - image_height) / 2.0
    px = x / canvas_width_m * canvas_size - pad_x
    py = canvas_size - y / canvas_width_m * canvas_size - pad_y
    return int(round(px)), int(round(py))


def render_annotated(
    image_path: Path,
    output_path: Path,
    *,
    canvas_width_m: float,
    calibration_path: Path | None = None,
) -> dict:
    image = _read_image(image_path)
    model = raster_slices_image_to_editor_model(
        image_path,
        canvas_width_m=canvas_width_m,
    )
    overlay = image.copy()
    height, width = image.shape[:2]
    wall_by_id = {wall["id"]: wall for wall in model["paredes"]}

    for wall in model["paredes"]:
        start = _world_to_image(
            wall["ax"], wall["ay"],
            canvas_width_m=canvas_width_m,
            image_width=width,
            image_height=height,
        )
        end = _world_to_image(
            wall["bx"], wall["by"],
            canvas_width_m=canvas_width_m,
            image_width=width,
            image_height=height,
        )
        thickness_px = max(2, int(round(wall["espessura"] / canvas_width_m * max(width, height))))
        cv2.line(overlay, start, end, WALL_COLOR, max(2, thickness_px), cv2.LINE_AA)

    calibration: list[dict] | None = None
    if calibration_path is not None:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration = list(payload.get("openings") or [])

    rendered_openings: list[dict] = []
    if calibration is not None:
        for opening in calibration:
            start = tuple(int(round(value)) for value in opening["start_px"])
            end = tuple(int(round(value)) for value in opening["end_px"])
            kind = str(opening["type"])
            color = DOOR_COLOR if kind == "door" else WINDOW_COLOR
            cv2.line(overlay, start, end, color, 8, cv2.LINE_AA)
            rendered_openings.append(opening)
    else:
        for opening in model["aberturas"]:
            wall = wall_by_id.get(opening["parede_id"])
            if wall is None:
                continue
            dx = float(wall["bx"]) - float(wall["ax"])
            dy = float(wall["by"]) - float(wall["ay"])
            length = max(1e-9, math.hypot(dx, dy))
            ux, uy = dx / length, dy / length
            center_x = float(wall["ax"]) + ux * float(opening["s_centro"])
            center_y = float(wall["ay"]) + uy * float(opening["s_centro"])
            half = float(opening["largura"]) / 2.0
            start = _world_to_image(
                center_x - ux * half, center_y - uy * half,
                canvas_width_m=canvas_width_m,
                image_width=width,
                image_height=height,
            )
            end = _world_to_image(
                center_x + ux * half, center_y + uy * half,
                canvas_width_m=canvas_width_m,
                image_width=width,
                image_height=height,
            )
            color = DOOR_COLOR if opening["tipo"] == "door" else WINDOW_COLOR
            cv2.line(overlay, start, end, color, 8, cv2.LINE_AA)
            rendered_openings.append({"type": opening["tipo"], "start_px": start, "end_px": end})

    plan = cv2.addWeighted(image, 0.58, overlay, 0.72, 0.0)
    header_height = 96
    annotated = np.full((height + header_height, width, 3), 255, dtype=np.uint8)
    annotated[header_height:, :] = plan
    cv2.putText(annotated, "PLANTA VETORIZADA - FATIAS 1D + CALIBRACAO VISUAL", (20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(annotated, (22, 51), (70, 51), WALL_COLOR, 7, cv2.LINE_AA)
    cv2.putText(annotated, "PAREDE", (82, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(annotated, (205, 51), (253, 51), DOOR_COLOR, 7, cv2.LINE_AA)
    cv2.putText(annotated, "PORTA", (265, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(annotated, (376, 51), (424, 51), WINDOW_COLOR, 7, cv2.LINE_AA)
    cv2.putText(annotated, "JANELA", (436, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 1, cv2.LINE_AA)
    doors = sum(item["type"] == "door" for item in rendered_openings)
    windows = sum(item["type"] == "window" for item in rendered_openings)
    summary = f"{len(model['paredes'])} eixos de parede | {doors} portas | {windows} janelas | escala visual: {canvas_width_m:g} m"
    cv2.putText(annotated, summary, (22, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (65, 65, 65), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", annotated)
    if not ok:
        raise RuntimeError("Falha ao codificar o PNG anotado")
    encoded.tofile(str(output_path))
    output_path.with_suffix(".json").write_text(
        json.dumps({"model": model, "visual_calibration": calibration}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--canvas-width", type=float, default=16.0)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    model = render_annotated(
        args.image,
        args.output,
        canvas_width_m=args.canvas_width,
        calibration_path=args.calibration,
    )
    print(f"{args.output}: {len(model['paredes'])} eixos de parede")


if __name__ == "__main__":
    main()
