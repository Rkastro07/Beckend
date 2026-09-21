"""Gera pranchas ampliadas com grade de coordenadas para revisar uma planta raster."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Não foi possível abrir {path}")
    return image


def render_sheet(
    image_path: Path,
    output_path: Path,
    *,
    rows: int,
    cols: int,
    scale: float,
    crop: tuple[int, int, int, int] | None = None,
    grid_step: int = 50,
) -> None:
    image = _read_image(image_path)
    offset_x = 0
    offset_y = 0
    if crop is not None:
        x0, y0, x1, y1 = crop
        image = image[y0:y1, x0:x1]
        offset_x = x0
        offset_y = y0
    height, width = image.shape[:2]
    panels: list[np.ndarray] = []
    for row in range(rows):
        y0 = round(row * height / rows)
        y1 = round((row + 1) * height / rows)
        row_panels: list[np.ndarray] = []
        for col in range(cols):
            local_x0 = round(col * width / cols)
            local_x1 = round((col + 1) * width / cols)
            panel = image[y0:y1, local_x0:local_x1].copy()
            x0 = local_x0 + offset_x
            x1 = local_x1 + offset_x
            absolute_y0 = y0 + offset_y
            absolute_y1 = y1 + offset_y
            if grid_step > 0:
                first_x = ((x0 + grid_step - 1) // grid_step) * grid_step
                for x in range(first_x, x1, grid_step):
                    local_x = x - x0
                    cv2.line(panel, (local_x, 0), (local_x, panel.shape[0] - 1), (220, 220, 220), 1)
                    cv2.putText(panel, str(x), (local_x + 3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
                first_y = ((absolute_y0 + grid_step - 1) // grid_step) * grid_step
                for y in range(first_y, absolute_y1, grid_step):
                    local_y = y - absolute_y0
                    cv2.line(panel, (0, local_y), (panel.shape[1] - 1, local_y), (220, 220, 220), 1)
                    cv2.putText(panel, str(y), (3, local_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
            panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            header = np.full((38, panel.shape[1], 3), 255, dtype=np.uint8)
            cv2.putText(
                header,
                f"x={x0}:{x1}  y={absolute_y0}:{absolute_y1}",
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (35, 35, 35),
                1,
                cv2.LINE_AA,
            )
            row_panels.append(np.vstack([header, panel]))
        max_height = max(item.shape[0] for item in row_panels)
        row_panels = [
            np.vstack([item, np.full((max_height - item.shape[0], item.shape[1], 3), 255, dtype=np.uint8)])
            for item in row_panels
        ]
        panels.append(np.hstack(row_panels))
    max_width = max(item.shape[1] for item in panels)
    panels = [
        np.hstack([item, np.full((item.shape[0], max_width - item.shape[1], 3), 255, dtype=np.uint8)])
        for item in panels
    ]
    sheet = np.vstack(panels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", sheet)
    if not ok:
        raise RuntimeError("Falha ao codificar a prancha")
    encoded.tofile(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--crop", help="x0,y0,x1,y1 em pixels")
    parser.add_argument("--grid-step", type=int, default=50, help="0 desliga a grade")
    args = parser.parse_args()
    crop = tuple(int(value) for value in args.crop.split(",")) if args.crop else None
    if crop is not None and len(crop) != 4:
        parser.error("--crop deve ter quatro inteiros: x0,y0,x1,y1")
    render_sheet(
        args.image,
        args.output,
        rows=args.rows,
        cols=args.cols,
        scale=args.scale,
        crop=crop,
        grid_step=max(0, args.grid_step),
    )


if __name__ == "__main__":
    main()
