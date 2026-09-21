"""Exporta uma revisao geometrica metrificada sobre uma planta raster.

O arquivo de revisao e deliberadamente simples: paredes e aberturas usam o
mesmo contrato editavel do front, enquanto ``calibration`` informa como
projetar as coordenadas metricas sobre a imagem retificada.  Assim o PNG de
auditoria e o IFC sao produzidos a partir da mesma geometria.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .planta_to_ifc_v1 import dict_para_modelo, gerar_ifc_do_modelo


WALL_COLOR = (0, 153, 255)  # BGR: laranja
DOOR_COLOR = (40, 190, 40)
WINDOW_COLOR = (235, 120, 35)
SLAB_COLOR = (210, 190, 80)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _point_mapper(calibration: dict[str, Any]):
    origin_x, origin_y = (float(value) for value in calibration["origin_px"])
    ppm = calibration.get("pixels_per_meter", 100.0)
    if isinstance(ppm, list):
        ppm_x, ppm_y = (float(value) for value in ppm)
    else:
        ppm_x = ppm_y = float(ppm)

    def point(x: float, y: float) -> tuple[int, int]:
        return (
            int(round(origin_x + float(x) * ppm_x)),
            int(round(origin_y - float(y) * ppm_y)),
        )

    return point, (ppm_x + ppm_y) / 2.0


def _wall_by_id(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(wall["id"]): wall for wall in model.get("paredes", [])}


def _opening_segment(
    opening: dict[str, Any], wall: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([float(wall["ax"]), float(wall["ay"])], dtype=float)
    b = np.array([float(wall["bx"]), float(wall["by"])], dtype=float)
    length = float(np.linalg.norm(b - a))
    if length <= 1e-9:
        raise ValueError(f"Parede degenerada: {wall['id']}")
    unit = (b - a) / length
    width = float(opening["largura"])
    center = float(opening["s_centro"])
    if width <= 0.0 or center - width / 2.0 < -1e-6 or center + width / 2.0 > length + 1e-6:
        raise ValueError(
            f"Abertura {opening['id']} nao cabe na parede {wall['id']} "
            f"(L={length:.3f}, centro={center:.3f}, largura={width:.3f})."
        )
    midpoint = a + unit * center
    return midpoint - unit * width / 2.0, midpoint + unit * width / 2.0


def validate_model(model: dict[str, Any]) -> None:
    walls = _wall_by_id(model)
    if not walls:
        raise ValueError("A revisao nao contem paredes.")
    for opening in model.get("aberturas", []):
        wall = walls.get(str(opening.get("parede_id")))
        if wall is None:
            raise ValueError(f"Abertura orfa: {opening.get('id')}")
        _opening_segment(opening, wall)
    contour = model.get("laje", {}).get("contorno", [])
    if len(contour) < 3:
        raise ValueError("O contorno da laje precisa de pelo menos tres pontos.")


def _draw_review(image: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    calibration = model["calibration"]
    to_px, ppm = _point_mapper(calibration)
    overlay = image.copy()

    contour = model.get("laje", {}).get("contorno", [])
    if len(contour) >= 3:
        polygon = np.array([to_px(*point) for point in contour], dtype=np.int32)
        cv2.fillPoly(overlay, [polygon], SLAB_COLOR)

    for wall in model.get("paredes", []):
        start = to_px(float(wall["ax"]), float(wall["ay"]))
        end = to_px(float(wall["bx"]), float(wall["by"]))
        thickness = max(4, int(round(float(wall["espessura"]) * ppm)))
        cv2.line(overlay, start, end, WALL_COLOR, thickness, cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, 0.42, image, 0.58, 0.0)
    walls = _wall_by_id(model)
    for opening in model.get("aberturas", []):
        wall = walls[str(opening["parede_id"])]
        a, b = _opening_segment(opening, wall)
        p1, p2 = to_px(*a), to_px(*b)
        color = DOOR_COLOR if opening["tipo"] == "door" else WINDOW_COLOR
        host_width = max(7, int(round(float(wall["espessura"]) * ppm)) + 5)
        cv2.line(blended, p1, p2, (250, 250, 250), host_width, cv2.LINE_AA)
        cv2.line(blended, p1, p2, color, max(5, host_width - 3), cv2.LINE_AA)
        cv2.circle(blended, p1, 4, color, -1, cv2.LINE_AA)
        cv2.circle(blended, p2, 4, color, -1, cv2.LINE_AA)
        center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        label = str(opening.get("label") or opening["id"])
        cv2.putText(
            blended,
            label,
            (center[0] + 5, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    for dimension in model.get("dimensions", []):
        p1 = to_px(*dimension["start"])
        p2 = to_px(*dimension["end"])
        color = (165, 60, 170)
        cv2.arrowedLine(blended, p1, p2, color, 1, cv2.LINE_AA, tipLength=0.025)
        cv2.arrowedLine(blended, p2, p1, color, 1, cv2.LINE_AA, tipLength=0.025)
        center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(
            blended,
            str(dimension["label"]),
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )

    header_height = 92
    canvas = np.full((image.shape[0] + header_height, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[header_height:] = blended
    cv2.putText(canvas, "PLAN TO BIM - REVISAO METRIFICADA", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2, cv2.LINE_AA)
    scale_note = str(model.get("scale_note", "Escala calibrada por cotas impressas"))
    cv2.putText(canvas, scale_note, (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)
    legend = [
        (WALL_COLOR, f"{len(model.get('paredes', []))} paredes-mae"),
        (DOOR_COLOR, f"{sum(item['tipo'] == 'door' for item in model.get('aberturas', []))} portas"),
        (WINDOW_COLOR, f"{sum(item['tipo'] == 'window' for item in model.get('aberturas', []))} janelas"),
    ]
    x = 18
    for color, label in legend:
        cv2.rectangle(canvas, (x, 70), (x + 18, 84), color, -1)
        cv2.putText(canvas, label, (x + 25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45, 45, 45), 1, cv2.LINE_AA)
        x += 25 + max(100, len(label) * 8)

    bar_start = to_px(0.25, 0.27)
    bar_end = to_px(2.25, 0.27)
    bar_start = (bar_start[0], bar_start[1] + header_height)
    bar_end = (bar_end[0], bar_end[1] + header_height)
    cv2.line(canvas, bar_start, bar_end, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(canvas, "2,00 m", (bar_start[0], bar_start[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def export_case(
    image_path: Path,
    review_path: Path,
    png_path: Path,
    ifc_path: Path,
    model_path: Path | None = None,
) -> dict[str, Any]:
    model = _read_json(review_path)
    validate_model(model)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Nao foi possivel ler a imagem: {image_path}")

    reviewed = _draw_review(image, model)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(png_path), reviewed):
        raise ValueError(f"Nao foi possivel gravar: {png_path}")

    internal = dict_para_modelo(model)
    ifc_path.parent.mkdir(parents=True, exist_ok=True)
    gerar_ifc_do_modelo(
        internal["paredes"],
        internal["aberturas"],
        str(ifc_path),
        {
            "altura": float(model.get("altura", 2.8)),
            "projeto": str(model.get("nome", "Plan to BIM raster")),
            "cobertura": False,
            "esquadria_detalhada": True,
        },
        laje=internal["laje"],
        spaces=internal["spaces"],
    )
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("w", encoding="utf-8") as stream:
            json.dump(model, stream, ensure_ascii=False, indent=2)

    return {
        "png": str(png_path),
        "ifc": str(ifc_path),
        "model": str(model_path) if model_path else None,
        "walls": len(model.get("paredes", [])),
        "doors": sum(item["tipo"] == "door" for item in model.get("aberturas", [])),
        "windows": sum(item["tipo"] == "window" for item in model.get("aberturas", [])),
        "slab_vertices": len(model.get("laje", {}).get("contorno", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("png", type=Path)
    parser.add_argument("ifc", type=Path)
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_case(args.image, args.review, args.png, args.ifc, args.model_output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
