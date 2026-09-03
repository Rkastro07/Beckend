"""Vetorização 2D de plantas raster por morfologia, contornos e símbolos.

Este módulo é deliberadamente independente do detector por fatias 1D. Ele
trabalha com regiões 2D preenchidas para paredes, componentes cromáticos para
janelas e componentes de arco/folha para portas.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np


RASTER_2D_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class Raster2DError(RuntimeError):
    """Falha controlada do vetorizador raster 2D."""


@dataclass(frozen=True)
class _Component:
    x: int
    y: int
    width: int
    height: int
    area: int

    @property
    def x2(self) -> int:
        return self.x + self.width - 1

    @property
    def y2(self) -> int:
        return self.y + self.height - 1


def _read_color(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    color = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if color is None:
        raise Raster2DError("Não foi possível abrir a imagem da planta.")
    return color


def _merge_axis_segments(
    segments: list[dict[str, Any]],
    *,
    fixed_tolerance: float,
    maximum_gap: float,
) -> list[dict[str, Any]]:
    working = sorted(
        ({**segment} for segment in segments),
        key=lambda item: (item["orientation"], float(item["fixed"]), float(item["start"])),
    )
    changed = True
    while changed:
        changed = False
        merged: list[dict[str, Any]] = []
        consumed: set[int] = set()
        for first_index, first in enumerate(working):
            if first_index in consumed:
                continue
            current = {**first}
            for second_index in range(first_index + 1, len(working)):
                if second_index in consumed:
                    continue
                second = working[second_index]
                if current["orientation"] != second["orientation"]:
                    continue
                if abs(float(current["fixed"]) - float(second["fixed"])) > fixed_tolerance:
                    continue
                gap = max(
                    float(current["start"]), float(second["start"])
                ) - min(float(current["end"]), float(second["end"]))
                if gap > maximum_gap:
                    continue
                first_length = max(1.0, float(current["end"]) - float(current["start"]))
                second_length = max(1.0, float(second["end"]) - float(second["start"]))
                total = first_length + second_length
                current = {
                    "orientation": current["orientation"],
                    "fixed": (
                        float(current["fixed"]) * first_length
                        + float(second["fixed"]) * second_length
                    ) / total,
                    "start": min(float(current["start"]), float(second["start"])),
                    "end": max(float(current["end"]), float(second["end"])),
                    "thickness": (
                        float(current["thickness"]) * first_length
                        + float(second["thickness"]) * second_length
                    ) / total,
                    "confidence": max(float(current["confidence"]), float(second["confidence"])),
                    "source": "2d-morphology-merged",
                }
                consumed.add(second_index)
                changed = True
            consumed.add(first_index)
            merged.append(current)
        working = merged
    return working


def detect_wall_regions_2d(
    color: np.ndarray,
    *,
    canvas_width_m: float,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Extrai eixos de parede a partir de massas 2D escuras e espessas."""
    if color.ndim != 3 or color.shape[2] != 3:
        raise Raster2DError("A imagem 2D precisa estar em BGR.")
    height, width = color.shape[:2]
    canvas_size = max(height, width)
    pixel_m = canvas_width_m / float(canvas_size)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    # O limiar rígido retém as massas cinza-escuras e elimina os símbolos
    # antialiasados mais claros. A abertura 3x3 remove texto e linhas finas.
    strict = np.asarray(gray <= 138, dtype=np.uint8)
    thick = cv2.morphologyEx(strict, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    kernel_length = max(9, int(round(canvas_size * 0.016)))
    horizontal_raw = cv2.morphologyEx(
        thick,
        cv2.MORPH_OPEN,
        np.ones((3, kernel_length), np.uint8),
    )
    vertical_raw = cv2.morphologyEx(
        thick,
        cv2.MORPH_OPEN,
        np.ones((kernel_length, 3), np.uint8),
    )
    # Interseções conectariam toda a malha em um único componente. Removê-las
    # temporariamente permite extrair cada faixa 2D; depois os pequenos cortes
    # são novamente costurados pelo alinhamento geométrico.
    horizontal = horizontal_raw.copy()
    vertical = vertical_raw.copy()
    # O kernel curto de extração também cabe transversalmente dentro de uma
    # parede grossa e a confundiria com uma interseção. Para cortar apenas
    # cruzamentos reais, a confirmação perpendicular exige pelo menos 55 cm.
    intersection_kernel_length = max(kernel_length + 2, int(round(0.55 / pixel_m)))
    horizontal_cross = cv2.morphologyEx(
        thick,
        cv2.MORPH_OPEN,
        np.ones((3, intersection_kernel_length), np.uint8),
    )
    vertical_cross = cv2.morphologyEx(
        thick,
        cv2.MORPH_OPEN,
        np.ones((intersection_kernel_length, 3), np.uint8),
    )
    intersection_padding = np.ones((3, 3), np.uint8)
    horizontal[cv2.dilate(vertical_cross, intersection_padding) > 0] = 0
    vertical[cv2.dilate(horizontal_cross, intersection_padding) > 0] = 0
    minimum_length = max(18.0, 0.36 / pixel_m)
    maximum_thickness = max(18.0, 0.38 / pixel_m)

    segments: list[dict[str, Any]] = []
    for orientation, mask in (("horizontal", horizontal), ("vertical", vertical)):
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for index in range(1, count):
            x, y, box_width, box_height, area = map(int, stats[index])
            length = float(box_width if orientation == "horizontal" else box_height)
            thickness = float(box_height if orientation == "horizontal" else box_width)
            if length < minimum_length or not 3.0 <= thickness <= maximum_thickness:
                continue
            elongation = length / max(1.0, thickness)
            if elongation < 1.65:
                continue
            fill = area / max(1.0, box_width * box_height)
            if fill < 0.43:
                continue
            if orientation == "horizontal":
                fixed = y + (box_height - 1) / 2.0
                start, end = float(x), float(x + box_width - 1)
            else:
                fixed = x + (box_width - 1) / 2.0
                start, end = float(y), float(y + box_height - 1)
            segments.append({
                "orientation": orientation,
                "fixed": fixed,
                "start": start,
                "end": end,
                "thickness": thickness,
                "confidence": min(0.97, 0.64 + 0.20 * fill + 0.08 * min(1.0, elongation / 8.0)),
                "source": "2d-morphology",
            })

    axes = _merge_axis_segments(
        segments,
        fixed_tolerance=max(2.0, 0.055 / pixel_m),
        maximum_gap=max(5.0, 0.30 / pixel_m),
    )
    axes.sort(
        key=lambda item: (
            item["orientation"],
            round(float(item["fixed"]), 3),
            float(item["start"]),
        )
    )
    diagnostic = {
        "wall_count": len(axes),
        "horizontal_walls": sum(item["orientation"] == "horizontal" for item in axes),
        "vertical_walls": sum(item["orientation"] == "vertical" for item in axes),
        "strict_wall_pixels": int(strict.sum()),
        "thick_wall_pixels": int(thick.sum()),
        "kernel_length_px": kernel_length,
        "intersection_kernel_length_px": intersection_kernel_length,
    }
    return axes, thick, diagnostic


def _axis_bbox(axis: dict[str, Any]) -> tuple[float, float, float, float]:
    half = float(axis["thickness"]) / 2.0
    if axis["orientation"] == "vertical":
        return (
            float(axis["fixed"]) - half,
            float(axis["start"]),
            float(axis["fixed"]) + half,
            float(axis["end"]),
        )
    return (
        float(axis["start"]),
        float(axis["fixed"]) - half,
        float(axis["end"]),
        float(axis["fixed"]) + half,
    )


def detect_slab_contour_2d(
    wall_axes: list[dict[str, Any]],
    image_shape: tuple[int, ...],
    *,
    canvas_width_m: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Reconstrói a projeção da laje a partir da envoltória das paredes.

    A envoltória raster preserva reentrâncias (plantas em L/U), ao contrário
    do hull convexo usado apenas como fallback quando a malha externa não
    fecha. O resultado permanece editável no front antes de virar IfcSlab.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    canvas_size = max(height, width)
    pixel_m = canvas_width_m / float(canvas_size)
    wall_mask = np.zeros((height, width), dtype=np.uint8)

    thicknesses_m = [
        float(axis["thickness"]) * pixel_m
        for axis in wall_axes
        if float(axis["end"]) > float(axis["start"])
    ]
    typical_thickness_m = (
        float(np.percentile(thicknesses_m, 50)) if thicknesses_m else 0.15
    )
    minimum_envelope_thickness_m = max(0.075, typical_thickness_m * 0.72)
    envelope_axes = [
        axis
        for axis in wall_axes
        if float(axis["thickness"]) * pixel_m >= minimum_envelope_thickness_m
        and (float(axis["end"]) - float(axis["start"])) * pixel_m >= 0.35
    ]
    if len(envelope_axes) < 3:
        envelope_axes = wall_axes

    for axis in envelope_axes:
        fixed = int(round(float(axis["fixed"])))
        start = int(round(float(axis["start"])))
        end = int(round(float(axis["end"])))
        thickness = max(3, int(round(float(axis["thickness"]))))
        if axis["orientation"] == "vertical":
            first, second = (fixed, start), (fixed, end)
        else:
            first, second = (start, fixed), (end, fixed)
        cv2.line(wall_mask, first, second, 255, thickness, cv2.LINE_8)

    wall_pixels = int(np.count_nonzero(wall_mask))
    if wall_pixels == 0:
        return [], {
            "slab_detected": False,
            "slab_method": "none",
            "slab_vertex_count": 0,
            "slab_area_m2": 0.0,
            "slab_enclosure_ratio": 0.0,
            "slab_confidence": 0.0,
        }

    # Fecha apenas folgas de encontro entre eixos. Portas/janelas reconhecidas
    # já foram costuradas em _build_opening_hosts antes desta etapa.
    join_size = max(3, int(round(0.18 / pixel_m)))
    if join_size % 2 == 0:
        join_size += 1
    joined = cv2.morphologyEx(
        wall_mask,
        cv2.MORPH_CLOSE,
        np.ones((join_size, join_size), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if len(contour) >= 3]
    best = max(contours, key=cv2.contourArea) if contours else None
    area_px = float(cv2.contourArea(best)) if best is not None else 0.0
    enclosure_ratio = area_px / max(1.0, float(wall_pixels))
    method = "wall-envelope"

    # Uma rede aberta produz um contorno estreito ao redor das linhas. Nessa
    # situação, um hull dos retângulos de parede é uma proposta conservadora
    # e explícita para o usuário ajustar, em vez de fingir uma concavidade.
    if best is None or enclosure_ratio < 1.6:
        footprint_points: list[list[float]] = []
        for axis in envelope_axes:
            x0, y0, x1, y1 = _axis_bbox(axis)
            footprint_points.extend([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        if len(envelope_axes) < 3 or len(footprint_points) < 3:
            return [], {
                "slab_detected": False,
                "slab_method": "insufficient-envelope",
                "slab_vertex_count": 0,
                "slab_area_m2": 0.0,
                "slab_enclosure_ratio": round(enclosure_ratio, 4),
                "slab_confidence": 0.0,
            }
        best = cv2.convexHull(np.asarray(footprint_points, dtype=np.float32))
        area_px = float(cv2.contourArea(best))
        enclosure_ratio = area_px / max(1.0, float(wall_pixels))
        method = "convex-hull-fallback"

    epsilon = max(1.5, 0.10 / pixel_m)
    approximation = cv2.approxPolyDP(best, epsilon, True)
    contour = [
        [round(float(point[0][0]), 3), round(float(point[0][1]), 3)]
        for point in approximation
    ]
    if len(contour) < 3:
        contour = [
            [round(float(point[0][0]), 3), round(float(point[0][1]), 3)]
            for point in best
        ]

    area_m2 = area_px * pixel_m * pixel_m
    if method == "wall-envelope":
        confidence = min(0.98, 0.68 + 0.08 * min(3.0, max(0.0, enclosure_ratio - 1.0)))
    else:
        confidence = 0.52
    return contour, {
        "slab_detected": len(contour) >= 3,
        "slab_method": method,
        "slab_vertex_count": len(contour),
        "slab_area_m2": round(area_m2, 4),
        "slab_enclosure_ratio": round(enclosure_ratio, 4),
        "slab_confidence": round(confidence, 4),
        "slab_join_size_px": join_size,
        "slab_envelope_axis_count": len(envelope_axes),
        "slab_minimum_wall_thickness_m": round(minimum_envelope_thickness_m, 4),
    }


def _bbox_overlap_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ix = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    iy = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = ix * iy
    first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / min(first_area, second_area)


def classify_structural_regions_2d(
    wall_axes: list[dict[str, Any]],
    thick_wall_mask: np.ndarray,
    *,
    canvas_width_m: float,
    color: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Separa paredes comuns, paredes grossas e pilares compactos.

    O mesmo bloco raster pode ser comprido numa direção (parede) ou compacto
    nas duas direções (pilar). A referência de espessura é aprendida na própria
    prancha; isso evita fixar um número de pixels que só funcionaria no
    CubiCasa.
    """
    height, width = thick_wall_mask.shape
    canvas_size = max(height, width)
    pixel_m = canvas_width_m / float(canvas_size)
    usable_axes = [
        {**axis}
        for axis in wall_axes
        if float(axis["thickness"]) * pixel_m >= 0.05
    ]
    reference_thicknesses = [
        float(axis["thickness"]) * pixel_m
        for axis in usable_axes
        if (float(axis["end"]) - float(axis["start"])) * pixel_m >= 0.75
        and 0.055 <= float(axis["thickness"]) * pixel_m <= 0.60
    ]
    if reference_thicknesses:
        # O percentil inferior representa melhor a alvenaria recorrente quando
        # a própria prancha contém muitas paredes estruturais grossas.
        typical_thickness_m = float(np.percentile(reference_thicknesses, 35))
    else:
        typical_thickness_m = 0.15
    structural_cutoff_m = max(0.22, typical_thickness_m * 1.55)
    column_min_side_m = max(0.26, typical_thickness_m * 1.70)

    columns: list[dict[str, Any]] = []
    walls: list[dict[str, Any]] = []
    for axis in usable_axes:
        length_m = (float(axis["end"]) - float(axis["start"])) * pixel_m
        thickness_m = float(axis["thickness"]) * pixel_m
        aspect = max(length_m, thickness_m) / max(1e-6, min(length_m, thickness_m))
        compact = (
            min(length_m, thickness_m) >= 0.18
            and max(length_m, thickness_m) <= 1.25
            and thickness_m >= column_min_side_m
            and aspect <= 3.0
        )
        if compact:
            columns.append({
                **axis,
                "element_type": "column",
                "source": "2d-compact-structural-mass",
                "confidence": max(0.78, float(axis.get("confidence", 0.72))),
            })
            continue
        element_type = "structural-wall" if thickness_m >= structural_cutoff_m else "wall"
        walls.append({**axis, "element_type": element_type})

    # Uma abertura morfológica quadrada recupera pilares que não chegam à
    # razão mínima de alongamento usada pelo extrator de eixos de parede.
    compact_kernel = max(5, int(round(max(0.12, typical_thickness_m * 0.8) / pixel_m)))
    compact_kernel = min(compact_kernel, max(5, min(height, width) // 8))
    compact_mask = cv2.morphologyEx(
        thick_wall_mask,
        cv2.MORPH_OPEN,
        np.ones((compact_kernel, compact_kernel), np.uint8),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(compact_mask, 8)
    for index in range(1, count):
        x, y, box_width, box_height, area = map(int, stats[index])
        width_m = box_width * pixel_m
        height_m = box_height * pixel_m
        short_m, long_m = sorted((width_m, height_m))
        if short_m < column_min_side_m or long_m > 1.25:
            continue
        if long_m / max(1e-6, short_m) > 2.8:
            continue
        roi = thick_wall_mask[y:y + box_height, x:x + box_width]
        fill = float(roi.mean()) if roi.size else 0.0
        if fill < 0.55:
            continue
        if box_width >= box_height:
            candidate = {
                "orientation": "horizontal",
                "fixed": y + (box_height - 1) / 2.0,
                "start": float(x),
                "end": float(x + box_width - 1),
                "thickness": float(box_height),
            }
        else:
            candidate = {
                "orientation": "vertical",
                "fixed": x + (box_width - 1) / 2.0,
                "start": float(y),
                "end": float(y + box_height - 1),
                "thickness": float(box_width),
            }
        candidate.update({
            "element_type": "column",
            "source": "2d-compact-structural-mass",
            "confidence": min(0.96, 0.72 + 0.20 * fill),
        })
        candidate_box = _axis_bbox(candidate)
        if any(_bbox_overlap_ratio(candidate_box, _axis_bbox(item)) >= 0.55 for item in columns):
            continue
        columns.append(candidate)

    # Pranchas CAD renderizadas frequentemente distinguem a estrutura por uma
    # massa cinza neutra. Esse canal recupera engrossamentos ligados às paredes
    # (que formariam um único componente no limiar escuro) sem usar textos
    # coloridos ou mobiliário como evidência.
    if color is not None:
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        spread = color.max(axis=2).astype(np.int16) - color.min(axis=2).astype(np.int16)
        neutral = np.asarray(
            (gray >= 65) & (gray <= 210) & (spread <= 10),
            dtype=np.uint8,
        )
        neutral = cv2.morphologyEx(neutral, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(neutral, 8)
        for index in range(1, count):
            x, y, box_width, box_height, area = map(int, stats[index])
            width_m = box_width * pixel_m
            height_m = box_height * pixel_m
            short_m, long_m = sorted((width_m, height_m))
            fill = area / max(1.0, box_width * box_height)
            if short_m < column_min_side_m or long_m > 1.25 or fill < 0.50:
                continue
            if long_m / max(1e-6, short_m) > 2.8:
                continue
            if box_width >= box_height:
                candidate = {
                    "orientation": "horizontal",
                    "fixed": y + (box_height - 1) / 2.0,
                    "start": float(x),
                    "end": float(x + box_width - 1),
                    "thickness": float(box_height),
                }
            else:
                candidate = {
                    "orientation": "vertical",
                    "fixed": x + (box_width - 1) / 2.0,
                    "start": float(y),
                    "end": float(y + box_height - 1),
                    "thickness": float(box_width),
                }
            candidate.update({
                "element_type": "column",
                "source": "2d-neutral-structural-mass",
                "confidence": min(0.97, 0.76 + 0.20 * fill),
            })
            candidate_box = _axis_bbox(candidate)
            if any(_bbox_overlap_ratio(candidate_box, _axis_bbox(item)) >= 0.55 for item in columns):
                continue
            columns.append(candidate)

    columns.sort(key=lambda item: (float(item["fixed"]), float(item["start"])))
    diagnostic = {
        "typical_wall_thickness_m": round(typical_thickness_m, 4),
        "structural_cutoff_m": round(structural_cutoff_m, 4),
        "column_min_side_m": round(column_min_side_m, 4),
        "structural_wall_count": sum(item["element_type"] == "structural-wall" for item in walls),
        "column_count": len(columns),
        "thin_axes_rejected": len(wall_axes) - len(usable_axes),
    }
    return walls, columns, diagnostic


def detect_windows_2d(
    color: np.ndarray,
    *,
    canvas_width_m: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Detecta linhas de janela por cromaticidade e componentes alongados."""
    height, width = color.shape[:2]
    canvas_size = max(height, width)
    pixel_m = canvas_width_m / float(canvas_size)
    blue, green, red = cv2.split(color)
    blue_i = blue.astype(np.int16)
    window_mask = np.asarray(
        (blue_i - green.astype(np.int16) >= 4)
        & (blue_i - red.astype(np.int16) >= 4)
        & (blue >= 120)
        & (blue < 255),
        dtype=np.uint8,
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(window_mask, 8)
    windows: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, box_width, box_height, area = map(int, stats[index])
        long_side = max(box_width, box_height)
        short_side = min(box_width, box_height)
        length_m = long_side * pixel_m
        if area < 12 or not 0.35 <= length_m <= 3.5:
            continue
        if long_side / max(1, short_side) < 2.2:
            continue
        if box_height >= box_width:
            start = (x + (box_width - 1) / 2.0, float(y))
            end = (x + (box_width - 1) / 2.0, float(y + box_height - 1))
            orientation = "vertical"
        else:
            start = (float(x), y + (box_height - 1) / 2.0)
            end = (float(x + box_width - 1), y + (box_height - 1) / 2.0)
            orientation = "horizontal"
        windows.append({
            "type": "window",
            "orientation": orientation,
            "start_px": [round(start[0], 2), round(start[1], 2)],
            "end_px": [round(end[0], 2), round(end[1], 2)],
            "confidence": min(0.99, 0.78 + 0.21 * min(1.0, area / max(1.0, long_side * 8.0))),
            "source": "2d-window-chroma",
        })
    return windows, window_mask


def _box_gap(first: _Component, second: _Component) -> tuple[int, int]:
    dx = max(0, max(first.x, second.x) - min(first.x2, second.x2) - 1)
    dy = max(0, max(first.y, second.y) - min(first.y2, second.y2) - 1)
    return dx, dy


def _cluster_components(components: list[_Component], maximum_gap: int = 5) -> list[_Component]:
    groups: list[list[_Component]] = []
    for component in sorted(components, key=lambda item: item.area, reverse=True):
        matches: list[int] = []
        for group_index, group in enumerate(groups):
            if any(max(_box_gap(component, other)) <= maximum_gap for other in group):
                x0 = min([component.x, *[item.x for item in group]])
                y0 = min([component.y, *[item.y for item in group]])
                x1 = max([component.x2, *[item.x2 for item in group]])
                y1 = max([component.y2, *[item.y2 for item in group]])
                if x1 - x0 + 1 <= 72 and y1 - y0 + 1 <= 72:
                    matches.append(group_index)
        if not matches:
            groups.append([component])
            continue
        target = matches[0]
        groups[target].append(component)
        for redundant in reversed(matches[1:]):
            groups[target].extend(groups.pop(redundant))

    clustered: list[_Component] = []
    for group in groups:
        x0 = min(item.x for item in group)
        y0 = min(item.y for item in group)
        x1 = max(item.x2 for item in group)
        y1 = max(item.y2 for item in group)
        clustered.append(_Component(x0, y0, x1 - x0 + 1, y1 - y0 + 1, sum(item.area for item in group)))
    return clustered


def _near_wall_score(component: _Component, thick_wall_mask: np.ndarray) -> tuple[float, Literal["vertical", "horizontal"]]:
    height, width = thick_wall_mask.shape
    padding = 7
    x0 = max(0, component.x - padding)
    y0 = max(0, component.y - padding)
    x1 = min(width, component.x2 + padding + 1)
    y1 = min(height, component.y2 + padding + 1)
    local = thick_wall_mask[y0:y1, x0:x1]
    if local.size == 0:
        return 0.0, "horizontal"
    left = float(local[:, :min(padding + 2, local.shape[1])].mean())
    right = float(local[:, max(0, local.shape[1] - padding - 2):].mean())
    top = float(local[:min(padding + 2, local.shape[0]), :].mean())
    bottom = float(local[max(0, local.shape[0] - padding - 2):, :].mean())
    vertical_score = max(left, right)
    horizontal_score = max(top, bottom)
    if vertical_score >= horizontal_score:
        return vertical_score, "vertical"
    return horizontal_score, "horizontal"


def detect_doors_2d(
    color: np.ndarray,
    thick_wall_mask: np.ndarray,
    wall_axes: list[dict[str, Any]],
    *,
    canvas_width_m: float,
    window_openings: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    """Propõe portas a partir de componentes claros de arco e folha."""
    height, width = color.shape[:2]
    canvas_size = max(height, width)
    pixel_m = canvas_width_m / float(canvas_size)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 15, 60, apertureSize=3, L2gradient=True)
    strict_dark = np.asarray(gray <= 138, dtype=np.uint8)
    blocked = cv2.dilate(strict_dark, np.ones((3, 3), np.uint8))
    residual = np.asarray((edges > 0) & (blocked == 0), dtype=np.uint8)
    residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    count, _, stats, _ = cv2.connectedComponentsWithStats(residual, 8)
    components = [
        _Component(*map(int, stats[index]))
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= 10
    ]
    clusters = _cluster_components(components, maximum_gap=5)
    minimum_side = max(14.0, 0.28 / pixel_m)
    maximum_side = max(58.0, 1.35 / pixel_m)
    proposals: list[dict[str, Any]] = []

    for cluster in clusters:
        short_side = min(cluster.width, cluster.height)
        long_side = max(cluster.width, cluster.height)
        if short_side < minimum_side or long_side > maximum_side:
            continue
        if long_side / max(1.0, short_side) > 2.55:
            continue
        if cluster.area < max(58, int(round(long_side * 2.2))):
            continue
        wall_score, orientation = _near_wall_score(cluster, thick_wall_mask)
        if wall_score < 0.025:
            continue
        density = cluster.area / max(1.0, cluster.width * cluster.height)
        squareness = short_side / max(1.0, long_side)
        confidence = min(0.96, 0.43 + 0.24 * min(1.0, density * 4.0) + 0.20 * squareness + 0.13 * min(1.0, wall_score * 4.0))
        proposals.append({
            "type": "door",
            "orientation": orientation,
            "bbox_px": [cluster.x, cluster.y, cluster.width, cluster.height],
            "center_px": [cluster.x + (cluster.width - 1) / 2.0, cluster.y + (cluster.height - 1) / 2.0],
            "confidence": round(confidence, 4),
            "wall_score": round(wall_score, 4),
            "component_density": round(density, 4),
            "source": "2d-door-arc-component",
        })

    # Associa o arco à faixa de parede que toca uma das bordas de seu bbox. A
    # faixa precisa perder massa escura no intervalo: isso elimina armários e
    # louças encostados em uma parede contínua.
    associated: list[dict[str, Any]] = []
    pixel_m = canvas_width_m / float(max(height, width))
    maximum_edge_distance = max(7.0, 0.16 / pixel_m)
    host_axes = [{**axis} for axis in wall_axes]
    for window in window_openings or []:
        start = window["start_px"]
        end = window["end_px"]
        if window["orientation"] == "vertical":
            host_axes.append({
                "orientation": "vertical",
                "fixed": (float(start[0]) + float(end[0])) / 2.0,
                "start": min(float(start[1]), float(end[1])),
                "end": max(float(start[1]), float(end[1])),
                "thickness": 8.0,
                "source": "2d-window-host-evidence",
            })
        else:
            host_axes.append({
                "orientation": "horizontal",
                "fixed": (float(start[1]) + float(end[1])) / 2.0,
                "start": min(float(start[0]), float(end[0])),
                "end": max(float(start[0]), float(end[0])),
                "thickness": 8.0,
                "source": "2d-window-host-evidence",
            })
    for proposal in proposals:
        x, y, box_width, box_height = proposal["bbox_px"]
        x2 = x + box_width - 1
        y2 = y + box_height - 1
        matches: list[tuple[float, dict[str, Any], float]] = []
        for axis in host_axes:
            fixed = float(axis["fixed"])
            if axis["orientation"] == "vertical":
                edge_distance = min(abs(fixed - x), abs(fixed - x2))
                along_gap = max(0.0, y - float(axis["end"]), float(axis["start"]) - y2)
                if edge_distance > maximum_edge_distance or along_gap > 15.0:
                    continue
                half = max(2, int(round(float(axis["thickness"]) / 2.0)))
                cx0 = max(0, int(round(fixed)) - half)
                cx1 = min(width, int(round(fixed)) + half + 1)
                cy0, cy1 = max(0, y + 2), min(height, y2 - 1)
                corridor = thick_wall_mask[cy0:cy1, cx0:cx1]
            else:
                edge_distance = min(abs(fixed - y), abs(fixed - y2))
                along_gap = max(0.0, x - float(axis["end"]), float(axis["start"]) - x2)
                if edge_distance > maximum_edge_distance or along_gap > 15.0:
                    continue
                half = max(2, int(round(float(axis["thickness"]) / 2.0)))
                cy0 = max(0, int(round(fixed)) - half)
                cy1 = min(height, int(round(fixed)) + half + 1)
                cx0, cx1 = max(0, x + 2), min(width, x2 - 1)
                corridor = thick_wall_mask[cy0:cy1, cx0:cx1]
            gap_density = float(corridor.mean()) if corridor.size else 1.0
            score = edge_distance + 0.5 * along_gap + 14.0 * gap_density
            matches.append((score, axis, gap_density))
        if not matches:
            continue
        _, host, gap_density = min(matches, key=lambda item: item[0])
        if gap_density > 0.46:
            continue
        fixed = float(host["fixed"])
        if host["orientation"] == "vertical":
            start_px = [round(fixed, 2), float(y)]
            end_px = [round(fixed, 2), float(y2)]
        else:
            start_px = [float(x), round(fixed, 2)]
            end_px = [float(x2), round(fixed, 2)]
        associated.append({
            **proposal,
            "orientation": host["orientation"],
            "start_px": start_px,
            "end_px": end_px,
            "gap_density": round(gap_density, 4),
        })

    # Supressão de propostas sobrepostas: mantém a evidência mais forte.
    kept: list[dict[str, Any]] = []
    for proposal in sorted(associated, key=lambda item: float(item["confidence"]), reverse=True):
        cx, cy = proposal["center_px"]
        if any(math.hypot(cx - item["center_px"][0], cy - item["center_px"][1]) < 18 for item in kept):
            continue
        kept.append(proposal)
    return kept, residual, proposals


def vectorize_floorplan_2d(
    image_path: Path,
    *,
    canvas_width_m: float = 16.0,
) -> dict[str, Any]:
    image_path = Path(image_path)
    color = _read_color(image_path)
    walls, thick_wall_mask, wall_diagnostic = detect_wall_regions_2d(
        color,
        canvas_width_m=canvas_width_m,
    )
    walls, columns, structural_diagnostic = classify_structural_regions_2d(
        walls,
        thick_wall_mask,
        canvas_width_m=canvas_width_m,
        color=color,
    )
    windows, _ = detect_windows_2d(color, canvas_width_m=canvas_width_m)
    doors, _, door_proposals = detect_doors_2d(
        color,
        thick_wall_mask,
        walls,
        canvas_width_m=canvas_width_m,
        window_openings=windows,
    )
    return {
        "ok": True,
        "method": "raster-2d-morphology",
        "image_size": [int(color.shape[1]), int(color.shape[0])],
        "canvas_width_m": float(canvas_width_m),
        "walls": walls,
        "columns": columns,
        "openings": [*doors, *windows],
        "diagnostics": {
            **wall_diagnostic,
            **structural_diagnostic,
            "wall_count": len(walls),
            "door_count": len(doors),
            "window_count": len(windows),
            "door_proposal_count": len(door_proposals),
            "door_proposals": door_proposals,
        },
    }


def _opening_interval(opening: dict[str, Any]) -> tuple[str, float, float, float]:
    start = opening["start_px"]
    end = opening["end_px"]
    orientation = str(opening["orientation"])
    if orientation == "vertical":
        return orientation, (float(start[0]) + float(end[0])) / 2.0, min(float(start[1]), float(end[1])), max(float(start[1]), float(end[1]))
    return orientation, (float(start[1]) + float(end[1])) / 2.0, min(float(start[0]), float(end[0])), max(float(start[0]), float(end[0]))


def _parallel_gap_line_score(
    color: np.ndarray,
    axis: dict[str, Any],
    start: float,
    end: float,
) -> tuple[float, int]:
    """Mede caixilhos: duas ou mais linhas finas paralelas dentro do vão."""
    height, width = color.shape[:2]
    fixed = int(round(float(axis["fixed"])))
    first = max(0, int(math.floor(start)))
    last = int(math.ceil(end))
    half_strip = max(6, int(round(float(axis["thickness"]) * 1.8)))
    if axis["orientation"] == "horizontal":
        x0, x1 = first, min(width, last + 1)
        y0, y1 = max(0, fixed - half_strip), min(height, fixed + half_strip + 1)
    else:
        x0, x1 = max(0, fixed - half_strip), min(width, fixed + half_strip + 1)
        y0, y1 = first, min(height, last + 1)
    roi = color[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0, 0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 18, 70, apertureSize=3, L2gradient=True)
    along_length = roi.shape[1] if axis["orientation"] == "horizontal" else roi.shape[0]
    kernel_length = max(5, int(round(along_length * 0.38)))
    kernel = (
        np.ones((1, kernel_length), np.uint8)
        if axis["orientation"] == "horizontal"
        else np.ones((kernel_length, 1), np.uint8)
    )
    parallel = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(parallel, 8)
    supports: list[float] = []
    for index in range(1, count):
        box_width = int(stats[index, cv2.CC_STAT_WIDTH])
        box_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        component_length = box_width if axis["orientation"] == "horizontal" else box_height
        component_thickness = box_height if axis["orientation"] == "horizontal" else box_width
        support = component_length / max(1.0, float(along_length))
        if support >= 0.42 and component_thickness <= max(5, half_strip // 2):
            supports.append(support)
    if len(supports) < 2:
        return 0.0, len(supports)
    score = min(1.0, (len(supports) / 3.0) * min(1.0, max(supports) / 0.72))
    return round(score, 4), len(supports)


def _opening_wall_match(
    opening: dict[str, Any],
    axis: dict[str, Any],
    *,
    fixed_tolerance: float,
) -> tuple[float, float, float] | None:
    orientation, fixed, start, end = _opening_interval(opening)
    if orientation != axis["orientation"]:
        return None
    fixed_distance = abs(fixed - float(axis["fixed"]))
    if fixed_distance > fixed_tolerance * 1.75:
        return None
    wall_start, wall_end = float(axis["start"]), float(axis["end"])
    overlap = max(0.0, min(end, wall_end) - max(start, wall_start))
    along_gap = max(0.0, wall_start - end, start - wall_end)
    if overlap <= 0.0 and along_gap > fixed_tolerance:
        return None
    confidence = float(opening.get("confidence", 0.5))
    score = confidence + 0.18 * min(1.0, overlap / max(1.0, end - start))
    score -= 0.10 * min(1.0, fixed_distance / max(1.0, fixed_tolerance))
    score -= 0.08 * min(1.0, along_gap / max(1.0, fixed_tolerance))
    return score, start, end


def build_canonical_wall_hosts_2d(
    axes: list[dict[str, Any]],
    openings: list[dict[str, Any]],
    color: np.ndarray,
    *,
    door_proposals: list[dict[str, Any]] | None = None,
    fixed_tolerance: float,
    minimum_gap: float,
    maximum_gap: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Cria paredes-mãe primeiro; intervalos viram aberturas hospedadas depois."""
    if not axes:
        return [], [], {
            "source_wall_segment_count": 0,
            "canonical_wall_count": 0,
            "wall_gap_count": 0,
            "classified_wall_gaps": 0,
            "unclassified_wall_gaps": 0,
            "unmatched_openings": len(openings),
            "wall_gaps": [],
        }

    parent = list(range(len(axes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first_index, first in enumerate(axes):
        for second_index in range(first_index + 1, len(axes)):
            second = axes[second_index]
            if first["orientation"] != second["orientation"]:
                continue
            if abs(float(first["fixed"]) - float(second["fixed"])) > fixed_tolerance:
                continue
            first_thickness = max(1.0, float(first["thickness"]))
            second_thickness = max(1.0, float(second["thickness"]))
            if min(first_thickness, second_thickness) / max(first_thickness, second_thickness) < 0.42:
                continue
            gap = max(
                0.0,
                max(float(first["start"]), float(second["start"]))
                - min(float(first["end"]), float(second["end"])),
            )
            if gap <= maximum_gap:
                union(first_index, second_index)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, axis in enumerate(axes):
        groups.setdefault(find(index), []).append({"source_index": index, **axis})

    canonical_axes: list[dict[str, Any]] = []
    for members in groups.values():
        members.sort(key=lambda item: (float(item["start"]), float(item["end"])))
        coverage: list[list[float]] = []
        for member in members:
            start, end = float(member["start"]), float(member["end"])
            if not coverage or start > coverage[-1][1]:
                coverage.append([start, end])
            else:
                coverage[-1][1] = max(coverage[-1][1], end)
        lengths = [max(1.0, float(item["end"]) - float(item["start"])) for item in members]
        total_length = sum(lengths)
        fixed = sum(float(item["fixed"]) * length for item, length in zip(members, lengths)) / total_length
        gaps: list[dict[str, Any]] = []
        for previous, following in zip(coverage, coverage[1:]):
            gap_start, gap_end = previous[1], following[0]
            if minimum_gap <= gap_end - gap_start <= maximum_gap:
                gaps.append({
                    "start": gap_start,
                    "end": gap_end,
                    "left_support": previous[1] - previous[0],
                    "right_support": following[1] - following[0],
                })
        canonical_axes.append({
            "orientation": members[0]["orientation"],
            "fixed": fixed,
            "start": min(float(item["start"]) for item in members),
            "end": max(float(item["end"]) for item in members),
            "thickness": float(np.median([float(item["thickness"]) for item in members])),
            "confidence": min(float(item.get("confidence", 0.5)) for item in members),
            "source": "2d-canonical-wall-carrier",
            "element_type": (
                "structural-wall"
                if any(item.get("element_type") == "structural-wall" for item in members)
                else "wall"
            ),
            "source_segment_count": len(members),
            "_gaps": gaps,
        })

    canonical_axes.sort(key=lambda item: (item["orientation"], float(item["fixed"]), float(item["start"])))
    used_openings: set[int] = set()
    hosted: list[dict[str, Any]] = []
    gap_diagnostics: list[dict[str, Any]] = []

    for host_index, axis in enumerate(canonical_axes):
        for gap in axis.pop("_gaps"):
            gap_start, gap_end = float(gap["start"]), float(gap["end"])
            gap_length = gap_end - gap_start
            continuity = min(
                1.0,
                min(float(gap["left_support"]), float(gap["right_support"]))
                / max(1.0, minimum_gap * 1.5),
            )
            door_score = 0.0
            window_score = 0.0
            door_source_index: int | None = None
            window_source_index: int | None = None
            for opening_index, opening in enumerate(openings):
                match = _opening_wall_match(opening, axis, fixed_tolerance=fixed_tolerance)
                if match is None:
                    continue
                match_score, opening_start, opening_end = match
                overlap = max(0.0, min(gap_end, opening_end) - max(gap_start, opening_start))
                center = (opening_start + opening_end) / 2.0
                if overlap <= 0.0 and not gap_start - fixed_tolerance <= center <= gap_end + fixed_tolerance:
                    continue
                gap_support = min(1.0, overlap / max(1.0, min(gap_length, opening_end - opening_start)))
                score = match_score * 0.78 + 0.22 * gap_support
                if opening["type"] == "door" and score > door_score:
                    door_score, door_source_index = score, opening_index
                elif opening["type"] == "window" and score > window_score:
                    window_score, window_source_index = score, opening_index

            for proposal in door_proposals or []:
                x, y, box_width, box_height = proposal["bbox_px"]
                if axis["orientation"] == "horizontal":
                    along_start, along_end = float(x), float(x + box_width - 1)
                    perpendicular_start, perpendicular_end = float(y), float(y + box_height - 1)
                else:
                    along_start, along_end = float(y), float(y + box_height - 1)
                    perpendicular_start, perpendicular_end = float(x), float(x + box_width - 1)
                perpendicular_distance = max(
                    0.0,
                    perpendicular_start - float(axis["fixed"]),
                    float(axis["fixed"]) - perpendicular_end,
                )
                overlap = max(0.0, min(gap_end, along_end) - max(gap_start, along_start))
                if perpendicular_distance > fixed_tolerance * 1.75 or overlap <= 0.0:
                    continue
                proposal_score = (
                    float(proposal.get("confidence", 0.5)) * 0.72
                    + 0.20 * min(1.0, overlap / max(1.0, gap_length))
                    + 0.08 * continuity
                )
                if gap_length > maximum_gap * 0.72:
                    proposal_score -= 0.18
                door_score = max(door_score, proposal_score)

            parallel_score, parallel_count = _parallel_gap_line_score(color, axis, gap_start, gap_end)
            if parallel_score >= 0.42:
                window_score = max(window_score, 0.57 + 0.31 * parallel_score + 0.08 * continuity)

            kind: str | None = None
            confidence = 0.0
            source_index: int | None = None
            if door_score >= 0.58 or window_score >= 0.58:
                # Arcos e folhas também geram várias arestas paralelas no
                # Canny. Sem um componente de janela independente, uma
                # evidência forte de porta prevalece sobre esse falso caixilho.
                if (
                    window_score > door_score + 0.04
                    and (window_source_index is not None or door_score < 0.64)
                ):
                    kind, confidence, source_index = "window", window_score, window_source_index
                else:
                    kind, confidence, source_index = "door", door_score, door_source_index
            if source_index is not None:
                used_openings.add(source_index)
            if kind is not None:
                hosted.append({
                    "type": kind,
                    "orientation": axis["orientation"],
                    "start_px": (
                        [round(gap_start, 2), round(float(axis["fixed"]), 2)]
                        if axis["orientation"] == "horizontal"
                        else [round(float(axis["fixed"]), 2), round(gap_start, 2)]
                    ),
                    "end_px": (
                        [round(gap_end, 2), round(float(axis["fixed"]), 2)]
                        if axis["orientation"] == "horizontal"
                        else [round(float(axis["fixed"]), 2), round(gap_end, 2)]
                    ),
                    "confidence": round(min(0.99, confidence), 4),
                    "source": "2d-canonical-wall-gap",
                    "semantic_reason": "intervalo interno classificado depois da consolidação da parede-mãe",
                    "host_axis_index": host_index,
                })
            gap_diagnostics.append({
                "host_axis_index": host_index,
                "orientation": axis["orientation"],
                "fixed": round(float(axis["fixed"]), 3),
                "start": round(gap_start, 3),
                "end": round(gap_end, 3),
                "classification": kind or "unknown",
                "confidence": round(confidence, 4),
                "door_score": round(door_score, 4),
                "window_score": round(window_score, 4),
                "parallel_line_count": parallel_count,
            })

    # Componentes reconhecidos dentro de uma parede contínua também são
    # hospedados, mesmo quando a massa escura não gerou um gap explícito.
    for opening_index, opening in enumerate(openings):
        if opening_index in used_openings:
            continue
        matches: list[tuple[float, int, float, float]] = []
        for host_index, axis in enumerate(canonical_axes):
            match = _opening_wall_match(opening, axis, fixed_tolerance=fixed_tolerance)
            if match is not None:
                score, start, end = match
                matches.append((score, host_index, start, end))
        if not matches:
            continue
        score, host_index, start, end = max(matches, key=lambda item: item[0])
        if score < 0.52:
            continue
        axis = canonical_axes[host_index]
        start = max(float(axis["start"]), start)
        end = min(float(axis["end"]), end)
        if end - start < minimum_gap * 0.65:
            continue
        candidate = {
            **opening,
            "orientation": axis["orientation"],
            "start_px": (
                [round(start, 2), round(float(axis["fixed"]), 2)]
                if axis["orientation"] == "horizontal"
                else [round(float(axis["fixed"]), 2), round(start, 2)]
            ),
            "end_px": (
                [round(end, 2), round(float(axis["fixed"]), 2)]
                if axis["orientation"] == "horizontal"
                else [round(float(axis["fixed"]), 2), round(end, 2)]
            ),
            "host_axis_index": host_index,
            "source": "2d-canonical-wall-component",
            "semantic_reason": "componente hospedado na parede-mãe sem dividir sua identidade",
        }
        _, _, candidate_start, candidate_end = _opening_interval(candidate)
        duplicate = False
        for existing in hosted:
            if existing["host_axis_index"] != host_index or existing["type"] != candidate["type"]:
                continue
            _, _, existing_start, existing_end = _opening_interval(existing)
            overlap = max(0.0, min(candidate_end, existing_end) - max(candidate_start, existing_start))
            if overlap / max(1.0, min(candidate_end - candidate_start, existing_end - existing_start)) >= 0.45:
                duplicate = True
                break
        if not duplicate:
            hosted.append(candidate)
            used_openings.add(opening_index)

    unmatched = len(openings) - len(used_openings)
    classified = sum(item["classification"] != "unknown" for item in gap_diagnostics)
    return canonical_axes, hosted, {
        "source_wall_segment_count": len(axes),
        "canonical_wall_count": len(canonical_axes),
        "wall_segments_absorbed": len(axes) - len(canonical_axes),
        "wall_gap_count": len(gap_diagnostics),
        "classified_wall_gaps": classified,
        "unclassified_wall_gaps": len(gap_diagnostics) - classified,
        "unmatched_openings": unmatched,
        "wall_gaps": gap_diagnostics,
    }


def raster_2d_image_to_editor_model(
    image_path: Path,
    *,
    canvas_width_m: float = 20.0,
) -> dict[str, Any]:
    """Converte o resultado 2D no contrato editável ``ModeloPlanta``."""
    image_path = Path(image_path)
    if image_path.suffix.lower() not in RASTER_2D_IMAGE_EXTENSIONS:
        raise Raster2DError("Formato raster não suportado pelo detector 2D.")
    if not math.isfinite(canvas_width_m) or canvas_width_m <= 0:
        raise Raster2DError("A largura do canvas precisa ser positiva.")
    color = _read_color(image_path)
    result = vectorize_floorplan_2d(image_path, canvas_width_m=canvas_width_m)
    image_height, image_width = color.shape[:2]
    canvas_size = max(image_width, image_height)
    pad_x = (canvas_size - image_width) // 2
    pad_y = (canvas_size - image_height) // 2
    square = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    square[pad_y:pad_y + image_height, pad_x:pad_x + image_width] = color
    pixel_m = canvas_width_m / float(canvas_size)
    axes, hosted, host_diagnostic = build_canonical_wall_hosts_2d(
        result["walls"],
        result["openings"],
        color,
        door_proposals=result["diagnostics"].get("door_proposals", []),
        fixed_tolerance=max(7.0, 0.16 / pixel_m),
        minimum_gap=max(8.0, 0.28 / pixel_m),
        maximum_gap=max(30.0, 2.40 / pixel_m),
    )
    unmatched = int(host_diagnostic["unmatched_openings"])
    slab_contour_px, slab_diagnostic = detect_slab_contour_2d(
        axes,
        color.shape,
        canvas_width_m=canvas_width_m,
    )

    def world_point(x: float, y: float) -> tuple[float, float]:
        square_x = x + pad_x
        square_y = y + pad_y
        return (
            round(square_x * pixel_m, 5),
            round((canvas_size - square_y) * pixel_m, 5),
        )

    slab_contour = [world_point(point[0], point[1]) for point in slab_contour_px]

    walls: list[dict[str, Any]] = []
    for index, axis in enumerate(axes):
        if axis["orientation"] == "vertical":
            ax, ay = world_point(float(axis["fixed"]), float(axis["end"]))
            bx, by = world_point(float(axis["fixed"]), float(axis["start"]))
        else:
            ax, ay = world_point(float(axis["start"]), float(axis["fixed"]))
            bx, by = world_point(float(axis["end"]), float(axis["fixed"]))
        structural = axis.get("element_type") == "structural-wall"
        walls.append({
            "id": f"W-2D-{index + 1:03d}",
            "ax": ax,
            "ay": ay,
            "bx": bx,
            "by": by,
            "espessura": round(max(0.06, min(0.80, float(axis["thickness"]) * pixel_m)), 4),
            "altura": 2.8,
            "elevacao": 0.0,
            "layer": "Wall-Structural-Raster-2D" if structural else "Wall-Raster-2D",
            "nome": f"Parede estrutural 2D {index + 1}" if structural else f"Parede 2D {index + 1}",
            "tipo": "structural-wall" if structural else "wall",
            "ifc_class": "IfcWall",
            "origem": "raster-2d-morphology",
            "confidence": round(float(axis["confidence"]), 4),
        })

    wall_count = len(walls)
    for index, column in enumerate(result.get("columns", [])):
        if column["orientation"] == "vertical":
            ax, ay = world_point(float(column["fixed"]), float(column["end"]))
            bx, by = world_point(float(column["fixed"]), float(column["start"]))
        else:
            ax, ay = world_point(float(column["start"]), float(column["fixed"]))
            bx, by = world_point(float(column["end"]), float(column["fixed"]))
        walls.append({
            "id": f"C-2D-{index + 1:03d}",
            "ax": ax,
            "ay": ay,
            "bx": bx,
            "by": by,
            "espessura": round(max(0.18, min(1.25, float(column["thickness"]) * pixel_m)), 4),
            "altura": 2.8,
            "elevacao": 0.0,
            "layer": "Column-Raster-2D",
            "nome": f"Pilar 2D {index + 1}",
            "tipo": "column",
            "ifc_class": "IfcColumn",
            "origem": "raster-2d-column",
            "confidence": round(float(column["confidence"]), 4),
        })

    editor_openings: list[dict[str, Any]] = []
    for index, opening in enumerate(hosted):
        host_index = int(opening["host_axis_index"])
        host_axis = axes[host_index]
        wall = walls[host_index]
        _, _, opening_start, opening_end = _opening_interval(opening)
        opening_center = (opening_start + opening_end) / 2.0
        if host_axis["orientation"] == "vertical":
            center_along = (float(host_axis["end"]) - opening_center) * pixel_m
        else:
            center_along = (opening_center - float(host_axis["start"])) * pixel_m
        wall_length = math.hypot(float(wall["bx"]) - float(wall["ax"]), float(wall["by"]) - float(wall["ay"]))
        opening_width = min(max(0.30, (opening_end - opening_start) * pixel_m), max(0.30, wall_length - 0.04))
        center_along = max(opening_width / 2.0, min(wall_length - opening_width / 2.0, center_along))
        kind = str(opening["type"])
        editor_openings.append({
            "id": f"O-2D-{index + 1:03d}",
            "parede_id": wall["id"],
            "tipo": kind,
            "s_centro": round(center_along, 5),
            "largura": round(opening_width, 4),
            "nome": f"{'Porta' if kind == 'door' else 'Janela'} 2D {index + 1}",
            "altura": 2.1 if kind == "door" else 1.2,
            "peitoril": 0.0 if kind == "door" else 1.0,
            "origem": str(opening.get("source", "raster-2d-opening")),
            "confidence": round(float(opening.get("confidence", 0.5)), 4),
            "semantic_reason": str(
                opening.get(
                    "semantic_reason",
                    "componente 2D hospedado na parede-mãe",
                )
            ),
        })

    ok, encoded = cv2.imencode(".png", square)
    if not ok:
        raise Raster2DError("Não foi possível codificar a imagem alinhada.")
    door_count = sum(item["tipo"] == "door" for item in editor_openings)
    window_count = sum(item["tipo"] == "window" for item in editor_openings)
    result["diagnostics"]["hosted_openings"] = len(editor_openings)
    result["diagnostics"].update(host_diagnostic)
    result["diagnostics"].update(slab_diagnostic)
    slab_active = len(slab_contour) >= 3
    return {
        "ok": True,
        "escala": pixel_m,
        "single_line": False,
        "nome": image_path.stem,
        "bbox": {"xmin": 0.0, "ymin": 0.0, "xmax": canvas_width_m, "ymax": canvas_width_m},
        "diagnostico": {
            "sobras": 0,
            "cantos_costurados": 0,
            "blocos_esquadria": len(editor_openings),
            "elementos_lidos": len(walls) + len(editor_openings) + int(slab_active),
            "geometrias_aproximadas": unmatched,
        },
        "source": {
            "format": image_path.suffix.lower().lstrip(".") or "image",
            "family": "raster",
            "mode": "raster-2d-morphology",
            "semantic_level": "geometric",
            "scale_source": "user-canvas-width",
        },
        "reference": {
            "kind": "raster2seq",
            "engine": "morphology-2d",
            "label": "Regiões e símbolos detectados em 2D",
            "bounds": [0.0, 0.0, float(canvas_width_m), float(canvas_width_m)],
            "image_mime": "image/png",
            "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "canvas_size": [canvas_size, canvas_size],
            "canvas_width_m": canvas_width_m,
            "rooms": [],
            "openings": [],
        },
        "warnings": [
            "Protótipo 2D: massas escuras, espessuras estruturais, arcos e componentes cromáticos dependem do estilo gráfico da planta.",
            f"A topologia consolidou {host_diagnostic['source_wall_segment_count']} trecho(s) em {host_diagnostic['canonical_wall_count']} parede(s)-mãe antes de criar as aberturas.",
            f"A análise separou {len(walls) - wall_count} pilar(es) compacto(s) e {result['diagnostics'].get('structural_wall_count', 0)} parede(s) estrutural(is).",
            f"Foram hospedadas {door_count} porta(s) e {window_count} janela(s); revise falsos positivos de escadas e mobiliário.",
            *(
                [f"{host_diagnostic['unclassified_wall_gaps']} intervalo(s) permaneceram sem classificação; a parede-mãe foi mantida contínua."]
                if host_diagnostic["unclassified_wall_gaps"]
                else []
            ),
            *(
                ["A laje foi proposta pelo contorno côncavo externo das paredes e permanece editável."]
                if slab_diagnostic["slab_method"] == "wall-envelope"
                else []
            ),
            *(
                ["A malha externa não fechou; a laje usa um hull convexo provisório e precisa de revisão."]
                if slab_diagnostic["slab_method"] == "convex-hull-fallback"
                else []
            ),
            *(
                ["Não foi possível propor uma laje: faltou uma envoltória mínima de paredes."]
                if not slab_active
                else []
            ),
            *([f"{unmatched} abertura(s) ficaram sem parede hospedeira e não entraram no modelo editável."] if unmatched else []),
        ],
        "paredes": walls,
        "aberturas": editor_openings,
        "laje": {
            "contorno": slab_contour,
            "piso": {"ativo": slab_active, "espessura": 0.12},
            "teto": {"ativo": False, "espessura": 0.12},
        },
        "spaces": [],
        "raster_2d": result["diagnostics"],
    }
