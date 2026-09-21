"""Local, non-generative scale and floor-area estimation for paid preflight.

The detector output is deliberately reduced to measurements. Wall/opening
coordinates from this module must never be passed to the Astra geometry stage.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from statistics import median
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .area_preinspection import first_page_text
from .pre_wall_opening_import import run_pre_wall_pipeline


_SCALE_PATTERN = re.compile(
    r"(?:escala\s*)?1\s*(?:[:/])\s*(\d{1,4})",
    flags=re.IGNORECASE,
)


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def printed_scale_m_per_px(original_path: Path, image_path: Path) -> dict[str, Any] | None:
    """Read an unambiguous printed scale from PDF text and its raster metadata."""
    if original_path.suffix.lower() != ".pdf":
        return None
    denominators = {
        int(match)
        for match in _SCALE_PATTERN.findall(first_page_text(original_path))
        if 10 <= int(match) <= 1000
    }
    if len(denominators) != 1:
        return None
    render_path = image_path.parent / "pdf_render.json"
    if not render_path.is_file():
        return None
    try:
        render = json.loads(render_path.read_text(encoding="utf-8"))
        render_scale = _finite_positive(render.get("render_scale"))
    except (OSError, ValueError, TypeError):
        return None
    if render_scale is None:
        return None
    denominator = denominators.pop()
    # A PDF point is 1/72 inch. render_scale is pixels per PDF point.
    meters_per_pixel = (0.0254 / 72.0) * denominator / render_scale
    return {
        "meters_per_pixel": meters_per_pixel,
        "denominator": denominator,
        "confidence": 0.98,
        "source": "printed-scale",
    }


def door_wall_scale(result: dict[str, Any]) -> dict[str, Any] | None:
    """Estimate scale from plausible door widths, stabilized by wall thickness."""
    detection = result.get("detection_scale") or {}
    wall_scale = _finite_positive(detection.get("detection_pixel_m"))
    widths: list[float] = []
    for opening in result.get("openings") or []:
        if not isinstance(opening, dict) or opening.get("type") != "door":
            continue
        start = opening.get("start_px")
        end = opening.get("end_px")
        if not (
            isinstance(start, (list, tuple))
            and isinstance(end, (list, tuple))
            and len(start) >= 2
            and len(end) >= 2
        ):
            continue
        try:
            width = math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(width) or width <= 0:
            continue
        # The detector's wall-thickness scale is only a plausibility filter.
        if wall_scale is not None and not 0.60 <= width * wall_scale <= 1.20:
            continue
        widths.append(width)

    if len(widths) < 5:
        if wall_scale is None:
            return None
        return {
            "meters_per_pixel": wall_scale,
            "source": "wall-thickness",
            "confidence": 0.45,
            "door_candidates": len(widths),
        }

    door_scale = 0.90 / median(widths)
    if wall_scale is None:
        return {
            "meters_per_pixel": door_scale,
            "source": "door-width",
            "confidence": min(0.78, 0.58 + len(widths) / 200.0),
            "door_candidates": len(widths),
        }
    ratio = door_scale / wall_scale
    if not 0.70 <= ratio <= 1.40:
        return {
            "meters_per_pixel": wall_scale,
            "source": "wall-thickness",
            "confidence": 0.45,
            "door_candidates": len(widths),
        }
    agreement = 1.0 - min(1.0, abs(1.0 - ratio) / 0.40)
    return {
        "meters_per_pixel": 0.75 * door_scale + 0.25 * wall_scale,
        "source": "door-wall-consensus",
        "confidence": min(0.90, 0.65 + min(len(widths), 30) / 200.0 + agreement * 0.10),
        "door_candidates": len(widths),
        "door_meters_per_pixel": door_scale,
        "wall_meters_per_pixel": wall_scale,
    }


def footprint_pixels(mask_path: Path, typical_wall_thickness_px: float) -> dict[str, Any]:
    """Measure the largest enclosed footprint in a wall mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("A máscara de paredes da pré-análise não foi encontrada.")
    walls = mask > 0
    thickness = _finite_positive(typical_wall_thickness_px) or 15.0
    kernel_size = int(round(thickness))
    kernel_size = max(9, min(65, kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    closed = cv2.morphologyEx(
        walls.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    free = (closed == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    outside_labels: set[int] = set()
    outside_labels.update(int(value) for value in np.unique(labels[0, :]))
    outside_labels.update(int(value) for value in np.unique(labels[-1, :]))
    outside_labels.update(int(value) for value in np.unique(labels[:, 0]))
    outside_labels.update(int(value) for value in np.unique(labels[:, -1]))
    footprint = np.ones_like(free, dtype=np.uint8)
    for label in outside_labels:
        footprint[labels == label] = 0
    component_count, components, component_stats, _ = cv2.connectedComponentsWithStats(
        footprint, connectivity=8
    )
    if component_count <= 1:
        raise ValueError("Não foi possível fechar o contorno principal da planta.")
    largest = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
    area_px = int(component_stats[largest, cv2.CC_STAT_AREA])
    if area_px < mask.size * 0.01:
        raise ValueError("O contorno identificado é pequeno demais para uma planta.")
    selected = components == largest
    return {
        "area_px": area_px,
        "bbox_px": [
            int(component_stats[largest, cv2.CC_STAT_LEFT]),
            int(component_stats[largest, cv2.CC_STAT_TOP]),
            int(component_stats[largest, cv2.CC_STAT_WIDTH]),
            int(component_stats[largest, cv2.CC_STAT_HEIGHT]),
        ],
        "fill_ratio": round(area_px / float(mask.size), 5),
        "mask": selected,
    }


def estimate_local_area(
    *,
    original_path: Path,
    image_path: Path,
    output_dir: Path,
    fallback_canvas_width_m: float,
) -> dict[str, Any]:
    """Run local measurement and return a sanitized pricing/scale inspection."""
    if str(os.environ.get("PLAN_BIM_LOCAL_AREA_ESTIMATION_ENABLED", "true")).lower() in {
        "0", "false", "no",
    }:
        return {
            "status": "unavailable",
            "estimated_area_m2": None,
            "message": "A área não foi estimada; o valor aplicado permanece fixo.",
        }
    detector_dir = output_dir / "measurement_detector"
    result = run_pre_wall_pipeline(
        image_path,
        detector_dir,
        canvas_width_m=fallback_canvas_width_m,
        timeout_seconds=max(30, int(os.environ.get("PLAN_BIM_PREANALYSIS_TIMEOUT_SECONDS", "240"))),
    )
    detection = result.get("detection_scale") or {}
    footprint = footprint_pixels(
        detector_dir / "yolo_wall_mask.png",
        float(detection.get("typical_wall_thickness_px") or 15.0),
    )
    visual_scale = door_wall_scale(result)
    printed_scale = printed_scale_m_per_px(original_path, image_path)
    selected = printed_scale or visual_scale
    if not selected:
        return {
            "status": "unavailable",
            "estimated_area_m2": None,
            "message": "Não foi possível estimar a área com confiança; o valor aplicado permanece fixo.",
        }
    meters_per_pixel = float(selected["meters_per_pixel"])
    area_m2 = footprint["area_px"] * meters_per_pixel * meters_per_pixel
    if not math.isfinite(area_m2) or not 5 <= area_m2 <= 100000:
        raise ValueError("A estimativa local de área ficou fora dos limites esperados.")
    with Image.open(image_path) as image:
        image_width_px = int(image.width)
    public = {
        "status": "estimated",
        "estimated_area_m2": round(area_m2, 1),
        "area_m2": round(area_m2, 1),
        "scale_m_per_px": round(meters_per_pixel, 8),
        "scale_source": str(selected["source"]),
        "scale_confidence": round(float(selected["confidence"]), 3),
        "effective_canvas_width_m": round(image_width_px * meters_per_pixel, 3),
        "door_candidates": int((visual_scale or {}).get("door_candidates") or 0),
        "footprint_area_px": int(footprint["area_px"]),
        "message": "Área estimada localmente para calcular a referência do orçamento.",
    }
    (output_dir / "local_area_inspection.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cv2.imwrite(
        str(output_dir / "local_area_footprint.png"),
        footprint["mask"].astype(np.uint8) * 255,
    )
    return public
