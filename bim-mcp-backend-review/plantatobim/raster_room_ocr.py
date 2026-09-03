"""Room-aware OCR and conservative metric calibration for raster plans.

Wall geometry remains authoritative.  OCR contributes room names, declared
dimensions and areas, while a robust consensus decides whether those clues are
strong enough to replace the user-provided canvas width.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import tempfile
import unicodedata
from typing import Any, Callable

import cv2
import numpy as np

from .cad_raster_ocr import run_windows_ocr


_NUMBER = r"\d{1,5}(?:[.,]\d{1,3})?"
_DIMENSION_PAIR = re.compile(
    rf"(?<!\d)(?P<first>{_NUMBER})\s*(?P<first_unit>mm|cm|m)?\s*"
    rf"[xX×]\s*(?P<second>{_NUMBER})\s*(?P<second_unit>mm|cm|m)?(?!\d)",
    re.IGNORECASE,
)
_AREA = re.compile(
    rf"(?<!\d)(?P<value>{_NUMBER})\s*(?:m\s*[²2]|m2|sqm)(?!\w)",
    re.IGNORECASE,
)
_OBJECT_DIMENSION_CONTEXT = re.compile(
    r"\b(?:porta|door|janela|window|cama|bed|mesa|table|armario|armário|closet)\b",
    re.IGNORECASE,
)

_ROOM_LABELS: tuple[tuple[tuple[str, ...], str, int], ...] = (
    (("cozinha", "kitchen"), "Cozinha", 1),
    (("sala de estar", "estar", "living", "sala"), "Sala", 2),
    (("suite", "suíte", "dormitorio", "dormitório", "bedroom", "quarto"), "Quarto", 3),
    (("banheiro", "banho", "bathroom", "lavabo", "wc", "bwc"), "Banheiro", 4),
    (("entrada", "hall", "foyer"), "Entrada", 5),
    (("deposito", "depósito", "despensa", "storage"), "Depósito", 6),
    (("garagem", "garage"), "Garagem", 7),
    (("lavanderia", "area de servico", "área de serviço", "service"), "Lavanderia", 8),
    (("escritorio", "escritório", "office"), "Escritório", 8),
    (("varanda", "sacada", "terraco", "terraço", "balcony"), "Varanda", 8),
    (("circulacao", "circulação", "corredor", "corridor"), "Circulação", 8),
)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return " ".join(
        "".join(character for character in normalized if not unicodedata.combining(character))
        .lower()
        .split()
    )


def _measurement_to_meters(value: str, unit: str | None) -> float | None:
    try:
        number = float(value.replace(",", "."))
    except ValueError:
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    explicit_unit = str(unit or "").lower()
    if explicit_unit == "mm":
        meters = number / 1000.0
    elif explicit_unit == "cm":
        meters = number / 100.0
    elif explicit_unit == "m":
        meters = number
    elif number >= 1000:
        meters = number / 1000.0
    elif number >= 100:
        meters = number / 100.0
    elif "," in value or "." in value:
        meters = number
    else:
        return None
    return float(meters) if 0.45 <= meters <= 25.0 else None


def _room_label(texts: list[str], room_index: int) -> tuple[str, int]:
    normalized_lines = [_normalize_text(text) for text in texts]
    for aliases, label, category_id in _ROOM_LABELS:
        normalized_aliases = [_normalize_text(alias) for alias in aliases]
        if any(alias in line for alias in normalized_aliases for line in normalized_lines):
            return label, category_id
    return f"Ambiente {room_index}", 8


def _line_bbox(line: dict[str, Any], scale_x: float, scale_y: float) -> dict[str, float]:
    x = float(line.get("x") or 0.0) * scale_x
    y = float(line.get("y") or 0.0) * scale_y
    width = float(line.get("width") or 0.0) * scale_x
    height = float(line.get("height") or 0.0) * scale_y
    return {
        "xmin": x,
        "ymin": y,
        "xmax": x + width,
        "ymax": y + height,
    }


def _parse_room_measurements(lines: list[dict[str, Any]]) -> dict[str, Any]:
    dimension_pairs: list[dict[str, Any]] = []
    areas: list[dict[str, Any]] = []
    for line in lines:
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        if not _OBJECT_DIMENSION_CONTEXT.search(text):
            for match in _DIMENSION_PAIR.finditer(text):
                first = _measurement_to_meters(
                    match.group("first"),
                    match.group("first_unit") or match.group("second_unit"),
                )
                second = _measurement_to_meters(
                    match.group("second"),
                    match.group("second_unit") or match.group("first_unit"),
                )
                if first is None or second is None:
                    continue
                dimension_pairs.append({
                    "values_m": [round(first, 5), round(second, 5)],
                    "text": match.group(0).strip(),
                    "line_text": text,
                    "bbox_px": dict(line["bbox_px"]),
                    "position_px": list(line["position_px"]),
                })
        for match in _AREA.finditer(text):
            try:
                area = float(match.group("value").replace(",", "."))
            except ValueError:
                continue
            if 0.5 <= area <= 1000.0:
                areas.append({
                    "value_m2": round(area, 5),
                    "text": match.group(0).strip(),
                    "line_text": text,
                    "bbox_px": dict(line["bbox_px"]),
                    "position_px": list(line["position_px"]),
                })
    return {
        "dimension_pairs": dimension_pairs,
        "areas": areas,
    }


def extract_room_regions(
    axes: list[dict[str, Any]],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Polygonize enclosed free-space regions from continuous wall axes."""

    height, width = int(image_shape[0]), int(image_shape[1])
    wall_mask = np.zeros((height, width), dtype=np.uint8)
    thicknesses = [
        max(2.0, float(axis.get("thickness", 4.0)))
        for axis in axes
        if axis.get("orientation") in {"horizontal", "vertical"}
    ]
    typical_thickness = float(np.median(thicknesses)) if thicknesses else 4.0
    endpoint_extension = max(2, int(round(typical_thickness * 0.8)))
    for axis in axes:
        orientation = axis.get("orientation")
        if orientation not in {"horizontal", "vertical"}:
            continue
        fixed = int(round(float(axis["fixed"])))
        start = int(round(float(axis["start"]))) - endpoint_extension
        end = int(round(float(axis["end"]))) + endpoint_extension
        thickness = max(2, int(round(float(axis.get("thickness", typical_thickness)))))
        if orientation == "horizontal":
            first, second = (start, fixed), (end, fixed)
        else:
            first, second = (fixed, start), (fixed, end)
        cv2.line(wall_mask, first, second, 255, thickness, cv2.LINE_8)

    join_size = max(3, int(round(typical_thickness * 0.65)))
    if join_size % 2 == 0:
        join_size += 1
    wall_mask = cv2.morphologyEx(
        wall_mask,
        cv2.MORPH_CLOSE,
        np.ones((join_size, join_size), dtype=np.uint8),
    )
    free_space = np.asarray(wall_mask == 0, dtype=np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(free_space, 8)
    minimum_area = max(180, int(round(height * width * 0.0025)))
    minimum_side = max(8, int(round(min(height, width) * 0.025)))
    rooms: list[dict[str, Any]] = []
    for component_index in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[component_index])
        touches_border = (
            x <= 0
            or y <= 0
            or x + component_width >= width
            or y + component_height >= height
        )
        if (
            touches_border
            or area < minimum_area
            or component_width < minimum_side
            or component_height < minimum_side
        ):
            continue
        component = np.asarray(labels == component_index, dtype=np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.008), True)
        points = [[float(point[0][0]), float(point[0][1])] for point in polygon]
        if len(points) < 3:
            continue
        rooms.append({
            "id": f"ROOM-OCR-{len(rooms) + 1:03d}",
            "component_index": component_index,
            "points_px": points,
            "contour_px": contour.reshape(-1, 2).astype(np.float32).tolist(),
            "bbox_px": [float(x), float(y), float(x + component_width), float(y + component_height)],
            "centroid_px": [float(centroids[component_index][0]), float(centroids[component_index][1])],
            "area_px": float(area),
            "ocr_lines": [],
        })
    rooms.sort(key=lambda room: (room["bbox_px"][1], room["bbox_px"][0]))
    for index, room in enumerate(rooms, 1):
        room["id"] = f"ROOM-OCR-{index:03d}"
    return rooms, {
        "wall_axis_count": len(axes),
        "room_count": len(rooms),
        "typical_wall_thickness_px": round(typical_thickness, 4),
        "minimum_room_area_px": minimum_area,
    }


def associate_ocr_lines_to_rooms(
    rooms: list[dict[str, Any]],
    ocr_result: dict[str, Any],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> dict[str, Any]:
    height, width = int(image_shape[0]), int(image_shape[1])
    ocr_width = max(1.0, float(ocr_result.get("width") or width))
    ocr_height = max(1.0, float(ocr_result.get("height") or height))
    scale_x, scale_y = width / ocr_width, height / ocr_height
    matched = 0
    unmatched = 0
    for source_line in ocr_result.get("lines") or []:
        text = str(source_line.get("text") or "").strip()
        if not text:
            continue
        bbox = _line_bbox(source_line, scale_x, scale_y)
        center = [
            (bbox["xmin"] + bbox["xmax"]) / 2.0,
            (bbox["ymin"] + bbox["ymax"]) / 2.0,
        ]
        line = {
            "text": text,
            "bbox_px": bbox,
            "position_px": center,
        }
        containing: list[tuple[float, dict[str, Any]]] = []
        for room in rooms:
            contour = np.asarray(room["contour_px"], dtype=np.float32).reshape(-1, 1, 2)
            distance = cv2.pointPolygonTest(contour, (center[0], center[1]), True)
            if distance >= 0:
                containing.append((distance, room))
        if not containing:
            unmatched += 1
            continue
        _, room = max(containing, key=lambda item: item[0])
        room["ocr_lines"].append(line)
        matched += 1

    for index, room in enumerate(rooms, 1):
        texts = [line["text"] for line in room["ocr_lines"]]
        label, category_id = _room_label(texts, index)
        room["label"] = label
        room["category_id"] = category_id
        room.update(_parse_room_measurements(room["ocr_lines"]))
    return {
        "line_count": len(ocr_result.get("lines") or []),
        "matched_line_count": matched,
        "unmatched_line_count": unmatched,
    }


def _append_unique_room_line(room: dict[str, Any], line: dict[str, Any]) -> bool:
    normalized = _normalize_text(str(line.get("text") or ""))
    if not normalized:
        return False
    for existing in room.get("ocr_lines") or []:
        if _normalize_text(str(existing.get("text") or "")) == normalized:
            return False
    room.setdefault("ocr_lines", []).append(line)
    return True


def supplement_ocr_from_room_crops(
    image_path: str | Path,
    rooms: list[dict[str, Any]],
    *,
    runner: Callable[[str | Path], dict[str, Any]] = run_windows_ocr,
) -> dict[str, Any]:
    """Upscale each enclosed room so tiny names and areas become legible."""

    try:
        encoded_image = np.fromfile(str(Path(image_path)), dtype=np.uint8)
    except OSError:
        encoded_image = np.asarray([], dtype=np.uint8)
    image = (
        cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        if encoded_image.size
        else None
    )
    if image is None:
        return {
            "room_crop_count": 0,
            "room_crop_line_count": 0,
            "room_crop_matched_line_count": 0,
            "room_crop_failures": 1,
        }
    image_height, image_width = image.shape[:2]
    attempted = 0
    recognized = 0
    appended = 0
    failures = 0
    with tempfile.TemporaryDirectory(prefix="plant2bim_room_ocr_") as temporary:
        temporary_dir = Path(temporary)
        for room_index, room in enumerate(rooms, 1):
            x0, y0, x1, y1 = [int(round(value)) for value in room["bbox_px"]]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(image_width, x1), min(image_height, y1)
            if x1 - x0 < 12 or y1 - y0 < 12:
                continue
            crop = image[y0:y1, x0:x1]
            long_side = max(crop.shape[:2])
            upscale = float(np.clip(1200.0 / max(1, long_side), 3.0, 12.0))
            resized = cv2.resize(
                crop,
                None,
                fx=upscale,
                fy=upscale,
                interpolation=cv2.INTER_CUBIC,
            )
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (0, 0), 1.15)
            enhanced = cv2.addWeighted(gray, 1.65, blur, -0.65, 0)
            padding = 24
            prepared = cv2.copyMakeBorder(
                enhanced,
                padding,
                padding,
                padding,
                padding,
                cv2.BORDER_CONSTANT,
                value=255,
            )
            crop_path = temporary_dir / f"room_{room_index:03d}.png"
            ok, encoded = cv2.imencode(".png", prepared)
            if not ok:
                failures += 1
                continue
            encoded.tofile(str(crop_path))
            attempted += 1
            try:
                result = runner(crop_path)
            except Exception:
                failures += 1
                continue
            result_width = max(1.0, float(result.get("width") or prepared.shape[1]))
            result_height = max(1.0, float(result.get("height") or prepared.shape[0]))
            coordinate_scale_x = prepared.shape[1] / result_width
            coordinate_scale_y = prepared.shape[0] / result_height
            room_contour = np.asarray(room["contour_px"], dtype=np.float32).reshape(-1, 1, 2)
            for source_line in result.get("lines") or []:
                recognized += 1
                prepared_bbox = _line_bbox(
                    source_line,
                    coordinate_scale_x,
                    coordinate_scale_y,
                )
                bbox = {
                    "xmin": x0 + (prepared_bbox["xmin"] - padding) / upscale,
                    "ymin": y0 + (prepared_bbox["ymin"] - padding) / upscale,
                    "xmax": x0 + (prepared_bbox["xmax"] - padding) / upscale,
                    "ymax": y0 + (prepared_bbox["ymax"] - padding) / upscale,
                }
                center = [
                    (bbox["xmin"] + bbox["xmax"]) / 2.0,
                    (bbox["ymin"] + bbox["ymax"]) / 2.0,
                ]
                if cv2.pointPolygonTest(room_contour, (center[0], center[1]), False) < 0:
                    continue
                line = {
                    "text": str(source_line.get("text") or "").strip(),
                    "bbox_px": bbox,
                    "position_px": center,
                    "source": "room-crop-upscaled",
                }
                if _append_unique_room_line(room, line):
                    appended += 1
    return {
        "room_crop_count": attempted,
        "room_crop_line_count": recognized,
        "room_crop_matched_line_count": appended,
        "room_crop_failures": failures,
    }


def _weighted_median(items: list[dict[str, Any]]) -> float:
    ordered = sorted(items, key=lambda item: float(item["scale_m_per_px"]))
    total_weight = sum(float(item["weight"]) for item in ordered)
    cursor = 0.0
    for item in ordered:
        cursor += float(item["weight"])
        if cursor >= total_weight / 2.0:
            return float(item["scale_m_per_px"])
    return float(ordered[-1]["scale_m_per_px"])


def solve_room_metric_scale(
    rooms: list[dict[str, Any]],
    *,
    image_width: int,
    fallback_pixel_m: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for room in rooms:
        x0, y0, x1, y1 = [float(value) for value in room["bbox_px"]]
        room_width_px = max(1.0, x1 - x0)
        room_height_px = max(1.0, y1 - y0)
        for pair in room.get("dimension_pairs") or []:
            first, second = [float(value) for value in pair["values_m"]]
            assignments = []
            for width_m, height_m, swapped in (
                (first, second, False),
                (second, first, True),
            ):
                horizontal_scale = width_m / room_width_px
                vertical_scale = height_m / room_height_px
                mean_scale = math.sqrt(horizontal_scale * vertical_scale)
                axis_error = abs(horizontal_scale - vertical_scale) / max(mean_scale, 1e-9)
                assignments.append((axis_error, mean_scale, swapped, horizontal_scale, vertical_scale))
            axis_error, scale, swapped, horizontal_scale, vertical_scale = min(assignments)
            if axis_error > 0.28:
                continue
            candidates.append({
                "room_id": room["id"],
                "kind": "dimension-pair",
                "scale_m_per_px": scale,
                "weight": max(0.55, 1.2 - axis_error * 2.0),
                "axis_error": axis_error,
                "swapped": swapped,
                "horizontal_scale": horizontal_scale,
                "vertical_scale": vertical_scale,
                "source_text": pair["line_text"],
            })
        for area_record in room.get("areas") or []:
            scale = math.sqrt(float(area_record["value_m2"]) / max(1.0, float(room["area_px"])))
            candidates.append({
                "room_id": room["id"],
                "kind": "declared-area",
                "scale_m_per_px": scale,
                "weight": 0.62,
                "source_text": area_record["line_text"],
            })

    candidates = [
        item for item in candidates
        if 2.0 <= float(item["scale_m_per_px"]) * image_width <= 300.0
    ]
    if not candidates:
        return {
            "applied": False,
            "reason": "no-reliable-room-measurements",
            "pixel_m": fallback_pixel_m,
            "canvas_width_m": fallback_pixel_m * image_width,
            "confidence": 0.0,
            "candidate_count": 0,
            "inlier_count": 0,
            "candidates": [],
        }

    initial = _weighted_median(candidates)
    inliers = [
        item for item in candidates
        if abs(float(item["scale_m_per_px"]) / initial - 1.0) <= 0.20
    ]
    if not inliers:
        inliers = [min(candidates, key=lambda item: abs(float(item["scale_m_per_px"]) - initial))]
    scale = _weighted_median(inliers)
    deviations = [abs(float(item["scale_m_per_px"]) / scale - 1.0) for item in inliers]
    dispersion = float(np.median(deviations)) if deviations else 1.0
    strong_pair = any(
        item["kind"] == "dimension-pair" and float(item.get("axis_error", 1.0)) <= 0.12
        for item in inliers
    )
    independent_rooms = len({str(item["room_id"]) for item in inliers})
    corroborated = len(inliers) >= 2 and dispersion <= 0.12
    applied = bool(strong_pair or corroborated)
    confidence = 0.0
    if applied:
        confidence = min(
            0.98,
            0.68
            + (0.12 if strong_pair else 0.0)
            + min(0.12, 0.04 * (len(inliers) - 1))
            + min(0.06, 0.03 * (independent_rooms - 1))
            - min(0.18, dispersion),
        )
    return {
        "applied": applied,
        "reason": "room-ocr-consensus" if applied else "insufficient-room-ocr-consensus",
        "pixel_m": scale if applied else fallback_pixel_m,
        "candidate_pixel_m": scale,
        "canvas_width_m": (scale if applied else fallback_pixel_m) * image_width,
        "candidate_canvas_width_m": scale * image_width,
        "confidence": round(max(0.0, confidence), 4),
        "candidate_count": len(candidates),
        "inlier_count": len(inliers),
        "independent_room_count": independent_rooms,
        "dispersion": round(dispersion, 6),
        "candidates": [
            {
                **item,
                "scale_m_per_px": round(float(item["scale_m_per_px"]), 8),
                "weight": round(float(item["weight"]), 4),
                **(
                    {"axis_error": round(float(item["axis_error"]), 6)}
                    if "axis_error" in item
                    else {}
                ),
            }
            for item in candidates
        ],
    }


def analyze_room_ocr(
    image_path: str | Path,
    axes: list[dict[str, Any]],
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    fallback_pixel_m: float,
    runner: Callable[[str | Path], dict[str, Any]] = run_windows_ocr,
    use_room_crops: bool = True,
) -> dict[str, Any]:
    rooms, room_diagnostic = extract_room_regions(axes, image_shape)
    try:
        ocr_result = runner(image_path)
        ocr_diagnostic = {
            "status": "ok",
            "engine": str(ocr_result.get("engine") or "unknown"),
            "language": ocr_result.get("language"),
            "cache_hit": bool(ocr_result.get("cache_hit")),
        }
        association = associate_ocr_lines_to_rooms(rooms, ocr_result, image_shape)
        crop_diagnostic = (
            supplement_ocr_from_room_crops(image_path, rooms, runner=runner)
            if use_room_crops and rooms
            else {
                "room_crop_count": 0,
                "room_crop_line_count": 0,
                "room_crop_matched_line_count": 0,
                "room_crop_failures": 0,
            }
        )
        association.update(crop_diagnostic)
        for index, room in enumerate(rooms, 1):
            texts = [line["text"] for line in room["ocr_lines"]]
            room["label"], room["category_id"] = _room_label(texts, index)
            room.update(_parse_room_measurements(room["ocr_lines"]))
    except Exception as exc:
        ocr_diagnostic = {
            "status": "failed",
            "engine": "windows-media-ocr",
            "error": str(exc),
        }
        association = {
            "line_count": 0,
            "matched_line_count": 0,
            "unmatched_line_count": 0,
            "room_crop_count": 0,
            "room_crop_line_count": 0,
            "room_crop_matched_line_count": 0,
            "room_crop_failures": 0,
        }
        for index, room in enumerate(rooms, 1):
            room["label"], room["category_id"] = _room_label([], index)
            room.update({"dimension_pairs": [], "areas": []})

    scale = solve_room_metric_scale(
        rooms,
        image_width=int(image_shape[1]),
        fallback_pixel_m=fallback_pixel_m,
    )
    for room in rooms:
        room.pop("contour_px", None)
        room.pop("component_index", None)
    return {
        "rooms": rooms,
        "scale": scale,
        "room_detection": room_diagnostic,
        "ocr": {**ocr_diagnostic, **association},
    }
