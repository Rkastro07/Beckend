"""Detect openings before wall vectorization and reconcile them afterwards.

The detector first sees only the 2D building crop.  Wall geometry is computed
afterwards and is used only to reject off-wall proposals, merge duplicates and
produce editor-compatible opening segments.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from plantatobim.raster_2d_import import detect_wall_regions_2d, vectorize_floorplan_2d


DOOR_COLOUR = (35, 190, 55)
WINDOW_COLOUR = (235, 135, 25)
WALL_COLOUR = (0, 155, 245)
WALL_MASK_COLOUR = (30, 70, 245)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Unable to encode {path}")
    encoded.tofile(str(path))


def detect_building_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find thick, long ink while ignoring dimensions, text and view arrows."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Thin CAD drawings often have no thick wall core, but their architectural
    # contours enclose large regions. Join overlapping contour envelopes while
    # discarding sparse dimension lines.
    contour_binary = (gray < 180).astype(np.uint8) * 255
    contour_binary = cv2.morphologyEx(
        contour_binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    raw_contours = cv2.findContours(
        contour_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    contour_boxes: list[tuple[int, int, int, int, float]] = []
    image_area = float(width * height)
    for contour in raw_contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area = float(box_width * box_height)
        fill = float(cv2.contourArea(contour)) / max(1.0, box_area)
        if box_area / image_area >= 0.02 and fill >= 0.045:
            contour_boxes.append((x, y, x + box_width, y + box_height, float(cv2.contourArea(contour))))

    contour_groups: list[list[tuple[int, int, int, int, float]]] = []
    for box in sorted(contour_boxes, key=lambda item: item[4], reverse=True):
        matching_groups = []
        for group in contour_groups:
            if any(
                box[0] <= other[2] + 5
                and box[2] + 5 >= other[0]
                and box[1] <= other[3] + 5
                and box[3] + 5 >= other[1]
                for other in group
            ):
                matching_groups.append(group)
        if not matching_groups:
            contour_groups.append([box])
            continue
        primary = matching_groups[0]
        primary.append(box)
        for extra in matching_groups[1:]:
            primary.extend(extra)
            contour_groups.remove(extra)

    contour_candidate: tuple[int, int, int, int] | None = None
    if contour_groups:
        def group_bounds(
            group: list[tuple[int, int, int, int, float]],
        ) -> tuple[int, int, int, int]:
            return (
                min(item[0] for item in group),
                min(item[1] for item in group),
                max(item[2] for item in group),
                max(item[3] for item in group),
            )

        def box_area(box: tuple[int, int, int, int]) -> float:
            return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))

        def group_score(group: list[tuple[int, int, int, int, float]]) -> float:
            bounds = group_bounds(group)
            return box_area(bounds) + sum(item[4] for item in group) * 0.15

        best_group = max(contour_groups, key=group_score)
        best_bounds = group_bounds(best_group)
        best_area = box_area(best_bounds)

        # A planta pode ter alas separadas por circulacoes abertas. Selecionar
        # apenas o maior componente corta metade do pavimento. Agregamos grupos
        # arquitetonicos relevantes que compartilham a mesma faixa horizontal
        # ou vertical com o grupo principal; blocos de legenda isolados ficam de
        # fora por tamanho e falta de sobreposicao.
        selected_bounds: list[tuple[int, int, int, int]] = []
        for group in contour_groups:
            bounds = group_bounds(group)
            area = box_area(bounds)
            horizontal_overlap = interval_overlap(
                float(bounds[0]), float(bounds[2]),
                float(best_bounds[0]), float(best_bounds[2]),
            ) / max(1.0, min(bounds[2] - bounds[0], best_bounds[2] - best_bounds[0]))
            vertical_overlap = interval_overlap(
                float(bounds[1]), float(bounds[3]),
                float(best_bounds[1]), float(best_bounds[3]),
            ) / max(1.0, min(bounds[3] - bounds[1], best_bounds[3] - best_bounds[1]))
            relevant_size = area >= max(image_area * 0.018, best_area * 0.07)
            aligned_with_plan = max(horizontal_overlap, vertical_overlap) >= 0.20
            if group is best_group or (relevant_size and aligned_with_plan):
                selected_bounds.append(bounds)

        if selected_bounds:
            pad = max(1, int(round(min(width, height) * 0.002)))
            contour_candidate = (
                max(0, min(item[0] for item in selected_bounds) - pad),
                max(0, min(item[1] for item in selected_bounds) - pad),
                min(width, max(item[2] for item in selected_bounds) + pad),
                min(height, max(item[3] for item in selected_bounds) + pad),
            )
            if box_area(contour_candidate) / image_area >= 0.15:
                return contour_candidate

    dark = (gray < 135).astype(np.uint8)
    distance = cv2.distanceTransform(dark, cv2.DIST_L2, 3)
    thick = (distance >= 1.35).astype(np.uint8) * 255
    horizontal_size = max(7, int(round(width * 0.018)))
    vertical_size = max(7, int(round(height * 0.018)))
    horizontal = cv2.morphologyEx(
        thick,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1)),
    )
    vertical = cv2.morphologyEx(
        thick,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size)),
    )
    structural = cv2.bitwise_or(horizontal, vertical)
    if np.count_nonzero(structural) < 40:
        return 0, 0, width, height

    # Join wall fragments across normal door/window gaps. The largest resulting
    # component is the building; isolated title text and view arrows stay out.
    join_width = max(25, int(round(min(width, height) * 0.12)))
    join_height = max(17, int(round(min(width, height) * 0.075)))
    if join_width % 2 == 0:
        join_width += 1
    if join_height % 2 == 0:
        join_height += 1
    connected = cv2.morphologyEx(
        structural,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (join_width, join_height)),
    )
    contours = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return 0, 0, width, height
    building = max(contours, key=cv2.contourArea)
    x, y, box_width, box_height = cv2.boundingRect(building)
    pad = max(1, int(round(min(width, height) * 0.002)))
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(width, x + box_width + pad)
    bottom = min(height, y + box_height + pad)
    if (right - left) * (bottom - top) < width * height * 0.12:
        return 0, 0, width, height
    return left, top, right, bottom


def box_orientation(box: list[float]) -> str:
    return "horizontal" if box[2] - box[0] >= box[3] - box[1] else "vertical"


def interval_overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def boxes_match(first: dict[str, Any], second: dict[str, Any]) -> bool:
    a = first["box_crop_px"]
    b = second["box_crop_px"]
    orientation = box_orientation(a)
    if orientation != box_orientation(b):
        return False
    if orientation == "horizontal":
        a_major, b_major = (a[0], a[2]), (b[0], b[2])
        a_minor, b_minor = (a[1], a[3]), (b[1], b[3])
    else:
        a_major, b_major = (a[1], a[3]), (b[1], b[3])
        a_minor, b_minor = (a[0], a[2]), (b[0], b[2])
    major_overlap = interval_overlap(*a_major, *b_major)
    minimum_major = max(1.0, min(a_major[1] - a_major[0], b_major[1] - b_major[0]))
    major_centres = abs(sum(a_major) / 2 - sum(b_major) / 2)
    major_limit = max(a_major[1] - a_major[0], b_major[1] - b_major[0]) * 0.58
    minor_centres = abs(sum(a_minor) / 2 - sum(b_minor) / 2)
    minor_limit = max(16.0, max(a_minor[1] - a_minor[0], b_minor[1] - b_minor[0]) * 1.6)
    return (
        minor_centres <= minor_limit
        and (major_overlap / minimum_major >= 0.18 or major_centres <= major_limit)
    )


def predict_multiscale(
    model: YOLO,
    crop: np.ndarray,
    sizes: list[int],
    confidence: float,
    device: str,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for image_size in sizes:
        result = model.predict(
            crop,
            conf=confidence,
            iou=0.42,
            imgsz=image_size,
            device=device,
            max_det=120,
            verbose=False,
        )[0]
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "class": result.names[class_id],
                "class_id": class_id,
                "confidence": float(box.conf.item()),
                "scale": image_size,
                "box_crop_px": [float(value) for value in box.xyxy[0].tolist()],
            })
    return detections


def predict_wall_segmentation(
    model: YOLO,
    crop: np.ndarray,
    *,
    image_size: int,
    confidence: float,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aggregate wall-instance masks into one geometry-ready binary mask."""

    height, width = crop.shape[:2]
    result = model.predict(
        crop,
        conf=confidence,
        iou=0.50,
        imgsz=image_size,
        device=device,
        max_det=300,
        retina_masks=True,
        verbose=False,
    )[0]
    combined = np.zeros((height, width), dtype=np.uint8)
    confidences: list[float] = []
    if result.masks is not None and result.boxes is not None:
        masks = result.masks.data.cpu().numpy()
        confidences = [float(value) for value in result.boxes.conf.cpu().numpy()]
        for mask in masks:
            if mask.shape != combined.shape:
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
            combined[mask >= 0.50] = 255
    if np.count_nonzero(combined):
        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
    return combined, {
        "instance_count": len(confidences),
        "mean_confidence": round(float(np.mean(confidences)), 6) if confidences else 0.0,
        "maximum_confidence": round(max(confidences), 6) if confidences else 0.0,
        "mask_pixels": int(np.count_nonzero(combined)),
    }


def _short_foreground_runs(mask: np.ndarray, *, axis: int, maximum: int) -> list[int]:
    """Collect wall-thickness-sized runs without assuming a metric scale."""

    runs: list[int] = []
    line_count = mask.shape[1] if axis == 0 else mask.shape[0]
    for index in range(line_count):
        line = mask[:, index] if axis == 0 else mask[index, :]
        padded = np.pad(np.asarray(line > 0, dtype=np.int8), (1, 1))
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        lengths = ends - starts
        runs.extend(int(value) for value in lengths if 3 <= value <= maximum)
    return runs


def estimate_visual_detection_scale(
    crop: np.ndarray,
    wall_mask: np.ndarray | None,
    *,
    nominal_wall_thickness_m: float = 0.20,
) -> dict[str, Any]:
    """Calibrate morphology from observed wall thickness, not user scale.

    ``canvas_width_m`` is a metric/export parameter.  Feeding it into visual
    morphology made the same pixels produce different topology at 12 m and
    20 m.  This internal nominal extent is derived only from the raster and is
    therefore stable when the user changes the final metric width.
    """

    height, width = crop.shape[:2]
    calibration_mask = wall_mask
    method = "yolo-wall-mask-runs"
    if calibration_mask is None or not np.count_nonzero(calibration_mask):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        calibration_mask = np.asarray(gray <= 138, dtype=np.uint8) * 255
        calibration_mask = cv2.morphologyEx(
            calibration_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
        method = "dark-mass-runs"

    maximum_run = max(12, int(round(min(height, width) * 0.12)))
    runs = [
        *_short_foreground_runs(calibration_mask, axis=0, maximum=maximum_run),
        *_short_foreground_runs(calibration_mask, axis=1, maximum=maximum_run),
    ]
    if runs:
        histogram = np.bincount(np.asarray(runs, dtype=np.int32))
        dominant = int(np.argmax(histogram[3:]) + 3)
        band = max(2, int(round(dominant * 0.35)))
        neighbourhood = [value for value in runs if abs(value - dominant) <= band]
        typical_thickness_px = float(np.median(neighbourhood or runs))
    else:
        dominant = 0
        typical_thickness_px = max(3.0, min(height, width) * 0.015)
        method = f"{method}-fallback"

    pixel_m = nominal_wall_thickness_m / max(1.0, typical_thickness_px)
    raw_extent_m = max(height, width) * pixel_m
    detection_canvas_extent_m = float(np.clip(raw_extent_m, 4.0, 40.0))
    return {
        "mode": "visual-auto-wall-thickness",
        "method": method,
        "sample_count": len(runs),
        "dominant_wall_run_px": dominant,
        "typical_wall_thickness_px": round(typical_thickness_px, 4),
        "nominal_wall_thickness_m": nominal_wall_thickness_m,
        "detection_pixel_m": round(
            detection_canvas_extent_m / max(1, max(height, width)),
            8,
        ),
        "detection_canvas_extent_m": round(detection_canvas_extent_m, 6),
        "extent_was_clamped": not math.isclose(raw_extent_m, detection_canvas_extent_m),
    }


def walls_from_segmentation_mask(
    mask: np.ndarray,
    *,
    canvas_width_m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reuse the tested axis vectorizer on a semantic wall-only raster."""

    rendered = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    rendered[mask > 0] = 0
    walls, _, diagnostic = detect_wall_regions_2d(
        rendered,
        canvas_width_m=canvas_width_m,
    )
    return merge_collinear_walls(walls), diagnostic


def fuse_wall_sets(
    geometry_walls: list[dict[str, Any]],
    yolo_walls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create one deduplicated carrier set supported by 2D and/or YOLO."""

    tagged = [
        {**wall, "_fusion_origin": "geometry"}
        for wall in geometry_walls
    ] + [
        {**wall, "_fusion_origin": "yolo"}
        for wall in yolo_walls
    ]
    fused = merge_collinear_walls(tagged)
    for wall in fused:
        supports: list[str] = []
        for origin, source_walls in (
            ("geometry", geometry_walls),
            ("yolo", yolo_walls),
        ):
            for source in source_walls:
                if source.get("orientation") != wall.get("orientation"):
                    continue
                fixed_tolerance = max(
                    4.0,
                    float(source.get("thickness", 6.0)) * 0.9,
                    float(wall.get("thickness", 6.0)) * 0.9,
                )
                if abs(float(source["fixed"]) - float(wall["fixed"])) > fixed_tolerance:
                    continue
                overlap = interval_overlap(
                    float(source["start"]),
                    float(source["end"]),
                    float(wall["start"]),
                    float(wall["end"]),
                )
                source_length = max(1.0, float(source["end"]) - float(source["start"]))
                if overlap / source_length >= 0.18:
                    supports.append(origin)
                    break
        supports = sorted(set(supports))
        wall["fusion_sources"] = supports
        if supports == ["geometry", "yolo"]:
            wall["source"] = "2d-yolo-wall-fusion"
        elif supports == ["geometry"]:
            wall["source"] = "2d-wall-fusion-recovery"
        else:
            wall["source"] = "yolo-wall-fusion-recovery"

    def connected(first: dict[str, Any], second: dict[str, Any]) -> bool:
        tolerance = max(
            10.0,
            float(first.get("thickness", 6.0)) * 2.5,
            float(second.get("thickness", 6.0)) * 2.5,
        )
        if first["orientation"] == second["orientation"]:
            if abs(float(first["fixed"]) - float(second["fixed"])) > tolerance * 0.45:
                return False
            gap = max(
                float(first["start"]) - float(second["end"]),
                float(second["start"]) - float(first["end"]),
                0.0,
            )
            return gap <= tolerance
        horizontal, vertical = (
            (first, second)
            if first["orientation"] == "horizontal"
            else (second, first)
        )
        return (
            float(horizontal["start"]) - tolerance
            <= float(vertical["fixed"])
            <= float(horizontal["end"]) + tolerance
            and float(vertical["start"]) - tolerance
            <= float(horizontal["fixed"])
            <= float(vertical["end"]) + tolerance
        )

    reliable_network = [wall for wall in fused if "yolo" in wall["fusion_sources"]]
    network_coordinates = [
        coordinate
        for wall in reliable_network
        for coordinate in (float(wall["fixed"]), float(wall["start"]), float(wall["end"]))
    ]
    network_span = max(network_coordinates, default=360.0) - min(network_coordinates, default=0.0)
    minimum_recovery_length = max(18.0, network_span * 0.05)
    accepted: list[dict[str, Any]] = []
    rejected_geometry_only = 0
    for wall in fused:
        if wall["fusion_sources"] != ["geometry"]:
            accepted.append(wall)
            continue
        length = float(wall["end"]) - float(wall["start"])
        if length >= minimum_recovery_length and any(
            connected(wall, neighbour) for neighbour in reliable_network
        ):
            accepted.append(wall)
        else:
            rejected_geometry_only += 1

    both_count = sum(wall["fusion_sources"] == ["geometry", "yolo"] for wall in accepted)
    geometry_only_count = sum(wall["fusion_sources"] == ["geometry"] for wall in accepted)
    yolo_only_count = sum(wall["fusion_sources"] == ["yolo"] for wall in accepted)
    return accepted, {
        "geometry_input_count": len(geometry_walls),
        "yolo_input_count": len(yolo_walls),
        "fused_count": len(accepted),
        "supported_by_both": both_count,
        "geometry_only": geometry_only_count,
        "yolo_only": yolo_only_count,
        "rejected_isolated_geometry_only": rejected_geometry_only,
        "minimum_recovery_length_px": round(minimum_recovery_length, 3),
    }


def draw_wall_mask(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coloured = crop.copy()
    coloured[mask > 0] = WALL_MASK_COLOUR
    return cv2.addWeighted(crop, 0.55, coloured, 0.45, 0.0)


def consolidate_detections(
    detections: list[dict[str, Any]],
    *,
    minimum_confidence: float,
    minimum_scale_support: int,
) -> list[dict[str, Any]]:
    clusters: list[list[dict[str, Any]]] = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        matching = next(
            (cluster for cluster in clusters if boxes_match(cluster[0], detection)),
            None,
        )
        if matching is None:
            clusters.append([detection])
        else:
            matching.append(detection)

    consolidated: list[dict[str, Any]] = []
    for cluster in clusters:
        scales = sorted({int(item["scale"]) for item in cluster})
        strongest = max(cluster, key=lambda item: item["confidence"])
        if strongest["confidence"] < minimum_confidence or len(scales) < minimum_scale_support:
            continue
        class_scores: dict[str, float] = {}
        for item in cluster:
            class_scores[item["class"]] = class_scores.get(item["class"], 0.0) + item["confidence"]
        chosen_class = max(class_scores, key=class_scores.get)
        representative = max(
            (item for item in cluster if item["class"] == chosen_class),
            key=lambda item: item["confidence"],
        )
        consolidated.append({
            "type": chosen_class,
            "confidence": round(float(strongest["confidence"]), 6),
            "scale_support": scales,
            "vote_scores": {key: round(value, 6) for key, value in class_scores.items()},
            "box_crop_px": [round(float(value), 3) for value in representative["box_crop_px"]],
            "orientation": box_orientation(representative["box_crop_px"]),
            "raw_count": len(cluster),
            "source": "yolo-multiscale-consensus",
        })
    return sorted(consolidated, key=lambda item: item["confidence"], reverse=True)


def merge_collinear_walls(walls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for orientation in ("horizontal", "vertical"):
        candidates = sorted(
            (wall for wall in walls if wall.get("orientation") == orientation),
            key=lambda wall: (float(wall["fixed"]), float(wall["start"])),
        )
        fixed_groups: list[list[dict[str, Any]]] = []
        for wall in candidates:
            tolerance = max(3.0, float(wall.get("thickness", 6.0)) * 0.7)
            group = next(
                (
                    current
                    for current in fixed_groups
                    if abs(np.median([float(item["fixed"]) for item in current]) - float(wall["fixed"]))
                    <= tolerance
                ),
                None,
            )
            if group is None:
                fixed_groups.append([wall])
            else:
                group.append(wall)
        for group in fixed_groups:
            fixed = float(np.median([float(item["fixed"]) for item in group]))
            thickness = float(np.median([float(item.get("thickness", 6.0)) for item in group]))
            intervals = sorted((float(item["start"]), float(item["end"])) for item in group)
            maximum_gap = max(32.0, thickness * 7.0)
            current_start, current_end = intervals[0]
            source_count = 1
            for start, end in intervals[1:]:
                if start - current_end <= maximum_gap:
                    current_end = max(current_end, end)
                    source_count += 1
                else:
                    merged.append({
                        "orientation": orientation,
                        "fixed": fixed,
                        "start": current_start,
                        "end": current_end,
                        "thickness": thickness,
                        "source_segment_count": source_count,
                    })
                    current_start, current_end, source_count = start, end, 1
            merged.append({
                "orientation": orientation,
                "fixed": fixed,
                "start": current_start,
                "end": current_end,
                "thickness": thickness,
                "source_segment_count": source_count,
            })
    return merged


def detect_single_line_walls(image: np.ndarray) -> list[dict[str, Any]]:
    """Fallback axes for thin CAD/schematic plans without thick wall regions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray < 185).astype(np.uint8) * 255
    minimum_dimension = min(gray.shape)
    minimum_length = max(16, int(round(minimum_dimension * 0.11)))
    walls: list[dict[str, Any]] = []
    for orientation, kernel_shape in (
        ("horizontal", (minimum_length, 1)),
        ("vertical", (1, minimum_length)),
    ):
        lines = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
        for index in range(1, count):
            x, y, width, height, area = map(int, stats[index])
            length = width if orientation == "horizontal" else height
            if length < minimum_length or area < minimum_length:
                continue
            if orientation == "horizontal":
                fixed, start, end, thickness = y + height / 2, x, x + width, max(2.0, min(5.0, float(height)))
            else:
                fixed, start, end, thickness = x + width / 2, y, y + height, max(2.0, min(5.0, float(width)))
            walls.append({
                "orientation": orientation,
                "fixed": fixed,
                "start": float(start),
                "end": float(end),
                "thickness": thickness,
                "source": "single-line-morphology",
            })
    return walls


def collapse_parallel_wall_faces(
    walls: list[dict[str, Any]],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Turn two close outline strokes into one wall axis with real thickness."""

    minimum_dimension = min(int(image_shape[0]), int(image_shape[1]))
    maximum_separation = max(8.0, minimum_dimension * 0.045)
    candidates: list[tuple[float, int, int]] = []
    for first_index, first in enumerate(walls):
        if float(first.get("thickness", 5.0)) > 5.5:
            continue
        for second_index in range(first_index + 1, len(walls)):
            second = walls[second_index]
            if second.get("orientation") != first.get("orientation"):
                continue
            if float(second.get("thickness", 5.0)) > 5.5:
                continue
            separation = abs(float(first["fixed"]) - float(second["fixed"]))
            if not 3.0 <= separation <= maximum_separation:
                continue
            overlap = interval_overlap(
                float(first["start"]),
                float(first["end"]),
                float(second["start"]),
                float(second["end"]),
            )
            minimum_length = max(
                1.0,
                min(
                    float(first["end"]) - float(first["start"]),
                    float(second["end"]) - float(second["start"]),
                ),
            )
            overlap_ratio = overlap / minimum_length
            endpoint_delta = (
                abs(float(first["start"]) - float(second["start"]))
                + abs(float(first["end"]) - float(second["end"]))
            )
            if overlap_ratio < 0.58 or endpoint_delta > max(30.0, minimum_length * 0.35):
                continue
            candidates.append((separation - overlap_ratio * 2.0, first_index, second_index))

    consumed: set[int] = set()
    paired: list[dict[str, Any]] = []
    pair_count = 0
    for _, first_index, second_index in sorted(candidates):
        if first_index in consumed or second_index in consumed:
            continue
        first, second = walls[first_index], walls[second_index]
        separation = abs(float(first["fixed"]) - float(second["fixed"]))
        face_thickness = float(np.median([
            float(first.get("thickness", 2.0)),
            float(second.get("thickness", 2.0)),
        ]))
        paired.append({
            "orientation": first["orientation"],
            "fixed": (float(first["fixed"]) + float(second["fixed"])) / 2.0,
            "start": min(float(first["start"]), float(second["start"])),
            "end": max(float(first["end"]), float(second["end"])),
            "thickness": separation + face_thickness,
            "confidence": max(
                float(first.get("confidence", 0.55)),
                float(second.get("confidence", 0.55)),
            ),
            "source": "paired-parallel-wall-faces",
            "source_segment_count": int(first.get("source_segment_count", 1))
            + int(second.get("source_segment_count", 1)),
        })
        consumed.update((first_index, second_index))
        pair_count += 1

    paired.extend(wall for index, wall in enumerate(walls) if index not in consumed)
    collapsed = merge_collinear_walls(paired)
    return collapsed, {
        "input_count": len(walls),
        "paired_face_count": pair_count,
        "output_count": len(collapsed),
    }


def candidate_axis(candidate: dict[str, Any]) -> tuple[float, float, float]:
    x1, y1, x2, y2 = candidate["box_crop_px"]
    if candidate["orientation"] == "horizontal":
        return (y1 + y2) / 2, x1, x2
    return (x1 + x2) / 2, y1, y2


def associate_wall(candidate: dict[str, Any], walls: list[dict[str, Any]]) -> tuple[int, dict[str, Any]] | None:
    fixed, start, end = candidate_axis(candidate)
    choices: list[tuple[float, int, dict[str, Any]]] = []
    for index, wall in enumerate(walls):
        if wall["orientation"] != candidate["orientation"]:
            continue
        normal_distance = abs(fixed - float(wall["fixed"]))
        normal_limit = max(13.0, float(wall["thickness"]) * 2.8)
        if normal_distance > normal_limit:
            continue
        along_gap = max(float(wall["start"]) - end, start - float(wall["end"]), 0.0)
        along_limit = max(42.0, float(wall["thickness"]) * 7.0)
        if along_gap > along_limit:
            continue
        choices.append((normal_distance + along_gap * 0.25, index, wall))
    if not choices:
        return None
    _, index, wall = min(choices, key=lambda item: item[0])
    return index, wall


def filter_structural_wall_network(
    walls: list[dict[str, Any]],
    opening_candidates: list[dict[str, Any]],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discard detached thin frames, title boxes and dimension-line networks.

    The building component is selected from graph connectivity, orthogonal
    junctions, wall thickness and nearby opening evidence.  Detached geometry
    is retained when it has comparable structural thickness and complexity.
    """

    if len(walls) < 2:
        return walls, {
            "input_count": len(walls),
            "component_count": len(walls),
            "kept_component_count": len(walls),
            "rejected_wall_count": 0,
        }

    parent = list(range(len(walls)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    def connected(first: dict[str, Any], second: dict[str, Any]) -> tuple[bool, bool]:
        first_thickness = float(first.get("thickness", 5.0))
        second_thickness = float(second.get("thickness", 5.0))
        tolerance = max(4.0, (first_thickness + second_thickness) * 0.65)
        if first["orientation"] == second["orientation"]:
            if abs(float(first["fixed"]) - float(second["fixed"])) > tolerance:
                return False, False
            gap = max(
                float(first["start"]) - float(second["end"]),
                float(second["start"]) - float(first["end"]),
                0.0,
            )
            return gap <= max(9.0, tolerance * 2.0), False
        horizontal, vertical = (
            (first, second)
            if first["orientation"] == "horizontal"
            else (second, first)
        )
        intersects = (
            float(horizontal["start"]) - tolerance
            <= float(vertical["fixed"])
            <= float(horizontal["end"]) + tolerance
            and float(vertical["start"]) - tolerance
            <= float(horizontal["fixed"])
            <= float(vertical["end"]) + tolerance
        )
        return intersects, intersects

    orthogonal_edges: set[tuple[int, int]] = set()
    for first_index, first in enumerate(walls):
        for second_index in range(first_index + 1, len(walls)):
            is_connected, is_orthogonal = connected(first, walls[second_index])
            if not is_connected:
                continue
            union(first_index, second_index)
            if is_orthogonal:
                orthogonal_edges.add((first_index, second_index))

    groups: dict[int, list[int]] = {}
    for wall_index in range(len(walls)):
        groups.setdefault(find(wall_index), []).append(wall_index)
    if len(groups) == 1:
        return walls, {
            "input_count": len(walls),
            "component_count": 1,
            "kept_component_count": 1,
            "rejected_wall_count": 0,
        }

    minimum_dimension = float(min(int(image_shape[0]), int(image_shape[1])))
    components: list[dict[str, Any]] = []
    for component_index, indexes in enumerate(groups.values()):
        component_walls = [walls[index] for index in indexes]
        total_length = sum(
            max(0.0, float(wall["end"]) - float(wall["start"]))
            for wall in component_walls
        )
        thickness = float(np.median([
            float(wall.get("thickness", 5.0)) for wall in component_walls
        ]))
        junction_count = sum(
            first in indexes and second in indexes
            for first, second in orthogonal_edges
        )
        opening_support = sum(
            associate_wall(candidate, component_walls) is not None
            for candidate in opening_candidates
        )
        complexity = max(0, len(component_walls) - 4)
        structural_length = total_length * float(np.clip(thickness / 6.0, 0.45, 1.4))
        score = (
            structural_length
            + junction_count * minimum_dimension * 0.18
            + opening_support * minimum_dimension * 1.10
            + complexity * minimum_dimension * 0.08
            + thickness * minimum_dimension * 0.08
        )
        components.append({
            "component_index": component_index,
            "indexes": indexes,
            "wall_count": len(component_walls),
            "total_length_px": total_length,
            "median_thickness_px": thickness,
            "junction_count": junction_count,
            "opening_support": opening_support,
            "score": score,
        })

    main = max(components, key=lambda component: float(component["score"]))
    main_length = max(1.0, float(main["total_length_px"]))
    main_thickness = max(1.0, float(main["median_thickness_px"]))
    kept_components: list[dict[str, Any]] = []
    rejected_components: list[dict[str, Any]] = []
    for component in components:
        comparable_structure = (
            float(component["median_thickness_px"]) >= main_thickness * 0.65
            and float(component["total_length_px"]) >= main_length * 0.18
            and int(component["junction_count"]) >= 1
        )
        keep = (
            component is main
            or int(component["opening_support"]) >= 1
            or comparable_structure
        )
        (kept_components if keep else rejected_components).append(component)

    kept_indexes = {
        wall_index
        for component in kept_components
        for wall_index in component["indexes"]
    }
    filtered = [wall for index, wall in enumerate(walls) if index in kept_indexes]
    return filtered, {
        "input_count": len(walls),
        "output_count": len(filtered),
        "component_count": len(components),
        "kept_component_count": len(kept_components),
        "rejected_component_count": len(rejected_components),
        "rejected_wall_count": len(walls) - len(filtered),
        "main_component": {
            key: round(float(value), 4) if isinstance(value, float) else value
            for key, value in main.items()
            if key != "indexes"
        },
        "rejected_components": [
            {
                key: round(float(value), 4) if isinstance(value, float) else value
                for key, value in component.items()
                if key != "indexes"
            }
            for component in rejected_components
        ],
    }


def opening_interval(opening: dict[str, Any]) -> tuple[str, float, float, float]:
    start = opening["start_px"]
    end = opening["end_px"]
    if abs(float(start[0]) - float(end[0])) >= abs(float(start[1]) - float(end[1])):
        return "horizontal", (float(start[1]) + float(end[1])) / 2, min(float(start[0]), float(end[0])), max(float(start[0]), float(end[0]))
    return "vertical", (float(start[0]) + float(end[0])) / 2, min(float(start[1]), float(end[1])), max(float(start[1]), float(end[1]))


def opening_touches_exterior(
    opening: dict[str, Any],
    wall: dict[str, Any],
    walls: list[dict[str, Any]],
) -> bool:
    """Classifica a parede pela existencia de faixas paralelas nos dois lados."""
    start = np.asarray(opening["start_px"], dtype=float)
    end = np.asarray(opening["end_px"], dtype=float)
    center = (start + end) / 2.0
    along = float(center[0] if wall["orientation"] == "horizontal" else center[1])
    fixed = float(wall["fixed"])
    interval_tolerance = max(18.0, float(wall.get("thickness", 6.0)) * 3.0)
    has_negative_side = False
    has_positive_side = False
    for candidate in walls:
        if candidate is wall or candidate["orientation"] != wall["orientation"]:
            continue
        candidate_start = float(candidate["start"])
        candidate_end = float(candidate["end"])
        if not candidate_start - interval_tolerance <= along <= candidate_end + interval_tolerance:
            continue
        delta = float(candidate["fixed"]) - fixed
        if abs(delta) <= max(
            float(wall.get("thickness", 6.0)),
            float(candidate.get("thickness", 6.0)),
        ):
            continue
        has_negative_side = has_negative_side or delta < 0
        has_positive_side = has_positive_side or delta > 0
    return not (has_negative_side and has_positive_side)


def matching_heuristic_door(candidate: dict[str, Any], doors: list[dict[str, Any]]) -> dict[str, Any] | None:
    fixed, start, end = candidate_axis(candidate)
    for door in doors:
        orientation, door_fixed, door_start, door_end = opening_interval(door)
        if orientation != candidate["orientation"] or abs(fixed - door_fixed) > 18:
            continue
        along_gap = max(door_start - end, start - door_end, 0.0)
        if along_gap <= 6:
            return door
    return None


def wall_band_density(
    gray: np.ndarray,
    wall: dict[str, Any],
    start: float,
    end: float,
) -> float:
    along_start, along_end = sorted((int(round(start)), int(round(end))))
    fixed = int(round(float(wall["fixed"])))
    half_band = max(2, int(round(float(wall["thickness"]) * 0.55)))
    if wall["orientation"] == "horizontal":
        patch = gray[
            max(0, fixed - half_band) : min(gray.shape[0], fixed + half_band + 1),
            max(0, along_start) : min(gray.shape[1], along_end + 1),
        ]
    else:
        patch = gray[
            max(0, along_start) : min(gray.shape[0], along_end + 1),
            max(0, fixed - half_band) : min(gray.shape[1], fixed + half_band + 1),
        ]
    return float(np.mean(patch < 130)) if patch.size else 1.0


def reconcile_openings(
    candidates: list[dict[str, Any]],
    walls: list[dict[str, Any]],
    heuristic_openings: list[dict[str, Any]],
    gray: np.ndarray,
    *,
    add_unmatched_heuristic_doors: bool,
) -> list[dict[str, Any]]:
    doors = [item for item in heuristic_openings if item.get("type") == "door"]
    openings: list[dict[str, Any]] = []
    matched_door_ids: set[int] = set()
    for candidate in candidates:
        association = associate_wall(candidate, walls)
        if association is None:
            continue
        wall_index, wall = association
        fixed, start, end = candidate_axis(candidate)
        density = wall_band_density(gray, wall, start, end)
        if density > 0.55:
            continue
        door = matching_heuristic_door(candidate, doors)
        if door is not None and id(door) in matched_door_ids:
            continue
        opening_type = "door" if door is not None else candidate["type"]
        if door is not None:
            matched_door_ids.add(id(door))
            _, _, start, end = opening_interval(door)
        axis_fixed = float(wall["fixed"])
        if candidate["orientation"] == "horizontal":
            start_px, end_px = [start, axis_fixed], [end, axis_fixed]
        else:
            start_px, end_px = [axis_fixed, start], [axis_fixed, end]
        openings.append({
            **candidate,
            "type": opening_type,
            "wall_index": wall_index,
            "wall_orientation": wall["orientation"],
            "start_px": [round(value, 3) for value in start_px],
            "end_px": [round(value, 3) for value in end_px],
            "classification": "2d-door-arc" if door is not None else "yolo-vote",
            "wall_band_density": round(density, 5),
        })

    # The door-arc heuristic is high precision and recovers doors that YOLO did
    # not propose. It runs only after the independent YOLO pass.
    for door in doors:
        if id(door) in matched_door_ids:
            continue
        if not add_unmatched_heuristic_doors:
            continue
        orientation, fixed, start, end = opening_interval(door)
        candidate = {
            "type": "door",
            "confidence": round(float(door.get("confidence", 0.75)), 6),
            "orientation": orientation,
            "box_crop_px": [start, fixed - 2, end, fixed + 2] if orientation == "horizontal" else [fixed - 2, start, fixed + 2, end],
            "source": "2d-door-arc-after-yolo",
        }
        association = associate_wall(candidate, walls)
        if association is None:
            continue
        wall_index, wall = association
        openings.append({
            **candidate,
            "wall_index": wall_index,
            "wall_orientation": orientation,
            "start_px": [start, wall["fixed"]] if orientation == "horizontal" else [wall["fixed"], start],
            "end_px": [end, wall["fixed"]] if orientation == "horizontal" else [wall["fixed"], end],
            "classification": "2d-door-arc",
            "scale_support": [],
            "raw_count": 0,
        })
    return openings


def draw_candidates(crop: np.ndarray, candidates: list[dict[str, Any]]) -> np.ndarray:
    preview = crop.copy()
    for index, candidate in enumerate(candidates, 1):
        x1, y1, x2, y2 = (int(round(value)) for value in candidate["box_crop_px"])
        colour = DOOR_COLOUR if candidate["type"] == "door" else WINDOW_COLOUR
        cv2.rectangle(preview, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        cv2.putText(preview, f"P{index}", (x1 + 2, max(14, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    return preview


def draw_result(crop: np.ndarray, walls: list[dict[str, Any]], openings: list[dict[str, Any]]) -> np.ndarray:
    overlay = crop.copy()
    for wall in walls:
        if wall["orientation"] == "horizontal":
            point1 = (int(round(wall["start"])), int(round(wall["fixed"])))
            point2 = (int(round(wall["end"])), int(round(wall["fixed"])))
        else:
            point1 = (int(round(wall["fixed"])), int(round(wall["start"])))
            point2 = (int(round(wall["fixed"])), int(round(wall["end"])))
        cv2.line(overlay, point1, point2, WALL_COLOUR, 1, cv2.LINE_AA)
    door_index = 0
    window_index = 0
    for opening in openings:
        if opening["type"] == "door":
            door_index += 1
            label = f"D{door_index}"
            colour = DOOR_COLOUR
        else:
            window_index += 1
            label = f"W{window_index}"
            colour = WINDOW_COLOUR
        point1 = tuple(int(round(value)) for value in opening["start_px"])
        point2 = tuple(int(round(value)) for value in opening["end_px"])
        cv2.line(overlay, point1, point2, colour, 5, cv2.LINE_AA)
        middle = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
        cv2.putText(overlay, label, (middle[0] + 3, middle[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (middle[0] + 3, middle[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.43, colour, 1, cv2.LINE_AA)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canvas-width-m", type=float, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[640, 960, 1280])
    parser.add_argument("--raw-confidence", type=float, default=0.01)
    parser.add_argument("--consensus-confidence", type=float, default=0.02)
    parser.add_argument("--minimum-scale-support", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--disable-metric-refinement", action="store_true")
    parser.add_argument("--wall-weights", type=Path)
    parser.add_argument("--wall-size", type=int, default=960)
    parser.add_argument("--wall-confidence", type=float, default=0.15)
    parser.add_argument(
        "--detection-scale-mode",
        choices=("auto", "metric"),
        default="auto",
        help="Auto calibrates visual filters from wall thickness; metric preserves legacy behavior.",
    )
    parser.add_argument("--nominal-wall-thickness-m", type=float, default=0.20)
    parser.add_argument(
        "--wall-source",
        choices=("geometry", "yolo", "hybrid"),
        default="geometry",
        help="Use classic geometry, YOLO-Seg, or a true deduplicated 2D+YOLO fusion.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = read_image(args.image)
    crop_bbox = detect_building_bbox(image)
    left, top, right, bottom = crop_bbox
    crop = image[top:bottom, left:right].copy()
    crop_path = args.output_dir / "building_crop.png"
    write_image(crop_path, crop)

    opening_model = YOLO(str(args.weights))
    raw = predict_multiscale(opening_model, crop, args.sizes, args.raw_confidence, args.device)
    candidates = consolidate_detections(
        raw,
        minimum_confidence=args.consensus_confidence,
        minimum_scale_support=args.minimum_scale_support,
    )
    write_image(args.output_dir / "yolo_consensus_before_walls.png", draw_candidates(crop, candidates))

    predicted_wall_mask: np.ndarray | None = None
    wall_segmentation = {
        "requested_source": args.wall_source,
        "weights": str(args.wall_weights) if args.wall_weights else None,
        "instance_count": 0,
        "axis_count": 0,
        "used": False,
        "fallback": None,
        "fusion": None,
    }
    if args.wall_source != "geometry":
        if args.wall_weights is None or not args.wall_weights.exists():
            wall_segmentation["fallback"] = "wall_weights_missing"
        else:
            wall_model = YOLO(str(args.wall_weights))
            predicted_wall_mask, mask_diagnostic = predict_wall_segmentation(
                wall_model,
                crop,
                image_size=args.wall_size,
                confidence=args.wall_confidence,
                device=args.device,
            )
            wall_segmentation.update(mask_diagnostic)
            write_image(args.output_dir / "yolo_wall_mask.png", predicted_wall_mask)
            write_image(
                args.output_dir / "yolo_wall_mask_overlay.png",
                draw_wall_mask(crop, predicted_wall_mask),
            )

    if args.detection_scale_mode == "auto":
        detection_scale = estimate_visual_detection_scale(
            crop,
            predicted_wall_mask,
            nominal_wall_thickness_m=args.nominal_wall_thickness_m,
        )
        detection_canvas_extent_m = float(detection_scale["detection_canvas_extent_m"])
    else:
        detection_canvas_extent_m = args.canvas_width_m
        detection_scale = {
            "mode": "legacy-user-metric",
            "method": "canvas-width-m",
            "detection_canvas_extent_m": detection_canvas_extent_m,
        }

    geometry = vectorize_floorplan_2d(
        crop_path,
        canvas_width_m=detection_canvas_extent_m,
    )
    raster_wall_count = len(geometry["walls"])
    geometry_face_pairing = {
        "input_count": 0,
        "paired_face_count": 0,
        "output_count": 0,
    }
    if raster_wall_count >= 8:
        geometry_walls = merge_collinear_walls(geometry["walls"])
        geometry_source = "thick-wall-2d"
    else:
        single_line_walls = merge_collinear_walls(detect_single_line_walls(crop))
        geometry_walls, geometry_face_pairing = collapse_parallel_wall_faces(
            single_line_walls,
            crop.shape,
        )
        geometry_source = (
            "single-line-face-pairing"
            if geometry_face_pairing["paired_face_count"]
            else "single-line-fallback"
        )
    walls = geometry_walls
    wall_geometry_source = geometry_source
    if predicted_wall_mask is not None:
        yolo_walls, yolo_wall_diagnostic = walls_from_segmentation_mask(
            predicted_wall_mask,
            canvas_width_m=detection_canvas_extent_m,
        )
        wall_segmentation["axis_count"] = len(yolo_walls)
        wall_segmentation["vectorizer"] = yolo_wall_diagnostic
        minimum_reasonable_count = max(4, round(len(geometry_walls) * 0.45))
        if len(yolo_walls) >= minimum_reasonable_count:
            if args.wall_source == "hybrid":
                walls, fusion_diagnostic = fuse_wall_sets(geometry_walls, yolo_walls)
                wall_segmentation["fusion"] = fusion_diagnostic
                wall_geometry_source = "hybrid-2d-yolo-fusion"
                wall_segmentation["used"] = True
            elif args.wall_source == "yolo":
                walls = yolo_walls
                wall_geometry_source = "yolo-wall-seg"
                wall_segmentation["used"] = True
        else:
            wall_segmentation["fallback"] = "insufficient_yolo_wall_axes"
    walls, structural_filter = filter_structural_wall_network(
        walls,
        candidates,
        crop.shape,
    )
    wall_segmentation["structural_filter"] = structural_filter
    openings = reconcile_openings(
        candidates,
        walls,
        geometry["openings"],
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
        add_unmatched_heuristic_doors=raster_wall_count >= 8,
    )
    meters_per_pixel = args.canvas_width_m / max(1, crop.shape[1])
    detection_pixel_m = float(detection_scale.get("detection_pixel_m", meters_per_pixel))
    for opening in openings:
        start = np.asarray(opening["start_px"], dtype=float)
        end = np.asarray(opening["end_px"], dtype=float)
        width_px = float(np.linalg.norm(end - start))
        opening["width_m"] = round(width_px * meters_per_pixel, 4)
        opening["classification_width_m"] = round(width_px * detection_pixel_m, 4)
        wall = walls[int(opening["wall_index"])]
        on_exterior_boundary = opening_touches_exterior(
            opening,
            wall,
            walls,
        )
        opening["topology_exterior"] = on_exterior_boundary
        # Plantas simplificadas frequentemente omitem o arco da porta. Quando
        # um vao de largura plausivel separa dois espacos internos, a topologia
        # e uma evidencia melhor que a classe visual isolada do YOLO.
        if (
            not args.disable_metric_refinement
            and opening["classification"] == "yolo-vote"
        ):
            if (
                not on_exterior_boundary
                and opening["type"] == "window"
                and 0.55 <= opening["classification_width_m"] <= 1.45
            ):
                opening["type"] = "door"
                opening["classification"] = "interior-opening-prior"
        opening["start_original_px"] = [round(opening["start_px"][0] + left, 3), round(opening["start_px"][1] + top, 3)]
        opening["end_original_px"] = [round(opening["end_px"][0] + left, 3), round(opening["end_px"][1] + top, 3)]

    crop_overlay = draw_result(crop, walls, openings)
    write_image(args.output_dir / "openings_consolidated_crop.png", crop_overlay)
    full_overlay = image.copy()
    full_overlay[top:bottom, left:right] = crop_overlay
    cv2.rectangle(full_overlay, (left, top), (right - 1, bottom - 1), (125, 125, 125), 1, cv2.LINE_AA)
    write_image(args.output_dir / "openings_consolidated_full.png", full_overlay)

    review = {
        "openings": [
            {
                "type": item["type"],
                "start_px": item["start_px"],
                "end_px": item["end_px"],
                "confidence": item["confidence"],
                "reason": item["classification"],
            }
            for item in openings
        ]
    }
    (args.output_dir / "openings_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = {
        "schema": "plant2bim.pre-wall-opening-pipeline.v1",
        "source_image": str(args.image),
        "weights": str(args.weights),
        "crop_bbox_original_px": list(crop_bbox),
        "crop_size_px": [crop.shape[1], crop.shape[0]],
        "canvas_width_m": args.canvas_width_m,
        "meters_per_pixel": meters_per_pixel,
        "detection_scale": detection_scale,
        "metric_refinement": not args.disable_metric_refinement,
        "scales": args.sizes,
        "raw_detection_count": len(raw),
        "consensus_candidate_count": len(candidates),
        "merged_wall_count": len(walls),
        "raster_wall_count": raster_wall_count,
        "geometry_face_pairing": geometry_face_pairing,
        "wall_geometry_source": wall_geometry_source,
        "wall_segmentation": wall_segmentation,
        "opening_count": len(openings),
        "door_count": sum(item["type"] == "door" for item in openings),
        "window_count": sum(item["type"] == "window" for item in openings),
        "candidates": candidates,
        "walls": walls,
        "openings": openings,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key not in {"candidates", "walls", "openings"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
