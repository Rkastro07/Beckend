"""Renderiza e avalia 1D automático, 2D automático e gabarito CubiCasa."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

try:
    from .raster_2d_import import vectorize_floorplan_2d
    from .raster_slices_import import raster_slices_image_to_editor_model
except ImportError:
    from raster_2d_import import vectorize_floorplan_2d
    from raster_slices_import import raster_slices_image_to_editor_model


WALL_COLOR = (0, 140, 255)
DOOR_COLOR = (70, 210, 70)
WINDOW_COLOR = (240, 175, 45)
TEXT_COLOR = (45, 45, 45)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Não foi possível abrir {path}")
    return image


def _parse_points(raw: str, scale: float) -> np.ndarray:
    points = []
    for token in raw.strip().split():
        x, y = token.split(",")
        points.append([float(x) * scale, float(y) * scale])
    return np.asarray(points, dtype=np.float32)


def load_cubicasa_ground_truth(svg_path: Path, image_width: int) -> dict:
    root = ET.parse(svg_path).getroot()
    view_box = [float(value) for value in (root.get("viewBox") or "0 0 1 1").split()]
    scale = image_width / view_box[2]
    walls: list[np.ndarray] = []
    openings: list[dict] = []
    for element in root.iter():
        classes = set((element.get("class") or "").split())
        if not ({"Wall", "Door", "Window"} & classes):
            continue
        polygon = next((child for child in element if child.tag.endswith("polygon")), None)
        if polygon is None or not polygon.get("points"):
            continue
        points = _parse_points(polygon.get("points") or "", scale)
        if "Wall" in classes:
            walls.append(points)
            continue
        kind = "door" if "Door" in classes else "window"
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        if y1 - y0 >= x1 - x0:
            start = [(x0 + x1) / 2.0, y0]
            end = [(x0 + x1) / 2.0, y1]
            orientation = "vertical"
        else:
            start = [x0, (y0 + y1) / 2.0]
            end = [x1, (y0 + y1) / 2.0]
            orientation = "horizontal"
        openings.append({
            "type": kind,
            "orientation": orientation,
            "start_px": [float(start[0]), float(start[1])],
            "end_px": [float(end[0]), float(end[1])],
            "bbox_px": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
        })
    return {"scale": scale, "walls": walls, "openings": openings}


def _one_d_wall_lines(model: dict, width: int, height: int, canvas_width_m: float) -> list[dict]:
    canvas_size = max(width, height)
    pad_x = (canvas_size - width) / 2.0
    pad_y = (canvas_size - height) / 2.0

    def point(x: float, y: float) -> list[float]:
        return [
            x / canvas_width_m * canvas_size - pad_x,
            canvas_size - y / canvas_width_m * canvas_size - pad_y,
        ]

    return [
        {
            "start_px": point(float(wall["ax"]), float(wall["ay"])),
            "end_px": point(float(wall["bx"]), float(wall["by"])),
            "thickness": max(2.0, float(wall["espessura"]) / canvas_width_m * canvas_size),
            "id": wall["id"],
        }
        for wall in model["paredes"]
    ]


def _one_d_opening_lines(model: dict, wall_lines: list[dict]) -> list[dict]:
    walls = {item["id"]: item for item in wall_lines}
    model_walls = {item["id"]: item for item in model["paredes"]}
    openings: list[dict] = []
    for opening in model["aberturas"]:
        wall_px = walls.get(opening["parede_id"])
        wall_model = model_walls.get(opening["parede_id"])
        if wall_px is None or wall_model is None:
            continue
        start = np.asarray(wall_px["start_px"], dtype=float)
        end = np.asarray(wall_px["end_px"], dtype=float)
        vector = end - start
        pixel_length = float(np.linalg.norm(vector))
        model_length = math.hypot(
            float(wall_model["bx"]) - float(wall_model["ax"]),
            float(wall_model["by"]) - float(wall_model["ay"]),
        )
        if pixel_length <= 1e-9 or model_length <= 1e-9:
            continue
        unit = vector / pixel_length
        center = start + unit * (float(opening["s_centro"]) / model_length * pixel_length)
        half = float(opening["largura"]) / model_length * pixel_length / 2.0
        openings.append({
            "type": opening["tipo"],
            "start_px": (center - unit * half).tolist(),
            "end_px": (center + unit * half).tolist(),
        })
    return openings


def _axis_to_line(axis: dict) -> dict:
    if axis["orientation"] == "vertical":
        start = [float(axis["fixed"]), float(axis["start"])]
        end = [float(axis["fixed"]), float(axis["end"])]
    else:
        start = [float(axis["start"]), float(axis["fixed"])]
        end = [float(axis["end"]), float(axis["fixed"])]
    return {"start_px": start, "end_px": end, "thickness": float(axis["thickness"])}


def _rasterize_lines(shape: tuple[int, int], lines: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for line in lines:
        start = tuple(int(round(value)) for value in line["start_px"])
        end = tuple(int(round(value)) for value in line["end_px"])
        thickness = max(2, int(round(float(line["thickness"]))))
        cv2.line(mask, start, end, 1, thickness, cv2.LINE_8)
    return mask


def _wall_metrics(detected: np.ndarray, truth: np.ndarray, tolerance: int = 3) -> dict:
    kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)
    truth_dilated = cv2.dilate(truth, kernel)
    detected_dilated = cv2.dilate(detected, kernel)
    detected_pixels = int(detected.sum())
    truth_pixels = int(truth.sum())
    precision = float((detected & truth_dilated).sum()) / max(1, detected_pixels)
    recall = float((truth & detected_dilated).sum()) / max(1, truth_pixels)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _midpoint(opening: dict) -> np.ndarray:
    return (
        np.asarray(opening["start_px"], dtype=float)
        + np.asarray(opening["end_px"], dtype=float)
    ) / 2.0


def _point_box_distance(point: np.ndarray, bbox: list[float]) -> float:
    x, y, width, height = map(float, bbox)
    x2, y2 = x + width, y + height
    dx = max(x - point[0], 0.0, point[0] - x2)
    dy = max(y - point[1], 0.0, point[1] - y2)
    return math.hypot(dx, dy)


def _opening_metrics(predictions: list[dict], truth: list[dict], kind: str) -> dict:
    predicted = [item for item in predictions if item["type"] == kind]
    expected = [item for item in truth if item["type"] == kind]
    available = set(range(len(expected)))
    true_positives = 0
    orientation_correct = 0
    for item in predicted:
        candidates: list[tuple[float, int]] = []
        for index in available:
            target = expected[index]
            if item.get("orientation") and item["orientation"] != target["orientation"] and kind == "window":
                continue
            center = _midpoint(target)
            if kind == "door" and item.get("bbox_px"):
                distance = _point_box_distance(center, item["bbox_px"])
            else:
                distance = float(np.linalg.norm(_midpoint(item) - center))
            candidates.append((distance, index))
        if not candidates:
            continue
        distance, match = min(candidates)
        if distance <= (12.0 if kind == "door" else 14.0):
            available.remove(match)
            true_positives += 1
            if item.get("orientation") == expected[match].get("orientation"):
                orientation_correct += 1
    false_positives = len(predicted) - true_positives
    false_negatives = len(expected) - true_positives
    precision = true_positives / max(1, len(predicted))
    recall = true_positives / max(1, len(expected))
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return {
        "predicted": len(predicted),
        "expected": len(expected),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "host_orientation_correct": orientation_correct,
        "host_orientation_accuracy": round(orientation_correct / max(1, true_positives), 4),
    }


def _draw_lines(image: np.ndarray, lines: list[dict], color: tuple[int, int, int], thickness_override: int | None = None) -> None:
    for line in lines:
        start = tuple(int(round(value)) for value in line["start_px"])
        end = tuple(int(round(value)) for value in line["end_px"])
        thickness = thickness_override or max(2, int(round(float(line.get("thickness", 5)))))
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)


def _panel(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    header = 72
    panel = np.full((image.shape[0] + header, image.shape[1], 3), 255, dtype=np.uint8)
    panel[header:] = image
    cv2.putText(panel, title, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.66, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(panel, subtitle, (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (75, 75, 75), 1, cv2.LINE_AA)
    return panel


def render_comparison(
    image_path: Path,
    svg_path: Path,
    output_path: Path,
    *,
    canvas_width_m: float,
) -> dict:
    image = _read_image(image_path)
    height, width = image.shape[:2]
    one_d = raster_slices_image_to_editor_model(image_path, canvas_width_m=canvas_width_m)
    two_d = vectorize_floorplan_2d(image_path, canvas_width_m=canvas_width_m)
    truth = load_cubicasa_ground_truth(svg_path, width)

    one_d_walls = _one_d_wall_lines(one_d, width, height, canvas_width_m)
    one_d_openings = _one_d_opening_lines(one_d, one_d_walls)
    two_d_walls = [_axis_to_line(axis) for axis in two_d["walls"]]
    two_d_openings = list(two_d["openings"])

    truth_wall_mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in truth["walls"]:
        cv2.fillPoly(truth_wall_mask, [np.round(polygon).astype(np.int32)], 1)
    one_d_mask = _rasterize_lines((height, width), one_d_walls)
    two_d_mask = _rasterize_lines((height, width), two_d_walls)
    metrics = {
        "ground_truth": {
            "wall_objects": len(truth["walls"]),
            "doors": sum(item["type"] == "door" for item in truth["openings"]),
            "windows": sum(item["type"] == "window" for item in truth["openings"]),
        },
        "one_d": {
            "wall_axes": len(one_d_walls),
            "walls": _wall_metrics(one_d_mask, truth_wall_mask),
            "doors": _opening_metrics(one_d_openings, truth["openings"], "door"),
            "windows": _opening_metrics(one_d_openings, truth["openings"], "window"),
        },
        "two_d": {
            "wall_axes": len(two_d_walls),
            "walls": _wall_metrics(two_d_mask, truth_wall_mask),
            "doors": _opening_metrics(two_d_openings, truth["openings"], "door"),
            "windows": _opening_metrics(two_d_openings, truth["openings"], "window"),
        },
    }

    one_overlay = image.copy()
    _draw_lines(one_overlay, one_d_walls, WALL_COLOR)
    _draw_lines(one_overlay, one_d_openings, DOOR_COLOR, 8)
    one_overlay = cv2.addWeighted(image, 0.55, one_overlay, 0.72, 0.0)

    two_overlay = image.copy()
    _draw_lines(two_overlay, two_d_walls, WALL_COLOR)
    for opening in two_d_openings:
        color = DOOR_COLOR if opening["type"] == "door" else WINDOW_COLOR
        _draw_lines(two_overlay, [opening], color, 8)
        if opening["type"] == "door" and opening.get("bbox_px"):
            x, y, box_width, box_height = map(int, opening["bbox_px"])
            cv2.rectangle(two_overlay, (x, y), (x + box_width - 1, y + box_height - 1), color, 2, cv2.LINE_AA)
    two_overlay = cv2.addWeighted(image, 0.55, two_overlay, 0.72, 0.0)

    truth_overlay = image.copy()
    wall_layer = image.copy()
    wall_layer[truth_wall_mask > 0] = WALL_COLOR
    truth_overlay = cv2.addWeighted(image, 0.62, wall_layer, 0.50, 0.0)
    for opening in truth["openings"]:
        color = DOOR_COLOR if opening["type"] == "door" else WINDOW_COLOR
        _draw_lines(truth_overlay, [opening], color, 8)

    one_panel = _panel(
        one_overlay,
        "1D AUTOMATICO",
        f"{len(one_d_walls)} eixos candidatos | {len(one_d_openings)} aberturas",
    )
    two_panel = _panel(
        two_overlay,
        "2D AUTOMATICO",
        f"{len(two_d_walls)} trechos candidatos | {metrics['two_d']['doors']['predicted']} portas | {metrics['two_d']['windows']['predicted']} janelas",
    )
    truth_panel = _panel(
        truth_overlay,
        "GABARITO (SO AVALIACAO)",
        f"{metrics['ground_truth']['wall_objects']} objetos de parede | {metrics['ground_truth']['doors']} portas | {metrics['ground_truth']['windows']} janelas",
    )
    gap = np.full((one_panel.shape[0], 10, 3), 235, dtype=np.uint8)
    comparison = np.hstack([one_panel, gap, two_panel, gap, truth_panel])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", comparison)
    if not ok:
        raise RuntimeError("Falha ao codificar comparação PNG")
    encoded.tofile(str(output_path))

    automatic_2d_path = output_path.with_name(output_path.stem + "_2d_automatic.png")
    ok, encoded = cv2.imencode(".png", two_panel)
    if not ok:
        raise RuntimeError("Falha ao codificar PNG 2D")
    encoded.tofile(str(automatic_2d_path))

    payload = {"metrics": metrics, "one_d": one_d, "two_d": two_d}
    output_path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"metrics": metrics, "comparison": str(output_path), "automatic_2d": str(automatic_2d_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("svg", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--canvas-width", type=float, default=16.0)
    args = parser.parse_args()
    result = render_comparison(
        args.image,
        args.svg,
        args.output,
        canvas_width_m=args.canvas_width,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
