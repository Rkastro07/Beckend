"""Stitch tiled geometric walls and their YOLO opening proposals globally."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("wall_models_index", type=Path)
    parser.add_argument("yolo_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--storey", default="S01")
    parser.add_argument("--angle-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--axis-offset-m", type=float, default=0.45)
    parser.add_argument("--join-gap-m", type=float, default=1.50)
    parser.add_argument("--thickness-difference-m", type=float, default=0.45)
    return parser.parse_args()


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[b] = a


def geometry(wall: dict):
    start = np.array([wall["ax"], wall["ay"]], dtype=np.float64)
    end = np.array([wall["bx"], wall["by"]], dtype=np.float64)
    vector = end - start
    length = float(np.linalg.norm(vector))
    unit = vector / max(length, 1e-9)
    if unit[0] < 0.0 or (abs(unit[0]) < 1e-9 and unit[1] < 0.0):
        unit = -unit
    return start, end, unit, length


def interval_gap(first: tuple[float, float], second: tuple[float, float]) -> float:
    if first[1] < second[0]:
        return second[0] - first[1]
    if second[1] < first[0]:
        return first[0] - second[1]
    return 0.0


def mergeable(first: dict, second: dict, args) -> bool:
    a0, a1, ua, la = geometry(first)
    b0, b1, ub, lb = geometry(second)
    if la < 0.20 or lb < 0.20:
        return False
    alignment = abs(float(np.dot(ua, ub)))
    if alignment < math.cos(math.radians(args.angle_tolerance_deg)):
        return False
    normal = np.array([-ua[1], ua[0]])
    offset = abs(float(np.dot((b0 + b1 - a0 - a1) / 2.0, normal)))
    if offset > args.axis_offset_m:
        return False
    if abs(float(first["espessura"]) - float(second["espessura"])) > args.thickness_difference_m:
        return False
    origin = (a0 + a1) / 2.0
    ia = sorted((float(np.dot(a0 - origin, ua)), float(np.dot(a1 - origin, ua))))
    ib = sorted((float(np.dot(b0 - origin, ua)), float(np.dot(b1 - origin, ua))))
    return interval_gap((ia[0], ia[1]), (ib[0], ib[1])) <= args.join_gap_m


def weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    ordered_values = np.asarray(values)[order]
    ordered_weights = np.asarray(weights)[order]
    index = int(np.searchsorted(np.cumsum(ordered_weights), np.sum(ordered_weights) / 2.0))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def merge_cluster(cluster_id: int, members: list[dict]) -> dict:
    parts = [geometry(wall) for wall in members]
    reference = max(parts, key=lambda item: item[3])[2]
    vectors = []
    weights = []
    for (_, _, unit, length) in parts:
        if np.dot(unit, reference) < 0.0:
            unit = -unit
        vectors.append(unit)
        weights.append(length)
    tangent = np.average(np.vstack(vectors), axis=0, weights=np.asarray(weights))
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    offsets = []
    parameters = []
    for wall, (start, end, _, length) in zip(members, parts):
        midpoint = (start + end) / 2.0
        offsets.append(float(np.dot(midpoint, normal)))
        parameters.extend((float(np.dot(start, tangent)), float(np.dot(end, tangent))))
    offset = weighted_median(offsets, weights)
    start = tangent * min(parameters) + normal * offset
    end = tangent * max(parameters) + normal * offset
    thickness = weighted_median([float(item["espessura"]) for item in members], weights)
    return {
        "id": f"TW-S01-{cluster_id:03d}",
        "ax": round(float(start[0]), 6),
        "ay": round(float(start[1]), 6),
        "bx": round(float(end[0]), 6),
        "by": round(float(end[1]), 6),
        "espessura": round(thickness, 4),
        "comprimento": round(float(np.linalg.norm(end - start)), 4),
        "members": [item["id"] for item in members],
        "source_tiles": sorted({item["source_tile"] for item in members}),
        "stitch_method": "deterministic_collinear_geometry_v1",
    }


def vertical_iou(first: dict, second: dict) -> float:
    intersection = max(0.0, min(first["z_max"], second["z_max"]) - max(first["z_min"], second["z_min"]))
    union = max(first["z_max"], second["z_max"]) - min(first["z_min"], second["z_min"])
    return intersection / max(union, 1e-9)


def deduplicate_openings(openings: list[dict]) -> list[dict]:
    selected = []
    for candidate in sorted(openings, key=lambda item: item["confidence"], reverse=True):
        duplicate = any(
            candidate["wall_id"] == other["wall_id"]
            and abs(candidate["s_center"] - other["s_center"]) <= 0.60
            and vertical_iou(candidate, other) >= 0.40
            for other in selected
        )
        if not duplicate:
            selected.append(candidate)
    return sorted(selected, key=lambda item: (item["wall_id"], item["s_center"]))


def render_plan(walls: list[dict], openings: list[dict], output: Path) -> None:
    points = np.vstack(
        [np.array([[wall["ax"], wall["ay"]], [wall["bx"], wall["by"]]]) for wall in walls]
    )
    minimum = np.min(points, axis=0) - 1.0
    maximum = np.max(points, axis=0) + 1.0
    width, height = 2000, 1500
    scale = min((width - 100) / (maximum[0] - minimum[0]), (height - 100) / (maximum[1] - minimum[1]))

    def pixel(point):
        return (
            int(round(50 + (point[0] - minimum[0]) * scale)),
            int(round(height - 50 - (point[1] - minimum[1]) * scale)),
        )

    canvas = np.full((height, width, 3), 247, dtype=np.uint8)
    by_id = {wall["id"]: wall for wall in walls}
    for wall in walls:
        cv2.line(
            canvas,
            pixel(np.array([wall["ax"], wall["ay"]])),
            pixel(np.array([wall["bx"], wall["by"]])),
            (65, 65, 65),
            max(2, int(round(wall["espessura"] * scale))),
            cv2.LINE_AA,
        )
    colors = {"door": (45, 200, 45), "window": (225, 145, 30)}
    for opening in openings:
        wall = by_id[opening["wall_id"]]
        start = np.array([wall["ax"], wall["ay"]])
        end = np.array([wall["bx"], wall["by"]])
        tangent = (end - start) / np.linalg.norm(end - start)
        center = start + tangent * opening["s_center"]
        cv2.circle(canvas, pixel(center), 7, colors[opening["class"]], -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f'Tiled geometry | {len(walls)} walls | {len(openings)} openings',
        (35, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), canvas)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = json.loads(args.wall_models_index.read_text(encoding="utf-8"))
    walls = []
    retained_ids = set()
    for tile in index["tiles"]:
        if tile["status"] != "ready":
            continue
        model = json.loads(Path(tile["model"]).read_text(encoding="utf-8"))
        for wall in model["paredes"]:
            if f'-{args.storey}-' not in wall["source_wall_id"]:
                continue
            walls.append(wall)
            retained_ids.add(wall["id"])

    union = UnionFind(len(walls))
    for first in range(len(walls)):
        for second in range(first + 1, len(walls)):
            if mergeable(walls[first], walls[second], args):
                union.union(first, second)
    groups: dict[int, list[dict]] = {}
    for index_value, wall in enumerate(walls):
        groups.setdefault(union.find(index_value), []).append(wall)
    merged = [merge_cluster(number, members) for number, members in enumerate(groups.values(), start=1)]
    merged.sort(key=lambda wall: (wall["ax"], wall["ay"], wall["bx"], wall["by"]))
    for number, wall in enumerate(merged, start=1):
        wall["id"] = f"TW-S01-{number:03d}"
    member_to_wall = {member: wall["id"] for wall in merged for member in wall["members"]}

    openings = []
    for tile in index["tiles"]:
        if tile["status"] != "ready":
            continue
        detection_path = args.yolo_root / tile["tile_id"] / "tiled_yoloworld_m_detections.json"
        if not detection_path.exists():
            continue
        payload = json.loads(detection_path.read_text(encoding="utf-8"))
        for detection in payload["detections"]:
            if detection["wall_id"] not in retained_ids:
                continue
            stitched_id = member_to_wall[detection["wall_id"]]
            wall = next(item for item in merged if item["id"] == stitched_id)
            start = np.array([wall["ax"], wall["ay"]])
            end = np.array([wall["bx"], wall["by"]])
            tangent = (end - start) / np.linalg.norm(end - start)
            world_xy = np.asarray(detection["world_center"][:2], dtype=np.float64)
            s_center = float(np.dot(world_xy - start, tangent))
            openings.append(
                {
                    **detection,
                    "source_wall_id": detection["wall_id"],
                    "wall_id": stitched_id,
                    "s_center": round(s_center, 4),
                    "width": round(float(detection["s_max"] - detection["s_min"]), 4),
                }
            )
    openings = deduplicate_openings(openings)

    payload = {
        "schema": "cloud2bim.tiled-stitched.v1",
        "method": {
            "walls": "geometric detector per overlapping tile plus deterministic stitching",
            "wall_neural_evaluation": False,
            "openings": "YOLO-World-M per tile",
            "angle_tolerance_deg": args.angle_tolerance_deg,
            "axis_offset_m": args.axis_offset_m,
            "join_gap_m": args.join_gap_m,
        },
        "counts": {
            "owned_local_walls": len(walls),
            "stitched_walls": len(merged),
            "doors": sum(item["class"] == "door" for item in openings),
            "windows": sum(item["class"] == "window" for item in openings),
            "openings": len(openings),
        },
        "paredes": merged,
        "aberturas": openings,
    }
    output_json = args.output / "tiled_stitched_model.json"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_plan(merged, openings, args.output / "tiled_stitched_plan.png")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
