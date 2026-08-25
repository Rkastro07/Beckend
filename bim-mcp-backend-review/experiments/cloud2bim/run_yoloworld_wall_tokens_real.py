"""Run a trained wall-token detector on a real Cloud-to-BIM result.

The script consumes the accepted wall axes from ``review_model.json`` and the
XYZ cloud extracted from the original E57. It reproduces the exact metric RGB
representation used for training, detects openings, then maps every box back to
wall-local metres and world XYZ coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOWorld


CLASS_NAMES = ("door", "window")
CLASS_COLORS = {"door": (45, 210, 45), "window": (230, 150, 35)}


@dataclass(frozen=True)
class WallFrame:
    wall_id: str
    start: np.ndarray
    end: np.ndarray
    center: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    length: float
    thickness: float
    z_min: float
    z_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud-xyz", type=Path, required=True)
    parser.add_argument("--review-model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-e57", type=Path)
    parser.add_argument("--title", default="Kladno real")
    parser.add_argument("--output-prefix", default="kladno_yoloworld_m")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--floor-z", type=float, default=0.1907)
    parser.add_argument("--ceiling-z", type=float, default=3.787)
    parser.add_argument("--token-m", type=float, default=0.05)
    parser.add_argument("--tile-width-m", type=float, default=12.8)
    parser.add_argument("--tile-height-m", type=float, default=4.0)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--wall-band-m", type=float, default=0.35)
    parser.add_argument(
        "--point-keep-ratio",
        type=float,
        default=1.0,
        help="Deterministic uniform thinning applied per wall before rasterization.",
    )
    parser.add_argument("--thinning-seed", type=int, default=20260824)
    parser.add_argument(
        "--relabel-raised-doors",
        action="store_true",
        help="Relabel door predictions that do not reach the floor as windows.",
    )
    parser.add_argument("--door-floor-tolerance-m", type=float, default=0.35)
    return parser.parse_args()


def load_xyz(path: Path) -> np.ndarray:
    points = (
        np.asarray(np.load(path, mmap_mode="r")[:, :3], dtype=np.float32)
        if path.suffix.lower() == ".npy"
        else np.loadtxt(path, dtype=np.float32, skiprows=1, usecols=(0, 1, 2))
    )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid XYZ cloud: {path}")
    return points


def wall_frames(model: dict, floor_z: float, ceiling_z: float) -> list[WallFrame]:
    frames = []
    for wall in model["paredes"]:
        start = np.array([wall["ax"], wall["ay"]], dtype=np.float64)
        end = np.array([wall["bx"], wall["by"]], dtype=np.float64)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 0.30:
            continue
        tangent = vector / length
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        frames.append(
            WallFrame(
                wall_id=str(wall["id"]),
                start=start,
                end=end,
                center=(start + end) / 2.0,
                tangent=tangent,
                normal=normal,
                length=length,
                thickness=float(wall["espessura"]),
                z_min=floor_z,
                z_max=ceiling_z,
            )
        )
    return frames


def tile_starts(length: float, width: float, overlap: float) -> list[float]:
    if length <= width:
        return [0.0]
    stride = width * (1.0 - overlap)
    count = max(2, int(math.ceil((length - width) / stride)) + 1)
    return sorted({round(min(index * stride, length - width), 6) for index in range(count)})


def project_cloud(points: np.ndarray, frame: WallFrame, band_m: float):
    # Restrict the expensive wall-frame projection to an axis-aligned envelope.
    # The padding fully contains the oriented wall band, so this changes runtime
    # without changing the selected points.
    padding = band_m + 0.15
    minimum_xy = np.minimum(frame.start, frame.end) - padding
    maximum_xy = np.maximum(frame.start, frame.end) + padding
    coarse_keep = (
        (points[:, 0] >= minimum_xy[0])
        & (points[:, 0] <= maximum_xy[0])
        & (points[:, 1] >= minimum_xy[1])
        & (points[:, 1] <= maximum_xy[1])
        & (points[:, 2] >= frame.z_min - 0.10)
        & (points[:, 2] <= max(frame.z_min + 4.0, frame.z_max + 0.20))
    )
    local_points = points[coarse_keep]
    relative_xy = local_points[:, :2] - frame.start
    along = relative_xy @ frame.tangent
    normal = (local_points[:, :2] - frame.center) @ frame.normal
    height = local_points[:, 2] - frame.z_min
    keep = (
        (along >= -0.10)
        & (along <= frame.length + 0.10)
        & (np.abs(normal) <= band_m)
        & (height >= -0.10)
        & (height <= max(4.0, frame.z_max - frame.z_min + 0.20))
    )
    return along[keep], normal[keep], height[keep]


def thin_wall_points(
    along: np.ndarray,
    normal: np.ndarray,
    height: np.ndarray,
    keep_ratio: float,
    seed: int,
):
    """Uniformly thin a projected wall while keeping overlapping tiles consistent."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("--point-keep-ratio must be in (0, 1]")
    if keep_ratio >= 1.0 or along.size == 0:
        return along, normal, height
    rng = np.random.default_rng(seed)
    keep = rng.random(along.size) < keep_ratio
    return along[keep], normal[keep], height[keep]


def rasterize_tokens(
    along: np.ndarray,
    normal: np.ndarray,
    height: np.ndarray,
    frame: WallFrame,
    tile_start: float,
    token_m: float,
    tile_width_m: float,
    tile_height_m: float,
):
    width_tokens = int(round(tile_width_m / token_m))
    height_tokens = int(round(tile_height_m / token_m))
    x_index = np.floor((along - tile_start) / token_m).astype(np.int32)
    z_index = np.floor(height / token_m).astype(np.int32)
    valid = (
        (x_index >= 0)
        & (x_index < width_tokens)
        & (z_index >= 0)
        & (z_index < height_tokens)
    )
    x_index = x_index[valid]
    z_index = z_index[valid]
    normal = normal[valid]

    counts = np.zeros((height_tokens, width_tokens), dtype=np.float32)
    np.add.at(counts, (z_index, x_index), 1.0)
    minimum = np.full_like(counts, np.inf)
    maximum = np.full_like(counts, -np.inf)
    if normal.size:
        np.minimum.at(minimum, (z_index, x_index), normal)
        np.maximum.at(maximum, (z_index, x_index), normal)
    span = np.where(counts >= 2.0, maximum - minimum, 0.0)
    span[~np.isfinite(span)] = 0.0

    tolerance = max(0.035, min(0.10, frame.thickness * 0.35))
    face_a = np.zeros_like(counts)
    face_b = np.zeros_like(counts)
    if normal.size:
        mask_a = np.abs(normal + frame.thickness / 2.0) <= tolerance
        mask_b = np.abs(normal - frame.thickness / 2.0) <= tolerance
        np.add.at(face_a, (z_index[mask_a], x_index[mask_a]), 1.0)
        np.add.at(face_b, (z_index[mask_b], x_index[mask_b]), 1.0)
    dual_face = np.minimum(face_a, face_b)

    density = np.clip(np.log1p(counts) / np.log1p(12.0), 0.0, 1.0)
    depth = np.clip(span / 0.60, 0.0, 1.0)
    dual = np.clip(np.log1p(dual_face) / np.log1p(4.0), 0.0, 1.0)
    token_image = np.uint8(np.round(255.0 * np.flipud(np.stack((dual, depth, density), axis=-1))))
    stats = {
        "points": int(normal.size),
        "occupied_tokens": int(np.count_nonzero(counts)),
        "dual_face_tokens": int(np.count_nonzero(dual_face)),
        "max_points_per_token": int(np.max(counts)) if counts.size else 0,
    }
    return token_image, stats


def metric_iou(first: dict, second: dict) -> float:
    x0 = max(first["s_min"], second["s_min"])
    x1 = min(first["s_max"], second["s_max"])
    z0 = max(first["z_min"], second["z_min"])
    z1 = min(first["z_max"], second["z_max"])
    intersection = max(0.0, x1 - x0) * max(0.0, z1 - z0)
    first_area = (first["s_max"] - first["s_min"]) * (first["z_max"] - first["z_min"])
    second_area = (second["s_max"] - second["s_min"]) * (second["z_max"] - second["z_min"])
    return intersection / max(first_area + second_area - intersection, 1e-9)


def suppress_tile_duplicates(candidates: list[dict]) -> list[dict]:
    selected = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        duplicate = any(
            candidate["wall_id"] == other["wall_id"]
            and candidate["class"] == other["class"]
            and metric_iou(candidate, other) >= 0.45
            for other in selected
        )
        if not duplicate:
            selected.append(candidate)
    return sorted(selected, key=lambda item: (item["wall_id"], item["s_min"]))


def physical_post_filter(candidates: list[dict], frames: dict[str, WallFrame], floor_z: float, ceiling_z: float) -> list[dict]:
    """Remove tile-edge artifacts and conflicting class boxes without using ground truth."""
    plausible = []
    for candidate in candidates:
        frame = frames[candidate["wall_id"]]
        if candidate["s_min"] < 0.10 or candidate["s_max"] > frame.length - 0.10:
            continue
        if candidate["z_min"] < floor_z - 0.10 or candidate["z_max"] > ceiling_z + 0.10:
            continue
        if not (0.35 <= candidate["s_max"] - candidate["s_min"] <= 3.50):
            continue
        if not (0.50 <= candidate["z_max"] - candidate["z_min"] <= 3.50):
            continue
        plausible.append(candidate)

    selected = []
    for candidate in sorted(plausible, key=lambda item: item["confidence"], reverse=True):
        conflicts = any(
            candidate["wall_id"] == other["wall_id"]
            and metric_iou(candidate, other) >= 0.60
            for other in selected
        )
        if not conflicts:
            selected.append(candidate)
    return sorted(selected, key=lambda item: (item["wall_id"], item["s_min"]))


def correct_raised_door_classes(candidates: list[dict], floor_z: float, tolerance_m: float) -> list[dict]:
    """Apply the physical invariant that a door opening must reach the floor."""
    corrected = []
    for candidate in candidates:
        item = dict(candidate)
        item["model_class"] = item["class"]
        if item["class"] == "door" and item["z_min"] > floor_z + tolerance_m:
            item["class"] = "window"
            item["class_correction"] = "raised opening does not reach floor"
        corrected.append(item)
    return corrected


def draw_tile_review(image: np.ndarray, tile: dict, candidates: list[dict], args) -> np.ndarray:
    review = image.copy()
    for meter in np.arange(0.0, args.tile_width_m + 1e-6, 1.0):
        x = int(round(meter / args.tile_width_m * review.shape[1]))
        cv2.line(review, (x, 0), (x, review.shape[0] - 1), (55, 55, 55), 1)
    for meter in np.arange(0.0, args.tile_height_m + 1e-6, 1.0):
        y = int(round((1.0 - meter / args.tile_height_m) * review.shape[0]))
        cv2.line(review, (0, y), (review.shape[1] - 1, y), (55, 55, 55), 1)
    for candidate in candidates:
        left, top, right, bottom = candidate["pixel_box"]
        color = CLASS_COLORS[candidate["class"]]
        cv2.rectangle(review, (left, top), (right, bottom), color, 3)
        cv2.putText(
            review,
            f'{candidate["class"]} {candidate["confidence"]:.2f}',
            (max(2, left), max(24, top - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(review, (0, 0), (review.shape[1] - 1, 36), (15, 15, 15), -1)
    cv2.putText(
        review,
        f'{tile["wall_id"]} | s={tile["tile_start_m"]:.2f}-{tile["tile_start_m"] + args.tile_width_m:.2f}m',
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return review


def contact_sheet(paths: list[Path], output: Path) -> None:
    cards = [cv2.resize(cv2.imread(str(path)), (640, 320), interpolation=cv2.INTER_AREA) for path in paths]
    blank = np.zeros_like(cards[0])
    while len(cards) % 3:
        cards.append(blank.copy())
    rows = [np.hstack(cards[index:index + 3]) for index in range(0, len(cards), 3)]
    cv2.imwrite(str(output), np.vstack(rows))


def plan_overview(frames: list[WallFrame], candidates: list[dict], output: Path, title: str) -> None:
    all_xy = np.vstack([np.vstack((frame.start, frame.end)) for frame in frames])
    minimum = np.min(all_xy, axis=0) - 1.0
    maximum = np.max(all_xy, axis=0) + 1.0
    width, height = 1600, 1000
    scale = min((width - 100) / max(maximum[0] - minimum[0], 1e-9), (height - 100) / max(maximum[1] - minimum[1], 1e-9))

    def pixel(point):
        x = int(round(50 + (point[0] - minimum[0]) * scale))
        y = int(round(height - 50 - (point[1] - minimum[1]) * scale))
        return x, y

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    by_id = {frame.wall_id: frame for frame in frames}
    for frame in frames:
        cv2.line(canvas, pixel(frame.start), pixel(frame.end), (80, 80, 80), 7, cv2.LINE_AA)
        cv2.putText(canvas, frame.wall_id, pixel((frame.start + frame.end) / 2.0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    for index, candidate in enumerate(candidates, start=1):
        frame = by_id[candidate["wall_id"]]
        center = frame.start + frame.tangent * candidate["s_center"]
        color = CLASS_COLORS[candidate["class"]]
        cv2.circle(canvas, pixel(center), 10, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, f'{index}:{candidate["class"]} {candidate["confidence"]:.2f}', (pixel(center)[0] + 12, pixel(center)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f'{title} | YOLO-World-M | {len(candidates)} deteccoes >= limiar', (40, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / "images"
    reviews_dir = args.output / "reviews"
    images_dir.mkdir(exist_ok=True)
    reviews_dir.mkdir(exist_ok=True)

    points = load_xyz(args.cloud_xyz)
    model_data = json.loads(args.review_model.read_text(encoding="utf-8"))
    frames = wall_frames(model_data, args.floor_z, args.ceiling_z)
    tiles = []
    image_paths = []
    frame_lookup = {frame.wall_id: frame for frame in frames}
    for frame in frames:
        along, normal, height = project_cloud(points, frame, args.wall_band_m)
        projected_count = int(along.size)
        wall_seed = args.thinning_seed + sum((index + 1) * ord(char) for index, char in enumerate(frame.wall_id))
        along, normal, height = thin_wall_points(
            along,
            normal,
            height,
            args.point_keep_ratio,
            wall_seed,
        )
        for tile_index, start in enumerate(tile_starts(frame.length, args.tile_width_m, args.overlap)):
            tokens, stats = rasterize_tokens(
                along,
                normal,
                height,
                frame,
                start,
                args.token_m,
                args.tile_width_m,
                args.tile_height_m,
            )
            image = cv2.resize(tokens, (1280, 640), interpolation=cv2.INTER_NEAREST)
            path = images_dir / f"{frame.wall_id}_t{tile_index:02d}.png"
            cv2.imwrite(str(path), image)
            stats["projected_wall_points_before_thinning"] = projected_count
            stats["projected_wall_points_after_thinning"] = int(along.size)
            stats["point_keep_ratio"] = args.point_keep_ratio
            tiles.append({"wall_id": frame.wall_id, "tile_index": tile_index, "tile_start_m": start, "image": str(path.resolve()), "stats": stats})
            image_paths.append(path)

    detector = YOLOWorld(str(args.weights.resolve()))
    # A Python list is treated by Ultralytics as one in-memory source batch;
    # explicitly chunk it so large buildings do not warm up all tiles at once.
    results = []
    for first in range(0, len(image_paths), args.batch_size):
        chunk = image_paths[first:first + args.batch_size]
        results.extend(
            detector.predict(
                source=[str(path) for path in chunk],
                imgsz=1280,
                batch=len(chunk),
                device=0,
                conf=0.01,
                iou=0.50,
                verbose=False,
            )
        )
    raw_candidates = []
    for tile, path, result in zip(tiles, image_paths, results):
        frame = frame_lookup[tile["wall_id"]]
        if result.boxes is None:
            continue
        for xyxy, confidence, class_index in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            class_name = CLASS_NAMES[int(class_index)]
            left, top, right, bottom = [float(value) for value in xyxy]
            s_min = tile["tile_start_m"] + left / 1280.0 * args.tile_width_m
            s_max = tile["tile_start_m"] + right / 1280.0 * args.tile_width_m
            z_min = args.floor_z + (1.0 - bottom / 640.0) * args.tile_height_m
            z_max = args.floor_z + (1.0 - top / 640.0) * args.tile_height_m
            s_center = (s_min + s_max) / 2.0
            center_xy = frame.start + frame.tangent * s_center
            candidate = {
                "wall_id": frame.wall_id,
                "class": class_name,
                "confidence": round(float(confidence), 6),
                "s_min": round(float(s_min), 4),
                "s_max": round(float(s_max), 4),
                "s_center": round(float(s_center), 4),
                "z_min": round(float(z_min), 4),
                "z_max": round(float(z_max), 4),
                "world_center": [round(float(center_xy[0]), 4), round(float(center_xy[1]), 4), round(float((z_min + z_max) / 2.0), 4)],
                "tile": path.name,
                "pixel_box": [int(round(left)), int(round(top)), int(round(right)), int(round(bottom))],
            }
            raw_candidates.append(candidate)

    all_deduplicated = suppress_tile_duplicates(raw_candidates)
    raw_selected = [candidate for candidate in all_deduplicated if candidate["confidence"] >= args.confidence]
    selected = physical_post_filter(raw_selected, frame_lookup, args.floor_z, args.ceiling_z)
    if args.relabel_raised_doors:
        selected = correct_raised_door_classes(selected, args.floor_z, args.door_floor_tolerance_m)
    tile_candidates: dict[str, list[dict]] = {str(path): [] for path in image_paths}
    for candidate in selected:
        tile_candidates[str(images_dir / candidate["tile"])].append(candidate)
    review_paths = []
    for tile, path in zip(tiles, image_paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        review = draw_tile_review(image, tile, tile_candidates[str(path)], args)
        review_path = reviews_dir / path.name
        cv2.imwrite(str(review_path), review)
        review_paths.append(review_path)

    contact_sheet(review_paths, args.output / f"{args.output_prefix}_wall_contact_sheet.png")
    plan_overview(frames, selected, args.output / f"{args.output_prefix}_plan_overview.png", args.title)
    payload = {
        "schema": "cloud2bim.yoloworld-real-inference.v1",
        "source_cloud_xyz": str(args.cloud_xyz.resolve()),
        "source_e57": str(args.source_e57.resolve()) if args.source_e57 else None,
        "weights": str(args.weights.resolve()),
        "confidence_threshold": args.confidence,
        "point_keep_ratio": args.point_keep_ratio,
        "thinning_seed": args.thinning_seed,
        "relabel_raised_doors": args.relabel_raised_doors,
        "door_floor_tolerance_m": args.door_floor_tolerance_m,
        "floor_z": args.floor_z,
        "ceiling_z": args.ceiling_z,
        "point_count": int(points.shape[0]),
        "wall_count": len(frames),
        "tile_count": len(tiles),
        "counts": {
            "door": sum(candidate["class"] == "door" for candidate in selected),
            "window": sum(candidate["class"] == "window" for candidate in selected),
            "total": len(selected),
        },
        "raw_counts_before_physical_filter": {
            "door": sum(candidate["class"] == "door" for candidate in raw_selected),
            "window": sum(candidate["class"] == "window" for candidate in raw_selected),
            "total": len(raw_selected),
        },
        "threshold_sweep": {
            str(threshold): sum(candidate["confidence"] >= threshold for candidate in all_deduplicated)
            for threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.40, 0.60)
        },
        "tiles": tiles,
        "raw_detections_before_physical_filter": raw_selected,
        "detections": selected,
    }
    (args.output / f"{args.output_prefix}_detections.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "point_count",
                    "wall_count",
                    "tile_count",
                    "raw_counts_before_physical_filter",
                    "counts",
                    "threshold_sweep",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
