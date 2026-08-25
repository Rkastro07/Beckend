"""Run opening inference for all cloud tiles with one persistent YOLO model.

Wall geometry remains entirely deterministic.  Token rasters are produced per
tile, but inference is streamed in batches and no PNG intermediates are written
unless explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOWorld

from run_yoloworld_wall_tokens_real import (
    CLASS_NAMES,
    correct_raised_door_classes,
    load_xyz,
    physical_post_filter,
    project_cloud,
    rasterize_tokens,
    suppress_tile_duplicates,
    thin_wall_points,
    tile_starts,
    wall_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("wall_models_index", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--storey", default="S01")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--floor-z", type=float, default=0.1907)
    parser.add_argument("--ceiling-z", type=float, default=3.787)
    parser.add_argument("--token-m", type=float, default=0.05)
    parser.add_argument("--tile-width-m", type=float, default=12.8)
    parser.add_argument("--tile-height-m", type=float, default=4.0)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--wall-band-m", type=float, default=0.35)
    parser.add_argument("--point-keep-ratio", type=float, default=0.18)
    parser.add_argument("--thinning-seed", type=int, default=20260824)
    parser.add_argument("--door-floor-tolerance-m", type=float, default=0.35)
    parser.add_argument("--keep-token-images", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def signature(args: argparse.Namespace, tile: dict) -> dict:
    cloud = Path(tile["cloud_xyz"])
    model = Path(tile["model"])
    return {
        "weights": str(args.weights.resolve()),
        "weights_mtime_ns": args.weights.stat().st_mtime_ns,
        "cloud": str(cloud.resolve()),
        "cloud_mtime_ns": cloud.stat().st_mtime_ns,
        "wall_model": str(model.resolve()),
        "wall_model_mtime_ns": model.stat().st_mtime_ns,
        "storey": args.storey,
        "confidence": args.confidence,
        "floor_z": args.floor_z,
        "ceiling_z": args.ceiling_z,
        "token_m": args.token_m,
        "tile_width_m": args.tile_width_m,
        "tile_height_m": args.tile_height_m,
        "overlap": args.overlap,
        "wall_band_m": args.wall_band_m,
        "point_keep_ratio": args.point_keep_ratio,
        "thinning_seed": args.thinning_seed,
        "door_floor_tolerance_m": args.door_floor_tolerance_m,
    }


def output_path(args: argparse.Namespace, tile_id: str) -> Path:
    return args.output / tile_id / "tiled_yoloworld_m_detections.json"


def cache_valid(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("run_signature") == expected
    except (OSError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    index = json.loads(args.wall_models_index.read_text(encoding="utf-8"))
    ready = [item for item in index["tiles"] if item.get("status") == "ready"]
    pending_tiles = []
    reused = []
    signatures = {}
    for tile in ready:
        tile_id = tile["tile_id"]
        expected = signature(args, tile)
        signatures[tile_id] = expected
        if not args.force and cache_valid(output_path(args, tile_id), expected):
            reused.append(tile_id)
        else:
            pending_tiles.append(tile)

    started = time.perf_counter()
    raw_by_tile: dict[str, list[dict]] = defaultdict(list)
    frames_by_tile = {}
    stats_by_tile = defaultdict(list)
    point_counts = {}
    token_counts = defaultdict(int)
    detector = YOLOWorld(str(args.weights.resolve())) if pending_tiles else None
    pending_images: list[np.ndarray] = []
    pending_records: list[dict] = []

    def flush() -> None:
        if not pending_images:
            return
        results = detector.predict(
            source=list(pending_images),
            imgsz=1280,
            batch=len(pending_images),
            device=args.device,
            conf=0.01,
            iou=0.50,
            verbose=False,
        )
        for record, result in zip(pending_records, results):
            frame = record["frame"]
            tile_meta = record["tile_meta"]
            tile_id = record["tile_id"]
            if result.boxes is None:
                continue
            boxes = zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            )
            for xyxy, confidence, class_index in boxes:
                class_name = CLASS_NAMES[int(class_index)]
                left, top, right, bottom = [float(value) for value in xyxy]
                s_min = tile_meta["tile_start_m"] + left / 1280.0 * args.tile_width_m
                s_max = tile_meta["tile_start_m"] + right / 1280.0 * args.tile_width_m
                z_min = args.floor_z + (1.0 - bottom / 640.0) * args.tile_height_m
                z_max = args.floor_z + (1.0 - top / 640.0) * args.tile_height_m
                s_center = (s_min + s_max) / 2.0
                center_xy = frame.start + frame.tangent * s_center
                raw_by_tile[tile_id].append({
                    "wall_id": frame.wall_id,
                    "class": class_name,
                    "confidence": round(float(confidence), 6),
                    "s_min": round(float(s_min), 4),
                    "s_max": round(float(s_max), 4),
                    "s_center": round(float(s_center), 4),
                    "z_min": round(float(z_min), 4),
                    "z_max": round(float(z_max), 4),
                    "world_center": [
                        round(float(center_xy[0]), 4),
                        round(float(center_xy[1]), 4),
                        round(float((z_min + z_max) / 2.0), 4),
                    ],
                    "tile": tile_meta["name"],
                    "pixel_box": [int(round(value)) for value in (left, top, right, bottom)],
                })
        pending_images.clear()
        pending_records.clear()

    for tile in pending_tiles:
        tile_id = tile["tile_id"]
        model_data = json.loads(Path(tile["model"]).read_text(encoding="utf-8"))
        model_data["paredes"] = [
            wall for wall in model_data["paredes"]
            if f'-{args.storey}-' in wall["source_wall_id"]
        ]
        frames = wall_frames(model_data, args.floor_z, args.ceiling_z)
        frames_by_tile[tile_id] = {frame.wall_id: frame for frame in frames}
        points = load_xyz(Path(tile["cloud_xyz"]))
        point_counts[tile_id] = int(points.shape[0])
        token_dir = args.output / tile_id / "images"
        if args.keep_token_images:
            token_dir.mkdir(parents=True, exist_ok=True)

        for frame in frames:
            along, normal, height = project_cloud(points, frame, args.wall_band_m)
            projected_count = int(along.size)
            wall_seed = args.thinning_seed + sum(
                (index + 1) * ord(char) for index, char in enumerate(frame.wall_id)
            )
            along, normal, height = thin_wall_points(
                along, normal, height, args.point_keep_ratio, wall_seed
            )
            for tile_index, start in enumerate(tile_starts(frame.length, args.tile_width_m, args.overlap)):
                tokens, stats = rasterize_tokens(
                    along, normal, height, frame, start, args.token_m,
                    args.tile_width_m, args.tile_height_m,
                )
                image = cv2.resize(tokens, (1280, 640), interpolation=cv2.INTER_NEAREST)
                name = f"{frame.wall_id}_t{tile_index:02d}.png"
                if args.keep_token_images:
                    cv2.imwrite(str(token_dir / name), image)
                stats.update({
                    "projected_wall_points_before_thinning": projected_count,
                    "projected_wall_points_after_thinning": int(along.size),
                    "point_keep_ratio": args.point_keep_ratio,
                })
                stats_by_tile[tile_id].append(stats)
                token_counts[tile_id] += 1
                pending_images.append(image)
                pending_records.append({
                    "tile_id": tile_id,
                    "frame": frame,
                    "tile_meta": {
                        "name": name,
                        "tile_start_m": start,
                    },
                })
                if len(pending_images) >= args.batch_size:
                    flush()
        del points
    flush()

    generated = []
    for tile in pending_tiles:
        tile_id = tile["tile_id"]
        frame_lookup = frames_by_tile[tile_id]
        all_deduplicated = suppress_tile_duplicates(raw_by_tile[tile_id])
        raw_selected = [
            candidate for candidate in all_deduplicated
            if candidate["confidence"] >= args.confidence
        ]
        selected = physical_post_filter(
            raw_selected, frame_lookup, args.floor_z, args.ceiling_z
        )
        selected = correct_raised_door_classes(
            selected, args.floor_z, args.door_floor_tolerance_m
        )
        payload = {
            "schema": "cloud2bim.yoloworld-tiled-batch.v1",
            "run_signature": signatures[tile_id],
            "batch_model_load": "single persistent model for every pending tile",
            "source_cloud_xyz": tile["cloud_xyz"],
            "weights": str(args.weights.resolve()),
            "point_count": point_counts[tile_id],
            "wall_count": len(frame_lookup),
            "tile_count": token_counts[tile_id],
            "token_images_persisted": args.keep_token_images,
            "counts": {
                "door": sum(item["class"] == "door" for item in selected),
                "window": sum(item["class"] == "window" for item in selected),
                "total": len(selected),
            },
            "raw_counts_before_physical_filter": {
                "door": sum(item["class"] == "door" for item in raw_selected),
                "window": sum(item["class"] == "window" for item in raw_selected),
                "total": len(raw_selected),
            },
            "token_stats": stats_by_tile[tile_id],
            "raw_detections_before_physical_filter": raw_selected,
            "detections": selected,
        }
        path = output_path(args, tile_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        generated.append(tile_id)

    elapsed = time.perf_counter() - started
    summary = {
        "pending_tiles": len(pending_tiles),
        "generated_tiles": generated,
        "reused_tiles": reused,
        "model_loads": 1 if pending_tiles else 0,
        "elapsed_seconds": round(elapsed, 3),
    }
    (args.output / "batch_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
