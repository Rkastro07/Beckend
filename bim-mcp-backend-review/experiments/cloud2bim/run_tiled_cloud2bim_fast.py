"""Accelerated tiled Cloud-to-BIM orchestration.

The wall detector is geometric and runs in parallel per overlapping XY tile.
YOLO is loaded once and is used only for door/window tokens.  IFC authoring is
not performed for temporary tiles.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
CLOUD2ENTITIES = HERE / "cloud2bim_patched" / "cloud2entities.py"
BUILD_TILES = HERE / "build_overlapping_xy_tiles.py"
PREPARE_WALLS = HERE / "prepare_tiled_wall_models.py"
BATCH_YOLO = HERE / "run_yoloworld_wall_tokens_tiled_batch.py"
STITCH = HERE / "stitch_tiled_cloud2bim.py"
GEOMETRY_PYTHON = Path(getattr(sys, "_base_executable", sys.executable)).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cloud_xyz", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    parser.add_argument("--core-size", type=float, default=12.0)
    parser.add_argument("--halo", type=float, default=2.0)
    parser.add_argument("--minimum-core-points", type=int, default=50_000)
    parser.add_argument("--tile-format", choices=("npy", "xyz"), default="npy")
    parser.add_argument("--slab-threshold", type=float, default=0.5)
    parser.add_argument("--resolution", type=float, default=0.039)
    parser.add_argument("--minimum-wall-length", type=float, default=0.30)
    parser.add_argument("--yolo-batch-size", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--point-keep-ratio", type=float, default=0.18)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--force-tiles", action="store_true")
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--force-yolo", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def manifest_matches(path: Path, args: argparse.Namespace) -> bool:
    if args.force_tiles or not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            Path(payload["source"]).resolve() == args.cloud_xyz.resolve()
            and payload.get("source_mtime_ns") == args.cloud_xyz.stat().st_mtime_ns
            and payload.get("core_size_m") == args.core_size
            and payload.get("halo_m") == args.halo
            and payload.get("minimum_core_points") == args.minimum_core_points
            and payload.get("tile_format") == args.tile_format
            and all(Path(tile["path"]).exists() for tile in payload.get("tiles", []))
        )
    except (OSError, ValueError, KeyError):
        return False


def config_text(tile: dict, output: Path, args: argparse.Namespace) -> str:
    source = Path(tile["path"]).resolve().as_posix()
    dummy_ifc = (output / f'{tile["tile_id"]}_temporary.ifc').resolve().as_posix()
    name = tile["tile_id"]
    grid_coefficient = max(3, min(6, round(0.10 / args.resolution)))
    return f"""e57_input: false
xyz_files:
  - {source}
exterior_scan: false
dilute: false
dilution_factor: 10
pc_resolution: {args.resolution}
grid_coefficient: {grid_coefficient}

bfs_thickness: 0.3
tfs_thickness: 0.3

min_wall_length: {args.minimum_wall_length}
min_wall_thickness: 0.05
max_wall_thickness: 0.75
exterior_walls_thickness: 0.3

output_ifc: {dummy_ifc}
ifc_project_name: {name}
ifc_project_long_name: Tiled geometric detection ({name})
ifc_project_version: fast-v1

ifc_author_name: Rafael
ifc_author_surname: Corrigliano
ifc_author_organization: Beckend

ifc_building_name: {name}
ifc_building_type: Building
ifc_building_phase: As-built

ifc_site_latitude: [0, 0, 0]
ifc_site_longitude: [0, 0, 0]
ifc_site_elevation: 0.0
material_for_objects: Concrete
"""


def geometry_signature(tile: dict, args: argparse.Namespace) -> dict:
    source = Path(tile["path"])
    return {
        "source": str(source.resolve()),
        "source_mtime_ns": source.stat().st_mtime_ns,
        "cloud2entities_mtime_ns": CLOUD2ENTITIES.stat().st_mtime_ns,
        "wall_detector_mtime_ns": (
            HERE / "cloud2bim_patched" / "wall_detector_v2.py"
        ).stat().st_mtime_ns,
        "slab_threshold": args.slab_threshold,
        "resolution": args.resolution,
        "minimum_wall_length": args.minimum_wall_length,
        "geometry_only": True,
    }


def process_geometry(tile: dict, root: Path, args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    tile_id = tile["tile_id"]
    output = root / tile_id
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = output / "wall_diagnostics.csv"
    cache_path = output / "geometry_run.json"
    expected = geometry_signature(tile, args)
    if not args.force_geometry and diagnostics.exists() and cache_path.exists():
        try:
            if json.loads(cache_path.read_text(encoding="utf-8")).get("signature") == expected:
                return {"tile_id": tile_id, "status": "reused", "seconds": 0.0}
        except (OSError, ValueError):
            pass

    config = output / "config.yaml"
    config.write_text(config_text(tile, output, args), encoding="utf-8")
    for relative in ("images/pdf", "images/wall_outputs_images", "output_xyz"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CLOUD2BIM_GEOMETRY_ONLY": "1",
        "SLAB_THR": str(args.slab_threshold),
        "SLAB_DETECTOR": "v1_refined",
        "SLAB_WALL_REFERENCE": "v1",
        "WALL_DETECTOR": "v2",
        "WALL_ZLO": "0.1",
        "WALL_ZHI": "0.9",
        "SINGLE_LINE": "1",
        "SINGLE_LINE_MINLEN": "1.5",
        "SINGLE_LINE_THK": "0.15",
        "WALL_CONTOURS": "all",
        "PYTHONIOENCODING": "utf-8",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str((output / ".mplconfig").resolve()),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    log_path = output / "pipeline.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(GEOMETRY_PYTHON), "-u", str(CLOUD2ENTITIES), str(config)],
            cwd=output,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0 or not diagnostics.exists():
        raise RuntimeError(f"geometry failed for {tile_id}; see {log_path}")
    elapsed = time.perf_counter() - started
    cache_path.write_text(
        json.dumps({"signature": expected, "elapsed_seconds": elapsed}, indent=2),
        encoding="utf-8",
    )
    return {"tile_id": tile_id, "status": "generated", "seconds": round(elapsed, 3)}


def main() -> None:
    args = parse_args()
    args.cloud_xyz = args.cloud_xyz.resolve()
    args.weights = args.weights.resolve()
    args.output = args.output.resolve()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    timings = {}
    total_started = time.perf_counter()

    manifest_path = args.output / "tiles_manifest.json"
    started = time.perf_counter()
    if not manifest_matches(manifest_path, args):
        run([
            sys.executable, str(BUILD_TILES), str(args.cloud_xyz), str(args.output),
            "--core-size", str(args.core_size),
            "--halo", str(args.halo),
            "--minimum-core-points", str(args.minimum_core_points),
            "--tile-format", args.tile_format,
        ])
    timings["tiles_seconds"] = round(time.perf_counter() - started, 3)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    geometry_root = args.output / "geometry"
    geometry_root.mkdir(exist_ok=True)
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_geometry, tile, geometry_root, args): tile["tile_id"]
            for tile in manifest["tiles"]
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    timings["geometry_seconds"] = round(time.perf_counter() - started, 3)

    wall_models = args.output / "wall_models"
    started = time.perf_counter()
    run([
        sys.executable, str(PREPARE_WALLS), str(manifest_path),
        str(geometry_root), str(wall_models), "--ownership", "core-intersection",
    ])
    timings["prepare_seconds"] = round(time.perf_counter() - started, 3)

    yolo_root = args.output / "yolo"
    started = time.perf_counter()
    yolo_command = [
        sys.executable, str(BATCH_YOLO),
        str(wall_models / "wall_models_index.json"),
        str(args.weights), str(yolo_root),
        "--batch-size", str(args.yolo_batch_size),
        "--device", args.device,
        "--point-keep-ratio", str(args.point_keep_ratio),
        "--confidence", str(args.confidence),
    ]
    if args.force_yolo:
        yolo_command.append("--force")
    run(yolo_command)
    timings["yolo_seconds"] = round(time.perf_counter() - started, 3)

    stitched = args.output / "stitched"
    started = time.perf_counter()
    run([
        sys.executable, str(STITCH),
        str(wall_models / "wall_models_index.json"),
        str(yolo_root), str(stitched),
    ])
    timings["stitch_seconds"] = round(time.perf_counter() - started, 3)
    timings["total_seconds"] = round(time.perf_counter() - total_started, 3)
    payload = {
        "schema": "cloud2bim.tiled-fast-run.v1",
        "wall_neural_evaluation": False,
        "yolo_scope": "doors and windows only",
        "temporary_ifc_count": 0,
        "workers": args.workers,
        "geometry_tiles": results,
        "timings": timings,
        "stitched_model": str((stitched / "tiled_stitched_model.json").resolve()),
        "stitched_plan": str((stitched / "tiled_stitched_plan.png").resolve()),
    }
    metrics = args.output / "fast_pipeline_metrics.json"
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
