"""Create per-tile wall models from geometric Cloud-to-BIM diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("tiles_manifest", type=Path)
    parser.add_argument("geometry_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--ownership",
        choices=("core-intersection", "all", "midpoint"),
        default="core-intersection",
        help="Keep segments that actually cross the tile core; avoids midpoint holes and halo-only duplicates.",
    )
    parser.add_argument("--minimum-core-overlap", type=float, default=0.20)
    return parser.parse_args()


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else default


def write_json_if_changed(path: Path, payload: dict) -> None:
    """Preserve mtimes so downstream YOLO cache remains valid."""
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def segment_core_overlap_length(
    ax: float, ay: float, bx: float, by: float,
    x0: float, y0: float, x1: float, y1: float,
) -> float:
    """Length of a segment inside an axis-aligned rectangle (Liang-Barsky)."""
    dx, dy = bx - ax, by - ay
    start, end = 0.0, 1.0
    for p, q in (
        (-dx, ax - x0), (dx, x1 - ax),
        (-dy, ay - y0), (dy, y1 - ay),
    ):
        if abs(p) < 1e-12:
            if q < 0.0:
                return 0.0
            continue
        ratio = q / p
        if p < 0.0:
            start = max(start, ratio)
        else:
            end = min(end, ratio)
        if start > end:
            return 0.0
    return max(0.0, end - start) * ((dx * dx + dy * dy) ** 0.5)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.tiles_manifest.read_text(encoding="utf-8"))
    summary = []
    for tile in manifest["tiles"]:
        tile_id = tile["tile_id"]
        diagnostics = args.geometry_root / tile_id / "wall_diagnostics.csv"
        if not diagnostics.exists():
            summary.append({"tile_id": tile_id, "status": "missing_diagnostics"})
            continue
        with diagnostics.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        x0, y0, x1, y1 = tile["core"]
        walls = []
        for row in rows:
            ax = as_float(row, "start_x")
            ay = as_float(row, "start_y")
            bx = as_float(row, "end_x")
            by = as_float(row, "end_y")
            mx = (ax + bx) / 2.0
            my = (ay + by) / 2.0
            if args.ownership == "midpoint":
                if not (x0 <= mx < x1 and y0 <= my < y1):
                    continue
            elif args.ownership == "core-intersection":
                overlap = segment_core_overlap_length(ax, ay, bx, by, x0, y0, x1, y1)
                if overlap < args.minimum_core_overlap:
                    continue
            local_id = row["reference"]
            walls.append(
                {
                    "id": f"{tile_id}__{local_id}",
                    "source_tile": tile_id,
                    "source_wall_id": local_id,
                    "ax": ax,
                    "ay": ay,
                    "bx": bx,
                    "by": by,
                    "espessura": as_float(row, "thickness", 0.15),
                    "comprimento": as_float(row, "length"),
                    "confidence": row.get("confidence", ""),
                    "review_status": row.get("review_status", ""),
                    "evidence_type": row.get("evidence_type", ""),
                    "detection_score": as_float(row, "detection_score"),
                    "point_count": int(as_float(row, "point_count")),
                }
            )
        tile_output = args.output / tile_id
        tile_output.mkdir(exist_ok=True)
        model_path = tile_output / "wall_model.json"
        write_json_if_changed(
            model_path,
            {
                "schema": "cloud2bim.tiled-wall-model.v1",
                "tile": tile,
                "paredes": walls,
            },
        )
        summary.append(
            {
                "tile_id": tile_id,
                "status": "ready",
                "detected_walls": len(rows),
                "owned_walls": len(walls),
                "model": str(model_path.resolve()),
                "cloud_xyz": str(Path(tile["path"]).resolve()),
                "ownership": args.ownership,
                "minimum_core_overlap": args.minimum_core_overlap,
            }
        )
    payload = {
        "schema": "cloud2bim.tiled-wall-model-index.v1",
        "ownership": args.ownership,
        "minimum_core_overlap": args.minimum_core_overlap,
        "tiles": summary,
    }
    index_path = args.output / "wall_models_index.json"
    write_json_if_changed(index_path, payload)
    print(
        json.dumps(
            {
                "ready_tiles": sum(item["status"] == "ready" for item in summary),
                "owned_walls": sum(item.get("owned_walls", 0) for item in summary),
                "index": str(index_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
