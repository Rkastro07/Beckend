"""Split an XYZ point cloud into overlapping XY tiles without changing coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cloud_xyz", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--core-size", type=float, default=12.0)
    parser.add_argument("--halo", type=float, default=2.0)
    parser.add_argument("--minimum-core-points", type=int, default=50_000)
    parser.add_argument(
        "--tile-format",
        choices=("npy", "xyz"),
        default="npy",
        help="NPY is substantially faster and smaller; XYZ remains available for inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.core_size <= 0.0 or args.halo < 0.0:
        raise ValueError("core-size must be positive and halo cannot be negative")
    args.output.mkdir(parents=True, exist_ok=True)
    tiles_dir = args.output / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    cloud = (
        np.load(args.cloud_xyz, mmap_mode="r")
        if args.cloud_xyz.suffix.lower() == ".npy"
        else np.loadtxt(args.cloud_xyz, dtype=np.float32, skiprows=1)
    )
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(f"Invalid XYZ cloud: {args.cloud_xyz}")
    minimum = np.min(cloud[:, :2], axis=0)
    maximum = np.max(cloud[:, :2], axis=0)
    origin = np.floor(minimum / args.core_size) * args.core_size
    limit = np.ceil(maximum / args.core_size) * args.core_size

    entries = []
    x_starts = np.arange(origin[0], limit[0], args.core_size)
    y_starts = np.arange(origin[1], limit[1], args.core_size)
    for row, y0 in enumerate(y_starts):
        for column, x0 in enumerate(x_starts):
            core = (
                (cloud[:, 0] >= x0)
                & (cloud[:, 0] < x0 + args.core_size)
                & (cloud[:, 1] >= y0)
                & (cloud[:, 1] < y0 + args.core_size)
            )
            core_count = int(np.count_nonzero(core))
            if core_count < args.minimum_core_points:
                continue
            halo = (
                (cloud[:, 0] >= x0 - args.halo)
                & (cloud[:, 0] < x0 + args.core_size + args.halo)
                & (cloud[:, 1] >= y0 - args.halo)
                & (cloud[:, 1] < y0 + args.core_size + args.halo)
            )
            tile_id = f"tile_r{row:02d}_c{column:02d}"
            tile_path = tiles_dir / f"{tile_id}.{args.tile_format}"
            values = cloud[halo]
            if args.tile_format == "npy":
                np.save(tile_path, np.asarray(values, dtype=np.float32), allow_pickle=False)
            else:
                formats = ["%.4f", "%.4f", "%.4f"] + ["%.0f"] * (values.shape[1] - 3)
                header = "x y z" + (" r g b" if values.shape[1] >= 6 else "")
                np.savetxt(tile_path, values, fmt=formats, header=header, comments="")
            entries.append(
                {
                    "tile_id": tile_id,
                    "path": str(tile_path.resolve()),
                    "core": [x0, y0, x0 + args.core_size, y0 + args.core_size],
                    "halo": [
                        x0 - args.halo,
                        y0 - args.halo,
                        x0 + args.core_size + args.halo,
                        y0 + args.core_size + args.halo,
                    ],
                    "core_point_count": core_count,
                    "tile_point_count": int(values.shape[0]),
                }
            )

    payload = {
        "schema": "cloud2bim.xy-tiles.v1",
        "source": str(args.cloud_xyz.resolve()),
        "source_mtime_ns": args.cloud_xyz.stat().st_mtime_ns,
        "point_count": int(cloud.shape[0]),
        "core_size_m": args.core_size,
        "halo_m": args.halo,
        "minimum_core_points": args.minimum_core_points,
        "tile_format": args.tile_format,
        "bounds": [*minimum.tolist(), *maximum.tolist()],
        "tiles": entries,
    }
    manifest = args.output / "tiles_manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tiles": len(entries), "manifest": str(manifest.resolve())}, indent=2))


if __name__ == "__main__":
    main()
