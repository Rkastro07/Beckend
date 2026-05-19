"""Pré-voxeliza um PLY pesado pra reduzir antes de outras etapas.

Uso:
    python -m pipeline_v2._prevoxel <ply_in> <ply_out> [voxel]
"""
import sys
from pathlib import Path
import numpy as np
import open3d as o3d


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    ply_in = sys.argv[1]
    ply_out = sys.argv[2]
    voxel = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03

    print(f"Lendo {ply_in}...")
    pcd = o3d.io.read_point_cloud(ply_in)
    n_in = len(pcd.points)
    print(f"  {n_in:,} pts")

    print(f"Voxelizando ({voxel}m)...")
    pcd_v = pcd.voxel_down_sample(voxel)
    n_out = len(pcd_v.points)
    print(f"  {n_out:,} pts (reduz {n_in/max(n_out,1):.1f}x)")

    print(f"Salvando {ply_out}...")
    Path(ply_out).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(ply_out, pcd_v, write_ascii=False)
    sz = Path(ply_out).stat().st_size / 1024 / 1024
    print(f"  {sz:.1f} MB")


if __name__ == "__main__":
    main()
