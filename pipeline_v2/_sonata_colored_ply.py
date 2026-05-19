"""Roda Sonata num PLY e salva como PLY colorido por classe.

Uso:
    python -m pipeline_v2._sonata_colored_ply <ply_in> [ply_out] [voxel]
"""
import sys
from pathlib import Path
import numpy as np
import open3d as o3d

from pipeline_v2 import sonata_runner

CLASS_NAMES = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower_curtain", "toilet", "sink", "bathtub",
    "otherfurniture",
)

# Cores RGB [0-1] por classe (paleta consistente)
COLORS = {
    "wall":           (0.60, 0.30, 0.90),
    "floor":          (0.20, 0.70, 0.30),
    "cabinet":        (0.70, 0.50, 0.30),
    "bed":            (0.90, 0.70, 0.70),
    "chair":          (0.50, 0.50, 0.70),
    "sofa":           (0.80, 0.50, 0.50),
    "table":          (0.60, 0.60, 0.40),
    "door":           (1.00, 0.50, 0.10),
    "window":         (0.20, 0.50, 1.00),
    "bookshelf":      (0.50, 0.30, 0.20),
    "picture":        (0.90, 0.90, 0.40),
    "counter":        (0.60, 0.40, 0.30),
    "desk":           (0.70, 0.40, 0.50),
    "curtain":        (0.40, 0.60, 0.60),
    "refrigerator":   (0.80, 0.80, 0.90),
    "shower_curtain": (0.30, 0.80, 0.80),
    "toilet":         (0.80, 0.30, 0.80),
    "sink":           (0.40, 0.30, 0.60),
    "bathtub":        (0.30, 0.60, 0.80),
    "otherfurniture": (0.50, 0.50, 0.50),
}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ply_in = sys.argv[1]
    voxel  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    ply_out = sys.argv[2] if len(sys.argv) > 2 else \
        str(Path(ply_in).parent / f"{Path(ply_in).stem}_sonata_v{int(voxel*100):02d}.ply")

    print(f"PLY in : {ply_in}")
    print(f"voxel  : {voxel}m")
    print(f"PLY out: {ply_out}")
    print()

    print("[1/3] Rodando Sonata (cache hit pula se for o mesmo arquivo + voxel)...")
    r = sonata_runner.run_sonata(ply_in, voxel=voxel, verbose=True)
    pts_voxel = r["pts_voxel"]
    pred      = r["pred"]
    print(f"   {len(pts_voxel):,} pts no voxelize")
    print(f"   {len(pred):,} predições")

    n = min(len(pts_voxel), len(pred))
    pts = pts_voxel[:n]
    pred = pred[:n]
    if n < len(pts_voxel):
        print(f"   ATENÇÃO: truncado a {n} (pts_voxel={len(pts_voxel)}, pred={len(pred)})")

    print()
    print("[2/3] Distribuição por classe:")
    from collections import Counter
    dist = Counter(CLASS_NAMES[int(c)] for c in pred)
    for cls, cnt in dist.most_common():
        print(f"   {cls:18s} {cnt:>10,} pts ({100*cnt/n:.1f}%)")

    print()
    print("[3/3] Escrevendo PLY colorido por classe...")
    cols = np.array([
        COLORS.get(CLASS_NAMES[int(c)], (0.5, 0.5, 0.5)) for c in pred
    ], dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    Path(ply_out).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(ply_out, pcd, write_ascii=False)
    sz = Path(ply_out).stat().st_size / 1024 / 1024
    print(f"   PLY salvo: {ply_out}")
    print(f"   {n:,} pts, ~{sz:.1f} MB")


if __name__ == "__main__":
    main()
