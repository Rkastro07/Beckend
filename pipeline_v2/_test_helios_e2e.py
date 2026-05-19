"""POC end-to-end: IFC → meshes → walkable scanners → HELIOS++ → PLY colorido por tipo.

Uso:
    python -m pipeline_v2._test_helios_e2e [ifc_path] [out_ply]
"""

import sys
import time
from collections import Counter
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import numpy as np
import open3d as o3d

from .helios_wrapper import scan_meshes, compute_walkable_scanners


TIPOS = ["IfcWall", "IfcSlab", "IfcCovering", "IfcColumn", "IfcDoor",
         "IfcWindow", "IfcRoof", "IfcBeam", "IfcStair", "IfcRailing"]

COLORS = {
    "IfcWall":     [0.70, 0.40, 0.90],
    "IfcSlab":     [0.40, 0.80, 0.40],
    "IfcCovering": [0.30, 0.60, 0.30],
    "IfcColumn":   [0.60, 0.60, 0.60],
    "IfcDoor":     [1.00, 0.50, 0.20],
    "IfcWindow":   [0.30, 0.50, 0.90],
    "IfcRoof":     [0.80, 0.30, 0.30],
    "IfcBeam":     [0.50, 0.50, 0.20],
    "IfcStair":    [0.30, 0.30, 0.50],
    "IfcRailing":  [0.90, 0.90, 0.30],
}


def main():
    ifc_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/ifc/casapequena.ifc"
    out_ply  = sys.argv[2] if len(sys.argv) > 2 else f"/mnt/c/Users/Rafael/Downloads/RCP/RCP/{Path(ifc_path).stem}_helios.ply"

    print("=" * 70)
    print(f"HELIOS++ POC: {ifc_path}")
    print("=" * 70)

    # 1. Extrai meshes do IFC
    print(f"\n[1/3] Lendo IFC e extraindo meshes...")
    m = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)

    meshes = {}
    tipos_por_guid = {}
    ifc_objs = []
    t0 = time.time()
    for tipo in TIPOS:
        for el in m.by_type(tipo):
            try:
                shape = ifcopenshell.geom.create_shape(settings, el)
            except Exception:
                continue
            verts = np.array(shape.geometry.verts, dtype=np.float32).reshape(-1, 3)
            faces = np.array(shape.geometry.faces, dtype=np.int32).reshape(-1, 3)
            if len(verts) == 0 or len(faces) == 0:
                continue
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices  = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            if mesh.get_surface_area() < 0.1:
                continue
            guid = el.GlobalId
            meshes[guid] = mesh
            tipos_por_guid[guid] = tipo
            bb = {
                "xmin": float(verts[:, 0].min()), "xmax": float(verts[:, 0].max()),
                "ymin": float(verts[:, 1].min()), "ymax": float(verts[:, 1].max()),
                "zmin": float(verts[:, 2].min()), "zmax": float(verts[:, 2].max()),
            }
            ifc_objs.append({"guid": guid, "tipo": tipo, "bbox": bb})

    dt = time.time() - t0
    print(f"      {len(meshes)} meshes em {dt:.1f}s")
    dist_in = Counter(tipos_por_guid.values())
    for t, n in dist_in.most_common():
        print(f"        {t:18s} {n}")

    # 2. Walkable scanners
    print(f"\n[2/3] Escolhendo posicoes walkable...")
    scanners = compute_walkable_scanners(ifc_objs, n_scanners=5)
    print(f"      {len(scanners)} scanners:")
    for i, (x, y, z) in enumerate(scanners):
        print(f"        s{i}: ({x:.2f}, {y:.2f}, {z:.2f})")

    # 3. Roda HELIOS++
    print(f"\n[3/3] Rodando HELIOS++...")
    t0 = time.time()
    out = scan_meshes(meshes, scanners, tipos_por_guid=tipos_por_guid)
    dt = time.time() - t0
    print(f"      Tempo: {dt:.1f}s")
    n_pts = out["n_pts"]
    print(f"      Total pts: {n_pts:,}")

    # Distribuicao por tipo IFC (via guid -> tipo)
    hit_tipos = [tipos_por_guid.get(g, "?") for g in out["hit_guid"]]
    dist_out = Counter(hit_tipos)
    print(f"\n      Distribuicao por tipo:")
    for t, n in dist_out.most_common():
        pct = 100 * n / max(n_pts, 1)
        print(f"        {t:18s} {n:>7,} pts ({pct:.1f}%)")

    # Salva PLY colorido
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out["pts"].astype(np.float64))
    cols = np.array([COLORS.get(t, [0.5, 0.5, 0.5]) for t in hit_tipos])
    pcd.colors = o3d.utility.Vector3dVector(cols)
    Path(out_ply).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(out_ply, pcd, write_ascii=False)
    print(f"\n PLY salvo: {out_ply}")
    print(f"   ({n_pts:,} pts, ~{Path(out_ply).stat().st_size/1024/1024:.1f} MB)")
    print()
    print("Cores: roxo=wall, verde=slab, verde escuro=covering, cinza=column,")
    print("       laranja=door, azul=window, vermelho=roof, amarelo=railing")


if __name__ == "__main__":
    main()
