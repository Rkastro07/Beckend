"""Exporta cada instância Sonata+DBSCAN como PLY separado pra inspeção visual.

Gera duas pastas:
  - sem_pos_processamento/   : 1 PLY por instância como saíram do DBSCAN
  - com_pos_processamento/   : 1 PLY por instância após merge de fragmentos
                                (mesma classe + bboxes inflados se sobrepõem)

Uso:
    python -m pipeline_v2._export_instancias_ply [out_root]
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# Helpers
# ============================================================
def _color_for(idx: int, seed: int = 42) -> np.ndarray:
    """Cor estável por índice — distinta o suficiente pra ver no CloudCompare."""
    rng = np.random.default_rng(seed + idx)
    c = rng.random(3)
    # evita cores muito escuras
    c = c * 0.7 + 0.3
    return c.astype(np.float32)


def _bbox_aabb(pts: np.ndarray) -> dict:
    """AABB da nuvem."""
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    return {
        "xmin": float(mn[0]), "xmax": float(mx[0]),
        "ymin": float(mn[1]), "ymax": float(mx[1]),
        "zmin": float(mn[2]), "zmax": float(mx[2]),
    }


def _bbox_inflate(bb: dict, m: float) -> dict:
    return {
        "xmin": bb["xmin"] - m, "xmax": bb["xmax"] + m,
        "ymin": bb["ymin"] - m, "ymax": bb["ymax"] + m,
        "zmin": bb["zmin"] - m, "zmax": bb["zmax"] + m,
    }


def _bbox_overlap(a: dict, b: dict) -> bool:
    """True se as AABBs se sobrepõem (incluindo só tocar)."""
    return (
        a["xmax"] >= b["xmin"] and b["xmax"] >= a["xmin"] and
        a["ymax"] >= b["ymin"] and b["ymax"] >= a["ymin"] and
        a["zmax"] >= b["zmin"] and b["zmax"] >= a["zmin"]
    )


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


# ============================================================
# Merge heurístico de fragmentos
# ============================================================
def merge_fragments(instances: list[dict], pts_all: np.ndarray,
                     margins: dict | None = None) -> list[dict]:
    """Funde instâncias da mesma classe cujos bboxes (inflados) se sobrepõem.

    Args:
        instances: lista do output Sonata (cada item tem class_name, pts_idx, bbox, etc.)
        pts_all:   nuvem global (pts_voxel)
        margins:   dict por classe {class_name: margem_m} pro inflate. Default:
                   wall=0.6 (paredes longas, fragmentos podem estar até 0.6m de gap),
                   floor=0.5, ceiling=0.5, outros=0.3

    Returns:
        Lista nova de instâncias merged. Cada item tem 'merged_from': [idx_originais],
        'class_name', 'pts_idx', 'centroid', 'bbox', 'n_pts'.
    """
    if margins is None:
        margins = {"wall": 0.6, "floor": 0.5, "ceiling": 0.5}
    default_margin = 0.3

    # Agrupa por classe
    by_class: dict[str, list[int]] = {}
    for i, inst in enumerate(instances):
        by_class.setdefault(inst["class_name"], []).append(i)

    # Union-find por classe
    n = len(instances)
    uf = _UnionFind(n)
    for cls, ids in by_class.items():
        m = margins.get(cls, default_margin)
        inflated = [_bbox_inflate(instances[i]["bbox"], m) for i in ids]
        for ai, ia in enumerate(ids):
            for aj in range(ai + 1, len(ids)):
                ib = ids[aj]
                if _bbox_overlap(inflated[ai], inflated[aj]):
                    uf.union(ia, ib)

    # Coleta grupos
    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    # Constrói instâncias merged
    merged: list[dict] = []
    for root, members in groups.items():
        first = instances[members[0]]
        # Junta pts_idx (sem deduplicar — não há sobreposição entre instâncias DBSCAN)
        pts_idx = np.concatenate([np.asarray(instances[i]["pts_idx"]) for i in members])
        pts = pts_all[pts_idx]
        merged.append({
            "class_name":  first["class_name"],
            "class_id":    first.get("class_id"),
            "pts_idx":     pts_idx,
            "centroid":    pts.mean(axis=0),
            "bbox":        _bbox_aabb(pts),
            "n_pts":       int(len(pts_idx)),
            "merged_from": members,
        })
    return merged


# ============================================================
# Export
# ============================================================
def export_instancias(out_dir: Path, instances: list[dict], pts_all: np.ndarray,
                       prefix: str = "inst") -> int:
    """Escreve 1 PLY por instância. Returns número de PLYs escritos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for i, inst in enumerate(instances):
        pts_idx = np.asarray(inst["pts_idx"])
        pts = pts_all[pts_idx]
        if len(pts) == 0:
            continue
        cor = _color_for(i)
        colors = np.tile(cor, (len(pts), 1))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        cls = inst["class_name"]
        n_pts = len(pts)
        fname = f"{prefix}_{i:03d}_{cls}_{n_pts}pts.ply"
        o3d.io.write_point_cloud(str(out_dir / fname), pcd, write_ascii=False)
        n_ok += 1
    return n_ok


def main():
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Users/Rafael/Downloads/RCP/RCP")
    out_root.mkdir(parents=True, exist_ok=True)

    # Acha o pickle Sonata mais recente (0.05m do RH.RETROPORTO esperado)
    cache_dir = Path("C:/Users/Rafael/AppData/Local/Temp/bim_outputs/_sonata_cache")
    pkls = sorted(
        [(p, p.stat().st_mtime, p.stat().st_size) for p in cache_dir.glob("*.pkl")],
        key=lambda x: x[1], reverse=True,
    )
    pkls = [p for p in pkls if p[2] > 1_000_000]
    if not pkls:
        sys.exit(f"Nenhum pickle em {cache_dir}")
    target = pkls[0][0]
    print(f"Usando cache: {target.name}  ({pkls[0][2]/1e6:.1f}MB)")

    with open(target, "rb") as f:
        r = pickle.load(f)
    pts_all   = r["pts_voxel"]
    instances = r["instances"]

    from collections import Counter
    dist = Counter(i["class_name"] for i in instances)
    print(f"Pts: {len(pts_all):,}")
    print(f"Instâncias: {len(instances)}")
    print(f"Por classe: {dict(dist)}")
    print()

    # ---------- SEM pós-processamento ----------
    out_sem = out_root / "sem_pos_processamento"
    print(f"Exportando SEM pós-processamento → {out_sem}")
    n_sem = export_instancias(out_sem, instances, pts_all, prefix="inst")
    print(f"  {n_sem} PLYs escritos")

    # ---------- COM pós-processamento (merge) ----------
    print()
    print("Aplicando merge de fragmentos...")
    merged = merge_fragments(instances, pts_all)
    dist_m = Counter(i["class_name"] for i in merged)
    print(f"  {len(instances)} → {len(merged)} instâncias após merge")
    print(f"  Por classe: {dict(dist_m)}")

    out_com = out_root / "com_pos_processamento"
    print(f"Exportando COM pós-processamento → {out_com}")
    n_com = export_instancias(out_com, merged, pts_all, prefix="merged")
    print(f"  {n_com} PLYs escritos")

    print()
    print("=" * 60)
    print(f"Pronto. Abra no CloudCompare:")
    print(f"  {out_sem}")
    print(f"  {out_com}")
    print()
    print("Diff:")
    print(f"  Antes  : {len(instances)} fragmentos ({dict(dist)})")
    print(f"  Depois : {len(merged)} instâncias  ({dict(dist_m)})")


if __name__ == "__main__":
    main()
