"""Teste end-to-end da Frente B: Sonata runner + cache + class_mapping.

Roda Sonata, aplica reclassify_ceiling, filtra matchable, mostra resumo.

Esse é o teste que valida que a Frente B está pronta pra alimentar a Frente C
(Hungarian matching). A próxima etapa receberá `instancias_matchable` como input.

Uso:
    python -m pipeline_v2._test_frenteB_e2e <ply_path> [voxel]
"""

import sys
from pathlib import Path

import numpy as np

from . import sonata_runner, class_mapping


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ply = sys.argv[1]
    voxel = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15

    print("=" * 70)
    print(f"FRENTE B — E2E TEST")
    print("=" * 70)
    print(f"PLY:   {Path(ply).name}")
    print(f"Voxel: {voxel}m")
    print()

    # 1. Sonata (runner cuida de cache + subprocess)
    print("[1/4] Rodando Sonata via runner...")
    result = sonata_runner.run_sonata(ply, voxel=voxel, verbose=True)

    pts_voxel = result["pts_voxel"]
    instances = result["instances"]
    print(f"      Resultado: {len(pts_voxel):,} pts, {len(instances)} instâncias")
    print(f"      Cache: {'HIT' if result.get('from_cache') else 'MISS'}")

    # 2. Reclassify ceiling
    print()
    print("[2/4] Reclassificando 'floor' alto como 'ceiling'...")
    scene_bbox_z = (float(pts_voxel[:, 2].min()), float(pts_voxel[:, 2].max()))
    print(f"      BBox Z da cena: [{scene_bbox_z[0]:.2f}, {scene_bbox_z[1]:.2f}]m")

    n_floor_antes = sum(1 for i in instances if i["class_name"] == "floor")
    instances_rec = class_mapping.reclassify_ceiling(instances, scene_bbox_z)
    n_ceiling = sum(1 for i in instances_rec if i["class_name"] == "ceiling")
    n_floor_depois = sum(1 for i in instances_rec if i["class_name"] == "floor")
    print(f"      floor antes:  {n_floor_antes}")
    print(f"      ceiling após: {n_ceiling}  (reclassificadas como teto)")
    print(f"      floor após:   {n_floor_depois}  (continuam piso)")

    # 3. Filtrar matchable
    print()
    print("[3/4] Filtrando instâncias 'matchable' (com cobertura IFC)...")
    matchable = class_mapping.filter_matchable_instances(instances_rec)
    n_descartadas = len(instances_rec) - len(matchable)
    print(f"      Matchable: {len(matchable)}")
    print(f"      Descartadas: {n_descartadas}  (cadeiras/mesas/etc. — ruído)")

    # 4. Resumo pronto pro Hungarian
    print()
    print("[4/4] Resumo de instâncias matchable (input pra Frente C):")
    print()
    print(f"  {'#':>3}  {'classe':10s}  {'pts':>8s}  {'volume':>10s}  {'conf':>6s}  {'centroid (x,y,z)':>30s}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*6}  {'-'*30}")
    for i, inst in enumerate(matchable[:30]):  # primeiras 30 pra não inundar
        c = inst["centroid"]
        print(f"  {i+1:>3}  "
              f"{inst['class_name']:10s}  "
              f"{inst['n_pts']:>8,}  "
              f"{inst['volume']:>10.3f}  "
              f"{inst['mean_conf']:>6.2f}  "
              f"({c[0]:>7.2f}, {c[1]:>7.2f}, {c[2]:>6.2f})")
    if len(matchable) > 30:
        print(f"  ... +{len(matchable)-30} mais")

    print()
    print("=" * 70)
    print("FRENTE B PRONTA. Próximo: Frente C (Hungarian matching).")
    print("=" * 70)
    print()
    print(f"Estatísticas pra Frente C:")
    print(f"  - {len(matchable)} instâncias matchable do scan")
    print(f"  - Distribuição:")
    from collections import Counter
    dist = Counter(i["class_name"] for i in matchable)
    for cls, n in sorted(dist.items()):
        compat = class_mapping.ifc_types_for_class(cls)
        print(f"    {cls:10s} = {n:3d}  → pode matchar com {compat}")


if __name__ == "__main__":
    main()
