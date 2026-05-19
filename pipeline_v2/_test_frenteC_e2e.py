"""Teste end-to-end Frente C: Sonata + Hungarian matching com IFC real.

Pipeline:
  1. Carrega PLY → Sonata via runner (B)
  2. Reclassify ceiling + filter matchable (B)
  3. Carrega IFC + extrai objetos (importa do app_obb.py)
  4. Hungarian matching (C)
  5. Imprime matches, AUSENTE, ADICAO

Uso:
    python -m pipeline_v2._test_frenteC_e2e <ply_path> <ifc_path> [voxel]
"""

import sys
from pathlib import Path

# Importa app_obb.py pra usar extrair_objetos_por_pavimento
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from . import sonata_runner, class_mapping, matcher_hungarian


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    ply = sys.argv[1]
    ifc_path = sys.argv[2]
    voxel = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

    print("=" * 72)
    print("FRENTE C — E2E (Sonata + Hungarian com IFC real)")
    print("=" * 72)
    print(f"PLY:   {Path(ply).name}")
    print(f"IFC:   {Path(ifc_path).name}")
    print(f"Voxel: {voxel}m")
    print()

    # 1. Sonata via runner
    print("[1/4] Sonata (cache hit se disponível)...")
    sonata_result = sonata_runner.run_sonata(ply, voxel=voxel, verbose=True)
    pts_voxel = sonata_result["pts_voxel"]
    instances = sonata_result["instances"]

    # 2. Reclassify + filter (Frente B finalizada)
    scene_bbox_z = (float(pts_voxel[:, 2].min()), float(pts_voxel[:, 2].max()))
    instances_rec = class_mapping.reclassify_ceiling(instances, scene_bbox_z)
    instances_matchable = class_mapping.filter_matchable_instances(instances_rec)
    print(f"      {len(instances)} instâncias → "
          f"{len(instances_matchable)} matchable após reclassify+filter")

    # 3. Carrega IFC (usa app_obb pra reaproveitar a lógica existente)
    print()
    print("[2/4] Extraindo objetos do IFC...")
    import app_obb
    # Pavimento '__TODOS__' = sem filtro de storey
    ifc_objects = app_obb.extrair_objetos_por_pavimento(
        ifc_path,
        pavimento_alvo="__TODOS__",
        incluir_estrutura_cruzando=False,
    )
    print(f"      {len(ifc_objects)} objetos IFC extraídos")
    from collections import Counter
    tipo_dist = Counter(o["tipo"] for o in ifc_objects)
    for tipo, n in sorted(tipo_dist.items(), key=lambda x: -x[1]):
        print(f"        {tipo:25s} {n:3d}")

    # 4. Hungarian matching
    print()
    print("[3/4] Hungarian matching...")
    result, summary = matcher_hungarian.match_and_summarize(instances_matchable, ifc_objects)
    print(summary)

    # 5. Detalhes
    print()
    print("[4/4] Detalhes dos matches (top 15 por custo):")
    if result["matches"]:
        sorted_matches = sorted(result["matches"], key=lambda m: m["cost"])
        for k, m in enumerate(sorted_matches[:15]):
            si = m["scan_inst"]
            oi = m["ifc_obj"]
            print(f"  #{k+1:2d}  "
                  f"{si['class_name']:9s} → {oi['tipo']:13s}  "
                  f"cost={m['cost']:5.2f}  "
                  f"ifc={oi.get('nome', oi['guid'])[:50]}")
        if len(result["matches"]) > 15:
            print(f"  ... +{len(result['matches'])-15} mais")

    if result["unmatched_scan"]:
        print()
        print(f"ADIÇÃO ({len(result['unmatched_scan'])} construído sem IFC) — top 10:")
        for u in result["unmatched_scan"][:10]:
            c = u["centroid"]
            print(f"  {u['class_name']:9s}  centro=({c[0]:7.2f}, {c[1]:7.2f}, {c[2]:6.2f})  "
                  f"n_pts={u['n_pts']:5d}  conf={u['mean_conf']:.2f}")

    if result["unmatched_ifc"]:
        print()
        print(f"AUSENTE ({len(result['unmatched_ifc'])} planejado não detectado) — top 10:")
        for u in result["unmatched_ifc"][:10]:
            print(f"  {u['tipo']:13s}  {u.get('nome', u['guid'])[:60]}")

    if result["ifc_sem_cobertura"]:
        print()
        print(f"V1 ({len(result['ifc_sem_cobertura'])} IFC sem cobertura Sonata):")
        tipo_dist_v1 = Counter(o["tipo"] for o in result["ifc_sem_cobertura"])
        for tipo, n in sorted(tipo_dist_v1.items(), key=lambda x: -x[1]):
            print(f"  {tipo:13s} {n:3d}")

    print()
    print("=" * 72)
    print(f"FRENTE C OK. Estado final: {result['stats']['n_matches']} matches "
          f"+ {result['stats']['n_unmatched_scan']} ADICAO "
          f"+ {result['stats']['n_unmatched_ifc']} AUSENTE")
    print("=" * 72)


if __name__ == "__main__":
    main()
