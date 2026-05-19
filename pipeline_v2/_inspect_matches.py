"""Inspetor de matches Sonata↔IFC (Frente C).

Roda o orchestrator e despeja uma tabela detalhada dos matches Hungarian.
Pula o RF — foco é validar que os pareamentos fazem sentido semanticamente
(parede detectada bate com Wall do IFC, posição compatível, etc.).

Uso:
    python -m pipeline_v2._inspect_matches <ply> <ifc> [pavimento]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_v2 import (
    sonata_runner, class_mapping, matcher_hungarian, matcher_costs,
)
import app_obb
import open3d as o3d
import tempfile


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    ply_path = sys.argv[1]
    ifc_path = sys.argv[2]
    pavimento = sys.argv[3] if len(sys.argv) > 3 else "__TODOS__"

    print("=" * 80)
    print(f"INSPETOR DE MATCHES — Sonata + Hungarian (sem RF)")
    print("=" * 80)
    print(f"PLY: {Path(ply_path).name}")
    print(f"IFC: {Path(ifc_path).name}")
    print(f"Pavimento: {pavimento}\n")

    # 1. IFC
    print("[1/5] Carregando IFC...")
    ifc_objects = app_obb.extrair_objetos_por_pavimento(
        ifc_path, pavimento_alvo=pavimento, incluir_estrutura_cruzando=False,
    )
    print(f"      {len(ifc_objects)} objetos IFC totais")

    # 2. Align
    print("[2/5] Alinhando PLY ao IFC...")
    pcd = o3d.io.read_point_cloud(ply_path)
    pts_raw = np.asarray(pcd.points, dtype=np.float32)
    pts_aligned, _ = app_obb.alinhar_nuvem_com_ifc(pts_raw, ifc_objects)

    aligned = o3d.geometry.PointCloud()
    aligned.points = o3d.utility.Vector3dVector(pts_aligned.astype(np.float64))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
    tmp.close()
    o3d.io.write_point_cloud(tmp.name, aligned, write_ascii=False)

    # 3. Sonata
    print("[3/5] Rodando Sonata (pode demorar se cache MISS)...")
    sonata = sonata_runner.run_sonata(tmp.name, voxel=0.15, verbose=False)
    Path(tmp.name).unlink(missing_ok=True)
    print(f"      {len(sonata['instances'])} instâncias detectadas")

    # 4. Reclassify + filter
    pts_voxel = sonata["pts_voxel"]
    scene_z = (float(pts_voxel[:, 2].min()), float(pts_voxel[:, 2].max()))
    insts = class_mapping.reclassify_ceiling(sonata["instances"], scene_z)
    insts_m = class_mapping.filter_matchable_instances(insts)
    print(f"      {len(insts_m)} matchable (após reclassify+filter)")

    # Estatística por classe Sonata
    from collections import Counter
    cls_dist = Counter(i["class_name"] for i in insts_m)
    print(f"      Classes Sonata: {dict(cls_dist)}")

    # 5. Hungarian
    print("[4/5] Hungarian matching...")
    result = matcher_hungarian.match(insts_m, ifc_objects)
    matches = sorted(result["matches"], key=lambda m: m["cost"])

    print()
    print(f"[5/5] {len(matches)} matches válidos (após threshold por tipo)")
    print(f"      {len(result['unmatched_scan'])} ADIÇÃO (scan sem par IFC)")
    print(f"      {len(result['unmatched_ifc'])} AUSENTE (IFC sem par scan)")
    print(f"      {len(result['ifc_sem_cobertura'])} pulados (tipos sem cobertura Sonata)")
    print()

    # ======================== TABELA DOS MATCHES ========================
    print("=" * 95)
    print(f"{'#':>3}  {'Sonata':10s}  {'IFC tipo':15s}  {'cost':>6s}  {'dist_m':>7s}  {'IFC nome / guid'}")
    print("-" * 95)
    for k, m in enumerate(matches, 1):
        si = m["scan_inst"]
        oi = m["ifc_obj"]
        bbox = oi["bbox"]
        ifc_c = np.array([
            (bbox["xmin"] + bbox["xmax"]) * 0.5,
            (bbox["ymin"] + bbox["ymax"]) * 0.5,
            (bbox["zmin"] + bbox["zmax"]) * 0.5,
        ])
        scan_c = np.asarray(si["centroid"])
        dist = float(np.linalg.norm(scan_c - ifc_c))
        nome = (oi.get("nome") or oi.get("guid", ""))[:55]
        print(f"  {k:>3}  {si['class_name']:10s}  {oi['tipo']:15s}  "
              f"{m['cost']:>6.2f}  {dist:>7.2f}  {nome}")
    print("=" * 95)

    # ======================== DIAGNÓSTICO ========================
    print()
    print("Resumo de diagnóstico:")
    # Compatibilidade de classe — quantos % são pares "naturais"?
    n_ok = 0
    n_off = 0
    for m in matches:
        si_cls = m["scan_inst"]["class_name"]
        ifc_tp = m["ifc_obj"]["tipo"]
        if matcher_costs.cost_class_compat(si_cls, ifc_tp) < matcher_costs.W_CLASS:
            n_ok += 1
        else:
            n_off += 1
    print(f"  ✓ Pares com classe compatível: {n_ok}/{len(matches)}")
    if n_off:
        print(f"  ✗ Pares com classe incompatível: {n_off} (não deveria existir após threshold)")

    if matches:
        costs = [m["cost"] for m in matches]
        dists = []
        for m in matches:
            bbox = m["ifc_obj"]["bbox"]
            ifc_c = np.array([
                (bbox["xmin"] + bbox["xmax"]) * 0.5,
                (bbox["ymin"] + bbox["ymax"]) * 0.5,
                (bbox["zmin"] + bbox["zmax"]) * 0.5,
            ])
            scan_c = np.asarray(m["scan_inst"]["centroid"])
            dists.append(float(np.linalg.norm(scan_c - ifc_c)))
        print(f"  Custo Hungarian: min={min(costs):.2f}, mediana={float(np.median(costs)):.2f}, max={max(costs):.2f}")
        print(f"  Distância centroide-bbox: min={min(dists):.2f}m, mediana={float(np.median(dists)):.2f}m, max={max(dists):.2f}m")
        print()
        print(f"  → Match 'natural' = classe Sonata bate com tipo IFC + distância < 2m.")
        print(f"  → Se a tabela acima tá com nomes coerentes, Sonata+Hungarian estão OK.")


if __name__ == "__main__":
    main()
