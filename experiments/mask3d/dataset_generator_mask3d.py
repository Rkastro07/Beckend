# -*- coding: utf-8 -*-
"""
DATASET GENERATOR - Mask3D BIM Instance Segmentation
=====================================================
Converte cenas sinteticas (cena.ply + ifc_ref.json) em dados de treino
com labels de INSTANCIA (cada objeto IFC = instancia separada).

Saida por cena (.npz):
  - pts:              (N, 3) float32  -- coordenadas XYZ
  - normals:          (N, 3) float32  -- normais estimadas
  - semantic_labels:  (N,)   int32    -- classe por ponto (0=bg, 1-7=IFC)
  - instance_labels:  (N,)   int32    -- id da instancia (-1=bg)

Classes BIM:
  0 = background
  1 = IfcWall
  2 = IfcSlab
  3 = IfcColumn
  4 = IfcBeam
  5 = IfcStair
  6 = IfcRoof
  7 = IfcSanitaryTerminal

Uso:
  python dataset_generator_mask3d.py
  python dataset_generator_mask3d.py --max 200
  python dataset_generator_mask3d.py --stats
"""

import json
import argparse
import time
import numpy as np
import open3d as o3d
from pathlib import Path

SINTETICO_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "sintetico"
OUT_DIR = Path(__file__).resolve().parent / "data" / "train"

LABEL_MAP = {
    "IfcWall":             1,
    "IfcSlab":             2,
    "IfcColumn":           3,
    "IfcBeam":             4,
    "IfcStair":            5,
    "IfcRoof":             6,
    "IfcSanitaryTerminal": 7,
}

BIM_CLASSES = ["background", "IfcWall", "IfcSlab", "IfcColumn",
               "IfcBeam", "IfcStair", "IfcRoof", "IfcSanitaryTerminal"]


def rotular_instancias(pts, objetos, margem=0.02):
    """
    Rotula cada ponto com classe semantica + id de instancia.

    Para sobreposicoes de bbox, atribui ao objeto cujo centro de bbox
    eh mais proximo (melhor separacao de instancias).

    Args:
        pts:     (N, 3) float32
        objetos: dict {guid: {tipo, bbox, ...}}
        margem:  margem extra no bbox (metros)

    Returns:
        sem_labels:  (N,) int32  -- classe 0-7
        inst_labels: (N,) int32  -- id instancia, -1 = background
    """
    N = len(pts)
    sem_labels = np.zeros(N, dtype=np.int32)
    inst_labels = np.full(N, -1, dtype=np.int32)
    best_dist = np.full(N, np.inf, dtype=np.float32)

    for inst_id, (guid, obj) in enumerate(objetos.items()):
        cls = LABEL_MAP.get(obj.get("tipo", ""), 0)
        if cls == 0:
            continue

        b = obj["bbox"]
        mask = (
            (pts[:, 0] >= b["xmin"] - margem) & (pts[:, 0] <= b["xmax"] + margem) &
            (pts[:, 1] >= b["ymin"] - margem) & (pts[:, 1] <= b["ymax"] + margem) &
            (pts[:, 2] >= b["zmin"] - margem) & (pts[:, 2] <= b["zmax"] + margem)
        )

        if mask.sum() == 0:
            continue

        # Distancia ao centro do bbox
        centro = np.array([
            (b["xmin"] + b["xmax"]) / 2,
            (b["ymin"] + b["ymax"]) / 2,
            (b["zmin"] + b["zmax"]) / 2,
        ], dtype=np.float32)

        dist = np.linalg.norm(pts[mask] - centro, axis=1)
        idx = np.where(mask)[0]
        closer = dist < best_dist[idx]
        idx_closer = idx[closer]

        sem_labels[idx_closer] = cls
        inst_labels[idx_closer] = inst_id
        best_dist[idx_closer] = dist[closer]

    return sem_labels, inst_labels


def processar_cena(scene_dir, out_dir):
    """
    Processa uma cena sintetica e salva .npz com labels de instancia.

    Returns:
        str com resumo ou None se falhar.
    """
    ply_path = scene_dir / "cena.ply"
    ref_path = scene_dir / "ifc_ref.json"

    if not ply_path.exists() or not ref_path.exists():
        return None

    out_path = out_dir / (scene_dir.name + ".npz")
    if out_path.exists():
        return f"  [skip] {scene_dir.name} (ja existe)"

    # Carrega PLY
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points, dtype=np.float32)

    if len(pts) < 100:
        return None

    # Carrega IFC ref
    with open(ref_path, encoding="utf-8") as f:
        objetos = json.load(f)

    # Rotula instancias
    sem_labels, inst_labels = rotular_instancias(pts, objetos)

    n_labeled = int((sem_labels > 0).sum())
    if n_labeled < 50:
        return f"  [skip] {scene_dir.name} (poucas labels: {n_labeled})"

    # Estima normais
    pcd_n = o3d.geometry.PointCloud()
    pcd_n.points = o3d.utility.Vector3dVector(pts)
    pcd_n.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
    )
    pcd_n.orient_normals_towards_camera_location([0, 0, 100])
    normals = np.asarray(pcd_n.normals, dtype=np.float32)

    # Salva
    np.savez_compressed(
        out_path,
        pts=pts,
        normals=normals,
        semantic_labels=sem_labels,
        instance_labels=inst_labels,
    )

    n_inst = len(set(inst_labels[inst_labels >= 0].tolist()))
    return f"  {scene_dir.name}: {len(pts):,} pts, {n_inst} inst, {n_labeled:,} labeled"


def gerar_dataset(max_scenes=None):
    """Processa todas as cenas sinteticas."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scenes = sorted(SINTETICO_DIR.iterdir())
    scenes = [s for s in scenes if s.is_dir() and (s / "cena.ply").exists()]

    if max_scenes:
        scenes = scenes[:max_scenes]

    print(f"Processando {len(scenes)} cenas sinteticas...")
    print(f"Saida: {OUT_DIR}")
    print("-" * 60)

    t0 = time.time()
    ok = 0
    for i, scene_dir in enumerate(scenes):
        result = processar_cena(scene_dir, OUT_DIR)
        if result:
            print(result)
            if "[skip]" not in result:
                ok += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  --- {i+1}/{len(scenes)} ({elapsed:.0f}s) ---")

    elapsed = time.time() - t0
    print(f"\nPronto! {ok} cenas processadas em {elapsed:.0f}s")
    print(f"Dados em: {OUT_DIR}")


def estatisticas():
    """Imprime estatisticas do dataset gerado."""
    files = sorted(OUT_DIR.glob("*.npz"))
    if not files:
        print("Nenhum dado encontrado. Rode sem --stats primeiro.")
        return

    total_pts = 0
    total_inst = 0
    contagem_cls = np.zeros(8, dtype=np.int64)
    contagem_inst_por_cls = np.zeros(8, dtype=np.int64)

    for f in files:
        data = np.load(f)
        sem = data["semantic_labels"]
        inst = data["instance_labels"]
        total_pts += len(sem)

        for c in range(8):
            contagem_cls[c] += (sem == c).sum()

        # Conta instancias unicas por classe
        for iid in np.unique(inst[inst >= 0]):
            cls = sem[inst == iid][0]
            contagem_inst_por_cls[cls] += 1
            total_inst += 1

    print(f"\nDataset Mask3D BIM")
    print(f"{'='*60}")
    print(f"Cenas:      {len(files):,}")
    print(f"Pontos:     {total_pts:,}")
    print(f"Instancias: {total_inst:,}")
    print(f"\n{'Classe':<22} {'Pontos':>12} {'%':>7} {'Instancias':>12}")
    print("-" * 55)
    for i, nome in enumerate(BIM_CLASSES):
        pct = contagem_cls[i] / total_pts * 100 if total_pts > 0 else 0
        print(f"{i}: {nome:<18} {contagem_cls[i]:>12,} {pct:>6.1f}% {contagem_inst_por_cls[i]:>12,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dataset Mask3D BIM")
    parser.add_argument("--max", type=int, default=None, help="Max cenas a processar")
    parser.add_argument("--stats", action="store_true", help="Mostra estatisticas")
    args = parser.parse_args()

    if args.stats:
        estatisticas()
    else:
        gerar_dataset(max_scenes=args.max)
