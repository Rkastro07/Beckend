# -*- coding: utf-8 -*-
"""
DATASET GENERATOR — Instâncias para Segmentação de Instâncias BIM
==================================================================
Estende o dataset_generator.py para gerar labels de instância por ponto.

Cada ponto recebe:
  - semantic_label  : 0..7 (tipo IFC)
  - instance_label  : 0 = background, 1..N = ID único do objeto IFC

Isso é necessário para treinar arquiteturas de segmentação de instâncias
como PointGroup, SPFormer ou Mask3D — que identificam cada objeto
individualmente (ex: wall_241 vs wall_234), não apenas o tipo.

Uso:
  python run_batch_instances.py --dataset C:/caminho/dataset/
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from dataset_generator import _estimar_normais, _rotular_pontos, LABEL_MAP, DATA_DIR


def _rotular_instancias(
    pts: np.ndarray,
    objetos_ifc: List[Dict],
    margem: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Atribui label semântico E label de instância a cada ponto.

    Retorna:
        sem_labels  : (N,) uint8  — classe semântica (0..7)
        inst_labels : (N,) int32  — ID de instância (0=background, 1..M)

    O instance_id é o índice 1-based do objeto na lista objetos_ifc.
    Pontos em sobreposição de bboxes recebem o último objeto processado
    (ordenado por volume crescente — objetos maiores têm prioridade).
    """
    N = len(pts)
    sem_labels  = np.zeros(N, dtype=np.uint8)
    inst_labels = np.zeros(N, dtype=np.int32)

    # Ordena por volume crescente — objetos maiores sobrescrevem menores (igual ao semântico)
    ordenados = sorted(
        enumerate(objetos_ifc, start=1),
        key=lambda x: x[1].get('volume_ifc', 0)
    )

    for inst_id, obj in ordenados:
        classe = LABEL_MAP.get(obj['tipo'], 0)
        if classe == 0:
            continue  # tipo não mapeado → background

        bbox = obj['bbox']
        mask = (
            (pts[:, 0] >= bbox['xmin'] - margem) &
            (pts[:, 0] <= bbox['xmax'] + margem) &
            (pts[:, 1] >= bbox['ymin'] - margem) &
            (pts[:, 1] <= bbox['ymax'] + margem) &
            (pts[:, 2] >= bbox['zmin'] - margem) &
            (pts[:, 2] <= bbox['zmax'] + margem)
        )
        sem_labels[mask]  = classe
        inst_labels[mask] = inst_id

    return sem_labels, inst_labels


def salvar_cena_instancias(
    pts_alinhado: np.ndarray,
    objetos_ifc: List[Dict],
    nome_cena: str,
    computar_normais: bool = True
) -> Path:
    """
    Salva cena com labels semânticos + instância no formato .npz.

    Estrutura do arquivo:
        pts       : (N, 6) float32 — xyz + normais
        labels    : (N,)   uint8   — classe semântica (0..7)
        instances : (N,)   int32   — ID de instância (0=bg, 1..M)
        meta      : dict como string JSON — nome, n_objetos, contagens por classe
    """
    if pts_alinhado is None or len(pts_alinhado) == 0:
        print("  Nuvem vazia — cena nao salva.")
        return None

    # Labeling
    sem_labels, inst_labels = _rotular_instancias(pts_alinhado, objetos_ifc)

    n_bg   = int((sem_labels == 0).sum())
    n_rot  = int((sem_labels > 0).sum())
    n_inst = int(inst_labels.max())
    print(f"  {n_rot:,} pts rotulados | {n_bg:,} background | {n_inst} instancias")

    # Normais
    if computar_normais:
        normais = _estimar_normais(pts_alinhado)
        features = np.hstack([pts_alinhado.astype(np.float32), normais])
    else:
        features = pts_alinhado.astype(np.float32)

    # Contagens por classe
    import json
    label_counts = {
        str(k): int((sem_labels == k).sum())
        for k in range(8)
        if (sem_labels == k).sum() > 0
    }

    inst_dir = DATA_DIR.parent / "data_instances"
    inst_dir.mkdir(parents=True, exist_ok=True)
    out_path = inst_dir / f"{nome_cena}.npz"

    np.savez_compressed(
        out_path,
        pts=features,
        labels=sem_labels,
        instances=inst_labels,
    )

    # Salva meta separado (json pequeno)
    meta = {
        'nome': nome_cena,
        'n_pontos': len(pts_alinhado),
        'n_instancias': n_inst,
        'n_objetos_ifc': len(objetos_ifc),
        'label_counts': label_counts,
        'bg_ratio': round(n_bg / len(pts_alinhado), 3)
    }
    with open(inst_dir / f"{nome_cena}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Salvo: {out_path.name} ({out_path.stat().st_size // 1024} KB)")
    return out_path
