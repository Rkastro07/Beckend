# -*- coding: utf-8 -*-
"""
DATASET GENERATOR — BIM RandLA-Net
===================================
Gera automaticamente dados de treino rotulados a partir do pipeline atual.
Cada cena vira um arquivo .npz com:
  - pts    : (N, 6) float32 — xyz + normais
  - labels : (N,)   int8   — classe por ponto

Classes:
  0 = background
  1 = IfcWall
  2 = IfcSlab
  3 = IfcColumn
  4 = IfcBeam
  5 = IfcStair
  6 = IfcRoof
  7 = IfcSanitaryTerminal
"""

import os
import uuid
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Dict

# Diretório de dados de treino
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Mapeamento tipo IFC → classe inteira
LABEL_MAP = {
    "IfcWall":             1,
    "IfcSlab":             2,
    "IfcColumn":           3,
    "IfcBeam":             4,
    "IfcStair":            5,
    "IfcRoof":             6,
    "IfcSanitaryTerminal": 7,
}


def _estimar_normais(pts: np.ndarray, radius: float = 0.3, max_nn: int = 20) -> np.ndarray:
    """Estima normais via Open3D (PCA local). Orientação para cima (Z+)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    # Orientação rápida: aponta normais para cima (evita tangent_plane que é O(N²))
    pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 100])
    return np.asarray(pcd.normals, dtype=np.float32)


def _rotular_pontos(pts: np.ndarray, objetos_ifc: List[Dict], margem: float = 0.02) -> np.ndarray:
    """
    Atribui label 0..7 a cada ponto via bbox dos objetos IFC.
    Pontos fora de qualquer bbox recebem label 0 (background).
    Em caso de sobreposição, vence o objeto com maior volume_ifc.
    """
    labels = np.zeros(len(pts), dtype=np.int8)

    # Ordena por volume crescente — objetos maiores sobrescrevem menores
    ordenados = sorted(objetos_ifc, key=lambda o: o.get('volume_ifc', 0))

    for obj in ordenados:
        classe = LABEL_MAP.get(obj['tipo'], 0)
        if classe == 0:
            continue
        bbox = obj['bbox']
        mask = (
            (pts[:, 0] >= bbox['xmin'] - margem) &
            (pts[:, 0] <= bbox['xmax'] + margem) &
            (pts[:, 1] >= bbox['ymin'] - margem) &
            (pts[:, 1] <= bbox['ymax'] + margem) &
            (pts[:, 2] >= bbox['zmin'] - margem) &
            (pts[:, 2] <= bbox['zmax'] + margem)
        )
        labels[mask] = classe

    return labels


def salvar_cena(
    pts_alinhado: np.ndarray,
    objetos_ifc: List[Dict],
    session_id: str = None,
    computar_normais: bool = True
) -> Path:
    """
    Ponto de entrada principal — chamado pelo app.py após análise.

    Args:
        pts_alinhado : (N, 3) nuvem alinhada ao IFC
        objetos_ifc  : lista de objetos com bbox e tipo
        session_id   : identificador da sessão (usa uuid se None)
        computar_normais : estima normais xyz via Open3D

    Returns:
        Path do arquivo .npz gerado
    """
    if pts_alinhado is None or len(pts_alinhado) == 0:
        print("  ⚠️ [Dataset] Nuvem vazia — cena não salva.")
        return None

    sid = session_id or str(uuid.uuid4())[:8]
    out_path = DATA_DIR / f"cena_{sid}.npz"

    # Labels por ponto
    labels = _rotular_pontos(pts_alinhado, objetos_ifc)

    n_background = int((labels == 0).sum())
    n_rotulados  = int((labels > 0).sum())
    print(f"  📊 [Dataset] {n_rotulados:,} pts rotulados | {n_background:,} background")

    # Features: xyz + (normais opcionais)
    if computar_normais:
        print("  🔢 [Dataset] Calculando normais...")
        normais = _estimar_normais(pts_alinhado)
        features = np.hstack([pts_alinhado.astype(np.float32), normais])
    else:
        features = pts_alinhado.astype(np.float32)

    np.savez_compressed(
        out_path,
        pts=features,
        labels=labels
    )
    print(f"  💾 [Dataset] Salvo: {out_path} ({features.shape[0]:,} pts, {features.shape[1]} features)")
    return out_path


def listar_cenas() -> List[Path]:
    """Lista todos os .npz disponíveis para treino."""
    return sorted(DATA_DIR.glob("cena_*.npz"))


def carregar_cena(path: Path):
    """Carrega cena e retorna (pts, labels)."""
    data = np.load(path)
    return data['pts'], data['labels']


def estatisticas_dataset():
    """Imprime estatísticas do dataset acumulado."""
    cenas = listar_cenas()
    if not cenas:
        print("⚠️ Nenhuma cena no dataset ainda.")
        return

    total_pts = 0
    contagem = np.zeros(8, dtype=int)
    nomes_classes = ["background", "IfcWall", "IfcSlab", "IfcColumn",
                     "IfcBeam", "IfcStair", "IfcRoof", "IfcSanitaryTerminal"]

    for p in cenas:
        _, labels = carregar_cena(p)
        total_pts += len(labels)
        for i in range(8):
            contagem[i] += int((labels == i).sum())

    print(f"\n📦 Dataset: {len(cenas)} cenas | {total_pts:,} pontos totais")
    print("-" * 40)
    for i, nome in enumerate(nomes_classes):
        pct = contagem[i] / total_pts * 100 if total_pts > 0 else 0
        print(f"  {i}: {nome:<22} {contagem[i]:>10,} pts ({pct:.1f}%)")


if __name__ == "__main__":
    estatisticas_dataset()
