# -*- coding: utf-8 -*-
"""
INFERÊNCIA RANDLA-NET PARA BIM
================================
Substitui o loop per-objeto de bbox no app.py assim que o modelo estiver treinado.

Uso standalone:
  python inference.py --ply caminho/nuvem.ply --checkpoint checkpoints/best.pth

Uso integrado (dentro do app.py):
  from randlanet.inference import classificar_com_modelo
  resultados = classificar_com_modelo(pts_alinhado, objetos_ifc, output_dir)
"""

import sys
import argparse
import numpy as np
import torch
import open3d as o3d
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from model import RandLANetBIM, NUM_CLASSES
from dataset_generator import _estimar_normais, LABEL_MAP

CHECKPOINT_DEFAULT = Path(__file__).parent / "checkpoints" / "best.pth"
CHUNK_SIZE = 65536   # pontos por vez (evita OOM em GPU)

NOMES_CLASSES = [
    "background", "IfcWall", "IfcSlab", "IfcColumn",
    "IfcBeam", "IfcStair", "IfcRoof", "IfcSanitaryTerminal"
]

# Mapa inverso: classe int → tipo IFC
LABEL_TO_TIPO = {v: k for k, v in LABEL_MAP.items()}


def _ifc_bbox_to_threejs(bbox: Dict) -> Dict:
    """Converte bbox IFC (Z=altura) → Three.js (Y=altura). Y/Z swap."""
    return {
        'xmin': bbox['xmin'], 'xmax': bbox['xmax'],
        'ymin': bbox['zmin'], 'ymax': bbox['zmax'],
        'zmin': bbox['ymin'], 'zmax': bbox['ymax'],
    }


def _carregar_modelo(checkpoint: Path, device: torch.device) -> RandLANetBIM:
    model = RandLANetBIM(num_classes=NUM_CLASSES, d_in=6).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def _predizer_em_chunks(
    model: RandLANetBIM,
    pts_feat: np.ndarray,
    device: torch.device
) -> np.ndarray:
    """Prediz labels para N pontos em batches para não explodir RAM/VRAM."""
    N = len(pts_feat)
    preds = np.zeros(N, dtype=np.int8)

    # Normaliza XYZ
    centro = pts_feat[:, :3].mean(axis=0)
    scale  = np.abs(pts_feat[:, :3] - centro).max() + 1e-8
    pts_norm = pts_feat.copy().astype(np.float32)
    pts_norm[:, :3] = (pts_feat[:, :3] - centro) / scale

    for start in range(0, N, CHUNK_SIZE):
        chunk = pts_norm[start: start + CHUNK_SIZE]
        t = torch.from_numpy(chunk).unsqueeze(0).to(device)  # (1, M, 6)
        with torch.no_grad():
            logits = model(t)           # (1, M, 8)
        pred = logits.squeeze(0).argmax(dim=-1).cpu().numpy()
        preds[start: start + CHUNK_SIZE] = pred.astype(np.int8)

    return preds


def classificar_com_modelo(
    pts_alinhado: np.ndarray,
    objetos_ifc: List[Dict],
    output_dir: Path,
    checkpoint: Path = CHECKPOINT_DEFAULT,
    margem: float = 0.05
) -> Optional[List[Dict]]:
    """
    Classifica pontos via RandLA-Net e monta resultados no formato do app.py.
    Retorna None se o modelo não estiver disponível.

    Este método SUBSTITUI o loop per-objeto quando o modelo estiver treinado.
    """
    if not checkpoint.exists():
        print(f"  ⚠️ [RandLA-Net] Checkpoint não encontrado: {checkpoint}")
        print("     Usando análise geométrica padrão.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🤖 [RandLA-Net] Inferência ({len(pts_alinhado):,} pts) em {device}...")

    model = _carregar_modelo(checkpoint, device)

    # Computa normais
    normais = _estimar_normais(pts_alinhado)
    pts_feat = np.hstack([pts_alinhado.astype(np.float32), normais])

    # Predição por ponto
    pred_labels = _predizer_em_chunks(model, pts_feat, device)

    # Estatísticas
    for i, nome in enumerate(NOMES_CLASSES):
        cnt = int((pred_labels == i).sum())
        if cnt > 0:
            print(f"   {i}: {nome:<22} {cnt:>8,} pts")

    # Associa pontos preditos a cada objeto IFC (por classe + proximidade à bbox)
    output_dir = Path(output_dir)
    resultados = []

    for obj in objetos_ifc:
        tipo     = obj['tipo']
        classe   = LABEL_MAP.get(tipo, 0)
        bbox     = obj['bbox']

        # Pontos da classe correta + dentro da bbox (com margem)
        mask_cls = (pred_labels == classe)
        mask_bbox = (
            (pts_alinhado[:, 0] >= bbox['xmin'] - margem) &
            (pts_alinhado[:, 0] <= bbox['xmax'] + margem) &
            (pts_alinhado[:, 1] >= bbox['ymin'] - margem) &
            (pts_alinhado[:, 1] <= bbox['ymax'] + margem) &
            (pts_alinhado[:, 2] >= bbox['zmin'] - margem) &
            (pts_alinhado[:, 2] <= bbox['zmax'] + margem)
        )
        mask = mask_cls & mask_bbox
        pts_obj = pts_alinhado[mask]

        n_pts = len(pts_obj)
        vol   = max(
            (bbox['xmax'] - bbox['xmin']) *
            (bbox['ymax'] - bbox['ymin']) *
            (bbox['zmax'] - bbox['zmin']),
            1e-6
        )

        # Cobertura: razão de pontos capturados vs volume esperado
        cobertura = min(n_pts / (vol * 50), 1.0)   # 50 pts/m³ = cobertura completa

        if cobertura >= 0.80:
            status = {'code': 'COMPLETO',  'emoji': '✅', 'texto': 'Completo',  'cor': '#4caf50'}
        elif cobertura >= 0.40:
            status = {'code': 'PARCIAL',   'emoji': '⚠️', 'texto': 'Parcial',   'cor': '#ff9800'}
        elif cobertura >= 0.10:
            status = {'code': 'INICIADO',  'emoji': '🔶', 'texto': 'Iniciado',  'cor': '#2196f3'}
        else:
            status = {'code': 'AUSENTE',   'emoji': '❌', 'texto': 'Ausente',   'cor': '#f44336'}

        # Exporta PLY por objeto
        ply_filename  = None
        json_filename = None
        if n_pts > 0:
            from werkzeug.utils import secure_filename
            import json
            nome_safe = secure_filename(obj['nome'])[:30]
            ply_filename  = f"{nome_safe}_{obj['guid'][:8]}.ply"
            json_filename = f"{nome_safe}_{obj['guid'][:8]}.json"

            pcd_exp = o3d.geometry.PointCloud()
            pcd_exp.points = o3d.utility.Vector3dVector(pts_obj)
            cor_rgb = {
                'COMPLETO': [0.2, 0.8, 0.2],
                'PARCIAL':  [1.0, 0.6, 0.0],
                'INICIADO': [0.2, 0.6, 1.0],
                'AUSENTE':  [0.8, 0.2, 0.2]
            }[status['code']]
            pcd_exp.paint_uniform_color(cor_rgb)
            o3d.io.write_point_cloud(str(output_dir / ply_filename), pcd_exp)

            # JSON para Three.js (Y/Z trocados)
            pts_3js = pts_obj[:, [0, 2, 1]]
            pts_3js[:, 2] = -pts_3js[:, 2]
            json_data = {
                'positions': pts_3js.flatten().tolist(),
                'color': cor_rgb,
                'count': n_pts
            }
            with open(output_dir / json_filename, 'w') as f:
                json.dump(json_data, f)

        resultados.append({
            'guid':     obj['guid'],
            'nome':     obj['nome'],
            'tipo':     obj['tipo'],
            'pavimento': obj['pavimento'],
            'volume_ifc': round(vol, 2),
            'pontos':   n_pts,
            'densidade': round(n_pts / vol, 1),
            'cobertura': round(cobertura * 100, 1),
            'status':   status,
            'eh_conexao': obj.get('eh_conexao', False),
            'phantom':  False,
            'ply_file': ply_filename,
            'json_file': json_filename,
            'bbox_normalized': _ifc_bbox_to_threejs(bbox)
        })

        print(f"  {obj['nome']:<25} {tipo:<12} {n_pts:>8,} pts  {status['emoji']} {status['texto']}")

    return resultados


# =========================
# STANDALONE
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply",        required=True)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DEFAULT))
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply)
    pts = np.asarray(pcd.points, dtype=np.float32)
    print(f"PLY carregado: {len(pts):,} pontos")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _carregar_modelo(Path(args.checkpoint), device)
    normais = _estimar_normais(pts)
    pts_feat = np.hstack([pts, normais])
    labels = _predizer_em_chunks(model, pts_feat, device)

    print("\nResultado:")
    for i, nome in enumerate(NOMES_CLASSES):
        cnt = int((labels == i).sum())
        print(f"  {i}: {nome:<22} {cnt:>8,} pts ({cnt/len(pts)*100:.1f}%)")
