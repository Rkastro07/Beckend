# -*- coding: utf-8 -*-
"""
INFERÊNCIA DE INSTÂNCIAS — RandLA-Net Instance
===============================================
Dado uma nuvem de pontos alinhada, detecta e separa cada objeto BIM
individualmente sem depender das bboxes do IFC para dividir os objetos.

Pipeline:
  1. RandLANetInstance → logits semânticos + offset 3D por ponto
  2. shifted_xyz = xyz + offset_predito  (pontos "pulam" para o centróide)
  3. DBSCAN por classe em shifted_xyz → cada cluster = uma instância
  4. Cruza instâncias detectadas com objetos IFC por proximidade de centróide
  5. Monta resultados no mesmo formato do app.py

Uso standalone:
  python randlanet/inference_instances.py --ply nuvem.ply
  python randlanet/inference_instances.py --ply nuvem.ply --ifc obra.ifc --pavimento 0F

Uso integrado (app.py):
  from randlanet.inference_instances import segmentar_instancias
  resultados = segmentar_instancias(pts_alinhado, objetos_ifc, output_dir)
"""

import sys
import json
import argparse
import numpy as np
import torch
import open3d as o3d
from pathlib import Path
from typing import List, Dict, Optional, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from model_instance import RandLANetInstance
from model import NUM_CLASSES
from dataset_generator import _estimar_normais, LABEL_MAP
from inference import _ifc_bbox_to_threejs

# ─────────────────────────────────────────────────────────────
CHECKPOINT_DEFAULT = Path(__file__).parent / "checkpoints" / "best_instance.pth"
CHUNK_SIZE  = 32768   # pontos por chunk (instance model é maior)

NOMES_CLASSES = [
    "background", "IfcWall", "IfcSlab", "IfcColumn",
    "IfcBeam", "IfcStair", "IfcRoof", "IfcSanitaryTerminal"
]

# Parâmetros DBSCAN por classe
# eps  = raio de vizinhança no espaço deslocado (shifted_xyz = xyz + offset)
#        Calibrado pelo IFC: paredes típicas têm esp. 0.05–0.43 m e comp. 2.8–9.5 m
#        Quanto pior o offset predito, maior o eps necessário para agrupar fragmentos.
#        Mas eps > espessura_mínima_entre_paredes (~0.05 m) começa a fundir paredes adjacentes.
#        → Usamos eps conservador + fusão posterior por guid IFC (ver _fundir_por_ifc)
# min_samples = pontos mínimos para formar um cluster (filtra ruído)
#        Densidade típica: ~400 pts/m² de parede → 2.8m × 2.8m × 400 ≈ 3136 pts/wall
#        min_samples=15 elimina fragmentos com < ~0.04 m² de área (bem pequenos)
DBSCAN_PARAMS: Dict[int, Dict] = {
    # eps no espaço deslocado (shifted_xyz = xyz + offset_predito)
    # Com offsets perfeitos, todos os pontos de um objeto convergem para 1 ponto → eps=0.
    # Com offsets imperfeitos (modelo ainda treinando), usamos eps proporcional à
    # dispersão esperada (~20% do comprimento típico do objeto):
    #   Paredes:  comprimento médio 4m  → 20% = 0.8m → mas sep. entre paredes ~0.3m  → eps=0.25
    #   Lajes:    objetos únicos por pavimento → eps maior ok
    #   Cobertura: idem lajes
    1: {"eps": 0.25, "min_samples": 15},   # IfcWall    — sep. mín. ~0.05m; 0.25 é conservador
    2: {"eps": 0.40, "min_samples": 20},   # IfcSlab    — geralmente 1 por pavimento
    3: {"eps": 0.15, "min_samples": 10},   # IfcColumn  — compactos
    4: {"eps": 0.15, "min_samples": 10},   # IfcBeam
    5: {"eps": 0.25, "min_samples": 15},   # IfcStair
    6: {"eps": 0.40, "min_samples": 20},   # IfcRoof    — geralmente 1 por pavimento
    7: {"eps": 0.12, "min_samples":  8},   # IfcSanitaryTerminal — pequenos
}
DBSCAN_DEFAULT = {"eps": 0.25, "min_samples": 15}

# Clusters com menos pontos que isso são ignorados (ruído residual)
MIN_PTS_CLUSTER = 20


# =========================
# MODELO
# =========================

def _carregar_modelo_instancia(checkpoint: Path, device: torch.device) -> RandLANetInstance:
    model = RandLANetInstance(num_classes=NUM_CLASSES, d_in=6).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


# =========================
# INFERÊNCIA EM CHUNKS
# =========================

def _predizer_instancia_chunks(
    model: RandLANetInstance,
    pts_feat: np.ndarray,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Roda o modelo em chunks e retorna:
        pred_labels : (N,)    int8   — classe semântica por ponto
        pred_offsets: (N, 3)  float32 — offset 3D por ponto
    """
    N = len(pts_feat)
    pred_labels  = np.zeros(N, dtype=np.int8)
    pred_offsets = np.zeros((N, 3), dtype=np.float32)

    # Normaliza XYZ (igual ao treino)
    centro = pts_feat[:, :3].mean(axis=0)
    scale  = np.abs(pts_feat[:, :3] - centro).max() + 1e-8
    pts_norm = pts_feat.copy().astype(np.float32)
    pts_norm[:, :3] = (pts_feat[:, :3] - centro) / scale

    for start in range(0, N, CHUNK_SIZE):
        chunk = pts_norm[start: start + CHUNK_SIZE]
        t = torch.from_numpy(chunk).unsqueeze(0).to(device)  # (1, M, 6)
        with torch.no_grad():
            logits, offsets = model(t)                         # (1,M,8), (1,M,3)

        pred_labels[start: start + CHUNK_SIZE]     = logits.squeeze(0).argmax(dim=-1).cpu().numpy().astype(np.int8)
        pred_offsets[start: start + CHUNK_SIZE]    = offsets.squeeze(0).cpu().numpy()

    # Desnormaliza offsets para escala real (metros)
    pred_offsets *= scale

    return pred_labels, pred_offsets


# =========================
# DBSCAN POR CLASSE
# =========================

def _clusterizar_instancias(
    pts_xyz:     np.ndarray,   # (N, 3) coords originais (metros)
    pred_labels: np.ndarray,   # (N,)   classe semântica
    pred_offsets:np.ndarray,   # (N, 3) offsets preditos
) -> np.ndarray:
    """
    Para cada classe semântica, desloca os pontos pelo offset e roda DBSCAN.
    Retorna inst_ids: (N,) int — 0 = background/ruído, 1..M = instâncias
    """
    from sklearn.cluster import DBSCAN

    inst_ids     = np.zeros(len(pts_xyz), dtype=np.int32)
    next_inst_id = 1

    for classe in range(1, NUM_CLASSES):
        mask = (pred_labels == classe)
        if mask.sum() < 5:
            continue

        idx   = np.where(mask)[0]
        pts_c = pts_xyz[idx]
        off_c = pred_offsets[idx]

        # Desloca pontos em direção ao centróide predito
        shifted = pts_c + off_c

        params  = DBSCAN_PARAMS.get(classe, DBSCAN_DEFAULT)
        db      = DBSCAN(eps=params["eps"], min_samples=params["min_samples"], n_jobs=-1)
        cluster_labels = db.fit_predict(shifted)   # -1 = ruído

        for cid in np.unique(cluster_labels):
            if cid == -1:
                continue   # ruído → permanece 0
            pts_no_cluster = idx[cluster_labels == cid]
            # Descarta clusters minúsculos (ruído residual do DBSCAN)
            if len(pts_no_cluster) < MIN_PTS_CLUSTER:
                continue
            inst_ids[pts_no_cluster] = next_inst_id
            next_inst_id += 1

    return inst_ids


# =========================
# FUSÃO DE FRAGMENTOS POR IFC
# =========================

def _fundir_por_ifc(
    pts_xyz:     np.ndarray,
    pred_labels: np.ndarray,
    inst_ids:    np.ndarray,
    objetos_ifc: List[Dict],
    margem:      float = 0.05,
) -> np.ndarray:
    """
    Usa as bboxes do IFC para fundir fragmentos DBSCAN que pertencem
    ao mesmo objeto.

    Lógica:
      Para cada ponto de foreground, verifica em qual bbox IFC ele cai
      (classe semântica correta + dentro da bbox com margem).
      Todos os pontos do mesmo objeto IFC recebem o mesmo inst_id
      → fragmentos da mesma parede são unidos.

    Retorna inst_ids corrigido: (N,) com menos instâncias, mais coerentes.
    """
    # Pré-computa classe IFC → índice do label semântico
    from dataset_generator import LABEL_MAP  # já importado no topo via inference

    inst_ids_new = np.zeros_like(inst_ids)
    next_id = 1

    for obj in objetos_ifc:
        tipo   = obj.get('tipo', '')
        classe = LABEL_MAP.get(tipo, 0)
        if classe == 0:
            continue

        bbox = obj['bbox']
        mask_cls  = (pred_labels == classe)
        mask_bbox = (
            (pts_xyz[:, 0] >= bbox['xmin'] - margem) &
            (pts_xyz[:, 0] <= bbox['xmax'] + margem) &
            (pts_xyz[:, 1] >= bbox['ymin'] - margem) &
            (pts_xyz[:, 1] <= bbox['ymax'] + margem) &
            (pts_xyz[:, 2] >= bbox['zmin'] - margem) &
            (pts_xyz[:, 2] <= bbox['zmax'] + margem)
        )
        mask = mask_cls & mask_bbox & (inst_ids > 0)  # só pontos que o DBSCAN capturou

        if mask.sum() < MIN_PTS_CLUSTER:
            continue

        inst_ids_new[mask] = next_id
        next_id += 1

    return inst_ids_new


# =========================
# CRUZA INSTÂNCIAS COM IFC
# =========================

def _cruzar_com_ifc(
    pts_xyz:     np.ndarray,
    pred_labels: np.ndarray,
    inst_ids:    np.ndarray,
    objetos_ifc: List[Dict],
) -> Dict[int, Dict]:
    """
    Para cada instância detectada, encontra o objeto IFC mais próximo
    do mesmo tipo semântico por distância centróide-a-centróide.

    Retorna dict: inst_id → objeto IFC correspondente (ou None)
    """
    # Pré-computa centróides por instância
    inst_centroids: Dict[int, np.ndarray] = {}
    inst_classes:   Dict[int, int]        = {}

    for iid in np.unique(inst_ids):
        if iid == 0:
            continue
        mask = (inst_ids == iid)
        inst_centroids[iid] = pts_xyz[mask].mean(axis=0)
        inst_classes[iid]   = int(np.bincount(pred_labels[mask].astype(np.int64)).argmax())

    # Para cada objeto IFC, computa centróide
    ifc_centroids = []
    for obj in objetos_ifc:
        bbox = obj['bbox']
        cx = (bbox['xmin'] + bbox['xmax']) / 2
        cy = (bbox['ymin'] + bbox['ymax']) / 2
        cz = (bbox['zmin'] + bbox['zmax']) / 2
        ifc_centroids.append(np.array([cx, cy, cz]))

    # Atribui instância ao IFC mais próximo do mesmo tipo
    inst_to_ifc: Dict[int, Optional[Dict]] = {}

    for iid, centroide in inst_centroids.items():
        classe = inst_classes[iid]
        tipo_esperado = NOMES_CLASSES[classe] if classe < len(NOMES_CLASSES) else None

        melhor_dist = float('inf')
        melhor_obj  = None

        for j, obj in enumerate(objetos_ifc):
            if obj.get('tipo') != tipo_esperado:
                continue
            dist = np.linalg.norm(centroide - ifc_centroids[j])
            if dist < melhor_dist:
                melhor_dist = dist
                melhor_obj  = obj

        inst_to_ifc[iid] = melhor_obj

    return inst_to_ifc


# =========================
# RESULTADO FINAL
# =========================

def _exportar_instancia(
    pts_obj:   np.ndarray,
    inst_id:   int,
    obj:       Optional[Dict],
    classe:    int,
    output_dir: Path,
) -> Tuple[Optional[str], Optional[str], Dict]:
    """Exporta PLY + JSON Three.js para uma instância detectada."""
    from werkzeug.utils import secure_filename

    n_pts = len(pts_obj)
    if n_pts == 0:
        return None, None, {}

    # Nome: usa o objeto IFC se encontrado, senão gera automático
    if obj:
        nome_safe = secure_filename(obj['nome'])[:28]
        guid_safe = obj['guid'][:8]
    else:
        nome_safe = f"{NOMES_CLASSES[classe]}_{inst_id}"
        guid_safe = f"inst{inst_id:04d}"

    ply_filename  = f"{nome_safe}_{guid_safe}.ply"
    json_filename = f"{nome_safe}_{guid_safe}.json"

    # Cor por tipo
    CORES_TIPO = {
        1: [0.2, 0.5, 1.0],   # parede — azul
        2: [0.8, 0.8, 0.2],   # laje — amarelo
        3: [1.0, 0.4, 0.4],   # coluna — vermelho
        4: [0.4, 1.0, 0.4],   # viga — verde
        5: [0.8, 0.4, 1.0],   # escada — roxo
        6: [1.0, 0.6, 0.2],   # cobertura — laranja
        7: [0.2, 1.0, 0.8],   # sanitário — ciano
    }
    cor_rgb = CORES_TIPO.get(classe, [0.7, 0.7, 0.7])

    # PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_obj)
    pcd.paint_uniform_color(cor_rgb)
    o3d.io.write_point_cloud(str(output_dir / ply_filename), pcd)

    # JSON Three.js (Y/Z swap + Z negado)
    pts_3js = pts_obj[:, [0, 2, 1]].copy()
    pts_3js[:, 2] = -pts_3js[:, 2]
    json_data = {
        'positions': pts_3js.flatten().tolist(),
        'color':     cor_rgb,
        'count':     n_pts
    }
    with open(output_dir / json_filename, 'w') as f:
        json.dump(json_data, f)

    return ply_filename, json_filename, {'cor': cor_rgb}


# =========================
# FUNÇÃO PRINCIPAL
# =========================

def segmentar_instancias(
    pts_alinhado: np.ndarray,
    objetos_ifc:  List[Dict],
    output_dir:   Path,
    checkpoint:   Path = CHECKPOINT_DEFAULT,
) -> Optional[List[Dict]]:
    """
    Segmenta instâncias individuais via RandLANetInstance + DBSCAN.

    Retorna lista de resultados no mesmo formato do app.py, ou None
    se o checkpoint não existir.

    Cada resultado representa UMA INSTÂNCIA detectada, cruzada com
    o objeto IFC mais próximo do mesmo tipo.
    """
    if not checkpoint.exists():
        print(f"  [Instance] Checkpoint nao encontrado: {checkpoint}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Instance] Segmentando instancias ({len(pts_alinhado):,} pts) em {device}...")

    # ── 1. Modelo ──────────────────────────────────────────────────────
    model = _carregar_modelo_instancia(checkpoint, device)

    normais  = _estimar_normais(pts_alinhado)
    pts_feat = np.hstack([pts_alinhado.astype(np.float32), normais])

    # ── 2. Inferência ──────────────────────────────────────────────────
    pred_labels, pred_offsets = _predizer_instancia_chunks(model, pts_feat, device)

    print("  Classes preditas:")
    for i, nome in enumerate(NOMES_CLASSES):
        cnt = int((pred_labels == i).sum())
        if cnt > 0:
            print(f"    {i}: {nome:<22} {cnt:>8,} pts")

    # ── 3. DBSCAN → instâncias ────────────────────────────────────────
    inst_ids = _clusterizar_instancias(pts_alinhado, pred_labels, pred_offsets)
    n_dbscan = int(inst_ids.max()) if inst_ids.max() > 0 else 0
    print(f"  {n_dbscan} clusters brutos pelo DBSCAN")

    # ── 3b. Fusão por IFC (une fragmentos do mesmo objeto) ────────────
    if objetos_ifc:
        inst_ids = _fundir_por_ifc(pts_alinhado, pred_labels, inst_ids, objetos_ifc)
        n_fundido = int(inst_ids.max()) if inst_ids.max() > 0 else 0
        print(f"  {n_fundido} instancias apos fusao por bbox IFC  (eram {n_dbscan})")
    else:
        n_fundido = n_dbscan

    # ── 4. Cruza com IFC ──────────────────────────────────────────────
    inst_to_ifc = _cruzar_com_ifc(pts_alinhado, pred_labels, inst_ids, objetos_ifc)

    # ── 5. Monta resultados ───────────────────────────────────────────
    output_dir = Path(output_dir)
    resultados = []

    for iid in sorted(inst_to_ifc.keys()):
        obj    = inst_to_ifc[iid]
        mask   = (inst_ids == iid)
        pts_obj = pts_alinhado[mask]
        classe  = int(np.bincount(pred_labels[mask].astype(np.int64)).argmax())

        ply_f, json_f, _ = _exportar_instancia(pts_obj, iid, obj, classe, output_dir)

        n_pts  = len(pts_obj)
        tipo   = NOMES_CLASSES[classe] if classe < len(NOMES_CLASSES) else "unknown"

        # Volume e cobertura a partir do IFC se disponível
        if obj:
            bbox = obj['bbox']
            vol  = max(
                (bbox['xmax'] - bbox['xmin']) *
                (bbox['ymax'] - bbox['ymin']) *
                (bbox['zmax'] - bbox['zmin']),
                1e-6
            )
            cobertura = min(n_pts / (vol * 50), 1.0)
            nome      = obj['nome']
            guid      = obj['guid']
            pavimento = obj.get('pavimento', '')
            bbox_3js  = _ifc_bbox_to_threejs(bbox)
        else:
            # Instância sem correspondência IFC — bbox pelo próprio cluster
            vol       = 1.0
            cobertura = 1.0   # detectada pela IA, sem referência IFC
            nome      = f"{tipo}_inst{iid:03d}"
            guid      = f"inst_{iid:06d}"
            pavimento = ''
            # Bbox pelo bounding box dos pontos do cluster
            mn = pts_obj.min(axis=0)
            mx = pts_obj.max(axis=0)
            bbox_raw = {'xmin': mn[0], 'xmax': mx[0],
                        'ymin': mn[1], 'ymax': mx[1],
                        'zmin': mn[2], 'zmax': mx[2]}
            bbox_3js = _ifc_bbox_to_threejs(bbox_raw)

        if cobertura >= 0.80:
            status = {'code': 'COMPLETO', 'emoji': 'ok', 'texto': 'Completo', 'cor': '#4caf50'}
        elif cobertura >= 0.40:
            status = {'code': 'PARCIAL',  'emoji': 'av', 'texto': 'Parcial',  'cor': '#ff9800'}
        elif cobertura >= 0.10:
            status = {'code': 'INICIADO', 'emoji': 'in', 'texto': 'Iniciado', 'cor': '#2196f3'}
        else:
            status = {'code': 'AUSENTE',  'emoji': 'xx', 'texto': 'Ausente',  'cor': '#f44336'}

        resultados.append({
            'guid':          guid,
            'nome':          nome,
            'tipo':          tipo,
            'pavimento':     pavimento,
            'inst_id':       iid,
            'volume_ifc':    round(vol, 2),
            'pontos':        n_pts,
            'densidade':     round(n_pts / vol, 1),
            'cobertura':     round(cobertura * 100, 1),
            'status':        status,
            'eh_conexao':    obj.get('eh_conexao', False) if obj else False,
            'phantom':       False,
            'ply_file':      ply_f,
            'json_file':     json_f,
            'bbox_normalized': bbox_3js,
        })

        print(f"  [{iid:3d}] {nome:<28} {tipo:<22} {n_pts:>7,} pts  {status['texto']}")

    return resultados


# =========================
# STANDALONE
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentacao de instancias BIM via RandLA-Net")
    parser.add_argument("--ply",        required=True,               help="Nuvem de pontos .ply ja alinhada")
    parser.add_argument("--ifc",        default=None,                help="Arquivo IFC (opcional, para cruzamento)")
    parser.add_argument("--pavimento",  default=None,                help="Nome do pavimento no IFC")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DEFAULT))
    parser.add_argument("--output",     default="output_instances",  help="Diretorio de saida")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carrega nuvem
    pcd = o3d.io.read_point_cloud(args.ply)
    pts = np.asarray(pcd.points, dtype=np.float64)
    pts = np.unique(pts, axis=0)
    print(f"PLY: {len(pts):,} pontos")

    # Carrega objetos IFC + alinha PLY se fornecido
    objetos_ifc = []
    if args.ifc and args.pavimento:
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            from app import (
                extrair_objetos_por_pavimento,
                alinhar_nuvem_com_ifc,
                corrigir_orientacao_por_pico_vertical,
                normalizar_coordenadas,
                detectar_paredes_conexao,
                marcar_conexoes_piso_teto,
            )
            objetos_ifc = extrair_objetos_por_pavimento(args.ifc, args.pavimento)
            print(f"IFC: {len(objetos_ifc)} objetos no pavimento '{args.pavimento}'")

            # Alinha PLY ao IFC (igual ao pipeline do app.py)
            objetos_ifc, _ = detectar_paredes_conexao(objetos_ifc)
            objetos_ifc    = marcar_conexoes_piso_teto(objetos_ifc)
            pts, _         = alinhar_nuvem_com_ifc(pts, objetos_ifc)
            pts, _, _      = corrigir_orientacao_por_pico_vertical(pts, objetos_ifc)
            pts, _, objetos_ifc = normalizar_coordenadas(pts, objetos_ifc)
            print(f"PLY alinhada ao IFC: {len(pts):,} pontos")

        except Exception as e:
            print(f"Aviso: nao foi possivel alinhar ao IFC: {e}")
            import traceback; traceback.print_exc()

    # Segmenta
    resultados = segmentar_instancias(
        pts.astype(np.float32),
        objetos_ifc,
        output_dir,
        checkpoint=Path(args.checkpoint)
    )

    if resultados:
        print(f"\n{len(resultados)} instancias exportadas em: {output_dir}")
        # Salva resumo JSON
        resumo_path = output_dir / "instancias_resumo.json"
        resumo = [{k: v for k, v in r.items() if k not in ('bbox_normalized',)} for r in resultados]

        # Converte tipos numpy para Python nativo (JSON não aceita int32/float32)
        def _to_native(obj):
            if isinstance(obj, dict):
                return {k: _to_native(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_native(v) for v in obj]
            if hasattr(obj, 'item'):   # np.int32, np.float32, etc.
                return obj.item()
            return obj

        with open(resumo_path, 'w', encoding='utf-8') as f:
            json.dump(_to_native(resumo), f, ensure_ascii=False, indent=2)
        print(f"Resumo: {resumo_path}")
