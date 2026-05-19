# -*- coding: utf-8 -*-
"""
BIM ANALYSIS API - VERSÃO HÍBRIDA 2.0 (CORRIGIDA)
=================================================
✅ Alinhamento: Centro Real (Preciso) + Scoring Amostral (Rápido)
✅ Proteções Anti-Leaking Completas
✅ Segurança: API Key via Variável de Ambiente

Como rodar:
export DEEPSEEK_API_KEY="sua-chave-aqui"
python app.py
"""

import os
import uuid
import traceback
import json
import sys
import requests

# Garante saída UTF-8 no Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import random
import tempfile
import numpy as np
from scipy.spatial import cKDTree

# Hook RandLA-Net (opcional — só ativa se PyTorch estiver instalado)
_RANDLANET_DISPONIVEL = False
_RANDLANET_INFERENCE = False
_INSTANCE_INFERENCE = False
try:
    import sys as _sys
    import os.path as _osp
    _sys.path.insert(0, _osp.join(_osp.dirname(_osp.abspath(__file__)), "randlanet"))
    from randlanet.dataset_generator import salvar_cena as _salvar_cena_dataset
    _RANDLANET_DISPONIVEL = True
    # Tenta carregar inferência (requer PyTorch + checkpoint)
    from randlanet.inference import classificar_com_modelo as _classificar_ai
    import os as _os
    _CHECKPOINT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "randlanet", "checkpoints", "best.pth")
    _CHECKPOINT_EXISTS = _os.path.exists(_CHECKPOINT)
    if _CHECKPOINT_EXISTS:
        _RANDLANET_INFERENCE = True
        print("🤖 RandLA-Net: modelo carregado (use_ai disponível)")
    else:
        print("🤖 RandLA-Net: checkpoint não encontrado, use_ai indisponível")
    # Inferência de instâncias (requer best_instance.pth)
    _INSTANCE_INFERENCE = False
    from randlanet.inference_instances import segmentar_instancias as _segmentar_instancias
    _CHECKPOINT_INST = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "randlanet", "checkpoints", "best_instance.pth")
    if _os.path.exists(_CHECKPOINT_INST):
        _INSTANCE_INFERENCE = True
        print("🤖 RandLA-Net Instance: modelo carregado (analisar_instancias disponível)")
    else:
        print("🤖 RandLA-Net Instance: checkpoint não encontrado")
except Exception as _e:
    print(f"⚠️  RandLA-Net não disponível: {_e}")
import ifcopenshell
import ifcopenshell.geom
from ifcopenshell.util.placement import get_local_placement
import open3d as o3d
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import permutations, product
from collections import Counter

# =========================
# CONFIGURAÇÃO
# =========================
TIPOS_INTERESSE = (
    "IfcWall", "IfcSlab",
    "IfcColumn", "IfcBeam", "IfcStair", "IfcRoof", "IfcSanitaryTerminal",
    "IfcDoor", "IfcWindow",
    # Camada 1: elementos estruturais/composicionais que o RF não foi treinado
    # explicitamente, mas geometricamente são equivalentes aos tipos conhecidos.
    # Mapeados via _ML_TIPO_MAP / _ML_EIXO_TIPO pra reutilizar o aprendizado.
    "IfcCovering",  # forros, revestimentos       → trata como IfcSlab
    "IfcPlate",     # placas/chapas estruturais   → trata como IfcSlab
    "IfcMember",    # membros estruturais         → trata como IfcBeam
    "IfcRailing",   # guarda-corpos, parapeitos   → trata como IfcWall
)
# IfcDoor e IfcWindow reativados — OBB elimina o leaking que justificou a remoção

# Margens de tolerância
MARGEM_PADRAO = 0.02      # 2cm
MARGEM_PORTA_JAN = 0.05   # 5cm

# Parâmetros de análise
NUM_FATIAS_ALTURA = 10
DENSIDADE_MIN = 50  # pontos/m³

# Thresholds de classificação
COBERTURA_COMPLETO = 0.80
COBERTURA_PARCIAL = 0.40
COBERTURA_INICIADO = 0.10

# Detector de conexões
THRESHOLD_CONEXAO = 0.30
COBERTURA_CONEXAO_MIN = 0.05

# Detecção de fantasma (T-junction)
PHANTOM_NUM_BINS = 10            # bins ao longo do eixo principal
PHANTOM_COBERTURA_MIN = 0.40     # mínimo 40% dos bins ocupados (era 30%)
PHANTOM_CONCENTRACAO_MAX = 0.65  # máximo 65% dos pontos numa extremidade (era 80%)
PHANTOM_FRAC_EXTREMIDADE = 0.30  # zona de extremidade = 30% de cada ponta (era 25%)
PHANTOM_MIN_COMPRIMENTO = 0.40   # ignorar paredes < 40cm (era 50cm)
PHANTOM_MIN_PTS_POR_BIN = 3      # mínimo de pontos para bin contar (era 5)

# Escalas para teste de alinhamento
SCALES_TO_TEST = (0.001, 0.01, 1.0, 100.0, 1000.0)

# DeepSeek API (Segurança: Pegar do ambiente)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Diretórios (compatível com Windows e Linux)
BASE_DIR = Path(tempfile.gettempdir())
UPLOAD_FOLDER = BASE_DIR / "bim_uploads"
OUTPUT_FOLDER = BASE_DIR / "bim_outputs"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# =========================
# FLASK APP
# =========================
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 2048 * 1024 * 1024  # 2GB

@app.after_request
def _no_cache(response):
    # /outputs/* são imutáveis (nome tem session_id + guid) — deixa o browser cachear
    if request.path.startswith('/outputs/'):
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    else:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


def _valid_upload(f, exts: Tuple[str, ...]) -> bool:
    """True se o filename termina numa das extensões permitidas (case-insensitive)."""
    if f is None or not f.filename:
        return False
    return f.filename.lower().endswith(exts)


# =========================
# CACHE DE IFC (evita reparse + re-upload do mesmo arquivo)
# =========================
# Camada (a): parse do ifcopenshell é caro (~2–5s pra IFCs grandes).
#   Cacheamos o ifcopenshell.file em memória por hash do conteúdo.
# Camada (b): /api/listar_pavimentos devolve `ifc_token` (hash).
#   Endpoints de análise aceitam token em vez de reupload do arquivo.
import hashlib
import threading
_IFC_CACHE: Dict[str, Dict] = {}  # hash -> {"file": ifcopenshell.file, "path": str}
_IFC_CACHE_LOCK = threading.Lock()  # Flask threaded por padrão; serializa populate do cache


def _hash_ifc_path(path: str, sample_bytes: int = 1024 * 1024) -> str:
    """Hash do 1º MB do arquivo (rápido, suficiente pra distinguir IFCs)."""
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        h.update(fh.read(sample_bytes))
    # Inclui tamanho pra diferenciar truncados de completos do mesmo prefixo
    h.update(str(os.path.getsize(path)).encode())
    return h.hexdigest()


def _cache_ifc(path: str) -> str:
    """Abre o IFC (se ainda não estiver em cache) e retorna o token.

    Double-checked locking: check sem lock (rápido) → lock → re-check → parse.
    Evita que duas threads façam ifcopenshell.open() do mesmo arquivo em paralelo.
    """
    token = _hash_ifc_path(path)
    if token in _IFC_CACHE:
        print(f"⚡ IFC cache hit ({token[:8]}…) — reusando parse")
        return token
    with _IFC_CACHE_LOCK:
        if token not in _IFC_CACHE:
            print(f"📂 IFC cache miss ({token[:8]}…) — fazendo parse de {os.path.basename(path)}")
            try:
                ifc = ifcopenshell.open(path)
            except Exception as e:
                # ifcopenshell solta exceções diversas (RuntimeError, IOError, schema errors…);
                # normaliza pra ValueError com mensagem útil pro usuário final.
                raise ValueError(
                    f'Não foi possível abrir o IFC "{os.path.basename(path)}": {e}. '
                    f'Verifique se o arquivo é IFC2x3/IFC4 válido e não está corrompido.'
                ) from e
            _IFC_CACHE[token] = {"file": ifc, "path": path}
        else:
            print(f"⚡ IFC cache hit ({token[:8]}…) — outro thread completou o parse")
    return token


def _resolve_ifc_from_request(req, session_id: str) -> Tuple[Optional[str], Optional[str], Optional[Tuple[str, int]]]:
    """
    Resolve o IFC da requisição: via upload (`ifc_file`) ou via token (`ifc_token`).
    Retorna (ifc_path, token, err). `err` é (msg, status) ou None.
    """
    token = req.form.get('ifc_token')
    if token:
        entry = _IFC_CACHE.get(token)
        if entry is None:
            return None, None, ("ifc_token desconhecido — reenvie o arquivo IFC", 410)
        return entry['path'], token, None

    ifc_file = req.files.get('ifc_file')
    if not ifc_file:
        return None, None, ("IFC não enviado (use ifc_file ou ifc_token)", 400)
    if not _valid_upload(ifc_file, ('.ifc',)):
        return None, None, ("Arquivo IFC precisa ter extensão .ifc", 400)

    ifc_path = UPLOAD_FOLDER / f"{session_id}_{secure_filename(ifc_file.filename)}"
    ifc_file.save(str(ifc_path))
    try:
        token = _cache_ifc(str(ifc_path))
    except ValueError as e:
        return None, None, (str(e), 400)
    return str(ifc_path), token, None


def _ler_ply_validado(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[str, int]]]:
    """Lê um PLY e valida que não ficou vazio. Devolve cores quando o scanner trouxe RGB.

    o3d.io.read_point_cloud não lança exceção em arquivos corrompidos — só devolve
    um PointCloud com 0 pontos, o que antes causava análises inteiras retornarem
    tudo "AUSENTE" em silêncio. Aqui deduplicamos e falhamos rápido com mensagem útil.

    Retorno: (pts[N,3] float, colors[N,3] float em [0,1] ou None se PLY sem RGB, err)
    Dedup preserva a ordem original via return_index — necessário pra cores ficarem
    alinhadas com os pontos.
    """
    try:
        pcd = o3d.io.read_point_cloud(path)
    except Exception as e:
        return np.empty((0, 3)), None, (f'Falha ao ler PLY: {e}', 400)
    pts = np.asarray(pcd.points, dtype=float)
    if pts.size == 0:
        return pts, None, (
            f'PLY "{os.path.basename(path)}" está vazio ou ilegível. '
            f'Verifique se o arquivo não está corrompido.',
            400,
        )
    has_colors = len(pcd.colors) > 0
    colors = np.asarray(pcd.colors, dtype=float) if has_colors else None

    # Dedup mantendo a ordem (sort dos índices únicos) para colors permanecer alinhada
    _, idx_unique = np.unique(pts, axis=0, return_index=True)
    idx_unique.sort()
    pts = pts[idx_unique]
    if colors is not None:
        colors = colors[idx_unique]

    if pts.size == 0:
        return pts, None, (
            f'PLY "{os.path.basename(path)}" só tem pontos duplicados (0 únicos).',
            400,
        )
    return pts, colors, None

# JSON encoder que converte numpy types automaticamente (evita int32/float64 errors)
import json as _json
class _NumpyEncoder(_json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):      # np.int32, np.float64, etc.
            return obj.item()
        if hasattr(obj, 'tolist'):     # np.ndarray
            return obj.tolist()
        return super().default(obj)
app.json_encoder = _NumpyEncoder


# =========================
# CONVERSÃO DE EIXOS (IFC -> Three.js)
# =========================
# IFC usa Z-up right-handed: (X, Y, Z) onde Z é vertical.
# Three.js usa Y-up right-handed: (X, Y, Z) onde Y é vertical.
# A conversão correta preservando handedness é rotação de -90° em torno de X:
#     (x, y, z)_IFC  →  (x, z, -y)_Three.js
#
# O `-y` é crítico. Sem a negação, a transformação vira LEFT-HANDED e objetos
# com rotação (matrix_local) aparecem ESPELHADOS no viewer. Objetos axis-aligned
# não percebem o bug (bbox é simétrica), por isso o problema estava escondido —
# sintéticos antigos eram quase todos axis-aligned.
def converter_ifc_para_threejs(bbox_ifc: Dict) -> Dict:
    """Converte BBox AABB do IFC (Z-up) para Three.js (Y-up).

    Rotação de -90° em X + rotação 180° em Y (faz a câmera default do viewer
    olhar pra "frente" do prédio em vez da "traseira"):
        (x, y, z)_IFC → (-x, z, -y)_TJ
    """
    return {
        'xmin': -bbox_ifc['xmax'],   # X_TJ = -X_IFC → max_IFC vira min_TJ
        'xmax': -bbox_ifc['xmin'],
        'ymin': bbox_ifc['zmin'],    # Y_TJ = Z_IFC
        'ymax': bbox_ifc['zmax'],
        'zmin': -bbox_ifc['ymax'],   # Z_TJ = -Y_IFC → max_IFC vira min_TJ
        'zmax': -bbox_ifc['ymin'],
    }


def converter_pontos_ifc_para_threejs(pts: np.ndarray) -> np.ndarray:
    """Converte nuvem de pontos do IFC (Z-up) para Three.js (Y-up) preservando handedness."""
    if len(pts) == 0:
        return pts
    pts_converted = np.empty_like(pts)
    pts_converted[:, 0] = -pts[:, 0]   # X_TJ = -X_IFC
    pts_converted[:, 1] =  pts[:, 2]   # Y_TJ =  Z_IFC
    pts_converted[:, 2] = -pts[:, 1]   # Z_TJ = -Y_IFC
    return pts_converted


def calcular_obb_corners_threejs(obj: Dict) -> List[List[float]]:
    """
    Calcula os 8 vértices do OBB em coordenadas Three.js (Y=altura).
    Retorna None se o objeto não tiver placement local válido.
    Ordem dos cantos: 0-3 são a face inferior, 4-7 a face superior, em CCW.
    """
    bbox_l = obj.get('bbox_local_obb')
    ml     = obj.get('matrix_local')
    if bbox_l is None or ml is None:
        return None

    # 8 cantos da AABB no espaço local
    xmin, xmax = bbox_l['xmin'], bbox_l['xmax']
    ymin, ymax = bbox_l['ymin'], bbox_l['ymax']
    zmin, zmax = bbox_l['zmin'], bbox_l['zmax']
    cantos_local = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin],
        [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax],
        [xmax, ymax, zmax], [xmin, ymax, zmax],
    ], dtype=float)

    # Transforma para world (já em coords normalizadas porque ml[:3,3] foi
    # atualizado em normalizar_coordenadas)
    R = ml[:3, :3]
    t = ml[:3, 3]
    cantos_world = (R @ cantos_local.T).T + t  # (8, 3) em IFC

    # Se é cross-floor fatiado, clipa Z global ao range da fatia
    slice_z = obj.get('slice_z')
    if slice_z is not None:
        z_lo, z_hi = slice_z
        cantos_world[:, 2] = np.clip(cantos_world[:, 2], z_lo, z_hi)

    # Converte IFC (Z-up) → Three.js (Y-up) com flip X pra câmera ver de frente:
    # (x, y, z)_IFC → (-x, z, -y)_TJ
    cantos_tj = np.empty_like(cantos_world)
    cantos_tj[:, 0] = -cantos_world[:, 0]
    cantos_tj[:, 1] =  cantos_world[:, 2]
    cantos_tj[:, 2] = -cantos_world[:, 1]
    return cantos_tj.tolist()


# =========================
# EXTRAÇÃO IFC
# =========================
def extrair_pavimentos(ifc_path: str) -> List[str]:
    """Extrai lista de pavimentos do arquivo IFC."""
    token = _cache_ifc(ifc_path)
    f = _IFC_CACHE[token]['file']
    pavs = set()
    for s in f.by_type("IfcBuildingStorey"):
        if s.Name:
            pavs.add(s.Name)
    return sorted(list(pavs))


def extrair_objetos_por_pavimento(
        ifc_path: str,
        pavimento_alvo: str,
        incluir_estrutura_cruzando: bool = False,
        min_overlap_frac: float = 0.30,
        margem_z: float = 0.30) -> List[Dict]:
    """Extrai objetos IFC filtrados por pavimento com suas bounding boxes.

    Se pavimento_alvo == "__TODOS__" (ou vazio), carrega todos os pavimentos.

    Se `incluir_estrutura_cruzando=True`, faz 2 passadas:
      Passada 1: pega objetos por nome de storey (normalizado: 14º ≡ OSSO 14º).
      Passada 2: calcula Z-range da passada 1 (com margem `margem_z` em metros)
                 e inclui TAMBÉM objetos de OUTROS storeys cujo bbox Z intersecte
                 esse range com ≥ `min_overlap_frac` (fração da altura do objeto).
    Isso captura pilares, shafts, escadas e vigas que o projetista atribuiu a
    outro storey no IFC mas que fisicamente passam pelo pavimento analisado.
    """
    token = _cache_ifc(ifc_path)
    f = _IFC_CACHE[token]['file']
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    todos = (not pavimento_alvo) or pavimento_alvo == "__TODOS__"
    objetos = []
    guids_incluidos = set()

    # Agrupa storeys equivalentes: "14º PAVIMENTO" + "OSSO 14º PAVIMENTO" contam como o mesmo andar.
    def _norm_pav(nome):
        if not nome:
            return ""
        n = nome.strip()
        if n.upper().startswith("OSSO "):
            n = n[5:].strip()
        return n.upper()

    alvo_norm = _norm_pav(pavimento_alvo) if not todos else None
    usa_geom = incluir_estrutura_cruzando and not todos

    if todos:
        print("Filtrando objetos: TODOS OS PAVIMENTOS")
    else:
        print(f"Filtrando objetos para o pavimento: {pavimento_alvo} "
              f"(inclui OSSO equivalente, normalizado='{alvo_norm}')")

    n_por_storey = 0
    n_por_geom   = 0
    # Cache da geometria de produtos NÃO matched por storey (reaproveitar na 2ª passada)
    cache_nao_storey = []  # lista de (product, tipo_base, pav_do_objeto, verts, shape)
    # Z-range por storey normalizado.
    # FONTE PRIMÁRIA: IfcBuildingStorey.Elevation (valor oficial do projeto).
    #   zmin = Elevation do storey; zmax = Elevation do próximo storey.
    # FALLBACK: envelope dos objetos (só se Elevation ausente), atualizado no loop abaixo.
    # Usar Elevation evita que objetos cross-floor marcados num storey puxem o
    # range pra fora do limite físico real do pavimento.
    storeys_z: Dict[str, Tuple[float, float]] = {}
    storeys_z_from_elev: Dict[str, Tuple[float, float]] = {}
    try:
        _sts = [s for s in f.by_type("IfcBuildingStorey")
                if getattr(s, "Elevation", None) is not None]
        _sts.sort(key=lambda s: float(s.Elevation))
        for i, s in enumerate(_sts):
            z_lo = float(s.Elevation)
            if i + 1 < len(_sts):
                z_hi = float(_sts[i + 1].Elevation)
            else:
                # Último storey: sem próximo, estima teto com altura média dos outros
                if len(_sts) >= 2:
                    alturas = [float(_sts[j+1].Elevation) - float(_sts[j].Elevation)
                               for j in range(len(_sts)-1)]
                    h_med = sum(alturas) / len(alturas)
                else:
                    h_med = 3.0
                z_hi = z_lo + h_med
            storeys_z_from_elev[_norm_pav(s.Name)] = (z_lo, z_hi)
        if storeys_z_from_elev:
            print(f"   📏 Z-range de {len(storeys_z_from_elev)} storeys via IfcBuildingStorey.Elevation")
            storeys_z.update(storeys_z_from_elev)
    except Exception as _e:
        print(f"   ⚠️ Falha ao ler Elevation dos storeys: {_e}. Usando envelope fallback.")

    for product in f.by_type("IfcProduct"):
        tipo_base = None
        for t in TIPOS_INTERESSE:
            if product.is_a(t):
                tipo_base = t
                break
        if tipo_base is None:
            continue

        pav_do_objeto = None
        for rel in getattr(product, "ContainedInStructure", []):
            if rel.RelatingStructure.is_a("IfcBuildingStorey"):
                pav_do_objeto = rel.RelatingStructure.Name
                break

        storey_match = todos or (_norm_pav(pav_do_objeto) == alvo_norm)

        # Se não passa por storey e não tem filtro geom, pula SEM computar geometria
        if not storey_match and not usa_geom:
            continue

        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            verts = np.array(shape.geometry.verts).reshape(-1, 3)
            if verts.size == 0:
                continue

            # Atualiza Z-range SÓ se Elevation não deu conta (storey sem Elevation no IFC).
            # Se já veio de Elevation, NÃO mexe — senão objetos cross-floor puxam o range.
            pav_norm_obj = _norm_pav(pav_do_objeto)
            if pav_norm_obj and pav_norm_obj not in storeys_z_from_elev:
                z_lo = float(verts[:, 2].min())
                z_hi = float(verts[:, 2].max())
                if pav_norm_obj in storeys_z:
                    old_lo, old_hi = storeys_z[pav_norm_obj]
                    storeys_z[pav_norm_obj] = (min(old_lo, z_lo), max(old_hi, z_hi))
                else:
                    storeys_z[pav_norm_obj] = (z_lo, z_hi)

            # Objetos fora do storey vão pro cache — serão revisitados na 2ª passada
            if not storey_match:
                cache_nao_storey.append((product, tipo_base, pav_do_objeto, verts, shape))
                continue

            incluido_por = "storey"
            if todos and pav_do_objeto is None:
                pav_do_objeto = "sem_pavimento"

            xmin, xmax = float(verts[:, 0].min()), float(verts[:, 0].max())
            ymin, ymax = float(verts[:, 1].min()), float(verts[:, 1].max())
            zmin, zmax = float(verts[:, 2].min()), float(verts[:, 2].max())

            nome = getattr(product, 'Name',
                           None) or f"{product.is_a()}_{product.GlobalId[:8]}"

            # Área de superfície real (usada como feature pelo ML)
            try:
                faces = np.array(shape.geometry.faces).reshape(-1, 3)
                mesh_tmp = o3d.geometry.TriangleMesh()
                mesh_tmp.vertices  = o3d.utility.Vector3dVector(verts)
                mesh_tmp.triangles = o3d.utility.Vector3iVector(faces)
                area_ifc = float(mesh_tmp.get_surface_area())
            except Exception:
                area_ifc = 2 * ((xmax-xmin)*(ymax-ymin) + (xmax-xmin)*(zmax-zmin) + (ymax-ymin)*(zmax-zmin))

            # ===== OBB: matriz de placement local + bbox no espaço local =====
            matrix_local = None
            bbox_local_obb = None
            try:
                matrix_local = get_local_placement(product.ObjectPlacement)
                R = matrix_local[:3, :3]
                t = matrix_local[:3, 3]
                R_inv = np.linalg.inv(R)
                vl = (R_inv @ (verts - t).T).T
                bbox_local_obb = {
                    'xmin': float(vl[:, 0].min()), 'xmax': float(vl[:, 0].max()),
                    'ymin': float(vl[:, 1].min()), 'ymax': float(vl[:, 1].max()),
                    'zmin': float(vl[:, 2].min()), 'zmax': float(vl[:, 2].max()),
                }
            except Exception:
                matrix_local = None
                bbox_local_obb = None

            objetos.append({
                'guid': product.GlobalId,
                'tipo': tipo_base,  # tipo base (IfcWall, nao IfcWallStandardCase)
                'nome': nome,
                'pavimento': pav_do_objeto or pavimento_alvo,
                'bbox': {
                    'xmin': xmin, 'xmax': xmax,
                    'ymin': ymin, 'ymax': ymax,
                    'zmin': zmin, 'zmax': zmax
                },
                'volume_ifc': (xmax - xmin) * (ymax - ymin) * (zmax - zmin),
                'area_ifc':   area_ifc,
                'matrix_local':    matrix_local,     # OBB
                'bbox_local_obb':  bbox_local_obb,   # OBB
                'incluido_por':    incluido_por,     # "storey" ou "geom"
            })
            if incluido_por == "storey":
                n_por_storey += 1
            else:
                n_por_geom += 1
            guids_incluidos.add(product.GlobalId)
        except Exception:
            continue

    # ===== 2ª PASSADA: estrutura vertical longa que atravessa o storey =====
    # Regras:
    #   - NUNCA inclui IfcSlab (laje sempre pertence ao pavimento que o IFC diz)
    #   - Só aceita se altura do elemento > 1.5× altura do pavimento (vertical longo)
    #   - Overlap Z mínimo de 5cm
    #   - bbox é CLIPADA ao range do pavimento (mostra só a fatia)
    #   - Metadata cross_floor adicionada pra UI exibir contexto
    if usa_geom and alvo_norm in storeys_z and cache_nao_storey:
        storey_z_min, storey_z_max = storeys_z[alvo_norm]
        altura_storey = storey_z_max - storey_z_min
        _fonte_z = "Elevation" if alvo_norm in storeys_z_from_elev else "envelope (fallback)"
        print(f"   🔲 Z-range do storey '{alvo_norm}': "
              f"[{storey_z_min:.2f}, {storey_z_max:.2f}] m  (altura {altura_storey:.2f}m)  "
              f"← fonte: {_fonte_z}")
        print(f"      Candidatos de outros storeys: {len(cache_nao_storey)} objetos")

        MIN_OVERLAP_M    = 0.05    # 5cm
        HEIGHT_RATIO_MIN = 1.5     # elemento precisa ser >1.5× altura do pavimento

        n_rej_slab    = 0
        n_rej_baixo   = 0  # elemento não é vertical longo
        n_rej_sem_ov  = 0  # sem overlap suficiente
        for (product, tipo_base, pav_do_objeto, verts, shape) in cache_nao_storey:
            try:
                # ─── 1) Slabs ficam no storey do IFC, sempre ─────────────────
                if tipo_base == "IfcSlab" or product.is_a("IfcSlab"):
                    n_rej_slab += 1
                    continue

                zmin_v = float(verts[:, 2].min())
                zmax_v = float(verts[:, 2].max())
                altura_v = zmax_v - zmin_v
                if altura_v <= 0:
                    continue

                # ─── 2) Precisa ser vertical longo ───────────────────────────
                if altura_storey > 0 and altura_v <= HEIGHT_RATIO_MIN * altura_storey:
                    n_rej_baixo += 1
                    continue

                # ─── 3) Overlap Z com o pavimento ≥ 5cm ──────────────────────
                overlap_zmin = max(zmin_v, storey_z_min)
                overlap_zmax = min(zmax_v, storey_z_max)
                overlap     = overlap_zmax - overlap_zmin
                if overlap < MIN_OVERLAP_M:
                    n_rej_sem_ov += 1
                    continue

                xmin, xmax = float(verts[:, 0].min()), float(verts[:, 0].max())
                ymin, ymax = float(verts[:, 1].min()), float(verts[:, 1].max())

                nome = getattr(product, 'Name', None) or f"{product.is_a()}_{product.GlobalId[:8]}"

                try:
                    faces = np.array(shape.geometry.faces).reshape(-1, 3)
                    mesh_tmp = o3d.geometry.TriangleMesh()
                    mesh_tmp.vertices  = o3d.utility.Vector3dVector(verts)
                    mesh_tmp.triangles = o3d.utility.Vector3iVector(faces)
                    area_total = float(mesh_tmp.get_surface_area())
                except Exception:
                    area_total = 2 * (
                        (xmax-xmin)*(ymax-ymin)
                        + (xmax-xmin)*altura_v
                        + (ymax-ymin)*altura_v
                    )

                # Cross-floor: escala área/volume pela FATIA (overlap/altura_v)
                # pra AI não cobrar pontos da viga/pilar inteiro quando só vê
                # a parte do pavimento atual.
                frac_altura = overlap / altura_v if altura_v > 0 else 1.0
                area_ifc = area_total * frac_altura

                matrix_local = None
                bbox_local_obb = None
                try:
                    matrix_local = get_local_placement(product.ObjectPlacement)
                    R = matrix_local[:3, :3]
                    t = matrix_local[:3, 3]
                    R_inv = np.linalg.inv(R)
                    vl = (R_inv @ (verts - t).T).T
                    bbox_local_obb = {
                        'xmin': float(vl[:, 0].min()), 'xmax': float(vl[:, 0].max()),
                        'ymin': float(vl[:, 1].min()), 'ymax': float(vl[:, 1].max()),
                        'zmin': float(vl[:, 2].min()), 'zmax': float(vl[:, 2].max()),
                    }
                except Exception:
                    matrix_local = None
                    bbox_local_obb = None

                # Andares que esse elemento atravessa (pra exibição na UI)
                andares_atravessa = []
                for storey_nome, (s_zmin, s_zmax) in storeys_z.items():
                    if zmax_v >= s_zmin and zmin_v <= s_zmax:
                        andares_atravessa.append(storey_nome)
                andares_atravessa.sort(key=lambda n: storeys_z[n][0])

                # BBox CLIPADA ao pavimento (é o que a IA e o viewer vêem)
                objetos.append({
                    'guid': product.GlobalId,
                    'tipo': tipo_base,
                    'nome': nome,
                    'pavimento': pav_do_objeto or pavimento_alvo,
                    'bbox': {
                        'xmin': xmin, 'xmax': xmax,
                        'ymin': ymin, 'ymax': ymax,
                        'zmin': overlap_zmin, 'zmax': overlap_zmax,
                    },
                    'volume_ifc': (xmax - xmin) * (ymax - ymin) * overlap,
                    'area_ifc':   area_ifc,
                    'matrix_local':    matrix_local,
                    'bbox_local_obb':  bbox_local_obb,
                    'incluido_por':    "geom",
                    # ── metadata cross-floor ────────────────────────────────
                    'cross_floor':         True,
                    'slice_z':             (overlap_zmin, overlap_zmax),
                    'altura_total':        altura_v,
                    'z_total':             [zmin_v, zmax_v],
                    'andares_atravessa':   andares_atravessa,
                    'storey_original_ifc': pav_do_objeto or "sem_pavimento",
                    'fatia_atual': {
                        'pavimento': pavimento_alvo,
                        'z_range':   [overlap_zmin, overlap_zmax],
                        'altura':    overlap,
                    },
                })
                n_por_geom += 1
            except Exception:
                continue

        print(f"      Rejeitados: slab={n_rej_slab}  "
              f"não-vertical={n_rej_baixo}  sem-overlap={n_rej_sem_ov}")

    if todos:
        pavs_unicos = sorted(set(o['pavimento'] for o in objetos))
        print(f"Encontrados {len(objetos)} objetos em {len(pavs_unicos)} pavimentos: {pavs_unicos}")
    else:
        print(f"Encontrados {len(objetos)} objetos no pavimento {pavimento_alvo} "
              f"(storey={n_por_storey} + geom={n_por_geom})")
    return objetos


# =========================
# DETECTOR DE CONEXÕES (ANTI-LEAKING)
# =========================
def eh_parede_de_conexao(
        obj: Dict,
        todos_objetos: List[Dict],
        threshold_distancia: float = THRESHOLD_CONEXAO) -> bool:
    """Detecta se uma parede conecta 2+ outras paredes (formato T ou L)."""
    if obj['tipo'] != 'IfcWall':
        return False

    bbox = obj['bbox']
    dx = bbox['xmax'] - bbox['xmin']
    dy = bbox['ymax'] - bbox['ymin']

    # Determina extremidades baseado na orientação
    if dx > dy:
        extremidade_1 = np.array([bbox['xmin'],
                                  (bbox['ymin'] + bbox['ymax']) / 2,
                                  (bbox['zmin'] + bbox['zmax']) / 2])
        extremidade_2 = np.array([bbox['xmax'],
                                  (bbox['ymin'] + bbox['ymax']) / 2,
                                  (bbox['zmin'] + bbox['zmax']) / 2])
    else:
        extremidade_1 = np.array([(bbox['xmin'] + bbox['xmax']) / 2,
                                  bbox['ymin'], (bbox['zmin'] + bbox['zmax']) / 2])
        extremidade_2 = np.array([(bbox['xmin'] + bbox['xmax']) / 2,
                                  bbox['ymax'], (bbox['zmin'] + bbox['zmax']) / 2])

    conexoes = set()

    for outra in todos_objetos:
        if outra['guid'] == obj['guid'] or outra['tipo'] != 'IfcWall':
            continue

        outra_bbox = outra['bbox']
        z_outra = (outra_bbox['zmin'] + outra_bbox['zmax']) / 2
        z_obj = (bbox['zmin'] + bbox['zmax']) / 2

        # Ignora paredes em andares diferentes
        if abs(z_outra - z_obj) > 0.5:
            continue

        for idx, extremidade in enumerate([extremidade_1, extremidade_2]):
            dentro_ou_proximo = (
                extremidade[0] >= outra_bbox['xmin'] - threshold_distancia and
                extremidade[0] <= outra_bbox['xmax'] + threshold_distancia and
                extremidade[1] >= outra_bbox['ymin'] - threshold_distancia and
                extremidade[1] <= outra_bbox['ymax'] + threshold_distancia
            )

            if dentro_ou_proximo:
                conexoes.add((outra['guid'], idx))

    num_paredes_conectadas = len(set(guid for guid, _ in conexoes))
    return num_paredes_conectadas >= 2


def detectar_paredes_conexao(objetos: List[Dict]) -> Tuple[List[Dict], int]:
    """Detecta e marca todas as paredes de conexão."""
    print("\n" + "=" * 70)
    print("DETECTOR DE PAREDES DE CONEXÃO")
    print("=" * 70)

    num_conexao = 0

    for obj in objetos:
        if eh_parede_de_conexao(obj, objetos, THRESHOLD_CONEXAO):
            obj['eh_conexao'] = True
            num_conexao += 1
        else:
            obj['eh_conexao'] = False

    print(f"✅ Detectadas {num_conexao} paredes de conexão")

    return objetos, num_conexao


# =========================
# CONEXÕES PISO/TETO (ANTI-LEAKING)
# =========================
def marcar_conexoes_piso_teto(
    objetos: List[Dict],
    tol_z: float = 0.20,
    overlap_xy_min_frac: float = 0.2
) -> List[Dict]:
    """
    Marca em cada parede se ela conecta com piso (slab abaixo) e/ou teto (slab acima).
    Usado para cortes dinâmicos anti-leaking.
    """
    paredes = [o for o in objetos if o['tipo'] == "IfcWall"]
    slabs = [o for o in objetos if o['tipo'] == "IfcSlab"]

    # Inicializa flags
    for p in paredes:
        p['conecta_piso'] = False
        p['conecta_teto'] = False

    if not paredes or not slabs:
        return objetos

    def overlap_xy(b1, b2):
        x_overlap = max(0.0, min(b1['xmax'], b2['xmax']) - max(b1['xmin'], b2['xmin']))
        y_overlap = max(0.0, min(b1['ymax'], b2['ymax']) - max(b1['ymin'], b2['ymin']))
        return x_overlap * y_overlap

    for parede in paredes:
        bbox_p = parede['bbox']
        area_p = max(1e-6, (bbox_p['xmax'] - bbox_p['xmin'])
                     * (bbox_p['ymax'] - bbox_p['ymin']))

        for slab in slabs:
            bbox_s = slab['bbox']
            area_ov = overlap_xy(bbox_p, bbox_s)

            if area_ov / area_p < overlap_xy_min_frac:
                continue

            # Piso: slab embaixo, encostando no zmin da parede
            if abs(bbox_s['zmax'] - bbox_p['zmin']) <= tol_z:
                parede['conecta_piso'] = True

            # Teto: slab em cima, encostando no zmax da parede
            if abs(bbox_s['zmin'] - bbox_p['zmax']) <= tol_z:
                parede['conecta_teto'] = True

            if parede['conecta_piso'] and parede['conecta_teto']:
                break

    # Log
    piso_only = sum(
        1 for p in paredes if p['conecta_piso'] and not p['conecta_teto'])
    teto_only = sum(
        1 for p in paredes if p['conecta_teto'] and not p['conecta_piso'])
    piso_teto = sum(1 for p in paredes if p['conecta_piso'] and p['conecta_teto'])

    print(f"\n📊 Conexões Piso/Teto:")
    print(f"   Só piso: {piso_only} | Só teto: {teto_only} | Ambos: {piso_teto}")

    return objetos


# =========================
# GEOMETRIA E BOUNDS
# =========================
def _bounds_from_objs(objetos: List[Dict]):
    """Calcula bounds globais a partir de lista de objetos."""
    if not objetos:
        return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

    xs = [o['bbox']['xmin'] for o in objetos] + \
        [o['bbox']['xmax'] for o in objetos]
    ys = [o['bbox']['ymin'] for o in objetos] + \
        [o['bbox']['ymax'] for o in objetos]
    zs = [o['bbox']['zmin'] for o in objetos] + \
        [o['bbox']['zmax'] for o in objetos]

    bmin = np.array([min(xs), min(ys), min(zs)], dtype=float)
    bmax = np.array([max(xs), max(ys), max(zs)], dtype=float)
    center = (bmin + bmax) / 2.0
    extent = bmax - bmin

    return bmin, bmax, center, extent


def _perm_sign_to_R(perm, sign, scale=1.0):
    """Converte permutação e sinais para matriz de rotação/escala."""
    R = np.zeros((3, 3), dtype=float)
    for i, p in enumerate(perm):
        R[i, p] = float(sign[i]) * float(scale)
    return R


# =========================
# FILTRO DE PONTOS (ANTI-LEAKING)
# =========================
def filtrar_pontos_aabb(
    pts: np.ndarray,
    bbox: Dict,
    margem: float,
    frac_bottom: float = 0.0,
    frac_top: float = 0.0
) -> np.ndarray:
    """
    Filtra pontos pelo AABB com cortes dinâmicos em Z (anti-leaking).
    """
    if pts.size == 0 or not bbox:
        return np.empty((0, 3), dtype=float)

    # X e Y com margem normal
    xmin = bbox['xmin'] - margem
    xmax = bbox['xmax'] + margem
    ymin = bbox['ymin'] - margem
    ymax = bbox['ymax'] + margem

    # Z com corte dinâmico anti-leaking
    zmin_orig = bbox['zmin']
    zmax_orig = bbox['zmax']
    altura = zmax_orig - zmin_orig

    if altura > 0 and (frac_bottom > 0.0 or frac_top > 0.0):
        frac_bottom = max(0.0, min(0.9, frac_bottom))
        frac_top = max(0.0, min(0.9, frac_top))

        zmin = zmin_orig + frac_bottom * altura
        zmax = zmax_orig - frac_top * altura

        # Se exagerou no corte, volta ao original
        if zmin >= zmax:
            zmin, zmax = zmin_orig, zmax_orig
    else:
        zmin, zmax = zmin_orig, zmax_orig

    zmin -= margem
    zmax += margem

    mask = (
        (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
        (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
        (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
    )

    return pts[mask]


# =========================
# CANDIDATOS VIA KDTREE (pré-filtro espacial)
# =========================
def candidatos_obj(
    pts: np.ndarray,
    tree: Optional[cKDTree],
    bbox: Dict,
    margem: float = 0.05,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Retorna apenas os pontos da cena que podem estar dentro do objeto.

    Usa uma esfera envolvente da bbox (expandida pela `margem`) como pré-filtro
    barato via cKDTree.query_ball_point. O filtro OBB/AABB exato é feito depois
    no subset minúsculo retornado aqui.

    Retorna `(pts_subset, idx)`. `idx` é o array de índices originais (no `pts` da
    cena) que pode ser reusado para fatiar arrays paralelos como `colors`. Quando
    `tree` é None (fallback) ou `pts` está vazio, `idx` vem como None — nesse caso
    o chamador deve assumir que `pts_subset` é o array inteiro.
    """
    if tree is None or pts.size == 0:
        return pts, None
    cx = 0.5 * (bbox['xmin'] + bbox['xmax'])
    cy = 0.5 * (bbox['ymin'] + bbox['ymax'])
    cz = 0.5 * (bbox['zmin'] + bbox['zmax'])
    dx = bbox['xmax'] - bbox['xmin']
    dy = bbox['ymax'] - bbox['ymin']
    dz = bbox['zmax'] - bbox['zmin']
    raio = 0.5 * float(np.sqrt(dx * dx + dy * dy + dz * dz)) + float(margem)
    # return_sorted=False: não precisamos de ordem (pts[idx] ignora isso)
    # workers=-1: paraleliza usando todos os cores disponíveis
    idx = tree.query_ball_point([cx, cy, cz], raio, return_sorted=False, workers=-1)
    if not idx:
        return np.empty((0, 3), dtype=pts.dtype), np.empty((0,), dtype=np.int64)
    idx_arr = np.asarray(idx, dtype=np.int64)
    return pts[idx_arr], idx_arr


# =========================
# FILTRO DE PONTOS (OBB)
# =========================
def filtrar_pontos_obb(
    pts: np.ndarray,
    bbox_local_obb: Dict,
    matrix_local: np.ndarray,
    margem: float,
    frac_bottom: float = 0.0,
    frac_top: float = 0.0,
    slice_z: Optional[Tuple[float, float]] = None,
    extras: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Filtra pontos usando OBB: transforma pontos para o espaço local do objeto
    (via placement IFC) e faz teste AABB nesse espaço.
    Elimina volumes fantasma nos cantos de objetos diagonais/rotacionados.
    Fallback automático para AABB global se matrix_local ou bbox_local_obb for None.

    Se `slice_z` (zmin_global, zmax_global) for fornecido, aplica também um clip
    GLOBAL em Z antes do OBB — usado para cross-floor (pilar fatiado por andar).

    Se `extras` for um array N×K (mesma ordem de `pts`), recebe os mesmos masks aplicados
    e a função retorna a tupla (pts_filtrados, extras_filtrados) — usado para propagar
    cores RGB do scanner em paralelo aos pontos sem refazer a filtragem.
    """
    # Clip global Z primeiro (cross-floor slicing)
    if slice_z is not None and pts.size > 0:
        z_lo, z_hi = slice_z
        mask_g = (pts[:, 2] >= z_lo - margem) & (pts[:, 2] <= z_hi + margem)
        pts = pts[mask_g]
        if extras is not None:
            extras = extras[mask_g]

    if matrix_local is None or bbox_local_obb is None:
        out_pts = filtrar_pontos_aabb(pts, bbox_local_obb or {}, margem, frac_bottom, frac_top)
        if extras is None:
            return out_pts
        # Para propagar extras no fallback AABB precisaríamos do mask interno; como o
        # caso é raro (sem placement) e barato, refazemos o filtro AABB local aqui:
        if pts.size == 0:
            return out_pts, np.empty((0, extras.shape[1]) if extras.ndim == 2 else (0,))
        b = bbox_local_obb or {}
        if not b:
            return out_pts, extras  # sem bbox: AABB devolve tudo
        m = margem
        mask_a = (
            (pts[:, 0] >= b.get('xmin', -np.inf) - m) & (pts[:, 0] <= b.get('xmax', np.inf) + m) &
            (pts[:, 1] >= b.get('ymin', -np.inf) - m) & (pts[:, 1] <= b.get('ymax', np.inf) + m) &
            (pts[:, 2] >= b.get('zmin', -np.inf) - m) & (pts[:, 2] <= b.get('zmax', np.inf) + m)
        )
        return out_pts, extras[mask_a]

    if pts.size == 0:
        empty = np.empty((0, 3), dtype=float)
        return (empty, np.empty((0, extras.shape[1]) if (extras is not None and extras.ndim == 2) else (0,))) if extras is not None else empty

    # Transforma pontos do espaço global para o espaço local do objeto
    R = matrix_local[:3, :3]
    t = matrix_local[:3, 3]
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        out_pts = filtrar_pontos_aabb(pts, bbox_local_obb, margem, frac_bottom, frac_top)
        return (out_pts, extras) if extras is not None else out_pts

    pts_local = (R_inv @ (pts - t).T).T  # (N, 3) no espaço local

    # Aplica frac_bottom / frac_top no eixo Z local
    zmin = bbox_local_obb['zmin']
    zmax = bbox_local_obb['zmax']
    altura = zmax - zmin
    if altura > 0 and (frac_bottom > 0.0 or frac_top > 0.0):
        frac_bottom = max(0.0, min(0.9, frac_bottom))
        frac_top    = max(0.0, min(0.9, frac_top))
        zmin_c = zmin + frac_bottom * altura
        zmax_c = zmax - frac_top    * altura
        if zmin_c < zmax_c:
            zmin, zmax = zmin_c, zmax_c

    mask = (
        (pts_local[:, 0] >= bbox_local_obb['xmin'] - margem) &
        (pts_local[:, 0] <= bbox_local_obb['xmax'] + margem) &
        (pts_local[:, 1] >= bbox_local_obb['ymin'] - margem) &
        (pts_local[:, 1] <= bbox_local_obb['ymax'] + margem) &
        (pts_local[:, 2] >= zmin - margem) &
        (pts_local[:, 2] <= zmax + margem)
    )
    if extras is None:
        return pts[mask]
    return pts[mask], extras[mask]


# =========================
# ALINHAMENTO V3 — FGR + ICP (pipeline novo, robusto)
# =========================
# O algoritmo `alinhar_nuvem_com_ifc` original (mais abaixo) é "centroide +
# brute-force de rotação Z + scoring por pontos em bboxes". Funciona em sintético
# (mesma convenção de eixo e centro próximos), mas falha com scan real:
#   - não normaliza convenção de eixos (Y-up vs Z-up)
#   - rotação só em Z, não consegue endireitar nuvem deitada
#   - scoring confunde com simetrias do prédio (180° ambíguo)
#
# Este pipeline novo combina 3 estágios bem testados:
#   1. Normalização do eixo vertical (histograma de normais → up vira Z)
#   2. Registro grosso via FGR (Open3D, baseado em features FPFH)
#   3. Refinamento ICP point-to-point
#
# Tudo via Open3D nativo. Sem AI, sem GPU, sem brute-force de orientações.

def _ifc_full_mesh_cloud(
    ifc_path: str,
    max_pts: int = 100_000,
    density_per_m2: float = 80.0,
) -> np.ndarray:
    """Amostra pontos uniformemente nas FACES das malhas IFC.

    IMPORTANTE: pra ALINHAMENTO usamos TODA a geometria disponível no IFC,
    não só TIPOS_INTERESSE. Coverings (forros), railings, furniture etc.
    dão FORMA distintiva ao prédio — sem eles FPFH não tem feature pra casar.
    Análise ML continua usando só TIPOS_INTERESSE; isso é só pra ALIGNMENT REF.

    Aqui usamos `mesh.sample_points_uniformly()` do Open3D que amostra
    densamente nas superfícies, com densidade proporcional à área.
    """
    token = _cache_ifc(ifc_path)
    f = _IFC_CACHE[token]['file']
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    # Tipos que ajudam no alinhamento mas não estão em TIPOS_INTERESSE
    # (cobre quase tudo que tem geometria 3D útil pra forma do prédio)
    TIPOS_ALINHAMENTO_EXTRAS = (
        "IfcCovering",          # forros, revestimentos (chave pro teto)
        "IfcRailing",           # guarda-corpos
        "IfcRamp",              # rampas
        "IfcCurtainWall",       # paredes-cortina (vidro)
        "IfcMember",            # peças estruturais menores
        "IfcPlate",             # placas
        "IfcFurnishingElement", # móveis fixos
        "IfcBuildingElementProxy", # genérico (equipamentos no Vinhedo, etc.)
        "IfcFlowTerminal",      # bocas de ar/luz/água
        "IfcFlowSegment",       # tubulação
        "IfcDistributionElement", # elétrica
        "IfcSpace",             # espaços (volumes vazios mas dão forma)
    )
    TIPOS_USAR = TIPOS_INTERESSE + TIPOS_ALINHAMENTO_EXTRAS

    all_pts = []
    n_meshes_ok = 0
    n_meshes_fallback = 0
    n_skipped_type = 0

    for product in f.by_type("IfcProduct"):
        if not any(product.is_a(t) for t in TIPOS_USAR):
            n_skipped_type += 1
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            verts_arr = np.array(shape.geometry.verts).reshape(-1, 3)
            faces_arr = np.array(shape.geometry.faces).reshape(-1, 3) if hasattr(shape.geometry, 'faces') else None

            if verts_arr.size == 0:
                continue

            if faces_arr is not None and len(faces_arr) > 0:
                # Constrói malha e amostra densamente as faces
                try:
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(verts_arr)
                    mesh.triangles = o3d.utility.Vector3iVector(faces_arr)
                    area = float(mesh.get_surface_area())
                    n_sample = max(20, int(area * density_per_m2))
                    pcd_s = mesh.sample_points_uniformly(number_of_points=n_sample)
                    all_pts.append(np.asarray(pcd_s.points))
                    n_meshes_ok += 1
                    continue
                except Exception:
                    pass

            # Fallback: vértices brutos
            all_pts.append(verts_arr)
            n_meshes_fallback += 1
        except Exception:
            continue

    if not all_pts:
        return np.empty((0, 3))

    pts = np.vstack(all_pts)
    pts = np.unique(pts, axis=0)
    if len(pts) > max_pts:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
    print(f"      [mesh cloud] {n_meshes_ok} meshes amostradas + {n_meshes_fallback} fallback verts "
          f"(ignorados {n_skipped_type} de tipos não-geométricos) → {len(pts):,} pts")
    return pts


def _sample_obb_surface(obj: Dict, target_pts: int = 300) -> np.ndarray:
    """Amostra pontos uniformemente nas 6 faces da bbox de um objeto IFC.

    Usado pra construir uma 'nuvem de referência' do IFC pro FGR. Não precisa
    de mesh exata — bbox é suficiente pra registro grosso.
    """
    bbox = obj['bbox']
    x0, x1 = float(bbox['xmin']), float(bbox['xmax'])
    y0, y1 = float(bbox['ymin']), float(bbox['ymax'])
    z0, z1 = float(bbox['zmin']), float(bbox['zmax'])
    dx, dy, dz = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6), max(z1 - z0, 1e-6)

    A_xy, A_xz, A_yz = dx * dy, dx * dz, dy * dz
    A_sum = A_xy + A_xz + A_yz
    if A_sum <= 0:
        return np.empty((0, 3))

    # Aloca pts por face proporcional à área
    n_xy = max(2, int(target_pts * A_xy / A_sum / 2))
    n_xz = max(2, int(target_pts * A_xz / A_sum / 2))
    n_yz = max(2, int(target_pts * A_yz / A_sum / 2))

    rng = np.random.default_rng(seed=42)
    parts = []

    # Bottom + top
    for z_val in (z0, z1):
        u = rng.uniform(x0, x1, n_xy)
        v = rng.uniform(y0, y1, n_xy)
        parts.append(np.column_stack([u, v, np.full(n_xy, z_val)]))
    # Front + back
    for y_val in (y0, y1):
        u = rng.uniform(x0, x1, n_xz)
        w = rng.uniform(z0, z1, n_xz)
        parts.append(np.column_stack([u, np.full(n_xz, y_val), w]))
    # Left + right
    for x_val in (x0, x1):
        v = rng.uniform(y0, y1, n_yz)
        w = rng.uniform(z0, z1, n_yz)
        parts.append(np.column_stack([np.full(n_yz, x_val), v, w]))

    return np.vstack(parts)


def _normalizar_eixo_vertical(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """Detecta qual eixo é o 'up' da nuvem (X, Y ou Z) e rotaciona pra Z-up.

    Método: estima normais em downsample e identifica o eixo cardinal onde
    as normais mais se concentram (porque lajes+tetos sempre apontam pra
    vertical, independente do número de andares).

    Retorna (pts_rotacionados, R_aplicada, log).
    """
    if len(pts) < 200:
        return pts, np.eye(3), "skip_few_pts"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Downsample agressivo só pra estimar normais rápido
    pcd_down = pcd.voxel_down_sample(voxel_size=0.10)
    if len(pcd_down.points) < 100:
        return pts, np.eye(3), "skip_sparse"
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.30, max_nn=20)
    )
    normals = np.asarray(pcd_down.normals)

    # Score = soma de (normal · eixo)² → eixo com mais normais "perpendiculares"
    # é o eixo vertical (porque lajes+tetos têm normal nesse eixo).
    scores = (normals ** 2).sum(axis=0)
    up_idx = int(np.argmax(scores))
    axis_name = ['X', 'Y', 'Z'][up_idx]
    scores_pct = (scores / scores.sum() * 100).round(1).tolist()

    # Rotação pra trazer up_idx → +Z (preservando handedness/orientação)
    # Atenção: as matrizes anteriores estavam INVERTIDAS, flippavam ao invés de rotacionar.
    R = np.eye(3)
    if up_idx == 0:    # X → +Z (rotação -90° em torno de Y)
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
    elif up_idx == 1:  # Y → +Z (rotação +90° em torno de X)
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)

    if up_idx == 2:
        return pts, R, f"já Z-up (distrib={scores_pct}%)"

    pts_rot = (R @ pts.T).T
    return pts_rot, R, f"{axis_name}-up → rotacionou para Z-up (distrib={scores_pct}%)"


def _alinhar_fgr_icp(
    pts_scan: np.ndarray,
    pts_ifc: np.ndarray,
    voxel_size: float = 0.10,
) -> Tuple[np.ndarray, float, float]:
    """Registro grosso (FGR) + refinamento (ICP) via Open3D.

    Returns: (transformation 4x4, fitness 0-1, rmse).
    """
    if len(pts_scan) == 0 or len(pts_ifc) == 0:
        return np.eye(4), 0.0, float('inf')

    pcd_s = o3d.geometry.PointCloud()
    pcd_s.points = o3d.utility.Vector3dVector(pts_scan)
    pcd_s_d = pcd_s.voxel_down_sample(voxel_size)

    pcd_i = o3d.geometry.PointCloud()
    pcd_i.points = o3d.utility.Vector3dVector(pts_ifc)
    pcd_i_d = pcd_i.voxel_down_sample(voxel_size)

    # Estima normais (FPFH e ICP point-to-plane precisam)
    radius_n = voxel_size * 2.0
    pcd_s_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_n, max_nn=30))
    pcd_i_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_n, max_nn=30))

    # FPFH features
    radius_f = voxel_size * 5.0
    fpfh_s = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_s_d, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_f, max_nn=100)
    )
    fpfh_i = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_i_d, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_f, max_nn=100)
    )

    # FGR — Fast Global Registration (não precisa de chute inicial)
    result_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pcd_s_d, pcd_i_d, fpfh_s, fpfh_i,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=voxel_size * 0.5
        )
    )

    # ICP refinement (point-to-plane converge mais rápido com normais)
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd_s, pcd_i_d,
        max_correspondence_distance=voxel_size * 0.4,
        init=result_fgr.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    return np.asarray(result_icp.transformation), float(result_icp.fitness), float(result_icp.inlier_rmse)


def alinhar_via_fgr_icp(
    pts_scan: np.ndarray,
    objetos_ifc: List[Dict],
    voxel_size: Optional[float] = None,
    ifc_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """Pipeline novo de alinhamento (3 estágios) — substitui o brute-force antigo.

    Estágios:
      1. Normaliza eixo vertical (X/Y-up → Z-up)
      2. FGR multi-escala (Open3D) — rotação grossa baseada em features FPFH
      3. ICP point-to-plane — refinamento fino

    Voxel é adaptativo ao tamanho do prédio (1/50 da menor dimensão IFC) e
    multi-escala (coarse → fine) — features grandes resolvem rotação 180°,
    features pequenas refinam translação.

    Returns: (pts_alinhados, debug_info)
    """
    debug: Dict = {}

    # Estágio 1: normalizar eixo
    pts_n, R_axis, axis_log = _normalizar_eixo_vertical(pts_scan)
    print(f"   📐 Normalização eixo: {axis_log}")
    debug['axis_normalization'] = axis_log

    if not objetos_ifc:
        debug['error'] = 'no_ifc_objects'
        return pts_n, debug

    # Estágio 2: monta nuvem de referência do IFC
    pts_ifc = np.empty((0, 3))
    if ifc_path:
        pts_ifc = _ifc_full_mesh_cloud(ifc_path, max_pts=80_000)
        if pts_ifc.size > 0:
            print(f"   📦 IFC reference cloud: {len(pts_ifc):,} pts (mesh vertices reais)")
    if pts_ifc.size == 0:
        pts_ifc = np.vstack([_sample_obb_surface(o, target_pts=300) for o in objetos_ifc])
        print(f"   📦 IFC reference cloud: {len(pts_ifc):,} pts (fallback: bbox surfaces)")
    if pts_ifc.size == 0:
        debug['error'] = 'empty_ifc_cloud'
        return pts_n, debug

    # Voxel adaptativo: 1/50 da diagonal do IFC
    ifc_dims = pts_ifc.max(axis=0) - pts_ifc.min(axis=0)
    diag = float(np.linalg.norm(ifc_dims))
    if voxel_size is None:
        voxel_size = max(0.05, diag / 80.0)  # ex: prédio 25m → voxel 0.3m
    print(f"   📏 IFC diag={diag:.1f}m | voxel base={voxel_size:.3f}m")

    # Multi-escala: coarse pra resolver rotação 180°, fine pra refinar
    voxels = [voxel_size * 3, voxel_size, voxel_size * 0.4]
    best_T = np.eye(4)
    best_fit = 0.0
    best_rmse = float('inf')

    for v in voxels:
        T, fitness, rmse = _alinhar_fgr_icp(pts_n, pts_ifc, voxel_size=v)
        print(f"      [voxel {v:.3f}m] fitness={fitness:.3f} rmse={rmse:.3f}m")
        # Aceita melhoria SIGNIFICATIVA (não só marginal)
        if fitness > best_fit + 0.02:
            best_T = T
            best_fit = fitness
            best_rmse = rmse

    print(f"   ✅ FGR+ICP (melhor): fitness={best_fit:.3f} | rmse={best_rmse:.3f}m")
    debug.update({
        'fitness': best_fit,
        'rmse_m': best_rmse,
        'transformation': best_T.tolist(),
        'voxel_base': voxel_size,
    })

    # Aplica transformação na nuvem completa
    pts_homog = np.hstack([pts_n, np.ones((len(pts_n), 1))])
    pts_final = (best_T @ pts_homog.T).T[:, :3]

    return pts_final, debug


# =========================
# ALINHAMENTO HÍBRIDO 2.0 (LEGADO — mantido como fallback)
# =========================
def alinhar_nuvem_com_ifc(
    pts: np.ndarray,
    objetos_ifc: List[Dict],
    max_pts_amostra: int = 120_000
) -> Tuple[np.ndarray, Dict]:
    print(f"\n🔍 Alinhando nuvem (v2.1 robust)...")
    if pts.size == 0 or not objetos_ifc:
        return pts, {'R': np.eye(3), 't': np.zeros(3), 'scale': 1.0}
    # =========================
    # 1. BOUNDS IFC
    # =========================
    ifc_min, ifc_max, ifc_center, ifc_extent = _bounds_from_objs(objetos_ifc)
    diag_ifc = np.linalg.norm(ifc_extent)
    # =========================
    # 2. BOUNDS NUVEM
    # =========================
    pts_min = pts.min(axis=0)
    pts_max = pts.max(axis=0)
    pts_center = (pts_min + pts_max) / 2.0
    diag_pts = np.linalg.norm(pts_max - pts_min)
    # =========================
    # 3. AMOSTRA ROBUSTA (sem viés)
    # =========================
    if len(pts) > max_pts_amostra:
        step = len(pts) // max_pts_amostra
        pts_sample = pts[::step]
    else:
        pts_sample = pts
    print(f"   ✂️ Amostra: {len(pts_sample):,} pontos")
    # =========================
    # 4. OBJETOS PARA SCORE
    # =========================
    objetos_teste = objetos_ifc
    if len(objetos_ifc) > 60:
        objetos_teste = random.sample(objetos_ifc, 60)
    # =========================
    # 5. BUSCA
    # =========================
    perms = list(permutations((0, 1, 2), 3))
    signs = list(product([-1, 1], repeat=3))

    # Helper: score de um pts_test dado
    def _score(pts_test):
        s = 0.0
        for obj in objetos_teste:
            bbox = obj['bbox']
            dx = bbox['xmax'] - bbox['xmin']
            dy = bbox['ymax'] - bbox['ymin']
            dz = bbox['zmax'] - bbox['zmin']
            volume = max(dx * dy * dz, 1e-6)
            mm = 0.15
            mask = (
                (pts_test[:, 0] >= bbox['xmin'] - mm) &
                (pts_test[:, 0] <= bbox['xmax'] + mm) &
                (pts_test[:, 1] >= bbox['ymin'] - mm) &
                (pts_test[:, 1] <= bbox['ymax'] + mm) &
                (pts_test[:, 2] >= bbox['zmin'] - mm) &
                (pts_test[:, 2] <= bbox['zmax'] + mm)
            )
            pts_in = pts_test[mask]
            count = len(pts_in)
            if count < 5:
                continue
            densidade = count / volume
            std = np.std(pts_in, axis=0).mean()
            penal_spread = 1.0 / (1.0 + std)
            s += densidade * penal_spread
        return s

    # Testa identidade PRIMEIRO (PLY ja pode estar em coords IFC).
    # Fase inicial (few constructed objects) quebra centroid-matching,
    # entao preferimos identidade se ela ja funcionar razoavelmente.
    score_identity = _score(pts_sample)
    print(f"   🆔 Score identidade: {score_identity:.4f}")
    melhor = {
        'score': score_identity,
        'R': np.eye(3),
        't': np.zeros(3),
        'scale': 1.0
    }
    # Barrier: so trocamos por outra transformacao se ela superar a
    # identidade por > 30% (evita desalinhar PLYs ja OK).
    IDENTITY_BARRIER = 1.30

    for scl in SCALES_TO_TEST:
        ratio = (diag_pts * scl) / (diag_ifc + 1e-6)
        # 🔥 restrição mais forte
        if ratio < 0.6 or ratio > 1.8:
            continue
        print(f"   ⚡ escala {scl} (ratio={ratio:.2f})")
        for perm in perms:
            for sign in signs:
                # transforma sample
                P = pts_sample[:, perm]
                S = (P * np.array(sign)) * scl
                # centro correto
                center_rot = pts_center[list(perm)] * np.array(sign) * scl
                t_centro = ifc_center - center_rot

                # estratégia base Z (chão com chão)
                smin = S.min(axis=0)
                t_base = t_centro.copy()
                t_base[2] = ifc_min[2] - smin[2]

                for t in (t_centro, t_base):
                    pts_test = S + t
                    score_total = _score(pts_test)
                    # So troca identidade se ganhar com folga
                    threshold = score_identity * IDENTITY_BARRIER if melhor['score'] == score_identity else melhor['score']
                    if score_total > threshold:
                        melhor = {
                            'score': score_total,
                            'R': _perm_sign_to_R(perm, sign, scl),
                            't': t,
                            'scale': scl
                        }
    # =========================
    # 6. VALIDAÇÃO FINAL
    # =========================
    if melhor['score'] < 1e-3:
        print("⚠️ Alinhamento falhou (score muito baixo)")
        return pts, {'R': np.eye(3), 't': np.zeros(3), 'scale': 1.0}
    print(f"   ✅ Melhor score: {melhor['score']:.4f} | escala={melhor['scale']}")
    # aplica transformação
    pts_alinhado = (pts @ melhor['R'].T) + melhor['t']
    return pts_alinhado, {
        'R': melhor['R'],
        't': melhor['t'],
        'scale': melhor['scale']
    }


# =========================
# VALIDAÇÃO DE ORIENTAÇÃO (ANTI-LEAKING)
# =========================
def corrigir_orientacao_por_pico_vertical(
    pts: np.ndarray,
    objetos_ifc: List[Dict],
    bins: int = 80,
    margem: float = 0.30
) -> Tuple[np.ndarray, bool, Dict]:
    """
    Usa o pico de densidade em Z para decidir se a nuvem está invertida.
    """
    debug_info: Dict = {}

    if pts.size == 0 or not objetos_ifc:
        return pts, False, {"motivo": "sem_dados"}

    # Bounds em Z do IFC
    z_min_ifc = min(o["bbox"]["zmin"] for o in objetos_ifc)
    z_max_ifc = max(o["bbox"]["zmax"] for o in objetos_ifc)
    altura_ifc = z_max_ifc - z_min_ifc

    if altura_ifc <= 0:
        return pts, False, {"motivo": "altura_zero"}

    # Histograma de Z
    zs = pts[:, 2]
    if zs.size < 10:
        return pts, False, {"motivo": "poucos_pontos"}

    counts, edges = np.histogram(zs, bins=bins)
    if counts.max() == 0:
        return pts, False, {"motivo": "histograma_vazio"}

    idx_mode = int(np.argmax(counts))
    z_mode = float((edges[idx_mode] + edges[idx_mode + 1]) / 2.0)

    dist_floor = abs(z_mode - z_min_ifc)
    dist_roof = abs(z_mode - z_max_ifc)

    debug_info = {
        "z_mode": z_mode,
        "z_floor": z_min_ifc,
        "z_roof": z_max_ifc,
        "dist_floor": dist_floor,
        "dist_roof": dist_roof
    }

    print(f"\n🔍 Checagem de orientação vertical:")
    print(f"   Pico Z: {z_mode:.2f} | Dist chão: {dist_floor:.2f} | Dist teto: {dist_roof:.2f}")

    # Se pico mais perto do teto, aplica flip
    if dist_roof + margem < dist_floor:
        print("   ❌ Pico perto do TETO → aplicando flip...")
        z_centro = (z_min_ifc + z_max_ifc) / 2.0
        pts_flipped = pts.copy()
        pts_flipped[:, 2] = 2.0 * z_centro - pts[:, 2]
        return pts_flipped, True, debug_info

    print("   ✅ Orientação OK")
    return pts, False, debug_info


# =========================
# NORMALIZAÇÃO DE COORDENADAS
# =========================
def normalizar_coordenadas(pts: np.ndarray,
                           objetos_ifc: List[Dict]) -> Tuple[np.ndarray,
                                                             Dict,
                                                             List[Dict]]:
    """Normaliza coordenadas para visualização (centro em 0,0 e Z>=0)."""

    ifc_min, ifc_max, ifc_center, _ = _bounds_from_objs(objetos_ifc)

    if len(pts) > 0:
        pts_min = pts.min(axis=0)
        pts_max = pts.max(axis=0)
        pts_center = (pts_min + pts_max) / 2.0

        # Centro X/Y combinado
        global_center_xy = np.array([
            (ifc_center[0] + pts_center[0]) / 2.0,
            (ifc_center[1] + pts_center[1]) / 2.0,
            0.0
        ])
        z_min = min(ifc_min[2], pts_min[2])
    else:
        global_center_xy = np.array([ifc_center[0], ifc_center[1], 0.0])
        z_min = ifc_min[2]

    global_center = np.array([global_center_xy[0], global_center_xy[1], z_min])

    # Aplica translação
    pts_normalized = pts - global_center if len(pts) > 0 else pts

    # Normaliza objetos
    objetos_normalized = []
    for obj in objetos_ifc:
        obj_norm = obj.copy()
        bbox = obj['bbox']
        obj_norm['bbox'] = {
            'xmin': bbox['xmin'] - global_center[0],
            'xmax': bbox['xmax'] - global_center[0],
            'ymin': bbox['ymin'] - global_center[1],
            'ymax': bbox['ymax'] - global_center[1],
            'zmin': bbox['zmin'] - global_center[2],
            'zmax': bbox['zmax'] - global_center[2]
        }
        # Atualiza translação do placement local para OBB acompanhar a normalização
        if obj.get('matrix_local') is not None:
            ml = obj['matrix_local'].copy()
            ml[:3, 3] -= global_center
            obj_norm['matrix_local'] = ml

        # Desloca também os campos cross-floor em Z global (IFC → normalizado)
        if obj.get('slice_z') is not None:
            sz = obj['slice_z']
            obj_norm['slice_z'] = (sz[0] - global_center[2], sz[1] - global_center[2])
        if obj.get('z_total') is not None:
            zt = obj['z_total']
            obj_norm['z_total'] = [zt[0] - global_center[2], zt[1] - global_center[2]]
        if obj.get('fatia_atual') is not None:
            fa = dict(obj['fatia_atual'])
            zr = fa.get('z_range')
            if zr is not None:
                fa['z_range'] = [zr[0] - global_center[2], zr[1] - global_center[2]]
            obj_norm['fatia_atual'] = fa

        objetos_normalized.append(obj_norm)

    transform_info = {'translation': global_center.tolist()}

    return pts_normalized, transform_info, objetos_normalized


# =========================
# CÁLCULO DE DENSIDADE
# =========================
def calcular_densidade_detalhada(
        pts: np.ndarray,
        bbox: Dict,
        num_fatias: int = NUM_FATIAS_ALTURA) -> Dict:
    """Calcula densidade com análise vertical por fatias."""
    if len(pts) == 0:
        return {
            'num_pontos': 0,
            'densidade_global': 0.0,
            'cobertura_vertical': 0.0,
            'densidade_por_altura': []
        }

    dx = bbox['xmax'] - bbox['xmin']
    dy = bbox['ymax'] - bbox['ymin']
    dz = bbox['zmax'] - bbox['zmin']
    volume = dx * dy * dz
    area_horizontal = dx * dy

    densidade_global = len(pts) / volume if volume > 0 else 0.0

    altura_fatia = dz / num_fatias if dz > 0 else 1.0
    fatias_com_pontos = 0
    densidade_por_altura = []

    for i in range(num_fatias):
        z_min_fatia = bbox['zmin'] + i * altura_fatia
        z_max_fatia = z_min_fatia + altura_fatia

        pts_fatia = pts[(pts[:, 2] >= z_min_fatia) & (pts[:, 2] < z_max_fatia)]

        volume_fatia = area_horizontal * altura_fatia
        densidade_fatia = len(pts_fatia) / volume_fatia if volume_fatia > 0 else 0.0

        densidade_por_altura.append({
            'fatia': i + 1,
            'z_min': round(z_min_fatia, 2),
            'z_max': round(z_max_fatia, 2),
            'pontos': len(pts_fatia),
            'densidade': round(densidade_fatia, 1)
        })

        if densidade_fatia > DENSIDADE_MIN:
            fatias_com_pontos += 1

    cobertura_vertical = fatias_com_pontos / num_fatias

    return {
        'num_pontos': len(pts),
        'densidade_global': densidade_global,
        'cobertura_vertical': cobertura_vertical,
        'densidade_por_altura': densidade_por_altura
    }


def calcular_densidade_horizontal(
        pts: np.ndarray,
        bbox: Dict,
        nx: int = 10,
        ny: int = 10) -> Dict:
    """Calcula densidade horizontal para lajes/pisos/tetos."""
    if pts.size == 0:
        return {
            'num_pontos': 0,
            'densidade_global': 0.0,
            'cobertura_vertical': 0.0,
            'densidade_por_altura': []
        }

    dx = bbox['xmax'] - bbox['xmin']
    dy = bbox['ymax'] - bbox['ymin']
    area = dx * dy

    if area <= 0:
        return {
            'num_pontos': len(pts),
            'densidade_global': 0.0,
            'cobertura_vertical': 0.0,
            'densidade_por_altura': []
        }

    densidade_global = len(pts) / area

    xs = pts[:, 0]
    ys = pts[:, 1]

    x_edges = np.linspace(bbox['xmin'], bbox['xmax'], nx + 1)
    y_edges = np.linspace(bbox['ymin'], bbox['ymax'], ny + 1)

    H, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])

    cells_ocupadas = np.count_nonzero(H)
    total_cells = H.size
    cobertura_horizontal = cells_ocupadas / total_cells if total_cells > 0 else 0.0

    return {
        'num_pontos': len(pts),
        'densidade_global': densidade_global,
        'cobertura_vertical': cobertura_horizontal,
        'densidade_por_altura': []
    }


# =========================
# CÁLCULO DE DIMENSÕES
# =========================
def calcular_dimensoes_reais(pts: np.ndarray, bbox_ifc: Dict) -> Dict:
    """Calcula dimensões executadas vs planejadas."""
    dim_plan = {
        'x': bbox_ifc['xmax'] - bbox_ifc['xmin'],
        'y': bbox_ifc['ymax'] - bbox_ifc['ymin'],
        'z': bbox_ifc['zmax'] - bbox_ifc['zmin']
    }

    if len(pts) < 10:
        return {
            'executado': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'planejado': dim_plan,
            'progresso': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'delta': {'x': -dim_plan['x'], 'y': -dim_plan['y'], 'z': -dim_plan['z']}
        }

    try:
        min_p = np.percentile(pts, 1, axis=0)
        max_p = np.percentile(pts, 99, axis=0)
        dim_exec = {
            'x': max_p[0] - min_p[0],
            'y': max_p[1] - min_p[1],
            'z': max_p[2] - min_p[2]
        }
    except Exception:
        dim_exec = {
            'x': pts[:, 0].max() - pts[:, 0].min(),
            'y': pts[:, 1].max() - pts[:, 1].min(),
            'z': pts[:, 2].max() - pts[:, 2].min()
        }

    progresso = {
        'x': min(dim_exec['x'] / dim_plan['x'], 1.2) if dim_plan['x'] > 0 else 0,
        'y': min(dim_exec['y'] / dim_plan['y'], 1.2) if dim_plan['y'] > 0 else 0,
        'z': min(dim_exec['z'] / dim_plan['z'], 1.2) if dim_plan['z'] > 0 else 0
    }

    return {
        'executado': {k: round(v, 2) for k, v in dim_exec.items()},
        'planejado': {k: round(v, 2) for k, v in dim_plan.items()},
        'progresso': {k: round(v * 100, 1) for k, v in progresso.items()},
        'delta': {
            'x': round(dim_exec['x'] - dim_plan['x'], 2),
            'y': round(dim_exec['y'] - dim_plan['y'], 2),
            'z': round(dim_exec['z'] - dim_plan['z'], 2)
        }
    }


# =========================
# CLASSIFICAÇÃO (ANTI-LEAKING)
# =========================
def classificar_status(
    densidade_info: Dict,
    eh_conexao: bool = False,
    tipo: str = None,
    dimensoes: Dict = None
) -> Dict:
    """
    Classificação com trava anti-leaking para paredes.
    Verifica altura executada antes de classificar.
    """
    cobertura = densidade_info['cobertura_vertical']
    densidade = densidade_info['densidade_global']

    threshold_min = COBERTURA_CONEXAO_MIN if eh_conexao else COBERTURA_INICIADO

    # TRAVA ANTI-LEAKING: Se altura executada < 25%, considera AUSENTE
    if tipo == "IfcWall" and dimensoes is not None:
        try:
            prog_altura = float(dimensoes['progresso']['z']) / 100.0
        except Exception:
            prog_altura = 0.0

        if prog_altura < 0.25:
            return {
                'code': "AUSENTE",
                'emoji': "❌",
                'texto': "Ausente (altura insuf.)",
                'cor': "#f44336"
            }

    # Classificação normal
    if cobertura >= COBERTURA_COMPLETO and densidade > DENSIDADE_MIN:
        return {
            'code': "COMPLETO",
            'emoji': "✅",
            'texto': "Completo",
            'cor': "#4caf50"
        }

    if cobertura >= COBERTURA_PARCIAL and densidade > DENSIDADE_MIN * 0.5:
        texto = "Parcial (Conexão)" if eh_conexao else "Parcial"
        return {'code': "PARCIAL", 'emoji': "⚠️", 'texto': texto, 'cor': "#ff9800"}

    if cobertura >= threshold_min:
        texto = "Iniciado (Conexão)" if eh_conexao else "Iniciado"
        return {'code': "INICIADO", 'emoji': "🔶", 'texto': texto, 'cor': "#2196f3"}

    return {'code': "AUSENTE", 'emoji': "❌", 'texto': "Ausente", 'cor': "#f44336"}


# =========================
# PÓS-PROCESSAMENTO: FANTASMA (v3 - SUBTRAÇÃO DE PONTOS)
# =========================
def detectar_objetos_fantasma(
    resultados: List[Dict],
    objetos_ifc: List[Dict],
    pts: np.ndarray
) -> List[Dict]:
    """
    Detecta paredes fantasma via subtração de pontos compartilhados.
    Se a maioria dos pontos de uma parede também pertence a outra parede
    com MAIS pontos, ela é fantasma (leaking de T-junction).
    """
    if len(resultados) == 0 or len(pts) == 0:
        return resultados

    ifc_map = {obj['guid']: obj for obj in objetos_ifc}
    reclassificados = 0

    # Pré-calcula pontos por parede (só paredes existentes)
    wall_data = {}
    for resultado in resultados:
        if resultado.get('tipo') != 'IfcWall':
            continue
        if resultado['status']['code'] not in ('COMPLETO', 'PARCIAL', 'INICIADO'):
            continue
        obj = ifc_map.get(resultado['guid'])
        if obj is None:
            continue
        pts_obj = filtrar_pontos_obb(
            pts,
            obj.get('bbox_local_obb') or obj['bbox'],
            obj.get('matrix_local'),
            MARGEM_PADRAO,
        )
        wall_data[resultado['guid']] = {
            'resultado': resultado,
            'obj': obj,
            'num_pontos': len(pts_obj),
            'pts': pts_obj
        }

    print("\n=== DETECCAO FANTASMA v3 (SUBTRACAO DE PONTOS) ===")

    for guid, data in wall_data.items():
        resultado = data['resultado']
        obj = data['obj']
        bbox = obj['bbox']
        nome = resultado.get('nome', guid[:8])
        status_code = resultado['status']['code']
        meus_pontos = data['pts']

        if len(meus_pontos) == 0:
            continue

        # Para cada ponto desta parede, verifica se está TAMBÉM
        # dentro de outra parede que tem MAIS pontos
        pontos_compartilhados = np.zeros(len(meus_pontos), dtype=bool)
        vizinho_principal = None
        max_pontos_vizinho = 0

        for other_guid, other_data in wall_data.items():
            if other_guid == guid:
                continue
            # Só compara se o outro tem mais pontos
            if other_data['num_pontos'] <= data['num_pontos']:
                continue

            other_obj = other_data['obj']
            m = MARGEM_PADRAO

            # OBB do vizinho (espaço local) — evita falsos positivos em paredes
            # diagonais, onde a AABB world engloba o canto vazio.
            other_ml      = other_obj.get('matrix_local')
            other_bbox_l  = other_obj.get('bbox_local_obb')
            if other_ml is not None and other_bbox_l is not None:
                try:
                    R_inv = np.linalg.inv(other_ml[:3, :3])
                    t     = other_ml[:3, 3]
                    pts_local = (R_inv @ (meus_pontos - t).T).T
                    mask_in_other = (
                        (pts_local[:, 0] >= other_bbox_l['xmin'] - m) &
                        (pts_local[:, 0] <= other_bbox_l['xmax'] + m) &
                        (pts_local[:, 1] >= other_bbox_l['ymin'] - m) &
                        (pts_local[:, 1] <= other_bbox_l['ymax'] + m) &
                        (pts_local[:, 2] >= other_bbox_l['zmin'] - m) &
                        (pts_local[:, 2] <= other_bbox_l['zmax'] + m)
                    )
                except np.linalg.LinAlgError:
                    other_bbox = other_obj['bbox']
                    mask_in_other = (
                        (meus_pontos[:, 0] >= other_bbox['xmin'] - m) &
                        (meus_pontos[:, 0] <= other_bbox['xmax'] + m) &
                        (meus_pontos[:, 1] >= other_bbox['ymin'] - m) &
                        (meus_pontos[:, 1] <= other_bbox['ymax'] + m) &
                        (meus_pontos[:, 2] >= other_bbox['zmin'] - m) &
                        (meus_pontos[:, 2] <= other_bbox['zmax'] + m)
                    )
            else:
                # Fallback AABB world quando objeto não tem placement
                other_bbox = other_obj['bbox']
                mask_in_other = (
                    (meus_pontos[:, 0] >= other_bbox['xmin'] - m) &
                    (meus_pontos[:, 0] <= other_bbox['xmax'] + m) &
                    (meus_pontos[:, 1] >= other_bbox['ymin'] - m) &
                    (meus_pontos[:, 1] <= other_bbox['ymax'] + m) &
                    (meus_pontos[:, 2] >= other_bbox['zmin'] - m) &
                    (meus_pontos[:, 2] <= other_bbox['zmax'] + m)
                )

            pontos_compartilhados |= mask_in_other

            n_shared = int(np.count_nonzero(mask_in_other))
            if n_shared > max_pontos_vizinho:
                max_pontos_vizinho = n_shared
                vizinho_principal = other_data['resultado'].get('nome', other_guid[:8])

        total = len(meus_pontos)
        compartilhados = int(np.count_nonzero(pontos_compartilhados))
        frac_compartilhada = compartilhados / total

        # Se >60% dos pontos são compartilhados com parede maior → fantasma
        if frac_compartilhada > 0.60:
            resultado['status'] = {
                'code': "AUSENTE",
                'emoji': "❌",
                'texto': "Ausente (fantasma T-junction)",
                'cor': "#f44336"
            }
            resultado['phantom'] = True
            reclassificados += 1
            print(f"  \"{nome}\" ({status_code}): {compartilhados}/{total} pontos "
                  f"({frac_compartilhada:.0%}) compartilhados com \"{vizinho_principal}\"")
            print(f"    -> RECLASSIFICADO -> AUSENTE (fantasma)")
        else:
            print(f"  \"{nome}\" ({status_code}): {compartilhados}/{total} "
                  f"({frac_compartilhada:.0%}) compartilhados -> OK")

    print(f"  Total reclassificados: {reclassificados}")
    print("=" * 60)

    return resultados


# =========================
# ANÁLISE PRINCIPAL
# =========================
def analisar_pavimento_completo(
    objetos_ifc: List[Dict],
    ply_path: str,
    output_dir: Path
) -> Tuple[List[Dict], Dict, np.ndarray, List[Dict]]:
    """
    Análise completa com todas as proteções anti-leaking.
    """
    print("\n" + "=" * 70)
    print("ANÁLISE DE PROGRESSO (VERSÃO HÍBRIDA 2.0)")
    print("=" * 70)

    # 1. Carrega nuvem (valida vazio/corrompido — open3d não lança por conta própria)
    # BBox helper: ignora colors (modo BBox não usa RGB do scanner)
    print("\n📦 Carregando nuvem de pontos...")
    pts, _, err_ply = _ler_ply_validado(ply_path)
    if err_ply:
        raise ValueError(err_ply[0])
    print(f"   ✓ {len(pts):,} pontos únicos")

    # 2. Detecta conexões
    objetos_ifc, num_conexoes = detectar_paredes_conexao(objetos_ifc)
    objetos_ifc = marcar_conexoes_piso_teto(objetos_ifc)

    # 3. Alinhamento robusto (v2.1)
    pts, _ = alinhar_nuvem_com_ifc(pts, objetos_ifc)

    # 4. Correção de orientação
    pts, flipped, _ = corrigir_orientacao_por_pico_vertical(pts, objetos_ifc)

    # 5. Normalização
    pts, transform_info, objetos_ifc = normalizar_coordenadas(pts, objetos_ifc)

    # 6. Análise de cada objeto
    resultados = []

    print("\n" + "=" * 120)
    print(f"{'Nome':<25} {'Tipo':<12} {'Pontos':>8} {'Cobert.':>8} {'Altura':>15} {'Status':<20}")
    print("=" * 120)

    n_obb = sum(
        1 for o in objetos_ifc
        if o.get('matrix_local') is not None and o.get('bbox_local_obb') is not None
    )
    print(f"\n🔷 OBB ativo: {n_obb}/{len(objetos_ifc)} objetos (restante usa AABB fallback)")

    for obj in objetos_ifc:
        # Margem baseada no tipo
        margem = MARGEM_PORTA_JAN if obj['tipo'] in (
            'IfcDoor', 'IfcWindow') else MARGEM_PADRAO

        # Cortes dinâmicos anti-leaking
        frac_bottom = 0.0
        frac_top = 0.0

        if obj['tipo'] == "IfcWall":
            if obj.get('conecta_piso') and not obj.get('conecta_teto'):
                frac_bottom = 0.10
            elif obj.get('conecta_piso') and obj.get('conecta_teto'):
                frac_bottom = 0.10
                frac_top = 0.10
            elif obj.get('conecta_teto') and not obj.get('conecta_piso'):
                frac_top = 0.10

        # OBB: usa bbox no espaço local (cópia pra não modificar o original)
        bbox_obb = dict(obj['bbox_local_obb']) if obj.get('bbox_local_obb') else None

        # Encolhe bbox para paredes de conexão (aplicado no espaço LOCAL — mais preciso)
        if obj['tipo'] == "IfcWall" and obj.get('eh_conexao') and bbox_obb:
            dx_l = bbox_obb['xmax'] - bbox_obb['xmin']
            dy_l = bbox_obb['ymax'] - bbox_obb['ymin']
            factor = 0.10
            if dx_l >= dy_l:
                shrink = dx_l * factor
                bbox_obb['xmin'] += shrink
                bbox_obb['xmax'] -= shrink
            else:
                shrink = dy_l * factor
                bbox_obb['ymin'] += shrink
                bbox_obb['ymax'] -= shrink

        # Filtra pontos com OBB (fallback automático para AABB se sem placement)
        pts_obj = filtrar_pontos_obb(
            pts,
            bbox_obb,
            obj.get('matrix_local'),
            margem,
            frac_bottom=frac_bottom,
            frac_top=frac_top,
        )

        # Densidade (horizontal para lajes, vertical para o resto)
        if obj['tipo'] in ("IfcSlab", "IfcRoof"):
            densidade_info = calcular_densidade_horizontal(pts_obj, obj['bbox'])
        else:
            densidade_info = calcular_densidade_detalhada(pts_obj, obj['bbox'])

        # Dimensões
        dimensoes = calcular_dimensoes_reais(pts_obj, obj['bbox'])

        # Classificação
        eh_conexao = obj.get('eh_conexao', False)
        status = classificar_status(
            densidade_info,
            eh_conexao=eh_conexao,
            tipo=obj['tipo'],
            dimensoes=dimensoes
        )

        # Nomes de arquivos
        nome_safe = secure_filename(obj['nome'])[:30]
        ply_filename = f"{nome_safe}_{obj['guid'][:8]}.ply"
        json_filename = f"{nome_safe}_{obj['guid'][:8]}.json"

        # BBox para Three.js
        bbox_threejs = converter_ifc_para_threejs(obj['bbox'])

        # Exporta PLY e JSON
        if len(pts_obj) > 0:
            try:
                pcd_export = o3d.geometry.PointCloud()
                pcd_export.points = o3d.utility.Vector3dVector(pts_obj)

                cor = {
                    'COMPLETO': [0.2, 0.8, 0.2],
                    'PARCIAL': [1.0, 0.6, 0.0],
                    'INICIADO': [0.2, 0.6, 1.0]
                }.get(status['code'], [0.8, 0.2, 0.2])

                pcd_export.paint_uniform_color(cor)
                o3d.io.write_point_cloud(str(output_dir / ply_filename), pcd_export)

                pts_threejs = converter_pontos_ifc_para_threejs(pts_obj)
                json_data = {
                    'positions': pts_threejs.flatten().tolist(),
                    'color': cor,
                    'count': len(pts_obj)
                }
                with open(output_dir / json_filename, 'w') as f:
                    json.dump(json_data, f)
            except Exception as e:
                print(f"   ❌ Erro exportando {obj['nome']}: {e}")

        # Log
        altura_str = f"{dimensoes['executado']['z']:.1f}/{dimensoes['planejado']['z']:.1f}m"
        print(
            f"{obj['nome']:<25} {obj['tipo']:<12} {densidade_info['num_pontos']:>8,} "
            f"{densidade_info['cobertura_vertical']:>7.0%} {altura_str:>15} "
            f"{status['emoji']} {status['texto']:<15}")

        resultados.append({
            'guid': obj['guid'],
            'nome': obj['nome'],
            'tipo': obj['tipo'],
            'pavimento': obj['pavimento'],
            'volume_ifc': round(obj['volume_ifc'], 2),
            'pontos': densidade_info['num_pontos'],
            'densidade': round(densidade_info['densidade_global'], 1),
            'cobertura': round(densidade_info['cobertura_vertical'] * 100, 1),
            'status': status,
            'eh_conexao': eh_conexao,
            'dimensoes': dimensoes,
            'ply_file': ply_filename if len(pts_obj) > 0 else None,
            'json_file': json_filename if len(pts_obj) > 0 else None,
            'bbox_normalized': bbox_threejs
        })

    print("=" * 120)

    # Estatísticas
    stats = Counter([r['status']['code'] for r in resultados])
    total = len(resultados)

    estatisticas = {
        'total': total,
        'completos': stats.get('COMPLETO', 0),
        'parciais': stats.get('PARCIAL', 0),
        'iniciados': stats.get('INICIADO', 0),
        'ausentes': stats.get('AUSENTE', 0),
        'conexoes': num_conexoes,
        'progresso_geral': round(
            (stats.get('COMPLETO', 0) +
             stats.get('PARCIAL', 0) * 0.5 +
             stats.get('INICIADO', 0) * 0.1) /
            total * 100, 1) if total > 0 else 0
    }

    print(f"\n📊 RESUMO:")
    print(f"   ✅ Completos: {estatisticas['completos']}")
    print(f"   ⚠️ Parciais:  {estatisticas['parciais']}")
    print(f"   🔶 Iniciados: {estatisticas['iniciados']}")
    print(f"   ❌ Ausentes:  {estatisticas['ausentes']}")
    print(f"   🔗 Conexões:  {estatisticas['conexoes']}")
    print(f"   📈 Progresso: {estatisticas['progresso_geral']:.1f}%")

    # === HOOK RANDLA-NET: salva cena rotulada para treino ===
    if _RANDLANET_DISPONIVEL:
        try:
            _salvar_cena_dataset(pts, objetos_ifc)
        except Exception as _e:
            print(f"  ⚠️ [Dataset] Erro ao salvar cena: {_e}")

    return resultados, estatisticas, pts, objetos_ifc


# =========================
# ROTAS API
# =========================
@app.route('/')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'online',
        'service': 'BIM Analysis API',
        'version': '2.0.0 (Hybrid Fixed)',
        'endpoints': [
            'GET  /                         - Health check',
            'GET  /outputs/<file>          - Download arquivos',
            'POST /api/listar_pavimentos   - Lista pavimentos do IFC',
            'POST /api/analisar_pavimento  - Análise completa com anti-leaking',
            'POST /api/generate_report     - Gera relatório via DeepSeek',
            'POST /api/chat                - Chat livre com DeepSeek'
        ]
    })


@app.route('/outputs/<path:filename>')
def download_output(filename):
    return send_from_directory(str(OUTPUT_FOLDER), filename)


@app.route('/api/listar_pavimentos', methods=['POST'])
def listar_pavimentos():
    """Lista pavimentos disponíveis no IFC."""
    try:
        f = request.files.get('ifc_file') or request.files.get('file')
        if not f:
            return jsonify({'error': 'Arquivo IFC não enviado'}), 400
        if not _valid_upload(f, ('.ifc',)):
            return jsonify({'error': 'Arquivo precisa ter extensão .ifc'}), 400

        filename = f"{uuid.uuid4()}_{secure_filename(f.filename)}"
        path = UPLOAD_FOLDER / filename
        f.save(str(path))

        token = _cache_ifc(str(path))
        pavs = extrair_pavimentos(str(path))

        return jsonify({
            'pavimentos': pavs,
            'total': len(pavs),
            'ifc_token': token,  # front pode reusar no /api/analisar_ai pra pular reupload
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/analisar_pavimento', methods=['POST'])
def analisar_pavimento():
    """Analisa um pavimento com todas as proteções anti-leaking."""
    try:
        ply_file = request.files.get('ply_file')
        pav_alvo = request.form.get('pavimento')
        use_ai = request.form.get('use_ai', 'false').lower() == 'true'

        if not ply_file:
            return jsonify({'error': 'Arquivo PLY não enviado'}), 400
        if not _valid_upload(ply_file, ('.ply',)):
            return jsonify({'error': 'Arquivo PLY precisa ter extensão .ply'}), 400
        if not pav_alvo:
            return jsonify({'error': 'Pavimento não especificado'}), 400

        session_id = str(uuid.uuid4())
        ifc_path_str, ifc_token, err = _resolve_ifc_from_request(request, session_id)
        if err:
            return jsonify({'error': err[0]}), err[1]
        ifc_path = Path(ifc_path_str)

        ply_path = UPLOAD_FOLDER / f"{session_id}_{secure_filename(ply_file.filename)}"
        ply_file.save(str(ply_path))

        # Extrai objetos do pavimento (inclui estrutura cruzando — consistente com /api/analisar_ai)
        objetos = extrair_objetos_por_pavimento(str(ifc_path), pav_alvo, incluir_estrutura_cruzando=True)
        if not objetos:
            return jsonify(
                {'error': f'Nenhum objeto encontrado no pavimento "{pav_alvo}"'}), 400

        # Prepara output
        output_session = OUTPUT_FOLDER / session_id
        output_session.mkdir(parents=True, exist_ok=True)

        # Análise completa (bbox-based) — ValueError indica PLY inválido (400), não 500
        try:
            resultados, estatisticas, pts_alinhados, objetos_processados = analisar_pavimento_completo(
                objetos,
                str(ply_path),
                output_session
            )
        except ValueError as e_ply:
            return jsonify({'error': str(e_ply)}), 400

        # ── MODO AI (opcional) ─────────────────────────────────
        modo_usado = "bbox"
        if use_ai and _RANDLANET_INFERENCE:
            print("\n🤖 Modo AI ativado — rodando RandLA-Net...")
            try:
                resultados_ai = _classificar_ai(
                    pts_alinhados, objetos_processados, output_session
                )
                if resultados_ai is not None:
                    resultados = resultados_ai
                    modo_usado = "ai"
                    print("✅ Análise AI concluída")
                else:
                    print("⚠️  AI retornou None — usando bbox")
            except Exception as e_ai:
                print(f"❌ Erro AI: {e_ai} — usando bbox")
        elif use_ai and not _RANDLANET_INFERENCE:
            print("⚠️  use_ai=true mas modelo não disponível — usando bbox")

        # ── PÓS-PROCESSAMENTO ──────────────────────────────────
        if modo_usado == "bbox":
            # Phantom detection só faz sentido no modo bbox
            resultados = detectar_objetos_fantasma(resultados, objetos_processados, pts_alinhados)

        # Recalcular estatísticas
        stats = Counter([r['status']['code'] for r in resultados])
        total = len(resultados)
        estatisticas = {
            'total': total,
            'completos': stats.get('COMPLETO', 0),
            'parciais': stats.get('PARCIAL', 0),
            'iniciados': stats.get('INICIADO', 0),
            'ausentes': stats.get('AUSENTE', 0),
            'conexoes': estatisticas.get('conexoes', 0),
            'progresso_geral': round(
                (stats.get('COMPLETO', 0) +
                 stats.get('PARCIAL', 0) * 0.5 +
                 stats.get('INICIADO', 0) * 0.1) /
                total * 100, 1) if total > 0 else 0
        }

        # Ajusta caminhos dos arquivos
        for r in resultados:
            if r.get('json_file'):
                r['json_file'] = f"{session_id}/{r['json_file']}"
            if r.get('ply_file'):
                r['ply_file'] = f"{session_id}/{r['ply_file']}"

        return jsonify({
            'pavimento': pav_alvo,
            'session_id': session_id,
            'modo': modo_usado,
            'estatisticas': estatisticas,
            'resultados': resultados
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# (RandLA-Net AI removido — substituido pelo Random Forest ML no final do arquivo)


@app.route('/api/analisar_instancias', methods=['POST'])
def analisar_instancias():
    """Análise de instâncias individuais via RandLA-Net Instance + DBSCAN."""
    try:
        if not _INSTANCE_INFERENCE:
            return jsonify({'error': 'RandLA-Net Instance não disponível (checkpoint ausente)'}), 503

        ply_file = request.files.get('ply_file')
        pav_alvo = request.form.get('pavimento')

        if not ply_file:
            return jsonify({'error': 'Arquivo PLY não enviado'}), 400
        if not _valid_upload(ply_file, ('.ply',)):
            return jsonify({'error': 'Arquivo PLY precisa ter extensão .ply'}), 400
        if not pav_alvo:
            return jsonify({'error': 'Pavimento não especificado'}), 400

        session_id = str(uuid.uuid4())
        ifc_path_str, _, err = _resolve_ifc_from_request(request, session_id)
        if err:
            return jsonify({'error': err[0]}), err[1]
        ifc_path = Path(ifc_path_str)

        ply_path = UPLOAD_FOLDER / f"{session_id}_{secure_filename(ply_file.filename)}"
        ply_file.save(str(ply_path))

        objetos = extrair_objetos_por_pavimento(str(ifc_path), pav_alvo, incluir_estrutura_cruzando=True)
        if not objetos:
            return jsonify({'error': f'Nenhum objeto no pavimento "{pav_alvo}"'}), 400

        output_session = OUTPUT_FOLDER / session_id
        output_session.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("ANALISE INSTANCIAS (RandLA-Net Instance + DBSCAN)")
        print("=" * 70)

        # Modo Instancias: ignora colors (RandLA-Net atual nao consome RGB)
        pts, _, err_ply = _ler_ply_validado(str(ply_path))
        if err_ply:
            return jsonify({'error': err_ply[0]}), err_ply[1]
        print(f"   Pontos: {len(pts):,}")

        objetos, _ = detectar_paredes_conexao(objetos)
        objetos = marcar_conexoes_piso_teto(objetos)
        pts, _ = alinhar_nuvem_com_ifc(pts, objetos)
        pts, _, _ = corrigir_orientacao_por_pico_vertical(pts, objetos)
        pts, _, objetos = normalizar_coordenadas(pts, objetos)

        from pathlib import Path as _Path
        resultados = _segmentar_instancias(pts, objetos, output_session, checkpoint=_Path(_CHECKPOINT_INST))

        if resultados is None:
            return jsonify({'error': 'Segmentação de instâncias falhou'}), 500

        total = len(resultados)
        detectados = sum(1 for r in resultados if r['status']['code'] == 'DETECTADO')
        estatisticas = {
            'total': total,
            'detectados': detectados,
            'completos': 0, 'parciais': 0, 'iniciados': 0, 'ausentes': 0,
            'conexoes': 0,
            'progresso_geral': 100.0 if total > 0 else 0
        }

        def _to_native(obj):
            if isinstance(obj, dict):
                return {k: _to_native(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_native(v) for v in obj]
            if hasattr(obj, 'item'):
                return obj.item()
            return obj

        for r in resultados:
            if r.get('json_file'):
                r['json_file'] = f"{session_id}/{r['json_file']}"
            if r.get('ply_file'):
                r['ply_file'] = f"{session_id}/{r['ply_file']}"

        return jsonify(_to_native({
            'pavimento': pav_alvo,
            'session_id': session_id,
            'modo': 'instancias',
            'estatisticas': estatisticas,
            'resultados': resultados
        }))

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """Proxy para API DeepSeek - gera relatório executivo."""
    try:
        if not DEEPSEEK_API_KEY:
            return jsonify({'error': 'DEEPSEEK_API_KEY não configurada no servidor'}), 500

        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'Prompt não fornecido'}), 400

        print(f"📝 Gerando relatório com DeepSeek (prompt: {len(prompt)} chars)")

        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
            },
            json={
                'model': 'deepseek-chat',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'Você é um especialista em planejamento e controle de obras. Gere relatórios técnicos concisos e objetivos em português.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.7,
                'max_tokens': 2000,
                'stream': False
            },
            timeout=30
        )

        if not response.ok:
            error_msg = f'DeepSeek API Error: {response.status_code}'
            print(f"❌ {error_msg}")
            print(f"Response: {response.text[:200]}")
            return jsonify({
                'error': error_msg,
                'details': response.text
            }), 500

        result = response.json()
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

        print(f"✅ Relatório gerado com sucesso ({len(content)} chars)")

        return jsonify({
            'content': content,
            'usage': result.get('usage', {})
        })

    except requests.exceptions.Timeout:
        print("❌ Timeout na requisição DeepSeek")
        return jsonify({'error': 'Timeout ao gerar relatório. Tente novamente.'}), 504

    except Exception as e:
        print(f"❌ Erro no proxy DeepSeek: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint de chat para conversa com DeepSeek."""
    try:
        if not DEEPSEEK_API_KEY:
            return jsonify({'error': 'DEEPSEEK_API_KEY não configurada no servidor'}), 500

        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'Prompt não fornecido'}), 400

        print(f"💬 Chat com DeepSeek (prompt: {len(prompt)} chars)")

        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
            },
            json={
                'model': 'deepseek-chat',
                'messages': [
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.7,
                'max_tokens': 300,
                'stream': False
            },
            timeout=20
        )

        if not response.ok:
            error_msg = f'DeepSeek API Error: {response.status_code}'
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 500

        result = response.json()
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

        print(f"✅ Resposta gerada ({len(content)} chars)")

        return jsonify({'content': content})

    except requests.exceptions.Timeout:
        return jsonify({'error': 'Timeout. Tente novamente.'}), 504

    except Exception as e:
        print(f"❌ Erro no chat: {e}")
        return jsonify({'error': str(e)}), 500


# =========================
# ANÁLISE ML (Random Forest)
# =========================

_ML_MODEL  = None
_ML_SCALER = None
_ML_MODEL_PATH = Path(__file__).parent / "ml" / "models" / "random_forest.pkl"
_ML_LOCK = threading.Lock()  # serializa o pickle.load na primeira requisição concorrente

def _carregar_modelo_ml():
    global _ML_MODEL, _ML_SCALER
    if _ML_MODEL is not None:
        return True
    with _ML_LOCK:
        if _ML_MODEL is not None:  # outro thread carregou enquanto esperávamos
            return True
        try:
            import pickle
            with open(_ML_MODEL_PATH, 'rb') as f:
                saved = pickle.load(f)
            _ML_MODEL  = saved['model']
            _ML_SCALER = saved['scaler']
            print(f"ML: Random Forest carregado")
            return True
        except Exception as e:
            print(f"ML: modelo nao disponivel — {e}")
            return False

_ML_DISPONIVEL = _carregar_modelo_ml()

_ML_TIPO_MAP = {
    "IfcWall": 0, "IfcSlab": 1, "IfcColumn": 2, "IfcBeam": 3,
    "IfcStair": 4, "IfcRoof": 5, "IfcDoor": 6, "IfcWindow": 7,
    # Novos tipos mapeados ao equivalente geométrico que o RF já conhece:
    "IfcCovering": 1,  # como IfcSlab (forros = placa horizontal)
    "IfcPlate":    1,  # como IfcSlab (placa)
    "IfcMember":   3,  # como IfcBeam (membro linear)
    "IfcRailing":  0,  # como IfcWall (extrusão vertical)
}
_ML_EIXO_MAP  = {"z_up": 0, "horiz_longest": 1, "length_axis": 2}
_ML_EIXO_TIPO = {
    "IfcWall": "z_up", "IfcColumn": "z_up", "IfcStair": "z_up",
    "IfcDoor": "z_up", "IfcWindow": "z_up",
    "IfcSlab": "horiz_longest", "IfcRoof": "horiz_longest",
    "IfcBeam": "length_axis",
    # Novos tipos, eixo da feature geométrica equivalente:
    "IfcCovering": "horiz_longest",  # forro deita no plano horizontal
    "IfcPlate":    "horiz_longest",
    "IfcMember":   "length_axis",     # membro estrutural longo
    "IfcRailing":  "z_up",            # guarda-corpo vertical
}
_ML_STATUS = {
    0: {'code': 'COMPLETO', 'texto': 'Executado',              'cor': '#22c55e'},
    1: {'code': 'PARCIAL',  'texto': 'Parcialmente Executado', 'cor': '#f59e0b'},
    2: {'code': 'AUSENTE',  'texto': 'Nao Executado',          'cor': '#ef4444'},
}


def _extrair_features_ml(
    pts_cena: np.ndarray,
    obj: Dict,
    colors_cena: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extrai 11 features ML + devolve pts/colors filtrados pelo OBB do objeto.

    Quando `colors_cena` é fornecido (PLY com RGB), retorna `colors_obj` com a mesma
    máscara aplicada — permite reusar no loop de geração dos JSONs sem refiltrar.
    """
    bbox = obj['bbox']
    area = float(obj.get('area_ifc') or obj.get('area') or 1.0)
    vol  = float(obj.get('volume_ifc') or obj.get('volume') or 1.0)
    tipo = _ML_TIPO_MAP.get(obj.get('tipo', ''), 0)
    eixo = _ML_EIXO_MAP.get(_ML_EIXO_TIPO.get(obj.get('tipo', ''), 'z_up'), 0)

    bw = max(bbox['xmax'] - bbox['xmin'], 1e-6)
    bd = max(bbox['ymax'] - bbox['ymin'], 1e-6)
    bh = max(bbox['zmax'] - bbox['zmin'], 1e-6)
    expected_pts = max(area * 130, 1)

    m = 0.05
    colors_obj: Optional[np.ndarray] = None
    if pts_cena is not None and len(pts_cena) > 0:
        # OBB com fallback automático para AABB se o objeto não tiver placement
        if colors_cena is not None:
            pts_obj, colors_obj = filtrar_pontos_obb(
                pts_cena,
                obj.get('bbox_local_obb') or bbox,
                obj.get('matrix_local'),
                m,
                slice_z=obj.get('slice_z'),
                extras=colors_cena,
            )
        else:
            pts_obj = filtrar_pontos_obb(
                pts_cena,
                obj.get('bbox_local_obb') or bbox,
                obj.get('matrix_local'),
                m,
                slice_z=obj.get('slice_z'),
            )
    else:
        pts_obj = np.zeros((0, 3))
        if colors_cena is not None:
            colors_obj = np.zeros((0, 3))

    n = len(pts_obj)
    if n > 0:
        zp = pts_obj[:, 2]
        zlo, zhi, zmean = float(zp.min()), float(zp.max()), float(zp.mean())
        height_fill     = (zhi - zlo) / bh
        z_bottom_norm   = (zlo  - bbox['zmin']) / bh
        z_top_norm      = (zhi  - bbox['zmin']) / bh
        centroid_z_norm = (zmean - bbox['zmin']) / bh
        xy_spread_norm  = float(pts_obj[:, :2].std()) / max(bw, bd)
        density_obs     = n / max(vol, 1e-6)
        completeness_r  = n / expected_pts
        is_empty        = 0.0
    else:
        height_fill = z_bottom_norm = z_top_norm = centroid_z_norm = 0.0
        xy_spread_norm = density_obs = completeness_r = 0.0
        is_empty = 1.0

    density_area = n / max(area, 1e-6)

    feats = np.array([
        completeness_r, is_empty, height_fill,
        z_bottom_norm, z_top_norm, centroid_z_norm, xy_spread_norm, density_area,
        float(tipo), float(eixo), bh,
    ], dtype=np.float32)
    return feats, pts_obj, colors_obj


@app.route('/api/analisar_ai', methods=['POST'])
def analisar_ai_ml():
    """Analisa pavimento usando Random Forest (substitui RandLA-Net no front)."""
    try:
        if not _ML_DISPONIVEL:
            return jsonify({'error': 'Modelo ML nao disponivel. Rode ml/train.py primeiro.'}), 503

        ply_file = request.files.get('ply_file')
        pav_alvo = request.form.get('pavimento')

        if not ply_file:
            return jsonify({'error': 'PLY nao enviado'}), 400
        if not _valid_upload(ply_file, ('.ply',)):
            return jsonify({'error': 'Arquivo PLY precisa ter extensão .ply'}), 400
        if not pav_alvo:
            return jsonify({'error': 'Pavimento nao especificado'}), 400

        session_id = str(uuid.uuid4())
        ifc_path_str, _, err = _resolve_ifc_from_request(request, session_id)
        if err:
            return jsonify({'error': err[0]}), err[1]
        ifc_path = Path(ifc_path_str)

        ply_path = UPLOAD_FOLDER / f"{session_id}_{secure_filename(ply_file.filename)}"
        ply_file.save(str(ply_path))

        # Extrai objetos do IFC:
        #   - 1ª passada: por nome de storey (inclui OSSO normalizado)
        #   - 2ª passada: Z-range do storey + overlap geométrico pra pegar pilares/shafts
        #                 que o projetista atribuiu a outro storey mas atravessam o pavimento
        objetos = extrair_objetos_por_pavimento(
            str(ifc_path), pav_alvo, incluir_estrutura_cruzando=True
        )
        if not objetos:
            return jsonify({'error': f'Nenhum objeto no pavimento "{pav_alvo}"'}), 400

        pts, ply_colors, err_ply = _ler_ply_validado(str(ply_path))
        if err_ply:
            return jsonify({'error': err_ply[0]}), err_ply[1]
        if ply_colors is not None:
            print(f"🎨 PLY traz cores RGB do scanner — {len(ply_colors):,} pontos coloridos")
        objetos, _ = detectar_paredes_conexao(objetos)
        objetos     = marcar_conexoes_piso_teto(objetos)

        # Alinha usando APENAS objetos do storey-match (não os da 2ª passada),
        # pois objetos verticais longos (pilares, shafts) distorcem o scoring
        # e podem escolher uma rotação espúria.
        objetos_para_alinhar = [o for o in objetos if o.get('incluido_por') == 'storey']
        if not objetos_para_alinhar:
            objetos_para_alinhar = objetos  # fallback
        print(f"   🎯 Alinhamento usará {len(objetos_para_alinhar)} objetos do storey "
              f"(ignora {len(objetos) - len(objetos_para_alinhar)} cross-floor)")
        # V3 (alinhar_via_fgr_icp) temporariamente DESABILITADO — está com bug
        # em dado real e o fallback legado lida bem com nuvens já pré-alinhadas
        # (como as exportadas do CloudCompare após alinhamento manual).
        # Reativar quando V3 estiver corrigido.
        # pts_v3, dbg_v3 = alinhar_via_fgr_icp(pts, objetos_para_alinhar, ifc_path=str(ifc_path))
        # if dbg_v3.get('fitness', 0.0) >= 0.10:
        #     pts = pts_v3
        #     print(f"   🎯 Usando alinhamento V3 (FGR+ICP) — fitness={dbg_v3['fitness']:.3f}")
        # else:
        #     print(f"   ⚠️  V3 com fitness baixa ({dbg_v3.get('fitness', 0):.3f}) — caindo pro algoritmo legado")
        #     pts, _ = alinhar_nuvem_com_ifc(pts, objetos_para_alinhar)
        pts, _ = alinhar_nuvem_com_ifc(pts, objetos_para_alinhar)
        print(f"   ℹ️  V3 desabilitado nesse teste — usando só legado")
        pts, _, objetos = normalizar_coordenadas(pts, objetos)

        import sys as _s
        print(f"ML: {len(pts):,} pts | {len(objetos)} objetos", flush=True)

        # ── DIAGNÓSTICO OBB ──────────────────────────────────────────────
        n_obb   = sum(1 for o in objetos if o.get('matrix_local') is not None
                                        and o.get('bbox_local_obb') is not None)
        n_aabb  = len(objetos) - n_obb
        print(f"\n🔷 OBB coverage: {n_obb}/{len(objetos)} usam OBB  |  {n_aabb} fallback AABB")

        # Ângulo de rotação Z do prédio (via 1º objeto com matrix_local válida)
        for o in objetos:
            ml = o.get('matrix_local')
            if ml is not None:
                import math as _math
                # Coluna X da rotação → ângulo em relação ao eixo global X
                angle_deg = _math.degrees(_math.atan2(float(ml[1, 0]), float(ml[0, 0])))
                print(f"   📐 Rotação Z do IFC (1º obj com placement): {angle_deg:.1f}°")
                break

        # Amostra dos primeiros 5 objetos: OBB ou AABB?
        print("   Primeiros 5 objetos:")
        for o in objetos[:5]:
            modo = "OBB ✅" if (o.get('matrix_local') is not None
                               and o.get('bbox_local_obb') is not None) else "AABB ⚠️"
            print(f"     {o['nome'][:35]:<35}  [{o['tipo']:<20}]  {modo}")
        print()
        # ─────────────────────────────────────────────────────────────────

        # KDTree construído UMA vez: pré-filtro espacial pra cada objeto
        # receber só os ~centenas de pontos dentro da sua esfera envolvente,
        # em vez de varrer os ~13M a cada objeto.
        import time as _time
        _t0 = _time.time()
        tree = cKDTree(pts) if len(pts) > 0 else None
        print(f"🌲 KDTree construída em {(_time.time()-_t0):.2f}s "
              f"({len(pts):,} pontos indexados)")

        _t0 = _time.time()
        feats_list = []
        pts_obj_cache: List[np.ndarray] = []                # pts filtrados p/ reusar no loop de JSON
        colors_obj_cache: List[Optional[np.ndarray]] = []   # cores RGB paralelas (None se PLY sem RGB)
        for obj in objetos:
            pts_cand, idx_cand = candidatos_obj(pts, tree, obj['bbox'], margem=0.10)
            # Sem custo extra de KDTree: `idx_cand` é o resultado de query_ball_point
            # já feito; só fatiamos o array global de cores com os mesmos índices.
            if ply_colors is not None and idx_cand is not None and idx_cand.size > 0:
                colors_cand = ply_colors[idx_cand]
            elif ply_colors is not None and idx_cand is None:
                # tree=None (PLY pequeno demais p/ KDTree) — passa o array inteiro
                colors_cand = ply_colors
            else:
                colors_cand = None
            feats, pts_obj, colors_obj = _extrair_features_ml(pts_cand, obj, colors_cena=colors_cand)
            feats_list.append(feats)
            pts_obj_cache.append(pts_obj)
            colors_obj_cache.append(colors_obj)
        feats = np.stack(feats_list)
        print(f"⚡ Features extraídas em {(_time.time()-_t0):.2f}s "
              f"para {len(objetos)} objetos")

        # RF foi treinado com dados SEM scaling (scaler eh so pro MLP)
        preds   = _ML_MODEL.predict(feats)
        probas  = _ML_MODEL.predict_proba(feats)

        # Cria pasta de output para os JSONs de pontos
        output_dir = OUTPUT_FOLDER / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Evidência do cache reuse: quantos objetos vão escrever JSON de pontos
        _n_cache_hit = sum(1 for p in pts_obj_cache if len(p) > 0)
        print(f"♻️  Cache pts_obj reusado: {_n_cache_hit}/{len(pts_obj_cache)} objetos "
              f"(sem refiltrar via KDTree/OBB)")

        _t0_json = _time.time()
        resultados = []
        for i, (obj, pred, proba) in enumerate(zip(objetos, preds, probas)):
            status = _ML_STATUS[int(pred)]
            bbox_threejs = converter_ifc_para_threejs(obj['bbox'])
            bbox = obj['bbox']

            # Reusa pts já filtrados na extração de features (evita candidatos_obj
            # + filtrar_pontos_obb duplicados — mesmos bbox/margem/slice_z)
            pts_obj = pts_obj_cache[i]
            colors_obj = colors_obj_cache[i]  # None se PLY sem RGB

            json_filename = None
            if len(pts_obj) > 0:
                # SUBSAMPLE pra viewer: scanners profissionais geram milhões de pts/objeto,
                # Three.js trava acima de uns 100k pts. Análise ML já rodou com todos os
                # pontos (features acima), então subsample aqui afeta SÓ a visualização.
                MAX_PTS_VIEWER = 50_000
                if len(pts_obj) > MAX_PTS_VIEWER:
                    rng = np.random.default_rng(seed=42)  # reproduzível
                    idx_sub = rng.choice(len(pts_obj), MAX_PTS_VIEWER, replace=False)
                    pts_obj_view = pts_obj[idx_sub]
                    colors_obj_view = colors_obj[idx_sub] if colors_obj is not None else None
                    print(f"   🔻 {obj.get('nome', '?')[:35]}: subsample {len(pts_obj):,} → {MAX_PTS_VIEWER:,} pra viewer")
                else:
                    pts_obj_view = pts_obj
                    colors_obj_view = colors_obj

                # Cor por status (fallback para quando o front escolhe modo "status")
                cor = {
                    'COMPLETO': [0.2, 0.8, 0.2],
                    'PARCIAL':  [1.0, 0.6, 0.0],
                    'AUSENTE':  [0.8, 0.2, 0.2],
                }.get(status['code'], [0.5, 0.5, 0.5])
                nome_safe = secure_filename(obj.get('nome', 'obj'))[:30]
                json_filename = f"{nome_safe}_{obj.get('guid', '')[:8]}.json"
                pts_threejs = converter_pontos_ifc_para_threejs(pts_obj_view)
                json_data = {
                    'positions': pts_threejs.flatten().tolist(),
                    'color':     cor,            # cor única por status (verde/laranja/vermelho)
                    'count':     len(pts_obj_view),
                    'count_total': len(pts_obj),  # quantos foram analisados de fato (pre-subsample)
                }
                # RGB do scanner desabilitado por enquanto — reduz peso do JSON em ~30%
                # e simplifica visualização (cor uniforme por status é mais legível
                # que mistura de status + tinta real). Reativar quando viewer escalar.
                with open(output_dir / json_filename, 'w') as f:
                    json.dump(json_data, f)
                json_filename = f"{session_id}/{json_filename}"

            # OBB corners (8 vertices, Three.js coords) — para desenhar caixa orientada no viewer
            obb_corners = calcular_obb_corners_threejs(obj)

            item_out = {
                'guid':         obj.get('guid', ''),
                'nome':         obj.get('nome', obj.get('tipo', '')),
                'tipo':         obj['tipo'],
                'status':       status,
                'confianca_ml': round(float(proba[int(pred)]), 3),
                'bbox':         obj['bbox'],
                'bbox_normalized': bbox_threejs,
                'obb_corners':  obb_corners,      # ← 8 vértices OBB ou None
                'json_file':    json_filename,
                'n_pts':        int(feats[i][0]),
            }

            # Metadata cross-floor (apenas para UI/tabela — NADA de 3js extra)
            if obj.get('cross_floor'):
                item_out['cross_floor']         = True
                item_out['altura_total']        = round(float(obj.get('altura_total', 0.0)), 3)
                z_tot = obj.get('z_total') or [0.0, 0.0]
                item_out['z_total']             = [round(float(z_tot[0]), 3), round(float(z_tot[1]), 3)]
                item_out['andares_atravessa']   = list(obj.get('andares_atravessa') or [])
                item_out['storey_original_ifc'] = obj.get('storey_original_ifc', '')
                fa = obj.get('fatia_atual') or {}
                if fa:
                    item_out['fatia_atual'] = {
                        'pavimento': fa.get('pavimento', ''),
                        'z_range':   [round(float(fa.get('z_range', [0, 0])[0]), 3),
                                      round(float(fa.get('z_range', [0, 0])[1]), 3)],
                        'altura':    round(float(fa.get('altura', 0.0)), 3),
                    }

            resultados.append(item_out)

        print(f"📝 Loop de JSON+output em {(_time.time()-_t0_json):.2f}s "
              f"({len(resultados)} objetos)")

        # libera caches por objeto (pode somar centenas de MB num IFC grande)
        del pts_obj_cache
        del colors_obj_cache

        stats = Counter(r['status']['code'] for r in resultados)
        total = len(resultados)
        estatisticas = {
            'total':         total,
            'completos':     stats.get('COMPLETO', 0),
            'parciais':      stats.get('PARCIAL',  0),
            'iniciados':     0,
            'ausentes':      stats.get('AUSENTE',  0),
            'progresso_geral': round(
                (stats.get('COMPLETO', 0) + stats.get('PARCIAL', 0) * 0.5) / total * 100, 1
            ) if total > 0 else 0,
        }

        print(f"ML: C={stats.get('COMPLETO',0)} P={stats.get('PARCIAL',0)} A={stats.get('AUSENTE',0)}", flush=True)

        # === Nuvem global pra visualização (debug + diagnóstico de alinhamento) ===
        # Sem isso, o viewer só mostra pontos dentro de OBBs — pontos "órfãos"
        # (fora de qualquer objeto IFC) ficam invisíveis. Escrever esta nuvem
        # subsampleada permite ver a FORMA completa do scan e diagnosticar
        # alinhamento visualmente.
        MAX_PTS_GLOBAL = 200_000
        if len(pts) > MAX_PTS_GLOBAL:
            rng_g = np.random.default_rng(seed=42)
            idx_g = rng_g.choice(len(pts), MAX_PTS_GLOBAL, replace=False)
            pts_global = pts[idx_g]
        else:
            pts_global = pts
        pts_global_threejs = converter_pontos_ifc_para_threejs(pts_global)
        global_json = {
            'positions': pts_global_threejs.flatten().tolist(),
            'count':     len(pts_global),
            'count_total': len(pts),
            'color':     [0.45, 0.5, 0.55],  # cinza-azulado neutro
        }
        with open(output_dir / "_global.json", 'w') as f:
            json.dump(global_json, f)
        print(f"🌐 Nuvem global salva: {len(pts_global):,}/{len(pts):,} pts → _global.json")

        return jsonify({
            'pavimento':    pav_alvo,
            'session_id':   session_id,
            'modo':         'ai',
            'estatisticas': estatisticas,
            'resultados':   resultados,
            'global_cloud': f"{session_id}/_global.json",
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ================================================================
# /api/analisar_ai_v2 — pipeline Sonata + OBB do IFC (sem DBSCAN)
# ================================================================
# REESCRITO 2026-05-18: pipeline antigo (Sonata+DBSCAN+Hungarian+RF) substituido
# por Sonata classifica nuvem inteira -> filtra pontos por OBB do IFC ->
# verdict por classe dominante dentro da OBB.
# Codigo antigo permanece em pipeline_v2/orchestrator.py mas nao e chamado.
@app.route('/api/analisar_ai_v2', methods=['POST'])
def analisar_ai_v2():
    """Pipeline Sonata + OBB do IFC.

    Etapas:
      1. Le IFC -> extrai OBB de cada elemento (TIPOS_INTERESSE)
      2. Le PLY -> alinha com IFC (resolve Y-up vs Z-up)
      3. Sonata classifica nuvem alinhada (ScanNet 20 classes por ponto)
      4. Pra cada OBB IFC: cropa pts dentro, conta classes, decide verdict:
           - poucos pts        -> AUSENTE
           - classe dominante coerente -> COMPLETO
           - classe diverge    -> PARCIAL

    Sem DBSCAN, sem Hungarian, sem RF. Tudo via subprocess no venv_sonata.
    """
    try:
        from pipeline_v2 import bbox_runner, bbox_to_v2_schema

        ply_file = request.files.get('ply_file')
        pav_alvo = request.form.get('pavimento') or '__TODOS__'
        # Flag opcional pra ativar alinhamento semantico por classe Sonata
        sem_align = (request.form.get('semantic_align') or '').lower() in ('1', 'true', 'yes', 'on')

        if not ply_file:
            return jsonify({'error': 'PLY nao enviado'}), 400
        if not _valid_upload(ply_file, ('.ply',)):
            return jsonify({'error': 'Arquivo PLY precisa ter extensao .ply'}), 400

        session_id = str(uuid.uuid4())
        ifc_path_str, _, err = _resolve_ifc_from_request(request, session_id)
        if err:
            return jsonify({'error': err[0]}), err[1]

        ply_path = UPLOAD_FOLDER / f"{session_id}_{secure_filename(ply_file.filename)}"
        ply_file.save(str(ply_path))

        print(f"v2: PLY={ply_path.name} IFC={Path(ifc_path_str).name} pav={pav_alvo}")

        output_dir = OUTPUT_FOLDER / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Roda bbox_features.py em subprocess (sonata venv)
        result_raw = bbox_runner.run_bbox_features(
            ply_path=str(ply_path),
            ifc_path=ifc_path_str,
            pavimento=pav_alvo,
            output_dir=str(output_dir),
            timeout=600.0,
            verbose=True,
            semantic_align=sem_align,
        )

        # Adapta pro schema do front v2
        adapted = bbox_to_v2_schema.adaptar(result_raw)

        return jsonify({
            'pavimento':    pav_alvo,
            'session_id':   session_id,
            'modo':         'sonata_bbox_v1',
            'estatisticas': adapted['estatisticas'],
            'resultados':   adapted['resultados'],
            'adicoes':      adapted['adicoes'],
            'global_cloud': adapted['meta'].get('global_cloud'),
            'meta':         adapted['meta'],
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': f'pipeline sonata_bbox falhou: {e}'}), 500


# =========================
# MAIN
# =========================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8081))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    print("=" * 70)
    print("🏗️ BIM ANALYSIS API - VERSÃO OBB 2.1 (ORIENTED BOUNDING BOX)")
    print("=" * 70)
    print("\n✅ PROTEÇÕES ANTI-LEAKING ATIVAS:")
    print("   • Alinhamento: Centro Real + Scoring Amostral (Híbrido)")
    print("   • Detecção de paredes de conexão (T)")
    print("   • Conexões piso/teto com cortes dinâmicos")
    print("   • Validação de orientação vertical")
    print("   • Encolhimento de bbox para conexões (espaço local)")
    print("   • 🔷 OBB: membership test no espaço local do objeto (elimina fantasmas em cantos diagonais)")

    if not DEEPSEEK_API_KEY:
        print("\n⚠️  AVISO: DEEPSEEK_API_KEY não encontrada nas variáveis de ambiente.")
        print("   As funções de Chat e Relatório não funcionarão.")

    print(f"\n🚀 Servidor: http://localhost:{port}")
    print(f"📁 Uploads:  {UPLOAD_FOLDER}")
    print(f"📁 Outputs:  {OUTPUT_FOLDER}")
    print("=" * 70)

    app.run(host='0.0.0.0', port=port, debug=debug)
