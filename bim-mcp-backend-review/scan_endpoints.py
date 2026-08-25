# -*- coding: utf-8 -*-
"""
Endpoints do SCAN -> BIM interativo (visualizador de thresholds).

Ideia (Rafael, jul-2026): o cliente sobe a nuvem, o servidor calcula UMA VEZ a
parte cara (grade de ocupacao voxel + histograma de z) e cacheia em memoria;
cada ajuste de slider so re-roda as operacoes de imagem (fatiar -> binarizar ->
contorno -> segmentos), que sao sub-segundo. Quando o preview estiver bom,
"Gerar IFC" roda a cadeia completa (experiments/cloud2bim) em background com
os thresholds escolhidos.

Registrado no app_obb via `register_scan(app, UPLOAD_FOLDER, OUTPUT_FOLDER)`.
"""
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
from flask import request, jsonify
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
C2B_DIR = BASE_DIR / 'experiments' / 'cloud2bim'
sys.path.insert(0, str(C2B_DIR))
sys.path.insert(0, str(C2B_DIR / 'cloud2bim_patched'))

HYBRID_TILED_SCRIPT = C2B_DIR / 'run_tiled_cloud2bim_fast.py'
HYBRID_WALL_ML_SCRIPT = C2B_DIR / 'run_wall_candidate_classifier_real.py'
HYBRID_OPENING_WEIGHTS = (
    BASE_DIR / 'artifacts' / 'cloud2bim_yoloworld_m_training'
    / 'wall_tokens_m_1280_v1' / 'weights' / 'best.pt'
)
HYBRID_WALL_WEIGHTS = (
    BASE_DIR / 'artifacts' / 'cloud2bim_wall_candidate_training_v2' / 'best.pt'
)
_LOCAL_ML_PYTHON = (
    BASE_DIR / '.codex_tmp' / 'yolo_world_zero_shot'
    / 'venv' / 'Scripts' / 'python.exe'
)

_SESSOES = {}
_JOBS = {}
_LOCK = threading.Lock()
_DETECTOR_LOCK = threading.Lock()   # identify_walls usa env vars (processo-global)
Z_STEP = 0.15          # mesmo passo do identify_slabs (autodiag comparavel)

# --- descarte automatico (o scan e' "usa e joga fora") ---------------------
# Nada do fluxo pode sobreviver ao uso: a nuvem enviada e a area de trabalho do
# job somam ~2,3 GB por scan grande. So o IFC final (em output_folder) fica,
# porque e' o entregavel. Sem isto o disco enche em dias de uso.
SESSAO_TTL_S = 30 * 60          # sessao inativa por 30 min -> descartada
JOB_TTL_S = 60 * 60             # status de job entregue expira em 1 h
_GC_INTERVALO_S = 5 * 60        # varredura a cada 5 min


def _descartar_sessao(sid, sessao):
    """Some com a nuvem enviada e libera a grade voxel da RAM."""
    removidos = 0
    for key in ('path', 'full_cloud_path'):
        try:
            p = Path(sessao.get(key, ''))
            if p.is_file():
                removidos += p.stat().st_size
                p.unlink()
        except OSError as e:
            print(f'[gc] sessao {sid}: nao consegui apagar {key} ({e})')
    if removidos:
        print(f'[gc] sessao {sid}: nuvem descartada ({removidos / 1e6:.0f} MB)')
    sessao.clear()   # solta occ/amostra pro coletor do Python


def _gc_passada(agora=None):
    """Descarta sessoes inativas e status de job velhos. Devolve o que limpou."""
    import time
    agora = agora if agora is not None else time.monotonic()
    with _LOCK:
        expiradas = [sid for sid, s in _SESSOES.items()
                     if agora - s.get('ultimo_acesso', agora) > SESSAO_TTL_S]
        for sid in expiradas:
            _descartar_sessao(sid, _SESSOES.pop(sid))
        jobs_velhos = [jid for jid, j in _JOBS.items()
                       if j.get('status') in ('pronto', 'erro')
                       and agora - j.get('fim', agora) > JOB_TTL_S]
        for jid in jobs_velhos:
            _JOBS.pop(jid, None)
    return len(expiradas), len(jobs_velhos)


def _gc_loop():
    import time
    while True:
        time.sleep(_GC_INTERVALO_S)
        try:
            _gc_passada()
        except Exception:
            traceback.print_exc()   # o coletor nunca pode derrubar o server


def _limpar_cache_scan_de_restart(upload_folder: Path):
    """Remove only expired Scan-to-BIM files orphaned by an old process."""
    import time
    agora = time.time()
    removidos = 0
    bytes_removidos = 0
    bruto_re = re.compile(
        r'^[0-9a-f]{12}_.+\.(?:e57|ply|xyz|las|laz|pts|csv)$', re.IGNORECASE)
    cache_re = re.compile(
        r'^[0-9a-f]{12}_(?:session(?:\.tmp)?\.npz|cloud\.npy)$', re.IGNORECASE)
    try:
        candidates = list(upload_folder.iterdir())
    except OSError:
        return 0, 0
    for path in candidates:
        if not path.is_file() or not (bruto_re.fullmatch(path.name)
                                      or cache_re.fullmatch(path.name)):
            continue
        try:
            if agora - path.stat().st_mtime <= SESSAO_TTL_S:
                continue
            size = path.stat().st_size
            path.unlink()
            removidos += 1
            bytes_removidos += size
        except OSError as exc:
            print(f'[gc] nao consegui remover cache scan orfao {path.name}: {exc}')
    if removidos:
        print(f'[gc] restart: {removidos} caches scan expirados removidos '
              f'({bytes_removidos / 1e6:.0f} MB)')
    return removidos, bytes_removidos


def _tocar(sessao):
    """Marca uso — adia o descarte enquanto o cliente esta' calibrando."""
    import time
    if sessao is not None:
        sessao['ultimo_acesso'] = time.monotonic()
    return sessao


def _classificar_multifatia(occ, cell, zbin, xmin, ymin, zmin, zlo, zhi,
                            eixos, fatias):
    """Classifica cada eixo em trechos parede/porta/janela/oclusao pela
    ASSINATURA VERTICAL de 3 fatias (ideia do Rafael). O eixo vem do
    identify_walls (motor bom de geometria); aqui so' se decide O QUE cada
    trecho e', consultando a grade voxel em 3 alturas.

    fatias = {'baixa':[flo,fhi], 'media':[...], 'alta':[...]} em fracao do
    pe-direito [zlo, zhi]. Tabela de decisao:
      baixa sim, media NAO           -> janela (peitoril embaixo, vao no meio)
      alta sim, media+baixa NAO      -> porta  (vao ate o chao, verga acima)
      as 3 NAO                        -> oclusao (o laser nao viu; candidata a parede)
      resto (solido na media)         -> parede
    """
    H = max(zhi - zlo, 1e-3)

    def mask(par):
        z0, z1 = zlo + par[0] * H, zlo + par[1] * H
        iz0 = max(0, int((z0 - zmin) / zbin))
        iz1 = min(occ.shape[0], max(iz0 + 1, int(np.ceil((z1 - zmin) / zbin))))
        m = (occ[iz0:iz1].sum(0) >= 1).astype(np.uint8)
        # corredor: tolera o eixo nao cair exatamente no pixel da parede
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    Db, Dm, Da = mask(fatias['baixa']), mask(fatias['media']), mask(fatias['alta'])
    NY, NX = Da.shape
    passo = max(cell, 0.10)
    out = []
    for eixo_idx, e in enumerate(eixos):
        x1, y1, x2, y2 = float(e[0]), float(e[1]), float(e[2]), float(e[3])
        L = float(np.hypot(x2 - x1, y2 - y1))
        n = max(int(L / passo), 2)
        cls = []
        for k in range(n + 1):
            t = k / n
            ix = int((x1 + t * (x2 - x1) - xmin) / cell)
            iy = int((y1 + t * (y2 - y1) - ymin) / cell)
            if not (0 <= ix < NX and 0 <= iy < NY):
                cls.append('parede'); continue
            lo, mi, hi = Db[iy, ix] > 0, Dm[iy, ix] > 0, Da[iy, ix] > 0
            if lo and not mi:            c = 'janela'
            elif hi and not mi and not lo: c = 'porta'
            elif not hi and not mi and not lo: c = 'oclusao'
            else:                        c = 'parede'
            cls.append(c)
        # suaviza: run isolado de 1 amostra (~10cm) vira o vizinho
        for i in range(1, len(cls) - 1):
            if cls[i - 1] == cls[i + 1] != cls[i]:
                cls[i] = cls[i - 1]
        # runs -> segmentos [x1,y1,x2,y2,classe,eixo_idx]. O indice ancora
        # portas/janelas na parede exata quando o preview entra no editor 3D.
        i = 0
        while i < len(cls):
            j = i
            while j + 1 < len(cls) and cls[j + 1] == cls[i]:
                j += 1
            t0, t1 = i / n, min(j + 1, n) / n
            out.append([round(x1 + t0 * (x2 - x1), 3), round(y1 + t0 * (y2 - y1), 3),
                        round(x1 + t1 * (x2 - x1), 3), round(y1 + t1 * (y2 - y1), 3),
                        cls[i], eixo_idx])
            i = j + 1
    return out


def _json_errors(contexto):
    """Impede que excecoes de rotas scan virem a pagina HTML 500 do Flask."""
    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                traceback.print_exc()
                return jsonify({
                    'error': f'{contexto}: {type(exc).__name__}: {exc}'
                }), 500
        return wrapped
    return decorator


def _normalizar_eixos_preview(raw):
    """Validate the geometry snapshot received from the trusted front flow."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError('eixos do preview devem ser uma lista')
    if len(raw) > 2000:
        raise ValueError('preview excede 2000 paredes')

    result = []
    valid_labels = {'single', 'interior', 'exterior'}
    for index, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            raise ValueError(f'eixo {index + 1} invalido')
        values = [float(value) for value in row[:5]]
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f'eixo {index + 1} contem coordenada invalida')
        if np.hypot(values[2] - values[0], values[3] - values[1]) < 0.05:
            continue
        values[4] = float(np.clip(values[4], 0.03, 0.75))
        label = str(row[5])
        if label not in valid_labels:
            label = 'interior'
        result.append([*values, label])
    return result


def _normalizar_aberturas_preview(raw, n_eixos):
    """Valida as esquadrias vindas do editor. None = fluxo direto (sem editor).
    Lista (mesmo vazia) = o cliente passou pelo editor e ela é a verdade."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError('aberturas devem ser uma lista')
    if len(raw) > 2000:
        raise ValueError('excede 2000 aberturas')
    result = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f'abertura {index + 1} invalida')
        try:
            eixo_idx = int(row['eixo_idx'])
            s_centro = float(row['s_centro'])
            largura = float(row['largura'])
        except (KeyError, ValueError, TypeError):
            raise ValueError(f'abertura {index + 1}: campos invalidos')
        if not (0 <= eixo_idx < n_eixos):
            # eixo referenciado nao existe nesta receita: ignora silenciosamente
            continue
        if not (np.isfinite(s_centro) and np.isfinite(largura)) or largura < 0.1:
            continue
        tipo = 'window' if str(row.get('tipo')) == 'window' else 'door'
        item = {'eixo_idx': eixo_idx, 'tipo': tipo,
                's_centro': round(s_centro, 4), 'largura': round(largura, 4)}
        if row.get('altura') is not None:
            try:
                altura = float(row['altura'])
            except (ValueError, TypeError):
                raise ValueError(f'abertura {index + 1}: altura invalida')
            if np.isfinite(altura) and 0.05 <= altura <= 20.0:
                item['altura'] = round(altura, 4)
        if tipo == 'window' and row.get('peitoril') is not None:
            try:
                peitoril = float(row['peitoril'])
            except (ValueError, TypeError):
                raise ValueError(f'abertura {index + 1}: peitoril invalido')
            if np.isfinite(peitoril) and 0.0 <= peitoril <= 20.0:
                item['peitoril'] = round(peitoril, 4)
        result.append(item)
    return result


def _normalizar_modelo_preview(raw, n_eixos):
    """Sanitiza o snapshot BIM aprovado usado para alturas, slabs e spaces.

    Os eixos e aberturas continuam passando pelos validadores dedicados acima;
    aqui entram apenas os dados derivados que a finalizacao automatica exibiu
    ao cliente antes da exportacao.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError('modelo aprovado deve ser um objeto')

    walls_raw = raw.get('paredes', [])
    if not isinstance(walls_raw, list) or len(walls_raw) != n_eixos:
        raise ValueError('snapshot de paredes difere dos eixos aprovados')
    walls = []
    for index, wall in enumerate(walls_raw):
        if not isinstance(wall, dict):
            raise ValueError(f'parede {index + 1} do snapshot invalida')
        item = {'id': str(wall.get('id') or f'W-{index + 1:03d}')}
        for field, low, high in (
                ('altura', 0.10, 20.0), ('elevacao', -100.0, 1000.0)):
            if wall.get(field) is not None:
                value = float(wall[field])
                if not np.isfinite(value) or not low <= value <= high:
                    raise ValueError(f'{item["id"]}: {field} invalida')
                item[field] = value
        walls.append(item)

    slab_raw = raw.get('laje') or {}
    if not isinstance(slab_raw, dict):
        raise ValueError('laje do snapshot invalida')
    contour = []
    for value in slab_raw.get('contorno', [])[:10000]:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError('ponto invalido no contorno da laje')
        x, y = float(value[0]), float(value[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError('coordenada invalida no contorno da laje')
        contour.append([x, y])

    slab = {'contorno': contour}
    for face in ('piso', 'teto'):
        source = slab_raw.get(face) or {}
        thickness = float(source.get('espessura', 0.12))
        if not np.isfinite(thickness) or not 0.01 <= thickness <= 2.0:
            raise ValueError(f'espessura de {face} invalida')
        slab[face] = {
            'ativo': bool(source.get('ativo', True)),
            'espessura': thickness,
        }

    spaces = []
    spaces_raw = raw.get('spaces') or []
    if not isinstance(spaces_raw, list) or len(spaces_raw) > 2000:
        raise ValueError('spaces do snapshot invalidos')
    for index, space in enumerate(spaces_raw):
        if not isinstance(space, dict):
            continue
        vertices = []
        for value in space.get('contorno', [])[:10000]:
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            x, y = float(value[0]), float(value[1])
            if np.isfinite(x) and np.isfinite(y):
                vertices.append([x, y])
        if len(vertices) >= 3:
            spaces.append({
                'id': str(space.get('id') or f'SPACE-{index + 1:03d}'),
                'contorno': vertices,
                'area': float(space.get('area', 0.0)),
            })
    return {'paredes': walls, 'laje': slab, 'spaces': spaces}


def _auditar_paredes_ifc(ifc_path, eixos):
    """Prova do contrato tela->IFC: compara cada eixo aprovado no preview com a
    parede correspondente no IFC FINAL (pos-bake). Devolve string pro relatorio
    de etapas do front. Le a malha direto do IfcPolygonalFaceSet (create_shape
    e instavel com faceset nesta versao do ifcopenshell); se a parede nao foi
    assada, cai pro Axis."""
    import ifcopenshell
    import ifcopenshell.util.placement as _P

    m = ifcopenshell.open(str(ifc_path))
    malhas = []
    for w in m.by_type('IfcWall') + m.by_type('IfcWallStandardCase'):
        # O gerador acrescenta a referencia depois do marcador, por exemplo
        # ``preview-locked; reference=W-S01-001``. O marcador e o contrato;
        # a referencia e apenas rastreabilidade.
        if not (w.Description or '').startswith('preview-locked'):
            continue
        M = _P.get_local_placement(w.ObjectPlacement)
        vs = []
        for r in (w.Representation.Representations if w.Representation else []):
            for it in r.Items:
                if it.is_a('IfcPolygonalFaceSet'):
                    pts = np.asarray(it.Coordinates.CoordList, dtype=float)
                    vs.append((M @ np.c_[pts, np.ones(len(pts))].T).T[:, :2])
            if not vs and r.RepresentationIdentifier == 'Axis':
                for it in r.Items:
                    if it.is_a('IfcPolyline'):
                        pts = np.asarray([p.Coordinates[:2] for p in it.Points], dtype=float)
                        vs.append((M @ np.c_[pts, np.zeros(len(pts)), np.ones(len(pts))].T).T[:, :2])
        if vs:
            malhas.append(np.vstack(vs))

    # eixo pode ter sido descartado de proposito (<5cm): audita so os validos
    validos = [e for e in eixos
               if np.hypot(e[2] - e[0], e[3] - e[1]) >= 0.05]
    if len(malhas) != len(validos):
        return (f'DIVERGIU: {len(validos)} paredes aprovadas na tela, '
                f'{len(malhas)} no IFC')

    # erro combinado (centro + comprimento + espessura) por par eixo<->malha
    E = np.full((len(validos), len(malhas)), np.inf)
    for i, e in enumerate(validos):
        a, b, esp = np.array(e[0:2], float), np.array(e[2:4], float), float(e[4])
        L = np.linalg.norm(b - a)
        u = (b - a) / L
        n = np.array([-u[1], u[0]])
        cx = (a + b) / 2
        for j, v in enumerate(malhas):
            t = (v - cx) @ u
            s = (v - cx) @ n
            E[i, j] = (abs((t.max() + t.min()) / 2) + abs((s.max() + s.min()) / 2)
                       + abs((t.max() - t.min()) - L) + abs((s.max() - s.min()) - esp))
    # pareamento guloso global (sem scipy): pega sempre o menor erro restante
    pior = 0.0
    livres_i, livres_j = set(range(len(validos))), set(range(len(malhas)))
    while livres_i:
        i, j = min(((i, j) for i in livres_i for j in livres_j),
                   key=lambda ij: E[ij[0], ij[1]])
        pior = max(pior, float(E[i, j]))
        livres_i.discard(i)
        livres_j.discard(j)
    if pior > 0.02:
        return f'DIVERGIU: pior erro {pior:.3f} m (tolerancia 0.02)'
    return f'ok — {len(validos)}/{len(validos)} paredes exatas (erro máx {pior:.3f} m)'


# O preview usa o MESMO detector do gerador (identify_walls) — motor unico.
# As funcoes de plot do cloud2bim sao neutralizadas (no servidor so custam tempo).
def _validar_ifc_download(
        ifc_path,
        expected_walls=None,
        expected_openings=None,
        expected_spaces=None,
        require_covering=False):
    """Bloqueia o download de um IFC estruturalmente incompleto."""
    import ifcopenshell

    model = ifcopenshell.open(str(ifc_path))
    if len(model.by_type('IfcProject')) != 1:
        raise RuntimeError('IFC final invalido: projeto ausente ou duplicado')

    if expected_walls is not None:
        wall_count = len(model.by_type('IfcWall'))
        if wall_count != int(expected_walls):
            raise RuntimeError(
                'IFC final invalido: %d paredes aprovadas, %d exportadas'
                % (int(expected_walls), wall_count)
            )

    openings = model.by_type('IfcOpeningElement')
    orphan_openings = []
    for opening in openings:
        voids = [
            relation for relation in model.get_inverse(opening)
            if relation.is_a('IfcRelVoidsElement')
            and relation.RelatedOpeningElement == opening
        ]
        if len(voids) != 1:
            orphan_openings.append(opening.id())
    if orphan_openings:
        raise RuntimeError(
            'IFC final invalido: %d aberturas sem vinculo unico ao hospedeiro'
            % len(orphan_openings)
        )

    if expected_openings is not None:
        filling_count = (
            len(model.by_type('IfcDoor')) + len(model.by_type('IfcWindow'))
        )
        if filling_count != int(expected_openings):
            raise RuntimeError(
                'IFC final invalido: %d esquadrias aprovadas, %d exportadas'
                % (int(expected_openings), filling_count)
            )

    if expected_spaces is not None:
        space_count = len(model.by_type('IfcSpace'))
        if space_count != int(expected_spaces):
            raise RuntimeError(
                'IFC final invalido: %d spaces aprovados, %d exportados'
                % (int(expected_spaces), space_count)
            )

    # IFC4 spatial hierarchy: IfcSpace must decompose an IfcBuildingStorey.
    # A mere IfcRelContainedInSpatialStructure is schema-loadable, but leaves
    # Bonsai without the parent collection and crashes its IFC importer.
    invalid_spaces = []
    for space in model.by_type('IfcSpace'):
        aggregate_rels = [
            relation for relation in model.get_inverse(space)
            if relation.is_a('IfcRelAggregates')
            and space in relation.RelatedObjects
        ]
        valid_parent = (
            len(aggregate_rels) == 1
            and aggregate_rels[0].RelatingObject is not None
            and aggregate_rels[0].RelatingObject.is_a('IfcBuildingStorey')
        )
        if not valid_parent:
            invalid_spaces.append(space.id())
    if invalid_spaces:
        raise RuntimeError(
            'IFC final invalido: %d IfcSpace sem agregacao unica ao pavimento'
            % len(invalid_spaces)
        )

    spaces = model.by_type('IfcSpace')
    coverings = model.by_type('IfcCovering')
    if require_covering:
        if not spaces:
            raise RuntimeError(
                'IFC final invalido: forro ativo, mas nenhuma area fechada '
                'foi reconhecida; revise os gaps das paredes'
            )
        if len(coverings) != len(spaces):
            raise RuntimeError(
                'IFC final invalido: %d spaces reconhecidos, %d forros exportados'
                % (len(spaces), len(coverings))
            )

    invalid_window_types = [
        item.id() for item in model.by_type('IfcWindowType')
        if not getattr(item, 'PartitioningType', None)
    ]
    if invalid_window_types:
        raise RuntimeError(
            'IFC final invalido: %d tipos de janela sem PartitioningType'
            % len(invalid_window_types)
        )

    orphan_shapes = [
        shape.id() for shape in model.by_type('IfcProductDefinitionShape')
        if not model.get_inverse(shape)
    ]
    if orphan_shapes:
        raise RuntimeError(
            'IFC final invalido: %d representacoes de produto orfas'
            % len(orphan_shapes)
        )

    return (
        'ok: %d aberturas vinculadas, %d produtos, %d spaces, %d forros'
        % (
            len(openings),
            len(model.by_type('IfcProduct')),
            len(spaces),
            len(coverings),
        )
    )


import os as _os
_os.environ.setdefault('MPLBACKEND', 'Agg')
import aux_functions as _AF
for _nome in ('plot_histogram', 'plot_binary_image', 'plot_contours',
              'plot_segments_with_random_colors', 'plot_parallel_wall_groups'):
    if hasattr(_AF, _nome):
        setattr(_AF, _nome, lambda *a, **k: None)


def _carregar_nuvem(path: Path):
    """ply/e57/xyz -> (N,3) float32. Reusa a logica do rodar.py.

    O loader do rodar.py usa sys.exit() em erro (comportamento de CLI); num
    servidor isso mataria a thread da request sem resposta — converte pra
    excecao normal.
    """
    try:
        from rodar import carregar_nuvem
        pts, _rgb = carregar_nuvem(path)
    except SystemExit as e:
        raise ValueError(str(e) or 'nuvem invalida')
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 1000:
        raise ValueError(f'nuvem vazia ou invalida ({len(pts)} pontos)')
    return pts


def _autodiag_from_hist(zmin, zmax, cnt, merge_dist=0.9):
    """Rebuild the slab recipe from the compact vertical histogram."""
    zmin, zmax = float(zmin), float(zmax)
    cnt = np.asarray(cnt, dtype=np.int64)
    n = len(cnt)
    if n == 0:
        raise ValueError('histograma vertical vazio')
    mx = max(1, int(cnt.max()))

    def grupos_para(thr):
        passa = cnt > thr * mx
        intervals, start = [], None
        for i in range(n):
            if passa[i] and start is None:
                start = i
            elif not passa[i] and start is not None:
                intervals.append((start, i - 1))
                start = None
        if start is not None:
            intervals.append((start, n - 1))
        grupos, last_end = [], None
        for a, b in intervals:
            za, zb = zmin + a * Z_STEP, zmin + b * Z_STEP + Z_STEP
            if last_end is None or za - last_end >= merge_dist:
                grupos.append([za, zb])
            else:
                grupos[-1][1] = zb
            last_end = zb
        return grupos

    tabela = []
    for thr in (0.5, 0.4, 0.3, 0.25, 0.2, 0.15):
        g = grupos_para(thr)
        tabela.append({'thr': thr, 'grupos': len(g)})
    validos = [t for t in tabela if t['grupos'] >= 2]
    melhor = max(validos, key=lambda t: (t['grupos'], t['thr']))['thr'] if validos else 0.15
    return tabela, melhor, grupos_para, (zmin, zmax, cnt.tolist())


def _autodiag(z, merge_dist=0.9):
    """Tabela thr -> (fatias, grupos de laje) + z dos grupos por thr."""
    z = np.round(z, 3)
    zmin, zmax = float(z.min()), float(z.max())
    n = int((zmax - zmin) / Z_STEP + 1)
    idx = np.clip(((z - zmin) / Z_STEP).astype(np.int64), 0, n - 1)
    cnt = np.bincount(idx, minlength=n).astype(np.int64)
    return _autodiag_from_hist(zmin, zmax, cnt, merge_dist=merge_dist)


def _construir_sessao(path: Path, full_cloud_path: Path | None = None):
    """Build the reusable in-memory scan representation from a stored cloud."""
    pts = _carregar_nuvem(path)
    if full_cloud_path is not None:
        full_cloud_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            full_cloud_path,
            np.asarray(pts, dtype=np.float32),
            allow_pickle=False,
        )
    ext = pts.max(0) - pts.min(0)
    area = float(ext[0] * ext[1])
    cell = float(np.clip(np.sqrt(area / 500_000), 0.04, 0.15))
    zbin = float(max(Z_STEP, ext[2] / 48))
    xmin, ymin, zmin = (float(v) for v in pts.min(0))
    NX = int(ext[0] / cell) + 1
    NY = int(ext[1] / cell) + 1
    NZ = int(ext[2] / zbin) + 1
    occ = np.zeros((NZ, NY, NX), dtype=np.uint16)
    ix = np.clip(((pts[:, 0] - xmin) / cell).astype(np.int32), 0, NX - 1)
    iy = np.clip(((pts[:, 1] - ymin) / cell).astype(np.int32), 0, NY - 1)
    iz = np.clip(((pts[:, 2] - zmin) / zbin).astype(np.int32), 0, NZ - 1)
    np.add.at(occ, (iz, iy, ix), 1)
    tabela, thr_sugerido, grupos_para, zhist = _autodiag(pts[:, 2])
    if len(pts) > 400_000:
        amostra = pts[np.random.default_rng(0).choice(
            len(pts), 400_000, replace=False)]
    else:
        amostra = pts
    import time as _time
    return {
        'path': str(path),
        'full_cloud_path': str(full_cloud_path) if full_cloud_path else None,
        'occ': occ, 'cell': cell, 'zbin': zbin,
        'source_stem': path.stem,
        'xmin': xmin, 'ymin': ymin, 'zmin': zmin,
        'npts': int(len(pts)), 'z_counts': zhist,
        'grupos_para': grupos_para, 'amostra': amostra,
        'extent': [round(float(v), 2) for v in ext],
        'tabela_thr': tabela, 'thr_sugerido': thr_sugerido,
        'ultimo_acesso': _time.monotonic(),
    }


def _salvar_cache_sessao(sid, sessao, upload_folder: Path):
    """Persist the compact editor state without retaining the raw upload."""
    cache_path = upload_folder / f'{sid}_session.npz'
    temp_path = upload_folder / f'{sid}_session.tmp.npz'
    zhist_min, zhist_max, zhist_counts = sessao['z_counts']
    try:
        np.savez_compressed(
            temp_path,
            occ=np.asarray(sessao['occ'], dtype=np.uint16),
            amostra=np.asarray(sessao['amostra'], dtype=np.float32),
            cell=np.float64(sessao['cell']),
            zbin=np.float64(sessao['zbin']),
            xmin=np.float64(sessao['xmin']),
            ymin=np.float64(sessao['ymin']),
            zmin=np.float64(sessao['zmin']),
            npts=np.int64(sessao['npts']),
            extent=np.asarray(sessao['extent'], dtype=np.float64),
            zhist_min=np.float64(zhist_min),
            zhist_max=np.float64(zhist_max),
            zhist_counts=np.asarray(zhist_counts, dtype=np.int64),
            source_stem=np.asarray(str(sessao.get('source_stem') or sid)),
        )
        temp_path.replace(cache_path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
    sessao['path'] = str(cache_path)
    full_cloud_path = upload_folder / f'{sid}_cloud.npy'
    if full_cloud_path.is_file():
        sessao['full_cloud_path'] = str(full_cloud_path)
    return cache_path


def _carregar_cache_sessao(cache_path: Path):
    """Restore the voxel/editor session without reopening the source E57."""
    with np.load(cache_path, allow_pickle=False) as cache:
        zhist_min = float(cache['zhist_min'])
        zhist_max = float(cache['zhist_max'])
        zhist_counts = np.asarray(cache['zhist_counts'], dtype=np.int64)
        tabela, thr_sugerido, grupos_para, zhist = _autodiag_from_hist(
            zhist_min, zhist_max, zhist_counts)
        import time as _time
        full_cloud_path = cache_path.with_name(
            cache_path.name.replace('_session.npz', '_cloud.npy'))
        return {
            'path': str(cache_path),
            'full_cloud_path': (
                str(full_cloud_path) if full_cloud_path.is_file() else None),
            'source_stem': str(cache['source_stem'].item()),
            'occ': np.asarray(cache['occ'], dtype=np.uint16),
            'amostra': np.asarray(cache['amostra'], dtype=np.float32),
            'cell': float(cache['cell']),
            'zbin': float(cache['zbin']),
            'xmin': float(cache['xmin']),
            'ymin': float(cache['ymin']),
            'zmin': float(cache['zmin']),
            'npts': int(cache['npts']),
            'extent': [round(float(v), 2) for v in cache['extent']],
            'z_counts': zhist,
            'grupos_para': grupos_para,
            'tabela_thr': tabela,
            'thr_sugerido': thr_sugerido,
            'ultimo_acesso': _time.monotonic(),
        }


def _obter_ou_recuperar_sessao(sid, upload_folder: Path):
    """Keep a browser scan session alive across a local backend restart.

    The source upload already exists on disk. Rebuilding the voxel cache is
    slower than a normal request, but avoids asking the client to upload a
    large E57 again merely because code was reloaded during review.
    """
    session = _tocar(_SESSOES.get(sid))
    if session is not None:
        return session
    if not isinstance(sid, str) or not re.fullmatch(r'[0-9a-f]{12}', sid):
        return None
    cache_path = upload_folder / f'{sid}_session.npz'
    if cache_path.is_file():
        rebuilt = _carregar_cache_sessao(cache_path)
        with _LOCK:
            session = _SESSOES.setdefault(sid, rebuilt)
        print(f'[scan] sessao {sid} recuperada do cache compacto')
        return _tocar(session)

    # Compatibilidade de uma unica passagem com uploads feitos pela versao
    # anterior: converte para cache e apaga o E57/PLY/XYZ bruto imediatamente.
    cloud_suffixes = {'.e57', '.ply', '.xyz', '.las', '.laz', '.pts', '.csv'}
    candidates = sorted(upload_folder.glob(f'{sid}_*'))
    source = next((path for path in candidates
                   if path.is_file() and path.suffix.lower() in cloud_suffixes), None)
    if source is None:
        return None
    rebuilt = _construir_sessao(
        source, upload_folder / f'{sid}_cloud.npy')
    _salvar_cache_sessao(sid, rebuilt, upload_folder)
    source.unlink(missing_ok=True)
    with _LOCK:
        session = _SESSOES.setdefault(sid, rebuilt)
    print(f'[scan] sessao {sid} migrada para cache compacto')
    return _tocar(session)


def _hybrid_runtime_paths():
    configured = os.environ.get('CLOUD2BIM_ML_PYTHON', '').strip()
    ml_python = Path(configured) if configured else _LOCAL_ML_PYTHON
    if not ml_python.is_file():
        raise RuntimeError(
            'runtime da ML nao instalado; configure CLOUD2BIM_ML_PYTHON')
    required = (
        HYBRID_TILED_SCRIPT,
        HYBRID_WALL_ML_SCRIPT,
        HYBRID_OPENING_WEIGHTS,
        HYBRID_WALL_WEIGHTS,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('modelos do fluxo hibrido ausentes: ' + ', '.join(missing))
    return ml_python


def _hybrid_result_payload(stitched_path: Path, predictions_path: Path,
                           png_url: str, floor_z: float, ceiling_z: float):
    stitched = json.loads(stitched_path.read_text(encoding='utf-8'))
    report = json.loads(predictions_path.read_text(encoding='utf-8'))
    predictions = {
        item['wall_id']: item for item in report.get('predictions', [])
    }
    walls = []
    for wall in stitched.get('paredes', []):
        prediction = predictions.get(wall.get('id'), {})
        predicted_class = str(prediction.get('predicted_class') or 'uncertain')
        predicted_probability = float(
            prediction.get('predicted_probability') or 0.0)
        display_class = (
            predicted_class if predicted_probability >= 0.60 else 'uncertain')
        walls.append({
            'id': wall['id'],
            'ax': float(wall['ax']), 'ay': float(wall['ay']),
            'bx': float(wall['bx']), 'by': float(wall['by']),
            'espessura': float(wall['espessura']),
            'ml_class': display_class,
            'ml_predicted_class': predicted_class,
            'ml_probability': predicted_probability,
            'wall_probability': float(prediction.get('wall_probability') or 0.0),
            'proposed_keep': bool(prediction.get('proposed_keep', True)),
        })
    openings = []
    for index, opening in enumerate(stitched.get('aberturas', []), start=1):
        kind = str(opening.get('class') or '')
        if kind not in {'door', 'window'}:
            continue
        z_min = float(opening.get('z_min', floor_z))
        z_max = float(opening.get('z_max', ceiling_z))
        openings.append({
            'id': f'HYB-O-{index:03d}',
            'wall_id': opening['wall_id'],
            'class': kind,
            's_center': float(opening['s_center']),
            'width': float(opening.get('width') or 0.90),
            'height': max(0.30, z_max - z_min),
            'sill': 0.0 if kind == 'door' else max(0.0, z_min - floor_z),
            'confidence': float(opening.get('confidence') or 0.0),
        })
    if walls:
        xs = [value for wall in walls for value in (wall['ax'], wall['bx'])]
        ys = [value for wall in walls for value in (wall['ay'], wall['by'])]
        bounds = [min(xs), min(ys), max(xs), max(ys)]
    else:
        bounds = [0.0, 0.0, 1.0, 1.0]
    counts = dict(report.get('counts') or {})
    counts.update({
        'doors': sum(item['class'] == 'door' for item in openings),
        'windows': sum(item['class'] == 'window' for item in openings),
        'openings': len(openings),
    })
    return {
        'png_url': png_url,
        'bounds': bounds,
        'floor_z': floor_z,
        'ceiling_z': ceiling_z,
        'counts': counts,
        # Todos os candidatos seguem para o editor. A rede colore e recomenda;
        # somente a confirmacao humana remove uma parede do modelo BIM.
        'walls': walls,
        'openings': openings,
        'automatic_geometry_change': False,
    }


def register_scan(app, upload_folder, output_folder):
    upload_folder = Path(upload_folder)
    output_folder = Path(output_folder)
    upload_folder.mkdir(parents=True, exist_ok=True)
    _limpar_cache_scan_de_restart(upload_folder)

    @app.route('/api/scan/recover-ifc', methods=['POST'])
    def _scan_recover_ifc():
        """Reopen a generated IFC in the focused Scan-to-BIM editor."""
        uploaded = request.files.get('file')
        if not uploaded or not str(uploaded.filename or '').lower().endswith('.ifc'):
            return jsonify({'error': 'envie um arquivo .ifc'}), 400
        recovery_path = upload_folder / (
            f"recover_{uuid.uuid4().hex[:12]}_{secure_filename(uploaded.filename)}"
        )
        try:
            uploaded.save(str(recovery_path))
            from bim_editing.ifc_recovery import recover_editor_model
            force_ceiling = str(request.form.get('force_ceiling', '')).lower() in (
                '1', 'true', 'yes', 'on')
            return jsonify(recover_editor_model(
                recovery_path,
                force_ceiling=force_ceiling,
            ))
        except Exception as exc:
            traceback.print_exc()
            return jsonify({'error': f'falha ao recuperar IFC: {exc}'}), 500
        finally:
            try:
                recovery_path.unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # 1) UPLOAD: fase cara (1x) — grade voxel + histograma de z
    # ------------------------------------------------------------------
    @app.route('/api/scan/upload', methods=['POST'])
    def _scan_upload():
        path = None
        try:
            f = request.files.get('file')
            if not f:
                return jsonify({'error': 'arquivo ausente'}), 400
            sid = uuid.uuid4().hex[:12]
            path = upload_folder / f"{sid}_{secure_filename(f.filename)}"
            f.save(str(path))

            sessao = _construir_sessao(
                path, upload_folder / f'{sid}_cloud.npy')
            _salvar_cache_sessao(sid, sessao, upload_folder)
            path.unlink(missing_ok=True)
            path = None
            with _LOCK:
                _SESSOES[sid] = sessao
            grupos = sessao['grupos_para'](sessao['thr_sugerido'])
            zh_min, zh_max, zh_cnt = sessao['z_counts']
            return jsonify({
                'sid': sid, 'n_pontos': sessao['npts'],
                'extent': sessao['extent'],
                'tabela_thr': sessao['tabela_thr'],
                'thr_sugerido': sessao['thr_sugerido'],
                'lajes': grupos,
                'z_hist': {'zmin': round(zh_min, 3), 'step': Z_STEP,
                           'counts': zh_cnt},
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            # f.save pode falhar no meio (por exemplo disco cheio). Nunca deixe
            # um E57 parcial ocupando o disco depois de uma resposta de erro.
            try:
                if path is not None:
                    path.unlink(missing_ok=True)
            except OSError:
                pass
            return jsonify({'error': str(e)}), 500

    # ------------------------------------------------------------------
    # 2) LAJES: re-agrupa com outro threshold (instantaneo)
    # ------------------------------------------------------------------
    @app.route('/api/scan/lajes', methods=['POST'])
    def _scan_lajes():
        d = request.get_json(force=True)
        s = _obter_ou_recuperar_sessao(d.get('sid'), upload_folder)
        if not s:
            return jsonify({'error': 'sessao expirada — refaça o upload'}), 404
        grupos = s['grupos_para'](float(d.get('thr', 0.3)))
        bandas = [[grupos[i][1], grupos[i + 1][0]] for i in range(len(grupos) - 1)]
        return jsonify({'lajes': grupos, 'bandas': bandas})

    # ------------------------------------------------------------------
    # 2b) NUVEM 3D SOB DEMANDA: usa a amostra ja mantida na sessao.
    #     Nao rele o E57 e nao envia os milhoes de pontos originais.
    # ------------------------------------------------------------------
    @app.route('/api/scan/cloud-preview', methods=['POST'])
    @_json_errors('preview 3D da nuvem')
    def _scan_cloud_preview():
        d = request.get_json(force=True)
        s = _obter_ou_recuperar_sessao(d.get('sid'), upload_folder)
        if not s:
            return jsonify({'error': 'sessao expirada - refaca o upload'}), 404

        max_points = int(np.clip(int(d.get('max_points', 120_000)), 1_000, 150_000))
        points = np.asarray(s['amostra'], dtype=np.float32)

        requested_base = d.get('base_elevation')
        if requested_base is None:
            vertical_base = float(np.min(points[:, 2])) if len(points) else 0.0
            discarded_below_base = 0
        else:
            vertical_base = float(requested_base)
            if not np.isfinite(vertical_base):
                raise ValueError('base_elevation invalida')
            # A visualizacao representa somente o pavimento selecionado. Pontos
            # abaixo da face inferior da laje sao ruido ou outro nivel e nao
            # podem atravessar o plano local Y=0 do editor.
            original_count = len(points)
            points = points[points[:, 2] >= vertical_base]
            discarded_below_base = original_count - len(points)

        if len(points) > max_points:
            # Amostra deterministica e espacialmente imparcial mesmo quando o
            # arquivo de origem veio ordenado por faixa/linha do scanner.
            indices = np.random.default_rng(0).choice(
                len(points), max_points, replace=False)
            points = points[indices]

        # Three.js usa Y como altura; a nuvem original chega em X,Y,Z.
        # O editor trabalha com a base do pavimento em Y=0. Quando o cliente
        # informa a cota da faixa selecionada, usamos exatamente essa cota em
        # vez do menor ponto da amostra (que pode ser ruido ou outro andar).
        positions_3d = points[:, [0, 2, 1]].copy()
        positions_3d[:, 1] -= vertical_base
        positions = positions_3d.reshape(-1)
        return jsonify({
            'positions': positions.tolist(),
            'count': int(len(points)),
            'source_count': int(s['npts']),
            'coordinate_order': 'x,z,y',
            'vertical_base': vertical_base,
            'normalized_to_ground': True,
            'discarded_below_base': int(discarded_below_base),
        })

    # ------------------------------------------------------------------
    # 3) PAREDES: MOTOR REAL (identify_walls) na banda + camada de percepcao.
    #    Vermelho no front = eixos pareados = exatamente o que vira IfcWall.
    # ------------------------------------------------------------------
    @app.route('/api/scan/paredes', methods=['POST'])
    @_json_errors('preview de paredes')
    def _scan_paredes():
        import os
        d = request.get_json(force=True)
        s = _obter_ou_recuperar_sessao(d.get('sid'), upload_folder)
        if not s:
            return jsonify({'error': 'sessao expirada — refaça o upload'}), 404
        zlo, zhi = float(d['zlo']), float(d['zhi'])
        zlo_frac = float(d.get('zlo_frac', 0.1))
        zhi_frac = float(d.get('zhi_frac', 0.9))
        min_len = float(d.get('min_len', 0.3))
        single_minlen = float(d.get('single_minlen', 1.5))
        contours_all = bool(d.get('contours_all', True))

        occ, cell, zbin = s['occ'], s['cell'], s['zbin']
        i0 = max(0, int((zlo - s['zmin']) / zbin))
        i1 = min(occ.shape[0], max(i0 + 1, int(np.ceil((zhi - s['zmin']) / zbin))))
        banda = occ[i0:i1]
        grade = banda.sum(0)
        NY, NX = grade.shape

        # ---- camada PERCEPCAO (cinza): contornos crus da fatia escolhida ----
        f0 = i0 + int(zlo_frac * (i1 - i0))
        f1 = max(f0 + 1, i0 + int(np.ceil(zhi_frac * (i1 - i0))))
        fatia = occ[f0:f1].sum(0)
        binimg = (fatia >= 1).astype(np.uint8) * 255
        binimg = cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        retr = cv2.RETR_LIST if contours_all else cv2.RETR_EXTERNAL
        contornos, _ = cv2.findContours(binimg, retr, cv2.CHAIN_APPROX_NONE)
        segs = []
        eps_px = max(0.04 / cell, 1.5)
        for cnt in contornos:
            if len(cnt) < 4:
                continue
            aprox = cv2.approxPolyDP(cnt, eps_px, True).reshape(-1, 2)
            for i in range(len(aprox)):
                p1, p2 = aprox[i], aprox[(i + 1) % len(aprox)]
                x1 = s['xmin'] + float(p1[0]) * cell
                y1 = s['ymin'] + float(p1[1]) * cell
                x2 = s['xmin'] + float(p2[0]) * cell
                y2 = s['ymin'] + float(p2[1]) * cell
                if np.hypot(x2 - x1, y2 - y1) >= min_len:
                    segs.append([round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)])

        # ---- camada INSTANCIA (vermelho): identify_walls de verdade ----
        # pseudo-nuvem = centros dos voxels ocupados da banda (nuvem decimada
        # na resolucao da grade — mesmo dado, motor identico ao gerador)
        eixos = []
        try:
            iz, iy, ix = np.nonzero(banda)
            pc = np.column_stack([
                s['xmin'] + (ix + 0.5) * cell,
                s['ymin'] + (iy + 0.5) * cell,
                zlo + (iz + 0.5) * zbin,
            ])
            poly = None
            try:
                from matplotlib.patches import Polygon as _MplPoly
                poly = _MplPoly([(s['xmin'], s['ymin']), (s['xmin'] + NX * cell, s['ymin']),
                                 (s['xmin'] + NX * cell, s['ymin'] + NY * cell),
                                 (s['xmin'], s['ymin'] + NY * cell)])
            except Exception:
                pass
            with _DETECTOR_LOCK:
                antes = {k: os.environ.get(k) for k in
                         ('WALL_DETECTOR', 'WALL_ZLO', 'WALL_ZHI', 'SINGLE_LINE', 'SINGLE_LINE_MINLEN',
                          'SINGLE_LINE_THK', 'WALL_CONTOURS')}
                os.environ.update({
                    'WALL_DETECTOR': 'v2',
                    'WALL_ZLO': str(zlo_frac), 'WALL_ZHI': str(zhi_frac),
                    'SINGLE_LINE': '0', 'SINGLE_LINE_MINLEN': str(single_minlen),
                    'SINGLE_LINE_THK': '0.15',
                    'WALL_CONTOURS': 'all' if contours_all else 'external',
                })
                try:
                    # O V2 devolve tambem o diagnostico de confianca de cada
                    # parede. O preview antigo ainda desempacotava o contrato
                    # de seis itens; a excecao era capturada abaixo e parecia
                    # uma deteccao valida com zero paredes.
                    (starts, ends, esps, _mats, _grupos, labels,
                     _diagnosticos) = _AF.identify_walls(
                        [tuple(p) for p in pc], cell, min_len, 0.05, 0.75,
                        zlo, zhi, grid_coefficient=1, slab_polygon=poly,
                        exterior_scan=False, exterior_walls_thickness=0.3)
                    for st, en, esp, lb in zip(starts, ends, esps, labels):
                        vals = [float(st[0]), float(st[1]), float(en[0]), float(en[1]), float(esp)]
                        # pareamento degenerado gera inf/nan — JSON invalido no browser
                        if not all(np.isfinite(v) for v in vals):
                            continue
                        # O modelador tambem descarta eixos menores que 5 cm;
                        # nao os prometa no preview como se virassem paredes.
                        if np.hypot(vals[2] - vals[0], vals[3] - vals[1]) < 0.05:
                            continue
                        eixos.append([round(vals[0], 3), round(vals[1], 3),
                                      round(vals[2], 3), round(vals[3], 3),
                                      round(max(vals[4], 0.03), 3), str(lb)])
                finally:
                    for k, v in antes.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v
        except Exception:
            # Nunca transforme uma falha do motor em "0 paredes": isso seria
            # indistinguivel de uma nuvem realmente sem paredes para o cliente.
            import traceback
            traceback.print_exc()
            raise

        # ---- classificacao multi-fatia (parede/porta/janela/oclusao) ----
        # roda POR CIMA dos eixos do identify_walls quando o front manda as
        # 3 faixas dos sliders; a decisao vem da assinatura vertical.
        classificacao = []
        fatias = d.get('fatias')
        if fatias and eixos:
            try:
                classificacao = _classificar_multifatia(
                    occ, cell, zbin, s['xmin'], s['ymin'], s['zmin'],
                    zlo, zhi, eixos, fatias)
            except Exception:
                traceback.print_exc()   # classificacao e' acessoria

        # backdrop: densidade em PNG (row 0 = topo = ymax -> flipud)
        from PIL import Image
        img = np.log1p(fatia.astype(np.float32))
        img = (255 * img / max(img.max(), 1e-6)).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(np.flipud(img), mode='L').save(buf, format='PNG')
        png_b64 = base64.b64encode(buf.getvalue()).decode()

        # ---- contorno do TETO/footprint (pra "ver a leitura" no editor) ----
        # maior contorno externo da densidade da banda, fechado e simplificado.
        contorno_teto = []
        try:
            occ_bin = (grade >= 1).astype(np.uint8) * 255
            occ_bin = cv2.morphologyEx(occ_bin, cv2.MORPH_CLOSE,
                                       np.ones((7, 7), np.uint8))
            cnts, _ = cv2.findContours(occ_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                maior = max(cnts, key=cv2.contourArea)
                eps = max(0.15 / cell, 2.0)   # ~15cm de tolerancia no contorno
                poly = cv2.approxPolyDP(maior, eps, True).reshape(-1, 2)
                contorno_teto = [[round(s['xmin'] + float(px) * cell, 3),
                                  round(s['ymin'] + float(py) * cell, 3)]
                                 for px, py in poly]
        except Exception:
            contorno_teto = []

        return jsonify({
            'segmentos': segs, 'n_segmentos': len(segs),
            'eixos': eixos, 'n_paredes': len(eixos),
            'classificacao': classificacao,
            'png': png_b64,
            'bounds': [s['xmin'], s['ymin'],
                       s['xmin'] + NX * cell, s['ymin'] + NY * cell],
            'contorno_teto': contorno_teto,
        })

    # ------------------------------------------------------------------
    # 3b) ESCADAS: detector real de lances na banda (fase 3 do wizard)
    # ------------------------------------------------------------------
    @app.route('/api/scan/escadas', methods=['POST'])
    def _scan_escadas():
        d = request.get_json(force=True)
        s = _obter_ou_recuperar_sessao(d.get('sid'), upload_folder)
        if not s:
            return jsonify({'error': 'sessao expirada — refaça o upload'}), 404
        zlo, zhi = float(d['zlo']), float(d['zhi'])
        area_min = float(d.get('area_min', 0.8))
        try:
            sys.path.insert(0, str(C2B_DIR / 'prototipos'))
            from detect_escada_reta import detectar_lances, fundir_lances
            lances = fundir_lances(detectar_lances(
                s['amostra'].astype(np.float64), zlo, zhi,
                cell=0.12, area_min=area_min))
            out = []
            for L in lances:
                riser = L['riser_h']
                out.append({
                    'cx': round(L['cx'], 2), 'cy': round(L['cy'], 2),
                    'ux': round(L['ux'], 3), 'uy': round(L['uy'], 3),
                    'comprimento': round(L['comprimento'], 2),
                    'largura': round(L['largura'], 2),
                    'z0': round(L['z0'], 2), 'z1': round(L['z1'], 2),
                    'declividade': round(L['slope'], 2),
                    'espelho_cm': round(riser * 100, 1) if riser == riser else None,
                    'degraus_vistos': L['n_degraus_vistos'],
                })
            return jsonify({'lances': out, 'n_lances': len(out)})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'detector de escada: {e}'}), 500

    # ------------------------------------------------------------------
    # 3c) FLUXO HIBRIDO: tiles geometricos + YOLO de aberturas + ML que
    #     revisa candidatos de parede. A ML nunca remove geometria sozinha.
    # ------------------------------------------------------------------
    @app.route('/api/scan/hibrido', methods=['POST'])
    def _scan_hibrido():
        d = request.get_json(force=True)
        sid = d.get('sid')
        s = _obter_ou_recuperar_sessao(sid, upload_folder)
        if not s:
            return jsonify({'error': 'sessao expirada - refaca o upload'}), 404
        full_cloud_path = Path(s.get('full_cloud_path') or '')
        if not full_cloud_path.is_file():
            return jsonify({
                'error': (
                    'esta sessao foi criada antes do fluxo hibrido; refaca o '
                    'upload para preservar a nuvem completa')
            }), 409
        try:
            floor_z = float(d.get('floor_z', s['zmin']))
            ceiling_z = float(d.get('ceiling_z', floor_z + 3.0))
        except (TypeError, ValueError):
            return jsonify({'error': 'cotas do pavimento invalidas'}), 400
        if not np.isfinite(floor_z) or not np.isfinite(ceiling_z):
            return jsonify({'error': 'cotas do pavimento invalidas'}), 400
        if ceiling_z <= floor_z + 0.30:
            return jsonify({'error': 'pe-direito insuficiente para a analise'}), 400
        try:
            ml_python = _hybrid_runtime_paths()
        except RuntimeError as exc:
            return jsonify({'error': str(exc)}), 503

        jid = uuid.uuid4().hex[:12]
        _JOBS[jid] = {
            'status': 'rodando',
            'kind': 'hybrid-review',
            'etapa': 'preparando blocos',
            'url': None,
            'erro': None,
        }
        source_stem = secure_filename(str(s.get('source_stem') or sid)) or sid

        def _job_hibrido():
            import time as _time
            outdir = upload_folder / f'hybrid_{jid}'
            heuristic_dir = outdir / 'heuristic'
            ml_dir = outdir / 'wall_ml'
            started = _time.monotonic()

            def run_stage(stage, command, timeout):
                _JOBS[jid]['etapa'] = stage
                env = dict(os.environ)
                dependency_dir = BASE_DIR.parent / '.codex_tmp' / 'cloud2bim_deps'
                if dependency_dir.is_dir():
                    previous = env.get('PYTHONPATH', '')
                    env['PYTHONPATH'] = (
                        str(dependency_dir)
                        + (os.pathsep + previous if previous else ''))
                completed = subprocess.run(
                    [str(value) for value in command],
                    cwd=BASE_DIR,
                    env=env,
                    capture_output=True,
                    text=True,
                    errors='replace',
                    timeout=timeout,
                )
                if completed.returncode:
                    log = ((completed.stdout or '') + '\n'
                           + (completed.stderr or '')).strip()[-1200:]
                    raise RuntimeError(f'{stage}: {log}')

            try:
                outdir.mkdir(parents=True, exist_ok=True)
                run_stage('heuristica em blocos + YOLO', [
                    ml_python,
                    HYBRID_TILED_SCRIPT,
                    full_cloud_path,
                    HYBRID_OPENING_WEIGHTS,
                    heuristic_dir,
                    '--workers', str(min(3, os.cpu_count() or 1)),
                ], 3600)
                stitched_path = (
                    heuristic_dir / 'stitched' / 'tiled_stitched_model.json')
                if not stitched_path.is_file():
                    raise RuntimeError('a costura nao produziu o modelo geometrico')

                run_stage('ML revisando paredes', [
                    ml_python,
                    HYBRID_WALL_ML_SCRIPT,
                    full_cloud_path,
                    stitched_path,
                    HYBRID_WALL_WEIGHTS,
                    ml_dir,
                    '--floor-z', str(floor_z),
                    '--ceiling-z', str(ceiling_z),
                    '--point-keep-ratio', '1.0',
                ], 1800)
                review_png = ml_dir / 'wall_ml_review.png'
                predictions_path = ml_dir / 'wall_ml_predictions.json'
                if not review_png.is_file() or not predictions_path.is_file():
                    raise RuntimeError('a ML nao produziu o pacote de revisao')

                png_name = f'{jid}_{source_stem}_hibrido_ml.png'
                json_name = f'{jid}_{source_stem}_hibrido_ml.json'
                model_name = f'{jid}_{source_stem}_hibrido_model.json'
                shutil.copy2(review_png, output_folder / png_name)
                shutil.copy2(predictions_path, output_folder / json_name)
                shutil.copy2(stitched_path, output_folder / model_name)
                hybrid = _hybrid_result_payload(
                    stitched_path,
                    predictions_path,
                    f'/outputs/{png_name}',
                    floor_z,
                    ceiling_z,
                )
                hybrid['predictions_url'] = f'/outputs/{json_name}'
                hybrid['model_url'] = f'/outputs/{model_name}'
                hybrid['elapsed_seconds'] = round(_time.monotonic() - started, 2)
                _JOBS[jid].update(
                    status='pronto',
                    etapa='revisao pronta',
                    url=hybrid['png_url'],
                    hybrid=hybrid,
                )
            except Exception as exc:
                traceback.print_exc()
                _JOBS[jid].update(status='erro', erro=str(exc)[-800:])
            finally:
                _JOBS[jid]['fim'] = _time.monotonic()
                if outdir.exists():
                    shutil.rmtree(outdir, ignore_errors=True)

        if os.environ.get('SCAN_JOBS_SYNC', 'false').lower() in {'1', 'true', 'yes'}:
            _job_hibrido()
        else:
            threading.Thread(target=_job_hibrido, daemon=True).start()
        return jsonify({'job': jid})

    # ------------------------------------------------------------------
    # 4) GERAR IFC: cadeia completa em background + poll
    # ------------------------------------------------------------------
    @app.route('/api/scan/gerar-ifc', methods=['POST'])
    def _scan_gerar():
        d = request.get_json(force=True)
        s = _obter_ou_recuperar_sessao(d.get('sid'), upload_folder)
        if not s:
            return jsonify({'error': 'sessao expirada — refaça o upload'}), 404
        jid = uuid.uuid4().hex[:12]
        _JOBS[jid] = {'status': 'rodando', 'etapa': 'pipeline', 'url': None, 'erro': None}

        env_extra = {
            'WALL_DETECTOR': 'v2',
            'WALL_ZLO': str(d.get('zlo_frac', 0.1)),
            'WALL_ZHI': str(d.get('zhi_frac', 0.9)),
            'SINGLE_LINE': '0',
            'SINGLE_LINE_MINLEN': str(d.get('single_minlen', 1.5)),
            'WALL_CONTOURS': 'all' if d.get('contours_all', True) else 'external',
        }
        thr = float(d.get('thr', 0.3))
        min_len = float(d.get('min_len', 0.3))
        stair_area_min = max(0.1, float(d.get('stair_area_min', 0.8)))
        banda_idx = max(0, int(d.get('banda_idx', 0)))
        eixos_preview = _normalizar_eixos_preview(d.get('eixos'))
        grupos_preview = s['grupos_para'](thr)
        n_bandas = max(0, len(grupos_preview) - 1)
        if eixos_preview is not None and banda_idx >= n_bandas:
            return jsonify({'error': 'pavimento do preview nao existe nesta receita'}), 400
        # aberturas do editor (opcional): None = fluxo direto (redetecta),
        # lista (mesmo vazia) = editor mandou, vira a verdade das esquadrias
        try:
            aberturas_preview = _normalizar_aberturas_preview(
                d.get('aberturas'), len(eixos_preview) if eixos_preview else 0)
            modelo_preview = _normalizar_modelo_preview(
                d.get('modelo'), len(eixos_preview) if eixos_preview else 0)
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400
        config_preview = d.get('config') if isinstance(d.get('config'), dict) else None
        src = Path(s['path'])
        source_stem = str(s.get('source_stem') or src.stem)

        def _job():
            import os
            import time as _time
            detalhes = {}
            detalhes['tempos_s'] = {}
            _JOBS[jid]['detalhes'] = detalhes
            # fora do try: o finally descarta esta pasta mesmo se o job explodir
            outdir = upload_folder / f"scanjob_{jid}"

            def passo(nome, cmd, timeout, env=None):
                _JOBS[jid]['etapa'] = nome
                started_at = _time.monotonic()
                try:
                    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                       timeout=timeout)
                finally:
                    detalhes['tempos_s'][nome] = round(
                        _time.monotonic() - started_at, 2)
                cauda = ((r.stdout or '') + (r.stderr or '')).strip()[-300:]
                return r.returncode == 0, cauda

            try:
                outdir.mkdir(parents=True, exist_ok=True)
                pipeline_env = {**os.environ, **env_extra}
                if eixos_preview is not None:
                    override_path = outdir / 'wall_overrides.json'
                    storey_override = {'eixos': eixos_preview}
                    if aberturas_preview is not None:
                        storey_override['aberturas'] = aberturas_preview
                    if config_preview:
                        storey_override['config'] = config_preview
                    if modelo_preview:
                        storey_override['modelo'] = modelo_preview
                    # A amostra comprimida pode perder uma das superfícies
                    # horizontais por densidade. Os níveis abaixo vêm do
                    # histograma completo já aprovado no preview e garantem o
                    # par piso+teto necessário para existir um pavimento.
                    storey_override['vertical'] = {
                        'floor_bottom_z': float(grupos_preview[banda_idx][0]),
                        'ceiling_bottom_z': float(grupos_preview[banda_idx + 1][0]),
                    }
                    override_path.write_text(json.dumps({
                        'storeys': {str(banda_idx): storey_override}
                    }, ensure_ascii=False), encoding='utf-8')
                    pipeline_env['WALL_OVERRIDE_FILE'] = str(override_path)
                    detalhes['paredes_preview'] = 'ok'
                    _ab_msg = ('' if aberturas_preview is None
                               else f', {len(aberturas_preview)} esquadrias do editor')
                    print(f'[scan {jid}] pavimento {banda_idx + 1}: '
                          f'{len(eixos_preview)} eixos travados pelo preview{_ab_msg}')
                pipeline_source = src
                approved_pipeline_flags = []
                if eixos_preview is not None or src.suffix.lower() == '.npz':
                    # Depois da aprovacao, a nuvem serve apenas como suporte
                    # vertical do motor legado. Nao releia/escreva dezenas de
                    # milhoes de pontos para uma geometria que ja esta travada.
                    # A mesma amostra permite que sessoes compactas sobrevivam
                    # a um restart sem conservar o E57 original no servidor.
                    approved_sample = np.asarray(s['amostra'], dtype=np.float32)
                    approved_source = outdir / 'approved_cloud_sample.xyz'
                    with approved_source.open('w', encoding='ascii') as sample_file:
                        sample_file.write('x y z r g b\n')
                        sample_rgb = np.full((len(approved_sample), 3), 128, dtype=np.int16)
                        np.savetxt(
                            sample_file,
                            np.column_stack((approved_sample, sample_rgb)),
                            fmt=['%.4f', '%.4f', '%.4f', '%d', '%d', '%d'],
                        )
                    pipeline_source = approved_source
                    if eixos_preview is not None:
                        approved_pipeline_flags = [
                            '--sem-pilar',
                            '--sem-ceiling-detector',
                        ]
                    detalhes['nuvem_ifc'] = (
                        f'amostra comprimida: {len(approved_sample):,} de '
                        f'{s["npts"]:,} pontos')
                ok, log = passo('pipeline (lajes/paredes/aberturas)',
                                [sys.executable, str(C2B_DIR / 'rodar.py'), str(pipeline_source),
                                 '--thr', str(thr), '--min-wall-len', str(min_len),
                                 '--saida', str(outdir), '--sem-escada',
                                 *approved_pipeline_flags],
                                3600, env=pipeline_env)
                if not ok:
                    raise RuntimeError(f'pipeline: {log}')
                detalhes['pipeline'] = 'ok'
                ifc = next(outdir.glob('*_cloud2bim.ifc'))
                atual = ifc

                # ---- telhado curvo/inclinado: reivindica a zona acima da
                # ultima laje e apara paredes no beiral (sem isso, telhado
                # inclinado vira escadinha de paredes = "serrilhado") ----
                try:
                    import ifcopenshell
                    import ifcopenshell.util.placement as _P
                    mm = ifcopenshell.open(str(atual))
                    tops = []
                    for sl in mm.by_type('IfcSlab'):
                        z0 = _P.get_local_placement(sl.ObjectPlacement)[2, 3]
                        depths = []
                        for rr in sl.Representation.Representations:
                            for it in rr.Items:
                                if it.is_a('IfcExtrudedAreaSolid'):
                                    depths.append(float(it.Depth))
                        if depths:
                            tops.append(float(z0 + max(depths)))
                    # so aciona em multi-pavimento: em laje plana (galpao) nao
                    # ha zona de telhado curvo, acharia falso positivo
                    zmin_telhado = sorted(tops)[-2] + 0.1 if len(tops) >= 3 else None
                    del mm
                except Exception:
                    zmin_telhado = None
                if zmin_telhado is not None:
                    com_telhado = outdir / 'com_telhado.ifc'
                    ok, log = passo('telhado',
                                    [sys.executable,
                                     str(C2B_DIR / 'prototipos' / 'detect_casca_curva.py'),
                                     str(outdir / 'nuvem.xyz'), str(atual),
                                     str(com_telhado), '--zmin', str(zmin_telhado)],
                                    1800)
                    if ok and com_telhado.exists():
                        atual = com_telhado
                        detalhes['telhado'] = 'ok'
                    else:
                        detalhes['telhado'] = f'pulado: {log[-150:]}'
                else:
                    detalhes['telhado'] = 'pulado: pavimento único (telhado plano)'

                # ---- escadas: falha aqui NAO pode ser silenciosa ----
                pronto = outdir / 'pronto.ifc'
                if eixos_preview is not None:
                    shutil.copy(atual, pronto)
                    detalhes['escadas'] = 'pulado: geometria aprovada no editor'
                else:
                    ok, log = passo('escadas',
                                    [sys.executable,
                                     str(C2B_DIR / 'prototipos' / 'montar_escada_gabarito.py'),
                                     str(outdir / 'nuvem.xyz'), str(atual), str(pronto),
                                     '--cell', '0.12',
                                     '--area-min', str(stair_area_min)],
                                    1800)
                    if ok and pronto.exists():
                        detalhes['escadas'] = ('ok' if 'total: 0' not in log
                                               else 'nenhum lance detectado')
                    else:
                        shutil.copy(atual, pronto)
                        detalhes['escadas'] = f'falhou: {log[-150:]}'

                bake_types = ('IfcWall' if eixos_preview is not None
                              else 'IfcWall,IfcSlab')
                ok, log = passo('assando geometria (furos consumados na malha)',
                                [sys.executable,
                                 str(C2B_DIR / 'prototipos' / 'assar_geometria.py'),
                                 str(pronto), '--tipos', bake_types],
                                1800)
                detalhes['bake'] = 'ok' if ok else f'falhou: {log[-150:]}'

                _JOBS[jid]['etapa'] = 'validando IFC final'
                require_covering = bool(
                    config_preview
                    and (config_preview.get('forro') or {}).get('ativo', False)
                )
                expected_spaces = None
                if modelo_preview is not None:
                    approved_space_count = len(modelo_preview.get('spaces') or [])
                    # Zero no snapshot significa "derive das paredes", nao
                    # "fabrique um Space geral" e nem "espere exatamente zero".
                    if approved_space_count > 0:
                        expected_spaces = approved_space_count
                detalhes['validacao_ifc'] = _validar_ifc_download(
                    pronto,
                    expected_walls=(
                        len(eixos_preview) if eixos_preview is not None else None),
                    expected_openings=(
                        len(aberturas_preview)
                        if aberturas_preview is not None else None),
                    expected_spaces=expected_spaces,
                    require_covering=require_covering,
                )

                # Prova geométrica complementar: além da contagem bloqueante
                # acima, os eixos e espessuras precisam coincidir com a tela.
                if eixos_preview is not None:
                    _JOBS[jid]['etapa'] = 'auditando paredes (tela = IFC?)'
                    detalhes['paredes'] = _auditar_paredes_ifc(
                        pronto, eixos_preview)
                    if not detalhes['paredes'].startswith('ok'):
                        raise RuntimeError(detalhes['paredes'])

                nome = f"{jid}_{source_stem}_scan2bim.ifc"
                shutil.copy(pronto, output_folder / nome)
                _JOBS[jid].update(
                    status='pronto', etapa='concluido', url=f'/outputs/{nome}')
            except Exception as e:
                _JOBS[jid].update(status='erro', erro=str(e)[-400:])
            finally:
                import time as _t
                _JOBS[jid]['fim'] = _t.monotonic()
                # o IFC ja' esta' salvo em output_folder; a area de trabalho
                # (nuvem.xyz ~1,1 GB + intermediarios) nao serve mais pra nada
                try:
                    if outdir.exists():
                        shutil.rmtree(outdir, ignore_errors=True)
                        print(f'[gc] job {jid}: area de trabalho descartada')
                except Exception:
                    traceback.print_exc()

        # Cloud Run pode suspender CPU assim que a resposta HTTP termina e o
        # status vive na memoria desta instancia. Em producao, mantenha o job
        # dentro da requisicao; localmente o modo assincorno continua disponivel.
        if os.environ.get('SCAN_JOBS_SYNC', 'false').lower() in {'1', 'true', 'yes'}:
            _job()
        else:
            threading.Thread(target=_job, daemon=True).start()
        return jsonify({'job': jid})

    @app.route('/api/scan/job/<jid>')
    def _scan_job(jid):
        j = _JOBS.get(jid)
        if not j:
            return jsonify({'error': 'job desconhecido'}), 404
        return jsonify(j)

    threading.Thread(target=_gc_loop, daemon=True).start()
    print("🔍 Scan→BIM registrado: /api/scan/{upload,lajes,paredes,hibrido,gerar-ifc,job}")
    print(f"🧹 Descarte automático: nuvem enviada some após {SESSAO_TTL_S // 60} min "
          f"sem uso; área de trabalho do job some ao terminar (IFC fica).")
