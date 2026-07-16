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

_SESSOES = {}
_JOBS = {}
_LOCK = threading.Lock()
_DETECTOR_LOCK = threading.Lock()   # identify_walls usa env vars (processo-global)
Z_STEP = 0.15          # mesmo passo do identify_slabs (autodiag comparavel)


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
        result.append({'eixo_idx': eixo_idx, 'tipo': tipo,
                       's_centro': round(s_centro, 4), 'largura': round(largura, 4)})
    return result


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
        if (w.Description or '') != 'preview-locked':
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


def _autodiag(z, merge_dist=0.9):
    """Tabela thr -> (fatias, grupos de laje) + z dos grupos por thr."""
    z = np.round(z, 3)
    zmin, zmax = float(z.min()), float(z.max())
    n = int((zmax - zmin) / Z_STEP + 1)
    idx = np.clip(((z - zmin) / Z_STEP).astype(np.int64), 0, n - 1)
    cnt = np.bincount(idx, minlength=n).astype(np.int64)
    mx = int(cnt.max())

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


def register_scan(app, upload_folder, output_folder):
    upload_folder = Path(upload_folder)
    output_folder = Path(output_folder)

    # ------------------------------------------------------------------
    # 1) UPLOAD: fase cara (1x) — grade voxel + histograma de z
    # ------------------------------------------------------------------
    @app.route('/api/scan/upload', methods=['POST'])
    def _scan_upload():
        try:
            f = request.files.get('file')
            if not f:
                return jsonify({'error': 'arquivo ausente'}), 400
            sid = uuid.uuid4().hex[:12]
            path = upload_folder / f"{sid}_{secure_filename(f.filename)}"
            f.save(str(path))

            pts = _carregar_nuvem(path)
            ext = pts.max(0) - pts.min(0)

            # grade voxel: XY limitado a ~500k celulas, z a <=48 bins
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
            # amostra pra fase da escada (espelhos precisam de z fino, nao voxel)
            if len(pts) > 400_000:
                amostra = pts[np.random.default_rng(0).choice(len(pts), 400_000,
                                                              replace=False)]
            else:
                amostra = pts
            with _LOCK:
                _SESSOES[sid] = {
                    'path': str(path), 'occ': occ, 'cell': cell, 'zbin': zbin,
                    'xmin': xmin, 'ymin': ymin, 'zmin': zmin,
                    'npts': int(len(pts)), 'z_counts': zhist,
                    'grupos_para': grupos_para, 'amostra': amostra,
                }
            grupos = grupos_para(thr_sugerido)
            zh_min, zh_max, zh_cnt = zhist
            return jsonify({
                'sid': sid, 'n_pontos': int(len(pts)),
                'extent': [round(float(v), 2) for v in ext],
                'tabela_thr': tabela, 'thr_sugerido': thr_sugerido,
                'lajes': grupos,
                'z_hist': {'zmin': round(zh_min, 3), 'step': Z_STEP,
                           'counts': zh_cnt},
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    # ------------------------------------------------------------------
    # 2) LAJES: re-agrupa com outro threshold (instantaneo)
    # ------------------------------------------------------------------
    @app.route('/api/scan/lajes', methods=['POST'])
    def _scan_lajes():
        d = request.get_json(force=True)
        s = _SESSOES.get(d.get('sid'))
        if not s:
            return jsonify({'error': 'sessao expirada — refaça o upload'}), 404
        grupos = s['grupos_para'](float(d.get('thr', 0.3)))
        bandas = [[grupos[i][1], grupos[i + 1][0]] for i in range(len(grupos) - 1)]
        return jsonify({'lajes': grupos, 'bandas': bandas})

    # ------------------------------------------------------------------
    # 3) PAREDES: MOTOR REAL (identify_walls) na banda + camada de percepcao.
    #    Vermelho no front = eixos pareados = exatamente o que vira IfcWall.
    # ------------------------------------------------------------------
    @app.route('/api/scan/paredes', methods=['POST'])
    @_json_errors('preview de paredes')
    def _scan_paredes():
        import os
        d = request.get_json(force=True)
        s = _SESSOES.get(d.get('sid'))
        if not s:
            return jsonify({'error': 'sessao expirada — refaça o upload'}), 404
        zlo, zhi = float(d['zlo']), float(d['zhi'])
        zlo_frac = float(d.get('zlo_frac', 0.1))
        zhi_frac = float(d.get('zhi_frac', 0.9))
        min_len = float(d.get('min_len', 0.3))
        single_minlen = float(d.get('single_minlen', 1.0))
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
                         ('WALL_ZLO', 'WALL_ZHI', 'SINGLE_LINE', 'SINGLE_LINE_MINLEN',
                          'SINGLE_LINE_THK', 'WALL_CONTOURS')}
                os.environ.update({
                    'WALL_ZLO': str(zlo_frac), 'WALL_ZHI': str(zhi_frac),
                    'SINGLE_LINE': '1', 'SINGLE_LINE_MINLEN': str(single_minlen),
                    'SINGLE_LINE_THK': '0.15',
                    'WALL_CONTOURS': 'all' if contours_all else 'external',
                })
                try:
                    (starts, ends, esps, _mats, _grupos, labels) = _AF.identify_walls(
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
            import traceback
            traceback.print_exc()      # preview de eixos falhou; manda so percepcao

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
        s = _SESSOES.get(d.get('sid'))
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
    # 4) GERAR IFC: cadeia completa em background + poll
    # ------------------------------------------------------------------
    @app.route('/api/scan/gerar-ifc', methods=['POST'])
    def _scan_gerar():
        d = request.get_json(force=True)
        s = _SESSOES.get(d.get('sid'))
        if not s:
            return jsonify({'error': 'sessao expirada — refaça o upload'}), 404
        jid = uuid.uuid4().hex[:12]
        _JOBS[jid] = {'status': 'rodando', 'etapa': 'pipeline', 'url': None, 'erro': None}

        env_extra = {
            'WALL_ZLO': str(d.get('zlo_frac', 0.1)),
            'WALL_ZHI': str(d.get('zhi_frac', 0.9)),
            'SINGLE_LINE_MINLEN': str(d.get('single_minlen', 1.0)),
            'WALL_CONTOURS': 'all' if d.get('contours_all', True) else 'external',
        }
        thr = float(d.get('thr', 0.3))
        min_len = float(d.get('min_len', 0.3))
        banda_idx = max(0, int(d.get('banda_idx', 0)))
        eixos_preview = _normalizar_eixos_preview(d.get('eixos'))
        n_bandas = max(0, len(s['grupos_para'](thr)) - 1)
        if eixos_preview is not None and banda_idx >= n_bandas:
            return jsonify({'error': 'pavimento do preview nao existe nesta receita'}), 400
        # aberturas do editor (opcional): None = fluxo direto (redetecta),
        # lista (mesmo vazia) = editor mandou, vira a verdade das esquadrias
        try:
            aberturas_preview = _normalizar_aberturas_preview(
                d.get('aberturas'), len(eixos_preview) if eixos_preview else 0)
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400
        config_preview = d.get('config') if isinstance(d.get('config'), dict) else None
        src = Path(s['path'])

        def _job():
            import os
            detalhes = {}
            _JOBS[jid]['detalhes'] = detalhes

            def passo(nome, cmd, timeout, env=None):
                _JOBS[jid]['etapa'] = nome
                r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                   timeout=timeout)
                cauda = ((r.stdout or '') + (r.stderr or '')).strip()[-300:]
                return r.returncode == 0, cauda

            try:
                outdir = upload_folder / f"scanjob_{jid}"
                outdir.mkdir(parents=True, exist_ok=True)
                pipeline_env = {**os.environ, **env_extra}
                if eixos_preview is not None:
                    override_path = outdir / 'wall_overrides.json'
                    storey_override = {'eixos': eixos_preview}
                    if aberturas_preview is not None:
                        storey_override['aberturas'] = aberturas_preview
                        if config_preview:
                            storey_override['config'] = config_preview
                    override_path.write_text(json.dumps({
                        'storeys': {str(banda_idx): storey_override}
                    }, ensure_ascii=False), encoding='utf-8')
                    pipeline_env['WALL_OVERRIDE_FILE'] = str(override_path)
                    detalhes['paredes_preview'] = 'ok'
                    _ab_msg = ('' if aberturas_preview is None
                               else f', {len(aberturas_preview)} esquadrias do editor')
                    print(f'[scan {jid}] pavimento {banda_idx + 1}: '
                          f'{len(eixos_preview)} eixos travados pelo preview{_ab_msg}')
                ok, log = passo('pipeline (lajes/paredes/aberturas)',
                                [sys.executable, str(C2B_DIR / 'rodar.py'), str(src),
                                 '--thr', str(thr), '--min-wall-len', str(min_len),
                                 '--saida', str(outdir)],
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
                        for rr in sl.Representation.Representations:
                            for it in rr.Items:
                                if it.is_a('IfcExtrudedAreaSolid'):
                                    tops.append(float(z0 + it.Depth))
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
                ok, log = passo('escadas',
                                [sys.executable,
                                 str(C2B_DIR / 'prototipos' / 'montar_escada_gabarito.py'),
                                 str(outdir / 'nuvem.xyz'), str(atual), str(pronto)],
                                1800)
                if ok and pronto.exists():
                    detalhes['escadas'] = ('ok' if 'total: 0' not in log
                                           else 'nenhum lance detectado')
                else:
                    shutil.copy(atual, pronto)
                    detalhes['escadas'] = f'falhou: {log[-150:]}'

                ok, log = passo('assando geometria (furos consumados na malha)',
                                [sys.executable,
                                 str(C2B_DIR / 'prototipos' / 'assar_geometria.py'),
                                 str(pronto), '--tipos', 'IfcWall,IfcSlab'],
                                1800)
                detalhes['bake'] = 'ok' if ok else f'falhou: {log[-150:]}'

                # prova do contrato: o IFC final contem EXATAMENTE as paredes
                # aprovadas na tela (auditoria nunca derruba o job)
                if eixos_preview is not None:
                    _JOBS[jid]['etapa'] = 'auditando paredes (tela = IFC?)'
                    try:
                        detalhes['paredes'] = _auditar_paredes_ifc(
                            pronto, eixos_preview)
                    except Exception as audit_err:
                        detalhes['paredes'] = f'auditoria falhou: {audit_err}'

                nome = f"{jid}_{src.stem}_scan2bim.ifc"
                shutil.copy(pronto, output_folder / nome)
                _JOBS[jid].update(status='pronto', url=f'/outputs/{nome}')
            except Exception as e:
                _JOBS[jid].update(status='erro', erro=str(e)[-400:])

        threading.Thread(target=_job, daemon=True).start()
        return jsonify({'job': jid})

    @app.route('/api/scan/job/<jid>')
    def _scan_job(jid):
        j = _JOBS.get(jid)
        if not j:
            return jsonify({'error': 'job desconhecido'}), 404
        return jsonify(j)

    print("🔍 Scan→BIM registrado: /api/scan/{upload,lajes,paredes,gerar-ifc,job}")
