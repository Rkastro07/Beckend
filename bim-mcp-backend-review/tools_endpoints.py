# -*- coding: utf-8 -*-
"""
Endpoints das FERRAMENTAS do produto (conversores + geradores de nuvem).

Registrados no app_obb via `register_tools(app, UPLOAD_FOLDER, OUTPUT_FOLDER,
_valid_upload)`. Reutiliza o codigo ja existente dos scripts CLI em vez de
duplicar logica:
  - usdz_to_ply.py            -> OBJ->PLY e USDZ->PLY
  - experiments/sonata/asc_to_ply_subsample.py -> ASC->PLY (subsample)
  - dataset/gerar_sintetico.py -> IFC->nuvem sintetica (estagios de obra)

Cada endpoint salva o resultado em OUTPUT_FOLDER e devolve JSON com a URL de
download (/outputs/<arquivo>), no mesmo padrao dos endpoints de analise.
"""
import sys
import io
import uuid
import json
import contextlib
import traceback
from pathlib import Path

import numpy as np
import open3d as o3d
from flask import request, jsonify
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent


def _escrever_ply(pts, path, cores=None):
    """Array (N,3) -> PLY binario. cores opcional (N,3) uint8/0-255."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    if cores is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(cores, dtype=np.float64) / 255.0)
    o3d.io.write_point_cloud(str(path), pcd)


def register_tools(app, upload_folder, output_folder, valid_upload):
    upload_folder = Path(upload_folder)
    output_folder = Path(output_folder)

    # -------------------------------------------------------------------
    # CONVERSOR 1: OBJ -> PLY (sampling da malha)
    # -------------------------------------------------------------------
    @app.route('/api/tools/obj-to-ply', methods=['POST'])
    def _obj_to_ply():
        try:
            f = request.files.get('file')
            if not f or not valid_upload(f, ('.obj',)):
                return jsonify({'error': 'Envie um arquivo .obj'}), 400
            densidade = int(float(request.form.get('densidade', 130)))

            sid = uuid.uuid4().hex[:8]
            obj_path = upload_folder / f"{sid}_{secure_filename(f.filename)}"
            f.save(str(obj_path))
            ply_name = f"{sid}_{Path(f.filename).stem}.ply"
            ply_path = output_folder / ply_name

            from usdz_to_ply import obj_para_ply
            obj_para_ply(obj_path, ply_path, densidade)

            n = len(o3d.io.read_point_cloud(str(ply_path)).points)
            return jsonify({'ok': True, 'download_url': f'/outputs/{ply_name}',
                            'n_pontos': n, 'densidade': densidade})
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Falha na conversao OBJ->PLY: {e}'}), 500

    # -------------------------------------------------------------------
    # CONVERSOR 2: USDZ -> PLY (RoomPlan do iPhone)
    # -------------------------------------------------------------------
    @app.route('/api/tools/usdz-to-ply', methods=['POST'])
    def _usdz_to_ply():
        try:
            f = request.files.get('file')
            if not f or not valid_upload(f, ('.usdz',)):
                return jsonify({'error': 'Envie um arquivo .usdz'}), 400
            densidade = int(float(request.form.get('densidade', 130)))
            so_estrutura = request.form.get('apenas_estrutura', 'true').lower() == 'true'

            sid = uuid.uuid4().hex[:8]
            usdz_path = upload_folder / f"{sid}_{secure_filename(f.filename)}"
            f.save(str(usdz_path))
            stem = Path(f.filename).stem
            obj_path = upload_folder / f"{sid}_{stem}.obj"
            ply_name = f"{sid}_{stem}.ply"
            ply_path = output_folder / ply_name

            from pxr import Usd
            from usdz_to_ply import coletar_meshes, salvar_obj, obj_para_ply
            stage = Usd.Stage.Open(str(usdz_path))
            if stage is None:
                return jsonify({'error': 'Falha ao abrir o USDZ'}), 400
            meshes = coletar_meshes(stage, apenas_estrutura=so_estrutura)
            if not meshes and so_estrutura:
                meshes = coletar_meshes(stage, apenas_estrutura=False)
            if not meshes:
                return jsonify({'error': 'Nenhuma malha encontrada no USDZ'}), 400
            salvar_obj(meshes, obj_path)
            obj_para_ply(obj_path, ply_path, densidade)

            n = len(o3d.io.read_point_cloud(str(ply_path)).points)
            return jsonify({'ok': True, 'download_url': f'/outputs/{ply_name}',
                            'n_pontos': n, 'n_meshes': len(meshes),
                            'apenas_estrutura': so_estrutura})
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Falha na conversao USDZ->PLY: {e}'}), 500

    # -------------------------------------------------------------------
    # CONVERSOR 3: ASC -> PLY (nuvem de scanner, com subsample)
    # -------------------------------------------------------------------
    @app.route('/api/tools/asc-to-ply', methods=['POST'])
    def _asc_to_ply():
        try:
            f = request.files.get('file')
            if not f or not valid_upload(f, ('.asc', '.txt', '.xyz', '.zip')):
                return jsonify({'error': 'Envie um arquivo .asc/.xyz/.txt/.zip'}), 400
            subsample = max(1, int(float(request.form.get('subsample', 20))))

            sid = uuid.uuid4().hex[:8]
            src = upload_folder / f"{sid}_{secure_filename(f.filename)}"
            f.save(str(src))
            ply_name = f"{sid}_{Path(f.filename).stem}.ply"
            ply_path = output_folder / ply_name

            sys.path.insert(0, str(BASE_DIR / 'experiments' / 'sonata'))
            from asc_to_ply_subsample import stream_convert
            # o script printa progresso com emoji -> quebra no console cp1252 do
            # Windows; suprime o stdout dele durante a conversao
            with contextlib.redirect_stdout(io.StringIO()):
                if src.suffix.lower() == '.zip':
                    import zipfile
                    zf = zipfile.ZipFile(str(src))
                    inner = next((n for n in zf.namelist()
                                  if n.lower().endswith(('.asc', '.xyz', '.txt'))), None)
                    if not inner:
                        return jsonify({'error': 'ZIP sem .asc/.xyz/.txt dentro'}), 400
                    with zf.open(inner) as stream:
                        stream_convert(stream, str(ply_path), subsample_rate=subsample)
                else:
                    with open(src, 'rb') as stream:
                        stream_convert(stream, str(ply_path), subsample_rate=subsample)

            n = len(o3d.io.read_point_cloud(str(ply_path)).points)
            return jsonify({'ok': True, 'download_url': f'/outputs/{ply_name}',
                            'n_pontos': n, 'subsample': subsample})
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Falha na conversao ASC->PLY: {e}'}), 500

    # -------------------------------------------------------------------
    # GERADOR: IFC -> nuvem sintetica (estagio de obra)
    #   modo 'estagio'  -> perfil pronto v00..v09 (ou aleatorio)
    #   modo 'manual'   -> usuario define % ausente e % parcial
    # -------------------------------------------------------------------
    @app.route('/api/tools/gerar-nuvem', methods=['POST'])
    def _gerar_nuvem():
        try:
            f = request.files.get('file')
            if not f or not valid_upload(f, ('.ifc',)):
                return jsonify({'error': 'Envie um arquivo .ifc'}), 400

            modo = request.form.get('modo', 'estagio')
            seed = int(float(request.form.get('seed', 0)))

            sys.path.insert(0, str(BASE_DIR / 'dataset'))
            from gerar_sintetico import (extrair_objetos, gerar_variante,
                                         PERFIS_VARIANTES)
            import ifcopenshell

            if modo == 'manual':
                pa = float(request.form.get('pct_ausente', 0.3))
                pp = float(request.form.get('pct_parcial', 0.15))
                frac_ausente, frac_parcial = (pa, pa), (pp, pp)
                rotulo = f"manual_a{int(pa*100)}_p{int(pp*100)}"
            else:  # estagio
                est = request.form.get('estagio', 'aleatorio')
                if est == 'aleatorio':
                    import random
                    est = random.Random(seed).randrange(len(PERFIS_VARIANTES))
                else:
                    est = max(0, min(len(PERFIS_VARIANTES) - 1, int(est)))
                frac_ausente, frac_parcial = PERFIS_VARIANTES[est]
                rotulo = f"estagio_v{est:02d}"

            sid = uuid.uuid4().hex[:8]
            ifc_path = upload_folder / f"{sid}_{secure_filename(f.filename)}"
            f.save(str(ifc_path))
            ifc_model = ifcopenshell.open(str(ifc_path))
            objetos = extrair_objetos(ifc_model)
            if not objetos:
                return jsonify({'error': 'IFC sem objetos com geometria valida'}), 400

            nuvens, labels = gerar_variante(objetos, frac_ausente, frac_parcial, seed)
            if not nuvens:
                return jsonify({'error': 'Nenhum ponto gerado (tudo ausente?)'}), 400
            pts = np.vstack(nuvens)

            stem = Path(f.filename).stem
            ply_name = f"{sid}_{stem}_{rotulo}.ply"
            lbl_name = f"{sid}_{stem}_{rotulo}.labels.json"
            _escrever_ply(pts, output_folder / ply_name)
            (output_folder / lbl_name).write_text(json.dumps(labels, indent=2))

            from collections import Counter
            st = Counter(v['status'] for v in labels.values())
            return jsonify({
                'ok': True,
                'download_url': f'/outputs/{ply_name}',
                'labels_url': f'/outputs/{lbl_name}',
                'n_pontos': int(len(pts)),
                'modo': modo, 'rotulo': rotulo,
                'stats': {'completo': st.get('COMPLETO', 0),
                          'parcial': st.get('PARCIAL', 0),
                          'ausente': st.get('AUSENTE', 0),
                          'total_objetos': len(labels)},
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Falha ao gerar nuvem: {e}'}), 500

    # -------------------------------------------------------------------
    # PLANTA -> BIM: importacao geometrica unificada + geracao IFC
    # -------------------------------------------------------------------
    def _import_planta():
        sys.path.insert(0, str(BASE_DIR / 'plantatobim'))
        import planta_to_ifc_v1 as pl
        return pl

    def _import_geometry_importers():
        sys.path.insert(0, str(BASE_DIR / 'plantatobim'))
        import geometry_importers as gi
        return gi

    @app.route('/api/planta/formatos', methods=['GET'])
    def _planta_formatos():
        """Catalogo de capacidades; pode virar um resource do servidor MCP."""
        gi = _import_geometry_importers()
        return jsonify({'ok': True, **gi.format_capabilities()})

    @app.route('/api/planta/importar', methods=['POST'])
    @app.route('/api/planta/parse', methods=['POST'])
    def _planta_parse():
        """IFC/IFCZIP/DXF/SVG -> modelo editavel (JSON).

        ``/api/planta/parse`` continua como alias para clientes antigos.
        """
        try:
            f = request.files.get('file')
            gi = _import_geometry_importers()
            known_extensions = tuple(gi.FORMAT_CAPABILITIES)
            if not f or not valid_upload(f, known_extensions):
                return jsonify({
                    'error': 'Envie um arquivo geometrico reconhecido.',
                    **gi.format_capabilities(),
                }), 400
            ext = Path(f.filename).suffix.lower()
            if ext not in gi.DIRECT_EDIT_FORMATS:
                capability = gi.FORMAT_CAPABILITIES.get(ext)
                return jsonify({
                    'error': (
                        f"{ext} tem geometria, mas deve seguir pela rota "
                        f"'{capability['rota']}'."
                    ),
                    'formato': ext,
                    'capability': capability,
                }), 422
            escala = request.form.get('escala')
            escala = float(escala) if escala else None
            esp_default = float(request.form.get('esp_default', 0.15))
            pavimento = request.form.get('pavimento') or None

            sid = uuid.uuid4().hex[:8]
            source_path = upload_folder / f"{sid}_{secure_filename(f.filename)}"
            f.save(str(source_path))

            pl = _import_planta()
            with contextlib.redirect_stdout(io.StringIO()):
                modelo = gi.importar_geometria(
                    source_path,
                    escala_forcada=escala,
                    esp_default=esp_default,
                    pavimento=pavimento,
                )
                d = pl.modelo_para_dict(modelo)
            d['ok'] = True
            d['nome'] = Path(f.filename).stem
            return jsonify(d)
        except gi.GeometryImportError as e:
            ext = Path(f.filename).suffix.lower() if f else ''
            return jsonify({
                'error': str(e),
                'formato': ext,
                'capability': gi.FORMAT_CAPABILITIES.get(ext),
            }), 422
        except SystemExit as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Falha ao importar geometria: {e}'}), 500

    @app.route('/api/planta/gerar', methods=['POST'])
    def _planta_gerar():
        """modelo (editado no front) -> IFC + preview PLY. Body JSON:
        {modelo: {paredes, aberturas}, config: {altura, ...}, nome: str}."""
        try:
            body = request.get_json(force=True, silent=True) or {}
            modelo_dict = body.get('modelo') or body
            config = body.get('config', {})
            nome = secure_filename(body.get('nome', 'planta')) or 'planta'
            if not modelo_dict.get('paredes'):
                return jsonify({'error': 'Modelo sem paredes'}), 400

            pl = _import_planta()
            interno = pl.dict_para_modelo(modelo_dict)
            if not interno['paredes']:
                return jsonify({'error': 'Nenhuma parede valida no modelo'}), 400

            sid = uuid.uuid4().hex[:8]
            ifc_name = f"{sid}_{nome}.ifc"
            ply_name = f"{sid}_{nome}_preview.ply"
            ifc_path = output_folder / ifc_name
            ply_path = output_folder / ply_name

            with contextlib.redirect_stdout(io.StringIO()):
                pl.gerar_ifc_do_modelo(interno['paredes'], interno['aberturas'],
                                       ifc_path, config, laje=interno.get('laje'))
                pl.ifc_para_ply(ifc_path, ply_path)

            return jsonify({
                'ok': True,
                'ifc_url': f'/outputs/{ifc_name}',
                'preview_url': f'/outputs/{ply_name}',
                'n_paredes': len(interno['paredes']),
                'n_aberturas': len(interno['aberturas']),
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Falha ao gerar IFC: {e}'}), 500

    return app
