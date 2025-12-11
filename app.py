# -*- coding: utf-8 -*-
"""
BIM ANALYZER BACKEND - CLOUD RUN EDITION
========================================
API para processamento de IFC e Nuvens de Pontos (Z-Scan + Slicing).
Pronto para deploy no Google Cloud Run.
"""

import os
import shutil
import uuid
import traceback
import random
import numpy as np
import ifcopenshell
import ifcopenshell.geom
import open3d as o3d
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS  # <--- CRUCIAL PARA O AI STUDIO
from werkzeug.utils import secure_filename
from pathlib import Path
from itertools import permutations, product
from collections import Counter, defaultdict

# =========================
# CONFIGURAÇÃO
# =========================
TIPOS_INTERESSE = (
    "IfcWall", "IfcSlab", "IfcDoor", "IfcWindow", 
    "IfcColumn", "IfcBeam", "IfcStair", "IfcRoof", "IfcSanitaryTerminal"
)
MARGEM_PADRAO = 0.05      
MARGEM_PORTA_JAN = 0.10   
NUM_FATIAS_ALTURA = 10
DENSIDADE_MIN = 5.0
COBERTURA_COMPLETO = 0.60
COBERTURA_PARCIAL = 0.30
COBERTURA_INICIADO = 0.05

# Pastas Temporárias (Cloud Run usa /tmp para escrita)
BASE_DIR = Path("/tmp") if os.environ.get("K_SERVICE") else Path(".")
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"

# Limpa e recria pastas
if UPLOAD_FOLDER.exists(): shutil.rmtree(UPLOAD_FOLDER)
if OUTPUT_FOLDER.exists(): shutil.rmtree(OUTPUT_FOLDER)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# App Flask
app = Flask(__name__)
CORS(app)  # Permite que qualquer frontend (AI Studio) chame esta API
app.config['MAX_CONTENT_LENGTH'] = 2048 * 1024 * 1024 # 2GB

# =========================
# LÓGICA DE ALINHAMENTO E ANÁLISE
# =========================

def extrair_pavimentos(ifc_path):
    f = ifcopenshell.open(ifc_path)
    pavs = set()
    for s in f.by_type("IfcBuildingStorey"):
        if s.Name: pavs.add(s.Name)
    return sorted(list(pavs))

def extrair_objetos_por_pavimento(ifc_path, pavimento_alvo):
    f = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    objetos = []
    
    for product in f.by_type("IfcProduct"):
        if product.is_a() not in TIPOS_INTERESSE: continue
        eh_do_pavimento = False
        for rel in getattr(product, "ContainedInStructure", []):
            if rel.RelatingStructure.is_a("IfcBuildingStorey"):
                if rel.RelatingStructure.Name == pavimento_alvo: eh_do_pavimento = True; break
        if not eh_do_pavimento: continue

        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            verts = np.array(shape.geometry.verts).reshape(-1, 3)
            if verts.size == 0: continue
            xmin, xmax = float(verts[:,0].min()), float(verts[:,0].max())
            ymin, ymax = float(verts[:,1].min()), float(verts[:,1].max())
            zmin, zmax = float(verts[:,2].min()), float(verts[:,2].max())
            
            objetos.append({
                'guid': product.GlobalId, 'tipo': product.is_a(),
                'nome': getattr(product, 'Name', None) or product.GlobalId[:8],
                'pavimento': pavimento_alvo,
                'bbox': {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax},
                'volume_ifc': (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
            })
        except: continue
    return objetos

def _bounds_from_objs(objetos):
    if not objetos: return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    xs = [o['bbox']['xmin'] for o in objetos] + [o['bbox']['xmax'] for o in objetos]
    ys = [o['bbox']['ymin'] for o in objetos] + [o['bbox']['ymax'] for o in objetos]
    zs = [o['bbox']['zmin'] for o in objetos] + [o['bbox']['zmax'] for o in objetos]
    bmin = np.array([min(xs), min(ys), min(zs)], dtype=float)
    bmax = np.array([max(xs), max(ys), max(zs)], dtype=float)
    center = (bmin + bmax) / 2.0
    extent = bmax - bmin
    return bmin, bmax, center, extent

def filtrar_pontos_aabb(pts, bbox, margem):
    if pts.size == 0 or not bbox: return np.empty((0, 3), dtype=float)
    mask = (
        (pts[:,0] >= bbox['xmin']-margem) & (pts[:,0] <= bbox['xmax']+margem) &
        (pts[:,1] >= bbox['ymin']-margem) & (pts[:,1] <= bbox['ymax']+margem) &
        (pts[:,2] >= bbox['zmin']-margem) & (pts[:,2] <= bbox['zmax']+margem)
    )
    return pts[mask]

def alinhar_especifico_pavimento(pts, objetos_ifc):
    # Guias
    guia = [o for o in objetos_ifc if o['tipo'] in ('IfcWall', 'IfcColumn')]
    if not guia: guia = objetos_ifc
    
    # Salto Quântico (Centro com Centro)
    ifc_min, _, ifc_center, _ = _bounds_from_objs(guia)
    pts_min, pts_max = pts.min(axis=0), pts.max(axis=0)
    pts_center = (pts_min + pts_max) / 2.0
    delta_z_inicial = ifc_center[2] - pts_center[2]
    
    melhor = {'pontos': -1, 'pts': pts, 'transf': np.eye(4)}
    perms = list(permutations((0, 1, 2), 3))
    signs = list(product([-1, 1], repeat=3))
    z_range = np.arange(-2.0, 2.0, 0.20)

    for perm in perms:
        P = pts[:, perm]
        for sign in signs:
            if sign[2] == -1: continue # Trava gravidade
            S = P * np.array(sign, dtype=float)
            smin, smax = S.min(axis=0), S.max(axis=0)
            scenter = (smin + smax) / 2.0
            
            t_xy = guia_center - scenter
            t_xy[2] = 0
            delta_z_rot = ifc_center[2] - scenter[2]
            
            for z_off in z_range:
                t_atual = t_xy.copy()
                t_atual[2] = delta_z_rot + z_off
                pts_teste = S + t_atual
                
                score = 0
                for obj in guia:
                    b = obj['bbox']
                    m = 0.20
                    mask = (
                        (pts_teste[:,0]>=b['xmin']-m) & (pts_teste[:,0]<=b['xmax']+m) &
                        (pts_teste[:,1]>=b['ymin']-m) & (pts_teste[:,1]<=b['ymax']+m) &
                        (pts_teste[:,2]>=b['zmin']-m) & (pts_teste[:,2]<=b['zmax']+m)
                    )
                    score += np.count_nonzero(mask)
                
                if score > melhor['pontos']:
                    melhor = {'pontos': score, 'pts': pts_teste.copy(), 'z_used': delta_z_rot + z_off}

    # Fine Tuning XY
    pts_base = melhor['pts']
    best_fine_score = melhor['pontos']
    best_offset = np.zeros(3)
    
    for dx in np.linspace(-0.2, 0.2, 5):
        for dy in np.linspace(-0.2, 0.2, 5):
            off = np.array([dx, dy, 0])
            p_try = pts_base + off
            s = 0
            for obj in guia:
                b = obj['bbox']
                m = 0.05
                mask = (
                    (p_try[:,0]>=b['xmin']-m) & (p_try[:,0]<=b['xmax']+m) &
                    (p_try[:,1]>=b['ymin']-m) & (p_try[:,1]<=b['ymax']+m) &
                    (p_try[:,2]>=b['zmin']-m) & (p_try[:,2]<=b['zmax']+m)
                )
                s += np.count_nonzero(mask)
            if s > best_fine_score:
                best_fine_score = s
                best_offset = off
                
    return pts_base + best_offset, None

# Métricas e Classificação
def calcular_densidade_detalhada(pts, bbox):
    if len(pts) == 0: return {'num_pontos':0, 'cobertura_vertical':0.0}
    dz = bbox['zmax'] - bbox['zmin']
    h_slice = dz / NUM_FATIAS_ALTURA
    ok = 0
    for i in range(NUM_FATIAS_ALTURA):
        z0 = bbox['zmin'] + i*h_slice
        z1 = z0 + h_slice
        if np.count_nonzero((pts[:,2]>=z0) & (pts[:,2]<z1)) > 0: ok += 1
    return {'num_pontos': len(pts), 'cobertura_vertical': ok/NUM_FATIAS_ALTURA}

def calcular_densidade_horizontal(pts, bbox):
    if len(pts) == 0: return {'num_pontos':0, 'cobertura_vertical':0.0}
    dx, dy = bbox['xmax']-bbox['xmin'], bbox['ymax']-bbox['ymin']
    dens = len(pts) / (dx*dy) if (dx*dy) > 0 else 0
    return {'num_pontos': len(pts), 'densidade_global': dens, 'cobertura_vertical': 1.0 if dens > 2 else 0.0}

def calcular_dimensoes(pts, bbox):
    dim_plan = {'x': bbox['xmax']-bbox['xmin'], 'y': bbox['ymax']-bbox['ymin'], 'z': bbox['zmax']-bbox['zmin']}
    if len(pts) < 5: return {'progresso': {'z': 0}}
    dim_exec = {'z': pts[:,2].max() - pts[:,2].min()}
    prog = min(dim_exec['z']/dim_plan['z'], 1.2) if dim_plan['z']>0 else 0
    return {'progresso': {'z': round(prog*100, 1)}}

def classificar_status(dens, dimensoes):
    cob = dens['cobertura_vertical']
    alt = dimensoes['progresso']['z'] / 100.0
    if cob >= COBERTURA_COMPLETO or alt >= 0.90: return {'code':'COMPLETO', 'texto':'Completo', 'cor':'#4caf50'}
    if cob >= COBERTURA_PARCIAL or alt >= 0.50: return {'code':'PARCIAL', 'texto':'Parcial', 'cor':'#ff9800'}
    if cob >= COBERTURA_INICIADO or alt >= 0.10: return {'code':'INICIADO', 'texto':'Iniciado', 'cor':'#2196f3'}
    return {'code':'AUSENTE', 'texto':'Ausente', 'cor':'#f44336'}

def converter_ifc_three(bbox):
    return {'xmin':bbox['xmin'],'xmax':bbox['xmax'], 'ymin':bbox['zmin'],'ymax':bbox['zmax'], 'zmin':bbox['ymin'],'zmax':bbox['ymax']}
def converter_pts_three(pts):
    p = pts.copy()
    p[:, [1, 2]] = p[:, [2, 1]]
    return p

# =========================
# ROTAS API
# =========================

@app.route('/')
def health_check():
    return "BIM Analyzer Backend is Running! 🚀", 200

@app.route('/api/listar_pavimentos', methods=['POST'])
def listar_pavimentos():
    try:
        f = request.files['ifc_file']
        path = UPLOAD_FOLDER / f"{uuid.uuid4()}_{secure_filename(f.filename)}"
        f.save(str(path))
        pavs = extrair_pavimentos(str(path))
        return jsonify({'pavimentos': pavs, 'temp_id': str(path)}) # Retorna path para reuso
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analisar_pavimento', methods=['POST'])
def analisar_pavimento():
    try:
        # Recupera IFC (re-upload ou path temporário se o frontend suportar)
        ifc_file = request.files.get('ifc_file')
        ply_file = request.files.get('ply_file')
        pav_alvo = request.form.get('pavimento')
        
        if not ifc_file or not ply_file or not pav_alvo: 
            return jsonify({'error': 'Arquivos faltando'}), 400

        # Salva com ID único para evitar colisão
        session_id = str(uuid.uuid4())
        ifc_path = UPLOAD_FOLDER / f"{session_id}_ifc.ifc"
        ply_path = UPLOAD_FOLDER / f"{session_id}_ply.ply"
        ifc_file.save(str(ifc_path))
        ply_file.save(str(ply_path))
        
        # Processamento
        objetos = extrair_objetos_por_pavimento(str(ifc_path), pav_alvo)
        if not objetos: return jsonify({'error': f'Pavimento {pav_alvo} vazio'}), 400
        
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points, dtype=float)
        
        pts_alinhados, _ = alinhar_especifico_pavimento(pts, objetos)
        
        # Resultados
        out_subdir = OUTPUT_FOLDER / session_id
        out_subdir.mkdir(exist_ok=True)
        resultados = []
        
        _, _, center, _ = _bounds_from_objs(objetos)
        global_offset = np.array([center[0], center[1], objetos[0]['bbox']['zmin']])
        
        for obj in objetos:
            m = MARGEM_PORTA_JAN if obj['tipo'] in ('IfcDoor','IfcWindow') else MARGEM_PADRAO
            pts_obj = filtrar_pontos_aabb(pts_alinhados, obj['bbox'], m)
            
            if obj['tipo'] in ('IfcSlab', 'IfcRoof'): dens = calcular_densidade_horizontal(pts_obj, obj['bbox'])
            else: dens = calcular_densidade_detalhada(pts_obj, obj['bbox'])
            
            dim = calcular_dimensoes(pts_obj, obj['bbox'])
            status = classificar_status(dens, dim)
            
            json_filename = f"{secure_filename(obj['nome'])}.json"
            
            if len(pts_obj) > 0:
                pts_view = pts_obj - global_offset
                cor = [0.2,0.8,0.2] if status['code']=='COMPLETO' else [1,0.6,0] if status['code']=='PARCIAL' else [0.2,0.6,1] if status['code']=='INICIADO' else [0.8,0.2,0.2]
                pts_three = converter_pts_three(pts_view)
                
                with open(out_subdir / json_filename, 'w') as f:
                    json.dump({'positions': pts_three.flatten().tolist(), 'color': cor}, f)
            
            b = obj['bbox'].copy()
            b['xmin']-=global_offset[0]; b['xmax']-=global_offset[0]
            b['ymin']-=global_offset[1]; b['ymax']-=global_offset[1]
            b['zmin']-=global_offset[2]; b['zmax']-=global_offset[2]
            
            resultados.append({
                'nome': obj['nome'], 'tipo': obj['tipo'], 'pontos': dens['num_pontos'],
                'status': status, 'dimensoes': dim,
                'json_url': f"/outputs/{session_id}/{json_filename}" if len(pts_obj)>0 else None,
                'bbox_normalized': converter_ifc_three(b)
            })
            
        return jsonify({'resultados': resultados})
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/outputs/<path:filepath>')
def download_file(filepath):
    return send_from_directory(OUTPUT_FOLDER, filepath)

if __name__ == '__main__':
    # Cloud Run injeta a porta na variável PORT
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)