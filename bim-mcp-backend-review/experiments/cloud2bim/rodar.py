#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner unico scan -> IFC (Cloud2BIM adaptado + detectores de pilar e escada).

    python rodar.py <nuvem.ply|.e57|.xyz> [--thr 0.3] [--saida DIR]
                    [--sem-pilar] [--sem-escada]

Faz, em sequencia:
  1. converte a nuvem pra .xyz (formato do Cloud2BIM)
  2. AUTODIAGNOSTICO do threshold de laje (o SLAB_THR que era calibrado na mao):
     testa varios valores e escolhe o que revela mais niveis de laje distintos
  3. gera o config .yaml e roda o pipeline (lajes -> pavimentos -> paredes ->
     aberturas -> comodos -> IFC) com a receita validada:
     projecao de altura cheia + resgate single-line (vidro/scan interno)
  4. roda o detector de pilar por pavimento e insere IfcColumn no IFC
  5. detecta lances, monta o gabarito e insere IfcStair no IFC
  6. renderiza planta por pavimento (paredes vermelho, pilares azul) pra conferencia

Saida: <pasta_da_nuvem>/<nome>_cloud2bim/  (ifc, pngs, config, logs)

Limitacoes conhecidas (ver README.md): laje CURVA vira fatia plana arbitraria,
viga nao detectada.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
C2B = PKG / 'cloud2bim_patched'
sys.path.insert(0, str(PKG))
from detect_pilar import find_pillars  # noqa: E402


# ---------------------------------------------------------------- entrada
def carregar_nuvem(path: Path):
    suf = path.suffix.lower()
    if suf == '.ply':
        import open3d as o3d
        p = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(p.points)
        cols = np.asarray(p.colors)
        rgb = (cols * 255).round().astype(int) if len(cols) else np.full((len(pts), 3), 128, int)
    elif suf == '.e57':
        import e57
        d = e57.read_points(str(path))
        pts = np.asarray(d.points)
        c = None
        try:
            c = np.asarray(d.color)
        except Exception:
            pass
        rgb = (c * 255).round().astype(int) if c is not None and len(c) else np.full((len(pts), 3), 128, int)
    elif suf == '.xyz':
        arr = np.loadtxt(path, skiprows=1)
        pts = arr[:, :3]
        rgb = arr[:, 3:6].astype(int) if arr.shape[1] >= 6 else np.full((len(pts), 3), 128, int)
    else:
        sys.exit(f'[X] formato nao suportado: {suf} (use .ply, .e57 ou .xyz)')
    if len(pts) == 0:
        sys.exit('[X] nuvem vazia')
    return pts, rgb


def escrever_xyz(pts, rgb, dst: Path):
    with open(dst, 'w') as f:
        f.write('x y z r g b\n')
        np.savetxt(f, np.hstack([pts, rgb]), fmt=['%.4f', '%.4f', '%.4f', '%d', '%d', '%d'])


# ------------------------------------------------- autodiagnostico do SLAB_THR
def autodiag_slab_thr(z, step=0.15, merge_dist=0.9):
    """Replica o histograma do identify_slabs (borda inclusiva, round 3) e testa
    thresholds decrescentes. Escolhe o que revela MAIS grupos de laje distintos
    (empate -> threshold mais alto, mais conservador)."""
    z = np.round(z, 3)
    zmin, zmax = z.min(), z.max()
    n = int((zmax - zmin) / step + 1)
    idx = np.clip(((z - zmin) / step).astype(np.int64), 0, n - 1)
    cnt = np.bincount(idx, minlength=n)
    mx = cnt.max()

    tabela = []
    for thr in (0.5, 0.4, 0.3, 0.25, 0.2, 0.15):
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
        grupos, last_end = 0, None
        for a, b in intervals:
            za = zmin + a * step
            zb = zmin + b * step + step
            if last_end is None or za - last_end >= merge_dist:
                grupos += 1
            last_end = zb
        tabela.append((thr, len(intervals), grupos))

    validos = [t for t in tabela if t[2] >= 2]
    if validos:
        melhor = max(validos, key=lambda t: (t[2], t[0]))
    else:
        melhor = tabela[-1]
        print('[!] nenhum threshold achou 2+ niveis de laje — usando o mais baixo; confira o resultado')
    print('    thr   fatias  grupos-de-laje')
    for thr, ni, g in tabela:
        marca = '  <-- escolhido' if (thr, ni, g) == melhor else ''
        print(f'    {thr:<5} {ni:^7} {g:^6}{marca}')
    return melhor[0]


# ------------------------------------------------------------------- config
CONFIG_TEMPLATE = """e57_input: false
xyz_files:
  - {xyz}
exterior_scan: false
dilute: false
dilution_factor: 10
pc_resolution: {pc_res}
grid_coefficient: {grid_coef}

bfs_thickness: 0.3
tfs_thickness: 0.3

min_wall_length: {min_wall_len}
min_wall_thickness: 0.05
max_wall_thickness: 0.75
exterior_walls_thickness: 0.3

output_ifc: {ifc}
ifc_project_name: {nome}
ifc_project_long_name: Scan-to-BIM automatico ({nome})
ifc_project_version: v1.0

ifc_author_name: Rafael
ifc_author_surname: Corrigliano
ifc_author_organization: Beckend

ifc_building_name: {nome}
ifc_building_type: Building
ifc_building_phase: As-built

ifc_site_latitude: [0, 0, 0]
ifc_site_longitude: [0, 0, 0]
ifc_site_elevation: 0.0
material_for_objects: Concrete
"""


def estimar_pc_resolution(pts):
    """Mediana da distancia ao vizinho mais proximo, numa amostra (rapido)."""
    import open3d as o3d
    amostra = pts if len(pts) <= 200_000 else pts[np.random.default_rng(0).choice(len(pts), 200_000, replace=False)]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(amostra)
    d = np.asarray(pc.compute_nearest_neighbor_distance())
    res = float(np.clip(np.median(d), 0.005, 0.05))
    return round(res, 3)


# --------------------------------------------------------- pilares por pavimento
def bandas_dos_pavimentos(ifc_path):
    """Le as lajes do IFC gerado e devolve [(z_topo_laje_i, z_fundo_laje_i+1)] + storeys."""
    import ifcopenshell
    import ifcopenshell.util.placement as P
    m = ifcopenshell.open(str(ifc_path))
    lajes = []
    for s in m.by_type('IfcSlab'):
        z0 = P.get_local_placement(s.ObjectPlacement)[2, 3]
        depths = []
        for r in s.Representation.Representations:
            for it in r.Items:
                if it.is_a('IfcExtrudedAreaSolid'):
                    depths.append(float(it.Depth))
        if depths:
            # Uma IfcSlab pode ter varios solidos XY desconectados, mas todos
            # pertencem ao mesmo nivel e nao criam pavimentos adicionais.
            lajes.append((float(z0), float(z0 + max(depths))))
    lajes.sort()
    bandas = [(lajes[i][1], lajes[i + 1][0]) for i in range(len(lajes) - 1)]
    storeys = sorted(m.by_type('IfcBuildingStorey'),
                     key=lambda s: P.get_local_placement(s.ObjectPlacement)[2, 3])
    return bandas, storeys, m


def inserir_pilares(ifc_path, pts):
    """Roda find_pillars por pavimento e insere IfcColumn no IFC. Retorna lista."""
    import ifcopenshell
    bandas, storeys, m = bandas_dos_pavimentos(ifc_path)
    owner = m.by_type('IfcOwnerHistory')[0]
    ctx = m.by_type('IfcGeometricRepresentationContext')[0]
    zdir = m.create_entity('IfcDirection', DirectionRatios=(0.0, 0.0, 1.0))
    rel = m.by_type('IfcRelContainedInSpatialStructure')
    todos = []
    cid = 0
    for k, (zlo, zhi) in enumerate(bandas):
        if zhi - zlo < 1.0:
            continue
        achados = [p for p in find_pillars(pts, zlo + 0.05, zhi - 0.05) if p['ok']]
        for p in achados:
            cid += 1
            profile = m.create_entity('IfcRectangleProfileDef', ProfileType='AREA',
                                      ProfileName=f'Pilar {p["w"]*100:.0f}x{p["h"]*100:.0f}',
                                      XDim=float(p['w']), YDim=float(p['h']))
            ax = m.create_entity('IfcAxis2Placement3D',
                                 Location=m.create_entity('IfcCartesianPoint', Coordinates=(0.0, 0.0, 0.0)))
            solid = m.create_entity('IfcExtrudedAreaSolid', SweptArea=profile, Position=ax,
                                    ExtrudedDirection=zdir, Depth=float(zhi - zlo))
            shape = m.create_entity('IfcShapeRepresentation', ContextOfItems=ctx,
                                    RepresentationIdentifier='Body',
                                    RepresentationType='SweptSolid', Items=[solid])
            pds = m.create_entity('IfcProductDefinitionShape', Representations=[shape])
            lp = m.create_entity('IfcLocalPlacement', RelativePlacement=m.create_entity(
                'IfcAxis2Placement3D',
                Location=m.create_entity('IfcCartesianPoint',
                                         Coordinates=(float(p['cx']), float(p['cy']), float(zlo)))))
            col = m.create_entity('IfcColumn', GlobalId=ifcopenshell.guid.new(),
                                  OwnerHistory=owner, Name=f'C{cid:02d}',
                                  Description='Detectado por perfil de ocupacao vertical',
                                  ObjectPlacement=lp, Representation=pds)
            if rel:
                r = rel[0]
                r.RelatedElements = list(r.RelatedElements) + [col]
            p['banda'] = k
            todos.append(p)
    if todos:
        m.write(str(ifc_path))
    return todos


# ----------------------------------------------------------------- render
def render_pavimentos(pts, ifc_path, out_png, pilares):
    import ifcopenshell
    import ifcopenshell.util.element as E
    import ifcopenshell.util.placement as P
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    bandas, storeys, m = bandas_dos_pavimentos(ifc_path)

    def geometry_of(w):
        mat = P.get_local_placement(w.ObjectPlacement)
        axis = None
        thickness = None
        for r in w.Representation.Representations:
            if r.RepresentationIdentifier == 'Axis':
                for it in r.Items:
                    if it.is_a('IfcPolyline'):
                        axis = np.array([
                            (mat @ np.array(list(pp.Coordinates)[:2] + [0, 1.0]))[:2]
                            for pp in it.Points])
            elif r.RepresentationIdentifier == 'Body':
                for it in r.Items:
                    if it.is_a('IfcExtrudedAreaSolid') and \
                            it.SweptArea.is_a('IfcRectangleProfileDef'):
                        thickness = float(it.SweptArea.YDim)
        return axis, thickness

    def footprint(axis, thickness):
        vector = axis[1] - axis[0]
        length = np.linalg.norm(vector)
        if length <= 1e-9 or thickness is None:
            return None
        normal = np.array([-vector[1], vector[0]]) / length * (thickness / 2)
        return np.array([axis[0] + normal, axis[1] + normal,
                         axis[1] - normal, axis[0] - normal])

    storey_walls = {}
    for rel in m.by_type('IfcRelContainedInSpatialStructure'):
        nome = rel.RelatingStructure.Name
        for el in rel.RelatedElements:
            if el.is_a('IfcWall'):
                a, thickness = geometry_of(el)
                if a is not None:
                    diagnostics = E.get_psets(el).get(
                        'Pset_Cloud2BIM_WallDiagnostics', {})
                    reference = (
                        diagnostics.get('Reference')
                        or el.Tag or el.Name or 'W-?')
                    confidence = str(
                        diagnostics.get('Confidence', 'UNSCORED')).upper()
                    storey_walls.setdefault(nome, []).append(
                        (a, thickness, reference, confidence))

    names = [s.Name for s in storeys[:len(bandas)]]
    ncol = min(3, max(1, len(bandas)))
    nrow = max(1, int(np.ceil(len(bandas) / ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.3 * nrow), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for k, (band, nm) in enumerate(zip(bandas, names)):
        ax = axes[k // ncol, k % ncol]
        sl = pts[(pts[:, 2] >= band[0] + 0.25) & (pts[:, 2] < band[1] - 0.05)]
        if len(sl) > 250_000:
            sl = sl[np.random.default_rng(0).choice(len(sl), 250_000, replace=False)]
        ax.scatter(sl[:, 0], sl[:, 1], s=0.25, c='0.75', linewidths=0)
        confidence_colours = {
            'HIGH': '#2e7d32',
            'MEDIUM': '#f9a825',
            'LOW': '#c62828',
            'REVIEWED': '#1565c0',
            'UNSCORED': '#616161',
        }
        used_confidences = set()
        label_boxes = []
        for a, thickness, reference, confidence in storey_walls.get(nm, []):
            colour = confidence_colours.get(confidence, '#616161')
            used_confidences.add(
                confidence if confidence in confidence_colours
                else 'UNSCORED')
            polygon = footprint(a, thickness)
            if polygon is not None:
                ax.add_patch(Polygon(
                    polygon, closed=True, facecolor=colour, edgecolor=colour,
                    alpha=0.20, linewidth=0.9, zorder=2))
            ax.plot(a[:, 0], a[:, 1], '-', c=colour, lw=1.0, zorder=3)
            midpoint = np.mean(a[:2], axis=0)
            direction = a[1] - a[0]
            direction /= max(np.linalg.norm(direction), 1e-9)
            normal = np.array([-direction[1], direction[0]])
            candidates = [
                midpoint + normal * normal_offset + direction * tangent_offset
                for normal_offset in (0.28, -0.28, 0.65, -0.65, 1.0, -1.0)
                for tangent_offset in (0.0, 0.8, -0.8)
            ]
            label_width = 2.15
            label_height = 0.46

            def label_score(position):
                x0 = position[0] - label_width / 2
                x1 = position[0] + label_width / 2
                y0 = position[1] - label_height / 2
                y1 = position[1] + label_height / 2
                overlap = 0.0
                for bx0, bx1, by0, by1 in label_boxes:
                    overlap += max(0.0, min(x1, bx1) - max(x0, bx0)) * \
                        max(0.0, min(y1, by1) - max(y0, by0))
                distance = np.linalg.norm(position - midpoint)
                return overlap * 100.0 + distance

            label_position = min(candidates, key=label_score)
            label_boxes.append((
                label_position[0] - label_width / 2,
                label_position[0] + label_width / 2,
                label_position[1] - label_height / 2,
                label_position[1] + label_height / 2,
            ))
            ax.annotate(
                reference,
                xy=(midpoint[0], midpoint[1]),
                xytext=(label_position[0], label_position[1]),
                textcoords='data',
                fontsize=6.2,
                color=colour,
                weight='bold',
                bbox=dict(
                    facecolor='white', edgecolor=colour,
                    alpha=0.82, linewidth=0.35, pad=0.8),
                arrowprops=dict(
                    arrowstyle='-', color=colour, linewidth=0.35,
                    shrinkA=1.5, shrinkB=1.5),
                zorder=5,
            )
        for p in [p for p in pilares if p.get('banda') == k]:
            ax.plot(p['cx'], p['cy'], 's', ms=9, mfc='none', mec='blue', mew=2.2)
        n = len(storey_walls.get(nm, []))
        np_ = len([p for p in pilares if p.get('banda') == k])
        ax.set_aspect('equal')
        if used_confidences:
            legend = [
                Line2D(
                    [0], [0], color=confidence_colours[confidence], lw=2,
                    label=confidence)
                for confidence in (
                    'HIGH', 'MEDIUM', 'LOW', 'REVIEWED', 'UNSCORED')
                if confidence in used_confidences
            ]
            ax.legend(
                handles=legend, title='Confianca', fontsize=7,
                title_fontsize=7, loc='upper right')
        ax.set_title(f'{nm} ({band[0]:.1f} a {band[1]:.1f}) — {n} paredes, {np_} pilares', fontsize=10)
    for k in range(len(bandas), nrow * ncol):
        axes[k // ncol, k % ncol].axis('off')
    fig.suptitle(
        'Scan-to-BIM: IDs iguais no PNG e no IFC; cor = confianca',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description='Scan -> IFC (Cloud2BIM adaptado + pilares + escadas)')
    ap.add_argument('nuvem', help='caminho da nuvem (.ply, .e57 ou .xyz)')
    ap.add_argument('--thr', type=float, default=None,
                    help='SLAB_THR manual (pula o autodiagnostico)')
    ap.add_argument('--res', type=float, default=None,
                    help='pc_resolution manual em m (pula a estimativa; menor = paredes mais finas detectadas, mais lento)')
    ap.add_argument('--min-wall-len', type=float, default=0.30,
                    help='comprimento minimo de parede em m (default 0.30)')
    ap.add_argument('--saida', default=None, help='pasta de saida (default: <nuvem>_cloud2bim/)')
    ap.add_argument('--sem-pilar', action='store_true', help='pula o detector de pilar')
    ap.add_argument('--sem-escada', action='store_true', help='pula o detector de escada')
    ap.add_argument('--opening-detector-v2-review', action='store_true',
                    help='gera propostas JSON/PNG de portas e janelas; nao altera o IFC')
    ap.add_argument(
        '--open-leaf-review',
        action='store_true',
        help=(
            'apos detectar aberturas, procura folhas abertas reconstruidas '
            'como paredes e gera PNG/JSON de revisao; nao altera o IFC'
        ),
    )
    single_line_group = ap.add_mutually_exclusive_group()
    single_line_group.add_argument(
        '--single-line',
        dest='single_line',
        action='store_true',
        help='resgata paredes observadas por uma unica face',
    )
    single_line_group.add_argument(
        '--sem-single-line',
        dest='single_line',
        action='store_false',
        help='desliga o resgate de uma unica face (mais conservador)',
    )
    ap.set_defaults(single_line=None)
    ap.add_argument('--stair-cell', type=float, default=0.12,
                    help='resolucao XY do detector de escada em m (default 0.12)')
    ap.add_argument('--stair-area-min', type=float, default=0.8,
                    help='area minima de lance de escada em m2 (default 0.8)')
    ap.add_argument(
        '--slab-detector',
        choices=['v1_refined', 'v1', 'grouped'],
        default=os.environ.get('SLAB_DETECTOR', 'v1_refined'),
        help='lajes V1 com bordas limpas (default), V1 puro ou agrupamento experimental',
    )
    ap.add_argument(
        '--slab-wall-reference',
        choices=['v1', 'refined'],
        default=os.environ.get('SLAB_WALL_REFERENCE', 'v1'),
        help='contorno da laje usado como evidencia auxiliar de fachada',
    )
    args = ap.parse_args()

    src = Path(args.nuvem).resolve()
    if not src.exists():
        sys.exit(f'[X] arquivo nao existe: {src}')
    outdir = Path(args.saida).resolve() if args.saida else src.parent / f'{src.stem}_cloud2bim'
    outdir.mkdir(parents=True, exist_ok=True)
    nome = src.stem

    print(f'[1/6] Carregando {src.name}...')
    pts, rgb = carregar_nuvem(src)
    print(f'      {len(pts):,} pontos | extent {np.round(pts.max(0)-pts.min(0), 1)}')

    xyz_path = outdir / 'nuvem.xyz'
    if not xyz_path.exists():
        escrever_xyz(pts, rgb, xyz_path)
        print(f'      convertido -> {xyz_path.name}')
    else:
        print(f'      reusando {xyz_path.name} ja convertido')

    print('[2/6] Autodiagnostico do threshold de laje...')
    thr = args.thr if args.thr is not None else autodiag_slab_thr(pts[:, 2])
    print(f'      SLAB_THR = {thr}')

    print('[3/6] Rodando pipeline Cloud2BIM (lajes/paredes/aberturas/comodos)...')
    pc_res = args.res if args.res is not None else estimar_pc_resolution(pts)
    grid_coef = int(np.clip(round(0.10 / pc_res), 3, 6))
    ifc_path = outdir / f'{nome}_cloud2bim.ifc'
    cfg_path = outdir / 'config.yaml'
    cfg_path.write_text(CONFIG_TEMPLATE.format(
        xyz=str(xyz_path).replace('\\', '/'), pc_res=pc_res, grid_coef=grid_coef,
        ifc=str(ifc_path).replace('\\', '/'), nome=nome,
        min_wall_len=args.min_wall_len), encoding='utf-8')
    print(f'      pc_resolution={pc_res} grid_coefficient={grid_coef}')

    # thresholds da receita; ja setados no ambiente (ex.: pelo visualizador do
    # front) tem prioridade sobre os defaults
    single_line = (
        os.environ.get('SINGLE_LINE', '1')
        if args.single_line is None
        else ('1' if args.single_line else '0')
    )
    env = {**os.environ,
           'SLAB_THR': str(thr),
           'SLAB_DETECTOR': args.slab_detector,
           'SLAB_WALL_REFERENCE': args.slab_wall_reference,
           'WALL_DETECTOR': os.environ.get('WALL_DETECTOR', 'v2'),
           'WALL_ZLO': os.environ.get('WALL_ZLO', '0.1'),
           'WALL_ZHI': os.environ.get('WALL_ZHI', '0.9'),
           'SINGLE_LINE': single_line,
           'SINGLE_LINE_MINLEN': os.environ.get('SINGLE_LINE_MINLEN', '1.5'),
           'SINGLE_LINE_THK': os.environ.get('SINGLE_LINE_THK', '0.15'),
           'WALL_CONTOURS': os.environ.get('WALL_CONTOURS', 'all'),
           'PYTHONIOENCODING': 'utf-8',
           'MPLBACKEND': 'Agg'}
    # pastas que o Cloud2BIM assume existir no cwd (plots/debug)
    for d in ('images/pdf', 'images/wall_outputs_images', 'output_xyz'):
        (outdir / d).mkdir(parents=True, exist_ok=True)
    log_path = outdir / 'pipeline.log'
    with open(log_path, 'w', encoding='utf-8') as lf:
        r = subprocess.run([sys.executable, '-u', str(C2B / 'cloud2entities.py'), str(cfg_path)],
                           cwd=outdir, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not ifc_path.exists():
        sys.exit(f'[X] pipeline falhou (veja {log_path})')

    if args.sem_pilar:
        pilares = []
        print('[4/6] Detector de pilar pulado (--sem-pilar)')
    else:
        print('[4/6] Detectando pilares por pavimento...')
        pilares = inserir_pilares(ifc_path, pts)
        print(f'      {len(pilares)} pilares inseridos no IFC')

    if args.sem_escada:
        print('[5/6] Detector de escada pulado (--sem-escada)')
    else:
        print('[5/6] Detectando e inserindo escadas...')
        stair_ifc = outdir / f'{nome}_com_escadas.ifc'
        stair_log = outdir / 'stair.log'
        with open(stair_log, 'w', encoding='utf-8') as lf:
            stair_result = subprocess.run(
                [
                    sys.executable,
                    str(PKG / 'prototipos' / 'montar_escada_gabarito.py'),
                    str(xyz_path),
                    str(ifc_path),
                    str(stair_ifc),
                    '--cell', str(args.stair_cell),
                    '--area-min', str(args.stair_area_min),
                ],
                cwd=outdir,
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
        if stair_result.returncode == 0 and stair_ifc.exists():
            stair_ifc.replace(ifc_path)
            import ifcopenshell
            stair_count = len(ifcopenshell.open(str(ifc_path)).by_type('IfcStair'))
            print(f'      {stair_count} escadas inseridas no IFC')
        else:
            print(f'[!] detector de escada falhou; veja {stair_log}')

    print('[6/6] Renderizando plantas por pavimento...')
    png_path = outdir / f'{nome}_pavimentos.png'
    render_pavimentos(pts, ifc_path, png_path, pilares)

    if args.opening_detector_v2_review or args.open_leaf_review:
        diagnostics_path = outdir / 'wall_diagnostics.csv'
        opening_output = outdir / 'opening_detector_v2'
        if diagnostics_path.exists():
            print('[review] Gerando propostas do Opening Detector V2...')
            opening_result = subprocess.run(
                [sys.executable, str(PKG / 'run_opening_detector_v2.py'),
                 str(ifc_path), str(xyz_path), str(diagnostics_path),
                 str(opening_output), '--geometry-mode', 'ifc'],
                cwd=outdir, env=env)
            if opening_result.returncode == 0:
                print(f'      propostas: {opening_output / "opening_candidates_v2.json"}')
                if args.open_leaf_review:
                    leaf_output = outdir / 'open_leaf_detector'
                    print('[review] Procurando folhas abertas reconstruidas como paredes...')
                    leaf_result = subprocess.run(
                        [
                            sys.executable,
                            str(PKG / 'detect_open_leaves.py'),
                            str(xyz_path),
                            str(diagnostics_path),
                            str(opening_output / 'opening_candidates_v2.json'),
                            str(leaf_output),
                        ],
                        cwd=outdir,
                        env=env,
                    )
                    if leaf_result.returncode == 0:
                        print(
                            '      revisao: '
                            f'{leaf_output / "open_leaf_before_after.png"}'
                        )
                    else:
                        print(
                            '[!] Detector de folhas abertas falhou; '
                            'o IFC nao foi alterado'
                        )
            else:
                print('[!] Opening Detector V2 falhou; o IFC nao foi alterado')
        else:
            print('[!] wall_diagnostics.csv ausente; detector V2 nao executado')

    # resumo
    import ifcopenshell
    m = ifcopenshell.open(str(ifc_path))
    print('\n===== RESUMO =====')
    for t in ['IfcBuildingStorey', 'IfcSlab', 'IfcWall', 'IfcWindow', 'IfcDoor',
              'IfcSpace', 'IfcColumn', 'IfcStair']:
        n = len(m.by_type(t))
        if n:
            print(f'  {t:18s} {n}')
    print(f'\n  IFC:    {ifc_path}')
    print(f'  Planta: {png_path}')
    print(f'  Log:    {log_path}')


if __name__ == '__main__':
    main()
