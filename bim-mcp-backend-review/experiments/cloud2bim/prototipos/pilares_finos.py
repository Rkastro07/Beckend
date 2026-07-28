# -*- coding: utf-8 -*-
"""Passada FINA de pilares: colunas esbeltas (Ø15-25cm, ex. porticos de entrada)
que a passada normal perde por amostragem esparsa.

Parametros relaxados (celula 10cm, 1 pt/bin, secao >=8cm) acham as colunas
finas MAS tambem disparam nos montantes de fachada entre janelas. Dois filtros
de reconciliacao resolvem:
  - deriva do centroide <= 5cm (coluna livre e mais 'reta' que montante)
  - candidato a menos de 35cm de um EIXO DE PAREDE ja detectado = montante
    (pilar embutido em parede pertence a parede) -> descartado

Uso:
  python pilares_finos.py <nuvem.xyz> <ifc_em_edicao> [--zlo 0.05 --zhi 2.65]
(edita o IFC no lugar, adicionando IfcColumn)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detect_pilar import find_pillars  # noqa: E402


def eixos_de_parede(m):
    import ifcopenshell.util.placement as P
    segs = []
    for w in m.by_type('IfcWall'):
        mat = P.get_local_placement(w.ObjectPlacement)
        for r in w.Representation.Representations:
            if r.RepresentationIdentifier == 'Axis':
                for it in r.Items:
                    if it.is_a('IfcPolyline'):
                        pts = [(mat @ np.array(list(p.Coordinates)[:2] + [0, 1.0]))[:2]
                               for p in it.Points]
                        segs.append(np.array(pts))
    return segs


def dist_ponto_seg(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('xyz')
    ap.add_argument('ifc')
    ap.add_argument('--zlo', type=float, default=0.05)
    ap.add_argument('--zhi', type=float, default=2.65)
    ap.add_argument('--drift-max', type=float, default=0.05)
    ap.add_argument('--dist-parede', type=float, default=0.35)
    args = ap.parse_args()

    print('[1/3] Passada fina de pilares...')
    pts = np.loadtxt(args.xyz, skiprows=1, usecols=(0, 1, 2))
    cands = find_pillars(pts, args.zlo, args.zhi, cell=0.10, min_pts_bin=1,
                         full_frac=0.60, sec_min=0.08, sec_aspect=2.0,
                         drift_max=args.drift_max, consist_min=0.90,
                         minratio_min=0.40)
    finos = [p for p in cands if p['ok']]
    print(f'      {len(finos)} candidatos passaram os filtros de forma/constancia/deriva')

    print('[2/3] Reconciliacao com eixos de parede (montante != pilar)...')
    import ifcopenshell
    import ifcopenshell.guid
    m = ifcopenshell.open(args.ifc)
    segs = eixos_de_parede(m)
    livres = []
    for p in finos:
        c = np.array([p['cx'], p['cy']])
        d = min((dist_ponto_seg(c, s[i], s[i + 1])
                 for s in segs for i in range(len(s) - 1)), default=99.0)
        if d > args.dist_parede:
            p['dist_parede'] = d
            livres.append(p)
    print(f'      {len(livres)} pilares LIVRES (nao-montante):')
    for p in livres:
        print(f'        ({p["cx"]:.2f},{p["cy"]:.2f}) secao {p["w"]:.2f}x{p["h"]:.2f} '
              f'deriva {p["drift"]*100:.1f}cm dist.parede {p["dist_parede"]:.2f}m')

    print('[3/3] Inserindo IfcColumn...')
    owner = m.by_type('IfcOwnerHistory')[0]
    ctx = m.by_type('IfcGeometricRepresentationContext')[0]
    zdir = m.create_entity('IfcDirection', DirectionRatios=(0.0, 0.0, 1.0))
    rel = m.by_type('IfcRelContainedInSpatialStructure')
    n0 = len(m.by_type('IfcColumn'))
    for k, p in enumerate(livres):
        # secao esbelta subestimada pela amostragem: usa circular Ø = max(lado)*2
        raio = max(p['w'], p['h'])
        prof = m.create_entity('IfcCircleProfileDef', ProfileType='AREA',
                               ProfileName=f'Coluna D{raio*200:.0f}', Radius=float(raio))
        pos = m.create_entity('IfcAxis2Placement3D',
                              Location=m.create_entity('IfcCartesianPoint',
                                                       Coordinates=(0.0, 0.0, 0.0)))
        solid = m.create_entity('IfcExtrudedAreaSolid', SweptArea=prof, Position=pos,
                                ExtrudedDirection=zdir, Depth=float(args.zhi - args.zlo))
        shape = m.create_entity('IfcShapeRepresentation', ContextOfItems=ctx,
                                RepresentationIdentifier='Body',
                                RepresentationType='SweptSolid', Items=[solid])
        pds = m.create_entity('IfcProductDefinitionShape', Representations=[shape])
        lp = m.create_entity('IfcLocalPlacement', RelativePlacement=m.create_entity(
            'IfcAxis2Placement3D',
            Location=m.create_entity('IfcCartesianPoint',
                                     Coordinates=(float(p['cx']), float(p['cy']), float(args.zlo)))))
        col = m.create_entity('IfcColumn', GlobalId=ifcopenshell.guid.new(),
                              OwnerHistory=owner, Name=f'C{n0+k+1:02d}',
                              Description='Coluna esbelta (passada fina + reconciliacao)',
                              ObjectPlacement=lp, Representation=pds)
        if rel:
            r0 = rel[0]
            r0.RelatedElements = list(r0.RelatedElements) + [col]
    m.write(args.ifc)
    print(f'      {len(livres)} IfcColumn adicionados a {args.ifc}')


if __name__ == '__main__':
    main()
