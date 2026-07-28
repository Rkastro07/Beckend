# -*- coding: utf-8 -*-
"""ASSA a geometria no IFC: substitui a representacao parametrica (extrusao +
IfcRelVoidsElement) pela malha triangulada JA COM OS FUROS consumados.

Por que: varios viewers (Bonsai do Rafael incluso, e qualquer three.js) nao
aplicam os booleans dos vaos — parede/laje aparecem lisas com as janelas/vaos
"escondidos dentro". O ifcopenshell aplica; entao tesselamos aqui e o arquivo
fica a prova de viewer.

Custo: o elemento vira malha (perde edicao parametrica no Bonsai) e perde o
Axis — rodar SEMPRE por ultimo na cadeia (depois do montador de escada).

Uso:
  python assar_geometria.py <ifc> [--tipos IfcWall,IfcSlab]
(edita no lugar)
"""
import argparse

import numpy as np
import ifcopenshell
import ifcopenshell.geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ifc')
    ap.add_argument('--tipos', default='IfcWall,IfcSlab')
    args = ap.parse_args()

    m = ifcopenshell.open(args.ifc)
    s = ifcopenshell.geom.settings()          # coords locais: preserva placement
    ctx = m.by_type('IfcGeometricRepresentationContext')[0]
    tipos = [t.strip() for t in args.tipos.split(',')]

    for tipo in tipos:
        n_ok = 0
        for el in list(m.by_type(tipo)):
            try:
                sh = ifcopenshell.geom.create_shape(s, el)
                v = np.array(sh.geometry.verts).reshape(-1, 3)
                f = np.array(sh.geometry.faces).reshape(-1, 3)
            except Exception:
                continue
            if len(f) == 0:
                continue
            pts = m.create_entity('IfcCartesianPointList3D',
                                  CoordList=[[float(a), float(b), float(c)] for a, b, c in v])
            faces = [m.create_entity('IfcIndexedPolygonalFace',
                                     CoordIndex=[int(i) + 1 for i in tri]) for tri in f]
            fs = m.create_entity('IfcPolygonalFaceSet', Coordinates=pts, Closed=False,
                                 Faces=faces)
            rep = m.create_entity('IfcShapeRepresentation', ContextOfItems=ctx,
                                  RepresentationIdentifier='Body',
                                  RepresentationType='Tessellation', Items=[fs])
            el.Representation = m.create_entity('IfcProductDefinitionShape',
                                                Representations=[rep])
            n_ok += 1
        # remove os vaos ja consumados na malha
        n_rv = 0
        for r in list(m.by_type('IfcRelVoidsElement')):
            if r.RelatingBuildingElement.is_a(tipo):
                m.remove(r)
                n_rv += 1
        print(f'{tipo}: {n_ok} elementos assados, {n_rv} vaos consumidos')

    m.write(args.ifc)
    print(f'salvo: {args.ifc}')


if __name__ == '__main__':
    main()
