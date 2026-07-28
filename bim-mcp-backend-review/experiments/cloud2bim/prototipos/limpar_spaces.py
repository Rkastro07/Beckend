# -*- coding: utf-8 -*-
"""Remove IfcSpace degenerados ("zonas-lasca").

Quando duas paredes quase coincidentes sao detectadas a poucos cm uma da outra,
o gerador de zonas cria um "comodo" de centimetros de largura entre elas — que
no viewer parece uma parede. Comodo de verdade tem area e largura minimas.

Uso:  python limpar_spaces.py <ifc> [--area-min 1.5] [--largura-min 0.45]
(edita no lugar)
"""
import argparse

import numpy as np
import ifcopenshell


def poligono_do_space(sp):
    try:
        for r in sp.Representation.Representations:
            for it in r.Items:
                if it.is_a('IfcExtrudedAreaSolid'):
                    curve = it.SweptArea.OuterCurve
                    return np.array([p.Coordinates for p in curve.Points], dtype=float)
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ifc')
    ap.add_argument('--area-min', type=float, default=1.5)
    ap.add_argument('--largura-min', type=float, default=0.45)
    args = ap.parse_args()

    m = ifcopenshell.open(args.ifc)
    removidos = 0
    for sp in list(m.by_type('IfcSpace')):
        poly = poligono_do_space(sp)
        if poly is None or len(poly) < 3:
            continue
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        # largura minima aproximada: area / maior dimensao do bbox
        maior = max(x.max() - x.min(), y.max() - y.min())
        largura = area / maior if maior > 0 else 0.0
        if area < args.area_min or largura < args.largura_min:
            for rel in list(m.by_type('IfcRelAggregates')) + \
                       list(m.by_type('IfcRelContainedInSpatialStructure')):
                if hasattr(rel, 'RelatedObjects') and sp in (rel.RelatedObjects or ()):
                    rel.RelatedObjects = [e for e in rel.RelatedObjects if e != sp]
                if hasattr(rel, 'RelatedElements') and sp in (rel.RelatedElements or ()):
                    rel.RelatedElements = [e for e in rel.RelatedElements if e != sp]
            m.remove(sp)
            removidos += 1
    m.write(args.ifc)
    n = len(m.by_type('IfcSpace'))
    print(f'{removidos} spaces-lasca removidos | {n} comodos legitimos mantidos')


if __name__ == '__main__':
    main()
