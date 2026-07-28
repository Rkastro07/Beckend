# -*- coding: utf-8 -*-
"""MONTADOR DE GABARITO de escada — ferradura de 3 lances.

Filosofia (decidida com o Rafael): o detector coleta EVIDENCIAS (lances, direcoes,
espelho, poco); o montador escolhe um GABARITO parametrico e instancia a escada
completa — inclusive lances que a deteccao perdeu (deduzidos pela subida que
falta). Dimensoes medidas quando a medicao e limpa, norma como fallback, e a
PROVENIENCIA gravada no IFC (Description).

Gabarito FERRADURA-3: lance A sobe -> patamar (vira 90) -> lance B -> patamar
(vira 90) -> lance C chega no piso de cima, dando a volta no poco.

Uso:
  python montar_escada_gabarito.py <nuvem.xyz> <entrada.ifc> <saida.ifc>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_escada_reta import detectar_lances, fundir_lances, _lajes_entidades  # noqa: E402

RISER_NORMA = 0.175
TREAD_NORMA = 0.30


def rot90(ux, uy, sentido):
    """Gira o vetor 90 graus (sentido +1 = anti-horario, -1 = horario)."""
    return (-sentido * uy, sentido * ux)


def montar_ferradura(lances, piso_z, teto_z, eixos_parede=()):
    """Evidencias -> parametros da ferradura-3. Retorna dict ou None."""
    if not lances:
        return None
    mid = max(lances, key=lambda L: L['comprimento'])       # lance do meio (maior)
    outros = [L for L in lances if L is not mid]
    d2 = np.array([mid['ux'], mid['uy']])
    # snap ortogonal: predio Manhattan — a direcao do gradiente tem erro de
    # alguns graus e o gabarito nao pode propagar inclinacao
    ang = np.arctan2(d2[1], d2[0])
    ang_snap = np.round(ang / (np.pi / 2)) * (np.pi / 2)
    if abs(((ang - ang_snap + np.pi) % (2 * np.pi)) - np.pi) < np.radians(25):
        d2 = np.array([np.cos(ang_snap), np.sin(ang_snap)]).round(6)
    ini = np.array([mid['cx'], mid['cy']]) - d2 * mid['comprimento'] / 2
    fim = np.array([mid['cx'], mid['cy']]) + d2 * mid['comprimento'] / 2

    # espelho: medido se os picos foram limpos, senao norma
    espelhos = [L['riser_h'] for L in lances if L['riser_h'] == L['riser_h']]
    degraus_vistos = sum(L['n_degraus_vistos'] for L in lances)
    if espelhos and degraus_vistos >= 8:
        riser, fonte = float(np.median(espelhos)), 'medido'
    else:
        riser, fonte = RISER_NORMA, 'norma'
    subida = teto_z - piso_z
    n_total = max(3, int(round(subida / riser)))
    riser = subida / n_total                                  # fecha exato no piso

    w = float(np.median([L['largura'] for L in lances]))
    w = min(max(w, 1.0), 1.6)

    # POCO da escada: bbox das evidencias nos eixos (s ao longo de B, t perpendicular)
    sh = d2
    nh = np.array([-d2[1], d2[0]])
    cantos = []
    for L in lances:
        for xx in (L['x0'], L['x1']):
            for yy in (L['y0'], L['y1']):
                cantos.append((xx, yy))
    cantos = np.array(cantos, dtype=float)
    s_all = cantos @ sh
    t_all = cantos @ nh
    s_lo, s_hi = float(s_all.min()), float(s_all.max())
    # lado do poco onde B corre: o lado do centro do lance do meio
    t_mid = float(np.array([mid['cx'], mid['cy']]) @ nh)
    if t_mid < (t_all.min() + t_all.max()) / 2:
        nh = -nh                                     # nh aponta pro lado do B
        t_all = -t_all
    t_lo, t_hi = float(t_all.min()), float(t_all.max())

    # RECONCILIACAO com as paredes do nucleo: expande o poco ate a parede
    # envolvente mais proxima (lances nao-detectados deixam o poco curto)
    cand_oeste, cand_leste, cand_fundo = [], [], []
    for a, b in eixos_parede:
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        nrm = np.linalg.norm(d)
        if nrm < 1.0:
            continue
        d = d / nrm
        sa, sb = a @ sh, b @ sh
        ta, tb = a @ nh, b @ nh
        if abs(d @ sh) < 0.3:                         # parede perpendicular a B (lateral)
            if min(ta, tb) < t_hi - 0.5 and max(ta, tb) > t_lo + 0.5:
                s_w = (sa + sb) / 2
                # so paredes FORA do suporte (parede do olho do poco fica dentro)
                if s_lo - 2.0 <= s_w <= s_lo - 0.2:
                    cand_oeste.append(s_w)
                if s_hi + 0.2 <= s_w <= s_hi + 2.0:
                    cand_leste.append(s_w)
        elif abs(d @ nh) < 0.3:                       # parede paralela a B (fundo)
            if min(sa, sb) < s_hi - 0.5 and max(sa, sb) > s_lo + 0.5:
                t_w = (ta + tb) / 2
                if t_hi + 0.2 <= t_w <= t_hi + 2.0:
                    cand_fundo.append(t_w)
    # sempre a parede MAIS INTERNA (nucleo tem paredes duplas/externas)
    if cand_oeste:
        s_lo = min(s_lo, max(cand_oeste) + 0.15)
    if cand_leste:
        s_hi = max(s_hi, min(cand_leste) - 0.15)
    if cand_fundo:
        t_hi = max(t_hi, min(cand_fundo) - 0.15)
    d1 = nh.copy()                                   # A sobe EM DIRECAO a faixa do B
    d3 = -nh                                         # C desce afastando (chega no piso de cima)

    # distribui degraus
    n2 = int(round((mid['z1'] - mid['z0']) / riser)) or 1
    n2 = min(max(n2, 3), n_total - 4)
    resto = n_total - n2
    n3 = resto // 2
    n1 = resto - n3
    B_len = max((s_hi - s_lo) - 2 * w, 1.5)
    tread = min(max(B_len / n2, 0.22), 0.40)
    run_lat = max(t_hi - t_lo - w, 1.2)              # espaco lateral p/ lances A e C
    tread_ac = min(max(run_lat / max(n1, n3), 0.22), 0.38)

    def P(s, t):
        return sh * s + nh * t
    t_B = t_hi - w / 2
    p1c = P(s_lo + w / 2, t_B)                       # patamar 1 (canto inicial)
    p2c = P(s_hi - w / 2, t_B)                       # patamar 2 (canto final)
    f2c = P((s_lo + s_hi) / 2, t_B)                  # lance B atravessa o poco
    f1c = p1c + d3 * (w / 2 + n1 * tread_ac / 2)     # lance A na lateral inicial
    f3c = p2c + d3 * (w / 2 + n3 * tread_ac / 2)     # lance C na lateral final
    t_edgeB = t_hi - w                                # borda interna da faixa do B
    return {
        'riser': riser, 'fonte': fonte, 'tread': tread, 'w': w,
        'n1': n1, 'n2': n2, 'n3': n3, 'piso': piso_z, 'teto': teto_z,
        'frame': {'sh': sh, 'nh': nh, 's_lo': s_lo, 's_hi': s_hi, 't_hi': t_hi,
                  'w': w, 't_As': t_edgeB - n1 * tread_ac, 't_Cs': t_edgeB - n3 * tread_ac},
        'lances': [
            {'c': f1c, 'd': d1, 'n': n1, 'z_base': piso_z, 'tread': tread_ac},
            {'c': f2c, 'd': d2, 'n': n2, 'z_base': piso_z + n1 * riser, 'tread': tread},
            {'c': f3c, 'd': d3, 'n': n3, 'z_base': piso_z + (n1 + n2) * riser,
             'tread': tread_ac},
        ],
        'patamares': [
            {'c': p1c, 'z': piso_z + n1 * riser},
            {'c': p2c, 'z': piso_z + (n1 + n2) * riser},
        ],
    }


def _caixa(m, quad, z0, dz, ctx, zdir):
    pts = [m.create_entity('IfcCartesianPoint', Coordinates=(float(a), float(b)))
           for a, b in quad]
    poly = m.create_entity('IfcPolyline', Points=pts + [pts[0]])
    prof = m.create_entity('IfcArbitraryClosedProfileDef', ProfileType='AREA', OuterCurve=poly)
    pos = m.create_entity('IfcAxis2Placement3D',
                          Location=m.create_entity('IfcCartesianPoint',
                                                   Coordinates=(0.0, 0.0, float(z0))))
    return m.create_entity('IfcExtrudedAreaSolid', SweptArea=prof, Position=pos,
                           ExtrudedDirection=zdir, Depth=float(dz))


def _rect(c, d, comp, w):
    ux, uy = d
    px, py = -uy, ux
    b = c - np.array([ux, uy]) * comp / 2
    e = c + np.array([ux, uy]) * comp / 2
    h = np.array([px, py]) * w / 2
    return [tuple(b + h), tuple(e + h), tuple(e - h), tuple(b - h)]


def construir_ifc_escada(m, G, nome, ctx, owner, zdir):
    import ifcopenshell.guid
    solids = []
    for LC in G['lances']:
        ux, uy = LC['d']
        tr = LC.get('tread', G['tread'])
        b = LC['c'] - LC['d'] * (LC['n'] * tr) / 2
        for i in range(LC['n']):
            s0, s1 = i * tr, (i + 1) * tr
            quad = [tuple(b + LC['d'] * s0 + np.array([-uy, ux]) * G['w'] / 2),
                    tuple(b + LC['d'] * s1 + np.array([-uy, ux]) * G['w'] / 2),
                    tuple(b + LC['d'] * s1 - np.array([-uy, ux]) * G['w'] / 2),
                    tuple(b + LC['d'] * s0 - np.array([-uy, ux]) * G['w'] / 2)]
            zt = LC['z_base'] + (i + 1) * G['riser']
            solids.append(_caixa(m, quad, zt - G['riser'], G['riser'], ctx, zdir))
    for P in G['patamares']:
        quad = _rect(P['c'], (1, 0), G['w'], G['w'])
        solids.append(_caixa(m, quad, P['z'] - 0.15, 0.15, ctx, zdir))
    shape = m.create_entity('IfcShapeRepresentation', ContextOfItems=ctx,
                            RepresentationIdentifier='Body',
                            RepresentationType='SweptSolid', Items=solids)
    pds = m.create_entity('IfcProductDefinitionShape', Representations=[shape])
    lp = m.create_entity('IfcLocalPlacement', RelativePlacement=m.create_entity(
        'IfcAxis2Placement3D',
        Location=m.create_entity('IfcCartesianPoint', Coordinates=(0.0, 0.0, 0.0))))
    st = m.create_entity('IfcStair', GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner,
                         Name=nome,
                         Description=(f'Gabarito ferradura-3: {G["n1"]}+{G["n2"]}+{G["n3"]} degraus, '
                                      f'espelho {G["riser"]*100:.1f}cm ({G["fonte"]}), '
                                      f'piso {G["tread"]*100:.0f}cm'),
                         ObjectPlacement=lp, Representation=pds,
                         PredefinedType='HALF_TURN_STAIR')
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('xyz')
    ap.add_argument('ifc_in')
    ap.add_argument('ifc_out')
    ap.add_argument('--cell', type=float, default=0.12,
                    help='resolucao XY do detector de lances em metros')
    ap.add_argument('--area-min', type=float, default=0.8,
                    help='area minima de um lance candidato em m2')
    ap.add_argument('--slope-min', type=float, default=0.35,
                    help='declividade minima do lance')
    ap.add_argument('--slope-max', type=float, default=1.2,
                    help='declividade maxima do lance')
    args = ap.parse_args()

    import ifcopenshell
    import ifcopenshell.guid
    import ifcopenshell.util.placement as P

    print('[1/5] Carregando nuvem e modelo...')
    pts = np.loadtxt(args.xyz, skiprows=1, usecols=(0, 1, 2))
    m = ifcopenshell.open(args.ifc_in)
    lajes = _lajes_entidades(m)
    bandas = [(lajes[i][0] + lajes[i][1], lajes[i + 1][0], lajes[i + 1])
              for i in range(len(lajes) - 1)]   # (piso=topo da laje i, fundo laje i+1, laje i+1)

    print('[2/5] Removendo escadas e vaos de escada antigos...')
    n_rm = 0
    for st in list(m.by_type('IfcStair')):
        for rel in m.by_type('IfcRelContainedInSpatialStructure'):
            if st in rel.RelatedElements:
                rel.RelatedElements = [e for e in rel.RelatedElements if e != st]
        m.remove(st)
        n_rm += 1
    for op in list(m.by_type('IfcOpeningElement')):
        if op.Name == 'Vao de escada':
            for rv in list(m.by_type('IfcRelVoidsElement')):
                if rv.RelatedOpeningElement == op:
                    m.remove(rv)
            m.remove(op)
    print(f'      {n_rm} escadas antigas removidas')

    print('[3/5] Detectando evidencias e montando gabaritos...')
    ctx = m.by_type('IfcGeometricRepresentationContext')[0]
    owner = m.by_type('IfcOwnerHistory')[0]
    zdir = m.create_entity('IfcDirection', DirectionRatios=(0.0, 0.0, 1.0))
    rel0 = m.by_type('IfcRelContainedInSpatialStructure')[0]
    # eixos de parede (pra reconciliar o poco com o nucleo)
    eixos = []
    for w in m.by_type('IfcWall'):
        try:
            mat = P.get_local_placement(w.ObjectPlacement)
            for r in w.Representation.Representations:
                if r.RepresentationIdentifier == 'Axis':
                    for it in r.Items:
                        if it.is_a('IfcPolyline'):
                            ptsw = [(mat @ np.array(list(p.Coordinates)[:2] + [0, 1.0]))[:2]
                                    for p in it.Points]
                            eixos.append((ptsw[0], ptsw[-1]))
        except Exception:
            continue
    gabaritos = []
    for k, (piso, fundo_prox, laje_prox) in enumerate(bandas):
        if fundo_prox - piso < 1.5:
            continue
        lances = fundir_lances(detectar_lances(
            pts, piso, fundo_prox,
            cell=float(args.cell),
            area_min=float(args.area_min),
            slope_min=float(args.slope_min),
            slope_max=float(args.slope_max),
        ))
        if not lances:
            continue
        teto = laje_prox[0] + laje_prox[1]        # topo da laje de chegada
        G = montar_ferradura(lances, piso, teto, eixos)
        if G is None:
            continue
        st = construir_ifc_escada(m, G, f'Escada pav {k+1}', ctx, owner, zdir)
        rel0.RelatedElements = list(rel0.RelatedElements) + [st]
        gabaritos.append((k, G, laje_prox))
        print(f'      banda {k}: ferradura {G["n1"]}+{G["n2"]}+{G["n3"]} degraus, '
              f'espelho {G["riser"]*100:.1f}cm ({G["fonte"]}), largura {G["w"]:.1f}m')

    print('[4/5] Cortando vaos em FERRADURA nas lajes de chegada...')
    n_open = 0
    for k, G, (z0l, espl, entl) in gabaritos:
        # poligono em U: as 3 faixas de lance + patamares abertos; o olho do
        # poco e a chegada sul ficam como peninsula de laje (piso de circulacao)
        F = G['frame']
        sh_, nh_ = F['sh'], F['nh']
        mg = 0.10
        s0, s1 = F['s_lo'] - mg, F['s_hi'] + mg
        tt = F['t_hi'] + mg
        tA, tC = F['t_As'] - mg, F['t_Cs'] - mg
        si0, si1 = F['s_lo'] + F['w'] + 0.05, F['s_hi'] - F['w'] - 0.05
        ti = F['t_hi'] - F['w'] - 0.05

        def Pw(s, t):
            p = sh_ * s + nh_ * t
            return (float(p[0]), float(p[1]))
        quadU = [Pw(s0, tA), Pw(s0, tt), Pw(s1, tt), Pw(s1, tC),
                 Pw(si1, tC), Pw(si1, ti), Pw(si0, ti), Pw(si0, tA)]
        for quad in (quadU,):
            solid = _caixa(m, quad, z0l - 0.05, espl + 0.10, ctx, zdir)
            shape = m.create_entity('IfcShapeRepresentation', ContextOfItems=ctx,
                                    RepresentationIdentifier='Body',
                                    RepresentationType='SweptSolid', Items=[solid])
            pds = m.create_entity('IfcProductDefinitionShape', Representations=[shape])
            lp = m.create_entity('IfcLocalPlacement', RelativePlacement=m.create_entity(
                'IfcAxis2Placement3D',
                Location=m.create_entity('IfcCartesianPoint', Coordinates=(0.0, 0.0, 0.0))))
            op = m.create_entity('IfcOpeningElement', GlobalId=ifcopenshell.guid.new(),
                                 OwnerHistory=owner, Name='Vao de escada',
                                 ObjectPlacement=lp, Representation=pds)
            m.create_entity('IfcRelVoidsElement', GlobalId=ifcopenshell.guid.new(),
                            OwnerHistory=owner, RelatingBuildingElement=entl,
                            RelatedOpeningElement=op)
            n_open += 1
    print(f'      {n_open} vaos cortados')

    m.write(args.ifc_out)
    print(f'[5/5] Salvo em {args.ifc_out}')

    # verificacao: top-view da banda 0 + os 3 lances/2 patamares
    if gabaritos:
        k, G, _ = gabaritos[0]
        fig, ax = plt.subplots(figsize=(8, 7))
        piso, fundo = G['piso'], G['teto']
        sl = pts[(pts[:, 2] >= piso + 0.2) & (pts[:, 2] < fundo - 0.3)]
        reg = sl[(sl[:, 0] > 16) & (sl[:, 0] < 26) & (sl[:, 1] > 10) & (sl[:, 1] < 17)]
        ax.scatter(reg[:, 0], reg[:, 1], s=1.2, c=reg[:, 2], cmap='Blues', linewidths=0)
        cores = ['#c22', '#d70', '#291']
        for i, LC in enumerate(G['lances']):
            q = np.array(_rect(LC['c'], tuple(LC['d']), LC['n'] * LC.get('tread', G['tread']),
                               G['w']), dtype=float)
            q = np.vstack([q, q[:1]])
            ax.plot(q[:, 0], q[:, 1], '-', c=cores[i], lw=2.5, label=f'lance {chr(65+i)} ({LC["n"]} dg)')
        for Pt in G['patamares']:
            q = np.array(_rect(Pt['c'], (1, 0), G['w'], G['w']), dtype=float)
            q = np.vstack([q, q[:1]])
            ax.plot(q[:, 0], q[:, 1], '--', c='#555', lw=2)
        ax.legend(); ax.set_aspect('equal')
        ax.set_title(f'Gabarito ferradura-3 montado (banda {k})')
        plt.tight_layout()
        png = str(Path(args.ifc_out).with_suffix('')) + '_gabarito.png'
        plt.savefig(png, dpi=130)
        print(f'      verificacao: {png}')


if __name__ == '__main__':
    main()
