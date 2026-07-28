# -*- coding: utf-8 -*-
"""Detector de ESCADA RETA (lances lineares) — protótipo.

Assinatura do lance reto (irmão linear da caracol):
  - célula XY FINA em z (como laje/degrau, oposto de parede)
  - mas a altura AVANÇA linearmente ao longo de uma direção horizontal
    (plano inclinado com declividade de escada, rise/run ~0.5-0.9)
  - dentro do lance, z é DISCRETO e periódico (espelhos) — find_peaks

Pipeline por banda de pavimento (lida das lajes do IFC):
  células finas entre lajes → componentes conexos → fit de plano por componente
  → declividade de escada = lance → projeção na direção de subida → espelhos
  → IfcStair (caixas de degrau empilhadas).

Uso:
  python detect_escada_reta.py <nuvem.xyz> <entrada.ifc> <saida.ifc>
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def bandas_das_lajes(ifc_path):
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
            lajes.append((float(z0), float(z0 + max(depths))))
    lajes.sort()
    return [(lajes[i][1], lajes[i + 1][0]) for i in range(len(lajes) - 1)]


def _lajes_entidades(m):
    """Lajes do modelo aberto, ordenadas por cota: [(z_bottom, thickness, entity)]."""
    import ifcopenshell.util.placement as P
    out = []
    for s in m.by_type('IfcSlab'):
        z0 = float(P.get_local_placement(s.ObjectPlacement)[2, 3])
        depths = []
        for r in s.Representation.Representations:
            for it in r.Items:
                if it.is_a('IfcExtrudedAreaSolid'):
                    depths.append(float(it.Depth))
        if depths:
            out.append((z0, max(depths), s))
    out.sort(key=lambda t: t[0])
    return out


def _dentro_do_lance(px_, py_, L, folga=0.3):
    """Ponto (x,y) cai no retangulo orientado do lance L (+folga)?"""
    ux, uy = L['ux'], L['uy']
    bx = L['cx'] - ux * L['comprimento'] / 2
    by = L['cy'] - uy * L['comprimento'] / 2
    s = (px_ - bx) * ux + (py_ - by) * uy
    t = (px_ - bx) * (-uy) + (py_ - by) * ux
    return (-folga <= s <= L['comprimento'] + folga) and (abs(t) <= L['largura'] / 2 + folga)


def detectar_lances(pts, zlo, zhi, cell=0.10, margem=0.20,
                    slope_min=0.35, slope_max=1.2, area_min=1.0):
    """Acha lances retos na banda [zlo+margem, zhi-margem).

    Robustez: (a) usa a superficie de BAIXO da celula (o degrau) — o guarda-corpo
    acima nao engrossa a celula; (b) paredes = celulas que ocupam a banda quase
    inteira, excluidas; (c) agrupa por DIRECAO do gradiente — lances opostos de
    uma escada em U viram componentes separados (fit de plano unico cancelaria).
    """
    banda_h = zhi - zlo - 2 * margem
    sel = pts[(pts[:, 2] >= zlo + margem) & (pts[:, 2] < zhi - margem)]
    if len(sel) < 500:
        return []
    xmin, ymin = sel[:, 0].min(), sel[:, 1].min()
    ix = ((sel[:, 0] - xmin) / cell).astype(np.int64)
    iy = ((sel[:, 1] - ymin) / cell).astype(np.int64)
    NX, NY = ix.max() + 1, iy.max() + 1
    key = iy * NX + ix
    order = np.argsort(key)
    ks, zs = key[order], sel[order, 2]
    uniq, starts = np.unique(ks, return_index=True)
    z_deg = np.full(NX * NY, np.nan)
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(ks)
        zc = zs[s:e]
        if len(zc) < 3:
            continue
        ext = zc.max() - zc.min()
        if ext > 0.6 * banda_h:
            continue                        # parede (ocupa a banda quase toda)
        # superficie de baixo = degrau; corrimao/balaustre acima e ignorado
        z_deg[uniq[i]] = np.median(zc[zc <= zc.min() + 0.15])
    z_deg = z_deg.reshape(NY, NX)

    # RAMPA: suaviza o mapa numa janela ~0.7m (2-3 degraus) — sem isso o
    # gradiente enxerga degrau por degrau (0 no piso, pico no espelho) e nunca
    # a declividade media do lance
    J = max(3, int(round(0.7 / cell)) | 1)
    h = J // 2
    zpad = np.pad(z_deg, h, constant_values=np.nan)
    stack = [zpad[dy:dy + NY, dx:dx + NX] for dy in range(J) for dx in range(J)]
    with np.errstate(all='ignore'):
        z_rampa = np.nanmean(np.stack(stack), axis=0)
        gy, gx = np.gradient(z_rampa, cell)
        slope = np.hypot(gx, gy)
    ok = np.isfinite(slope) & (slope >= slope_min) & (slope <= slope_max)
    with np.errstate(all='ignore'):
        ang = np.arctan2(gy, gx)

    # setores de direcao SOBREPOSTOS (largura 90 graus, centros a cada 45):
    # lances opostos se separam, jitter de direcao nao fragmenta o lance
    lances = []
    aceitos_geo = []      # (cx, cy, ux, uy) p/ dedup entre setores sobrepostos
    for sbin in range(8):
        centro = -np.pi + sbin * (np.pi / 4)
        dist_ang = np.abs((ang - centro + np.pi) % (2 * np.pi) - np.pi)
        mask = (ok & (dist_ang <= np.pi / 4)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n_comp, labels = cv2.connectedComponents(mask, connectivity=8)
        for c in range(1, n_comp):
            yy, xx = np.nonzero(labels == c)
            if len(yy) * cell * cell < area_min:
                continue
            X = xmin + xx * cell + cell / 2
            Y = ymin + yy * cell + cell / 2
            Z = z_deg[yy, xx]
            fin = np.isfinite(Z)
            if fin.sum() < 10:
                continue
            X, Y, Z, yy2, xx2 = X[fin], Y[fin], Z[fin], yy[fin], xx[fin]
            A = np.column_stack([X, Y, np.ones(len(X))])
            (a, b, cc), *_ = np.linalg.lstsq(A, Z, rcond=None)
            sl = float(np.hypot(a, b))
            if not (slope_min <= sl <= slope_max):
                continue
            rms = float(np.sqrt(((A @ np.array([a, b, cc]) - Z) ** 2).mean()))
            if rms > 0.15:
                continue
            ux, uy = a / sl, b / sl         # direcao de subida (horizontal)
            # pontos crus das celulas do lance pra medir espelhos
            cel_ids = np.unique(yy2 * NX + xx2)
            pin = np.isin(ks, cel_ids)
            zraw = zs[pin]
            zraw = zraw[zraw <= np.nanmax(Z) + 0.2]   # corta corrimao
            if len(zraw) < 50:
                continue
            hist, edges = np.histogram(zraw, bins=np.arange(zraw.min(), zraw.max() + 0.02, 0.02))
            pk, _ = find_peaks(hist, distance=5, prominence=0.15 * hist.max())
            z_pk = edges[pk] + 0.01
            risers = np.diff(np.sort(z_pk))
            risers = risers[(risers > 0.10) & (risers < 0.25)]
            riser_h = float(np.median(risers)) if len(risers) else float('nan')
            comprimento = float(np.ptp((X - X.mean()) * ux + (Y - Y.mean()) * uy))
            largura = float(np.ptp((X - X.mean()) * (-uy) + (Y - Y.mean()) * ux))
            if comprimento < 0.9 or largura < 0.6 or (Z.max() - Z.min()) < 0.45:
                continue                    # curto/estreito/raso demais pra lance
            # dedup entre setores sobrepostos: mesmo lance detectado 2x
            cx_, cy_ = float(X.mean()), float(Y.mean())
            dup = any(np.hypot(cx_ - g[0], cy_ - g[1]) < 1.0 and
                      (ux * g[2] + uy * g[3]) > 0.7 for g in aceitos_geo)
            if dup:
                continue
            aceitos_geo.append((cx_, cy_, ux, uy))
            lances.append({
                'n_cells': int(fin.sum()), 'area': float(fin.sum()) * cell * cell,
                'cx': float(X.mean()), 'cy': float(Y.mean()),
                'x0': float(X.min()), 'x1': float(X.max()),
                'y0': float(Y.min()), 'y1': float(Y.max()),
                'z0': float(np.percentile(Z, 2)), 'z1': float(np.percentile(Z, 98)),
                'slope': sl, 'rms': rms, 'ux': float(ux), 'uy': float(uy),
                'riser_h': riser_h, 'n_degraus_vistos': int(len(z_pk)),
                'largura': largura, 'comprimento': comprimento,
            })
    return lances


def fundir_lances(lances):
    """Funde fragmentos do mesmo lance: mesma direcao, colineares, com vao curto."""
    fundidos = []
    usados = set()
    for i, A in enumerate(lances):
        if i in usados:
            continue
        grupo = dict(A)
        for j, B in enumerate(lances[i + 1:], i + 1):
            if j in usados:
                continue
            if grupo['ux'] * B['ux'] + grupo['uy'] * B['uy'] < 0.85:
                continue
            ux, uy = grupo['ux'], grupo['uy']
            ds = (B['cx'] - grupo['cx']) * ux + (B['cy'] - grupo['cy']) * uy
            dt = (B['cx'] - grupo['cx']) * (-uy) + (B['cy'] - grupo['cy']) * ux
            gap = abs(ds) - (grupo['comprimento'] + B['comprimento']) / 2
            if gap > 1.2 or abs(dt) > 0.8:
                continue
            # funde: extensao ao longo de u + z range
            s_lo = min(-grupo['comprimento'] / 2, ds - B['comprimento'] / 2)
            s_hi = max(grupo['comprimento'] / 2, ds + B['comprimento'] / 2)
            grupo['cx'] += ux * (s_lo + s_hi) / 2
            grupo['cy'] += uy * (s_lo + s_hi) / 2
            grupo['comprimento'] = s_hi - s_lo
            grupo['largura'] = max(grupo['largura'], B['largura'])
            grupo['z0'] = min(grupo['z0'], B['z0'])
            grupo['z1'] = max(grupo['z1'], B['z1'])
            grupo['n_degraus_vistos'] += B['n_degraus_vistos']
            usados.add(j)
        fundidos.append(grupo)
    return fundidos


def inserir_escadas(ifc_path_in, ifc_path_out, lances_por_banda):
    import ifcopenshell
    import ifcopenshell.guid
    m = ifcopenshell.open(str(ifc_path_in))
    owner = m.by_type('IfcOwnerHistory')[0]
    ctx = m.by_type('IfcGeometricRepresentationContext')[0]
    zdir = m.create_entity('IfcDirection', DirectionRatios=(0.0, 0.0, 1.0))
    rel = m.by_type('IfcRelContainedInSpatialStructure')
    lajes_ent = _lajes_entidades(m)
    n = 0
    for banda, lances in lances_por_banda.items():
        for L in lances:
            riser = L['riser_h'] if L['riser_h'] == L['riser_h'] else 0.17
            # ANCORAGEM: estica o lance ate o piso de partida e a laje de chegada
            # (a margem da deteccao come os primeiros/ultimos degraus; sem isso
            # lances parciais terminam no ar e nao cortam vao na laje)
            piso = max((z0 + esp for z0, esp, _ in lajes_ent if z0 + esp <= L['z0'] + 0.15),
                       default=None)
            chegada = min(((z0, esp) for z0, esp, _ in lajes_ent
                           if L['z1'] - 0.3 <= z0 <= L['z1'] + 1.0),
                          default=None)
            sl = max(L['slope'], 0.2)
            if piso is not None and 0 < L['z0'] - piso <= 0.7:
                d = (L['z0'] - piso) / sl
                L['cx'] -= L['ux'] * d / 2
                L['cy'] -= L['uy'] * d / 2
                L['comprimento'] += d
                L['z0'] = piso
            if chegada is not None:
                topo_alvo = chegada[0] + chegada[1]          # topo da laje de chegada
                if 0 < topo_alvo - L['z1'] <= 1.0:
                    d = (topo_alvo - L['z1']) / sl
                    L['cx'] += L['ux'] * d / 2
                    L['cy'] += L['uy'] * d / 2
                    L['comprimento'] += d
                    L['z1'] = topo_alvo
            n_ris = max(2, int(round((L['z1'] - L['z0']) / riser)))
            tread = L['comprimento'] / max(n_ris - 1, 1)
            ux, uy = L['ux'], L['uy']
            px, py = -uy, ux                      # perpendicular (largura)
            w2 = L['largura'] / 2
            # ponto de partida: base do lance (menor z ao longo de u)
            bx = L['cx'] - ux * L['comprimento'] / 2
            by = L['cy'] - uy * L['comprimento'] / 2
            solids = []
            for i in range(n_ris):
                s0, s1 = i * tread, (i + 1) * tread
                zt = L['z0'] + (i + 1) * riser
                quad = [(bx + ux * s0 + px * w2, by + uy * s0 + py * w2),
                        (bx + ux * s1 + px * w2, by + uy * s1 + py * w2),
                        (bx + ux * s1 - px * w2, by + uy * s1 - py * w2),
                        (bx + ux * s0 - px * w2, by + uy * s0 - py * w2)]
                ptsi = [m.create_entity('IfcCartesianPoint', Coordinates=(float(q[0]), float(q[1])))
                        for q in quad]
                poly = m.create_entity('IfcPolyline', Points=ptsi + [ptsi[0]])
                prof = m.create_entity('IfcArbitraryClosedProfileDef', ProfileType='AREA',
                                       OuterCurve=poly)
                pos = m.create_entity('IfcAxis2Placement3D',
                                      Location=m.create_entity('IfcCartesianPoint',
                                                               Coordinates=(0.0, 0.0, float(zt - riser))))
                solids.append(m.create_entity('IfcExtrudedAreaSolid', SweptArea=prof,
                                              Position=pos, ExtrudedDirection=zdir,
                                              Depth=float(riser)))
            shape = m.create_entity('IfcShapeRepresentation', ContextOfItems=ctx,
                                    RepresentationIdentifier='Body',
                                    RepresentationType='SweptSolid', Items=solids)
            pds = m.create_entity('IfcProductDefinitionShape', Representations=[shape])
            lp = m.create_entity('IfcLocalPlacement', RelativePlacement=m.create_entity(
                'IfcAxis2Placement3D',
                Location=m.create_entity('IfcCartesianPoint', Coordinates=(0.0, 0.0, 0.0))))
            n += 1
            stair = m.create_entity('IfcStair', GlobalId=ifcopenshell.guid.new(),
                                    OwnerHistory=owner, Name=f'Escada {n}',
                                    Description=f'Lance reto: {n_ris} degraus, espelho '
                                                f'{riser*100:.0f}cm, declividade {L["slope"]:.2f}',
                                    ObjectPlacement=lp, Representation=pds,
                                    PredefinedType='STRAIGHT_RUN_STAIR')
            if rel:
                r0 = rel[0]
                r0.RelatedElements = list(r0.RelatedElements) + [stair]

    # ---- RECONCILIACAO escada <-> laje/parede -------------------------------
    import ifcopenshell.util.placement as P
    lajes = _lajes_entidades(m)
    zdir2 = m.create_entity('IfcDirection', DirectionRatios=(0.0, 0.0, 1.0))
    n_open = 0
    for banda, lances in lances_por_banda.items():
        for L in lances:
            # laje de CHEGADA: a primeira cujo fundo esta logo acima do topo do lance
            alvo = None
            for z0, esp, ent in lajes:
                if L['z1'] - 0.3 <= z0 <= L['z1'] + 0.6:
                    alvo = (z0, esp, ent)
                    break
            if alvo is None:
                continue
            z0, esp, ent = alvo
            # recorte sobre o trecho FINAL do lance (onde ele atravessa a laje)
            ux, uy = L['ux'], L['uy']
            px_, py_ = -uy, ux
            w2 = L['largura'] / 2 + 0.15
            bx = L['cx'] - ux * L['comprimento'] / 2
            by = L['cy'] - uy * L['comprimento'] / 2
            s0, s1 = 0.35 * L['comprimento'], L['comprimento'] + 0.30
            quad = [(bx + ux * s0 + px_ * w2, by + uy * s0 + py_ * w2),
                    (bx + ux * s1 + px_ * w2, by + uy * s1 + py_ * w2),
                    (bx + ux * s1 - px_ * w2, by + uy * s1 - py_ * w2),
                    (bx + ux * s0 - px_ * w2, by + uy * s0 - py_ * w2)]
            ptsq = [m.create_entity('IfcCartesianPoint', Coordinates=(float(q[0]), float(q[1])))
                    for q in quad]
            poly = m.create_entity('IfcPolyline', Points=ptsq + [ptsq[0]])
            prof = m.create_entity('IfcArbitraryClosedProfileDef', ProfileType='AREA',
                                   OuterCurve=poly)
            pos = m.create_entity('IfcAxis2Placement3D',
                                  Location=m.create_entity('IfcCartesianPoint',
                                                           Coordinates=(0.0, 0.0, float(z0 - 0.05))))
            solid = m.create_entity('IfcExtrudedAreaSolid', SweptArea=prof, Position=pos,
                                    ExtrudedDirection=zdir2, Depth=float(esp + 0.10))
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
                            OwnerHistory=owner, RelatingBuildingElement=ent,
                            RelatedOpeningElement=op)
            n_open += 1

    # paredes-fantasma: eixo dentro do footprint de um lance = contorno do vao
    removidas = 0
    for w in list(m.by_type('IfcWall')):
        zw = float(P.get_local_placement(w.ObjectPlacement)[2, 3])
        eixo = None
        for r in w.Representation.Representations:
            if r.RepresentationIdentifier == 'Axis':
                for it in r.Items:
                    if it.is_a('IfcPolyline'):
                        mat = P.get_local_placement(w.ObjectPlacement)
                        eixo = [(mat @ np.array(list(p.Coordinates)[:2] + [0, 1.0]))[:2]
                                for p in it.Points]
        if eixo is None:
            continue
        mx_, my_ = (eixo[0][0] + eixo[-1][0]) / 2, (eixo[0][1] + eixo[-1][1]) / 2
        for banda, lances in lances_por_banda.items():
            for L in lances:
                if L['z0'] - 1.0 <= zw <= L['z1'] + 0.5 and _dentro_do_lance(mx_, my_, L):
                    for rel2 in m.by_type('IfcRelContainedInSpatialStructure'):
                        if w in rel2.RelatedElements:
                            rel2.RelatedElements = [e for e in rel2.RelatedElements if e != w]
                    m.remove(w)
                    removidas += 1
                    break
            else:
                continue
            break
    print(f'      reconciliacao: {n_open} vaos de escada cortados nas lajes, '
          f'{removidas} paredes-fantasma (contorno do vao) removidas')

    m.write(str(ifc_path_out))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('xyz')
    ap.add_argument('ifc_in')
    ap.add_argument('ifc_out')
    ap.add_argument('--png', default=None)
    args = ap.parse_args()

    print(f'[1/4] Carregando {Path(args.xyz).name}...')
    pts = np.loadtxt(args.xyz, skiprows=1, usecols=(0, 1, 2))
    bandas = bandas_das_lajes(args.ifc_in)
    print(f'      {len(pts):,} pontos | {len(bandas)} bandas de pavimento')

    print('[2/4] Detectando lances por banda...')
    lances_por_banda = {}
    for k, (zlo, zhi) in enumerate(bandas):
        if zhi - zlo < 1.2:
            continue
        lances = detectar_lances(pts, zlo, zhi)
        lances = fundir_lances(lances)
        if lances:
            lances_por_banda[k] = lances
        for L in lances:
            print(f'      banda {k} (z {zlo:.1f}..{zhi:.1f}): lance em ({L["cx"]:.1f},{L["cy"]:.1f}) '
                  f'| {L["comprimento"]:.1f}m x {L["largura"]:.1f}m | declividade {L["slope"]:.2f} '
                  f'| espelho {L["riser_h"]*100 if L["riser_h"]==L["riser_h"] else float("nan"):.0f}cm '
                  f'| {L["n_degraus_vistos"]} degraus vistos | rms {L["rms"]*100:.0f}cm')
    total = sum(len(v) for v in lances_por_banda.values())
    print(f'      total: {total} lances')

    print('[3/4] Inserindo IfcStair...')
    n = inserir_escadas(args.ifc_in, args.ifc_out, lances_por_banda)
    print(f'      {n} IfcStair (STRAIGHT_RUN_STAIR) salvos em {args.ifc_out}')

    print('[4/4] Renderizando verificacao...')
    png = args.png or str(Path(args.ifc_out).with_suffix('')) + '_escadas.png'
    ncol = max(len(lances_por_banda), 1)
    fig, axes = plt.subplots(1, ncol, figsize=(5.5 * ncol, 5))
    axes = np.atleast_1d(axes)
    for ax, (k, lances) in zip(axes, sorted(lances_por_banda.items())):
        zlo, zhi = bandas[k]
        sl = pts[(pts[:, 2] >= zlo + 0.2) & (pts[:, 2] < zhi - 0.2)]
        reg = sl[(sl[:, 0] > min(L['x0'] for L in lances) - 1.5) &
                 (sl[:, 0] < max(L['x1'] for L in lances) + 1.5) &
                 (sl[:, 1] > min(L['y0'] for L in lances) - 1.5) &
                 (sl[:, 1] < max(L['y1'] for L in lances) + 1.5)]
        ax.scatter(reg[:, 0], reg[:, 1], s=1.0, c=reg[:, 2], cmap='Blues', linewidths=0)
        for L in lances:
            ax.plot([L['x0'], L['x1'], L['x1'], L['x0'], L['x0']],
                    [L['y0'], L['y0'], L['y1'], L['y1'], L['y0']], '-', c='#c22', lw=2)
            ax.annotate(f"{L['n_degraus_vistos']}dg", (L['cx'], L['cy']),
                        color='#c22', fontsize=10, fontweight='bold', ha='center')
        ax.set_aspect('equal')
        ax.set_title(f'banda {k} (z {zlo:.1f}..{zhi:.1f}): {len(lances)} lances', fontsize=10)
    plt.tight_layout()
    plt.savefig(png, dpi=130, bbox_inches='tight')
    print(f'      verificacao: {png}')


if __name__ == '__main__':
    main()
