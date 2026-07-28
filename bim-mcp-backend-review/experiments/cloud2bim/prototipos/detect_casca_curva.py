# -*- coding: utf-8 -*-
"""Detector de CASCA CURVA (telhado em arco/barril) — protótipo v2.

A laje plana falha em telhado curvo porque a premissa "laje = pico estreito no
histograma de z" é falsa: a curva espalha os pontos por toda a faixa de z.
Assinatura da casca: LOCALMENTE FINA (spread de z pequeno na célula XY — o
oposto de parede) mas GLOBALMENTE CURVA (z varia suave entre células).

v2: SEGMENTA a cobertura em trechos de perfil constante ao longo de x antes de
ajustar (prédio pode ter mais de um barril — ex. Allplan: barra principal +
núcleo central mais alto). Um arco Kasa + um IfcRoof por trecho.

Uso:
  python detect_casca_curva.py <nuvem.xyz> <entrada.ifc> <saida.ifc> --zmin 9.1
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def kasa_fit(uv):
    A = np.column_stack([uv[:, 0], uv[:, 1], np.ones(len(uv))])
    b = (uv ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cu, cv = sol[0] / 2, sol[1] / 2
    r = float(np.sqrt(sol[2] + cu ** 2 + cv ** 2))
    return float(cu), float(cv), r


def fit_arco_robusto(uv, iters=4):
    keep = np.ones(len(uv), bool)
    cu = cv = r = None
    for _ in range(iters):
        cu, cv, r = kasa_fit(uv[keep])
        res = np.abs(np.hypot(uv[:, 0] - cu, uv[:, 1] - cv) - r)
        mad = np.median(np.abs(res - np.median(res))) + 1e-9
        keep = res < 4 * mad + 0.05
    rms = float(np.sqrt(((np.hypot(uv[keep, 0] - cu, uv[keep, 1] - cv) - r) ** 2).mean()))
    return cu, cv, r, keep, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('xyz')
    ap.add_argument('ifc_in')
    ap.add_argument('ifc_out')
    ap.add_argument('--zmin', type=float, required=True,
                    help='cota acima da qual procurar a casca (topo do ultimo pavimento)')
    ap.add_argument('--cell', type=float, default=0.20)
    ap.add_argument('--esp-max', type=float, default=0.45,
                    help='espessura local maxima pra célula contar como casca')
    ap.add_argument('--esp-casca', type=float, default=0.25,
                    help='espessura construida do IfcRoof')
    ap.add_argument('--corte-perfil', type=float, default=0.30,
                    help='salto medio de z entre colunas x que corta um novo trecho')
    ap.add_argument('--png', default=None)
    args = ap.parse_args()

    print(f'[1/6] Carregando {Path(args.xyz).name} (z > {args.zmin})...')
    pts = np.loadtxt(args.xyz, skiprows=1, usecols=(0, 1, 2))
    sel = pts[pts[:, 2] > args.zmin]
    print(f'      {len(sel):,} pontos candidatos a cobertura')

    # ---- 1. mapa de altura + espessura local por célula
    print('[2/6] Mapa de altura (topo + espessura local por célula)...')
    cell = args.cell
    xmin, ymin = sel[:, 0].min(), sel[:, 1].min()
    ix = ((sel[:, 0] - xmin) / cell).astype(np.int64)
    iy = ((sel[:, 1] - ymin) / cell).astype(np.int64)
    NX, NY = ix.max() + 1, iy.max() + 1
    key = iy * NX + ix
    order = np.argsort(key)
    ks, zs = key[order], sel[order, 2]
    uniq, starts = np.unique(ks, return_index=True)
    z_top = np.full(NX * NY, np.nan)
    esp = np.full(NX * NY, np.nan)
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(ks)
        zc = zs[s:e]
        if len(zc) < 4:
            continue
        z_top[uniq[i]] = np.median(zc[zc >= zc.max() - 0.15])
        esp[uniq[i]] = np.percentile(zc, 95) - np.percentile(zc, 5)
    z_top = z_top.reshape(NY, NX)
    esp = esp.reshape(NY, NX)
    casca = np.isfinite(z_top) & (esp < args.esp_max)
    zt = np.where(casca, z_top, np.nan)
    print(f'      {int(casca.sum())} células de casca | '
          f'{int((np.isfinite(z_top) & ~casca).sum())} grossas excluídas (paredes/empenas)')

    # ---- 2. segmenta em trechos de perfil constante ao longo de x
    print('[3/6] Segmentando trechos de perfil constante em x...')
    validas = np.where(np.isfinite(zt).sum(0) >= 5)[0]   # colunas x com casca
    # compara cada coluna valida com a PROXIMA valida (pulando cortinas de NaN
    # deixadas pelas empenas excluidas — e cortando tambem em vaos largos)
    trechos, atual = [], [int(validas[0])]
    for j1, j2 in zip(validas[:-1], validas[1:]):
        c1, c2 = zt[:, j1], zt[:, j2]
        ambos = np.isfinite(c1) & np.isfinite(c2)
        salto = float(np.abs(c1[ambos] - c2[ambos]).mean()) if ambos.sum() >= 3 else np.inf
        vao = (j2 - j1) * cell
        if salto > args.corte_perfil or vao > 0.8:
            trechos.append((atual[0], int(j1) + 1))
            atual = [int(j2)]
    trechos.append((atual[0], int(validas[-1]) + 1))
    trechos = [(a, b) for a, b in trechos if (b - a) * cell >= 1.5]
    print(f'      {len(trechos)} trecho(s): ' + ', '.join(
        f'x {xmin+a*cell:.1f}..{xmin+b*cell:.1f}' for a, b in trechos))

    # ---- 3. fit de arco por trecho
    print('[4/6] Fit de arco (Kasa robusto) por trecho...')
    resultados = []
    for a, b in trechos:
        yy, xx = np.nonzero(casca[:, a:b])
        uv = np.column_stack([ymin + yy * cell + cell / 2, z_top[yy, xx + a]])
        # extrusao dentro do trecho
        sub = zt[:, a:b]
        var_x = np.nanmean(np.nanstd(sub, axis=1))
        cu, cv, R, keep, rms = fit_arco_robusto(uv)
        y_lo, y_hi = float(uv[keep, 0].min()), float(uv[keep, 0].max())
        resultados.append({'x_lo': xmin + a * cell, 'x_hi': xmin + b * cell,
                           'cy': cu, 'cz': cv, 'R': R, 'rms': rms,
                           'y_lo': y_lo, 'y_hi': y_hi, 'apex': cv + R,
                           'var_x': var_x, 'uv': uv, 'keep': keep})
        print(f'      x {xmin+a*cell:6.1f}..{xmin+b*cell:6.1f} | R={R:6.2f}m | '
              f'cumeeira z={cv+R:5.2f} | vao {y_hi-y_lo:4.1f}m | RMS {rms*100:4.1f}cm | '
              f'extrusao(var_x)={var_x:.3f}m')

    # ---- 4. IFC: um IfcRoof por trecho + remove laje plana espuria
    print('[5/6] Inserindo IfcRoof(s) e removendo laje(s) plana(s) espuria(s)...')
    import ifcopenshell
    import ifcopenshell.guid
    import ifcopenshell.util.placement as P
    m = ifcopenshell.open(args.ifc_in)

    z_shell_min = float(np.nanmin(zt))
    removidas = 0
    for s in list(m.by_type('IfcSlab')):
        z0 = P.get_local_placement(s.ObjectPlacement)[2, 3]
        if z0 > z_shell_min - 0.3:
            for rel in m.by_type('IfcRelContainedInSpatialStructure'):
                if s in rel.RelatedElements:
                    rel.RelatedElements = [e for e in rel.RelatedElements if e != s]
            m.remove(s)
            removidas += 1
    print(f'      {removidas} laje(s) espuria(s) removida(s)')

    owner = m.by_type('IfcOwnerHistory')[0]
    ctx = m.by_type('IfcGeometricRepresentationContext')[0]
    rel = m.by_type('IfcRelContainedInSpatialStructure')
    for k, t in enumerate(resultados):
        cy, cz, R = t['cy'], t['cz'], t['R']
        # angulo direto da geometria (ramo superior do circulo) — sem interp
        def th_at(y):
            dz = np.sqrt(max(R ** 2 - (y - cy) ** 2, 0.0))
            return float(np.arctan2(dz, y - cy))
        ths = np.linspace(th_at(t['y_lo']), th_at(t['y_hi']), 24)
        externo = [(cy + R * np.cos(a), cz + R * np.sin(a)) for a in ths]
        Ri = R - args.esp_casca
        interno = [(cy + Ri * np.cos(a), cz + Ri * np.sin(a)) for a in ths[::-1]]
        pts_ifc = [m.create_entity('IfcCartesianPoint', Coordinates=(float(u), float(v)))
                   for u, v in externo + interno]
        poly = m.create_entity('IfcPolyline', Points=pts_ifc + [pts_ifc[0]])
        prof = m.create_entity('IfcArbitraryClosedProfileDef', ProfileType='AREA',
                               ProfileName=f'Arco {k+1}', OuterCurve=poly)
        # sistema local: X_local=Y_mundo, Y_local=Z_mundo, Z_local(extrusao)=X_mundo
        pos = m.create_entity(
            'IfcAxis2Placement3D',
            Location=m.create_entity('IfcCartesianPoint', Coordinates=(float(t['x_lo']), 0.0, 0.0)),
            Axis=m.create_entity('IfcDirection', DirectionRatios=(1.0, 0.0, 0.0)),
            RefDirection=m.create_entity('IfcDirection', DirectionRatios=(0.0, 1.0, 0.0)))
        solid = m.create_entity('IfcExtrudedAreaSolid', SweptArea=prof, Position=pos,
                                ExtrudedDirection=m.create_entity(
                                    'IfcDirection', DirectionRatios=(0.0, 0.0, 1.0)),
                                Depth=float(t['x_hi'] - t['x_lo']))
        shape = m.create_entity('IfcShapeRepresentation', ContextOfItems=ctx,
                                RepresentationIdentifier='Body',
                                RepresentationType='SweptSolid', Items=[solid])
        pds = m.create_entity('IfcProductDefinitionShape', Representations=[shape])
        lp = m.create_entity('IfcLocalPlacement', RelativePlacement=m.create_entity(
            'IfcAxis2Placement3D',
            Location=m.create_entity('IfcCartesianPoint', Coordinates=(0.0, 0.0, 0.0))))
        roof = m.create_entity('IfcRoof', GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner,
                               Name=f'Telhado em arco {k+1}',
                               Description=f'R={R:.2f}m, cumeeira z={t["apex"]:.2f}m, '
                                           f'RMS={t["rms"]*100:.1f}cm',
                               ObjectPlacement=lp, Representation=pds,
                               PredefinedType='BARREL_ROOF')
        if rel:
            r0 = rel[-1]
            r0.RelatedElements = list(r0.RelatedElements) + [roof]
    # ---- reconciliacao: o telhado reivindicou a zona; ajusta o resto do modelo
    z_beiral = float(min(t['uv'][t['keep']][:, 1].min() for t in resultados))
    print(f'      reconciliacao (beiral z={z_beiral:.2f}):')

    # paredes que invadem a zona do telhado: apara no beiral (nao deleta —
    # parede de apoio ate o beiral e real); parede inteira acima do beiral sai
    aparadas, removidas_w = 0, 0
    for w in list(m.by_type('IfcWall')):
        z0 = P.get_local_placement(w.ObjectPlacement)[2, 3]
        corpo = None
        for r in w.Representation.Representations:
            if r.RepresentationIdentifier == 'Body':
                for it in r.Items:
                    if it.is_a('IfcExtrudedAreaSolid'):
                        corpo = it
        if corpo is None:
            continue
        topo = z0 + corpo.Depth
        if z0 >= z_beiral - 0.1:
            for rel2 in m.by_type('IfcRelContainedInSpatialStructure'):
                if w in rel2.RelatedElements:
                    rel2.RelatedElements = [e for e in rel2.RelatedElements if e != w]
            m.remove(w)
            removidas_w += 1
        elif topo > z_beiral + 0.15:
            corpo.Depth = float(max(z_beiral - z0, 0.3))
            aparadas += 1
    print(f'        {aparadas} paredes aparadas no beiral, {removidas_w} removidas')

    # spaces do sotao fantasma (caixas em cima do telhado no viewer)
    removidos_sp = 0
    for sp in list(m.by_type('IfcSpace')):
        zsp = P.get_local_placement(sp.ObjectPlacement)[2, 3]
        if zsp >= z_beiral - 1.2:
            for rel2 in list(m.by_type('IfcRelAggregates')) + list(m.by_type('IfcRelContainedInSpatialStructure')):
                if hasattr(rel2, 'RelatedObjects') and sp in (rel2.RelatedObjects or ()):
                    rel2.RelatedObjects = [e for e in rel2.RelatedObjects if e != sp]
                if hasattr(rel2, 'RelatedElements') and sp in (rel2.RelatedElements or ()):
                    rel2.RelatedElements = [e for e in rel2.RelatedElements if e != sp]
            m.remove(sp)
            removidos_sp += 1
    print(f'        {removidos_sp} spaces do sotao removidos')

    # storeys fantasmas vazios acima do beiral
    removidos_st = 0
    for st in list(m.by_type('IfcBuildingStorey')):
        zst = P.get_local_placement(st.ObjectPlacement)[2, 3]
        if zst < z_beiral:
            continue
        tem_algo = False
        for rel2 in m.by_type('IfcRelContainedInSpatialStructure'):
            if rel2.RelatingStructure == st and len(rel2.RelatedElements) > 0:
                tem_algo = True
        if not tem_algo:
            for rel2 in list(m.by_type('IfcRelContainedInSpatialStructure')):
                if rel2.RelatingStructure == st:
                    m.remove(rel2)
            for rel2 in m.by_type('IfcRelAggregates'):
                if st in (rel2.RelatedObjects or ()):
                    rel2.RelatedObjects = [e for e in rel2.RelatedObjects if e != st]
            m.remove(st)
            removidos_st += 1
    print(f'        {removidos_st} storey(s) fantasma(s) removido(s)')

    m.write(args.ifc_out)
    print(f'      {len(resultados)} IfcRoof (BARREL_ROOF) salvos em {args.ifc_out}')

    # ---- 5. verificacao visual: um corte por trecho + mapa de altura
    print('[6/6] Renderizando verificacao...')
    png = args.png or str(Path(args.ifc_out).with_suffix('')) + '_telhado.png'
    ncols = len(resultados) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
    axes = np.atleast_1d(axes)
    for k, t in enumerate(resultados):
        ax = axes[k]
        xc = (t['x_lo'] + t['x_hi']) / 2
        faixa = sel[np.abs(sel[:, 0] - xc) < 0.4]
        ax.scatter(faixa[:, 1], faixa[:, 2], s=1.5, c='#9ecae1')
        tt = np.linspace(np.arctan2(t['uv'][t['keep']][:, 1].min() - t['cz'], t['y_lo'] - t['cy']),
                         np.arctan2(t['uv'][t['keep']][:, 1].min() - t['cz'], t['y_hi'] - t['cy']), 100)
        yyv = np.linspace(t['y_lo'], t['y_hi'], 100)
        # desenha o arco resolvendo z(y) do circulo (ramo superior)
        dz = np.sqrt(np.maximum(t['R'] ** 2 - (yyv - t['cy']) ** 2, 0))
        ax.plot(yyv, t['cz'] + dz, '-', c='#c22', lw=2.2,
                label=f"R={t['R']:.2f}m RMS {t['rms']*100:.1f}cm")
        ax.set_title(f"Trecho {k+1}: corte em x={xc:.1f}", fontsize=10)
        ax.set_xlabel('y (m)'); ax.set_ylabel('z (m)')
        ax.legend(fontsize=8); ax.set_aspect('equal')
    ax = axes[-1]
    im = ax.imshow(zt, origin='lower', cmap='Blues',
                   extent=(xmin, xmin + NX * cell, ymin, ymin + NY * cell), aspect='equal')
    for t in resultados:
        ax.axvline(t['x_lo'], color='#c22', lw=1.2, ls='--')
        ax.axvline(t['x_hi'], color='#c22', lw=1.2, ls='--')
    plt.colorbar(im, ax=ax, fraction=0.03, label='z topo (m)')
    ax.set_title('Mapa de altura + cortes de trecho', fontsize=10)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    plt.tight_layout(); plt.savefig(png, dpi=130, bbox_inches='tight')
    print(f'      verificacao: {png}')


if __name__ == '__main__':
    main()
