# -*- coding: utf-8 -*-
"""Detector de PILARES v2 — filtro de constancia de secao ao longo de z.

Pilar real (concreto/aco): mesma secao transversal em TODA altura, centroide
parado. Rack/movel: secao varia por nivel (prateleia cheia/vazia), centroide
deriva. Esse teste e o que separa P3 (real) dos falsos P1/P2/P4 da v1.

Cacheia a grade 3D em occ_cache.npz pra iterar sem reler os 32M de pontos.
"""
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

XYZ = 'input_xyz/real.xyz'
IFC = 'output_IFC/real_cloud2bim.ifc'
OUT = 'real_pilares_v2.png'
CACHE = 'occ_cache.npz'

CELL = 0.05
ZLO, ZHI = -0.95, 1.25
ZBIN = 0.10
MIN_PTS_BIN = 2
FULL_FRAC = 0.70

# candidato (fase 1, frouxo)
CAND_MIN, CAND_MAX, CAND_ASPECT = 0.10, 1.2, 4.0
# aceitacao (fase 2): constancia de secao + forma de secao de pilar
CONSIST_MIN = 0.95     # fracao de z-bins com area >= 60% da mediana
AREA_RATIO = 0.60
DRIFT_MAX = 0.10       # deriva maxima do centroide (m)
MINRATIO_MIN = 0.55    # pior bin vs mediana (armario vazio no meio cai aqui)
SEC_MIN = 0.15         # lado minimo da secao (m) — pilar estrutural
SEC_ASPECT = 2.5       # aspecto maximo da secao (armario 30x90 = 3.0 cai)

# ---------- grade 3D (com cache) ----------
if os.path.exists(CACHE):
    print('[1/4] Carregando cache da grade 3D...')
    d = np.load(CACHE)
    occ, total_cnt = d['occ'], d['total_cnt']
    xmin, ymin = float(d['xmin']), float(d['ymin'])
    NZ, NY, NX = occ.shape
else:
    print('[1/4] Lendo nuvem em chunks (primeira vez, vai cachear)...')
    mins, maxs = [], []
    for c in pd.read_csv(XYZ, sep='\t', usecols=[0, 1, 2], chunksize=4_000_000):
        a = c.to_numpy(); mins.append(a.min(0)); maxs.append(a.max(0))
    mn, mx = np.min(mins, 0), np.max(maxs, 0)
    xmin, ymin = mn[0], mn[1]
    NX = int(np.ceil((mx[0] - xmin) / CELL)) + 1
    NY = int(np.ceil((mx[1] - ymin) / CELL)) + 1
    NZ = int(np.ceil((ZHI - ZLO) / ZBIN))
    occ = np.zeros((NZ, NY, NX), dtype=np.uint16)
    total_cnt = np.zeros((NY, NX), dtype=np.uint32)
    for c in pd.read_csv(XYZ, sep='\t', usecols=[0, 1, 2], chunksize=4_000_000):
        a = c.to_numpy()
        ix = ((a[:, 0] - xmin) / CELL).astype(np.int32)
        iy = ((a[:, 1] - ymin) / CELL).astype(np.int32)
        np.add.at(total_cnt, (iy, ix), 1)
        m = (a[:, 2] >= ZLO) & (a[:, 2] < ZHI)
        iz = ((a[m, 2] - ZLO) / ZBIN).astype(np.int32)
        np.add.at(occ, (iz, iy[m], ix[m]), 1)
    np.savez_compressed(CACHE, occ=occ, total_cnt=total_cnt, xmin=xmin, ymin=ymin)
    print(f'   cache salvo em {CACHE}')

print('[2/4] Mascara full-height + candidatos compactos...')
occ_bin = (occ >= MIN_PTS_BIN)
frac = occ_bin.sum(0) / float(NZ)
full_mask = (frac >= FULL_FRAC).astype(np.uint8) * 255
full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cands = []
for cnt in contours:
    if len(cnt) < 3:
        continue
    (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
    lo, hi = sorted([w * CELL, h * CELL])
    if lo < 0.06 or hi > CAND_MAX or lo < CAND_MIN or hi / max(lo, 1e-6) > CAND_ASPECT:
        continue
    cands.append({'cnt': cnt, 'cx': xmin + cx * CELL, 'cy': ymin + cy * CELL,
                  'w': w * CELL, 'h': h * CELL})
print(f'   {len(cands)} candidatos compactos full-height')

print('[3/4] Teste de constancia de secao ao longo de z...')
aceitos, rejeitados = [], []
for cd in cands:
    blob = np.zeros((NY, NX), dtype=np.uint8)
    cv2.drawContours(blob, [cd['cnt']], -1, 1, thickness=-1)
    blob = blob.astype(bool)
    areas, cents = [], []
    for z in range(NZ):
        sel = occ_bin[z] & blob
        n = int(sel.sum())
        areas.append(n)
        if n > 0:
            ys, xs = np.nonzero(sel)
            cents.append((xs.mean(), ys.mean()))
    areas = np.array(areas)
    med = np.median(areas[areas > 0]) if (areas > 0).any() else 0
    consist = float((areas >= AREA_RATIO * med).sum()) / NZ if med > 0 else 0.0
    if len(cents) >= 2:
        cents = np.array(cents)
        drift = float(np.linalg.norm(cents - cents.mean(0), axis=1).max()) * CELL
    else:
        drift = 99.0
    nz_areas = areas[areas > 0]
    area_cv = float(nz_areas.std() / med) if med > 0 and len(nz_areas) > 1 else 9.9
    min_ratio = float(areas.min() / med) if med > 0 else 0.0
    cd['consist'], cd['drift'] = consist, drift
    cd['area_cv'], cd['min_ratio'] = area_cv, min_ratio
    lo_s, hi_s = sorted([cd['w'], cd['h']])
    cd['ok'] = bool(consist >= CONSIST_MIN and drift <= DRIFT_MAX
                    and min_ratio >= MINRATIO_MIN
                    and lo_s >= SEC_MIN and hi_s / max(lo_s, 1e-6) <= SEC_ASPECT)
    (aceitos if cd['ok'] else rejeitados).append(cd)

print(f'   {len(aceitos)} aceitos | {len(rejeitados)} rejeitados')
print('\n   candidato          secao (m)   consist  drift(m)  areaCV  minRatio  veredito')
for cd in cands:
    v = 'PILAR' if cd['ok'] else 'rejeitado'
    print(f'   ({cd["cx"]:7.2f},{cd["cy"]:6.2f})  {cd["w"]:.2f}x{cd["h"]:.2f}   '
          f'{cd["consist"]:.2f}     {cd["drift"]:.3f}   {cd["area_cv"]:.2f}    '
          f'{cd["min_ratio"]:.2f}      {v}')

print('[4/4] Renderizando...')
import ifcopenshell, ifcopenshell.util.placement as P
def wall_axes(fn):
    m = ifcopenshell.open(fn); segs = []
    for w in m.by_type('IfcWall'):
        mat = P.get_local_placement(w.ObjectPlacement); rep = None
        for r in w.Representation.Representations:
            if r.RepresentationIdentifier == 'Axis':
                rep = r
        if not rep:
            continue
        for it in rep.Items:
            if it.is_a('IfcPolyline'):
                pts = [(mat @ np.array(list(p.Coordinates)[:2] + [0, 1.0]))[:2] for p in it.Points]
                segs.append(np.array(pts))
    return segs

fig, ax = plt.subplots(figsize=(16, 7))
ax.imshow(np.log1p(total_cnt), cmap='Greys', origin='lower', alpha=0.75,
          extent=(xmin, xmin + NX * CELL, ymin, ymin + NY * CELL), aspect='equal')
for s in wall_axes(IFC):
    ax.plot(s[:, 0], s[:, 1], '-', c='#ffb3b3', lw=1.4, zorder=2)
for i, cd in enumerate(rejeitados):
    ax.plot(cd['cx'], cd['cy'], 'x', ms=10, c='orange', mew=2.5, zorder=4)
    ax.annotate(f'R{i+1}', (cd['cx'], cd['cy']), xytext=(6, 6), textcoords='offset points',
                color='#b36b00', fontsize=10, fontweight='bold', zorder=5)
for i, cd in enumerate(aceitos):
    ax.plot(cd['cx'], cd['cy'], 's', ms=11, mfc='none', mec='blue', mew=2.5, zorder=4)
    ax.annotate(f'P{i+1}', (cd['cx'], cd['cy']), xytext=(7, 7), textcoords='offset points',
                color='blue', fontsize=12, fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='blue', alpha=0.9))
ax.set_title(f'Pilares v2: aceitos (azul, {len(aceitos)}) | rejeitados pelo teste de '
             f'constancia (laranja X, {len(rejeitados)})')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
plt.tight_layout(); plt.savefig(OUT, dpi=130)
print(f'OK -> {OUT}')
