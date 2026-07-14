# -*- coding: utf-8 -*-
"""Detector de pilares por perfil de ocupacao vertical (versao integrada).

Pilar = celula XY com ocupacao continua do chao ao teto (como parede), mas com
footprint COMPACTO em planta (blob, nao linha) e SECAO CONSTANTE em z.
A combinacao validada no scan RCP (1 pilar real, 0 falsos):
  - constancia de secao ao longo de z (rack/prateleira varia por nivel -> cai)
  - deriva do centroide baixa (tralha encostada deriva -> cai)
  - forma de secao de pilar: lado >= 15cm, aspecto <= 2.5 (armario 30x90 -> cai)

Uso como modulo (pelo rodar.py):
    from detect_pilar import find_pillars
    pilares = find_pillars(pts, zlo=-1.0, zhi=1.3)
"""
import numpy as np
import cv2


def find_pillars(pts, zlo, zhi, cell=0.05, zbin=0.10, min_pts_bin=2, full_frac=0.70,
                 cand_min=0.10, cand_max=1.2, cand_aspect=4.0,
                 consist_min=0.95, area_ratio=0.60, drift_max=0.10,
                 minratio_min=0.55, sec_min=0.15, sec_aspect=2.5):
    """Detecta pilares na banda [zlo, zhi) de uma nuvem (N,3).

    Retorna lista de dicts: cx, cy, w, h (secao em m), consist, drift, ok.
    So os com ok=True passaram em todos os filtros; os demais sao candidatos
    rejeitados (uteis pra debug/visualizacao).
    """
    sel = pts[(pts[:, 2] >= zlo) & (pts[:, 2] < zhi)]
    if len(sel) < 1000:
        return []

    xmin, ymin = float(sel[:, 0].min()), float(sel[:, 1].min())
    NX = int((sel[:, 0].max() - xmin) / cell) + 1
    NY = int((sel[:, 1].max() - ymin) / cell) + 1
    NZ = max(3, int((zhi - zlo) / zbin))

    occ = np.zeros((NZ, NY, NX), dtype=np.uint16)
    ix = np.clip(((sel[:, 0] - xmin) / cell).astype(np.int32), 0, NX - 1)
    iy = np.clip(((sel[:, 1] - ymin) / cell).astype(np.int32), 0, NY - 1)
    iz = np.clip(((sel[:, 2] - zlo) / zbin).astype(np.int32), 0, NZ - 1)
    np.add.at(occ, (iz, iy, ix), 1)

    occ_bin = occ >= min_pts_bin
    frac = occ_bin.sum(0) / float(NZ)
    mask = (frac >= full_frac).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resultados = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        (ccx, ccy), (w, h), _ang = cv2.minAreaRect(cnt)
        lo, hi = sorted([w * cell, h * cell])
        if lo < 0.06 or hi > cand_max or lo < cand_min or hi / max(lo, 1e-6) > cand_aspect:
            continue

        blob = np.zeros((NY, NX), dtype=np.uint8)
        cv2.drawContours(blob, [cnt], -1, 1, thickness=-1)
        blob = blob.astype(bool)

        areas, cents = [], []
        for z in range(NZ):
            s = occ_bin[z] & blob
            n = int(s.sum())
            areas.append(n)
            if n > 0:
                ys, xs = np.nonzero(s)
                cents.append((xs.mean(), ys.mean()))
        areas = np.array(areas)
        med = np.median(areas[areas > 0]) if (areas > 0).any() else 0
        consist = float((areas >= area_ratio * med).sum()) / NZ if med > 0 else 0.0
        min_ratio = float(areas.min() / med) if med > 0 else 0.0
        if len(cents) >= 2:
            cents = np.array(cents)
            drift = float(np.linalg.norm(cents - cents.mean(0), axis=1).max()) * cell
        else:
            drift = 99.0

        ok = bool(consist >= consist_min and drift <= drift_max
                  and min_ratio >= minratio_min
                  and lo >= sec_min and hi / max(lo, 1e-6) <= sec_aspect)
        resultados.append({
            'cx': xmin + ccx * cell, 'cy': ymin + ccy * cell,
            'w': w * cell, 'h': h * cell,
            'consist': consist, 'drift': drift, 'min_ratio': min_ratio,
            'ok': ok,
        })
    return resultados
