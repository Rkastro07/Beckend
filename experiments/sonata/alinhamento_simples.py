"""Cópia standalone das funções de alinhamento do app_obb.py.

NÃO depende de Flask/sklearn/etc. Só numpy + random + itertools.

ATENÇÃO: Mantém em sync com app_obb.alinhar_nuvem_com_ifc se houver mudanças.
Idealmente, no futuro, app_obb deve importar daqui (ao invés de duplicar).
"""
import random
from itertools import permutations, product
from typing import Dict, List, Tuple

import numpy as np


SCALES_TO_TEST = (0.001, 0.01, 1.0, 100.0, 1000.0)


def _bounds_from_objs(objetos: List[Dict]):
    if not objetos:
        return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    xs = [o['bbox']['xmin'] for o in objetos] + [o['bbox']['xmax'] for o in objetos]
    ys = [o['bbox']['ymin'] for o in objetos] + [o['bbox']['ymax'] for o in objetos]
    zs = [o['bbox']['zmin'] for o in objetos] + [o['bbox']['zmax'] for o in objetos]
    bmin = np.array([min(xs), min(ys), min(zs)], dtype=float)
    bmax = np.array([max(xs), max(ys), max(zs)], dtype=float)
    return bmin, bmax, (bmin + bmax) / 2.0, bmax - bmin


def _perm_sign_to_R(perm, sign, scale=1.0):
    R = np.zeros((3, 3), dtype=float)
    for i, p in enumerate(perm):
        R[i, p] = float(sign[i]) * float(scale)
    return R


def alinhar_nuvem_com_ifc(
    pts: np.ndarray,
    objetos_ifc: List[Dict],
    max_pts_amostra: int = 120_000,
) -> Tuple[np.ndarray, Dict]:
    """Alinha pts com IFC testando permutações de eixos + sinais + escala.

    Cada objeto IFC precisa ter chave 'bbox' = {xmin,xmax,ymin,ymax,zmin,zmax}.
    Retorna (pts_alinhado, transform_dict).
    """
    print(f"\n[ALIGN] alinhando nuvem (v2.1)...")
    if pts.size == 0 or not objetos_ifc:
        return pts, {'R': np.eye(3), 't': np.zeros(3), 'scale': 1.0}

    ifc_min, ifc_max, ifc_center, ifc_extent = _bounds_from_objs(objetos_ifc)
    diag_ifc = np.linalg.norm(ifc_extent)

    pts_min = pts.min(axis=0)
    pts_max = pts.max(axis=0)
    pts_center = (pts_min + pts_max) / 2.0
    diag_pts = np.linalg.norm(pts_max - pts_min)

    if len(pts) > max_pts_amostra:
        step = len(pts) // max_pts_amostra
        pts_sample = pts[::step]
    else:
        pts_sample = pts
    print(f"   amostra: {len(pts_sample):,} pts")

    objetos_teste = objetos_ifc
    if len(objetos_ifc) > 60:
        objetos_teste = random.sample(objetos_ifc, 60)

    perms = list(permutations((0, 1, 2), 3))
    signs = list(product([-1, 1], repeat=3))

    def _score(pts_test):
        s = 0.0
        for obj in objetos_teste:
            bbox = obj['bbox']
            dx = bbox['xmax'] - bbox['xmin']
            dy = bbox['ymax'] - bbox['ymin']
            dz = bbox['zmax'] - bbox['zmin']
            volume = max(dx * dy * dz, 1e-6)
            mm = 0.15
            mask = (
                (pts_test[:, 0] >= bbox['xmin'] - mm) &
                (pts_test[:, 0] <= bbox['xmax'] + mm) &
                (pts_test[:, 1] >= bbox['ymin'] - mm) &
                (pts_test[:, 1] <= bbox['ymax'] + mm) &
                (pts_test[:, 2] >= bbox['zmin'] - mm) &
                (pts_test[:, 2] <= bbox['zmax'] + mm)
            )
            pts_in = pts_test[mask]
            count = len(pts_in)
            if count < 5:
                continue
            densidade = count / volume
            std = np.std(pts_in, axis=0).mean()
            penal_spread = 1.0 / (1.0 + std)
            s += densidade * penal_spread
        return s

    score_identity = _score(pts_sample)
    print(f"   score identidade: {score_identity:.4f}")
    melhor = {'score': score_identity, 'R': np.eye(3), 't': np.zeros(3), 'scale': 1.0}
    IDENTITY_BARRIER = 1.30

    for scl in SCALES_TO_TEST:
        ratio = (diag_pts * scl) / (diag_ifc + 1e-6)
        if ratio < 0.6 or ratio > 1.8:
            continue
        for perm in perms:
            for sign in signs:
                P = pts_sample[:, perm]
                S = (P * np.array(sign)) * scl
                center_rot = pts_center[list(perm)] * np.array(sign) * scl
                t_centro = ifc_center - center_rot
                smin = S.min(axis=0)
                t_base = t_centro.copy()
                t_base[2] = ifc_min[2] - smin[2]
                for t in (t_centro, t_base):
                    pts_test = S + t
                    score_total = _score(pts_test)
                    threshold = (score_identity * IDENTITY_BARRIER
                                 if melhor['score'] == score_identity
                                 else melhor['score'])
                    if score_total > threshold:
                        melhor = {
                            'score': score_total,
                            'R': _perm_sign_to_R(perm, sign, scl),
                            't': t,
                            'scale': scl,
                        }

    if melhor['score'] < 1e-3:
        print("   [WARN] alinhamento falhou (score muito baixo)")
        return pts, {'R': np.eye(3), 't': np.zeros(3), 'scale': 1.0}
    print(f"   melhor score: {melhor['score']:.4f} | escala={melhor['scale']}")
    pts_alinhado = (pts @ melhor['R'].T) + melhor['t']
    return pts_alinhado, melhor
