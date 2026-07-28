# -*- coding: utf-8 -*-
"""Localiza a escada: pinta em azul (sequencial por profundidade) todos os
pontos ABAIXO do piso principal. A espiral que desce pro subsolo aparece
sozinha sobre o fundo cinza da planta."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

XYZ = 'input_xyz/real.xyz'
CACHE = 'occ_cache.npz'
OUT = 'real_escada_localizacao.png'
CELL = 0.05
Z_PISO = -1.35   # abaixo do fundo da laje do piso principal (-1.311)

print('[1/3] Fundo (grade cacheada, todos os z)...')
d = np.load(CACHE)
total_cnt = d['total_cnt']
xmin, ymin = float(d['xmin']), float(d['ymin'])
NY, NX = total_cnt.shape

print('[2/3] Pontos abaixo do piso principal...')
deep = []
for c in pd.read_csv(XYZ, sep='\t', usecols=[0, 1, 2], chunksize=4_000_000):
    a = c.to_numpy()
    m = a[:, 2] < Z_PISO
    if m.any():
        deep.append(a[m])
deep = np.vstack(deep) if deep else np.empty((0, 3))
print(f'   {len(deep):,} pontos com z < {Z_PISO}')
if len(deep) > 400_000:
    idx = np.random.default_rng(0).choice(len(deep), 400_000, replace=False)
    deep = deep[idx]

print('[3/3] Renderizando...')
fig, ax = plt.subplots(figsize=(16, 7))
ax.imshow(np.log1p(total_cnt), cmap='Greys', origin='lower', alpha=0.55,
          extent=(xmin, xmin + NX * CELL, ymin, ymin + NY * CELL), aspect='equal')
if len(deep):
    depth = -(deep[:, 2])  # maior = mais fundo
    sc = ax.scatter(deep[:, 0], deep[:, 1], c=depth, cmap='Blues', s=0.5,
                    linewidths=0, vmin=-Z_PISO, vmax=depth.max(), zorder=3)
    cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label('profundidade abaixo do piso (m)')
    # centro da regiao funda -> anotacao
    cx, cy = deep[:, 0].mean(), deep[:, 1].mean()
    ax.annotate('escada desce aqui', (cx, cy), xytext=(cx - 6, cy + 4.5),
                fontsize=12, fontweight='bold', color='#1a4a7a',
                arrowprops=dict(arrowstyle='->', color='#1a4a7a', lw=1.8), zorder=5)
ax.set_title(f'Onde a escada desce: pontos abaixo do piso principal (z < {Z_PISO}m), '
             'azul mais escuro = mais fundo')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
plt.tight_layout(); plt.savefig(OUT, dpi=130)
print(f'OK -> {OUT}')
