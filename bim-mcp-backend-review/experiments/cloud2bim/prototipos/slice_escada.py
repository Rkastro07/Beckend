# -*- coding: utf-8 -*-
"""Fatia a caixa da escada em cortes de 25cm e plota uma mini-planta por corte.
Fundo cinza fixo (densidade de todos os z da regiao) + pontos da fatia em azul:
o que 'anda' de um painel pro outro e a espiral."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

XYZ = 'input_xyz/real.xyz'
OUT = 'real_escada_fatias.png'

# caixa da escada (do mapa de localizacao)
X0, X1 = 4.5, 15.7
Y0, Y1 = -3.2, 7.0
Z0, Z1 = -2.15, 1.35
DZ = 0.25
CELL = 0.04

print('[1/3] Lendo pontos da regiao da escada...')
parts = []
for c in pd.read_csv(XYZ, sep='\t', usecols=[0, 1, 2], chunksize=4_000_000):
    a = c.to_numpy(dtype=np.float32)
    m = (a[:, 0] >= X0) & (a[:, 0] <= X1) & (a[:, 1] >= Y0) & (a[:, 1] <= Y1)
    if m.any():
        parts.append(a[m])
pts = np.vstack(parts)
print(f'   {len(pts):,} pontos na caixa')

print('[2/3] Grades por fatia...')
NX = int((X1 - X0) / CELL) + 1
NY = int((Y1 - Y0) / CELL) + 1
xe = np.linspace(X0, X1, NX + 1)
ye = np.linspace(Y0, Y1, NY + 1)
bg, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=[ye, xe])  # fundo: todos os z

zlos = np.arange(Z0, Z1, DZ)
n = len(zlos)
print(f'   {n} fatias de {DZ}m')

print('[3/3] Renderizando grade...')
ncol = 5
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(18, 3.4 * nrow), sharex=True, sharey=True)
axes = np.atleast_2d(axes)
for i, zlo in enumerate(zlos):
    ax = axes[i // ncol, i % ncol]
    ax.imshow(np.log1p(bg), cmap='Greys', origin='lower', alpha=0.30,
              extent=(X0, X1, Y0, Y1), aspect='equal')
    m = (pts[:, 2] >= zlo) & (pts[:, 2] < zlo + DZ)
    sl = pts[m]
    if len(sl) > 120_000:
        sl = sl[np.random.default_rng(0).choice(len(sl), 120_000, replace=False)]
    ax.scatter(sl[:, 0], sl[:, 1], s=0.4, c='#2166ac', linewidths=0, alpha=0.8)
    ax.set_title(f'z {zlo:+.2f} a {zlo+DZ:+.2f} m   ({m.sum():,} pts)', fontsize=9)
    ax.tick_params(labelsize=7)
for j in range(n, nrow * ncol):
    axes[j // ncol, j % ncol].axis('off')
fig.suptitle('Caixa da escada fatiada (25cm por corte, de baixo pra cima) — '
             'azul = pontos da fatia, cinza = referencia da caixa toda', fontsize=13, y=1.0)
plt.tight_layout()
plt.savefig(OUT, dpi=110, bbox_inches='tight')
print(f'OK -> {OUT}')
