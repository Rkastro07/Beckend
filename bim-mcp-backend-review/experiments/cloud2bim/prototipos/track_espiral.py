# -*- coding: utf-8 -*-
"""Rastreia a espiral da escada: ajusta centro comum (fit de circulo robusto),
mede avanco angular vs z e periodicidade de espelho -> parametros do create_stair.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

XYZ = 'input_xyz/real.xyz'
OUT = 'real_espiral_tracking.png'

# caixa do leque (trecho curvo visivel nos cortes 7-13)
X0, X1 = 11.3, 15.7
Y0, Y1 = -1.8, 2.6
ZLO, ZHI = -0.65, 1.10

print('[1/4] Lendo pontos do leque...')
parts = []
for c in pd.read_csv(XYZ, sep='\t', usecols=[0, 1, 2], chunksize=4_000_000):
    a = c.to_numpy(dtype=np.float64)
    m = ((a[:, 0] >= X0) & (a[:, 0] <= X1) & (a[:, 1] >= Y0) & (a[:, 1] <= Y1)
         & (a[:, 2] >= ZLO) & (a[:, 2] <= ZHI))
    if m.any():
        parts.append(a[m])
pts = np.vstack(parts)
print(f'   {len(pts):,} pontos')

# --- remove estruturas VERTICAIS (paredes/guarda-corpo): celula XY cuja
# extensao em z e grande nao e degrau (degrau = lamina fina de z) ---
CELLF = 0.05
gx = ((pts[:, 0] - X0) / CELLF).astype(np.int32)
gy = ((pts[:, 1] - Y0) / CELLF).astype(np.int32)
cell = gy.astype(np.int64) * 10000 + gx
order = np.argsort(cell)
cs, zs = cell[order], pts[order, 2]
uniq, start = np.unique(cs, return_index=True)
ext = np.zeros(len(cs))
for i, s in enumerate(start):
    e = start[i + 1] if i + 1 < len(start) else len(cs)
    ext[s:e] = zs[s:e].max() - zs[s:e].min()
horiz = np.empty(len(pts), bool)
horiz[order] = ext < 0.35
pts = pts[horiz]
print(f'   {len(pts):,} apos remover celulas verticais (paredes)')

print('[2/4] Fit de circulo robusto (centro comum da helice)...')
def kasa_fit(xy):
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    b = (xy ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    return cx, cy, r

xy = pts[:, :2].copy()
keep = np.ones(len(xy), bool)
for it in range(4):
    cx, cy, r = kasa_fit(xy[keep])
    ri = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    res = np.abs(ri - r)
    mad = np.median(np.abs(res - np.median(res))) + 1e-9
    keep = res < 5 * mad + 0.15
cx, cy, r = kasa_fit(xy[keep])
ri_all = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
flight = keep
r_in, r_out = np.percentile(ri_all[flight], [5, 95])
print(f'   centro ({cx:.2f}, {cy:.2f}) | raio medio {r:.2f} m | '
      f'faixa do lance {r_in:.2f}..{r_out:.2f} m ({flight.sum():,} pts no lance)')

print('[3/4] Angulo vs z + periodicidade de espelho...')
fp = pts[flight]
theta = np.degrees(np.arctan2(fp[:, 1] - cy, fp[:, 0] - cx))
# desembaraca wrap: centra o setor
t_med = np.median(theta)
theta = (theta - t_med + 180) % 360 - 180 + t_med

z = fp[:, 2]
# fit linear robusto theta(z): usa mediana por bin fino de z
zb = np.arange(z.min(), z.max(), 0.05)
tz, zz = [], []
for lo in zb:
    m = (z >= lo) & (z < lo + 0.05)
    if m.sum() >= 30:
        tz.append(np.median(theta[m])); zz.append(lo + 0.025)
tz, zz = np.array(tz), np.array(zz)
slope, intercept = np.polyfit(zz, tz, 1)   # graus por metro
print(f'   avanco angular: {slope:.1f} graus/m de subida')

# espelhos: picos no histograma fino de z (patamares dos degraus)
hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max(), 0.02))
pk, _ = find_peaks(hist, distance=5, prominence=0.10 * hist.max())
z_pk = edges[pk] + 0.01

# passo angular: picos no histograma de THETA (bordas radiais dos degraus)
# -> mais robusto que z-histograma global, que a oclusao embaralha
th_hist, th_edges = np.histogram(theta, bins=np.arange(theta.min(), theta.max(), 1.0))
th_pk, _ = find_peaks(th_hist, distance=8, prominence=0.15 * th_hist.max())
th_peaks = th_edges[th_pk] + 0.5
pitches = np.diff(np.sort(th_peaks))
pitches = pitches[(pitches > 5) & (pitches < 45)]  # descarta vaos de oclusao
ang_step = float(np.median(pitches)) if len(pitches) else float('nan')
riser_h = ang_step / abs(slope) if ang_step == ang_step else float('nan')
sweep = theta.max() - theta.min()
n_risers = int(round(sweep / ang_step)) if ang_step == ang_step else 0
print(f'   {len(th_peaks)} bordas de degrau | passo angular {ang_step:.1f} graus | '
      f'espelho {riser_h*100:.1f} cm | ~{n_risers} degraus no trecho visivel ({sweep:.0f} graus)')

import json
with open('espiral_params.json', 'w') as fj:
    json.dump({'cx': cx, 'cy': cy, 'r_in': r_in, 'r_out': r_out,
               'slope_deg_m': slope, 'intercept_deg': intercept,
               'ang_step': ang_step, 'riser_h': riser_h}, fj, indent=1)
print('   parametros salvos em espiral_params.json')

print('[4/4] Renderizando...')
fig = plt.figure(figsize=(20, 5.6))
# (a) top view do leque + centro/raios
ax = fig.add_subplot(141)
sc = ax.scatter(fp[:, 0], fp[:, 1], c=fp[:, 2], cmap='Blues', s=1.2, linewidths=0)
plt.colorbar(sc, ax=ax, fraction=0.045, label='z (m)')
tt = np.linspace(np.radians(theta.min()), np.radians(theta.max()), 100)
for rr in (r_in, r_out):
    ax.plot(cx + rr * np.cos(tt), cy + rr * np.sin(tt), '--', c='#888', lw=1)
ax.plot(cx, cy, 'x', ms=11, c='#c22', mew=2.5)
ax.annotate(f'centro\n({cx:.2f}, {cy:.2f})', (cx, cy), xytext=(8, 8),
            textcoords='offset points', fontsize=9, color='#c22')
ax.set_aspect('equal'); ax.set_title('Leque: pontos por altura + circulo ajustado')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
# (b) theta vs z
ax = fig.add_subplot(142)
ax.scatter(z, theta, s=1.0, c='#9ecae1', linewidths=0)
ax.plot(zz, tz, 'o', ms=4, c='#2166ac', label='mediana por bin')
zfit = np.array([z.min(), z.max()])
ax.plot(zfit, slope * zfit + intercept, '-', c='#c22', lw=2,
        label=f'{slope:.0f} graus/m')
for zp in z_pk:
    ax.axvline(zp, color='#bbb', lw=0.6, zorder=0)
ax.set_xlabel('z (m)'); ax.set_ylabel('angulo em torno do centro (graus)')
ax.set_title('Helice: angulo x altura (linhas = patamares)')
ax.legend(fontsize=8)
# (c) histograma de theta (bordas de degrau -> passo angular)
ax = fig.add_subplot(143)
ax.bar(th_edges[:-1], th_hist, width=1.0, color='#9ecae1')
ax.plot(th_peaks, th_hist[th_pk], 'v', c='#c22', ms=8)
ax.set_xlabel('angulo (graus)'); ax.set_ylabel('pontos')
ax.set_title(f'Bordas de degrau: passo {ang_step:.1f} graus')
# (d) histograma z (patamares, com oclusao no meio)
ax = fig.add_subplot(144)
ax.bar(edges[:-1], hist, width=0.02, color='#9ecae1')
ax.plot(z_pk, hist[pk], 'v', c='#c22', ms=7)
ax.set_xlabel('z (m)'); ax.set_ylabel('pontos')
ax.set_title('Patamares em z (vao central = oclusao)')
fig.suptitle(f'Tracking da espiral -> create_stair: inner_radius={r_in:.2f}m, '
             f'flight_width={r_out-r_in:.2f}m, raiser_height={riser_h:.3f}m, '
             f'angle_per_step={ang_step:.1f} graus, ~{n_risers} degraus visiveis', fontsize=12)
plt.tight_layout(); plt.savefig(OUT, dpi=130, bbox_inches='tight')
print(f'OK -> {OUT}')
