# -*- coding: utf-8 -*-
"""Constroi a escada caracol (parametros do track_espiral) como IfcStair no IFC
real: 1 solido em cunha anular por degrau, empilhados em helice.
Tambem renderiza verificacao 3D + top-view."""
import json
import numpy as np
import ifcopenshell
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FN = 'output_IFC/real_cloud2bim.ifc'
OUT_PNG = 'real_escada_ifc.png'

p = json.load(open('espiral_params.json'))
CX, CY = p['cx'], p['cy']
R_IN, R_OUT = p['r_in'], p['r_out']
SLOPE, INTERC = p['slope_deg_m'], p['intercept_deg']
ANG, RISER = p['ang_step'], p['riser_h']

Z_FLOOR, Z_TOP = -1.011, 1.302
N_RISERS = int(round((Z_TOP - Z_FLOOR) / RISER))
print(f'{N_RISERS} degraus | espelho {RISER*100:.1f}cm | passo {ANG:.1f} graus')

def tread_polygon(a_deg):
    """Setor anular de ANG graus comecando em a_deg (5 seg de arco)."""
    a = np.radians(np.linspace(a_deg, a_deg - ANG, 6))  # helice desce em theta
    outer = [(CX + R_OUT * np.cos(t), CY + R_OUT * np.sin(t)) for t in a]
    inner = [(CX + R_IN * np.cos(t), CY + R_IN * np.sin(t)) for t in a[::-1]]
    return outer + inner

treads = []
for i in range(N_RISERS):
    z_top = Z_FLOOR + (i + 1) * RISER
    a_deg = INTERC + SLOPE * z_top
    treads.append({'z_top': z_top, 'poly': tread_polygon(a_deg)})

# ---------- IFC ----------
f = ifcopenshell.open(FN)
owner = f.by_type('IfcOwnerHistory')[0]
storey = f.by_type('IfcBuildingStorey')[0]
ctx = f.by_type('IfcGeometricRepresentationContext')[0]
zdir = f.create_entity('IfcDirection', DirectionRatios=(0.0, 0.0, 1.0))

solids = []
for t in treads:
    pts = [f.create_entity('IfcCartesianPoint', Coordinates=(float(x), float(y)))
           for x, y in t['poly']]
    poly = f.create_entity('IfcPolyline', Points=pts + [pts[0]])
    prof = f.create_entity('IfcArbitraryClosedProfileDef', ProfileType='AREA',
                           OuterCurve=poly)
    pos = f.create_entity('IfcAxis2Placement3D',
                          Location=f.create_entity('IfcCartesianPoint',
                                                   Coordinates=(0.0, 0.0, float(t['z_top'] - RISER))))
    solids.append(f.create_entity('IfcExtrudedAreaSolid', SweptArea=prof,
                                  Position=pos, ExtrudedDirection=zdir,
                                  Depth=float(RISER)))

shape = f.create_entity('IfcShapeRepresentation', ContextOfItems=ctx,
                        RepresentationIdentifier='Body',
                        RepresentationType='SweptSolid', Items=solids)
pds = f.create_entity('IfcProductDefinitionShape', Representations=[shape])
lp = f.create_entity('IfcLocalPlacement', RelativePlacement=f.create_entity(
    'IfcAxis2Placement3D',
    Location=f.create_entity('IfcCartesianPoint', Coordinates=(0.0, 0.0, 0.0))))
stair = f.create_entity('IfcStair', GlobalId=ifcopenshell.guid.new(),
                        OwnerHistory=owner, Name='Caracol telhado',
                        Description='Helice rastreada da nuvem (espelho a confirmar - resolucao fraca)',
                        ObjectPlacement=lp, Representation=pds,
                        PredefinedType='SPIRAL_STAIR')
rel = f.by_type('IfcRelContainedInSpatialStructure')[0]
rel.RelatedElements = list(rel.RelatedElements) + [stair]
f.write(FN)

m = ifcopenshell.open(FN)
print('Conteudo final do IFC:')
for t in ['IfcSlab', 'IfcWall', 'IfcWindow', 'IfcDoor', 'IfcColumn', 'IfcStair']:
    n = len(m.by_type(t))
    if n:
        print(f'  {t:10s} {n}')

# ---------- verificacao visual ----------
fig = plt.figure(figsize=(15, 6.5))
ax = fig.add_subplot(121, projection='3d')
cmap = plt.get_cmap('Blues')
for i, t in enumerate(treads):
    poly = np.array(t['poly'])
    z0, z1 = t['z_top'] - RISER, t['z_top']
    col = cmap(0.3 + 0.6 * i / len(treads))
    top = [list(zip(poly[:, 0], poly[:, 1], [z1] * len(poly)))]
    ax.add_collection3d(Poly3DCollection(top, facecolors=col, edgecolors='#456', linewidths=0.3))
ax.set_xlim(CX - 2, CX + 2); ax.set_ylim(CY - 2, CY + 2); ax.set_zlim(Z_FLOOR, Z_TOP)
ax.set_title(f'IfcStair: {N_RISERS} degraus em helice (3D)')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.view_init(elev=32, azim=-50)

ax2 = fig.add_subplot(122)
for i, t in enumerate(treads):
    poly = np.array(t['poly'])
    ax2.fill(poly[:, 0], poly[:, 1], color=cmap(0.3 + 0.6 * i / len(treads)),
             alpha=0.55, edgecolor='#456', linewidth=0.4)
ax2.plot(CX, CY, 'x', c='#c22', ms=10, mew=2.5)
ax2.set_aspect('equal'); ax2.set_title('Top-view: leque completo da caracol')
ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)')
fig.suptitle(f'Escada caracol inserida no IFC — centro ({CX:.2f},{CY:.2f}), '
             f'r {R_IN:.2f}-{R_OUT:.2f}m, {abs(SLOPE):.0f} graus/m', fontsize=12)
plt.tight_layout(); plt.savefig(OUT_PNG, dpi=130, bbox_inches='tight')
print(f'OK -> {OUT_PNG}')
