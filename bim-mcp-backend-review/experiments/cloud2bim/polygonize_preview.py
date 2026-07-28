from pathlib import Path
import sys, numpy as np, matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize
import ifcopenshell
from render_wall_labels_large import axis_points
from render_review_preview import EXCLUDE

ifc=ifcopenshell.open(sys.argv[1]); xyz=np.loadtxt(sys.argv[2],skiprows=1,usecols=(0,1,2)); xyz=xyz[::max(1,len(xyz)//300000)]
lines=[]; walls={}
for w in ifc.by_type('IfcWall'):
 r=w.Name; a=axis_points(w)
 if r and a and r not in EXCLUDE: walls[r]=a; lines.append(LineString(a))
for a,b in [('W-S01-027','W-S01-031'),('W-S01-010','W-S01-026')]:
 if a in walls and b in walls: lines.append(LineString([walls[a][0],walls[b][1]]))
u=unary_union(lines); polys=[p for p in polygonize(u) if p.area>2]
fig,ax=plt.subplots(figsize=(28,20)); ax.scatter(xyz[:,0],xyz[:,1],s=.15,c='#cbd5e1',alpha=.12)
for i,p in enumerate(polys,1):
 x,y=p.exterior.xy; ax.fill(x,y,color='#60a5fa',alpha=.2,ec='#2563eb'); c=p.representative_point(); ax.text(c.x,c.y,f'COMODO {i}\n{p.area:.1f} m²',ha='center',fontsize=13,fontweight='bold',color='#1d4ed8')
for r,(a,b) in walls.items(): ax.plot([a[0],b[0]],[a[1],b[1]],c='#145da0',lw=2)
for r,(a,b) in walls.items():
 m=(np.asarray(a)+np.asarray(b))/2; ax.annotate(r.replace('W-S01-','W-'),m,fontsize=10,fontweight='bold',color='#071a2b',bbox=dict(boxstyle='round,pad=.18',fc='white',ec='#145da0',alpha=.9))
ax.set_aspect('equal'); ax.grid(alpha=.15); ax.set_title('Kladno — prévia de cômodos fechados (sem IFC)',fontsize=24,fontweight='bold'); fig.tight_layout(); fig.savefig(sys.argv[3],dpi=220); print('polygons',len(polys))
