from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import ifcopenshell
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize, snap
from render_wall_labels_large import axis_points

EXCLUDE = {6,7,8,9,11,17,18,19,20,21,24,25,28,29,30}

def rid(n): return f"W-S01-{n:03d}"

def infinite_intersection(seg_a, seg_b):
    p,r=np.asarray(seg_a[0],float),np.asarray(seg_a[1],float)-np.asarray(seg_a[0],float)
    q,s=np.asarray(seg_b[0],float),np.asarray(seg_b[1],float)-np.asarray(seg_b[0],float)
    cross=lambda a,b:a[0]*b[1]-a[1]*b[0]
    d=cross(r,s)
    if abs(d)<1e-9:return None
    return p+r*(cross(q-p,s)/d)

def set_nearest_endpoint(seg, point):
    a,b=np.asarray(seg[0],float),np.asarray(seg[1],float); p=np.asarray(point,float)
    return (p,b) if np.linalg.norm(a-p)<np.linalg.norm(b-p) else (a,p)

def merge_segments(records, ids):
    pts=np.vstack([np.vstack(records[i]) for i in ids if i in records]); c=pts.mean(0)
    _,_,vh=np.linalg.svd(pts-c,full_matrices=False); v=vh[0]; t=(pts-c)@v
    return c+v*t.min(),c+v*t.max()

def main(ifc_path,xyz_path,out_path):
    m=ifcopenshell.open(str(ifc_path)); rec={}
    for w in m.by_type("IfcWall"):
        a=axis_points(w)
        if a is not None and w.Name: rec[w.Name]=(np.asarray(a[0]),np.asarray(a[1]))
    g={k:v for k,v in rec.items() if int(k[-3:]) not in EXCLUDE}
    g[rid(27)]=merge_segments(rec,[rid(22),rid(27),rid(31),rid(32)]); g.pop(rid(22),None); g.pop(rid(31),None); g.pop(rid(32),None)
    g[rid(10)]=merge_segments(rec,[rid(10),rid(26)]); g.pop(rid(26),None)
    # Actual collinear extensions to the infinite supporting lines.
    for a,b in [(1,15),(3,27),(3,10),(16,27),(16,10),(13,23)]:
        ia,ib=rid(a),rid(b)
        if ia in g and ib in g:
            p=infinite_intersection(g[ia],g[ib])
            if p is not None:g[ia]=set_nearest_endpoint(g[ia],p)
    # Close sub-door-size gaps between approximately perpendicular walls.
    # This absorbs scan/detection tolerances while preserving larger door openings.
    names=list(g)
    for i,na in enumerate(names):
        for nb in names[i+1:]:
            if {na,nb}=={rid(13),rid(10)}: continue
            sa,sb=g[na],g[nb]; va=sa[1]-sa[0]; vb=sb[1]-sb[0]
            cosa=abs(float(np.dot(va,vb))/((np.linalg.norm(va)*np.linalg.norm(vb)) or 1.0))
            if cosa>0.25: continue
            p=infinite_intersection(sa,sb)
            if p is None: continue
            da=min(np.linalg.norm(p-sa[0]),np.linalg.norm(p-sa[1])); db=min(np.linalg.norm(p-sb[0]),np.linalg.norm(p-sb[1]))
            if da<=0.80 and db<=0.80:
                g[na]=set_nearest_endpoint(sa,p); g[nb]=set_nearest_endpoint(sb,p)
    # New wall W-005.1: perpendicular to W-005 and ending on W-003.
    a,b=g[rid(5)]; v=(b-a)/(np.linalg.norm(b-a) or 1); n=np.array([-v[1],v[0]])
    p=a; line_n=(p-n*100,p+n*100); q=infinite_intersection(line_n,g[rid(3)])
    if q is not None:g["W-S01-005.1"]=(p,q)
    lines=[LineString(v) for v in g.values()]
    network=unary_union(lines); polys=[p for p in polygonize(network) if p.area>1]
    pts=np.loadtxt(xyz_path,skiprows=1,usecols=(0,1,2)); pts=pts[::max(1,len(pts)//350000)]
    fig,ax=plt.subplots(figsize=(28,20)); ax.scatter(pts[:,0],pts[:,1],s=.13,c="#cbd5e1",alpha=.12)
    colors=plt.cm.Set3(np.linspace(0,1,max(1,len(polys))))
    for i,(p,c) in enumerate(zip(polys,colors),1):
        x,y=p.exterior.xy; ax.fill(x,y,color=c,alpha=.38,ec="#334155",lw=1.2); rp=p.representative_point(); ax.text(rp.x,rp.y,f"SPACE {i}\n{p.area:.2f} m²",ha="center",fontsize=13,fontweight="bold")
    for name,(a,b) in sorted(g.items()):
        ax.plot([a[0],b[0]],[a[1],b[1]],c="#075985",lw=4,solid_capstyle="round")
        mid=(a+b)/2; label=name.replace("W-S01-","W-"); length=np.linalg.norm(b-a)
        ax.annotate(f"{label}\n{length:.2f} m",mid,fontsize=11,fontweight="bold",ha="center",bbox=dict(boxstyle="round,pad=.22",fc="white",ec="#075985",alpha=.94))
    ax.set_title(f"Kladno — geometria 2D realmente corrigida | {len(polys)} espaços fechados",fontsize=23,fontweight="bold")
    ax.set_aspect("equal"); ax.grid(alpha=.16); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.text(.01,.01,"Azul = segmentos finais reais. Merges substituídos por linha única; junções calculadas por interseção; W-005.1 perpendicular à W-005.",transform=ax.transAxes,fontsize=13,bbox=dict(fc="white",alpha=.92))
    fig.tight_layout(); fig.savefig(out_path,dpi=220,bbox_inches="tight"); print(out_path); print("spaces",len(polys))

if __name__=="__main__": main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
