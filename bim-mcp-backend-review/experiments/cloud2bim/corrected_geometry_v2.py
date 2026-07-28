from __future__ import annotations
import numpy as np
from render_wall_labels_large import axis_points

EXCLUDE={6,7,8,9,11,17,18,19,20,21,24,25,28,29,30}
def rid(n):return f"W-S01-{n:03d}"
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
def intersection(a,b):
    p=np.asarray(a[0]); r=np.asarray(a[1])-p; q=np.asarray(b[0]); s=np.asarray(b[1])-q; d=cross(r,s)
    return None if abs(d)<1e-9 else p+r*cross(q-p,s)/d
def merge_faces(rec,ids):
    pts=np.vstack([np.vstack(rec[i]) for i in ids if i in rec]); c=pts.mean(0); _,_,vh=np.linalg.svd(pts-c,full_matrices=False)
    v=vh[0]; n=np.array([-v[1],v[0]]); along=(pts-c)@v; across=(pts-c)@n
    center=c+n*((across.min()+across.max())/2); seg=(center+v*along.min(),center+v*along.max())
    return seg,max(.15,float(across.max()-across.min()))
def assign_two_ends(seg,p0,p1):
    a,b=np.asarray(seg[0]),np.asarray(seg[1]); d1=np.linalg.norm(a-p0)+np.linalg.norm(b-p1); d2=np.linalg.norm(a-p1)+np.linalg.norm(b-p0)
    return (p0,p1) if d1<=d2 else (p1,p0)
def nearest_end(seg,p):
    a,b=np.asarray(seg[0]),np.asarray(seg[1]); return (p,b) if np.linalg.norm(a-p)<=np.linalg.norm(b-p) else (a,p)
def build(model):
    rec={w.Name:tuple(map(np.asarray,axis_points(w))) for w in model.by_type("IfcWall") if w.Name and axis_points(w) is not None}
    g={k:v for k,v in rec.items() if int(k[-3:]) not in EXCLUDE}; overrides={}
    g[rid(27)],overrides[rid(27)]=merge_faces(rec,[rid(22),rid(27),rid(31),rid(32)])
    for k in (rid(22),rid(31),rid(32)):g.pop(k,None)
    g[rid(10)],overrides[rid(10)]=merge_faces(rec,[rid(10),rid(26)]); g.pop(rid(26),None)
    # Walls explicitly required to meet two opposite boundaries.
    for wall,t0,t1 in [(rid(3),rid(27),rid(10)),(rid(16),rid(27),rid(10))]:
        p0,p1=intersection(g[wall],g[t0]),intersection(g[wall],g[t1])
        if p0 is not None and p1 is not None:g[wall]=assign_two_ends(g[wall],p0,p1)
    for wall,target in [(rid(1),rid(15)),(rid(13),rid(23))]:
        p=intersection(g[wall],g[target])
        if p is not None:g[wall]=nearest_end(g[wall],p)
    # New W-005.1 from the correct endpoint, perpendicular to W-005, ending at W-003.
    a,b=g[rid(5)]; v=(b-a)/(np.linalg.norm(b-a) or 1); n=np.array([-v[1],v[0]]); p=a
    q=intersection((p-n*100,p+n*100),g[rid(3)])
    if q is not None:g["W-S01-005.1"]=(p,q); overrides["W-S01-005.1"]=.15
    # Global endpoint closure: stage candidates first, then update both ends once.
    candidates={k:[[],[]] for k in g}; names=list(g)
    for i,na in enumerate(names):
        for nb in names[i+1:]:
            if {na,nb}=={rid(13),rid(10)}:continue
            sa,sb=g[na],g[nb]; va=sa[1]-sa[0]; vb=sb[1]-sb[0]
            if abs(np.dot(va,vb))/((np.linalg.norm(va)*np.linalg.norm(vb)) or 1)>.25:continue
            p=intersection(sa,sb)
            if p is None:continue
            for name,seg in ((na,sa),(nb,sb)):
                ds=[np.linalg.norm(p-seg[0]),np.linalg.norm(p-seg[1])]; idx=int(ds[1]<ds[0])
                if ds[idx]<=.80:candidates[name][idx].append((ds[idx],p))
    for name,ends in candidates.items():
        a,b=map(np.asarray,g[name]); out=[a,b]
        for idx,opts in enumerate(ends):
            if opts:out[idx]=min(opts,key=lambda x:x[0])[1]
        g[name]=(out[0],out[1])
    return g,overrides
