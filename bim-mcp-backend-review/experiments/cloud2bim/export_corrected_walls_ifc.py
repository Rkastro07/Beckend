from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
import ifcopenshell
import ifcopenshell.api as api
from bim_authoring.engine import IfcAuthoringEngine
from render_wall_labels_large import axis_points
from build_corrected_plan_preview import rid, EXCLUDE, merge_segments, infinite_intersection, set_nearest_endpoint

def corrected_geometry(source):
    rec={}
    for w in source.by_type("IfcWall"):
        a=axis_points(w)
        if a is not None and w.Name: rec[w.Name]=(np.asarray(a[0]),np.asarray(a[1]))
    g={k:v for k,v in rec.items() if int(k[-3:]) not in EXCLUDE}
    g[rid(27)]=merge_segments(rec,[rid(22),rid(27),rid(31),rid(32)])
    for k in (rid(22),rid(31),rid(32)): g.pop(k,None)
    g[rid(10)]=merge_segments(rec,[rid(10),rid(26)]); g.pop(rid(26),None)
    for a,b in [(1,15),(3,27),(3,10),(16,27),(16,10),(13,23)]:
        ia,ib=rid(a),rid(b)
        if ia in g and ib in g:
            p=infinite_intersection(g[ia],g[ib])
            if p is not None:g[ia]=set_nearest_endpoint(g[ia],p)
    names=list(g)
    for i,na in enumerate(names):
        for nb in names[i+1:]:
            if {na,nb}=={rid(13),rid(10)}:continue
            sa,sb=g[na],g[nb]; va=sa[1]-sa[0]; vb=sb[1]-sb[0]
            cosa=abs(float(np.dot(va,vb))/((np.linalg.norm(va)*np.linalg.norm(vb)) or 1))
            if cosa>0.25:continue
            p=infinite_intersection(sa,sb)
            if p is None:continue
            da=min(np.linalg.norm(p-sa[0]),np.linalg.norm(p-sa[1])); db=min(np.linalg.norm(p-sb[0]),np.linalg.norm(p-sb[1]))
            if da<=.80 and db<=.80:g[na]=set_nearest_endpoint(sa,p); g[nb]=set_nearest_endpoint(sb,p)
    a,b=g[rid(5)]; v=(b-a)/(np.linalg.norm(b-a) or 1); n=np.array([-v[1],v[0]]); p=a
    q=infinite_intersection((p-n*100,p+n*100),g[rid(3)])
    if q is not None:g["W-S01-005.1"]=(p,q)
    return g

def main(src_path, diag_path, out_path):
    src=ifcopenshell.open(str(src_path)); geom=corrected_geometry(src)
    thick={}
    with open(diag_path,newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try: thick[r["reference"]]=max(.10,min(.35,float(r["thickness"])))
            except Exception: pass
    model=api.run("project.create_file",version="IFC4")
    project=api.run("root.create_entity",model,ifc_class="IfcProject",name="Kladno Cloud-to-BIM - paredes corrigidas")
    api.run("unit.assign_unit",model)
    ctx=api.run("context.add_context",model,context_type="Model")
    body=api.run("context.add_context",model,context_type="Model",context_identifier="Body",target_view="MODEL_VIEW",parent=ctx)
    site=api.run("root.create_entity",model,ifc_class="IfcSite",name="Kladno")
    building=api.run("root.create_entity",model,ifc_class="IfcBuilding",name="Kladno Saal")
    storey=api.run("root.create_entity",model,ifc_class="IfcBuildingStorey",name="Floor 0.1 m")
    api.run("aggregate.assign_object",model,products=[site],relating_object=project)
    api.run("aggregate.assign_object",model,products=[building],relating_object=site)
    api.run("aggregate.assign_object",model,products=[storey],relating_object=building)
    engine=IfcAuthoringEngine(model)
    for name,(a,b) in sorted(geom.items()):
        source_name=name if name in thick else rid(5)
        engine.create_wall(start=a,end=b,height=3.05,thickness=thick.get(source_name,.15),body_context=body,storey=storey,elevation=-.159,name=name)
    out_path.parent.mkdir(parents=True,exist_ok=True); model.write(str(out_path))
    print(out_path); print("IfcWall",len(model.by_type("IfcWall"))); print("IfcDoor",len(model.by_type("IfcDoor"))); print("IfcWindow",len(model.by_type("IfcWindow"))); print("IfcSpace",len(model.by_type("IfcSpace")))

if __name__=="__main__":main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
