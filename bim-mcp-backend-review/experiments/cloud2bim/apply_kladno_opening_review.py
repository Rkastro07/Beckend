from __future__ import annotations
import csv,json,sys
from dataclasses import replace
from pathlib import Path

import ifcopenshell
import matplotlib.pyplot as plt
import numpy as np

from cloud2bim_patched.opening_detector_v2 import OpeningCandidate,detect_wall_openings,render_wall_result
from corrected_geometry_v2 import build as corrected_geometry,intersection,rid
from run_opening_detector_v2 import floor_and_ceiling

def main(ifc_path,xyz_path,diag_path,proposal_path,outdir):
    outdir.mkdir(parents=True,exist_ok=True); (outdir/"walls").mkdir(exist_ok=True)
    model=ifcopenshell.open(str(ifc_path)); walls,overrides=corrected_geometry(model); points=np.loadtxt(xyz_path,skiprows=1,usecols=(0,1,2)); floor,ceiling=floor_and_ceiling(model)
    thickness={}
    with open(diag_path,newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:thickness[row["reference"]]=float(row["thickness"])
            except Exception:pass
    raw=json.loads(proposal_path.read_text(encoding="utf-8")); flat={c["id"]:c for w in raw["walls"] for c in w["candidates"]}
    keep=["D-W-S01-001-02","D-W-S01-003-01","D-W-S01-003-02","D-W-S01-004-01","D-W-S01-016-01","J-W-S01-027-02","J-W-S01-027-04"]
    final=[dict(flat[cid],status="approved",confidence=1.0,source="review") for cid in keep]
    anchor=flat["J-W-S01-012-02"]; width=float(anchor["width"]); z0=float(anchor["z_min"]); z1=float(anchor["z_max"])
    def reviewed_window(cid,wall,s):
        return {"id":cid,"host_wall":wall,"type":"window","s_center":round(float(s),4),"width":round(width,4),"z_min":round(z0,4),"z_max":round(z1,4),"height":round(z1-z0,4),"confidence":1.0,"status":"approved","source":"review-family","evidence":{"template":"J-W-S01-012-02"}}
    final += [
        reviewed_window("J-W-S01-012-01",rid(12),1.1380),
        reviewed_window("J-W-S01-012-02",rid(12),4.2600),
        reviewed_window("J-W-S01-012-03",rid(12),6.7084),
        reviewed_window("J-W-S01-015-01",rid(15),5.6334),
        reviewed_window("J-W-S01-015-02",rid(15),2.6500),
    ]
    topology=[]
    for index,(a,b) in enumerate([(rid(23),rid(16)),(rid(23),rid(15)),(rid(1),rid(5))],1):
        p=intersection(walls[a],walls[b]); ends=np.vstack(walls[a]); endpoint=ends[np.argmin(np.linalg.norm(ends-p,axis=1))]; center=(endpoint+p)/2; u=(walls[a][1]-walls[a][0])/(np.linalg.norm(walls[a][1]-walls[a][0]) or 1); s=float(np.dot(center-walls[a][0],u)); gap=float(np.linalg.norm(p-endpoint))
        topology.append({"id":f"D-GAP-{index:02d}","host_wall":a,"type":"door","s_center":round(s,4),"width":round(gap,4),"z_min":0.0,"z_max":2.1,"height":2.1,"confidence":1.0,"status":"approved","source":"topology-review","between":[a,b],"global_center":[float(center[0]),float(center[1])],"evidence":{"gap_to_intersection":round(gap,4)}})
    final+=topology
    by_wall={wall:[] for wall in walls}
    for c in final:by_wall.setdefault(c["host_wall"],[]).append(c)
    # Re-render the two reviewed window walls on their actual X-Z grids.
    for wall_id in (rid(12),rid(15)):
        a,b=walls[wall_id]; result=detect_wall_openings(points,wall_id=wall_id,start=a,end=b,thickness=overrides.get(wall_id,thickness.get(wall_id,.15)),floor_z=floor,ceiling_z=ceiling)
        result.candidates=[OpeningCandidate(id=c["id"],host_wall=c["host_wall"],type=c["type"],s_center=c["s_center"],width=c["width"],z_min=c["z_min"],z_max=c["z_max"],height=c["height"],confidence=1.0,status="proposed",evidence=c.get("evidence",{})) for c in by_wall[wall_id]]
        render_wall_result(result,outdir/"walls"/f"{wall_id}_reviewed.png")
    fig,ax=plt.subplots(figsize=(28,20)); sample=points[::max(1,len(points)//450000)]; ax.scatter(sample[:,0],sample[:,1],s=.13,c="#cbd5e1",alpha=.14)
    for wall_id,(a,b) in sorted(walls.items()):ax.plot([a[0],b[0]],[a[1],b[1]],c="#334155",lw=2.3)
    for c in final:
        if "global_center" in c:p=np.asarray(c["global_center"])
        else:
            a,b=walls[c["host_wall"]]; u=(b-a)/(np.linalg.norm(b-a) or 1); p=a+u*c["s_center"]
        color="#dc2626" if c["type"]=="door" else "#16a34a"; marker="s" if c["type"]=="door" else "o"
        ax.scatter(p[0],p[1],s=190,c=color,marker=marker,edgecolors="white",linewidths=1.4,zorder=8); ax.annotate(c["id"],p,xytext=p+np.array([.14,.14]),fontsize=10,fontweight="bold",color=color,bbox=dict(boxstyle="round,pad=.18",fc="white",ec=color,alpha=.94))
    ax.set_title("Kladno — teste da revisão de portas e janelas | 8 portas, 7 janelas",fontsize=23,fontweight="bold"); ax.set_aspect("equal"); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(outdir/"opening_review_test_overview.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    payload={"schema":"cloud2bim.opening-review-test.v1","floor_z":floor,"ceiling_z":ceiling,"candidates":final,"rejected":["J-W-S01-012-04","D-W-S01-015-01"]}
    (outdir/"opening_review_test.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"doors":sum(c["type"]=="door" for c in final),"windows":sum(c["type"]=="window" for c in final),"topology_doors":len(topology)},ensure_ascii=False)); print(outdir)

if __name__=="__main__":main(*map(Path,sys.argv[1:6]))
