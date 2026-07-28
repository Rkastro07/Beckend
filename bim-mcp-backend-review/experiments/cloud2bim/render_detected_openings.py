from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import ifcopenshell
import ifcopenshell.util.placement as Placement
from render_wall_labels_large import axis_points

def main(ifc_path, xyz_path, out_path):
    model=ifcopenshell.open(str(ifc_path))
    pts=np.loadtxt(xyz_path,skiprows=1,usecols=(0,1,2)); pts=pts[::max(1,len(pts)//400000)]
    fig,ax=plt.subplots(figsize=(28,20)); ax.scatter(pts[:,0],pts[:,1],s=.15,c="#a8b0ba",alpha=.16)
    for wall in model.by_type("IfcWall"):
        a=axis_points(wall)
        if a is None: continue
        p,q=a; ax.plot([p[0],q[0]],[p[1],q[1]],c="#334155",lw=2.3)
        mid=(p+q)/2; ax.annotate((wall.Name or "W").replace("W-S01-","W-"),mid,fontsize=9,color="#334155",bbox=dict(fc="white",ec="#94a3b8",alpha=.85,pad=.15))
    counts={"IfcDoor":0,"IfcWindow":0}
    for typ,code,color,marker in [("IfcDoor","D","#dc2626","s"),("IfcWindow","J","#16a34a","o")]:
        for n,obj in enumerate(model.by_type(typ),1):
            m=Placement.get_local_placement(obj.ObjectPlacement); pos=np.array([m[0,3],m[1,3]])
            counts[typ]+=1; ax.scatter(pos[0],pos[1],s=210,c=color,marker=marker,edgecolors="white",linewidths=1.6,zorder=8)
            ax.annotate(f"{code}{n}\n{obj.Name or typ}",pos,xytext=pos+np.array([.18,.18]),fontsize=11,fontweight="bold",color=color,bbox=dict(boxstyle="round,pad=.2",fc="white",ec=color,alpha=.95),zorder=9)
    ax.set_title(f"Detecção original Cloud-to-BIM — {counts['IfcDoor']} portas | {counts['IfcWindow']} janelas",fontsize=24,fontweight="bold")
    ax.text(.01,.01,"Vermelho = IfcDoor detectada | Verde = IfcWindow detectada | Cinza = paredes originais",transform=ax.transAxes,fontsize=14,bbox=dict(fc="white",alpha=.92))
    ax.set_aspect("equal"); ax.grid(alpha=.16); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); fig.tight_layout(); fig.savefig(out_path,dpi=220,bbox_inches="tight")
    print(out_path); print(counts)

if __name__=="__main__": main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
