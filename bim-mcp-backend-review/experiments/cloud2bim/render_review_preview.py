from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as Element
from render_wall_labels_large import axis_points

EXCLUDE = {"W-S01-024", "W-S01-028", "W-S01-029", "W-S01-007", "W-S01-008", "W-S01-011",
           "W-S01-006", "W-S01-009", "W-S01-020", "W-S01-021", "W-S01-019", "W-S01-018",
           "W-S01-025", "W-S01-030", "W-S01-017"}
MERGES = [("W-S01-027", "W-S01-031", "W-S01-032"), ("W-S01-010", "W-S01-026")]

def main(ifc_path, xyz_path, out_path):
    model = ifcopenshell.open(str(ifc_path))
    pts = np.loadtxt(xyz_path, skiprows=1, usecols=(0, 1, 2))
    if len(pts) > 450000:
        pts = pts[np.random.default_rng(42).choice(len(pts), 450000, replace=False)]
    recs = {}
    for wall in model.by_type("IfcWall"):
        ref = wall.Name or wall.GlobalId
        for pset in Element.get_psets(wall).values():
            if isinstance(pset, dict) and pset.get("Reference"):
                ref = str(pset["Reference"]); break
        axis = axis_points(wall)
        if axis is not None: recs[ref] = (axis[0], axis[1])
    fig, ax = plt.subplots(figsize=(28, 20))
    ax.scatter(pts[:,0], pts[:,1], s=.15, c="#b8bec6", alpha=.15, linewidths=0)
    # Highlight the closed spaces delivered by the detector.
    settings = ifcopenshell.geom.settings(); settings.set(settings.USE_WORLD_COORDS, True)
    for space in model.by_type("IfcSpace"):
        try:
            shape = ifcopenshell.geom.create_shape(settings, space)
            v = np.asarray(shape.geometry.verts).reshape(-1, 3)[:, :2]
            if len(v) >= 3:
                c = v.mean(0); order = np.argsort(np.arctan2(v[:,1]-c[1], v[:,0]-c[0])); v=v[order]
                ax.fill(v[:,0], v[:,1], color="#60a5fa", alpha=.12, ec="#2563eb", lw=1.2, zorder=1)
                ax.text(c[0], c[1], f"SPACE {space.Name}", color="#1d4ed8", fontsize=13, ha="center", fontweight="bold")
        except Exception:
            pass
    for ref, (p0,p1) in sorted(recs.items()):
        if ref in EXCLUDE or any(ref in g for g in MERGES): continue
        ax.plot([p0[0],p1[0]], [p0[1],p1[1]], color="#145da0", lw=3)
        mid=(p0+p1)/2
        ax.annotate(ref.replace("W-S01-", "W-"), mid, xytext=mid+np.array([.15,.16]), fontsize=14, fontweight="bold", bbox=dict(boxstyle="round,pad=.25",fc="white",ec="#145da0",alpha=.95))
        ax.text(mid[0], mid[1]-.22, f"{np.linalg.norm(p1-p0):.2f} m", fontsize=10, color="#0f3d67", ha="center")
    # Openings from the detected IFC, shown for review only.
    openings = []
    for kind, typ in (("D", "IfcDoor"), ("J", "IfcWindow")):
        for n, obj in enumerate(model.by_type(typ), 1):
            try:
                m = obj.ObjectPlacement.RelativePlacement.Location.Coordinates
                openings.append((kind, n, np.array([float(m[0]), float(m[1])]), typ))
            except Exception:
                pass
    for kind, n, pos, typ in openings if False else []:
        color = "#dc2626" if kind == "D" else "#16a34a"
        ax.scatter([pos[0]], [pos[1]], s=180, marker="s" if kind == "D" else "o", c=color, edgecolors="white", linewidths=1.5, zorder=8)
        ax.annotate(f"{kind}{n}", pos, xytext=pos+np.array([.18,.18]), fontsize=12, color=color, fontweight="bold", zorder=9,
                    bbox=dict(boxstyle="round,pad=.18", fc="white", ec=color, alpha=.95))
    # Required physical joins (no opening was specified at these meeting points).
    joins = [("01→15", "W-S01-001", "W-S01-015"), ("03→22", "W-S01-003", "W-S01-022"),
             ("16→22", "W-S01-016", "W-S01-022"), ("13→10+26", "W-S01-013", "W-S01-010"),
             ("16→10+26", "W-S01-016", "W-S01-010")]
    joins.append(("03 para merge 26+10", "W-S01-003", "W-S01-010"))
    # 05.1 is a new perpendicular wall, not a continuation of W-005.
    for label, a, b in joins:
        if a in recs and b in recs:
            pa=np.vstack(recs[a]); pb=np.vstack(recs[b]); da=((pa[:,None,:]-pb[None,:,:])**2).sum(2); ia,ib=np.unravel_index(np.argmin(da),da.shape)
            p,q=pa[ia],pb[ib]
            # Continue wall a along its own axis; never draw a new diagonal between walls.
            va=pa[1]-pa[0]; va=va/(np.linalg.norm(va) or 1.0)
            if np.dot(q-p,va) < 0: va=-va
            q = p + va * np.dot(q-p, va)
            ax.plot([p[0],q[0]],[p[1],q[1]],color="#f97316",lw=5,ls="-.",zorder=6)
            ax.annotate("JUNÇÃO RETA "+label,(p+q)/2,fontsize=12,color="#9a3412",fontweight="bold",bbox=dict(boxstyle="round,pad=.2",fc="#ffedd5",ec="#f97316"))
    if "W-S01-005" in recs and "W-S01-003" in recs:
        pa=np.vstack(recs["W-S01-005"]); pb=np.vstack(recs["W-S01-003"])
        da=((pa[:,None,:]-pb[None,:,:])**2).sum(2); ia,ib=np.unravel_index(np.argmin(da),da.shape)
        p=pa[ia]; target=pb[ib]; v=pa[1]-pa[0]; v=v/(np.linalg.norm(v) or 1.0); n=np.array([-v[1],v[0]])
        if np.dot(target-p,n)<0: n=-n
        q=p+n*np.dot(target-p,n)
        ax.plot([p[0],q[0]],[p[1],q[1]],color="#ea580c",lw=5,ls="-.",zorder=7)
        ax.annotate("NOVA PAREDE 05.1 (90°)",(p+q)/2,fontsize=12,color="#9a3412",fontweight="bold",bbox=dict(boxstyle="round,pad=.2",fc="#ffedd5",ec="#ea580c"))
    for group in MERGES:
        vals=[recs[r] for r in group if r in recs]
        if not vals: continue
        ends=np.vstack([np.vstack(v) for v in vals]); center=ends.mean(0); _,_,vh=np.linalg.svd(ends-center,full_matrices=False); axis=vh[0]; proj=(ends-center)@axis; a,b=center+axis*proj.min(),center+axis*proj.max()
        ax.plot([a[0],b[0]],[a[1],b[1]],color="#c026d3",lw=6,alpha=.9,ls="--")
        ax.annotate("MERGE: " + "+".join(r.replace("W-S01-", "W-") for r in group), (a+b)/2, fontsize=15, color="#86198f", fontweight="bold", bbox=dict(boxstyle="round,pad=.3",fc="#fae8ff",ec="#c026d3"))
    # W13 is marked for trimming to the W23 limit; openings are review annotations.
    if "W-S01-013" in recs and "W-S01-023" in recs:
        p0,p1=recs["W-S01-013"]; ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color="#f59e0b",lw=5,ls=":")
        ax.annotate("W-013: cortar no limite de W-023", (p0+p1)/2, fontsize=14, color="#92400e", bbox=dict(boxstyle="round,pad=.25",fc="#fef3c7",ec="#f59e0b"))
    ax.set_title("Kladno — prévia corrigida da leitura da nuvem (sem IFC editado)",fontsize=24,fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal"); ax.grid(alpha=.18)
    ax.text(.01,.01,"Removidas: W-006/007/008/009/011/017/018/019/020/021/024/025/028/029/030 + escada.\nMagenta = unificações; laranja = junções retas; pontos de janela automáticos anulados; amarelo = W-013 limitada por W-023.",transform=ax.transAxes,fontsize=14,va="bottom",bbox=dict(fc="white",alpha=.9))
    fig.tight_layout(); fig.savefig(out_path,dpi=220,bbox_inches="tight"); plt.close(fig)
    print(out_path)

if __name__ == "__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
