from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import ifcopenshell
import ifcopenshell.util.element as Element
import ifcopenshell.util.placement as Placement
import matplotlib.pyplot as plt
import numpy as np


def axis_points(wall):
    rep = next((r for r in (wall.Representation.Representations if wall.Representation else [])
                if r.RepresentationType == "Curve2D"), None)
    if not rep or not rep.Items:
        return None
    curve = rep.Items[0]
    pts = curve.Points
    if len(pts) < 2:
        return None
    m = Placement.get_local_placement(wall.ObjectPlacement)
    p0 = m @ np.array([pts[0].Coordinates[0], pts[0].Coordinates[1], 0.0, 1.0])
    p1 = m @ np.array([pts[1].Coordinates[0], pts[1].Coordinates[1], 0.0, 1.0])
    return p0[:2], p1[:2]


def wall_record(wall, storey):
    axis = axis_points(wall)
    if axis is None:
        return None
    p0, p1 = axis
    ref = wall.Name or wall.GlobalId
    for pset in Element.get_psets(wall).values():
        if isinstance(pset, dict) and pset.get("Reference"):
            ref = str(pset["Reference"])
            break
    return {"ref": ref, "storey": storey, "p0": p0, "p1": p1,
            "mid": (p0 + p1) / 2, "length": float(np.linalg.norm(p1-p0))}


def render(ifc_path: Path, xyz_path: Path, out_dir: Path, max_points: int = 450_000):
    model = ifcopenshell.open(str(ifc_path))
    points = np.loadtxt(xyz_path, skiprows=1, usecols=(0, 1, 2))
    if len(points) > max_points:
        rng = np.random.default_rng(42)
        points = points[rng.choice(len(points), max_points, replace=False)]
    by_storey = defaultdict(list)
    for wall in model.by_type("IfcWall"):
        storey = "S01"
        for rel in getattr(wall, "ContainedInStructure", []) or []:
            if rel.RelatingStructure and rel.RelatingStructure.is_a("IfcBuildingStorey"):
                storey = rel.RelatingStructure.Name or rel.RelatingStructure.LongName or storey
        rec = wall_record(wall, storey)
        if rec:
            by_storey[storey].append(rec)
    out_dir.mkdir(parents=True, exist_ok=True)
    stores = sorted(by_storey, key=str)
    fig, axes = plt.subplots(1, max(1, len(stores)), figsize=(30, 18), squeeze=False)
    axes = axes[0]
    for i, storey in enumerate(stores):
        ax = axes[i]
        recs = by_storey[storey]
        ax.scatter(points[:, 0], points[:, 1], s=0.15, c="#b8bec6", alpha=0.16, linewidths=0)
        for n, rec in enumerate(sorted(recs, key=lambda r: r["ref"])):
            p0, p1 = rec["p0"], rec["p1"]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#145da0", lw=2.8, solid_capstyle="round")
            dx, dy = p1 - p0
            normal = np.array([-dy, dx]) / (np.linalg.norm([dx, dy]) or 1.0)
            offset = normal * (0.24 + (n % 3) * 0.10)
            pos = rec["mid"] + offset
            ax.annotate(rec["ref"], xy=rec["mid"], xytext=pos,
                        fontsize=14, fontweight="bold", color="#071a2b",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#145da0", lw=1.2, alpha=0.94),
                        arrowprops=dict(arrowstyle="-", color="#145da0", lw=0.8, alpha=0.7))
        allxy = np.vstack([points[:, :2], *[np.vstack([r["p0"], r["p1"]]) for r in recs]])
        lo, hi = allxy.min(axis=0), allxy.max(axis=0)
        pad = max(1.0, 0.04 * np.max(hi-lo))
        ax.set_xlim(lo[0]-pad, hi[0]+pad); ax.set_ylim(lo[1]-pad, hi[1]+pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Pavimento {storey} — {len(recs)} paredes", fontsize=19, fontweight="bold", pad=14)
        ax.set_xlabel("X (m)", fontsize=12); ax.set_ylabel("Y (m)", fontsize=12)
        ax.grid(True, alpha=0.18)
    for ax in axes[len(stores):]:
        ax.axis("off")
    fig.suptitle("Cloud-to-BIM Kladno — paredes detectadas e identificadas", fontsize=24, fontweight="bold")
    fig.text(0.5, 0.02, f"Nuvem: {len(points):,} pontos amostrados | IFC: {ifc_path.name}", ha="center", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    combined = out_dir / "kladno_cloud2bim_wall_labels_large.png"
    fig.savefig(combined, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return combined, {s: len(by_storey[s]) for s in stores}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ifc", type=Path); ap.add_argument("xyz", type=Path); ap.add_argument("out", type=Path)
    args = ap.parse_args()
    path, counts = render(args.ifc, args.xyz, args.out)
    print(path)
    print(counts)
