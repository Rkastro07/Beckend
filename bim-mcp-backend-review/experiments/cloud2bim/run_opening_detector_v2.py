from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import ifcopenshell
import ifcopenshell.util.placement as Placement
import matplotlib.pyplot as plt
import numpy as np

from cloud2bim_patched.opening_detector_v2 import (
    detect_topology_openings,
    detect_wall_openings,
    render_wall_result,
)
from corrected_geometry_v2 import build as corrected_geometry, rid
from render_wall_labels_large import axis_points


def thickness_table(path):
    result = {}
    with open(path, newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            try:
                result[row["reference"]] = float(row["thickness"])
            except Exception:
                pass
    return result


def floor_and_ceiling(model):
    elevations = sorted(
        float(storey.Elevation)
        for storey in model.by_type("IfcBuildingStorey")
        if storey.Elevation is not None
    )
    floor = elevations[0] if elevations else 0.0
    slab_z = sorted(
        float(Placement.get_local_placement(slab.ObjectPlacement)[2, 3])
        for slab in model.by_type("IfcSlab")
    )
    ceiling = next(
        (z for z in slab_z if z > floor + .5),
        elevations[1] if len(elevations) > 1 else floor + 3.0,
    )
    return floor, ceiling


def render_overview(points, walls, results, topology, output):
    fig, ax = plt.subplots(figsize=(28, 20))
    sample = points[::max(1, len(points) // 450000)]
    ax.scatter(sample[:, 0], sample[:, 1], s=.13, c="#cbd5e1", alpha=.14)
    colors = {"door": "#dc2626", "window": "#16a34a", "unknown": "#f59e0b"}
    for wall_id, (start, end) in sorted(walls.items()):
        ax.plot([start[0], end[0]], [start[1], end[1]], c="#334155", lw=2.3)
        middle = (start + end) / 2
        ax.annotate(
            wall_id.replace("W-S01-", "W-"),
            middle,
            fontsize=9,
            color="#334155",
            bbox=dict(fc="white", ec="#94a3b8", alpha=.88, pad=.15),
        )
        direction = (end - start) / (np.linalg.norm(end - start) or 1)
        for candidate in results[wall_id].candidates:
            position = start + direction * candidate.s_center
            color = (
                colors[candidate.type]
                if candidate.status == "proposed"
                else "#f59e0b"
            )
            marker = (
                "x"
                if candidate.status == "review"
                else ("s" if candidate.type == "door" else "o")
            )
            ax.scatter(
                position[0],
                position[1],
                s=180,
                c=color,
                marker=marker,
                edgecolors="white" if marker != "x" else None,
                linewidths=1.4,
                zorder=8,
            )
            ax.annotate(
                candidate.id,
                position,
                xytext=position + np.array([.14, .14]),
                fontsize=10,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=.18", fc="white", ec=color, alpha=.94),
                zorder=9,
            )
    for candidate in topology:
        position = np.asarray(candidate.global_center)
        color = "#7c3aed" if candidate.status == "proposed" else "#f59e0b"
        ax.scatter(
            position[0],
            position[1],
            s=210,
            c=color,
            marker="D",
            edgecolors="white",
            linewidths=1.4,
            zorder=8,
        )
        ax.annotate(
            candidate.id,
            position,
            xytext=position + np.array([.14, .14]),
            fontsize=10,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=.18", fc="white", ec=color, alpha=.94),
            zorder=9,
        )
    proposed = sum(
        candidate.status == "proposed"
        for result in results.values()
        for candidate in result.candidates
    )
    review = sum(
        candidate.status == "review"
        for result in results.values()
        for candidate in result.candidates
    )
    ax.set_title(
        f"Opening Detector V2 — {proposed} propostas locais | "
        f"{len(topology)} portas topológicas | {review} para revisão",
        fontsize=23,
        fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.grid(alpha=.15)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.text(
        .01,
        .01,
        "Vermelho = porta local | Verde = janela | Roxo = porta em gap "
        "topológico | Laranja = revisão",
        transform=ax.transAxes,
        fontsize=13,
        bbox=dict(fc="white", alpha=.92),
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ifc", type=Path)
    parser.add_argument("xyz", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--geometry-mode",
        choices=("ifc", "kladno-reviewed"),
        default="ifc",
        help="usa eixos do IFC ou a revisão geométrica do benchmark Kladno",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "walls").mkdir(exist_ok=True)

    model = ifcopenshell.open(str(args.ifc))
    thicknesses = thickness_table(args.diagnostics)
    if args.geometry_mode == "kladno-reviewed":
        walls, overrides = corrected_geometry(model)
    else:
        walls = {}
        overrides = {}
        for wall in model.by_type("IfcWall"):
            axis = axis_points(wall)
            if axis is not None and wall.Name:
                walls[wall.Name] = (
                    np.asarray(axis[0]),
                    np.asarray(axis[1]),
                )

    points = np.loadtxt(args.xyz, skiprows=1, usecols=(0, 1, 2))
    floor, ceiling = floor_and_ceiling(model)
    results = {}
    for wall_id, (start, end) in sorted(walls.items()):
        source = wall_id if wall_id in thicknesses else rid(5)
        thickness = overrides.get(wall_id, thicknesses.get(source, .15))
        result = detect_wall_openings(
            points,
            wall_id=wall_id,
            start=start,
            end=end,
            thickness=thickness,
            floor_z=floor,
            ceiling_z=ceiling,
        )
        results[wall_id] = result
        render_wall_result(result, args.output / "walls" / f"{wall_id}.png")

    topology = detect_topology_openings(walls)
    payload = {
        "schema": "cloud2bim.opening-proposals.v2",
        "source_ifc": str(args.ifc),
        "floor_z": floor,
        "ceiling_z": ceiling,
        "walls": [results[key].to_dict() for key in sorted(results)],
        "topology_candidates": [
            candidate.to_dict() for candidate in topology
        ],
    }
    with open(
        args.output / "opening_candidates_v2.json",
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    render_overview(
        points,
        walls,
        results,
        topology,
        args.output / "opening_candidates_v2_overview.png",
    )
    totals = {
        kind: sum(
            candidate.type == kind
            for result in results.values()
            for candidate in result.candidates
        )
        for kind in ("door", "window", "unknown")
    }
    print(json.dumps({
        "floor_z": floor,
        "ceiling_z": ceiling,
        "walls": len(walls),
        **totals,
        "topology_doors": len(topology),
    }, ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
