"""Review open door/window leaves that were reconstructed as diagonal walls.

This is an approval-stage diagnostic.  It consumes the wall/opening geometry
already shown in the Cloud2BIM PNG, evaluates raw point support, and writes:

* ``open_leaf_candidates.json`` with auditable geometric evidence;
* ``open_leaf_before_after.png`` with the proposed wall suppression.

It does not mutate the IFC.  Approved ``suppress`` entries are intended to be
fed into the topology rebuild before IfcSpace generation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cloud2bim_patched.wall_detector_v2 import (
    detect_articulated_leaf_walls,
    keep_non_leaf_wall_indices,
)


COLORS = {
    "wall": "#263b55",
    "cloud": "#e1e8f0",
    "opening": "#0891b2",
    "leaf": "#d946ef",
    "review": "#f59e0b",
    "removed": "#cbd5e1",
}


def load_thicknesses(path: Path):
    result = {}
    with open(path, newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            try:
                result[str(row["reference"])] = float(row["thickness"])
            except (KeyError, TypeError, ValueError):
                continue
    return result


def load_geometry(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    walls = {}
    anchors = []
    for wall in payload.get("walls", []):
        identifier = str(wall["wall_id"])
        start = np.asarray(wall["start"], dtype=float)[:2]
        end = np.asarray(wall["end"], dtype=float)[:2]
        axis = [start.tolist(), end.tolist()]
        walls[identifier] = axis
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 0:
            continue
        direction /= length
        for candidate in wall.get("candidates", []):
            center = start + direction * float(candidate["s_center"])
            anchors.append({
                "id": str(candidate["id"]),
                "host_wall": identifier,
                "type": str(candidate["type"]),
                "width": float(candidate["width"]),
                "center": center.tolist(),
                "host_axis": axis,
                "status": str(candidate.get("status", "review")),
            })
    for candidate in payload.get("topology_candidates", []):
        host = str(candidate["host_wall"])
        if host not in walls:
            continue
        anchors.append({
            "id": str(candidate["id"]),
            "host_wall": host,
            "type": str(candidate.get("type", "door")),
            "width": float(candidate["width"]),
            "center": [float(value)
                       for value in candidate["global_center"][:2]],
            "host_axis": walls[host],
            "status": str(candidate.get("status", "review")),
        })
    return payload, walls, anchors


def font(size: int, bold: bool = False):
    filename = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / filename
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def render(points, wall_ids, axes, anchors, results, output: Path):
    width, height = 7000, 3400
    panel_width = width // 2
    outer_margin = 150
    title_height = 230
    footer_height = 170
    array_axes = [np.asarray(axis, dtype=float) for axis in axes]
    all_xy = np.vstack([points[:, :2], *array_axes])
    lower, upper = all_xy.min(axis=0), all_xy.max(axis=0)
    padding = max(0.8, 0.04 * float(np.max(upper - lower)))
    lower -= padding
    upper += padding
    scale = min(
        (panel_width - 2 * outer_margin) / float(upper[0] - lower[0]),
        (height - title_height - footer_height)
        / float(upper[1] - lower[1]),
    )

    def pixel(values, panel):
        values = np.asarray(values, dtype=float)
        result = np.empty_like(values)
        panel_offset = panel * panel_width
        result[..., 0] = (
            panel_offset + outer_margin + (values[..., 0] - lower[0]) * scale
        )
        result[..., 1] = (
            height - footer_height - (values[..., 1] - lower[1]) * scale
        )
        return np.rint(result).astype(int)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    sample = points[::max(1, len(points) // 400_000)]
    suppressed_ids = {
        result["wall_id"] for result in results if result.get("suppress")
    }
    result_by_wall = {result["wall_id"]: result for result in results}

    for panel in range(2):
        cloud_pixels = pixel(sample[:, :2], panel)
        draw.point([tuple(value) for value in cloud_pixels],
                   fill=COLORS["cloud"])
        for identifier, axis in zip(wall_ids, array_axes):
            suppressed = identifier in suppressed_ids
            if panel == 1 and suppressed:
                colour, line_width = COLORS["removed"], 4
            else:
                colour = (
                    COLORS["leaf"]
                    if suppressed
                    else COLORS["review"]
                    if identifier in result_by_wall
                    else COLORS["wall"]
                )
                line_width = 9 if identifier in result_by_wall else 6
            draw.line(
                [tuple(value) for value in pixel(axis, panel)],
                fill=colour,
                width=line_width,
            )
        for anchor in anchors:
            host_axis = np.asarray(anchor["host_axis"], dtype=float)
            direction = host_axis[1] - host_axis[0]
            direction /= np.linalg.norm(direction)
            center = np.asarray(anchor["center"], dtype=float)
            jambs = [
                center - direction * anchor["width"] * 0.5,
                center + direction * anchor["width"] * 0.5,
            ]
            for jamb in jambs:
                x, y = pixel(jamb, panel)
                draw.ellipse(
                    (x - 6, y - 6, x + 6, y + 6),
                    fill=COLORS["opening"],
                )

    title_font = font(48, bold=True)
    subtitle_font = font(34, bold=True)
    label_font = font(25, bold=True)
    body_font = font(27)
    draw.text(
        (width // 2, 32),
        "Detector de folhas abertas — parede falsa x painel articulado",
        font=title_font,
        fill="#111827",
        anchor="ma",
    )
    draw.text(
        (panel_width // 2, 115),
        "ANTES — paredes detectadas",
        font=subtitle_font,
        fill=COLORS["wall"],
        anchor="ma",
    )
    draw.text(
        (panel_width + panel_width // 2, 115),
        "DEPOIS — sugestão para aprovação",
        font=subtitle_font,
        fill=COLORS["leaf"],
        anchor="ma",
    )
    draw.line(
        [(panel_width, title_height - 25),
         (panel_width, height - footer_height + 20)],
        fill="#cbd5e1",
        width=3,
    )

    for panel in range(2):
        for result in results:
            axis_index = int(result["wall_index"])
            center = np.mean(array_axes[axis_index], axis=0)
            x, y = pixel(center, panel)
            colour = (
                COLORS["leaf"] if result.get("suppress") else COLORS["review"]
            )
            if result.get("source") == "user_ground_truth":
                label = f'{result["wall_id"]} — remoção aprovada'
            else:
                label = (
                    f'{result["wall_id"]} → {result["opening_id"]} '
                    f'({result["open_angle_deg"]:.0f}°)'
                )
            bbox = draw.textbbox((x + 15, y - 15), label, font=label_font)
            draw.rounded_rectangle(
                (bbox[0] - 7, bbox[1] - 5, bbox[2] + 7, bbox[3] + 5),
                radius=7,
                fill="white",
                outline=colour,
                width=2,
            )
            draw.text(
                (x + 15, y - 15),
                label,
                font=label_font,
                fill=colour,
            )

    proposed = sum(result.get("suppress") for result in results)
    review = len(results) - proposed
    footer = (
        f"Magenta = folha sugerida para supressão ({proposed})  |  "
        f"Laranja = revisar ({review})  |  "
        "Ciano = batentes das aberturas  |  "
        "Cinza no painel direito = parede retirada apenas da prévia"
    )
    draw.text(
        (width // 2, height - 90),
        footer,
        font=body_font,
        fill="#334155",
        anchor="mm",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def evaluate(results, expected):
    predicted = {
        result["wall_id"] for result in results if result.get("suppress")
    }
    expected = set(expected)
    true_positive = predicted & expected
    false_positive = predicted - expected
    false_negative = expected - predicted
    precision = (
        len(true_positive) / len(predicted) if predicted else 0.0
    )
    recall = (
        len(true_positive) / len(expected) if expected else 0.0
    )
    return {
        "expected_not_walls": sorted(expected),
        "predicted_open_leaves": sorted(predicted),
        "true_positive": sorted(true_positive),
        "false_positive": sorted(false_positive),
        "false_negative": sorted(false_negative),
        "precision": float(precision),
        "recall_against_all_non_walls": float(recall),
    }


def apply_approved_non_walls(results, wall_ids, axes, thicknesses, approved):
    """Merge explicit PNG approvals without teaching IDs to the detector."""
    index_by_id = {
        identifier: index for index, identifier in enumerate(wall_ids)
    }
    result_by_id = {
        result["wall_id"]: result for result in results
    }
    unknown = sorted(set(approved) - set(index_by_id))
    if unknown:
        raise ValueError(
            "paredes aprovadas ausentes na geometria: " + ", ".join(unknown)
        )
    for identifier in approved:
        if identifier in result_by_id:
            result = result_by_id[identifier]
            result["status"] = "approved"
            result["suppress"] = True
            result["approval_source"] = "user_ground_truth"
            continue
        index = index_by_id[identifier]
        axis = np.asarray(axes[index], dtype=float)
        result = {
            "wall_id": identifier,
            "wall_index": int(index),
            "opening_id": "",
            "host_wall": "",
            "type": "manual_non_wall",
            "hinge": None,
            "free_edge": None,
            "matched_jamb": None,
            "hinge_distance": None,
            "length": float(np.linalg.norm(axis[1] - axis[0])),
            "opening_width": None,
            "length_width_ratio": None,
            "thickness": float(thicknesses[index]),
            "open_angle_deg": None,
            "non_orthogonality_deg": None,
            "free_edge_clearance": None,
            "vertical_match": None,
            "profile": {},
            "geometry_score": None,
            "score": 1.0,
            "status": "approved",
            "suppress": True,
            "source": "user_ground_truth",
        }
        results.append(result)
        result_by_id[identifier] = result
    return sorted(results, key=lambda result: result["wall_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("openings", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--expected-not-walls",
        default="",
        help="IDs separados por virgula, usados somente na avaliacao do teste",
    )
    parser.add_argument(
        "--approved-non-walls",
        default="",
        help=(
            "IDs aprovados no PNG, separados por virgula; entram apenas na "
            "previa e no contrato de revisao, sem alterar o IFC"
        ),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    payload, walls, anchors = load_geometry(args.openings)
    thicknesses = load_thicknesses(args.diagnostics)
    wall_ids = list(walls)
    axes = [walls[identifier] for identifier in wall_ids]
    wall_thicknesses = [
        thicknesses.get(identifier, 0.15) for identifier in wall_ids
    ]
    points = np.loadtxt(args.xyz, skiprows=1, usecols=(0, 1, 2))
    results = detect_articulated_leaf_walls(
        axes,
        wall_thicknesses,
        wall_ids,
        anchors,
        points,
        float(payload["floor_z"]),
        float(payload["ceiling_z"]),
    )
    approved = [
        value.strip()
        for value in args.approved_non_walls.split(",")
        if value.strip()
    ]
    results = apply_approved_non_walls(
        results,
        wall_ids,
        axes,
        wall_thicknesses,
        approved,
    )
    kept = keep_non_leaf_wall_indices(len(axes), results)
    expected = [
        value.strip()
        for value in args.expected_not_walls.split(",")
        if value.strip()
    ]
    report = {
        "schema": "cloud2bim.open-leaf-review.v1",
        "source_xyz": str(args.xyz),
        "source_openings": str(args.openings),
        "wall_count_before": len(axes),
        "opening_anchor_count": len(anchors),
        "candidate_count": len(results),
        "approved_non_walls": sorted(approved),
        "suppression_proposal_count": sum(
            result.get("suppress") for result in results),
        "wall_count_after_preview": len(kept),
        "candidates": results,
    }
    if expected:
        report["evaluation"] = evaluate(results, expected)
    json_path = args.output / "open_leaf_candidates.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    png_path = args.output / "open_leaf_before_after.png"
    render(points, wall_ids, axes, anchors, results, png_path)
    print(json.dumps({
        "walls_before": len(axes),
        "opening_anchors": len(anchors),
        "candidates": len(results),
        "proposed_suppression": report["suppression_proposal_count"],
        "walls_after_preview": len(kept),
        "json": str(json_path),
        "png": str(png_path),
        "evaluation": report.get("evaluation"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
