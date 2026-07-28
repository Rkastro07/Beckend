#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure wall evidence separately in each height layer and on each face.

This is a diagnostic companion to ``wall_detector_v2``.  It reads an IFC
result and streams the original XYZ cloud, avoiding a second full-cloud copy
in memory.  The resulting CSV and PNG make furniture-like candidates visible:
they have longitudinal support in lower layers but disappear near the ceiling.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import ifcopenshell
import ifcopenshell.util.placement as placement
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class WallProbe:
    index: int
    axis: np.ndarray
    thickness: float
    length: float
    u: np.ndarray
    n: np.ndarray
    bins: int
    axis_hits: np.ndarray
    face_hits: np.ndarray


def _wall_geometry(ifc_path: Path, layers: int, longitudinal_step: float):
    model = ifcopenshell.open(str(ifc_path))
    probes = []
    for index, wall in enumerate(model.by_type("IfcWall")):
        transform = placement.get_local_placement(wall.ObjectPlacement)
        axis = None
        thickness = None
        for representation in wall.Representation.Representations:
            if representation.RepresentationIdentifier == "Axis":
                for item in representation.Items:
                    if not item.is_a("IfcPolyline"):
                        continue
                    points = []
                    for cartesian_point in item.Points[:2]:
                        coordinates = list(cartesian_point.Coordinates)
                        local = np.array(
                            [coordinates[0], coordinates[1], 0.0, 1.0],
                            dtype=float,
                        )
                        points.append((transform @ local)[:2])
                    axis = np.asarray(points, dtype=float)
            elif representation.RepresentationIdentifier == "Body":
                for item in representation.Items:
                    if (
                        item.is_a("IfcExtrudedAreaSolid")
                        and item.SweptArea.is_a("IfcRectangleProfileDef")
                    ):
                        thickness = float(
                            min(item.SweptArea.XDim, item.SweptArea.YDim)
                        )
        if axis is None or thickness is None:
            continue
        direction = axis[1] - axis[0]
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            continue
        u = direction / length
        n = np.array([-u[1], u[0]], dtype=float)
        bin_count = max(1, int(np.ceil(length / longitudinal_step)))
        probes.append(
            WallProbe(
                index=index,
                axis=axis,
                thickness=thickness,
                length=length,
                u=u,
                n=n,
                bins=bin_count,
                axis_hits=np.zeros((layers, bin_count), dtype=bool),
                face_hits=np.zeros((2, layers, bin_count), dtype=bool),
            )
        )
    return probes


def _mark_hits(hit_grid, z_index, s_index):
    np.logical_or.at(hit_grid, (z_index, s_index), True)


def measure(
    xyz_path: Path,
    probes: list[WallProbe],
    z_min: float,
    z_max: float,
    layers: int,
    longitudinal_step: float,
    face_tolerance: float,
    chunk_size: int,
):
    processed = 0
    for chunk in pd.read_csv(
        xyz_path,
        sep=r"\s+",
        usecols=[0, 1, 2],
        dtype=np.float32,
        chunksize=chunk_size,
    ):
        points = chunk.to_numpy(dtype=np.float32, copy=False)
        points = points[
            np.isfinite(points).all(axis=1)
            & (points[:, 2] >= z_min)
            & (points[:, 2] <= z_max)
        ]
        if not len(points):
            continue
        xy = points[:, :2]
        z_index = np.floor(
            (points[:, 2] - z_min) / max(z_max - z_min, 1e-9) * layers
        ).astype(np.int32)
        z_index = np.clip(z_index, 0, layers - 1)
        for probe in probes:
            relative = xy - probe.axis[0]
            longitudinal = relative @ probe.u
            normal = relative @ probe.n
            inside_length = (
                (longitudinal >= -longitudinal_step)
                & (longitudinal <= probe.length + longitudinal_step)
            )
            if not inside_length.any():
                continue
            s_index = np.floor(
                np.clip(longitudinal, 0.0, probe.length - 1e-9)
                / longitudinal_step
            ).astype(np.int32)
            s_index = np.clip(s_index, 0, probe.bins - 1)

            corridor = inside_length & (
                np.abs(normal) <= 0.5 * probe.thickness + face_tolerance
            )
            if corridor.any():
                _mark_hits(
                    probe.axis_hits,
                    z_index[corridor],
                    s_index[corridor],
                )

            for face_index, offset in enumerate(
                (-0.5 * probe.thickness, 0.5 * probe.thickness)
            ):
                on_face = inside_length & (
                    np.abs(normal - offset) <= face_tolerance
                )
                if on_face.any():
                    _mark_hits(
                        probe.face_hits[face_index],
                        z_index[on_face],
                        s_index[on_face],
                    )
        processed += len(points)
        if processed and processed % 5_000_000 < chunk_size:
            print("Measured %d in-band points..." % processed)


def _metrics(probe: WallProbe):
    axis_coverage = probe.axis_hits.mean(axis=1)
    face_coverage = probe.face_hits.mean(axis=2)
    best_face = face_coverage.max(axis=0)
    weak_face = face_coverage.min(axis=0)
    vertical_span_per_bin = probe.axis_hits.sum(axis=0)
    return {
        "axis": axis_coverage,
        "face0": face_coverage[0],
        "face1": face_coverage[1],
        "best_face": best_face,
        "weak_face": weak_face,
        "persistent_4": float((vertical_span_per_bin >= 4).mean()),
        "persistent_5": float((vertical_span_per_bin >= 5).mean()),
        "top2": float(axis_coverage[-2:].mean()),
    }


def write_outputs(
    probes: list[WallProbe],
    output_prefix: Path,
    layers: int,
    z_min: float,
    z_max: float,
):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for probe in probes:
        metrics = _metrics(probe)
        row = {
            "wall_index": probe.index,
            "length_m": probe.length,
            "thickness_m": probe.thickness,
            "persistent_4_layers": metrics["persistent_4"],
            "persistent_5_layers": metrics["persistent_5"],
            "top2_axis_coverage": metrics["top2"],
        }
        for layer in range(layers):
            row["axis_layer_%d" % (layer + 1)] = metrics["axis"][layer]
            row["face0_layer_%d" % (layer + 1)] = metrics["face0"][layer]
            row["face1_layer_%d" % (layer + 1)] = metrics["face1"][layer]
        rows.append(row)

    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    matrix = np.asarray(
        [[_metrics(probe)["axis"][layer] for layer in range(layers)]
         for probe in probes],
        dtype=float,
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, max(7, 0.35 * len(probes))),
        gridspec_kw={"width_ratios": [1.05, 1.55]},
    )
    image = axes[0].imshow(
        matrix,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axes[0].set_title("Cobertura longitudinal por camada")
    axes[0].set_xlabel("Camada (1=baixa, %d=alta)" % layers)
    axes[0].set_ylabel("Índice da parede no IFC")
    axes[0].set_xticks(range(layers), range(1, layers + 1))
    axes[0].set_yticks(range(len(probes)), [probe.index for probe in probes])
    figure.colorbar(image, ax=axes[0], label="fração do comprimento")

    for probe in probes:
        axis = probe.axis
        axes[1].plot(axis[:, 0], axis[:, 1], linewidth=2)
        midpoint = axis.mean(axis=0)
        axes[1].text(
            midpoint[0],
            midpoint[1],
            str(probe.index),
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    axes[1].axis("equal")
    axes[1].grid(alpha=0.2)
    axes[1].set_title("Mapa dos índices analisados")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    figure.suptitle(
        "Persistência vertical %.3f m a %.3f m" % (z_min, z_max)
    )
    figure.tight_layout()
    png_path = output_prefix.with_suffix(".png")
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return csv_path, png_path, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ifc", type=Path)
    parser.add_argument("xyz", type=Path)
    parser.add_argument("--z-min", type=float, required=True)
    parser.add_argument("--z-max", type=float, required=True)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--longitudinal-step", type=float, default=0.095)
    parser.add_argument("--face-tolerance", type=float, default=0.10)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--output-prefix", type=Path, required=True)
    arguments = parser.parse_args()

    probes = _wall_geometry(
        arguments.ifc,
        arguments.layers,
        arguments.longitudinal_step,
    )
    measure(
        arguments.xyz,
        probes,
        arguments.z_min,
        arguments.z_max,
        arguments.layers,
        arguments.longitudinal_step,
        arguments.face_tolerance,
        arguments.chunk_size,
    )
    csv_path, png_path, rows = write_outputs(
        probes,
        arguments.output_prefix,
        arguments.layers,
        arguments.z_min,
        arguments.z_max,
    )
    for row in rows:
        layer_values = [
            row["axis_layer_%d" % layer]
            for layer in range(1, arguments.layers + 1)
        ]
        print(
            "W%02d L=%5.2f th=%4.2f layers=%s top2=%.2f persistent4=%.2f"
            % (
                row["wall_index"],
                row["length_m"],
                row["thickness_m"],
                " ".join("%.2f" % value for value in layer_values),
                row["top2_axis_coverage"],
                row["persistent_4_layers"],
            )
        )
    print("CSV:", csv_path)
    print("PNG:", png_path)


if __name__ == "__main__":
    main()
