"""Exporta o contrato editavel do Plan to BIM como uma planta DXF 2D metrificada."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import ezdxf
import numpy as np


APP_ID = "PLAN2BIM"


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _layers(doc: ezdxf.document.Drawing) -> None:
    definitions = {
        "A-WALL-EXT": {"color": 30, "lineweight": 50},
        "A-WALL-INT": {"color": 7, "lineweight": 35},
        "A-WALL-HOST": {"color": 8, "linetype": "DASHED", "lineweight": 9},
        "A-DOOR": {"color": 3, "lineweight": 25},
        "A-WINDOW": {"color": 5, "lineweight": 25},
        "A-SLAB": {"color": 8, "lineweight": 25},
        "A-SPACE": {"color": 4, "lineweight": 13},
        "A-DIMS": {"color": 6, "lineweight": 13},
        "A-ANNO": {"color": 7, "lineweight": 13},
    }
    for name, attribs in definitions.items():
        if name not in doc.layers:
            doc.layers.add(name, dxfattribs=attribs)
    doc.layers.get("A-WALL-HOST").dxf.plot = 0


def _xdata(entity: Any, *values: Any) -> None:
    tags = []
    for value in values:
        if isinstance(value, (float, int)):
            tags.append((1040, float(value)))
        else:
            tags.append((1000, str(value)))
    entity.set_xdata(APP_ID, tags)


def _wall_geometry(wall: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    a = np.array([float(wall["ax"]), float(wall["ay"])], dtype=float)
    b = np.array([float(wall["bx"]), float(wall["by"])], dtype=float)
    length = float(np.linalg.norm(b - a))
    if length <= 1e-9:
        raise ValueError(f"Parede degenerada: {wall['id']}")
    unit = (b - a) / length
    normal = np.array([-unit[1], unit[0]])
    return a, unit, normal, length


def _arc_geometry(wall: dict[str, Any]) -> dict[str, Any] | None:
    if wall.get("geometria") != "arco" or not isinstance(wall.get("curva"), dict):
        return None
    ax, ay = float(wall["ax"]), float(wall["ay"])
    bx, by = float(wall["bx"]), float(wall["by"])
    cx, cy = float(wall["curva"]["x"]), float(wall["curva"]["y"])
    denominator = 2.0 * (
        ax * (cy - by) + cx * (by - ay) + bx * (ay - cy)
    )
    chord = math.hypot(bx - ax, by - ay)
    if chord <= 1e-9 or abs(denominator) < chord * chord * 1e-7:
        return None
    a2, c2, b2 = ax * ax + ay * ay, cx * cx + cy * cy, bx * bx + by * by
    ox = (a2 * (cy - by) + c2 * (by - ay) + b2 * (ay - cy)) / denominator
    oy = (a2 * (bx - cx) + c2 * (ax - bx) + b2 * (cx - ax)) / denominator
    radius = math.hypot(ax - ox, ay - oy)
    start = math.atan2(ay - oy, ax - ox)
    control = math.atan2(cy - oy, cx - ox)
    end = math.atan2(by - oy, bx - ox)
    ccw_sweep = (end - start) % (2.0 * math.pi)
    ccw_control = (control - start) % (2.0 * math.pi)
    sweep = ccw_sweep if ccw_control <= ccw_sweep + 1e-7 else ccw_sweep - 2.0 * math.pi
    return {
        "center": np.array([ox, oy], dtype=float),
        "radius": radius,
        "start": start,
        "sweep": sweep,
        "length": abs(sweep) * radius,
    }


def _wall_length(wall: dict[str, Any]) -> float:
    arc = _arc_geometry(wall)
    return float(arc["length"] if arc else _wall_geometry(wall)[3])


def _wall_frame(
    wall: dict[str, Any],
    distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arc = _arc_geometry(wall)
    length = _wall_length(wall)
    distance = max(0.0, min(length, float(distance)))
    if not arc:
        a, unit, normal, _ = _wall_geometry(wall)
        return a + unit * distance, unit, normal
    angle = arc["start"] + arc["sweep"] * distance / length
    direction = 1.0 if arc["sweep"] >= 0 else -1.0
    unit = np.array([-math.sin(angle) * direction,
                     math.cos(angle) * direction], dtype=float)
    normal = np.array([-unit[1], unit[0]], dtype=float)
    point = arc["center"] + arc["radius"] * np.array(
        [math.cos(angle), math.sin(angle)], dtype=float)
    return point, unit, normal


def _wall_path(
    wall: dict[str, Any],
    start: float,
    end: float,
    max_segment: float = 0.10,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    count = 1 if _arc_geometry(wall) is None else max(
        2, min(512, int(math.ceil((end - start) / max_segment)))
    )
    return [_wall_frame(wall, start + (end - start) * i / count)
            for i in range(count + 1)]


def _wall_strip(
    wall: dict[str, Any],
    start: float,
    end: float,
    half_width: float,
) -> list[tuple[float, float]]:
    frames = _wall_path(wall, start, end)
    left = [tuple(point + normal * half_width) for point, _, normal in frames]
    right = [tuple(point - normal * half_width)
             for point, _, normal in reversed(frames)]
    return left + right


def _rectangle(
    a: np.ndarray,
    unit: np.ndarray,
    normal: np.ndarray,
    start: float,
    end: float,
    half_width: float,
) -> list[tuple[float, float]]:
    p1 = a + unit * start
    p2 = a + unit * end
    return [
        tuple(p1 + normal * half_width),
        tuple(p2 + normal * half_width),
        tuple(p2 - normal * half_width),
        tuple(p1 - normal * half_width),
    ]


def export_model_to_dxf(model: dict[str, Any], output: Path) -> dict[str, Any]:
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 6  # metros
    doc.header["$MEASUREMENT"] = 1
    doc.header["$LTSCALE"] = 0.05
    if APP_ID not in doc.appids:
        doc.appids.add(APP_ID)
    _layers(doc)
    msp = doc.modelspace()

    walls = {str(wall["id"]): wall for wall in model.get("paredes", [])}
    hosted: dict[str, list[dict[str, Any]]] = {wall_id: [] for wall_id in walls}
    for opening in model.get("aberturas", []):
        wall_id = str(opening["parede_id"])
        if wall_id not in walls:
            raise ValueError(f"Abertura orfa: {opening.get('id')}")
        hosted[wall_id].append(opening)

    wall_polylines = 0
    for wall_id, wall in walls.items():
        a, unit, normal, chord_length = _wall_geometry(wall)
        length = _wall_length(wall)
        thickness = float(wall["espessura"])
        layer = "A-WALL-EXT" if "Exterior" in str(wall.get("layer", "")) else "A-WALL-INT"
        intervals = []
        for opening in hosted[wall_id]:
            center = float(opening["s_centro"])
            width = float(opening["largura"])
            start, end = center - width / 2.0, center + width / 2.0
            if start < -1e-6 or end > length + 1e-6:
                raise ValueError(f"Abertura {opening['id']} nao cabe em {wall_id}")
            intervals.append((max(0.0, start), min(length, end)))
        intervals.sort()
        cursor = 0.0
        segments: list[tuple[float, float]] = []
        for start, end in intervals:
            if start > cursor + 1e-6:
                segments.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < length - 1e-6:
            segments.append((cursor, length))
        for index, (start, end) in enumerate(segments, 1):
            points = _wall_strip(wall, start, end, thickness / 2.0)
            entity = msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})
            _xdata(entity, wall_id, wall.get("nome", wall_id), index, thickness, wall.get("altura", 2.8))
            wall_polylines += 1
        axis_points = [tuple(point) for point, _, _ in _wall_path(wall, 0.0, length)]
        axis = msp.add_lwpolyline(axis_points, dxfattribs={"layer": "A-WALL-HOST"})
        _xdata(axis, wall_id, "PAREDE_MAE_CONTINUA", length)

    for opening in model.get("aberturas", []):
        wall = walls[str(opening["parede_id"])]
        center = float(opening["s_centro"])
        width = float(opening["largura"])
        thickness = float(wall["espessura"])
        p1, unit1, normal1 = _wall_frame(wall, center - width / 2.0)
        p2, _unit2, normal2 = _wall_frame(wall, center + width / 2.0)
        jamb1 = msp.add_line(tuple(p1 - normal1 * thickness / 2.0), tuple(p1 + normal1 * thickness / 2.0), dxfattribs={"layer": "A-DOOR" if opening["tipo"] == "door" else "A-WINDOW"})
        jamb2 = msp.add_line(tuple(p2 - normal2 * thickness / 2.0), tuple(p2 + normal2 * thickness / 2.0), dxfattribs={"layer": "A-DOOR" if opening["tipo"] == "door" else "A-WINDOW"})
        _xdata(jamb1, opening["id"], opening.get("nome", ""), width)
        _xdata(jamb2, opening["id"], opening.get("nome", ""), width)
        if opening["tipo"] == "door":
            leaf_end = p1 + normal1 * width
            leaf = msp.add_line(tuple(p1), tuple(leaf_end), dxfattribs={"layer": "A-DOOR"})
            axis_angle = math.degrees(math.atan2(unit1[1], unit1[0]))
            arc = msp.add_arc(tuple(p1), width, axis_angle, axis_angle + 90.0, dxfattribs={"layer": "A-DOOR"})
            _xdata(leaf, opening["id"], "FOLHA", width)
            _xdata(arc, opening["id"], "GIRO", width)
        else:
            for offset in (-thickness * 0.22, thickness * 0.22):
                window_points = [
                    tuple(point + frame_normal * offset)
                    for point, _, frame_normal in _wall_path(
                        wall, center - width / 2.0, center + width / 2.0, 0.06
                    )
                ]
                line = msp.add_lwpolyline(window_points, dxfattribs={"layer": "A-WINDOW"})
                _xdata(line, opening["id"], opening.get("nome", ""), width, opening.get("peitoril", 1.0))

    slab = model.get("laje", {}).get("contorno", [])
    if len(slab) >= 3:
        entity = msp.add_lwpolyline(slab, close=True, dxfattribs={"layer": "A-SLAB"})
        _xdata(entity, "SLAB-01", model.get("laje", {}).get("piso", {}).get("espessura", 0.12))

    for space in model.get("spaces", []):
        contour = np.array(space.get("contorno", []), dtype=float)
        if len(contour) < 3:
            continue
        entity = msp.add_lwpolyline([tuple(point) for point in contour], close=True, dxfattribs={"layer": "A-SPACE"})
        _xdata(entity, space["id"], space.get("nome", ""))
        centroid = contour.mean(axis=0)
        text = msp.add_text(str(space.get("nome") or space["id"]), height=0.18, dxfattribs={"layer": "A-SPACE"})
        text.set_placement(tuple(centroid), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)

    dimensions = 0
    for dimension in model.get("dimensions", []):
        p1 = tuple(float(value) for value in dimension["start"])
        p2 = tuple(float(value) for value in dimension["end"])
        horizontal = abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1])
        dim = msp.add_linear_dim(
            base=p1,
            p1=p1,
            p2=p2,
            angle=0.0 if horizontal else 90.0,
            dimstyle="EZDXF",
            override={
                "dimtxt": 0.16,
                "dimasz": 0.10,
                "dimdec": 2,
                "dimlfac": 1.0,
                "dimpost": "<> m",
            },
            dxfattribs={"layer": "A-DIMS"},
        )
        dim.render()
        dimensions += 1

    title = msp.add_text("PLAN TO BIM - FUNDO - UNIDADES EM METROS", height=0.22, dxfattribs={"layer": "A-ANNO"})
    title.set_placement((0.0, 5.65))
    note = msp.add_text(str(model.get("scale_note", "Escala metrificada")), height=0.13, dxfattribs={"layer": "A-ANNO"})
    note.set_placement((0.0, 5.42))

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output)
    return {
        "output": str(output),
        "units": "meters",
        "walls": len(walls),
        "wall_polylines_split_at_openings": wall_polylines,
        "doors": sum(item["tipo"] == "door" for item in model.get("aberturas", [])),
        "windows": sum(item["tipo"] == "window" for item in model.get("aberturas", [])),
        "spaces": len(model.get("spaces", [])),
        "dimensions": dimensions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = export_model_to_dxf(_load(args.model), args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
