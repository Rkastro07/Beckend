"""Renderiza paredes e laje de um IFC em PNGs metricos e identificados.

O leitor STEP e propositalmente pequeno e cobre a geometria produzida pelo
modelador Planta-to-BIM deste repositorio. Nao substitui IfcOpenShell para IFCs
arbitrarios; serve para diagnostico visual quando as dependencias nativas nao
estao disponiveis.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[-+]?\d+)?"


@dataclass(frozen=True)
class StepEntity:
    ifc_class: str
    args: str


@dataclass(frozen=True)
class Wall2D:
    code: str
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float
    polygon: tuple[tuple[float, float], ...]

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)


def parse_step(path: Path) -> dict[int, StepEntity]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        int(entity_id): StepEntity(ifc_class, args)
        for entity_id, ifc_class, args in re.findall(
            r"#(\d+)=([A-Z0-9_]+)\((.*?)\);",
            text,
            re.DOTALL,
        )
    }


def refs(entity: StepEntity) -> list[int]:
    return [int(value) for value in re.findall(r"#(\d+)", entity.args)]


def find_descendants(
    entities: dict[int, StepEntity],
    root_id: int,
    ifc_class: str,
) -> list[int]:
    found = []
    pending = [root_id]
    visited = set()
    while pending:
        entity_id = pending.pop()
        if entity_id in visited or entity_id not in entities:
            continue
        visited.add(entity_id)
        entity = entities[entity_id]
        if entity.ifc_class == ifc_class:
            found.append(entity_id)
        pending.extend(reversed(refs(entity)))
    return found


def point(entities: dict[int, StepEntity], entity_id: int) -> tuple[float, ...]:
    entity = entities[entity_id]
    values_match = re.search(r"\((.*?)\)", entity.args)
    if not values_match:
        raise ValueError(f"#{entity_id} nao contem uma lista de coordenadas")
    return tuple(float(value) for value in re.findall(NUMBER, values_match.group(1)))


def entity_name(entity: StepEntity, fallback: str) -> str:
    strings = re.findall(r"'([^']*)'", entity.args)
    return strings[1] if len(strings) > 1 else fallback


def normalize2(vector: tuple[float, ...]) -> tuple[float, float]:
    x, y = vector[:2]
    length = math.hypot(x, y)
    return (x / length, y / length) if length > 1e-12 else (1.0, 0.0)


def local_frame(
    entities: dict[int, StepEntity],
    placement_id: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    axes = find_descendants(entities, placement_id, "IFCAXIS2PLACEMENT3D")
    if not axes:
        return (0.0, 0.0), (1.0, 0.0)
    axis = entities[axes[0]]
    axis_refs = refs(axis)
    origin3 = point(entities, axis_refs[0])
    x_axis = (
        normalize2(point(entities, axis_refs[2]))
        if len(axis_refs) >= 3
        else (1.0, 0.0)
    )
    return (origin3[0], origin3[1]), x_axis


def rectangle_profile(
    entities: dict[int, StepEntity],
    product_id: int,
) -> tuple[float, float, tuple[float, float]]:
    candidates = []
    for profile_id in find_descendants(
        entities,
        product_id,
        "IFCRECTANGLEPROFILEDEF",
    ):
        profile = entities[profile_id]
        parts = profile.args.rsplit(",", 2)
        if len(parts) != 3:
            continue
        length, thickness = float(parts[-2]), float(parts[-1])
        profile_refs = refs(profile)
        center = (0.0, 0.0)
        if profile_refs:
            centers = find_descendants(
                entities,
                profile_refs[0],
                "IFCCARTESIANPOINT",
            )
            if centers:
                raw_center = point(entities, centers[0])
                center = (
                    raw_center[0],
                    raw_center[1] if len(raw_center) > 1 else 0.0,
                )
        candidates.append((length * thickness, length, thickness, center))
    if not candidates:
        raise ValueError(f"Produto #{product_id} nao possui perfil retangular")
    _, length, thickness, center = max(candidates)
    return length, thickness, center


def extract_walls(entities: dict[int, StepEntity]) -> list[Wall2D]:
    walls = []
    named_wall_ids = []
    for entity_id, entity in entities.items():
        if entity.ifc_class != "IFCWALL":
            continue
        name = entity_name(entity, "")
        normalized = name.lower()
        if normalized.startswith("verga-"):
            continue
        if normalized.startswith("parede-"):
            code = f"W-{int(name.rsplit('-', 1)[-1]):03d}"
        elif normalized == "fechamento-traseiro":
            code = "W-018"
        elif normalized == "fechamento-frontal":
            code = "W-019"
        else:
            code = f"W-{len(named_wall_ids) + 1:03d}"
        named_wall_ids.append((code, entity_id))
    named_wall_ids.sort(key=lambda item: item[0])
    for code, wall_id in named_wall_ids:
        wall = entities[wall_id]
        wall_refs = refs(wall)
        placement_id = next(
            ref for ref in wall_refs if entities[ref].ifc_class == "IFCLOCALPLACEMENT"
        )
        origin, x_axis = local_frame(entities, placement_id)
        length, thickness, profile_center = rectangle_profile(entities, wall_id)
        y_axis = (-x_axis[1], x_axis[0])
        center = (
            origin[0] + x_axis[0] * profile_center[0] + y_axis[0] * profile_center[1],
            origin[1] + x_axis[1] * profile_center[0] + y_axis[1] * profile_center[1],
        )
        start = (
            center[0] - x_axis[0] * length / 2,
            center[1] - x_axis[1] * length / 2,
        )
        end = (
            center[0] + x_axis[0] * length / 2,
            center[1] + x_axis[1] * length / 2,
        )
        polygon = tuple(
            (
                center[0] + sx * x_axis[0] * length / 2 + sy * y_axis[0] * thickness / 2,
                center[1] + sx * x_axis[1] * length / 2 + sy * y_axis[1] * thickness / 2,
            )
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        )
        walls.append(
            Wall2D(
                code=code,
                name=entity_name(wall, code),
                start=start,
                end=end,
                thickness=thickness,
                polygon=polygon,
            )
        )
    return walls


def extract_roof_slab(
    entities: dict[int, StepEntity],
) -> tuple[str, list[tuple[float, float]]]:
    slabs = [
        (entity_id, entity)
        for entity_id, entity in entities.items()
        if entity.ifc_class == "IFCSLAB"
    ]
    selected = next(
        (
            item
            for item in slabs
            if "cobertura" in entity_name(item[1], "").lower()
        ),
        slabs[-1] if slabs else None,
    )
    if selected is None:
        raise ValueError("IFC sem IfcSlab")
    slab_id, slab = selected
    polylines = find_descendants(entities, slab_id, "IFCPOLYLINE")
    if not polylines:
        raise ValueError("IfcSlab sem IfcPolyline no perfil")
    point_ids = refs(entities[polylines[0]])
    contour = []
    for point_id in point_ids:
        xy = point(entities, point_id)
        value = (xy[0], xy[1])
        if not contour or value != contour[-1]:
            contour.append(value)
    if len(contour) > 1 and contour[0] == contour[-1]:
        contour.pop()
    return entity_name(slab, "Laje-Cobertura"), contour


class Canvas:
    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        *,
        width: int = 2600,
        height: int = 1900,
        drawing_width: int = 1950,
        margin: int = 150,
    ):
        xmin, ymin, xmax, ymax = bounds
        self.width = width
        self.height = height
        self.drawing_width = drawing_width
        self.margin = margin
        self.scale = min(
            (drawing_width - 2 * margin) / max(xmax - xmin, 1e-6),
            (height - 2 * margin) / max(ymax - ymin, 1e-6),
        )
        self.offset_x = margin + (
            drawing_width - 2 * margin - (xmax - xmin) * self.scale
        ) / 2
        self.offset_y = margin + (
            height - 2 * margin - (ymax - ymin) * self.scale
        ) / 2
        self.xmin = xmin
        self.ymax = ymax
        self.image = Image.new("RGB", (width, height), "#f7f8fa")
        self.draw = ImageDraw.Draw(self.image)

    def xy(self, value: tuple[float, float]) -> tuple[float, float]:
        return (
            self.offset_x + (value[0] - self.xmin) * self.scale,
            self.offset_y + (self.ymax - value[1]) * self.scale,
        )


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def text_box(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    text_font: ImageFont.ImageFont,
    *,
    fill: str = "#111827",
    background: str = "#ffffff",
    anchor: str = "mm",
) -> None:
    box = draw.textbbox(position, text, font=text_font, anchor=anchor, stroke_width=0)
    pad = 7
    draw.rounded_rectangle(
        (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
        radius=7,
        fill=background,
        outline="#cbd5e1",
        width=2,
    )
    draw.text(position, text, font=text_font, fill=fill, anchor=anchor)


def bounds_of(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    points = list(points)
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            start[0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * start[1]
            for index, start in enumerate(points)
        )
        / 2
    )


def draw_header(canvas: Canvas, title: str, subtitle: str) -> None:
    canvas.draw.text(
        (80, 55),
        title,
        font=font(46, bold=True),
        fill="#111827",
        anchor="la",
    )
    canvas.draw.text(
        (82, 112),
        subtitle,
        font=font(24),
        fill="#475569",
        anchor="la",
    )
    canvas.draw.line((80, 145, canvas.width - 80, 145), fill="#d7dde5", width=3)


def draw_north_and_scale(
    canvas: Canvas,
    *,
    x: float = 115,
    y: float | None = None,
    scale_x: float = 205,
) -> None:
    y = y if y is not None else canvas.height - 125
    canvas.draw.line((x, y, x, y - 75), fill="#111827", width=5)
    canvas.draw.polygon(
        [(x, y - 92), (x - 12, y - 69), (x + 12, y - 69)],
        fill="#111827",
    )
    canvas.draw.text((x, y - 106), "N", font=font(24, bold=True), fill="#111827", anchor="ms")
    scale_length_m = 5.0
    scale_px = scale_length_m * canvas.scale
    sx = scale_x
    canvas.draw.line((sx, y, sx + scale_px, y), fill="#111827", width=6)
    canvas.draw.line((sx, y - 9, sx, y + 9), fill="#111827", width=4)
    canvas.draw.line((sx + scale_px, y - 9, sx + scale_px, y + 9), fill="#111827", width=4)
    canvas.draw.text(
        (sx + scale_px / 2, y - 18),
        "5 m",
        font=font(22, bold=True),
        fill="#111827",
        anchor="ms",
    )


def render_walls(
    walls: list[Wall2D],
    slab: list[tuple[float, float]],
    output: Path,
    source_name: str,
) -> None:
    all_points = [point for wall in walls for point in wall.polygon]
    canvas = Canvas(bounds_of(all_points))
    draw_header(
        canvas,
        "Door Ground Floor — paredes",
        f"Interpretação métrica atual de {source_name} • códigos W-###",
    )
    label_font = font(20, bold=True)
    for wall in walls:
        canvas.draw.polygon(
            [canvas.xy(point) for point in wall.polygon],
            fill="#334155",
            outline="#f97316",
            width=4,
        )
        midpoint = (
            (wall.start[0] + wall.end[0]) / 2,
            (wall.start[1] + wall.end[1]) / 2,
        )
        text_box(
            canvas.draw,
            canvas.xy(midpoint),
            wall.code,
            label_font,
            fill="#9a3412",
            background="#fff7ed",
        )
    draw_north_and_scale(canvas)
    legend_x = canvas.drawing_width + 35
    canvas.draw.rounded_rectangle(
        (legend_x, 185, canvas.width - 55, canvas.height - 65),
        radius=18,
        fill="#ffffff",
        outline="#d7dde5",
        width=3,
    )
    canvas.draw.text(
        (legend_x + 30, 220),
        "PAREDES",
        font=font(28, bold=True),
        fill="#111827",
    )
    canvas.draw.text(
        (legend_x + 30, 265),
        "Código     comp.     esp.",
        font=font(20, bold=True),
        fill="#64748b",
    )
    y = 310
    for wall in walls:
        canvas.draw.text(
            (legend_x + 30, y),
            f"{wall.code}     {wall.length:5.2f} m     {wall.thickness:.2f} m",
            font=font(21),
            fill="#1f2937",
        )
        y += 42
    canvas.image.save(output, optimize=True)


def render_slab(
    walls: list[Wall2D],
    slab_name: str,
    slab: list[tuple[float, float]],
    output: Path,
    source_name: str,
) -> None:
    canvas = Canvas(bounds_of(slab))
    draw_header(
        canvas,
        "Door Ground Floor — teto/cobertura",
        f"{slab_name} • arestas SLAB-E-### • origem: {source_name}",
    )
    slab_pixels = [canvas.xy(point) for point in slab]
    canvas.draw.polygon(
        slab_pixels,
        fill="#dbeafe",
        outline="#2563eb",
        width=7,
    )
    for wall in walls:
        canvas.draw.line(
            [canvas.xy(wall.start), canvas.xy(wall.end)],
            fill="#94a3b8",
            width=2,
        )

    area = polygon_area(slab)
    perimeter = sum(
        math.dist(start, slab[(index + 1) % len(slab)])
        for index, start in enumerate(slab)
    )
    centroid = (
        sum(point[0] for point in slab) / len(slab),
        sum(point[1] for point in slab) / len(slab),
    )
    edge_rows = []
    edge_font = font(19, bold=True)
    vertex_font = font(17, bold=True)
    for index, start in enumerate(slab, start=1):
        end = slab[index % len(slab)]
        code = f"SLAB-E-{index:03d}"
        length = math.dist(start, end)
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        radial = (midpoint[0] - centroid[0], midpoint[1] - centroid[1])
        norm = max(math.hypot(*radial), 1e-9)
        normal = (radial[0] / norm, radial[1] / norm)
        label_distance = 1.40 if length < 1.5 else 0.90
        label_world = (
            midpoint[0] + normal[0] * label_distance,
            midpoint[1] + normal[1] * label_distance,
        )
        text_box(
            canvas.draw,
            canvas.xy(label_world),
            code,
            edge_font,
            fill="#1d4ed8",
            background="#eff6ff",
        )
        edge_rows.append((code, length, start, end))
    for index, vertex in enumerate(slab, start=1):
        pixel = canvas.xy(vertex)
        canvas.draw.ellipse(
            (pixel[0] - 8, pixel[1] - 8, pixel[0] + 8, pixel[1] + 8),
            fill="#1d4ed8",
            outline="#ffffff",
            width=3,
        )
        radial = (vertex[0] - centroid[0], vertex[1] - centroid[1])
        radial_norm = max(math.hypot(*radial), 1e-9)
        label_offset = (
            48 * radial[0] / radial_norm,
            -48 * radial[1] / radial_norm,
        )
        text_box(
            canvas.draw,
            (pixel[0] + label_offset[0], pixel[1] + label_offset[1]),
            f"V-{index:03d}",
            vertex_font,
            fill="#1e3a8a",
            background="#ffffff",
        )
    legend_x = canvas.drawing_width + 35
    canvas.draw.rounded_rectangle(
        (legend_x, 185, canvas.width - 55, canvas.height - 65),
        radius=18,
        fill="#ffffff",
        outline="#d7dde5",
        width=3,
    )
    canvas.draw.text(
        (legend_x + 30, 220),
        "ARESTAS DO SLAB",
        font=font(27, bold=True),
        fill="#111827",
    )
    canvas.draw.text(
        (legend_x + 30, 267),
        "Código                 comprimento",
        font=font(19, bold=True),
        fill="#64748b",
    )
    canvas.draw.text(
        (legend_x + 30, 310),
        f"Área: {area:.2f} m²   Perímetro: {perimeter:.2f} m",
        font=font(20, bold=True),
        fill="#1d4ed8",
    )
    y = 365
    for code, length, _, _ in edge_rows:
        canvas.draw.text(
            (legend_x + 30, y),
            f"{code}          {length:6.2f} m",
            font=font(22),
            fill="#1f2937",
        )
        y += 48
    draw_north_and_scale(
        canvas,
        x=legend_x + 35,
        y=canvas.height - 130,
        scale_x=legend_x + 95,
    )
    canvas.image.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ifc", type=Path)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    entities = parse_step(args.ifc)
    walls = extract_walls(entities)
    slab_name, slab = extract_roof_slab(entities)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_name = args.source_name or args.ifc.name
    walls_png = args.output_dir / "doorgrondfloor_walls_annotated.png"
    slab_png = args.output_dir / "doorgrondfloor_ceiling_slab_edges_annotated.png"
    labels_json = args.output_dir / "doorgrondfloor_geometry_labels.json"

    render_walls(walls, slab, walls_png, source_name)
    render_slab(walls, slab_name, slab, slab_png, source_name)
    labels_json.write_text(
        json.dumps(
            {
                "source": source_name,
                "derived_from_ifc": str(args.ifc),
                "coordinate_unit": "m",
                "walls": [
                    {
                        "code": wall.code,
                        "source_name": wall.name,
                        "start": wall.start,
                        "end": wall.end,
                        "length": wall.length,
                        "thickness": wall.thickness,
                    }
                    for wall in walls
                ],
                "ceiling_slab": {
                    "name": slab_name,
                    "area": polygon_area(slab),
                    "perimeter": sum(
                        math.dist(vertex, slab[index % len(slab)])
                        for index, vertex in enumerate(slab, start=1)
                    ),
                    "vertices": [
                        {"code": f"V-{index:03d}", "point": vertex}
                        for index, vertex in enumerate(slab, start=1)
                    ],
                    "edges": [
                        {
                            "code": f"SLAB-E-{index:03d}",
                            "start_vertex": f"V-{index:03d}",
                            "end_vertex": f"V-{index % len(slab) + 1:03d}",
                            "length": math.dist(vertex, slab[index % len(slab)]),
                        }
                        for index, vertex in enumerate(slab, start=1)
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "walls_png": str(walls_png),
                "slab_png": str(slab_png),
                "labels_json": str(labels_json),
                "wall_count": len(walls),
                "slab_edge_count": len(slab),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
