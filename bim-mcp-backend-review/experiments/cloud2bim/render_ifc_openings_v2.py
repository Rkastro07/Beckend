"""Render a top-down audit image from the authored IFC, not detector JSON.

The preview is intentionally independent from the proposal file.  Wall axes,
opening-fill geometry, candidate identifiers, and audit metadata are all read
back from the IFC so the image proves what was actually materialised.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as Element
import ifcopenshell.util.placement as Placement
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull, QhullError


COLORS = {
    "IfcDoor": "#e52b23",
    "IfcWindow": "#00a84f",
    "topology": "#7236e6",
}


def wall_axis(wall):
    representations = wall.Representation.Representations if wall.Representation else []
    representation = next(
        (
            item
            for item in representations
            if item.RepresentationType == "Curve2D"
        ),
        None,
    )
    if not representation or not representation.Items:
        return None
    curve = representation.Items[0]
    if not hasattr(curve, "Points") or len(curve.Points) < 2:
        return None
    matrix = Placement.get_local_placement(wall.ObjectPlacement)
    points = []
    for point in (curve.Points[0], curve.Points[1]):
        coordinates = point.Coordinates
        local = np.array([coordinates[0], coordinates[1], 0.0, 1.0])
        points.append((matrix @ local)[:2])
    return np.asarray(points)


def candidate_id(product):
    pset = Element.get_pset(product, "Pset_Cloud2BIMOpeningV2") or {}
    return str(pset.get("SourceCandidateId") or product.Name or product.GlobalId)


def product_polygon(product, settings):
    shape = ifcopenshell.geom.create_shape(settings, product)
    vertices = np.asarray(shape.geometry.verts, dtype=float).reshape((-1, 3))
    xy = np.unique(np.round(vertices[:, :2], decimals=6), axis=0)
    if len(xy) < 3:
        return xy
    try:
        return xy[ConvexHull(xy).vertices]
    except QhullError:
        return xy


def load_cloud(path: Path, max_points: int):
    points = np.loadtxt(path, skiprows=1, usecols=(0, 1, 2))
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(42)
    indices = rng.choice(len(points), max_points, replace=False)
    return points[indices]


def font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def text_box(draw, position, text, *, text_font, color, anchor="la"):
    bbox = draw.textbbox(position, text, font=text_font, anchor=anchor, stroke_width=0)
    padding = 9
    rectangle = (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    )
    draw.rounded_rectangle(
        rectangle,
        radius=8,
        fill="white",
        outline=color,
        width=3,
    )
    draw.text(
        position,
        text,
        font=text_font,
        fill=color,
        anchor=anchor,
    )
    return rectangle


def render(ifc_path: Path, xyz_path: Path, output_path: Path, max_points: int):
    model = ifcopenshell.open(str(ifc_path))
    cloud = load_cloud(xyz_path, max_points)

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    walls = []
    for wall in model.by_type("IfcWall"):
        axis = wall_axis(wall)
        if axis is not None:
            walls.append((wall.Name or wall.Tag or wall.GlobalId, axis))

    products = list(model.by_type("IfcDoor")) + list(model.by_type("IfcWindow"))
    authored = []
    for product in products:
        identifier = candidate_id(product)
        polygon = product_polygon(product, settings)
        if len(polygon) == 0:
            raise RuntimeError(f"Geometria vazia no preenchimento {identifier}")
        authored.append(
            {
                "id": identifier,
                "type": product.is_a(),
                "polygon": polygon,
                "center": polygon.mean(axis=0),
                "topology": identifier.startswith("D-GAP-"),
            }
        )

    combined_xy = [cloud[:, :2]]
    combined_xy.extend(axis for _, axis in walls)
    combined_xy.extend(item["polygon"] for item in authored)
    all_xy = np.vstack(combined_xy)
    lower, upper = all_xy.min(axis=0), all_xy.max(axis=0)
    data_padding = max(1.0, .04 * float(np.max(upper - lower)))
    lower -= data_padding
    upper += data_padding

    width, height = 5600, 4400
    margin_x, margin_top, margin_bottom = 240, 300, 260
    scale = min(
        (width - 2 * margin_x) / float(upper[0] - lower[0]),
        (height - margin_top - margin_bottom) / float(upper[1] - lower[1]),
    )

    def pixel(points):
        array = np.asarray(points, dtype=float)
        result = np.empty_like(array)
        result[..., 0] = margin_x + (array[..., 0] - lower[0]) * scale
        result[..., 1] = height - margin_bottom - (array[..., 1] - lower[1]) * scale
        return np.rint(result).astype(int)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    cloud_pixels = pixel(cloud[:, :2])
    draw.point(
        [tuple(value) for value in cloud_pixels],
        fill="#e3eaf1",
    )
    for _, axis in walls:
        draw.line(
            [tuple(value) for value in pixel(axis)],
            fill="#263b55",
            width=7,
            joint="curve",
        )

    labels = []
    for index, item in enumerate(sorted(authored, key=lambda value: value["id"])):
        color = COLORS["topology"] if item["topology"] else COLORS[item["type"]]
        polygon_pixels = pixel(item["polygon"])
        center = pixel(item["center"])
        if len(polygon_pixels) >= 3:
            draw.polygon(
                [tuple(value) for value in polygon_pixels],
                fill=color,
                outline="white",
                width=3,
            )
        else:
            x, y = center
            draw.rectangle(
                (x - 10, y - 10, x + 10, y + 10),
                fill=color,
                outline="white",
                width=3,
            )
        angle = (index % 8) * np.pi / 4.0
        distance = 74 + 26 * (index % 3)
        offset = np.array([np.cos(angle), -np.sin(angle)]) * distance
        labels.append((item, color, center, center + offset))

    label_font = font(29, bold=True)
    for item, color, center, label_position in labels:
        anchor = "la" if label_position[0] >= center[0] else "ra"
        draw.line(
            [tuple(center), tuple(label_position)],
            fill=color,
            width=3,
        )
        text_box(
            draw,
            tuple(label_position),
            item["id"],
            text_font=label_font,
            color=color,
            anchor=anchor,
        )

    door_count = sum(item["type"] == "IfcDoor" for item in authored)
    window_count = sum(item["type"] == "IfcWindow" for item in authored)
    opening_count = len(model.by_type("IfcOpeningElement"))
    title = "IFC Openings V2 — conferência do arquivo materializado"
    subtitle = (
        f"{door_count} IfcDoor  |  {window_count} IfcWindow  |  "
        f"{opening_count} IfcOpeningElement"
    )
    title_font = font(55, bold=True)
    subtitle_font = font(38, bold=True)
    draw.text(
        (width // 2, 40),
        title,
        font=title_font,
        fill="#111827",
        anchor="ma",
    )
    draw.text(
        (width // 2, 115),
        subtitle,
        font=subtitle_font,
        fill="#263b55",
        anchor="ma",
    )

    legend_font = font(29)
    legend_items = [
        ("#263b55", "IfcWall"),
        (COLORS["IfcDoor"], "IfcDoor local"),
        (COLORS["IfcWindow"], "IfcWindow"),
        (COLORS["topology"], "IfcDoor em gap topológico"),
    ]
    legend_x, legend_y = 90, height - 185
    draw.rounded_rectangle(
        (legend_x - 28, legend_y - 30, 1840, height - 45),
        radius=15,
        fill="white",
        outline="#263b55",
        width=3,
    )
    cursor_x = legend_x
    for color, label in legend_items:
        draw.rectangle(
            (cursor_x, legend_y, cursor_x + 30, legend_y + 30),
            fill=color,
        )
        draw.text(
            (cursor_x + 44, legend_y + 15),
            label,
            font=legend_font,
            fill="#111827",
            anchor="lm",
        )
        cursor_x += 88 + int(draw.textlength(label, font=legend_font))

    footer_font = font(25)
    draw.text(
        (width // 2, height - 70),
        "Fonte: geometria e propriedades reabertas do próprio IFC V2",
        font=footer_font,
        fill="#4b5563",
        anchor="mm",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)

    return {
        "output": str(output_path),
        "walls": len(walls),
        "doors": door_count,
        "windows": window_count,
        "openings": opening_count,
        "ids": sorted(item["id"] for item in authored),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ifc", type=Path)
    parser.add_argument("xyz", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-points", type=int, default=450_000)
    arguments = parser.parse_args()
    print(
        render(
            arguments.ifc,
            arguments.xyz,
            arguments.output,
            arguments.max_points,
        )
    )
