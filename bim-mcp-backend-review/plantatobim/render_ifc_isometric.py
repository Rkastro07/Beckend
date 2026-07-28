"""Render isometrico leve de um IFC usando a malha real do IfcOpenShell."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ifcopenshell
import ifcopenshell.geom
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class ElementMesh:
    entity: Any
    code: str
    name: str
    ifc_class: str
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[int, int, int]
    alpha: int
    label: bool


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    paths = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_box(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    value: str,
    *,
    fill: tuple[int, int, int] = (20, 30, 45),
    background: tuple[int, int, int, int] = (255, 255, 255, 235),
    text_font: ImageFont.ImageFont | None = None,
) -> None:
    text_font = text_font or font(17, bold=True)
    bounds = draw.textbbox(position, value, font=text_font, anchor="mm")
    pad_x, pad_y = 7, 5
    draw.rounded_rectangle(
        (
            bounds[0] - pad_x,
            bounds[1] - pad_y,
            bounds[2] + pad_x,
            bounds[3] + pad_y,
        ),
        radius=6,
        fill=background,
        outline=(190, 200, 214, 230),
        width=2,
    )
    draw.text(position, value, font=text_font, fill=fill, anchor="mm")


def element_style(entity: Any) -> tuple[str, tuple[int, int, int], int, bool]:
    ifc_class = entity.is_a()
    name = str(getattr(entity, "Name", "") or "")
    if ifc_class == "IfcWall" and name.lower().startswith("parede-"):
        suffix = name.rsplit("-", 1)[-1]
        return f"W-{suffix}", (202, 132, 55), 255, True
    if ifc_class == "IfcWall" and name.lower() == "fechamento-traseiro":
        return "W-018", (202, 132, 55), 255, True
    if ifc_class == "IfcWall" and name.lower() == "fechamento-frontal":
        return "W-019", (202, 132, 55), 255, True
    if ifc_class == "IfcWall":
        return "LINTEL", (165, 102, 45), 255, False
    if ifc_class == "IfcDoor":
        suffix = name.rsplit("-", 1)[-1]
        return f"D-{suffix}", (36, 161, 112), 255, True
    if ifc_class == "IfcWindow":
        suffix = name.rsplit("-", 1)[-1]
        return f"WIN-{suffix}", (66, 153, 225), 230, True
    if ifc_class == "IfcSlab" and "cobertura" in name.lower():
        return "SLAB-ROOF", (65, 145, 225), 105, True
    if ifc_class == "IfcSlab":
        return "SLAB-FLOOR", (105, 126, 148), 230, False
    return ifc_class, (150, 155, 165), 230, False


def load_meshes(ifc_path: Path, *, roof_explosion: float) -> list[ElementMesh]:
    model = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    meshes = []
    accepted = ("IfcWall", "IfcDoor", "IfcWindow", "IfcSlab")
    for entity in model.by_type("IfcProduct"):
        if entity.is_a() not in accepted:
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, entity)
        except Exception:
            continue
        vertices = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
        faces = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
        code, color, alpha, label = element_style(entity)
        name = str(getattr(entity, "Name", "") or entity.is_a())
        if code == "SLAB-ROOF":
            vertices = vertices.copy()
            vertices[:, 2] += roof_explosion
        meshes.append(
            ElementMesh(
                entity=entity,
                code=code,
                name=name,
                ifc_class=entity.is_a(),
                vertices=vertices,
                faces=faces,
                color=color,
                alpha=alpha,
                label=label,
            )
        )
    if not meshes:
        raise ValueError("Nenhum elemento BIM produziu malha.")
    return meshes


class Projection:
    def __init__(
        self,
        points: np.ndarray,
        *,
        azimuth: float,
        elevation: float,
        viewport: tuple[int, int, int, int],
    ):
        az = math.radians(azimuth)
        el = math.radians(elevation)
        self.view = np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
        )
        self.right = np.cross(self.view, np.array([0.0, 0.0, 1.0]))
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.view)
        self.up /= np.linalg.norm(self.up)
        self.center = (points.min(axis=0) + points.max(axis=0)) / 2
        projected = self.raw(points)
        xmin, ymin = projected[:, :2].min(axis=0)
        xmax, ymax = projected[:, :2].max(axis=0)
        left, top, right, bottom = viewport
        self.scale = min(
            (right - left) / max(xmax - xmin, 1e-9),
            (bottom - top) / max(ymax - ymin, 1e-9),
        )
        self.offset = np.array(
            [
                left + ((right - left) - (xmax - xmin) * self.scale) / 2 - xmin * self.scale,
                top + ((bottom - top) - (ymax - ymin) * self.scale) / 2 + ymax * self.scale,
            ]
        )

    def raw(self, points: np.ndarray) -> np.ndarray:
        centered = points - self.center
        return np.column_stack(
            (
                centered @ self.right,
                centered @ self.up,
                centered @ self.view,
            )
        )

    def project(self, points: np.ndarray) -> np.ndarray:
        raw = self.raw(points)
        return np.column_stack(
            (
                raw[:, 0] * self.scale + self.offset[0],
                -raw[:, 1] * self.scale + self.offset[1],
                raw[:, 2],
            )
        )


def shade(
    color: tuple[int, int, int],
    normal: np.ndarray,
) -> tuple[int, int, int]:
    light = np.array([-0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)
    norm = np.linalg.norm(normal)
    intensity = 0.72
    if norm > 1e-12:
        intensity = 0.55 + 0.45 * abs(float(normal @ light) / norm)
    return tuple(min(255, max(0, round(channel * intensity))) for channel in color)


def draw_ground_grid(
    draw: ImageDraw.ImageDraw,
    projection: Projection,
    points: np.ndarray,
) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    xmin, xmax = math.floor(minimum[0]), math.ceil(maximum[0])
    ymin, ymax = math.floor(minimum[1]), math.ceil(maximum[1])
    z = minimum[2] - 0.02
    for x in range(xmin, xmax + 1):
        line = projection.project(np.array([[x, ymin, z], [x, ymax, z]], dtype=float))
        draw.line(
            [tuple(line[0, :2]), tuple(line[1, :2])],
            fill=(187, 198, 212, 90),
            width=1 if x % 5 else 2,
        )
    for y in range(ymin, ymax + 1):
        line = projection.project(np.array([[xmin, y, z], [xmax, y, z]], dtype=float))
        draw.line(
            [tuple(line[0, :2]), tuple(line[1, :2])],
            fill=(187, 198, 212, 90),
            width=1 if y % 5 else 2,
        )


def render(
    meshes: list[ElementMesh],
    output: Path,
    *,
    source_name: str,
    roof_explosion: float,
    subtitle_note: str | None = None,
    diagnostic_note: str | None = None,
) -> None:
    width, height = 2700, 1850
    image = Image.new("RGBA", (width, height), (246, 248, 251, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    all_points = np.vstack([mesh.vertices for mesh in meshes])
    projection = Projection(
        all_points,
        azimuth=-55,
        elevation=29,
        viewport=(100, 185, 2070, 1710),
    )
    draw_ground_grid(draw, projection, all_points)

    triangles = []
    for mesh in meshes:
        projected = projection.project(mesh.vertices)
        for face in mesh.faces:
            world_triangle = mesh.vertices[face]
            normal = np.cross(
                world_triangle[1] - world_triangle[0],
                world_triangle[2] - world_triangle[0],
            )
            triangle = projected[face]
            triangles.append(
                (
                    float(triangle[:, 2].mean()),
                    triangle[:, :2],
                    shade(mesh.color, normal),
                    mesh.alpha,
                )
            )
    triangles.sort(key=lambda item: item[0])
    for _, triangle, color, alpha in triangles:
        pixels = [tuple(value) for value in triangle]
        draw.polygon(pixels, fill=(*color, alpha))

    label_font = font(17, bold=True)
    for mesh in meshes:
        if not mesh.label:
            continue
        top_center = np.array(
            [
                mesh.vertices[:, 0].mean(),
                mesh.vertices[:, 1].mean(),
                mesh.vertices[:, 2].max() + 0.08,
            ]
        )
        position = projection.project(top_center.reshape(1, 3))[0, :2]
        label_color = (
            (26, 92, 160) if mesh.code == "SLAB-ROOF"
            else (11, 112, 77) if mesh.code.startswith("D-")
            else (139, 70, 14)
        )
        text_box(
            draw,
            (float(position[0]), float(position[1])),
            mesh.code,
            fill=label_color,
            text_font=label_font,
        )

    draw.text(
        (80, 55),
        "Door Ground Floor — BIM gerado pelo app atual",
        font=font(45, bold=True),
        fill=(17, 24, 39, 255),
    )
    draw.text(
        (82, 112),
        f"Fonte: {source_name} • geometria IFC real • "
        f"{subtitle_note or f'teto explodido +{roof_explosion:.2f} m'}",
        font=font(24),
        fill=(70, 85, 105, 255),
    )
    draw.line((80, 150, width - 80, 150), fill=(205, 213, 224, 255), width=3)

    legend_x = 2110
    draw.rounded_rectangle(
        (legend_x, 190, width - 55, height - 65),
        radius=18,
        fill=(255, 255, 255, 245),
        outline=(209, 218, 229, 255),
        width=3,
    )
    draw.text(
        (legend_x + 35, 230),
        "MODELO BIM",
        font=font(29, bold=True),
        fill=(17, 24, 39, 255),
    )
    counts = {
        "Paredes": sum(
            1
            for mesh in meshes
            if mesh.ifc_class == "IfcWall" and mesh.name.lower().startswith("parede-")
        ),
        "Portas": sum(1 for mesh in meshes if mesh.ifc_class == "IfcDoor"),
        "Janelas": sum(1 for mesh in meshes if mesh.ifc_class == "IfcWindow"),
        "Slabs": sum(1 for mesh in meshes if mesh.ifc_class == "IfcSlab"),
    }
    y = 292
    for label, count in counts.items():
        draw.text(
            (legend_x + 35, y),
            f"{label}: {count}",
            font=font(23, bold=True),
            fill=(43, 55, 72, 255),
        )
        y += 48
    draw.line(
        (legend_x + 35, y + 5, width - 90, y + 5),
        fill=(219, 225, 233, 255),
        width=2,
    )
    y += 42
    legend_items = [
        ((202, 132, 55), "Parede / W-###"),
        ((36, 161, 112), "Porta / D-###"),
        ((105, 126, 148), "Laje de piso"),
        ((65, 145, 225), "Laje de cobertura"),
    ]
    for color, label in legend_items:
        draw.rounded_rectangle(
            (legend_x + 35, y, legend_x + 70, y + 35),
            radius=5,
            fill=(*color, 230),
        )
        draw.text(
            (legend_x + 88, y + 17),
            label,
            font=font(21),
            fill=(43, 55, 72, 255),
            anchor="lm",
        )
        y += 54
    draw.rounded_rectangle(
        (legend_x + 30, y + 20, width - 85, y + 190),
        radius=12,
        fill=(239, 246, 255, 255),
        outline=(147, 197, 253, 255),
        width=2,
    )
    note = diagnostic_note or (
        "O deslocamento vertical do teto é apenas visual.\n"
        "O IFC permanece com a cobertura em Z = 2,80 m.\n\n"
        "A faixa diagonal azul é o slab realmente\n"
        "calculado pelo app nesta execução."
    )
    draw.multiline_text(
        (legend_x + 50, y + 43),
        note,
        font=font(19),
        fill=(30, 64, 108, 255),
        spacing=8,
    )
    image.convert("RGB").save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ifc", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--explode-roof", type=float, default=1.20)
    parser.add_argument("--hide-slabs", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meshes = load_meshes(args.ifc, roof_explosion=args.explode_roof)
    subtitle_note = None
    diagnostic_note = None
    if args.hide_slabs:
        meshes = [mesh for mesh in meshes if mesh.ifc_class != "IfcSlab"]
        subtitle_note = "corte de inspeção com os slabs ocultos"
        diagnostic_note = (
            "Piso e cobertura foram ocultados somente nesta vista\n"
            "para permitir a inspeção das paredes e portas.\n\n"
            "O arquivo IFC entregue continua contendo os 2 slabs."
        )
    render(
        meshes,
        args.output,
        source_name=args.source_name or args.ifc.name,
        roof_explosion=args.explode_roof,
        subtitle_note=subtitle_note,
        diagnostic_note=diagnostic_note,
    )
    print(
        f"{args.output} | elements={len(meshes)} | "
        f"triangles={sum(len(mesh.faces) for mesh in meshes)}"
    )


if __name__ == "__main__":
    main()
