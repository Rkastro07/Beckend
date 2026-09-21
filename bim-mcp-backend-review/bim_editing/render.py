"""Layered PNG renderer for overview and element-edit views."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .geometry import wall_axis, wall_axis_segments, wall_frame, wall_length


COLORS = {
    "background": "#f8fafc",
    "grid": "#e2e8f0",
    "wall": "#155e75",
    "wall_selected": "#ea580c",
    "wall_muted": "#94a3b8",
    "label": "#0f172a",
    "door": "#16a34a",
    "window": "#0284c7",
    "p1": "#16a34a",
    "p2": "#7c3aed",
    "slab": "#64748b",
    "space_fill": "#dbeafe",
    "space_edge": "#60a5fa",
}


def _font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _compact(identifier: str) -> str:
    return str(identifier).replace("W-S01-", "W-")


def _selection(values: Iterable[str]) -> tuple[set[str], dict[str, set[str]], set[str]]:
    selected_walls: set[str] = set()
    selected_parts: dict[str, set[str]] = {}
    selected_openings: set[str] = set()
    for raw in values:
        value = str(raw)
        if "." in value:
            element, part = value.rsplit(".", 1)
            if part.upper() in {"P1", "P2", "AXIS"}:
                selected_walls.add(element)
                selected_parts.setdefault(element, set()).add(part.upper())
                continue
        if value.startswith(("D-", "J-", "O-")):
            selected_openings.add(value)
        else:
            selected_walls.add(value)
            selected_parts.setdefault(value, set()).update(("P1", "P2"))
    return selected_walls, selected_parts, selected_openings


def render_model(
    model: dict,
    output: str | Path,
    *,
    mode: str = "overview",
    selected: Iterable[str] = (),
    width: int = 2600,
    height: int = 1800,
) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), COLORS["background"])
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _font(36, bold=True)
    label_font = _font(22, bold=True)
    small_font = _font(18)
    endpoint_font = _font(20, bold=True)

    selected_walls, selected_parts, selected_openings = _selection(selected)
    walls = model.get("paredes", [])
    bbox = model.get("bbox", {})
    xmin = float(bbox.get("xmin", 0.0))
    ymin = float(bbox.get("ymin", 0.0))
    xmax = float(bbox.get("xmax", 1.0))
    ymax = float(bbox.get("ymax", 1.0))
    span_x, span_y = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)
    plot_left, plot_top, plot_right, plot_bottom = 120, 135, width - 90, height - 120
    scale_px = min(
        (plot_right - plot_left) / span_x,
        (plot_bottom - plot_top) / span_y,
    )
    pad_x = ((plot_right - plot_left) - span_x * scale_px) * 0.5
    pad_y = ((plot_bottom - plot_top) - span_y * scale_px) * 0.5

    def screen(value):
        x, y = value
        return (
            plot_left + pad_x + (x - xmin) * scale_px,
            plot_bottom - pad_y - (y - ymin) * scale_px,
        )

    # Sparse metric grid.
    grid_step = 1.0
    if max(span_x, span_y) > 40:
        grid_step = 5.0
    elif max(span_x, span_y) > 20:
        grid_step = 2.0
    gx = math.floor(xmin / grid_step) * grid_step
    while gx <= xmax + grid_step:
        x, _ = screen((gx, ymin))
        draw.line((x, plot_top, x, plot_bottom), fill=COLORS["grid"] + "88", width=1)
        gx += grid_step
    gy = math.floor(ymin / grid_step) * grid_step
    while gy <= ymax + grid_step:
        _, y = screen((xmin, gy))
        draw.line((plot_left, y, plot_right, y), fill=COLORS["grid"] + "88", width=1)
        gy += grid_step

    # Spaces stay visually quiet.
    for space in model.get("spaces", []):
        vertices = [screen(value) for value in space.get("contorno", [])]
        if len(vertices) >= 3:
            draw.polygon(vertices, fill=COLORS["space_fill"] + "55")
            draw.line(vertices + [vertices[0]], fill=COLORS["space_edge"] + "AA", width=2)
            cx = sum(value[0] for value in vertices) / len(vertices)
            cy = sum(value[1] for value in vertices) / len(vertices)
            text = f"{space['id']}  {float(space.get('area', 0.0)):.2f} m²"
            bounds = draw.textbbox((0, 0), text, font=small_font)
            draw.rounded_rectangle(
                (
                    cx - (bounds[2] - bounds[0]) / 2 - 7,
                    cy - 15,
                    cx + (bounds[2] - bounds[0]) / 2 + 7,
                    cy + 15,
                ),
                radius=6,
                fill="#ffffffcc",
                outline=COLORS["space_edge"] + "99",
            )
            draw.text((cx, cy), text, font=small_font, fill="#1d4ed8", anchor="mm")

    slab = [screen(value) for value in model.get("laje", {}).get("contorno", [])]
    if len(slab) >= 3:
        draw.line(slab + [slab[0]], fill=COLORS["slab"] + "99", width=3)

    label_boxes: list[tuple[float, float, float, float]] = []
    wall_label_annotations: list[
        tuple[str, tuple[float, float], tuple[float, float], tuple[float, float, float, float]]
    ] = []
    endpoint_annotations: list[tuple[str, str, tuple[float, float], str]] = []

    def overlaps(candidate):
        return any(
            not (
                candidate[2] < occupied[0]
                or candidate[0] > occupied[2]
                or candidate[3] < occupied[1]
                or candidate[1] > occupied[3]
            )
            for occupied in label_boxes
        )

    wall_by_id = {wall["id"]: wall for wall in walls}
    for wall in walls:
        identifier = wall["id"]
        axis = wall_axis(wall)
        a, b = screen(axis[0]), screen(axis[1])
        chosen = identifier in selected_walls
        color = (
            COLORS["wall_selected"]
            if chosen
            else COLORS["wall"] if mode == "overview" else COLORS["wall_muted"]
        )
        thickness_px = max(4, int(round(float(wall["espessura"]) * scale_px)))
        axis_segments = wall_axis_segments(wall)
        wall_pixels = [screen(axis_segments[0][0])] + [
            screen(segment[1]) for segment in axis_segments
        ]
        draw.line(
            wall_pixels,
            fill=color + ("ff" if chosen else "dd"),
            width=thickness_px,
            joint="curve",
        )

        text = _compact(identifier)
        if chosen and mode == "edit":
            text += f"  {float(wall.get('comprimento', 0.0)):.2f} m"
        box = draw.textbbox((0, 0), text, font=label_font)
        tw, th = box[2] - box[0], box[3] - box[1]
        midpoint = screen(wall_frame(wall, wall_length(wall) / 2)[0])
        vector = (b[0] - a[0], b[1] - a[1])
        vector_length = max(1.0, math.hypot(*vector))
        normal = (-vector[1] / vector_length, vector[0] / vector_length)
        candidates = []
        for offset in (24, -24, 48, -48, 72, -72):
            cx = midpoint[0] + normal[0] * offset
            cy = midpoint[1] + normal[1] * offset
            candidates.append((cx, cy))
        cx, cy = candidates[-1]
        candidate_box = (cx - tw / 2 - 8, cy - th / 2 - 6, cx + tw / 2 + 8, cy + th / 2 + 6)
        for proposed_x, proposed_y in candidates:
            proposed = (
                proposed_x - tw / 2 - 8,
                proposed_y - th / 2 - 6,
                proposed_x + tw / 2 + 8,
                proposed_y + th / 2 + 6,
            )
            if not overlaps(proposed):
                cx, cy, candidate_box = proposed_x, proposed_y, proposed
                break
        label_boxes.append(candidate_box)
        wall_label_annotations.append(
            (text, midpoint, (cx, cy), candidate_box)
        )

        if chosen and mode == "edit":
            requested = selected_parts.get(identifier, {"P1", "P2"})
            for part, world, pixel, part_color in (
                ("P1", axis[0], a, COLORS["p1"]),
                ("P2", axis[1], b, COLORS["p2"]),
            ):
                if part not in requested:
                    continue
                endpoint_annotations.append((identifier, part, pixel, part_color))

    # Openings are visible as geometry, but their IDs only appear when selected
    # or hosted by a selected wall in edit mode.
    for opening in model.get("aberturas", []):
        wall = wall_by_id.get(opening["parede_id"])
        if wall is None:
            continue
        center_distance = float(opening["s_centro"])
        half_width = float(opening["largura"]) * 0.5
        start = max(0.0, center_distance - half_width)
        end = min(wall_length(wall), center_distance + half_width)
        count = max(1, min(128, int(math.ceil((end - start) / 0.06))))
        opening_pixels = [
            screen(wall_frame(wall, start + (end - start) * index / count)[0])
            for index in range(count + 1)
        ]
        centre = wall_frame(wall, center_distance)[0]
        color = COLORS["door"] if opening.get("tipo") == "door" else COLORS["window"]
        draw.line(opening_pixels, fill=color + "ff",
                  width=max(5, int(wall["espessura"] * scale_px * 1.2)), joint="curve")
        if opening["id"] in selected_openings:
            cx, cy = screen(centre)
            draw.text(
                (cx, cy - 17),
                opening["id"],
                font=small_font,
                fill=color,
                anchor="ms",
                stroke_width=3,
                stroke_fill="#ffffff",
            )

    # Wall IDs are a review/control layer, so they must stay above wall and
    # opening geometry. A strong neutral outline avoids door/window colours
    # making the number illegible.
    for text, midpoint, centre, candidate_box in wall_label_annotations:
        cx, cy = centre
        draw.line((*midpoint, cx, cy), fill="#334155bb", width=3)
        draw.rounded_rectangle(
            candidate_box,
            radius=8,
            fill="#fffffff8",
            outline="#0f172add",
            width=3,
        )
        draw.text(
            (cx, cy),
            text,
            font=label_font,
            fill="#020617",
            anchor="mm",
            stroke_width=1,
            stroke_fill="#ffffff",
        )

    # Endpoint markers are always the topmost layer in edit mode.  This avoids
    # doors/windows hiding the exact point the client is being asked to name.
    for identifier, part, pixel, part_color in endpoint_annotations:
        radius = 11
        draw.ellipse(
            (
                pixel[0] - radius,
                pixel[1] - radius,
                pixel[0] + radius,
                pixel[1] + radius,
            ),
            fill="#ffffff",
            outline=part_color,
            width=5,
        )
        endpoint_text = f"{_compact(identifier)}.{part}"
        eb = draw.textbbox((0, 0), endpoint_text, font=endpoint_font)
        ew, eh = eb[2] - eb[0], eb[3] - eb[1]
        to_left = pixel[0] + ew + 35 > width
        ex = pixel[0] - 16 if to_left else pixel[0] + 16
        ey = pixel[1] - 19 if part == "P1" else pixel[1] + 19
        ey = min(height - 80, max(125, ey))
        if to_left:
            rect = (ex - ew - 5, ey - eh / 2 - 5, ex + 5, ey + eh / 2 + 5)
            anchor = "rm"
        else:
            rect = (ex - 5, ey - eh / 2 - 5, ex + ew + 5, ey + eh / 2 + 5)
            anchor = "lm"
        draw.rounded_rectangle(
            rect,
            radius=6,
            fill="#ffffffee",
            outline=part_color,
            width=2,
        )
        draw.text(
            (ex, ey),
            endpoint_text,
            font=endpoint_font,
            fill=part_color,
            anchor=anchor,
        )

    title = (
        f"Revisão {model.get('revision', 'R00')} — "
        f"{len(walls)} paredes · {len(model.get('aberturas', []))} aberturas · "
        f"{len(model.get('spaces', []))} spaces"
    )
    draw.text((width / 2, 45), title, font=title_font, fill=COLORS["label"], anchor="mm")
    subtitle = (
        "Visão geral: IDs compactos e camadas essenciais"
        if mode == "overview"
        else "Modo de edição: P1/P2 aparecem somente nos elementos selecionados"
    )
    draw.text((width / 2, 91), subtitle, font=small_font, fill="#475569", anchor="mm")

    legend = "Parede  |  Porta  |  Janela  |  Slab  |  P1 verde  |  P2 roxo"
    draw.text((width / 2, height - 45), legend, font=small_font, fill="#475569", anchor="mm")
    image.save(destination, format="PNG", optimize=True)
    return destination


def render_revision_set(
    model: dict,
    output_dir: str | Path,
    *,
    selected: Iterable[str] = (),
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "overview": render_model(
            model,
            directory / "revision_overview.png",
            mode="overview",
        ),
        "edit": render_model(
            model,
            directory / "revision_edit_endpoints.png",
            mode="edit",
            selected=selected,
        ),
    }
