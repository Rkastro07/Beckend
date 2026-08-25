"""Build reviewable wall-token histograms from synthetic point clouds.

The generator deliberately stops before training.  It unwraps each IFC wall
into a metric X-Z token grid, projects the matching synthetic cloud into that
grid and writes:

* a clean three-channel image for a detector;
* a YOLO label file generated from IFC hosted openings;
* a human-review image with metric grid and coloured labels;
* per-image JSON preserving the pixel/token-to-world transform.

Channel convention (written as a normal RGB PNG):

* red   = point density;
* green = observed depth/thickness span;
* blue  = support on both wall faces.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import ifcopenshell
import ifcopenshell.geom
import numpy as np
import open3d as o3d


CLASS_NAMES = ("door", "window", "column", "opening")
CLASS_COLORS = {
    "door": (40, 210, 40),       # BGR green
    "window": (230, 150, 35),    # BGR blue
    "column": (40, 180, 245),    # BGR orange
    "opening": (210, 60, 210),   # BGR magenta
}


@dataclass(frozen=True)
class WallFrame:
    center: np.ndarray
    start: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    length: float
    thickness: float
    z_min: float
    z_max: float


@dataclass(frozen=True)
class MetricBox:
    class_name: str
    guid: str
    name: str
    s_min: float
    s_max: float
    z_min: float
    z_max: float
    source_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metric wall-token histograms for visual review."
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        action="append",
        required=True,
        help="Synthetic variant directory containing cena.ply and labels.json.",
    )
    parser.add_argument(
        "--ifc-root",
        type=Path,
        required=True,
        help="Directory searched recursively for the IFC named by labels.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-m", type=float, default=0.05)
    parser.add_argument("--tile-width-m", type=float, default=12.8)
    parser.add_argument("--tile-height-m", type=float, default=4.0)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=640)
    parser.add_argument("--wall-band-m", type=float, default=0.35)
    parser.add_argument("--wall-status", default="COMPLETO")
    parser.add_argument("--include-negative", action="store_true")
    parser.add_argument("--max-walls", type=int, default=0)
    return parser.parse_args()


def shape_vertices(settings: ifcopenshell.geom.settings, element) -> np.ndarray:
    shape = ifcopenshell.geom.create_shape(settings, element)
    vertices = np.asarray(shape.geometry.verts, dtype=np.float64)
    if not vertices.size:
        raise ValueError(f"{element.is_a()} {element.GlobalId} has no vertices")
    return vertices.reshape((-1, 3))


def wall_frame(vertices: np.ndarray) -> WallFrame:
    xy = np.asarray(vertices[:, :2], dtype=np.float32)
    rectangle = cv2.minAreaRect(xy)
    corners = cv2.boxPoints(rectangle).astype(np.float64)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    longest_index = int(np.argmax(lengths))
    tangent = edges[longest_index] / max(float(lengths[longest_index]), 1e-9)
    if (
        abs(float(tangent[0])) >= abs(float(tangent[1]))
        and float(tangent[0]) < 0.0
    ) or (
        abs(float(tangent[1])) > abs(float(tangent[0]))
        and float(tangent[1]) < 0.0
    ):
        tangent *= -1.0
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    center = np.mean(corners, axis=0)
    length = float(np.max(lengths))
    thickness = float(np.min(lengths))
    start = center - tangent * length / 2.0
    return WallFrame(
        center=center,
        start=start,
        tangent=tangent,
        normal=normal,
        length=length,
        thickness=thickness,
        z_min=float(np.min(vertices[:, 2])),
        z_max=float(np.max(vertices[:, 2])),
    )


def project_metric_box(
    vertices: np.ndarray,
    frame: WallFrame,
    *,
    class_name: str,
    guid: str,
    name: str,
    status: str,
) -> MetricBox:
    along = (vertices[:, :2] - frame.start) @ frame.tangent
    return MetricBox(
        class_name=class_name,
        guid=guid,
        name=name,
        s_min=float(np.min(along)),
        s_max=float(np.max(along)),
        z_min=float(np.min(vertices[:, 2]) - frame.z_min),
        z_max=float(np.max(vertices[:, 2]) - frame.z_min),
        source_status=status,
    )


def opening_class(opening) -> tuple[str, object]:
    fillings = tuple(getattr(opening, "HasFillings", ()) or ())
    if fillings:
        filled = fillings[0].RelatedBuildingElement
        if filled.is_a("IfcDoor"):
            return "door", filled
        if filled.is_a("IfcWindow"):
            return "window", filled
        return "opening", filled
    return "opening", opening


def object_status(labels: dict, guid: str) -> str:
    return str((labels.get("objetos", {}).get(guid) or {}).get("status", "GABARITO"))


def hosted_boxes(
    wall,
    frame: WallFrame,
    settings: ifcopenshell.geom.settings,
    labels: dict,
) -> list[MetricBox]:
    result: list[MetricBox] = []
    for relation in tuple(getattr(wall, "HasOpenings", ()) or ()):
        opening = relation.RelatedOpeningElement
        try:
            vertices = shape_vertices(settings, opening)
        except Exception:
            continue
        class_name, filled = opening_class(opening)
        result.append(
            project_metric_box(
                vertices,
                frame,
                class_name=class_name,
                guid=str(getattr(filled, "GlobalId", opening.GlobalId)),
                name=str(getattr(filled, "Name", None) or class_name),
                status=object_status(labels, str(getattr(filled, "GlobalId", ""))),
            )
        )
    return result


def column_boxes(
    columns: Iterable[tuple[object, np.ndarray]],
    frame: WallFrame,
    labels: dict,
    band_m: float,
) -> list[MetricBox]:
    result: list[MetricBox] = []
    for column, vertices in columns:
        normal_values = (vertices[:, :2] - frame.center) @ frame.normal
        along = (vertices[:, :2] - frame.start) @ frame.tangent
        overlaps_wall_band = (
            float(np.min(normal_values)) <= band_m
            and float(np.max(normal_values)) >= -band_m
            and float(np.max(along)) >= 0.0
            and float(np.min(along)) <= frame.length
        )
        if not overlaps_wall_band:
            continue
        result.append(
            project_metric_box(
                vertices,
                frame,
                class_name="column",
                guid=str(column.GlobalId),
                name=str(column.Name or "column"),
                status=object_status(labels, str(column.GlobalId)),
            )
        )
    return result


def tile_starts(length: float, width: float, overlap: float) -> list[float]:
    if length <= width:
        return [0.0]
    stride = width * (1.0 - overlap)
    count = max(2, int(math.ceil((length - width) / stride)) + 1)
    starts = [min(index * stride, length - width) for index in range(count)]
    return sorted(set(round(value, 6) for value in starts))


def project_cloud(points: np.ndarray, frame: WallFrame, band_m: float) -> tuple[np.ndarray, ...]:
    relative_xy = points[:, :2] - frame.start
    along = relative_xy @ frame.tangent
    normal = (points[:, :2] - frame.center) @ frame.normal
    height = points[:, 2] - frame.z_min
    keep = (
        (along >= -0.10)
        & (along <= frame.length + 0.10)
        & (np.abs(normal) <= band_m)
        & (height >= -0.10)
        & (height <= max(4.0, frame.z_max - frame.z_min + 0.20))
    )
    return along[keep], normal[keep], height[keep]


def rasterize_tokens(
    along: np.ndarray,
    normal: np.ndarray,
    height: np.ndarray,
    *,
    tile_start: float,
    frame: WallFrame,
    token_m: float,
    tile_width_m: float,
    tile_height_m: float,
) -> tuple[np.ndarray, dict]:
    width_tokens = int(round(tile_width_m / token_m))
    height_tokens = int(round(tile_height_m / token_m))
    x_index = np.floor((along - tile_start) / token_m).astype(np.int32)
    z_index = np.floor(height / token_m).astype(np.int32)
    valid = (
        (x_index >= 0)
        & (x_index < width_tokens)
        & (z_index >= 0)
        & (z_index < height_tokens)
    )
    x_index = x_index[valid]
    z_index = z_index[valid]
    normal = normal[valid]

    counts = np.zeros((height_tokens, width_tokens), dtype=np.float32)
    np.add.at(counts, (z_index, x_index), 1.0)

    minimum = np.full_like(counts, np.inf)
    maximum = np.full_like(counts, -np.inf)
    if normal.size:
        np.minimum.at(minimum, (z_index, x_index), normal)
        np.maximum.at(maximum, (z_index, x_index), normal)
    span = np.where(counts >= 2.0, maximum - minimum, 0.0)
    span[~np.isfinite(span)] = 0.0

    face_tolerance = max(0.035, min(0.10, frame.thickness * 0.35))
    face_a = np.zeros_like(counts)
    face_b = np.zeros_like(counts)
    if normal.size:
        mask_a = np.abs(normal + frame.thickness / 2.0) <= face_tolerance
        mask_b = np.abs(normal - frame.thickness / 2.0) <= face_tolerance
        np.add.at(face_a, (z_index[mask_a], x_index[mask_a]), 1.0)
        np.add.at(face_b, (z_index[mask_b], x_index[mask_b]), 1.0)
    dual_face = np.minimum(face_a, face_b)

    density_channel = np.clip(np.log1p(counts) / np.log1p(12.0), 0.0, 1.0)
    span_channel = np.clip(span / 0.60, 0.0, 1.0)
    dual_channel = np.clip(np.log1p(dual_face) / np.log1p(4.0), 0.0, 1.0)

    # OpenCV stores BGR: blue=dual face, green=depth, red=density.
    image = np.stack((dual_channel, span_channel, density_channel), axis=-1)
    image = np.uint8(np.round(255.0 * np.flipud(image)))
    stats = {
        "points": int(normal.size),
        "occupied_tokens": int(np.count_nonzero(counts)),
        "dual_face_tokens": int(np.count_nonzero(dual_face)),
        "max_points_per_token": int(np.max(counts)) if counts.size else 0,
    }
    return image, stats


def clipped_box(box: MetricBox, tile_start: float, width: float, height: float):
    x0 = max(0.0, box.s_min - tile_start)
    x1 = min(width, box.s_max - tile_start)
    z0 = max(0.0, box.z_min)
    z1 = min(height, box.z_max)
    if x1 - x0 < 0.08 or z1 - z0 < 0.08:
        return None
    return x0, x1, z0, z1


def yolo_line(class_name: str, bounds, width: float, height: float) -> str:
    x0, x1, z0, z1 = bounds
    center_x = (x0 + x1) / (2.0 * width)
    center_y = 1.0 - (z0 + z1) / (2.0 * height)
    box_width = (x1 - x0) / width
    box_height = (z1 - z0) / height
    return (
        f"{CLASS_NAMES.index(class_name)} {center_x:.8f} {center_y:.8f} "
        f"{box_width:.8f} {box_height:.8f}"
    )


def draw_review(
    image: np.ndarray,
    labels: list[tuple[MetricBox, tuple[float, float, float, float]]],
    *,
    render_width: int,
    render_height: int,
    tile_width_m: float,
    tile_height_m: float,
    title: str,
) -> np.ndarray:
    review = cv2.resize(
        image, (render_width, render_height), interpolation=cv2.INTER_NEAREST
    )
    for meter in np.arange(0.0, tile_width_m + 1e-6, 1.0):
        x = int(round(meter / tile_width_m * render_width))
        cv2.line(review, (x, 0), (x, render_height - 1), (55, 55, 55), 1)
    for meter in np.arange(0.0, tile_height_m + 1e-6, 1.0):
        y = int(round((1.0 - meter / tile_height_m) * render_height))
        cv2.line(review, (0, y), (render_width - 1, y), (55, 55, 55), 1)

    for box, bounds in labels:
        x0, x1, z0, z1 = bounds
        left = int(round(x0 / tile_width_m * render_width))
        right = int(round(x1 / tile_width_m * render_width))
        top = int(round((1.0 - z1 / tile_height_m) * render_height))
        bottom = int(round((1.0 - z0 / tile_height_m) * render_height))
        color = CLASS_COLORS[box.class_name]
        cv2.rectangle(review, (left, top), (right, bottom), color, 3)
        label = f"{box.class_name} {box.name}"
        cv2.putText(
            review,
            label,
            (max(2, left), max(22, top - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(review, (0, 0), (render_width - 1, 35), (15, 15, 15), -1)
    cv2.putText(
        review,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return review


def find_ifc(ifc_root: Path, name: str) -> Path:
    direct = ifc_root / name
    if direct.exists():
        return direct
    matches = list(ifc_root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"IFC not found below {ifc_root}: {name}")
    return matches[0]


def safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def build_contact_sheet(review_paths: list[Path], output: Path) -> None:
    if not review_paths:
        return
    cards = []
    for path in review_paths[:16]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            cards.append(cv2.resize(image, (640, 320), interpolation=cv2.INTER_AREA))
    if not cards:
        return
    blank = np.zeros_like(cards[0])
    while len(cards) % 4:
        cards.append(blank.copy())
    rows = [np.hstack(cards[index:index + 4]) for index in range(0, len(cards), 4)]
    cv2.imwrite(str(output), np.vstack(rows))


def build_review_gallery(items: list[dict], output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    cards = []
    for item in items:
        review_path = Path(item["review"])
        relative = review_path.relative_to(output_dir).as_posix()
        classes = item["classes"]
        kind = "positive" if classes else "negative"
        class_text = ", ".join(classes) if classes else "sem abertura"
        cards.append(
            f'<article class="card {kind}">'
            f'<img loading="lazy" src="{html.escape(relative)}" '
            f'alt="{html.escape(item["id"])}">'
            f'<div><strong>{html.escape(item["sample"])}</strong><br>'
            f'{html.escape(class_text)} · {item["stats"]["points"]} pts</div>'
            f'</article>'
        )
    document = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cloud-to-BIM · revisão dos tokens de parede</title>
<style>
body{margin:0;background:#101114;color:#eee;font:14px system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;padding:16px 22px;background:#181a20ee;
border-bottom:1px solid #333}h1{font-size:20px;margin:0 0 8px}.controls{display:flex;gap:8px}
button{border:1px solid #555;background:#272a32;color:#eee;padding:7px 12px;border-radius:8px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px;padding:12px}
.card{background:#191b20;border:1px solid #30333b;border-radius:10px;overflow:hidden}
.card img{display:block;width:100%;aspect-ratio:2/1;object-fit:contain;background:#000}
.card div{padding:9px 12px;color:#bbb}.card strong{color:#fff}
body.only-positive .negative,body.only-negative .positive{display:none}
</style></head><body><header><h1>Histograma visual por parede · prévia antes do treino</h1>
<div class="controls"><button onclick="document.body.className=''">Todas</button>
<button onclick="document.body.className='only-positive'">Com abertura</button>
<button onclick="document.body.className='only-negative'">Sem abertura</button></div>
</header><main>""" + "\n".join(cards) + "</main></body></html>"
    output = output_dir / "review_gallery.html"
    output.write_text(document, encoding="utf-8")
    return output


def process_sample(args, sample_dir: Path, manifest: dict, settings) -> None:
    labels_path = sample_dir / "labels.json"
    cloud_path = sample_dir / "cena.ply"
    if not labels_path.exists() or not cloud_path.exists():
        raise FileNotFoundError(f"Incomplete synthetic sample: {sample_dir}")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    ifc_path = find_ifc(args.ifc_root, str(labels["ifc_file"]))
    ifc = ifcopenshell.open(str(ifc_path))
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if not points.size:
        raise ValueError(f"Empty point cloud: {cloud_path}")

    column_geometries = []
    for column in ifc.by_type("IfcColumn"):
        try:
            column_geometries.append((column, shape_vertices(settings, column)))
        except Exception:
            continue

    wall_count = 0
    for wall in ifc.by_type("IfcWall"):
        status = object_status(labels, str(wall.GlobalId))
        if args.wall_status and status != args.wall_status:
            continue
        if args.max_walls and wall_count >= args.max_walls:
            break
        try:
            vertices = shape_vertices(settings, wall)
            frame = wall_frame(vertices)
        except Exception as exc:
            manifest["warnings"].append(f"{sample_dir.name}/{wall.GlobalId}: {exc}")
            continue
        if frame.length < 0.30 or frame.z_max - frame.z_min < 0.30:
            continue

        boxes = hosted_boxes(wall, frame, settings, labels)
        boxes.extend(column_boxes(column_geometries, frame, labels, args.wall_band_m))
        if not args.include_negative and not boxes:
            continue
        along, normal, height = project_cloud(points, frame, args.wall_band_m)
        if not along.size:
            continue
        wall_count += 1

        for tile_index, start in enumerate(
            tile_starts(frame.length, args.tile_width_m, args.overlap)
        ):
            visible = []
            for box in boxes:
                bounds = clipped_box(
                    box, start, args.tile_width_m, args.tile_height_m
                )
                if bounds is not None:
                    visible.append((box, bounds))
            if not args.include_negative and not visible:
                continue
            token_image, stats = rasterize_tokens(
                along,
                normal,
                height,
                tile_start=start,
                frame=frame,
                token_m=args.token_m,
                tile_width_m=args.tile_width_m,
                tile_height_m=args.tile_height_m,
            )
            rendered = cv2.resize(
                token_image,
                (args.render_width, args.render_height),
                interpolation=cv2.INTER_NEAREST,
            )
            stem = safe_id(f"{sample_dir.name}__{wall.GlobalId}__t{tile_index:02d}")
            image_path = args.output_dir / "images" / "preview" / f"{stem}.png"
            label_path = args.output_dir / "labels" / "preview" / f"{stem}.txt"
            review_path = args.output_dir / "review" / f"{stem}.png"
            metadata_path = args.output_dir / "metadata" / f"{stem}.json"

            cv2.imwrite(str(image_path), rendered)
            label_path.write_text(
                "\n".join(
                    yolo_line(box.class_name, bounds, args.tile_width_m, args.tile_height_m)
                    for box, bounds in visible
                ) + ("\n" if visible else ""),
                encoding="utf-8",
            )
            title = (
                f"{sample_dir.name} | {wall.Name or wall.GlobalId} | "
                f"s={start:.2f}-{start + args.tile_width_m:.2f}m"
            )
            review = draw_review(
                token_image,
                visible,
                render_width=args.render_width,
                render_height=args.render_height,
                tile_width_m=args.tile_width_m,
                tile_height_m=args.tile_height_m,
                title=title,
            )
            cv2.imwrite(str(review_path), review)

            metadata = {
                "schema": "cloud2bim.wall-token-sample.v1",
                "sample": sample_dir.name,
                "cloud": str(cloud_path.resolve()),
                "ifc": str(ifc_path.resolve()),
                "wall": {
                    "guid": str(wall.GlobalId),
                    "name": str(wall.Name or wall.GlobalId),
                    "status": status,
                    "center_xy": frame.center.tolist(),
                    "start_xy": frame.start.tolist(),
                    "tangent_xy": frame.tangent.tolist(),
                    "normal_xy": frame.normal.tolist(),
                    "length_m": frame.length,
                    "thickness_m": frame.thickness,
                    "z_min_m": frame.z_min,
                    "z_max_m": frame.z_max,
                },
                "tile": {
                    "start_m": start,
                    "width_m": args.tile_width_m,
                    "height_m": args.tile_height_m,
                    "token_m": args.token_m,
                    "token_shape": list(token_image.shape[:2][::-1]),
                    "render_shape": [args.render_width, args.render_height],
                    "pixel_to_world": (
                        "world_xy = wall.start_xy + wall.tangent_xy * "
                        "(tile.start_m + pixel_x/render_width*tile.width_m); "
                        "world_z = wall.z_min_m + (1-pixel_y/render_height)*tile.height_m"
                    ),
                },
                "channels_rgb": {
                    "red": "log point density, fixed clip at 12 points/token",
                    "green": "normal-depth span, fixed clip at 0.60 m",
                    "blue": "minimum support on both wall faces",
                },
                "stats": stats,
                "objects": [
                    {
                        "class": box.class_name,
                        "guid": box.guid,
                        "name": box.name,
                        "status": box.source_status,
                        "metric_box_s_z": list(bounds),
                    }
                    for box, bounds in visible
                ],
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest["images"].append(
                {
                    "id": stem,
                    "sample": sample_dir.name,
                    "wall_guid": str(wall.GlobalId),
                    "image": str(image_path.resolve()),
                    "label": str(label_path.resolve()),
                    "review": str(review_path.resolve()),
                    "metadata": str(metadata_path.resolve()),
                    "classes": [box.class_name for box, _ in visible],
                    "stats": stats,
                }
            )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.overlap < 0.90:
        raise ValueError("--overlap must be between 0 and 0.9")
    if args.token_m <= 0.0:
        raise ValueError("--token-m must be positive")
    for relative in ("images/preview", "labels/preview", "review", "metadata"):
        (args.output_dir / relative).mkdir(parents=True, exist_ok=True)

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    manifest = {
        "schema": "cloud2bim.wall-token-dataset.v1",
        "stage": "visual-review-before-training",
        "classes": list(CLASS_NAMES),
        "config": {
            "token_m": args.token_m,
            "tile_width_m": args.tile_width_m,
            "tile_height_m": args.tile_height_m,
            "overlap": args.overlap,
            "render_size": [args.render_width, args.render_height],
            "wall_status": args.wall_status,
            "include_negative": args.include_negative,
        },
        "sources": [str(path.resolve()) for path in args.sample_dir],
        "images": [],
        "warnings": [],
    }
    for sample_dir in args.sample_dir:
        process_sample(args, sample_dir, manifest, settings)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    labelled_reviews = [
        Path(item["review"])
        for item in manifest["images"]
        if item["classes"]
    ]
    build_contact_sheet(labelled_reviews, args.output_dir / "review_contact_sheet.png")
    gallery_path = build_review_gallery(manifest["images"], args.output_dir)
    summary = {
        "output": str(args.output_dir.resolve()),
        "images": len(manifest["images"]),
        "labels": sum(len(item["classes"]) for item in manifest["images"]),
        "class_counts": {
            class_name: sum(
                item["classes"].count(class_name) for item in manifest["images"]
            )
            for class_name in CLASS_NAMES
        },
        "warnings": len(manifest["warnings"]),
        "contact_sheet": str((args.output_dir / "review_contact_sheet.png").resolve()),
        "gallery": str(gallery_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
