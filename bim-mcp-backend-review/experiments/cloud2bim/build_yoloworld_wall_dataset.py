"""Build the full grouped Cloud-to-BIM wall-token dataset for YOLO-World.

This builder uses the synthetic ``labels.json`` + ``ifc_ref.json`` pairs so it
also includes produced, partial and absent doors/windows when the original IFC
file is not available locally.  Splits are made by architectural family, never
by point-cloud variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from build_wall_token_dataset import (
    CLASS_COLORS,
    MetricBox,
    WallFrame,
    build_contact_sheet,
    draw_review,
    project_cloud,
    rasterize_tokens,
    tile_starts,
    yolo_line,
)


CLASSES = ("door", "window")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("synthetic_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-m", type=float, default=0.05)
    parser.add_argument("--tile-width-m", type=float, default=12.8)
    parser.add_argument("--tile-height-m", type=float, default=4.0)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--wall-band-m", type=float, default=0.35)
    parser.add_argument("--review-per-category", type=int, default=4)
    parser.add_argument("--max-models", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def family_key(ifc_name: str) -> str:
    value = Path(ifc_name).stem.lower()
    if "fzk" in value or "haus" in value:
        return "fzk_haus_family"
    if "convenience" in value or "renga" in value:
        return "convenience_store_family"
    if "office_model_cv2" in value:
        return "office_cv2_family"
    if "miniexample" in value:
        return "mini_example_family"
    value = re.sub(r"^\d{6,14}[-_ ]*", "", value)
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def inventory(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for labels_path in sorted(root.glob("*/labels.json")):
        try:
            labels = read_json(labels_path)
        except Exception:
            continue
        counts = Counter(
            obj.get("tipo") for obj in labels.get("objetos", {}).values()
        )
        if counts["IfcWall"] and (counts["IfcDoor"] or counts["IfcWindow"]):
            result[str(labels.get("ifc_file") or labels_path.parent.name)].append(
                labels_path.parent
            )
    return result


def split_families(
    models: dict[str, list[Path]], seed: int
) -> tuple[dict[str, str], dict[str, list[str]]]:
    families: dict[str, list[str]] = defaultdict(list)
    for model in models:
        families[family_key(model)].append(model)
    names = sorted(families)
    random.Random(seed).shuffle(names)
    n_test = max(1, round(len(names) * 0.15))
    n_val = max(1, round(len(names) * 0.15))
    family_split = {}
    for index, name in enumerate(names):
        if index < n_test:
            family_split[name] = "test"
        elif index < n_test + n_val:
            family_split[name] = "val"
        else:
            family_split[name] = "train"
    model_split = {
        model: family_split[family_key(model)]
        for model in models
    }
    return model_split, dict(families)


def bbox_corners(bbox: dict) -> np.ndarray:
    return np.array(
        [
            [x, y, z]
            for x in (float(bbox["xmin"]), float(bbox["xmax"]))
            for y in (float(bbox["ymin"]), float(bbox["ymax"]))
            for z in (float(bbox["zmin"]), float(bbox["zmax"]))
        ],
        dtype=np.float64,
    )


def frame_from_reference(reference: dict) -> WallFrame:
    bbox = reference["bbox"]
    corners = bbox_corners(bbox)
    center = np.mean(corners[:, :2], axis=0)
    extent = np.ptp(corners[:, :2], axis=0)

    candidates = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    matrix = np.asarray(reference.get("matrix_local") or [], dtype=float)
    if matrix.shape == (4, 4):
        for column in (matrix[:2, 0], matrix[:2, 1]):
            norm = float(np.linalg.norm(column))
            if norm > 1e-7:
                candidates.append(column / norm)

    best = None
    for tangent in candidates:
        normal = np.array([-tangent[1], tangent[0]])
        projected_length = float(np.ptp(corners[:, :2] @ tangent))
        projected_thickness = float(np.ptp(corners[:, :2] @ normal))
        aspect = projected_length / max(projected_thickness, 1e-6)
        alignment = abs(float(tangent @ np.eye(2)[int(np.argmax(extent))]))
        score = aspect + 0.05 * alignment
        if best is None or score > best[0]:
            best = (score, tangent, normal, projected_length, projected_thickness)
    assert best is not None
    _, tangent, normal, length, thickness = best
    if (
        abs(float(tangent[0])) >= abs(float(tangent[1]))
        and float(tangent[0]) < 0.0
    ) or (
        abs(float(tangent[1])) > abs(float(tangent[0]))
        and float(tangent[1]) < 0.0
    ):
        tangent *= -1.0
        normal *= -1.0
    start = center - tangent * length / 2.0
    return WallFrame(
        center=center,
        start=start,
        tangent=tangent,
        normal=normal,
        length=length,
        thickness=thickness,
        z_min=float(bbox["zmin"]),
        z_max=float(bbox["zmax"]),
    )


def object_box(
    guid: str,
    obj: dict,
    reference: dict,
    frame: WallFrame,
) -> MetricBox:
    corners = bbox_corners(reference["bbox"])
    along = (corners[:, :2] - frame.start) @ frame.tangent
    return MetricBox(
        class_name="door" if obj.get("tipo") == "IfcDoor" else "window",
        guid=guid,
        name=str(obj.get("nome") or guid),
        s_min=float(np.min(along)),
        s_max=float(np.max(along)),
        z_min=float(np.min(corners[:, 2]) - frame.z_min),
        z_max=float(np.max(corners[:, 2]) - frame.z_min),
        source_status=str(obj.get("status") or "GABARITO"),
    )


def assign_objects(
    labels: dict,
    references: dict,
    walls: dict[str, tuple[dict, WallFrame]],
) -> tuple[dict[str, list[MetricBox]], list[str]]:
    assigned: dict[str, list[MetricBox]] = defaultdict(list)
    warnings = []
    for guid, obj in labels.get("objetos", {}).items():
        if obj.get("tipo") not in ("IfcDoor", "IfcWindow") or guid not in references:
            continue
        corners = bbox_corners(references[guid]["bbox"])
        center = np.mean(corners[:, :2], axis=0)
        z0, z1 = float(np.min(corners[:, 2])), float(np.max(corners[:, 2]))
        choices = []
        for wall_guid, (_, frame) in walls.items():
            along = float((center - frame.start) @ frame.tangent)
            normal_center = abs(float((center - frame.center) @ frame.normal))
            normal_half = float(np.ptp(corners[:, :2] @ frame.normal)) / 2.0
            vertical_overlap = min(z1, frame.z_max) - max(z0, frame.z_min)
            if vertical_overlap <= 0.05:
                continue
            if not -0.35 <= along <= frame.length + 0.35:
                continue
            allowed = frame.thickness / 2.0 + normal_half + 0.22
            if normal_center > allowed:
                continue
            end_penalty = 0.0
            if along < 0.0:
                end_penalty = abs(along)
            elif along > frame.length:
                end_penalty = along - frame.length
            choices.append((normal_center + 2.0 * end_penalty, wall_guid))
        if not choices:
            warnings.append(f"unhosted {obj.get('tipo')} {guid} {obj.get('nome')}")
            continue
        _, wall_guid = min(choices)
        assigned[wall_guid].append(
            object_box(guid, obj, references[guid], walls[wall_guid][1])
        )
    return assigned, warnings


def crop_wall_points(
    points: np.ndarray,
    bbox: dict,
    frame: WallFrame,
    band_m: float,
) -> tuple[np.ndarray, ...]:
    padding = max(0.25, band_m)
    keep = (
        (points[:, 0] >= float(bbox["xmin"]) - padding)
        & (points[:, 0] <= float(bbox["xmax"]) + padding)
        & (points[:, 1] >= float(bbox["ymin"]) - padding)
        & (points[:, 1] <= float(bbox["ymax"]) + padding)
        & (points[:, 2] >= frame.z_min - 0.15)
        & (points[:, 2] <= frame.z_min + 4.20)
    )
    return project_cloud(points[keep], frame, band_m)


def clipped_bounds(box: MetricBox, start: float, width: float, height: float):
    bounds = (
        max(0.0, box.s_min - start),
        min(width, box.s_max - start),
        max(0.0, box.z_min),
        min(height, box.z_max),
    )
    if bounds[1] - bounds[0] < 0.08 or bounds[3] - bounds[2] < 0.08:
        return None
    return bounds


def safe_stem(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:90]}_{digest}"


def review_categories(visible) -> list[str]:
    if not visible:
        return ["negative"]
    return sorted(
        {
            f"{box.class_name}_{box.source_status.lower()}"
            for box, _ in visible
        }
    )


def process_variant(
    args,
    sample_dir: Path,
    split: str,
    manifest_handle,
    counters: Counter,
    review_counts: Counter,
    review_paths: dict[str, list[Path]],
) -> list[str]:
    labels = read_json(sample_dir / "labels.json")
    references = read_json(sample_dir / "ifc_ref.json")
    cloud = o3d.io.read_point_cloud(str(sample_dir / "cena.ply"))
    points = np.asarray(cloud.points, dtype=np.float64)
    if not points.size:
        return [f"empty cloud {sample_dir}"]

    all_walls: dict[str, tuple[dict, WallFrame]] = {}
    walls: dict[str, tuple[dict, WallFrame]] = {}
    for guid, obj in labels.get("objetos", {}).items():
        if obj.get("tipo") != "IfcWall":
            continue
        if guid not in references:
            continue
        try:
            frame = frame_from_reference(references[guid])
        except Exception:
            continue
        if frame.length >= 0.30 and frame.z_max - frame.z_min >= 0.30:
            all_walls[guid] = (obj, frame)
            if obj.get("status") == "COMPLETO":
                walls[guid] = (obj, frame)
    assigned_all, warnings = assign_objects(labels, references, all_walls)
    assigned = {
        wall_guid: boxes
        for wall_guid, boxes in assigned_all.items()
        if wall_guid in walls
    }
    for wall_guid, boxes in assigned_all.items():
        if wall_guid in walls:
            continue
        for box in boxes:
            counters[f"{split}:skipped_noncomplete_host"] += 1
            counters[
                f"{split}:skipped_noncomplete_host:{box.class_name}:{box.source_status}"
            ] += 1

    for wall_guid, (wall_obj, frame) in walls.items():
        along, normal, height = crop_wall_points(
            points, references[wall_guid]["bbox"], frame, args.wall_band_m
        )
        if along.size < 20:
            continue
        boxes = assigned.get(wall_guid, [])
        for tile_index, start in enumerate(
            tile_starts(frame.length, args.tile_width_m, args.overlap)
        ):
            visible = []
            for box in boxes:
                bounds = clipped_bounds(
                    box, start, args.tile_width_m, args.tile_height_m
                )
                if bounds is not None:
                    visible.append((box, bounds))
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
            if stats["occupied_tokens"] < 5:
                continue
            image = cv2.resize(token_image, (1280, 640), interpolation=cv2.INTER_NEAREST)
            identity = f"{sample_dir.name}|{wall_guid}|{tile_index}"
            stem = safe_stem(identity)
            image_path = args.output_dir / "images" / split / f"{stem}.png"
            label_path = args.output_dir / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(image_path), image)
            label_path.write_text(
                "\n".join(
                    yolo_line(box.class_name, bounds, args.tile_width_m, args.tile_height_m)
                    for box, bounds in visible
                ) + ("\n" if visible else ""),
                encoding="utf-8",
            )
            for box, _ in visible:
                counters[f"{split}:{box.class_name}"] += 1
                counters[f"{split}:{box.class_name}:{box.source_status}"] += 1
            counters[f"{split}:images"] += 1
            counters[f"{split}:negative"] += int(not visible)

            entry = {
                "id": stem,
                "split": split,
                "family": family_key(str(labels.get("ifc_file") or "")),
                "ifc_file": labels.get("ifc_file"),
                "variant": sample_dir.name,
                "wall": {
                    "guid": wall_guid,
                    "name": wall_obj.get("nome"),
                    "start_xy": frame.start.tolist(),
                    "tangent_xy": frame.tangent.tolist(),
                    "normal_xy": frame.normal.tolist(),
                    "z_min_m": frame.z_min,
                    "length_m": frame.length,
                    "thickness_m": frame.thickness,
                },
                "tile": {
                    "start_m": start,
                    "token_m": args.token_m,
                    "width_m": args.tile_width_m,
                    "height_m": args.tile_height_m,
                    "render_size": [1280, 640],
                },
                "image": str(image_path.resolve()),
                "label": str(label_path.resolve()),
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
                "stats": stats,
            }
            manifest_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

            categories = review_categories(visible)
            selected = any(
                review_counts[f"{split}:{category}"] < args.review_per_category
                for category in categories
            )
            if selected:
                for category in categories:
                    review_counts[f"{split}:{category}"] += 1
                title = (
                    f"{split} | {sample_dir.name} | {wall_obj.get('nome', wall_guid)} | "
                    f"s={start:.2f}m"
                )
                review = draw_review(
                    token_image,
                    visible,
                    render_width=1280,
                    render_height=640,
                    tile_width_m=args.tile_width_m,
                    tile_height_m=args.tile_height_m,
                    title=title,
                )
                review_path = args.output_dir / "review" / split / f"{stem}.png"
                cv2.imwrite(str(review_path), review)
                review_paths[split].append(review_path)
    return warnings


def write_dataset_yaml(output_dir: Path) -> None:
    yaml = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: door\n"
        "  1: window\n"
    )
    (output_dir / "dataset.yaml").write_text(yaml, encoding="utf-8")


def main() -> None:
    args = parse_args()
    models = inventory(args.synthetic_root)
    if args.max_models:
        models = dict(list(sorted(models.items()))[:args.max_models])
    model_split, families = split_families(models, args.seed)
    for split in ("train", "val", "test"):
        for group in ("images", "labels", "review"):
            (args.output_dir / group / split).mkdir(parents=True, exist_ok=True)

    split_info = {
        "schema": "cloud2bim.wall-token-splits.v1",
        "seed": args.seed,
        "rule": "architectural family grouped; variants never cross splits",
        "families": families,
        "models": {
            model: {
                "family": family_key(model),
                "split": model_split[model],
                "variants": [path.name for path in paths],
            }
            for model, paths in sorted(models.items())
        },
    }
    (args.output_dir / "splits.json").write_text(
        json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    counters: Counter = Counter()
    review_counts: Counter = Counter()
    review_paths: dict[str, list[Path]] = defaultdict(list)
    warnings = []
    manifest_path = args.output_dir / "manifest.jsonl"
    ordered = [
        (model, sample)
        for model, paths in sorted(models.items())
        for sample in sorted(paths)
    ]
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, (model, sample_dir) in enumerate(ordered, start=1):
            warnings.extend(
                process_variant(
                    args,
                    sample_dir,
                    model_split[model],
                    manifest,
                    counters,
                    review_counts,
                    review_paths,
                )
            )
            if index % 10 == 0 or index == len(ordered):
                print(f"processed {index}/{len(ordered)} variants", flush=True)

    for split in ("train", "val", "test"):
        build_contact_sheet(
            review_paths[split],
            args.output_dir / f"review_contact_sheet_{split}.png",
        )
    write_dataset_yaml(args.output_dir)
    summary = {
        "schema": "cloud2bim.wall-token-dataset-summary.v1",
        "target_model": "YOLO-World-M 1280",
        "classes": list(CLASSES),
        "token_m": args.token_m,
        "tile_m": [args.tile_width_m, args.tile_height_m],
        "render_size": [1280, 640],
        "base_ifcs": len(models),
        "families": len(families),
        "variants": len(ordered),
        "counts": dict(sorted(counters.items())),
        "review_counts": dict(sorted(review_counts.items())),
        "unhosted_warnings": len(warnings),
        "warning_examples": warnings[:50],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
