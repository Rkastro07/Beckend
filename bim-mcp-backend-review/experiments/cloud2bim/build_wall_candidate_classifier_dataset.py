"""Build a grouped wall-candidate classification dataset from synthetic BIM scans.

The classifier does not invent geometry.  It receives the metric X-Z token
image for an axis proposed by the geometric wall detector and predicts one of:
wall, door_leaf, or non_wall.  Architectural families remain isolated across
train/validation/test splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from build_wall_token_dataset import rasterize_tokens
from build_yoloworld_wall_dataset import crop_wall_points, frame_from_reference


CLASSES = ("wall", "door_leaf", "non_wall")
NON_WALL_TYPES = ("IfcBeam", "IfcColumn", "IfcSlab", "IfcStair", "IfcRoof")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("synthetic_root", type=Path)
    parser.add_argument("splits_json", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--token-m", type=float, default=0.05)
    parser.add_argument("--tile-width-m", type=float, default=6.4)
    parser.add_argument("--tile-height-m", type=float, default=4.0)
    parser.add_argument("--wall-band-m", type=float, default=0.35)
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=320)
    parser.add_argument("--minimum-partial-fraction", type=float, default=0.30)
    parser.add_argument("--max-tiles-per-object", type=int, default=2)
    parser.add_argument("--views-per-object", type=int, default=2)
    parser.add_argument("--max-leaves-per-variant", type=int, default=8)
    parser.add_argument("--max-other-negatives-per-variant", type=int, default=10)
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_stem(text: str) -> str:
    clean = "".join(char if char.isalnum() or char in "-_" else "_" for char in text)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{clean[:80]}_{digest}"


def candidate_class(obj: dict, minimum_partial: float) -> str | None:
    object_type = obj.get("tipo")
    status = str(obj.get("status") or "AUSENTE")
    fraction = float(obj.get("frac_pts", obj.get("frac_executada", 0.0)) or 0.0)
    present = status == "COMPLETO" or (status == "PARCIAL" and fraction >= minimum_partial)
    if object_type == "IfcWall":
        return "wall" if present else "non_wall"
    if object_type == "IfcDoor":
        return "door_leaf" if present else None
    if object_type in NON_WALL_TYPES:
        return "non_wall" if present else None
    return None


def centered_starts(length: float, width: float, maximum: int) -> list[float]:
    if length <= width:
        return [-(width - length) / 2.0]
    starts = [0.0, max(0.0, length - width)]
    if maximum > 2:
        starts.extend(np.linspace(0.0, length - width, maximum).tolist())
    return sorted(set(round(float(value), 6) for value in starts))[:maximum]


def perturbed_frame(frame, *, identity: str, view: int, class_name: str, seed: int):
    """Simulate the offset/thickness/angle errors of a geometric proposal."""
    rng = random.Random(f"{seed}:{identity}:{view}")
    if view == 0:
        angle = 0.0
        offset = 0.0
        thickness_scale = 1.0
        length_scale = 1.0
    else:
        angle = np.deg2rad(rng.uniform(-3.0, 3.0))
        offset_limit = 0.22 if class_name == "wall" else 0.15
        offset = rng.uniform(-offset_limit, offset_limit)
        thickness_scale = rng.uniform(0.65, 1.45)
        length_scale = rng.uniform(0.85, 1.10)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    tangent = np.array([
        frame.tangent[0] * cosine - frame.tangent[1] * sine,
        frame.tangent[0] * sine + frame.tangent[1] * cosine,
    ])
    normal = np.array([-tangent[1], tangent[0]])
    center = frame.center + frame.normal * offset
    length = max(0.30, frame.length * length_scale)
    return replace(
        frame,
        center=center,
        start=center - tangent * length / 2.0,
        tangent=tangent,
        normal=normal,
        length=length,
        thickness=float(np.clip(frame.thickness * thickness_scale, 0.04, 0.75)),
    ), rng.choice((0.15, 0.20, 0.30, 0.50, 0.75, 1.0))


def select_objects(labels: dict, references: dict, args, variant_name: str) -> list[tuple[str, dict, dict, str]]:
    grouped = {name: [] for name in CLASSES}
    for guid, obj in labels.get("objetos", {}).items():
        if guid not in references:
            continue
        label = candidate_class(obj, args.minimum_partial_fraction)
        if label is not None:
            grouped[label].append((guid, obj, references[guid], label))

    rng = random.Random(f"{args.seed}:{variant_name}")
    rng.shuffle(grouped["door_leaf"])
    grouped["door_leaf"] = grouped["door_leaf"][:args.max_leaves_per_variant]

    wall_absent = [
        item for item in grouped["non_wall"]
        if item[1].get("tipo") == "IfcWall"
    ]
    other = [
        item for item in grouped["non_wall"]
        if item[1].get("tipo") != "IfcWall"
    ]
    rng.shuffle(wall_absent)
    rng.shuffle(other)
    # Missing-wall axes are unusually valuable negatives because nearby BIM
    # objects may still populate their histogram.
    grouped["non_wall"] = (
        wall_absent[:args.max_other_negatives_per_variant]
        + other[:args.max_other_negatives_per_variant]
    )
    return grouped["wall"] + grouped["door_leaf"] + grouped["non_wall"]


def contact_sheet(paths: list[Path], output: Path) -> None:
    cards = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        cards.append(cv2.resize(image, (512, 320), interpolation=cv2.INTER_AREA))
    if not cards:
        return
    blank = np.zeros_like(cards[0])
    while len(cards) % 4:
        cards.append(blank.copy())
    rows = [np.hstack(cards[index:index + 4]) for index in range(0, len(cards), 4)]
    cv2.imwrite(str(output), np.vstack(rows))


def main() -> None:
    args = parse_args()
    args.synthetic_root = args.synthetic_root.resolve()
    args.output = args.output.resolve()
    split_payload = read_json(args.splits_json)
    variants = []
    for model_name, model in split_payload["models"].items():
        for variant in model["variants"]:
            variants.append((variant, model["split"], model["family"], model_name))
    if args.max_variants:
        variants = variants[:args.max_variants]

    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            (args.output / split / class_name).mkdir(parents=True, exist_ok=True)
    review_dir = args.output / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    records = []
    review_paths = []
    warnings = []
    for variant_index, (variant, split, family, model_name) in enumerate(variants, start=1):
        sample = args.synthetic_root / variant
        labels_path = sample / "labels.json"
        references_path = sample / "ifc_ref.json"
        cloud_path = sample / "cena.ply"
        if not (labels_path.exists() and references_path.exists() and cloud_path.exists()):
            warnings.append(f"missing files: {sample}")
            continue
        labels = read_json(labels_path)
        references = read_json(references_path)
        cloud = o3d.io.read_point_cloud(str(cloud_path))
        points = np.asarray(cloud.points, dtype=np.float64)
        if not points.size:
            warnings.append(f"empty cloud: {sample}")
            continue

        selected = select_objects(labels, references, args, variant)
        for guid, obj, reference, class_name in selected:
            try:
                base_frame = frame_from_reference(reference)
            except Exception as exc:
                warnings.append(f"{variant}:{guid}: {exc}")
                continue
            for view in range(args.views_per_object):
                identity_base = f"{variant}:{guid}:{class_name}"
                frame, keep_ratio = perturbed_frame(
                    base_frame,
                    identity=identity_base,
                    view=view,
                    class_name=class_name,
                    seed=args.seed,
                )
                along, normal, height = crop_wall_points(
                    points, reference["bbox"], frame, args.wall_band_m
                )
                if keep_ratio < 1.0 and len(along):
                    thinning = np.random.default_rng(
                        int(hashlib.sha1(f"{identity_base}:{view}".encode()).hexdigest()[:8], 16)
                    ).random(len(along)) < keep_ratio
                    along, normal, height = along[thinning], normal[thinning], height[thinning]
                for tile_index, start in enumerate(centered_starts(
                    frame.length, args.tile_width_m, args.max_tiles_per_object
                )):
                    token_image, stats = rasterize_tokens(
                        along, normal, height,
                        tile_start=start,
                        frame=frame,
                        token_m=args.token_m,
                        tile_width_m=args.tile_width_m,
                        tile_height_m=args.tile_height_m,
                    )
                    image = cv2.resize(
                        token_image,
                        (args.render_width, args.render_height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    identity = f"{identity_base}:{view}:{tile_index}"
                    filename = safe_stem(identity) + ".png"
                    image_path = args.output / split / class_name / filename
                    cv2.imwrite(str(image_path), image)
                    record = {
                        "image": str(image_path.resolve()),
                        "split": split,
                        "class": class_name,
                        "family": family,
                        "model": model_name,
                        "variant": variant,
                        "guid": guid,
                        "object_type": obj.get("tipo"),
                        "object_status": obj.get("status"),
                        "fraction_points": float(obj.get("frac_pts", 0.0) or 0.0),
                        "candidate_length_m": frame.length,
                        "candidate_thickness_m": frame.thickness,
                        "point_keep_ratio": keep_ratio,
                        "perturbed_view": view,
                        "tile_start_m": start,
                        "stats": stats,
                    }
                    records.append(record)
                    counters[f"{split}:{class_name}"] += 1
                    counters[f"{split}:images"] += 1
                    if len(review_paths) < 48 or counters[f"review:{split}:{class_name}"] < 6:
                        review = image.copy()
                        cv2.rectangle(review, (0, 0), (review.shape[1] - 1, 38), (15, 15, 15), -1)
                        cv2.putText(
                            review,
                            f"{split} | {class_name} | {obj.get('tipo')} {obj.get('status')} | keep {keep_ratio:.2f}",
                            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (245, 245, 245), 2, cv2.LINE_AA,
                        )
                        review_path = review_dir / filename
                        cv2.imwrite(str(review_path), review)
                        review_paths.append(review_path)
                        counters[f"review:{split}:{class_name}"] += 1
        print(
            json.dumps({
                "variant": variant_index,
                "total": len(variants),
                "name": variant,
                "images": len(records),
            }),
            flush=True,
        )

    (args.output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": "cloud2bim.wall-candidate-classifier-dataset.v1",
        "classes": list(CLASSES),
        "source_variants": len(variants),
        "token_m": args.token_m,
        "tile_m": [args.tile_width_m, args.tile_height_m],
        "render_size": [args.render_width, args.render_height],
        "split_rule": "architectural family grouped; inherited from opening dataset v1",
        "counts": dict(sorted(counters.items())),
        "warnings": len(warnings),
        "warning_examples": warnings[:30],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    contact_sheet(review_paths[:48], args.output / "review_contact_sheet.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
