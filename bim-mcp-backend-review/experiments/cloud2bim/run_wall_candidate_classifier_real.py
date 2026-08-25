"""Review stitched real-cloud wall candidates with the trained classifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from train_wall_candidate_classifier import WallCandidateNet


COLORS = {
    "wall": (45, 180, 45),
    "door_leaf": (30, 150, 235),
    "non_wall": (45, 45, 220),
    "uncertain": (150, 150, 150),
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


def project_cloud(points: np.ndarray, frame: WallFrame, band_m: float):
    relative_xy = points[:, :2] - frame.start
    along = relative_xy @ frame.tangent
    normal = (points[:, :2] - frame.center) @ frame.normal
    height = points[:, 2] - frame.z_min
    keep = (
        (along >= -0.10) & (along <= frame.length + 0.10)
        & (np.abs(normal) <= band_m)
        & (height >= -0.10) & (height <= max(4.0, frame.z_max - frame.z_min + 0.20))
    )
    return along[keep], normal[keep], height[keep]


def rasterize_tokens(
    along, normal, height, *, tile_start, frame, token_m,
    tile_width_m, tile_height_m,
):
    width_tokens = int(round(tile_width_m / token_m))
    height_tokens = int(round(tile_height_m / token_m))
    x_index = np.floor((along - tile_start) / token_m).astype(np.int32)
    z_index = np.floor(height / token_m).astype(np.int32)
    valid = (
        (x_index >= 0) & (x_index < width_tokens)
        & (z_index >= 0) & (z_index < height_tokens)
    )
    x_index, z_index, normal = x_index[valid], z_index[valid], normal[valid]
    counts = np.zeros((height_tokens, width_tokens), dtype=np.float32)
    np.add.at(counts, (z_index, x_index), 1.0)
    minimum = np.full_like(counts, np.inf)
    maximum = np.full_like(counts, -np.inf)
    if normal.size:
        np.minimum.at(minimum, (z_index, x_index), normal)
        np.maximum.at(maximum, (z_index, x_index), normal)
    span = np.where(counts >= 2.0, maximum - minimum, 0.0)
    span[~np.isfinite(span)] = 0.0
    tolerance = max(0.035, min(0.10, frame.thickness * 0.35))
    face_a = np.zeros_like(counts)
    face_b = np.zeros_like(counts)
    if normal.size:
        mask_a = np.abs(normal + frame.thickness / 2.0) <= tolerance
        mask_b = np.abs(normal - frame.thickness / 2.0) <= tolerance
        np.add.at(face_a, (z_index[mask_a], x_index[mask_a]), 1.0)
        np.add.at(face_b, (z_index[mask_b], x_index[mask_b]), 1.0)
    image = np.stack((
        np.clip(np.log1p(np.minimum(face_a, face_b)) / np.log1p(4.0), 0.0, 1.0),
        np.clip(span / 0.60, 0.0, 1.0),
        np.clip(np.log1p(counts) / np.log1p(12.0), 0.0, 1.0),
    ), axis=-1)
    return np.uint8(np.round(255.0 * np.flipud(image))), {"points": int(normal.size)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cloud_xyz", type=Path)
    parser.add_argument("stitched_model", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--floor-z", type=float, default=0.1907)
    parser.add_argument("--ceiling-z", type=float, default=3.787)
    parser.add_argument("--token-m", type=float, default=0.05)
    parser.add_argument("--tile-width-m", type=float, default=6.4)
    parser.add_argument("--tile-height-m", type=float, default=4.0)
    parser.add_argument("--wall-band-m", type=float, default=0.35)
    # Real Kladno calibration: aggressive point thinning erased the two-face
    # signature of thick exterior walls. Keep the full local crop by default;
    # the option remains available for controlled ablation/benchmark runs.
    parser.add_argument("--point-keep-ratio", type=float, default=1.0)
    parser.add_argument("--keep-threshold", type=float, default=0.50)
    parser.add_argument("--uncertain-threshold", type=float, default=0.60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_points(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, mmap_mode="r")[:, :3], dtype=np.float32)
    return np.loadtxt(path, dtype=np.float32, skiprows=1, usecols=(0, 1, 2))


def make_frame(wall: dict, floor_z: float, ceiling_z: float) -> WallFrame:
    start = np.array([wall["ax"], wall["ay"]], dtype=np.float64)
    end = np.array([wall["bx"], wall["by"]], dtype=np.float64)
    vector = end - start
    length = float(np.linalg.norm(vector))
    tangent = vector / max(length, 1e-9)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    return WallFrame(
        center=(start + end) / 2.0,
        start=start,
        tangent=tangent,
        normal=normal,
        length=length,
        thickness=float(wall["espessura"]),
        z_min=floor_z,
        z_max=ceiling_z,
    )


def starts(length: float, width: float) -> list[float]:
    if length <= width:
        return [-(width - length) / 2.0]
    count = max(2, int(np.ceil(length / width)))
    return np.linspace(0.0, max(0.0, length - width), count).tolist()


def local_points(points: np.ndarray, frame: WallFrame, band: float) -> np.ndarray:
    end = frame.start + frame.tangent * frame.length
    minimum = np.minimum(frame.start, end) - band - 0.15
    maximum = np.maximum(frame.start, end) + band + 0.15
    keep = (
        (points[:, 0] >= minimum[0]) & (points[:, 0] <= maximum[0])
        & (points[:, 1] >= minimum[1]) & (points[:, 1] <= maximum[1])
        & (points[:, 2] >= frame.z_min - 0.10)
        & (points[:, 2] <= frame.z_min + 4.10)
    )
    return points[keep]


def render_plan(
    walls: list[dict],
    openings: list[dict],
    predictions: list[dict],
    output: Path,
    uncertain: float,
    title: str = "ML + heuristic",
) -> None:
    points = np.vstack([
        np.array([[wall["ax"], wall["ay"]], [wall["bx"], wall["by"]]])
        for wall in walls
    ])
    minimum = points.min(axis=0) - 1.0
    maximum = points.max(axis=0) + 1.0
    width, height = 2000, 1500
    scale = min((width - 100) / np.ptp(points[:, 0]), (height - 100) / np.ptp(points[:, 1]))

    def pixel(point):
        return (
            int(round(50 + (point[0] - minimum[0]) * scale)),
            int(round(height - 50 - (point[1] - minimum[1]) * scale)),
        )

    canvas = np.full((height, width, 3), 247, np.uint8)
    by_id = {item["wall_id"]: item for item in predictions}
    for wall in walls:
        prediction = by_id[wall["id"]]
        predicted = prediction["predicted_class"]
        display_class = predicted if prediction["predicted_probability"] >= uncertain else "uncertain"
        color = COLORS[display_class]
        cv2.line(
            canvas,
            pixel(np.array([wall["ax"], wall["ay"]])),
            pixel(np.array([wall["bx"], wall["by"]])),
            color,
            max(2, int(round(float(wall["espessura"]) * scale))),
            cv2.LINE_AA,
        )
        midpoint = np.array([(wall["ax"] + wall["bx"]) / 2.0, (wall["ay"] + wall["by"]) / 2.0])
        cv2.putText(
            canvas,
            wall["id"].split("-")[-1],
            pixel(midpoint),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA,
        )
    opening_colors = {"door": (45, 200, 45), "window": (225, 145, 30)}
    wall_by_id = {wall["id"]: wall for wall in walls}
    for opening in openings:
        wall = wall_by_id.get(opening.get("wall_id"))
        if wall is None:
            continue
        start = np.array([wall["ax"], wall["ay"]], dtype=np.float64)
        end = np.array([wall["bx"], wall["by"]], dtype=np.float64)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            continue
        center = start + vector / length * float(opening["s_center"])
        color = opening_colors.get(opening.get("class"), (90, 90, 90))
        cv2.circle(canvas, pixel(center), 9, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(canvas, pixel(center), 6, color, -1, cv2.LINE_AA)
    counts = {name: sum(item["predicted_class"] == name for item in predictions) for name in COLORS if name != "uncertain"}
    doors = sum(item.get("class") == "door" for item in openings)
    windows = sum(item.get("class") == "window" for item in openings)
    cv2.putText(
        canvas,
        f'{title} | wall={counts["wall"]} leaf={counts["door_leaf"]} non-wall={counts["non_wall"]} door={doors} window={windows}',
        (35, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (15, 15, 15), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "lines: green=wall orange=door-leaf red=non-wall gray=uncertain | dots: green=door blue=window",
        (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (40, 40, 40), 2, cv2.LINE_AA,
    )
    cv2.imwrite(str(output), canvas)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    points = load_points(args.cloud_xyz)
    stitched = json.loads(args.stitched_model.read_text(encoding="utf-8"))
    walls = stitched["paredes"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model = WallCandidateNet(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    input_width, input_height = checkpoint["input_size"]

    images = []
    owners = []
    frames = {}
    for wall in walls:
        frame = make_frame(wall, args.floor_z, args.ceiling_z)
        frames[wall["id"]] = frame
        cropped = local_points(points, frame, args.wall_band_m)
        along, normal, height = project_cloud(cropped, frame, args.wall_band_m)
        if args.point_keep_ratio < 1.0 and len(along):
            seed = 20260825 + sum((index + 1) * ord(char) for index, char in enumerate(wall["id"]))
            keep = np.random.default_rng(seed).random(len(along)) < args.point_keep_ratio
            along, normal, height = along[keep], normal[keep], height[keep]
        for tile_start in starts(frame.length, args.tile_width_m):
            token, _ = rasterize_tokens(
                along, normal, height,
                tile_start=tile_start,
                frame=frame,
                token_m=args.token_m,
                tile_width_m=args.tile_width_m,
                tile_height_m=args.tile_height_m,
            )
            image = cv2.resize(token, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
            images.append(np.moveaxis(image.astype(np.float32) / 255.0, -1, 0))
            owners.append(wall["id"])

    probabilities = []
    with torch.inference_mode():
        for first in range(0, len(images), args.batch):
            batch = torch.from_numpy(np.stack(images[first:first + args.batch])).to(device)
            probabilities.extend(torch.softmax(model(batch), dim=1).cpu().numpy())
    by_wall = {wall["id"]: [] for wall in walls}
    for owner, probability in zip(owners, probabilities):
        by_wall[owner].append(probability)

    predictions = []
    for wall in walls:
        averaged = np.mean(np.vstack(by_wall[wall["id"]]), axis=0)
        class_index = int(np.argmax(averaged))
        predicted = classes[class_index]
        wall_probability = float(averaged[classes.index("wall")])
        predictions.append({
            "wall_id": wall["id"],
            "predicted_class": predicted,
            "predicted_probability": round(float(averaged[class_index]), 6),
            "wall_probability": round(wall_probability, 6),
            "probabilities": {
                name: round(float(averaged[index]), 6)
                for index, name in enumerate(classes)
            },
            "tile_count": len(by_wall[wall["id"]]),
            "proposed_keep": wall_probability >= args.keep_threshold,
        })

    keep_ids = {item["wall_id"] for item in predictions if item["proposed_keep"]}
    payload = {
        "schema": "cloud2bim.wall-candidate-real-review.v1",
        "source_model": str(args.stitched_model.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "automatic_geometry_change": False,
        "point_keep_ratio": args.point_keep_ratio,
        "keep_threshold": args.keep_threshold,
        "counts": {
            "input_walls": len(walls),
            "proposed_keep": len(keep_ids),
            "proposed_remove": len(walls) - len(keep_ids),
            **{
                name: sum(item["predicted_class"] == name for item in predictions)
                for name in classes
            },
        },
        "predictions": predictions,
    }
    (args.output / "wall_ml_predictions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_plan(
        walls,
        stitched.get("aberturas", []),
        predictions,
        args.output / "wall_ml_review.png",
        args.uncertain_threshold,
    )

    filtered = dict(stitched)
    filtered["schema"] = "cloud2bim.tiled-stitched-wall-ml-proposal.v1"
    filtered["wall_ml_review_only"] = True
    filtered["paredes"] = [wall for wall in walls if wall["id"] in keep_ids]
    filtered["aberturas"] = [
        opening for opening in stitched.get("aberturas", [])
        if opening["wall_id"] in keep_ids
    ]
    (args.output / "wall_ml_filtered_proposal.json").write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
