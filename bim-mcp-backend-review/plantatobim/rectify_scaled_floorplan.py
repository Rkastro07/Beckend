"""Retifica uma foto de planta e grava uma imagem com escala métrica uniforme."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Não foi possível abrir {path}")
    return image


def _parse_quad(value: str) -> np.ndarray:
    coordinates = [float(item) for item in value.split(",")]
    if len(coordinates) != 8:
        raise argparse.ArgumentTypeError("use x1,y1,x2,y2,x3,y3,x4,y4")
    return np.asarray(coordinates, dtype=np.float32).reshape(4, 2)


def rectify_floorplan(
    image_path: Path,
    output_path: Path,
    *,
    source_quad: np.ndarray,
    main_width_m: float,
    main_height_m: float,
    pixels_per_meter: float,
    margin_m: float,
    right_extra_m: float,
    normalize: bool,
    x_mapping: dict[str, object] | None = None,
    y_mapping: dict[str, object] | None = None,
) -> dict[str, object]:
    image = _read_image(image_path)
    margin_px = int(round(margin_m * pixels_per_meter))
    main_width_px = int(round(main_width_m * pixels_per_meter))
    main_height_px = int(round(main_height_m * pixels_per_meter))
    output_width = int(round((main_width_m + right_extra_m + 2 * margin_m) * pixels_per_meter))
    output_height = int(round((main_height_m + 2 * margin_m) * pixels_per_meter))
    destination_quad = np.asarray([
        [margin_px, margin_px],
        [margin_px + main_width_px, margin_px],
        [margin_px + main_width_px, margin_px + main_height_px],
        [margin_px, margin_px + main_height_px],
    ], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source_quad, destination_quad)
    unit_homography = cv2.getPerspectiveTransform(
        source_quad,
        np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
    )

    def mapping_arrays(
        mapping: dict[str, object] | None,
        default_extent_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if mapping:
            normalized = np.asarray(mapping.get("normalized", []), dtype=np.float64)
            meters = np.asarray(mapping.get("meters", []), dtype=np.float64)
            if (
                normalized.size >= 2
                and normalized.size == meters.size
                and np.all(np.diff(normalized) > 1e-8)
                and np.all(np.diff(meters) > 1e-8)
            ):
                return normalized, meters
        return (
            np.asarray([0.0, 1.0], dtype=np.float64),
            np.asarray([0.0, default_extent_m], dtype=np.float64),
        )

    def inverse_piecewise(
        metric_values: np.ndarray,
        normalized_controls: np.ndarray,
        metric_controls: np.ndarray,
    ) -> np.ndarray:
        result = np.interp(metric_values, metric_controls, normalized_controls)
        left = metric_values < metric_controls[0]
        right = metric_values > metric_controls[-1]
        if np.any(left):
            slope = (
                (normalized_controls[1] - normalized_controls[0])
                / (metric_controls[1] - metric_controls[0])
            )
            result[left] = normalized_controls[0] + (
                metric_values[left] - metric_controls[0]
            ) * slope
        if np.any(right):
            slope = (
                (normalized_controls[-1] - normalized_controls[-2])
                / (metric_controls[-1] - metric_controls[-2])
            )
            result[right] = normalized_controls[-1] + (
                metric_values[right] - metric_controls[-1]
            ) * slope
        return result

    x_normalized, x_meters = mapping_arrays(x_mapping, main_width_m)
    y_normalized, y_meters = mapping_arrays(y_mapping, main_height_m)
    target_x_m = np.arange(output_width, dtype=np.float64) / pixels_per_meter - margin_m
    target_y_m = np.arange(output_height, dtype=np.float64) / pixels_per_meter - margin_m
    source_u = inverse_piecewise(target_x_m, x_normalized, x_meters)
    source_v = inverse_piecewise(target_y_m, y_normalized, y_meters)
    grid_u, grid_v = np.meshgrid(source_u, source_v)
    inverse_unit = np.linalg.inv(unit_homography.astype(np.float64))
    denominator = (
        inverse_unit[2, 0] * grid_u
        + inverse_unit[2, 1] * grid_v
        + inverse_unit[2, 2]
    )
    map_x = (
        inverse_unit[0, 0] * grid_u
        + inverse_unit[0, 1] * grid_v
        + inverse_unit[0, 2]
    ) / denominator
    map_y = (
        inverse_unit[1, 0] * grid_u
        + inverse_unit[1, 1] * grid_v
        + inverse_unit[1, 2]
    ) / denominator
    rectified = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    if normalize:
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        illumination = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(18.0, pixels_per_meter * 0.22))
        normalized = cv2.divide(gray, np.maximum(illumination, 1), scale=245)
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        rectified = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", rectified)
    if not ok:
        raise RuntimeError("Falha ao codificar a planta retificada")
    encoded.tofile(str(output_path))
    metadata = {
        "source": str(image_path),
        "output": str(output_path),
        "source_quad_px": source_quad.round(3).tolist(),
        "destination_quad_px": destination_quad.round(3).tolist(),
        "main_width_m": main_width_m,
        "main_height_m": main_height_m,
        "pixels_per_meter": pixels_per_meter,
        "meters_per_pixel": 1.0 / pixels_per_meter,
        "margin_m": margin_m,
        "right_extra_m": right_extra_m,
        "canvas_width_m": output_width / pixels_per_meter,
        "canvas_height_m": output_height / pixels_per_meter,
        "homography": homography.round(10).tolist(),
        "unit_homography": unit_homography.round(12).tolist(),
        "axis_mappings": {
            "horizontal": {
                "normalized": x_normalized.round(9).tolist(),
                "meters": x_meters.round(9).tolist(),
            },
            "vertical": {
                "normalized": y_normalized.round(9).tolist(),
                "meters": y_meters.round(9).tolist(),
            },
        },
        "warp": "homography+separable-metric-anchors",
        "normalization": "illumination-division" if normalize else "none",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-quad", required=True, type=_parse_quad)
    parser.add_argument("--main-width-m", required=True, type=float)
    parser.add_argument("--main-height-m", required=True, type=float)
    parser.add_argument("--pixels-per-meter", type=float, default=100.0)
    parser.add_argument("--margin-m", type=float, default=0.5)
    parser.add_argument("--right-extra-m", type=float, default=1.5)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    metadata = rectify_floorplan(
        args.image,
        args.output,
        source_quad=args.source_quad,
        main_width_m=args.main_width_m,
        main_height_m=args.main_height_m,
        pixels_per_meter=args.pixels_per_meter,
        margin_m=args.margin_m,
        right_extra_m=args.right_extra_m,
        normalize=args.normalize,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
