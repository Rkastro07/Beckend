"""Vetorização determinística de plantas raster por fatias locais 1D.

O detector reduz faixas horizontais e verticais a perfis de ocupação 1D,
rastreia picos persistentes e agrupa faces paralelas em paredes. A imagem e
as faces detectadas são preservadas como evidência editável no frontend.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

try:
    from .cad_raster_ocr import (
        OCR_ENGINE,
        dimension_candidates_from_ocr,
        run_windows_ocr,
    )
except ImportError:  # suporte à execução direta por scripts legados
    from cad_raster_ocr import (
        OCR_ENGINE,
        dimension_candidates_from_ocr,
        run_windows_ocr,
    )


RASTER_SLICE_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class RasterSlicesError(RuntimeError):
    """Falha controlada do detector geométrico por fatias."""


@dataclass
class _Track:
    fixed: list[float] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    strengths: list[float] = field(default_factory=list)
    sweeps: list[float] = field(default_factory=list)
    last_sweep: float = 0.0

    def append(self, fixed: float, width: float, strength: float, sweep: float) -> None:
        self.fixed.append(float(fixed))
        self.widths.append(float(width))
        self.strengths.append(float(strength))
        self.sweeps.append(float(sweep))
        self.last_sweep = float(sweep)


def _dark_runs(profile: np.ndarray, threshold: float) -> list[tuple[float, float, float]]:
    active = np.asarray(profile >= threshold, dtype=np.uint8)
    padded = np.pad(active, (1, 1), constant_values=0)
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    detections: list[tuple[float, float, float]] = []
    for start, end in zip(starts, ends):
        width = float(end - start + 1)
        center = (float(start) + float(end)) / 2.0
        strength = float(np.mean(profile[start:end + 1]))
        detections.append((center, width, strength))
    return detections


def _track_slice_profiles(
    ink: np.ndarray,
    *,
    orientation: Literal["vertical", "horizontal"],
    slice_size: int,
    step: int,
    density_threshold: float,
    track_tolerance: float,
    gap_slices: int,
    minimum_length: float,
    minimum_support: float,
) -> list[dict[str, float | str]]:
    height, width = ink.shape
    sweep_limit = height if orientation == "vertical" else width
    tracks: list[_Track] = []
    max_gap = step * (gap_slices + 1) + 0.5

    for start in range(0, max(1, sweep_limit - slice_size + 1), step):
        end = min(sweep_limit, start + slice_size)
        sweep = (start + end - 1) / 2.0
        if orientation == "vertical":
            profile = ink[start:end, :].mean(axis=0)
        else:
            profile = ink[:, start:end].mean(axis=1)
        detections = _dark_runs(profile, density_threshold)
        available = {
            index for index, track in enumerate(tracks)
            if sweep - track.last_sweep <= max_gap
        }
        for fixed, run_width, strength in detections:
            match = min(
                available,
                key=lambda index: abs(float(np.median(tracks[index].fixed)) - fixed),
                default=None,
            )
            if match is not None:
                distance = abs(float(np.median(tracks[match].fixed)) - fixed)
                if distance > track_tolerance:
                    match = None
            if match is None:
                track = _Track()
                track.append(fixed, run_width, strength, sweep)
                tracks.append(track)
            else:
                tracks[match].append(fixed, run_width, strength, sweep)
                available.remove(match)

    lines: list[dict[str, float | str]] = []
    for track in tracks:
        if len(track.sweeps) < 2:
            continue
        start = max(0.0, min(track.sweeps) - slice_size / 2.0)
        end = min(float(sweep_limit - 1), max(track.sweeps) + slice_size / 2.0)
        length = end - start
        expected = max(1.0, length / float(step) + 1.0)
        support = min(1.0, len(track.sweeps) / expected)
        if length < minimum_length or support < minimum_support:
            continue
        lines.append({
            "orientation": orientation,
            "fixed": float(np.median(track.fixed)),
            "start": start,
            "end": end,
            "width": float(np.median(track.widths)),
            "strength": float(np.mean(track.strengths)),
            "support": support,
        })
    return lines


def _overlap(first: dict[str, Any], second: dict[str, Any]) -> tuple[float, float, float]:
    start = max(float(first["start"]), float(second["start"]))
    end = min(float(first["end"]), float(second["end"]))
    return start, end, max(0.0, end - start)


def _axis_candidates(
    lines: list[dict[str, Any]],
    *,
    minimum_separation: float,
    maximum_separation: float,
    minimum_length: float,
) -> list[dict[str, Any]]:
    axes: list[dict[str, Any]] = []
    thin: list[tuple[int, dict[str, Any]]] = []

    for index, line in enumerate(lines):
        width = float(line["width"])
        length = float(line["end"]) - float(line["start"])
        if minimum_separation <= width <= maximum_separation and length >= minimum_length:
            axes.append({
                "orientation": line["orientation"],
                "fixed": float(line["fixed"]),
                "start": float(line["start"]),
                "end": float(line["end"]),
                "thickness": width,
                "confidence": min(0.96, 0.55 + 0.22 * float(line["support"]) + 0.15 * float(line["strength"])),
                "source": "filled-band",
            })
        elif width < minimum_separation:
            thin.append((index, line))

    pair_options: list[tuple[float, int, int, dict[str, Any]]] = []
    for first_index in range(len(thin)):
        source_i, first = thin[first_index]
        for second_index in range(first_index + 1, len(thin)):
            source_j, second = thin[second_index]
            separation = abs(float(second["fixed"]) - float(first["fixed"]))
            if not minimum_separation <= separation <= maximum_separation:
                continue
            overlap_start, overlap_end, overlap_length = _overlap(first, second)
            if overlap_length < minimum_length:
                continue
            shorter = min(
                float(first["end"]) - float(first["start"]),
                float(second["end"]) - float(second["start"]),
            )
            overlap_ratio = overlap_length / max(1.0, shorter)
            if overlap_ratio < 0.48:
                continue
            support = (float(first["support"]) + float(second["support"])) / 2.0
            strength = (float(first["strength"]) + float(second["strength"])) / 2.0
            score = 0.50 * overlap_ratio + 0.30 * support + 0.20 * strength
            pair_options.append((score, source_i, source_j, {
                "orientation": first["orientation"],
                "fixed": (float(first["fixed"]) + float(second["fixed"])) / 2.0,
                "start": overlap_start,
                "end": overlap_end,
                "thickness": separation,
                "confidence": min(0.94, 0.42 + score * 0.52),
                "source": "paired-faces",
            }))

    used: set[int] = set()
    for _, source_i, source_j, axis in sorted(pair_options, reverse=True, key=lambda item: item[0]):
        if source_i in used or source_j in used:
            continue
        used.update((source_i, source_j))
        axes.append(axis)

    # Elimina reconstruções duplicadas quase coincidentes.
    deduplicated: list[dict[str, Any]] = []
    for axis in sorted(
        axes,
        key=lambda item: (float(item["confidence"]), float(item["end"]) - float(item["start"])),
        reverse=True,
    ):
        duplicate = False
        for kept in deduplicated:
            if axis["orientation"] != kept["orientation"]:
                continue
            if abs(float(axis["fixed"]) - float(kept["fixed"])) > minimum_separation * 0.75:
                continue
            _, _, overlap_length = _overlap(axis, kept)
            axis_length = float(axis["end"]) - float(axis["start"])
            kept_length = float(kept["end"]) - float(kept["start"])
            if overlap_length >= 0.65 * min(axis_length, kept_length):
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(axis)
    return deduplicated


def _snap_axis_intersections(axes: list[dict[str, Any]], tolerance: float) -> None:
    vertical = [axis for axis in axes if axis["orientation"] == "vertical"]
    horizontal = [axis for axis in axes if axis["orientation"] == "horizontal"]
    for wall_v in vertical:
        x = float(wall_v["fixed"])
        for wall_h in horizontal:
            y = float(wall_h["fixed"])
            if not (
                float(wall_h["start"]) - tolerance <= x <= float(wall_h["end"]) + tolerance
                and float(wall_v["start"]) - tolerance <= y <= float(wall_v["end"]) + tolerance
            ):
                continue
            if abs(float(wall_v["start"]) - y) <= tolerance:
                wall_v["start"] = y
            if abs(float(wall_v["end"]) - y) <= tolerance:
                wall_v["end"] = y
            if abs(float(wall_h["start"]) - x) <= tolerance:
                wall_h["start"] = x
            if abs(float(wall_h["end"]) - x) <= tolerance:
                wall_h["end"] = x


def _opening_features(
    ink: np.ndarray,
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    pixel_m: float,
    window_hint_distance_m: float | None = None,
) -> tuple[str | None, float, dict[str, float]]:
    gap_start = float(first["end"])
    gap_end = float(second["start"])
    gap = gap_end - gap_start
    fixed = (float(first["fixed"]) + float(second["fixed"])) / 2.0
    thickness = (float(first["thickness"]) + float(second["thickness"])) / 2.0
    radius = max(12, int(round(max(gap * 1.05, thickness * 3.0))))
    padding = max(4, int(round(thickness * 0.7)))
    height, width = ink.shape

    if first["orientation"] == "horizontal":
        x0 = max(0, int(math.floor(gap_start)) - padding)
        x1 = min(width, int(math.ceil(gap_end)) + padding + 1)
        y0 = max(0, int(math.floor(fixed)) - radius)
        y1 = min(height, int(math.ceil(fixed)) + radius + 1)
        local = ink[y0:y1, x0:x1]
        center_row = int(round(fixed)) - y0
        central_start = max(0, int(round(gap_start)) - x0)
        central_end = min(local.shape[1], int(round(gap_end)) - x0 + 1)
    else:
        x0 = max(0, int(math.floor(fixed)) - radius)
        x1 = min(width, int(math.ceil(fixed)) + radius + 1)
        y0 = max(0, int(math.floor(gap_start)) - padding)
        y1 = min(height, int(math.ceil(gap_end)) + padding + 1)
        local = ink[y0:y1, x0:x1].T
        center_row = int(round(fixed)) - x0
        central_start = max(0, int(round(gap_start)) - y0)
        central_end = min(local.shape[1], int(round(gap_end)) - y0 + 1)

    if local.size == 0 or central_end - central_start < 4:
        return None, 0.0, {}
    central = np.asarray(local[:, central_start:central_end], dtype=np.uint8)
    center_row = max(0, min(central.shape[0] - 1, center_row))
    longitudinal = max(1, central.shape[1])
    row_coverage = central.mean(axis=1)
    corridor_half = max(2, int(round(thickness * 0.8)))
    corridor_start = max(0, center_row - corridor_half)
    corridor_end = min(central.shape[0], center_row + corridor_half + 1)
    corridor_rows = row_coverage[corridor_start:corridor_end]
    parallel_coverage = float(corridor_rows.max(initial=0.0))
    parallel_rows = int(np.count_nonzero(corridor_rows >= 0.52))
    outside = central.copy()
    outside[corridor_start:corridor_end, :] = 0
    outside_density = float(outside.mean())

    binary = central * 255
    lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 180.0,
        threshold=max(8, int(round(gap * 0.18))),
        minLineLength=max(7, int(round(gap * 0.28))),
        maxLineGap=max(3, int(round(gap * 0.12))),
    )
    maximum_parallel = 0.0
    maximum_diagonal = 0.0
    maximum_perpendicular = 0.0
    if lines is not None:
        for raw in lines[:, 0, :]:
            dx = float(raw[2] - raw[0])
            dy = float(raw[3] - raw[1])
            length = math.hypot(dx, dy)
            angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
            angle = min(angle, 180.0 - angle)
            if angle <= 13.0:
                maximum_parallel = max(maximum_parallel, length)
            elif angle >= 76.0:
                maximum_perpendicular = max(maximum_perpendicular, length)
            else:
                maximum_diagonal = max(maximum_diagonal, length)

    parallel_ratio = min(1.0, maximum_parallel / longitudinal)
    diagonal_ratio = min(1.0, maximum_diagonal / longitudinal)
    perpendicular_ratio = min(1.0, maximum_perpendicular / longitudinal)
    outside_signal = min(1.0, outside_density * 38.0)
    gap_m = gap * pixel_m
    door_prior = 1.0 if 0.55 <= gap_m <= 1.65 else 0.58
    window_prior = 1.0 if 0.35 <= gap_m <= 3.0 else 0.55
    door_score = door_prior * (
        0.44 * max(diagonal_ratio, perpendicular_ratio * 0.88)
        + 0.34 * outside_signal
        + 0.22 * (1.0 - parallel_coverage)
    )
    window_score = window_prior * (
        0.52 * max(parallel_coverage, parallel_ratio)
        + 0.18 * min(1.0, parallel_rows / 2.0)
        + 0.30 * (1.0 - outside_signal)
    )
    if window_hint_distance_m is not None and window_hint_distance_m <= 1.4:
        window_score = min(
            0.96,
            window_score + 0.26 * (1.0 - window_hint_distance_m / 1.4),
        )
    features = {
        "gap_m": round(gap_m, 4),
        "parallel_coverage": round(parallel_coverage, 4),
        "parallel_rows": parallel_rows,
        "parallel_ratio": round(parallel_ratio, 4),
        "diagonal_ratio": round(diagonal_ratio, 4),
        "perpendicular_ratio": round(perpendicular_ratio, 4),
        "outside_density": round(outside_density, 5),
        "door_score": round(door_score, 4),
        "window_score": round(window_score, 4),
        "window_hint_distance_m": (
            round(window_hint_distance_m, 4)
            if window_hint_distance_m is not None
            else -1.0
        ),
    }
    if window_hint_distance_m is not None and window_hint_distance_m <= 1.4 and window_score >= 0.62:
        return "window", min(0.94, window_score), features
    if parallel_rows <= 3 and door_score >= 0.52:
        return "door", min(0.94, door_score), features
    if door_score >= 0.75 and max(diagonal_ratio, perpendicular_ratio) >= 0.80 and gap_m <= 1.3:
        return "door", min(0.94, door_score), features
    if 2 <= parallel_rows <= 6 and window_score >= 0.62 and window_score >= door_score + 0.04:
        return "window", min(0.94, window_score), features
    return None, max(door_score, window_score), features


def detect_slice_openings(
    axes: list[dict[str, Any]],
    ink: np.ndarray,
    *,
    canvas_width_m: float,
    window_hints: list[tuple[float, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Une trechos colineares quando o vão intermediário contém uma esquadria."""
    canvas_size = max(ink.shape)
    pixel_m = canvas_width_m / float(canvas_size)
    fixed_tolerance = max(3.0, 0.16 / pixel_m)
    minimum_gap = 0.35 / pixel_m
    maximum_gap = 2.2 / pixel_m
    working = [{**axis, "_openings": []} for axis in axes]
    evaluated: dict[tuple[Any, ...], dict[str, Any]] = {}

    while True:
        accepted: list[tuple[float, int, int, dict[str, Any]]] = []
        for first_index, first in enumerate(working):
            for second_index in range(first_index + 1, len(working)):
                second = working[second_index]
                if first["orientation"] != second["orientation"]:
                    continue
                if abs(float(first["fixed"]) - float(second["fixed"])) > fixed_tolerance:
                    continue
                ordered_first, ordered_second = (
                    (first, second)
                    if float(first["start"]) <= float(second["start"])
                    else (second, first)
                )
                gap = float(ordered_second["start"]) - float(ordered_first["end"])
                if not minimum_gap <= gap <= maximum_gap:
                    continue
                gap_start = float(ordered_first["end"])
                gap_end = float(ordered_second["start"])
                intervening = False
                for third_index, third in enumerate(working):
                    if third_index in (first_index, second_index):
                        continue
                    if third["orientation"] == first["orientation"]:
                        if abs(float(third["fixed"]) - float(first["fixed"])) > fixed_tolerance:
                            continue
                        covered = min(gap_end, float(third["end"])) - max(gap_start, float(third["start"]))
                        if covered > max(3.0, gap * 0.12):
                            intervening = True
                            break
                    else:
                        crossing = float(third["fixed"])
                        boundary = fixed_tolerance * 0.55
                        if not gap_start + boundary < crossing < gap_end - boundary:
                            continue
                        if float(third["start"]) - fixed_tolerance <= float(first["fixed"]) <= float(third["end"]) + fixed_tolerance:
                            intervening = True
                            break
                if intervening:
                    continue
                first_thickness = max(1.0, float(first["thickness"]))
                second_thickness = max(1.0, float(second["thickness"]))
                if max(first_thickness, second_thickness) / min(first_thickness, second_thickness) > 2.25:
                    continue
                key = (
                    first["orientation"],
                    round((float(first["fixed"]) + float(second["fixed"])) / 2.0, 1),
                    round(float(ordered_first["end"]), 1),
                    round(float(ordered_second["start"]), 1),
                )
                if first["orientation"] == "horizontal":
                    candidate_center = ((gap_start + gap_end) / 2.0, key[1])
                else:
                    candidate_center = (key[1], (gap_start + gap_end) / 2.0)
                hint_distance = min(
                    (
                        math.hypot(candidate_center[0] - hint[0], candidate_center[1] - hint[1])
                        * pixel_m
                        for hint in (window_hints or [])
                    ),
                    default=None,
                )
                kind, confidence, features = _opening_features(
                    ink,
                    ordered_first,
                    ordered_second,
                    pixel_m=pixel_m,
                    window_hint_distance_m=hint_distance,
                )
                evaluated[key] = {
                    "orientation": first["orientation"],
                    "fixed": key[1],
                    "gap_start": key[2],
                    "gap_end": key[3],
                    "kind": kind,
                    "confidence": round(confidence, 4),
                    **features,
                }
                if kind is None:
                    continue
                opening = {
                    "kind": kind,
                    "confidence": confidence,
                    "gap_start": float(ordered_first["end"]),
                    "gap_end": float(ordered_second["start"]),
                    "features": features,
                }
                accepted.append((confidence, first_index, second_index, opening))
        if not accepted:
            break
        _, first_index, second_index, opening = max(accepted, key=lambda item: item[0])
        first = working[first_index]
        second = working[second_index]
        first_length = float(first["end"]) - float(first["start"])
        second_length = float(second["end"]) - float(second["start"])
        total_length = max(1.0, first_length + second_length)
        merged = {
            "orientation": first["orientation"],
            "fixed": (
                float(first["fixed"]) * first_length
                + float(second["fixed"]) * second_length
            ) / total_length,
            "start": min(float(first["start"]), float(second["start"])),
            "end": max(float(first["end"]), float(second["end"])),
            "thickness": (
                float(first["thickness"]) * first_length
                + float(second["thickness"]) * second_length
            ) / total_length,
            "confidence": min(float(first["confidence"]), float(second["confidence"]), float(opening["confidence"])),
            "source": "opening-bridged",
            "_openings": [
                *first.get("_openings", []),
                *second.get("_openings", []),
                opening,
            ],
        }
        for index in sorted((first_index, second_index), reverse=True):
            working.pop(index)
        working.append(merged)

    openings: list[dict[str, Any]] = []
    clean_axes: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(working):
        local_openings = axis.pop("_openings", [])
        clean_axes.append(axis)
        for opening in local_openings:
            openings.append({
                **opening,
                "host_axis_index": axis_index,
                "orientation": axis["orientation"],
                "fixed": float(axis["fixed"]),
            })
    diagnostic = {
        "opening_candidates": len(evaluated),
        "classified_openings": len(openings),
        "door_count": sum(item["kind"] == "door" for item in openings),
        "window_count": sum(item["kind"] == "window" for item in openings),
        "candidates": list(evaluated.values())[:80],
    }
    return clean_axes, openings, diagnostic


def detect_slice_walls(
    ink: np.ndarray,
    *,
    canvas_width_m: float,
    window_hints: list[tuple[float, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Detecta faces e eixos ortogonais usando apenas perfis locais 1D."""
    if ink.ndim != 2 or min(ink.shape) < 32:
        raise RasterSlicesError("A máscara da planta é pequena ou inválida.")
    if not math.isfinite(canvas_width_m) or canvas_width_m <= 0:
        raise RasterSlicesError("A largura do canvas precisa ser positiva.")
    height, width = ink.shape
    canvas_size = max(height, width)
    pixel_m = canvas_width_m / float(canvas_size)
    slice_size = max(5, int(round(canvas_size * 0.009)))
    if slice_size % 2 == 0:
        slice_size += 1
    step = max(2, slice_size // 3)
    minimum_length = max(24.0, canvas_size * 0.060)
    minimum_separation = max(2.5, 0.065 / pixel_m)
    maximum_separation = max(minimum_separation + 2.0, 0.32 / pixel_m)

    common = {
        "slice_size": slice_size,
        "step": step,
        "density_threshold": 0.58,
        "track_tolerance": max(2.0, slice_size * 0.34),
        "gap_slices": 2,
        "minimum_length": minimum_length,
        "minimum_support": 0.42,
    }
    vertical_lines = _track_slice_profiles(ink, orientation="vertical", **common)
    horizontal_lines = _track_slice_profiles(ink, orientation="horizontal", **common)
    vertical_axes = _axis_candidates(
        vertical_lines,
        minimum_separation=minimum_separation,
        maximum_separation=maximum_separation,
        minimum_length=minimum_length,
    )
    horizontal_axes = _axis_candidates(
        horizontal_lines,
        minimum_separation=minimum_separation,
        maximum_separation=maximum_separation,
        minimum_length=minimum_length,
    )
    axes = vertical_axes + horizontal_axes
    _snap_axis_intersections(axes, tolerance=maximum_separation * 0.75)
    axes, openings, opening_diagnostic = detect_slice_openings(
        axes,
        ink,
        canvas_width_m=canvas_width_m,
        window_hints=window_hints,
    )
    diagnostics = {
        "canvas_size_px": canvas_size,
        "pixel_size_m": round(pixel_m, 7),
        "slice_size_px": slice_size,
        "slice_step_px": step,
        "vertical_face_tracks": len(vertical_lines),
        "horizontal_face_tracks": len(horizontal_lines),
        "vertical_walls": len(vertical_axes),
        "horizontal_walls": len(horizontal_axes),
        "wall_count": len(axes),
        **opening_diagnostic,
    }
    for opening in openings:
        opening["host_axis"] = axes[int(opening.pop("host_axis_index"))]
    diagnostics["openings"] = openings
    return axes, vertical_lines + horizontal_lines, diagnostics


def _load_square_image(image_path: Path, maximum_size: int = 1400) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    color = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if color is None:
        raise RasterSlicesError("Não foi possível abrir a imagem da planta.")
    original_height, original_width = color.shape[:2]
    resize_scale = min(1.0, maximum_size / float(max(original_height, original_width)))
    resized_width = max(1, int(round(original_width * resize_scale)))
    resized_height = max(1, int(round(original_height * resize_scale)))
    if (resized_width, resized_height) != (original_width, original_height):
        color = cv2.resize(color, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas_size = max(resized_width, resized_height)
    pad_x = (canvas_size - resized_width) // 2
    pad_y = (canvas_size - resized_height) // 2
    square = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    square[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = color
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return square, ink.astype(np.uint8), {
        "resize_scale": resize_scale,
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "original_width": float(original_width),
        "original_height": float(original_height),
        "canvas_size": float(canvas_size),
    }


def _mask_ocr_words(ink: np.ndarray, ocr_result: dict[str, Any], transform: dict[str, float]) -> int:
    masked = 0
    scale = transform["resize_scale"]
    pad_x = transform["pad_x"]
    pad_y = transform["pad_y"]
    height, width = ink.shape
    margin = max(1, int(round(transform["canvas_size"] * 0.0025)))
    for line in ocr_result.get("lines") or []:
        for word in line.get("words") or []:
            x0 = int(math.floor(pad_x + float(word.get("x") or 0) * scale)) - margin
            y0 = int(math.floor(pad_y + float(word.get("y") or 0) * scale)) - margin
            x1 = int(math.ceil(x0 + float(word.get("width") or 0) * scale)) + margin * 2
            y1 = int(math.ceil(y0 + float(word.get("height") or 0) * scale)) + margin * 2
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(width - 1, x1), min(height - 1, y1)
            if x1 > x0 and y1 > y0:
                ink[y0:y1 + 1, x0:x1 + 1] = 0
                masked += 1
    return masked


def _world_segment(item: dict[str, Any], canvas_size: int, canvas_width_m: float) -> tuple[float, float, float, float]:
    scale = canvas_width_m / float(canvas_size)
    if item["orientation"] == "vertical":
        x = float(item["fixed"]) * scale
        return (
            round(x, 5),
            round((canvas_size - float(item["end"])) * scale, 5),
            round(x, 5),
            round((canvas_size - float(item["start"])) * scale, 5),
        )
    y = (canvas_size - float(item["fixed"])) * scale
    return (
        round(float(item["start"]) * scale, 5),
        round(y, 5),
        round(float(item["end"]) * scale, 5),
        round(y, 5),
    )


def raster_slices_image_to_editor_model(
    image_path: Path,
    *,
    canvas_width_m: float = 20.0,
) -> dict[str, Any]:
    image_path = Path(image_path)
    if image_path.suffix.lower() not in RASTER_SLICE_IMAGE_EXTENSIONS:
        raise RasterSlicesError("Formato raster não suportado pelo detector de fatias.")
    square, ink, transform = _load_square_image(image_path)
    canvas_size = int(transform["canvas_size"])
    ocr_diagnostic: dict[str, Any]
    try:
        ocr_result = run_windows_ocr(image_path)
        masked_words = _mask_ocr_words(ink, ocr_result, transform)
        dimensions = dimension_candidates_from_ocr(
            ocr_result,
            canvas_width_m=canvas_width_m,
            canvas_size=canvas_size,
        )
        ocr_diagnostic = {
            "status": "ok",
            "engine": ocr_result.get("engine", OCR_ENGINE),
            "language": ocr_result.get("language"),
            "cache_hit": bool(ocr_result.get("cache_hit")),
            "line_count": len(ocr_result.get("lines") or []),
            "masked_words": masked_words,
            "dimension_count": len(dimensions),
        }
    except Exception as exc:
        dimensions = []
        ocr_diagnostic = {
            "status": "failed",
            "engine": OCR_ENGINE,
            "error": str(exc),
            "masked_words": 0,
        }

    world_to_pixel = canvas_size / float(canvas_width_m)
    window_hints = [
        (
            float(dimension["position"]["x"]) * world_to_pixel,
            canvas_size - float(dimension["position"]["y"]) * world_to_pixel,
        )
        for dimension in dimensions
        if dimension.get("kind") == "object-size"
    ]
    axes, faces, diagnostics = detect_slice_walls(
        ink,
        canvas_width_m=canvas_width_m,
        window_hints=window_hints,
    )
    scale = canvas_width_m / float(canvas_size)
    walls: list[dict[str, Any]] = []
    for index, axis in enumerate(axes):
        ax, ay, bx, by = _world_segment(axis, canvas_size, canvas_width_m)
        walls.append({
            "id": f"W-S1D-{index + 1:03d}",
            "ax": ax,
            "ay": ay,
            "bx": bx,
            "by": by,
            "espessura": round(max(0.065, min(0.32, float(axis["thickness"]) * scale)), 4),
            "altura": 2.8,
            "elevacao": 0.0,
            "layer": "Wall-Fatias-1D",
            "nome": f"Parede fatia {index + 1}",
            "origem": "raster-slices-1d",
            "confidence": round(float(axis["confidence"]), 4),
        })

    opening_specs = diagnostics.pop("openings", [])
    openings: list[dict[str, Any]] = []
    for index, opening in enumerate(opening_specs):
        host_axis = opening.pop("host_axis")
        try:
            host_index = axes.index(host_axis)
        except ValueError:
            continue
        wall = walls[host_index]
        gap_center = (float(opening["gap_start"]) + float(opening["gap_end"])) / 2.0
        if host_axis["orientation"] == "vertical":
            center_along_wall = (float(host_axis["end"]) - gap_center) * scale
        else:
            center_along_wall = (gap_center - float(host_axis["start"])) * scale
        wall_length = math.hypot(
            float(wall["bx"]) - float(wall["ax"]),
            float(wall["by"]) - float(wall["ay"]),
        )
        opening_width = min(
            max(0.35, (float(opening["gap_end"]) - float(opening["gap_start"])) * scale),
            max(0.35, wall_length - 0.04),
        )
        center_along_wall = max(
            opening_width / 2.0,
            min(wall_length - opening_width / 2.0, center_along_wall),
        )
        kind = str(opening["kind"])
        openings.append({
            "id": f"O-S1D-{index + 1:03d}",
            "parede_id": wall["id"],
            "tipo": kind,
            "s_centro": round(center_along_wall, 5),
            "largura": round(opening_width, 4),
            "nome": f"{'Porta' if kind == 'door' else 'Janela'} fatia {index + 1}",
            "altura": 2.1 if kind == "door" else 1.2,
            "peitoril": 0.0 if kind == "door" else 1.0,
            "origem": "raster-slices-opening",
            "confidence": round(float(opening["confidence"]), 4),
            "semantic_reason": "classificação local por vão, linhas e ocupação fora da parede",
        })

    slice_segments = []
    for index, face in enumerate(faces):
        ax, ay, bx, by = _world_segment(face, canvas_size, canvas_width_m)
        slice_segments.append({
            "id": f"S1D-FACE-{index + 1:04d}",
            "orientation": face["orientation"],
            "points": [[ax, ay], [bx, by]],
            "width_px": round(float(face["width"]), 3),
            "support": round(float(face["support"]), 4),
            "strength": round(float(face["strength"]), 4),
        })

    ok, png_buffer = cv2.imencode(".png", square)
    if not ok:
        raise RasterSlicesError("Não foi possível codificar a imagem alinhada.")
    bounds = [0.0, 0.0, float(canvas_width_m), float(canvas_width_m)]
    return {
        "ok": True,
        "escala": scale,
        "single_line": False,
        "nome": image_path.stem,
        "bbox": {"xmin": 0.0, "ymin": 0.0, "xmax": canvas_width_m, "ymax": canvas_width_m},
        "diagnostico": {
            "sobras": 0,
            "cantos_costurados": 0,
            "blocos_esquadria": len(openings),
            "elementos_lidos": len(walls) + len(openings),
            "geometrias_aproximadas": 0,
        },
        "source": {
            "format": image_path.suffix.lower().lstrip(".") or "image",
            "family": "raster",
            "mode": "raster-slices-1d",
            "semantic_level": "geometric",
            "scale_source": "user-canvas-width",
        },
        "reference": {
            "kind": "raster2seq",
            "engine": "slices-1d",
            "label": "Faces detectadas por fatias 1D",
            "bounds": bounds,
            "image_mime": "image/png",
            "image_base64": base64.b64encode(png_buffer.tobytes()).decode("ascii"),
            "canvas_size": [canvas_size, canvas_size],
            "canvas_width_m": canvas_width_m,
            "rooms": [],
            "openings": [],
            "dimensions": dimensions,
            "slice_segments": slice_segments,
        },
        "warnings": [
            "Protótipo geométrico: paredes ortogonais foram reconstruídas por persistência entre fatias 1D.",
            "Confirme paredes curtas, diagonais e símbolos antes de gerar o IFC.",
            f"A heurística classificou {len(openings)} abertura(s) por evidência local; revise tipo e largura.",
        ],
        "paredes": walls,
        "aberturas": openings,
        "laje": {
            "contorno": [],
            "piso": {"ativo": False, "espessura": 0.12},
            "teto": {"ativo": False, "espessura": 0.12},
        },
        "spaces": [],
        "raster_slices": diagnostics,
        "raster_ocr": ocr_diagnostic,
    }
