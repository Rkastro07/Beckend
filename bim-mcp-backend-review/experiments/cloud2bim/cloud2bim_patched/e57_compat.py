"""Stable E57 reader shared by the runner and patched Cloud2BIM."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pye57


def _normalized_channel(values: np.ndarray, count: int) -> np.ndarray:
    if values.size == 0:
        return np.full(count, 0.5, dtype=float)
    result = np.asarray(values, dtype=float)
    peak = float(np.nanmax(result)) if result.size else 0.0
    if peak > 1.0:
        result = result / (255.0 if peak <= 255.0 else 65535.0)
    return np.clip(result, 0.0, 1.0)


def read_e57_points(file_name: str):
    """Read every scan and return points, normalized RGB and intensity arrays."""
    source = pye57.E57(str(file_name))
    point_blocks: list[np.ndarray] = []
    color_blocks: list[np.ndarray] = []
    intensity_blocks: list[np.ndarray] = []

    for scan_index in range(source.scan_count):
        data = source.read_scan(
            scan_index,
            colors=True,
            intensity=True,
            row_column=False,
            ignore_missing_fields=True,
        )
        points = np.column_stack(
            (
                data["cartesianX"],
                data["cartesianY"],
                data["cartesianZ"],
            )
        ).astype(float, copy=False)
        count = len(points)
        colors = np.column_stack(
            (
                _normalized_channel(np.asarray(data.get("colorRed", [])), count),
                _normalized_channel(np.asarray(data.get("colorGreen", [])), count),
                _normalized_channel(np.asarray(data.get("colorBlue", [])), count),
            )
        )
        intensity = np.asarray(
            data.get("intensity", np.zeros(count)),
            dtype=float,
        ).reshape(-1, 1)
        point_blocks.append(points)
        color_blocks.append(colors)
        intensity_blocks.append(intensity)

    if not point_blocks:
        return SimpleNamespace(
            points=np.empty((0, 3), dtype=float),
            color=np.empty((0, 3), dtype=float),
            intensity=np.empty((0, 1), dtype=float),
        )
    return SimpleNamespace(
        points=np.concatenate(point_blocks, axis=0),
        color=np.concatenate(color_blocks, axis=0),
        intensity=np.concatenate(intensity_blocks, axis=0),
    )
