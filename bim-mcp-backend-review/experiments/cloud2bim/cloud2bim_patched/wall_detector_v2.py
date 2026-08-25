# -*- coding: utf-8 -*-
"""Geometric wall reconstruction helpers used by ``identify_walls`` V2.

The original Cloud2BIM wall detector treats every contour segment as a final
piece of geometry and then greedily groups nearby parallel segments.  That is
fragile when a wall face is interrupted by a door, window, occlusion or sparse
scan.  This module separates the two concepts:

* contour segments are *evidence*;
* a wall is a reconstructed entity supported by one or two evidence faces.

The public helpers intentionally operate on the same ``[[x, y], [x, y]]``
segment representation used by the original project so the IFC generation
contract does not change.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


_EPS = 1e-9


@dataclass(frozen=True)
class SegmentFrame:
    segment: list[list[float]]
    u: np.ndarray
    n: np.ndarray
    theta: float
    rho: float
    t0: float
    t1: float
    length: float


@dataclass
class RefinementContext:
    points: np.ndarray
    tree: object
    z_min: float
    z_max: float


def build_multislice_wall_grid(points, z_min: float, z_max: float,
                               pixel_size: float, n_slices: int = 6,
                               minimum_slice_fraction: float = 0.30):
    """Rasterize XY evidence while preserving its vertical persistence.

    Returns the density grid, binary persistent-wall mask, bin edges, cached
    per-height masks and the absolute number of slices required.  Furniture or
    noise visible in only one height band therefore never reaches contour
    vectorization.
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3 or len(array) < 2:
        raise ValueError("at least two XYZ points are required")
    if not (np.isfinite(array[:, :3]).all() and z_min < z_max and pixel_size > 0):
        raise ValueError("invalid point cloud, height interval or pixel size")
    n_slices = max(3, int(n_slices))
    in_band = array[(array[:, 2] >= z_min) & (array[:, 2] <= z_max), :3]
    if len(in_band) < 2:
        raise ValueError("not enough points in the selected wall-height interval")

    x_min, y_min = np.min(in_band[:, :2], axis=0)
    x_max, y_max = np.max(in_band[:, :2], axis=0)
    nx = max(1, int(np.ceil((x_max - x_min) / pixel_size)))
    ny = max(1, int(np.ceil((y_max - y_min) / pixel_size)))
    x_edges = x_min + np.arange(nx + 1) * pixel_size
    y_edges = y_min + np.arange(ny + 1) * pixel_size
    density, _, _ = np.histogram2d(
        in_band[:, 0], in_band[:, 1], bins=[x_edges, y_edges])
    density = density.T

    z_edges = np.linspace(z_min, z_max, n_slices + 1)
    masks = []
    for index in range(n_slices):
        lo, hi = z_edges[index:index + 2]
        if index == n_slices - 1:
            selected = ((array[:, 2] >= lo) & (array[:, 2] <= hi))
        else:
            selected = ((array[:, 2] >= lo) & (array[:, 2] < hi))
        if selected.any():
            slice_density, _, _ = np.histogram2d(
                array[selected, 0], array[selected, 1],
                bins=[x_edges, y_edges])
            masks.append(slice_density.T > 0)
        else:
            masks.append(np.zeros_like(density, dtype=bool))
    masks = np.asarray(masks, dtype=bool)
    required_slices = max(2, int(np.ceil(
        n_slices * float(minimum_slice_fraction))))
    binary = (masks.sum(axis=0) >= required_slices).astype(np.uint8) * 255
    grid = {
        "slice_masks": masks,
        "pixel_size": float(pixel_size),
        "x_min": float(x_min),
        "y_min": float(y_min),
    }
    return density, binary, x_edges, y_edges, grid, required_slices


def _normalised_segment(segment: Sequence[Sequence[float]]) -> list[list[float]]:
    a = np.asarray(segment[0], dtype=float)[:2]
    b = np.asarray(segment[1], dtype=float)[:2]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite wall segment")
    d = b - a
    length = float(np.linalg.norm(d))
    if length <= _EPS:
        raise ValueError("degenerate wall segment")
    u = d / length
    # Give an undirected line one deterministic orientation.  This also gives
    # its normal a deterministic sign, so rho values can be compared.
    if u[0] < -_EPS or (abs(u[0]) <= _EPS and u[1] < 0):
        a, b = b, a
    return [a.tolist(), b.tolist()]


def segment_frame(segment: Sequence[Sequence[float]]) -> SegmentFrame:
    seg = _normalised_segment(segment)
    a, b = np.asarray(seg[0]), np.asarray(seg[1])
    d = b - a
    length = float(np.linalg.norm(d))
    u = d / length
    n = np.array([-u[1], u[0]], dtype=float)
    theta = float(math.atan2(u[1], u[0]) % math.pi)
    ts = np.array([a @ u, b @ u])
    return SegmentFrame(
        segment=seg,
        u=u,
        n=n,
        theta=theta,
        rho=float(((a + b) * 0.5) @ n),
        t0=float(ts.min()),
        t1=float(ts.max()),
        length=length,
    )


def angle_difference_deg(a: SegmentFrame, b: SegmentFrame) -> float:
    delta = abs(a.theta - b.theta)
    delta = min(delta, math.pi - delta)
    return math.degrees(delta)


def _distance_to_infinite_line(points: np.ndarray, frame: SegmentFrame) -> np.ndarray:
    return np.abs(points @ frame.n - frame.rho)


def _same_supporting_line(a: SegmentFrame, b: SegmentFrame,
                          angle_tolerance: float, rho_tolerance: float) -> bool:
    if angle_difference_deg(a, b) > angle_tolerance:
        return False
    pa = np.asarray(a.segment)
    pb = np.asarray(b.segment)
    # Symmetric endpoint-to-line residual.  Using the endpoints, rather than
    # only the midpoints, prevents long segments with a small angular drift
    # from being fused into one bent wall face.
    residual = max(float(_distance_to_infinite_line(pa, b).mean()),
                   float(_distance_to_infinite_line(pb, a).mean()))
    return residual <= rho_tolerance


def _intervals_in_frame(a: SegmentFrame, b: SegmentFrame):
    """Return both segment intervals expressed along the longer frame."""
    ref = a if a.length >= b.length else b
    ia = sorted(float(np.asarray(p) @ ref.u) for p in a.segment)
    ib = sorted(float(np.asarray(p) @ ref.u) for p in b.segment)
    return ref, ia, ib


def _interval_gap(ia: Sequence[float], ib: Sequence[float]) -> float:
    if ia[1] < ib[0]:
        return float(ib[0] - ia[1])
    if ib[1] < ia[0]:
        return float(ia[0] - ib[1])
    return 0.0


def _grid_hits(segment: Sequence[Sequence[float]], grid,
               corridor_pixels: int = 1) -> np.ndarray:
    """Boolean longitudinal occupancy for every vertical slice of a segment."""
    if not grid or grid.get("slice_masks") is None:
        return np.zeros((0, 0), dtype=bool)
    masks = np.asarray(grid["slice_masks"], dtype=bool)
    if masks.ndim != 3 or masks.shape[0] == 0:
        return np.zeros((0, 0), dtype=bool)
    p = float(grid["pixel_size"])
    xmin, ymin = float(grid["x_min"]), float(grid["y_min"])
    a, b = np.asarray(segment[0], float), np.asarray(segment[1], float)
    length = float(np.linalg.norm(b - a))
    count = max(2, int(math.ceil(length / max(p * 0.75, 0.02))) + 1)
    samples = a[None, :] + np.linspace(0.0, 1.0, count)[:, None] * (b - a)[None, :]
    ix = np.floor((samples[:, 0] - xmin) / p).astype(int)
    iy = np.floor((samples[:, 1] - ymin) / p).astype(int)
    valid = ((ix >= 0) & (ix < masks.shape[2]) &
             (iy >= 0) & (iy < masks.shape[1]))
    ix, iy = ix[valid], iy[valid]
    if len(ix) == 0:
        return np.zeros((masks.shape[0], 0), dtype=bool)
    hits = np.zeros((masks.shape[0], len(ix)), dtype=bool)
    r = max(0, int(corridor_pixels))
    for dy in range(-r, r + 1):
        yy = np.clip(iy + dy, 0, masks.shape[1] - 1)
        for dx in range(-r, r + 1):
            xx = np.clip(ix + dx, 0, masks.shape[2] - 1)
            hits |= masks[:, yy, xx]
    return hits


def _grid_signature(segment: Sequence[Sequence[float]], grid,
                    corridor_pixels: int = 1) -> np.ndarray:
    """Occupancy coverage per vertical slice along a world-coordinate segment."""
    hits = _grid_hits(segment, grid, corridor_pixels)
    if hits.shape[1] == 0:
        return np.zeros(hits.shape[0], dtype=float)
    return hits.mean(axis=1)


def _common_face_hits(pair, grid, corridor_pixels: int):
    """Sample both proposed faces at identical longitudinal positions."""
    u, n, rho_a, rho_b, t0, t1 = _pair_geometry(pair)
    common_pair = [
        [(u * t0 + n * rho).tolist(), (u * t1 + n * rho).tolist()]
        for rho in (rho_a, rho_b)
    ]
    return [
        _grid_hits(face, grid, corridor_pixels=corridor_pixels)
        for face in common_pair
    ]


def _longest_true_run(values) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _face_hits_metrics(face_hits, persistent_slices: int = 4):
    """Summarise two aligned ``height x length`` occupancy matrices."""
    n_slices = max((hits.shape[0] for hits in face_hits), default=0)
    if n_slices == 0:
        return {
            "accepted_face": -1,
            "face_coverages": np.zeros((2, 0), dtype=float),
            "bottom_coverage": 0.0,
            "top_coverage": 0.0,
            "persistent_coverage": 0.0,
            "face_persistent_coverages": np.zeros(2, dtype=float),
            "second_face_persistent_coverage": 0.0,
            "paired_persistent_coverage": 0.0,
            "pair_coherence": 0.0,
            "upper_run_coverage": 0.0,
            "score": 0.0,
        }

    coverages = []
    persistences = []
    persistent_columns = []
    for hits in face_hits:
        if hits.shape[1] == 0:
            coverages.append(np.zeros(n_slices, dtype=float))
            persistences.append(0.0)
            persistent_columns.append(np.zeros(0, dtype=bool))
            continue
        coverage = hits.mean(axis=1)
        coverages.append(coverage)
        required = min(max(2, int(persistent_slices)), hits.shape[0])
        columns = hits.sum(axis=0) >= required
        persistent_columns.append(columns)
        persistences.append(float(columns.mean()))
    coverages = np.asarray(coverages, dtype=float)
    persistences = np.asarray(persistences, dtype=float)
    edge_count = max(1, min(2, n_slices // 2))
    bottom = coverages[:, :edge_count].mean(axis=1)
    top = coverages[:, -edge_count:].mean(axis=1)
    # One coherent observed face is enough for an exterior or single-line wall;
    # its synthetic hidden face is expected to have little or no cloud support.
    face_scores = 0.40 * top + 0.25 * bottom + 0.35 * persistences
    accepted_face = int(np.argmax(face_scores))
    same_width = (face_hits[0].shape == face_hits[1].shape and
                  face_hits[0].shape[1] > 0)
    if same_width:
        union = np.logical_or(face_hits[0], face_hits[1])
        coincidence = np.logical_and(face_hits[0], face_hits[1])
        pair_coherence = float(coincidence.sum() / max(int(union.sum()), 1))
        paired_columns = np.logical_and(
            persistent_columns[0], persistent_columns[1])
        paired_persistence = float(paired_columns.mean())
        upper_columns = face_hits[accepted_face][-edge_count:, :].any(axis=0)
        upper_run = (_longest_true_run(upper_columns) /
                     max(len(upper_columns), 1))
    else:
        pair_coherence = 0.0
        paired_persistence = 0.0
        upper_run = 0.0
    score = float(np.clip(
        0.80 * face_scores[accepted_face] +
        0.10 * pair_coherence + 0.10 * paired_persistence,
        0.0, 1.0))
    return {
        "accepted_face": accepted_face,
        "face_coverages": coverages,
        "bottom_coverage": float(bottom[accepted_face]),
        "top_coverage": float(top[accepted_face]),
        "persistent_coverage": float(persistences[accepted_face]),
        "face_persistent_coverages": persistences,
        "second_face_persistent_coverage": float(np.min(persistences)),
        "paired_persistent_coverage": paired_persistence,
        "pair_coherence": pair_coherence,
        "upper_run_coverage": float(upper_run),
        "score": score,
    }


def face_pair_vertical_metrics(pair, grid, *, corridor_pixels: int = 1,
                               persistent_slices: int = 4):
    """Measure vertical and lateral persistence on the coarse raster grid."""
    if len(pair) != 2:
        raise ValueError("a wall candidate must contain exactly two faces")
    metrics = _face_hits_metrics(
        _common_face_hits(pair, grid, corridor_pixels),
        persistent_slices=persistent_slices,
    )
    metrics.update({
        "profile_source": "raster",
        "profile_point_count": 0,
        "profile_face_point_counts": np.zeros(2, dtype=int),
        "measured_thickness": 0.0,
        "thickness_mad": 0.0,
        "face_rms": 0.0,
    })
    return metrics


def face_pair_point_metrics(pair, context: RefinementContext | None, *,
                            n_slices: int = 6,
                            persistent_slices: int = 4,
                            longitudinal_bin_size: float = 0.15,
                            minimum_points_per_cell: int = 2,
                            face_band: float | None = None):
    """Build a local ``z x s x face`` profile from original XYZ points."""
    if context is None or len(pair) != 2:
        return None
    u, n, rho_a, rho_b, t0, t1 = _pair_geometry(pair)
    length = float(t1 - t0)
    separation = float(abs(rho_b - rho_a))
    z_span = float(context.z_max - context.z_min)
    if length <= _EPS or separation <= _EPS or z_span <= _EPS:
        return None
    n_slices = max(3, int(n_slices))
    longitudinal_bin_size = max(0.06, float(longitudinal_bin_size))
    n_longitudinal = max(2, int(math.ceil(length / longitudinal_bin_size)))
    actual_bin_size = length / n_longitudinal
    sample_t = t0 + (np.arange(n_longitudinal) + 0.5) * actual_bin_size
    rho_mid = 0.5 * (rho_a + rho_b)
    sample_centres = (sample_t[:, None] * u[None, :] +
                      rho_mid * n[None, :])
    band = (max(0.025, min(0.060, 0.35 * separation))
            if face_band is None else max(0.015, float(face_band)))
    query_radius = 0.5 * separation + band + 0.04
    neighbours = context.tree.query_ball_point(sample_centres, r=query_radius)
    nonempty = [np.asarray(ids, dtype=np.int64)
                for ids in neighbours if len(ids)]
    if not nonempty:
        return None
    indices = np.unique(np.concatenate(nonempty))
    xyz = np.asarray(context.points[indices, :3], dtype=float)
    along = xyz[:, :2] @ u
    normal = xyz[:, :2] @ n
    valid = (
        (along >= t0) & (along <= t1) &
        (xyz[:, 2] >= context.z_min) & (xyz[:, 2] <= context.z_max) &
        (normal >= min(rho_a, rho_b) - band) &
        (normal <= max(rho_a, rho_b) + band)
    )
    if int(valid.sum()) < 12:
        return None
    xyz, along, normal = xyz[valid], along[valid], normal[valid]
    s_index = np.floor((along - t0) / length * n_longitudinal).astype(int)
    z_index = np.floor(
        (xyz[:, 2] - context.z_min) / z_span * n_slices).astype(int)
    s_index = np.clip(s_index, 0, n_longitudinal - 1)
    z_index = np.clip(z_index, 0, n_slices - 1)
    distances = np.column_stack([
        np.abs(normal - rho_a), np.abs(normal - rho_b)])
    nearest_face = np.argmin(distances, axis=1)
    face_hits = []
    face_point_counts = []
    face_residuals = []
    for face_index in range(2):
        selected = ((nearest_face == face_index) &
                    (distances[:, face_index] <= band))
        counts = np.zeros((n_slices, n_longitudinal), dtype=np.int32)
        np.add.at(counts, (z_index[selected], s_index[selected]), 1)
        face_hits.append(counts >= max(1, int(minimum_points_per_cell)))
        face_point_counts.append(int(selected.sum()))
        residual = distances[selected, face_index]
        face_residuals.append(float(np.sqrt(np.mean(residual ** 2)))
                              if len(residual) else 0.0)
    if max(face_point_counts, default=0) < max(12, n_longitudinal // 2):
        return None
    metrics = _face_hits_metrics(
        face_hits, persistent_slices=persistent_slices)
    thickness_samples = []
    for longitudinal_index in range(n_longitudinal):
        in_bin = s_index == longitudinal_index
        face_a = normal[in_bin & (nearest_face == 0) &
                        (distances[:, 0] <= band)]
        face_b = normal[in_bin & (nearest_face == 1) &
                        (distances[:, 1] <= band)]
        if len(face_a) >= 3 and len(face_b) >= 3:
            thickness_samples.append(abs(
                float(np.median(face_b)) - float(np.median(face_a))))
    if thickness_samples:
        measured_thickness = float(np.median(thickness_samples))
        thickness_mad = float(np.median(np.abs(
            np.asarray(thickness_samples) - measured_thickness)))
    else:
        measured_thickness = 0.0
        thickness_mad = 0.0
    metrics.update({
        "profile_source": "original_points",
        "profile_point_count": int(sum(face_point_counts)),
        "profile_face_point_counts": np.asarray(face_point_counts, dtype=int),
        "measured_thickness": measured_thickness,
        "thickness_mad": thickness_mad,
        "face_rms": float(max(face_residuals, default=0.0)),
        "longitudinal_bins": int(n_longitudinal),
        "longitudinal_bin_size": float(actual_bin_size),
        "face_band": float(band),
    })
    return metrics


def wall_pair_has_vertical_support(pair, grid, *, corridor_pixels: int = 1,
                                   persistent_slices: int = 4,
                                   minimum_bottom_coverage: float = 0.12,
                                   minimum_top_coverage: float = 0.25,
                                   minimum_persistent_coverage: float = 0.10,
                                   require_two_faces: bool = False,
                                   minimum_second_face_persistence: float = 0.08,
                                   minimum_pair_coherence: float = 0.08,
                                   minimum_paired_persistence: float = 0.05,
                                   context: RefinementContext | None = None,
                                   n_slices: int = 6,
                                   longitudinal_bin_size: float = 0.15,
                                   minimum_points_per_cell: int = 2,
                                   face_band: float | None = None):
    """Return whether the local height-by-length signature is wall-like."""
    raster_metrics = face_pair_vertical_metrics(
        pair,
        grid,
        corridor_pixels=corridor_pixels,
        persistent_slices=persistent_slices,
    )
    point_metrics = face_pair_point_metrics(
        pair, context, n_slices=n_slices,
        persistent_slices=persistent_slices,
        longitudinal_bin_size=longitudinal_bin_size,
        minimum_points_per_cell=minimum_points_per_cell,
        face_band=face_band)
    metrics = point_metrics if point_metrics is not None else raster_metrics
    metrics["raster_score"] = float(raster_metrics["score"])
    single_face_accepted = (
        metrics["bottom_coverage"] >= float(minimum_bottom_coverage)
        and metrics["top_coverage"] >= float(minimum_top_coverage)
        and metrics["persistent_coverage"] >= float(minimum_persistent_coverage)
    )
    accepted = single_face_accepted
    if require_two_faces:
        accepted = accepted and (
            metrics["second_face_persistent_coverage"] >=
            float(minimum_second_face_persistence)
            and metrics["pair_coherence"] >= float(minimum_pair_coherence)
            and metrics["paired_persistent_coverage"] >=
            float(minimum_paired_persistence)
        )
    metrics["single_face_accepted"] = bool(single_face_accepted)
    metrics["two_face_accepted"] = bool(
        accepted if require_two_faces else False)
    return bool(accepted), metrics


def _gap_segment(a: SegmentFrame, b: SegmentFrame):
    ref, ia, ib = _intervals_in_frame(a, b)
    if ia[1] < ib[0]:
        s0, s1 = ia[1], ib[0]
    elif ib[1] < ia[0]:
        s0, s1 = ib[1], ia[0]
    else:
        return None
    # Average both supporting-line positions in the reference normal frame.
    rho_a = float(np.mean([np.asarray(p) @ ref.n for p in a.segment]))
    rho_b = float(np.mean([np.asarray(p) @ ref.n for p in b.segment]))
    rho = (rho_a + rho_b) * 0.5
    return [(ref.u * s0 + ref.n * rho).tolist(),
            (ref.u * s1 + ref.n * rho).tolist()]


def _gap_can_be_closed(a: SegmentFrame, b: SegmentFrame, grid,
                       max_gap: float, max_unseen_gap: float,
                       minimum_slice_support: float) -> bool:
    ref, ia, ib = _intervals_in_frame(a, b)
    gap = _interval_gap(ia, ib)
    if gap <= 0:
        return True
    if gap > max_gap:
        return False
    if gap <= max_unseen_gap:
        return True
    gap_segment = _gap_segment(a, b)
    if gap_segment is None:
        return True
    signature = _grid_signature(gap_segment, grid, corridor_pixels=1)
    # One supported height band is enough to connect a door/window interval;
    # completely unseen long gaps remain separate walls.
    return bool(len(signature) and signature.max(initial=0.0) >= minimum_slice_support)


def _fit_component_segment(segments: Iterable[Sequence[Sequence[float]]]):
    points = np.vstack([np.asarray(s, dtype=float)[:2] for s in segments])
    center = np.median(points, axis=0)
    q = points - center
    covariance = q.T @ q
    values, vectors = np.linalg.eigh(covariance)
    u = vectors[:, int(np.argmax(values))]
    if u[0] < -_EPS or (abs(u[0]) <= _EPS and u[1] < 0):
        u = -u
    n = np.array([-u[1], u[0]])
    rho = float(np.median(points @ n))
    ts = points @ u
    return [(u * float(ts.min()) + n * rho).tolist(),
            (u * float(ts.max()) + n * rho).tolist()]


def merge_collinear_fragments(segments, grid=None, *, angle_tolerance=2.5,
                              rho_tolerance=0.10, max_gap=1.80,
                              max_unseen_gap=0.35,
                              minimum_slice_support=0.15):
    """Merge interrupted observations of the same physical wall face.

    Geometry establishes a shared supporting line.  For gaps larger than the
    small-occlusion allowance, the cached multi-height grid must provide point
    support in at least one height slice (e.g. a door lintel or window sill).
    """
    clean = []
    for segment in segments:
        try:
            clean.append(segment_frame(segment))
        except ValueError:
            continue
    n = len(clean)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if not _same_supporting_line(clean[i], clean[j],
                                         angle_tolerance, rho_tolerance):
                continue
            if _gap_can_be_closed(clean[i], clean[j], grid, max_gap,
                                  max_unseen_gap, minimum_slice_support):
                union(i, j)

    components = {}
    for i, frame in enumerate(clean):
        components.setdefault(find(i), []).append(frame.segment)
    merged = [_fit_component_segment(component)
              for component in components.values()]
    return merged


def _pair_candidate(a: SegmentFrame, b: SegmentFrame, min_thickness: float,
                    max_thickness: float, angle_tolerance: float,
                    minimum_overlap: float):
    angle = angle_difference_deg(a, b)
    if angle > angle_tolerance:
        return None
    ref, ia, ib = _intervals_in_frame(a, b)
    overlap = max(0.0, min(ia[1], ib[1]) - max(ia[0], ib[0]))
    shorter = min(a.length, b.length)
    required = max(minimum_overlap, min(0.75, 0.25 * shorter))
    if overlap < required:
        return None
    pa, pb = np.asarray(a.segment), np.asarray(b.segment)
    separation = 0.5 * (
        float(_distance_to_infinite_line(pa, b).mean()) +
        float(_distance_to_infinite_line(pb, a).mean()))
    if not (min_thickness <= separation <= max_thickness):
        return None
    overlap_ratio = overlap / max(shorter, _EPS)
    length_similarity = shorter / max(a.length, b.length, _EPS)
    # Prefer strong overlap, similar extents and the nearest plausible opposite
    # face.  Greedy selection below is then strictly one-to-one.
    score = (4.0 * overlap_ratio + 1.5 * length_similarity
             - angle / max(angle_tolerance, _EPS)
             - 0.8 * separation / max(max_thickness, _EPS))
    return score, separation, overlap


def _maximum_weight_pair_candidates(candidates, *, exact_component_limit=22):
    """Choose a globally best strict 1:1 matching in each candidate component.

    Greedy edge selection can consume a face needed by two slightly lower
    scoring but jointly better wall pairs.  Contour candidate graphs are small
    disconnected components, so an exact bit-mask search is both deterministic
    and cheap.  Very large pathological components retain a bounded greedy
    fallback rather than making runtime exponential.
    """
    if not candidates:
        return []
    adjacency = {}
    for item in candidates:
        i, j = int(item[1]), int(item[2])
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)

    components = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack = [start]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, ()))
        unseen.difference_update(component)
        components.append(sorted(component))

    selected = []
    for component in components:
        component_set = set(component)
        edges = [
            item for item in candidates
            if int(item[1]) in component_set and int(item[2]) in component_set
        ]
        if len(component) > int(exact_component_limit):
            used = set()
            for item in sorted(edges, reverse=True, key=lambda edge: edge[0]):
                i, j = int(item[1]), int(item[2])
                if item[0] > 0.0 and i not in used and j not in used:
                    selected.append(item)
                    used.update((i, j))
            continue

        local = {node: position for position, node in enumerate(component)}
        edge_for_local = {}
        local_adjacency = {position: [] for position in range(len(component))}
        for item in edges:
            i, j = local[int(item[1])], local[int(item[2])]
            if i > j:
                i, j = j, i
            edge_for_local[(i, j)] = item
            local_adjacency[i].append(j)
            local_adjacency[j].append(i)

        cache = {}

        def solve(mask):
            if mask == 0:
                return 0.0, ()
            if mask in cache:
                return cache[mask]
            first_bit = mask & -mask
            first = first_bit.bit_length() - 1
            remaining = mask & ~first_bit
            best_score, best_edges = solve(remaining)
            for neighbour in local_adjacency[first]:
                neighbour_bit = 1 << neighbour
                if not (remaining & neighbour_bit):
                    continue
                edge_key = (min(first, neighbour), max(first, neighbour))
                edge = edge_for_local[edge_key]
                tail_score, tail_edges = solve(remaining & ~neighbour_bit)
                candidate_score = float(edge[0]) + tail_score
                if candidate_score > best_score + 1e-12:
                    best_score = candidate_score
                    best_edges = (edge_key,) + tail_edges
            cache[mask] = (best_score, best_edges)
            return cache[mask]

        _, chosen_edges = solve((1 << len(component)) - 1)
        selected.extend(edge_for_local[edge] for edge in chosen_edges)
    return sorted(selected, reverse=True, key=lambda item: item[0])


def pair_wall_faces(segments, min_thickness: float, max_thickness: float, *,
                    angle_tolerance=3.0, minimum_overlap=0.20):
    """Return strict 1:1 face pairs and unpaired evidence segments."""
    frames = []
    for segment in segments:
        try:
            frames.append(segment_frame(segment))
        except ValueError:
            continue
    candidates = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            candidate = _pair_candidate(frames[i], frames[j], min_thickness,
                                        max_thickness, angle_tolerance,
                                        minimum_overlap)
            if candidate is not None:
                candidates.append((candidate[0], i, j, candidate))
    used = set()
    pairs = []
    diagnostics = []
    for _score, i, j, candidate in _maximum_weight_pair_candidates(candidates):
        used.update((i, j))
        pairs.append([frames[i].segment, frames[j].segment])
        diagnostics.append({
            "score": float(candidate[0]),
            "thickness": float(candidate[1]),
            "overlap": float(candidate[2]),
        })
    leftovers = [frame.segment for i, frame in enumerate(frames) if i not in used]
    return pairs, leftovers, diagnostics


def build_refinement_context(points, z_floor: float, z_ceiling: float,
                             max_points: int = 1_000_000):
    """Build a sampled XY spatial index without changing point precision."""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3 or len(array) == 0:
        return None
    finite = np.isfinite(array[:, :3]).all(axis=1)
    array = array[finite]
    lo, hi = sorted((float(z_floor), float(z_ceiling)))
    array = array[(array[:, 2] >= lo) & (array[:, 2] <= hi), :3]
    if len(array) == 0:
        return None
    if len(array) > max_points:
        step = int(math.ceil(len(array) / max_points))
        array = array[::step]
    array = np.asarray(array, dtype=np.float32)
    return RefinementContext(
        points=array,
        tree=cKDTree(array[:, :2]),
        z_min=lo,
        z_max=hi,
    )


def _pair_geometry(pair):
    a, b = segment_frame(pair[0]), segment_frame(pair[1])
    u = a.u if a.length >= b.length else b.u
    if u[0] < -_EPS or (abs(u[0]) <= _EPS and u[1] < 0):
        u = -u
    n = np.array([-u[1], u[0]])
    points_a, points_b = np.asarray(a.segment), np.asarray(b.segment)
    rho_a = float(np.mean(points_a @ n))
    rho_b = float(np.mean(points_b @ n))
    all_points = np.vstack([points_a, points_b])
    ts = all_points @ u
    return u, n, rho_a, rho_b, float(ts.min()), float(ts.max())


def _orientation_delta_deg(u, v):
    dot = float(np.clip(abs(np.dot(u, v)), 0.0, 1.0))
    return math.degrees(math.acos(dot))


def refine_face_pair(pair, context: RefinementContext | None, pixel_size: float,
                     min_thickness: float, max_thickness: float,
                     max_angle_change=5.0):
    """Refit direction and both face offsets against original 3D points."""
    if context is None:
        return pair
    u, n, rho_a, rho_b, t0, t1 = _pair_geometry(pair)
    pair_endpoints = np.vstack([
        np.asarray(pair[0], dtype=float),
        np.asarray(pair[1], dtype=float),
    ])
    length = t1 - t0
    if length <= _EPS:
        return pair
    separation = abs(rho_b - rho_a)
    sample_step = max(0.20, 2.0 * pixel_size)
    ts = np.linspace(t0, t1, max(2, int(math.ceil(length / sample_step)) + 1))
    rho_mid = (rho_a + rho_b) * 0.5
    centers = ts[:, None] * u[None, :] + rho_mid * n[None, :]
    radius = separation * 0.5 + max(0.12, 2.0 * pixel_size)
    neighbours = context.tree.query_ball_point(centers, r=radius)
    nonempty = [np.asarray(ids, dtype=np.int64) for ids in neighbours if len(ids)]
    if not nonempty:
        return pair
    indices = np.unique(np.concatenate(nonempty))
    if len(indices) < 30:
        return pair
    xy = np.asarray(context.points[indices, :2], dtype=float)
    along = xy @ u
    normal = xy @ n
    face_tolerance = max(0.06, 1.25 * pixel_size)
    # Estimate direction from the whole local wall corridor.  Selecting only
    # points close to the *proposed* raster faces biases TLS when the proposal
    # is a few degrees off: its endpoints then clip opposite halves of the two
    # real faces and can amplify, rather than correct, the angular error.
    rho_mid = (rho_a + rho_b) * 0.5
    direction_corridor = separation * 0.5 + max(0.10, 2.0 * pixel_size)
    near_wall = ((along >= t0 - 2 * pixel_size) &
                 (along <= t1 + 2 * pixel_size) &
                 (abs(normal - rho_mid) <= direction_corridor))
    xy = xy[near_wall]
    if len(xy) < 30:
        return pair

    # TLS direction refinement.  Keep the raster proposal if local points try
    # to rotate it too far (typical at intersections or on compact clutter).
    centered = xy - np.median(xy, axis=0)
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    candidate_u = vectors[:, int(np.argmax(values))]
    if np.dot(candidate_u, u) < 0:
        candidate_u = -candidate_u
    if (_orientation_delta_deg(candidate_u, u) <= max_angle_change and
            float(values.max()) > 4.0 * max(float(values.min()), _EPS)):
        u = candidate_u / np.linalg.norm(candidate_u)
        n = np.array([-u[1], u[0]])

        # ``t0`` and ``t1`` belong to the previous longitudinal basis. Using
        # them after rotating ``u`` translates a wall by an amount proportional
        # to its global coordinates. Reproject the raster endpoints into the
        # accepted TLS basis before refining their robust percentiles.
        endpoint_t = pair_endpoints @ u
        t0 = float(endpoint_t.min())
        t1 = float(endpoint_t.max())

    points_a, points_b = np.asarray(pair[0]), np.asarray(pair[1])
    predicted_a = float(np.mean(points_a @ n))
    predicted_b = float(np.mean(points_b @ n))
    normal = xy @ n

    def robust_face_offset(predicted):
        selected = normal[np.abs(normal - predicted) <= face_tolerance]
        if len(selected) < 12:
            return predicted, 0
        median = float(np.median(selected))
        mad = float(np.median(np.abs(selected - median))) + _EPS
        selected = selected[np.abs(selected - median) <= max(3.5 * mad, 0.015)]
        return (float(np.median(selected)), len(selected)) if len(selected) else (predicted, 0)

    refined_a, count_a = robust_face_offset(predicted_a)
    refined_b, count_b = robust_face_offset(predicted_b)
    refined_thickness = abs(refined_b - refined_a)
    if count_a < 12 or count_b < 12 or not (
            min_thickness <= refined_thickness <= max_thickness):
        refined_a, refined_b = predicted_a, predicted_b

    # Point support may move a raster endpoint slightly, but never radically
    # change the topology established by the interval reconstruction.
    evidence_normal = np.minimum(abs(normal - refined_a), abs(normal - refined_b))
    evidence = xy[evidence_normal <= face_tolerance]
    if len(evidence) >= 20:
        evidence_t = evidence @ u
        q0, q1 = np.quantile(evidence_t, [0.01, 0.99])
        allowance = max(0.15, 2.0 * pixel_size)
        t0 = float(np.clip(q0, t0 - allowance, t0 + allowance))
        t1 = float(np.clip(q1, t1 - allowance, t1 + allowance))
    if t1 - t0 <= 0.05:
        return pair
    return [
        [(u * t0 + n * refined_a).tolist(), (u * t1 + n * refined_a).tolist()],
        [(u * t0 + n * refined_b).tolist(), (u * t1 + n * refined_b).tolist()],
    ]


def refine_face_pairs(pairs, context, pixel_size, min_thickness, max_thickness):
    return [refine_face_pair(pair, context, pixel_size,
                             min_thickness, max_thickness)
            for pair in pairs]


def wall_axis_from_pair(pair):
    """Return the common mid-axis and thickness of an exact two-face pair."""
    u, n, rho_a, rho_b, t0, t1 = _pair_geometry(pair)
    rho = (rho_a + rho_b) * 0.5
    axis = [(u * t0 + n * rho).tolist(),
            (u * t1 + n * rho).tolist()]
    return axis, float(abs(rho_b - rho_a))


def plausible_wall_geometry(axis, thickness: float, minimum_length: float,
                            minimum_aspect: float = 2.5) -> bool:
    """Reject compact blobs that have two faces but do not look like walls.

    Vertical cabinets, columns and scan clutter may pass persistence and face
    pairing.  A wall still needs a longitudinal dimension meaningfully larger
    than its thickness.  Long thick masonry remains valid; squat elements are
    left for the column/object detectors.
    """
    values = np.asarray(axis, dtype=float)
    if values.shape != (2, 2) or not np.isfinite(values).all():
        return False
    if not np.isfinite(thickness) or thickness <= 0:
        return False
    length = float(np.linalg.norm(values[1] - values[0]))
    return (length >= float(minimum_length) and
            length / max(float(thickness), _EPS) >= float(minimum_aspect))


def _point_to_segment_distance(point, segment):
    point = np.asarray(point, dtype=float)[:2]
    a = np.asarray(segment[0], dtype=float)[:2]
    b = np.asarray(segment[1], dtype=float)[:2]
    vector = b - a
    denominator = float(vector @ vector)
    if denominator <= _EPS:
        return float(np.linalg.norm(point - a))
    parameter = float(np.clip((point - a) @ vector / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + parameter * vector)))


def _leaf_vertical_profile(points, hinge, free, thickness,
                           z_floor: float, z_ceiling: float,
                           along_bins: int = 12, height_bins: int = 12):
    """Measure a possible moving panel away from its hinge.

    Points close to the hinge are deliberately ignored because the structural
    jamb may otherwise lend full-storey support to a door or window leaf.
    Layer coverage, rather than only a Z bounding box, also makes isolated
    ceiling/background points much less influential.
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3 or len(array) == 0:
        return {
            "point_count": 0,
            "z_min": None,
            "z_max": None,
            "height": None,
            "bottom_offset": None,
            "top_gap": None,
            "layer_coverage": [],
        }
    hinge = np.asarray(hinge, dtype=float)[:2]
    free = np.asarray(free, dtype=float)[:2]
    direction = free - hinge
    length = float(np.linalg.norm(direction))
    if length <= _EPS:
        return {
            "point_count": 0,
            "z_min": None,
            "z_max": None,
            "height": None,
            "bottom_offset": None,
            "top_gap": None,
            "layer_coverage": [],
        }
    direction /= length
    normal = np.asarray([-direction[1], direction[0]])
    local = array[:, :2] - hinge
    along = local @ direction
    across = np.abs(local @ normal)
    corridor = max(0.07, 0.75 * float(thickness))
    selected = (
        (along >= max(0.10, 0.15 * length))
        & (along <= length + 0.05)
        & (across <= corridor)
        & (array[:, 2] >= float(z_floor) - 0.05)
        & (array[:, 2] <= float(z_ceiling) + 0.05)
    )
    panel = array[selected, :3]
    panel_along = along[selected]
    if len(panel) < 20:
        return {
            "point_count": int(len(panel)),
            "z_min": None,
            "z_max": None,
            "height": None,
            "bottom_offset": None,
            "top_gap": None,
            "layer_coverage": [],
        }

    z_span = max(float(z_ceiling) - float(z_floor), _EPS)
    t_edges = np.linspace(max(0.10, 0.15 * length), length + 0.05,
                          max(3, int(along_bins)) + 1)
    z_edges = np.linspace(float(z_floor), float(z_ceiling),
                          max(4, int(height_bins)) + 1)
    occupancy, _, _ = np.histogram2d(
        panel[:, 2],
        panel_along,
        bins=[z_edges, t_edges],
    )
    positive_cells = occupancy[occupancy > 0]
    # A single background/occlusion point must not make an entire height layer
    # look like panel support.  Dense leaf surfaces populate neighbouring
    # along-cells repeatedly, unlike sparse rays seen through glass/openings.
    occupied_cell_threshold = max(
        2.0,
        0.35 * float(np.median(positive_cells))
        if len(positive_cells) else 2.0,
    )
    coverage = (occupancy >= occupied_cell_threshold).mean(axis=1)
    active = np.flatnonzero(coverage >= 0.20)
    if len(active):
        z_min = float(z_edges[active[0]])
        z_max = float(z_edges[active[-1] + 1])
    else:
        z_min, z_max = np.quantile(panel[:, 2], [0.03, 0.97])
        z_min, z_max = float(z_min), float(z_max)
    return {
        "point_count": int(len(panel)),
        "z_min": z_min,
        "z_max": z_max,
        "height": float(z_max - z_min),
        "bottom_offset": float(z_min - float(z_floor)),
        "top_gap": float(float(z_ceiling) - z_max),
        "storey_height": float(z_span),
        "occupied_cell_threshold": float(occupied_cell_threshold),
        "layer_coverage": [float(value) for value in coverage],
        "occupancy_counts": [
            [int(value) for value in row] for row in occupancy],
    }


def detect_articulated_leaf_walls(
        axes, thicknesses, wall_ids, opening_anchors, points,
        z_floor: float, z_ceiling: float, *,
        wall_labels=None,
        minimum_length: float = 0.45,
        maximum_length: float = 2.10,
        maximum_thickness: float = 0.23,
        hinge_tolerance: float = 0.30,
        minimum_open_angle: float = 15.0,
        minimum_free_clearance: float = 0.24,
        allow_unanchored: bool = True,
        unanchored_maximum_length: float = 1.10,
        unanchored_hinge_tolerance: float = 0.25,
        exact_connection_tolerance: float = 0.05):
    """Detect wall candidates that are probably open door/window leaves.

    ``opening_anchors`` is a generic list of dictionaries containing ``id``,
    ``host_wall``, ``type``, ``width``, ``center`` and ``host_axis``.  The
    detector requires a hinge-sized endpoint-to-jamb match, compatible panel
    dimensions, a free outer edge and a non-full-storey vertical profile.

    Results are suggestions for the PNG approval stage.  Only entries with
    ``suppress=True`` should be removed from wall topology after approval.
    """
    if not (len(axes) == len(thicknesses) == len(wall_ids)):
        raise ValueError("axes, thicknesses and wall_ids must have equal size")
    if wall_labels is not None and len(wall_labels) != len(axes):
        raise ValueError("wall_labels must be absent or have one item per axis")
    frames = [segment_frame(axis) for axis in axes]
    wall_index = {str(identifier): index
                  for index, identifier in enumerate(wall_ids)}
    # Avoid circular evidence: a weak opening proposed on a compact panel must
    # not make that same panel structural.  Protect only credible openings that
    # physically fit inside their host with some solid material left.
    hosted_openings = {
        str(anchor.get("host_wall"))
        for anchor in opening_anchors
        if (
            anchor.get("host_wall")
            and str(anchor.get("status", "proposed")).lower()
            in {"approved", "proposed", "auto_accepted"}
            and float(anchor.get("confidence", 1.0)) >= 0.50
            and str(anchor.get("host_wall")) in wall_index
            and frames[wall_index[str(anchor.get("host_wall"))]].length
            >= float(anchor.get("width", 0.0)) + 0.20
        )
    }
    results = []
    for index, (frame, thickness, identifier) in enumerate(zip(
            frames, thicknesses, wall_ids)):
        thickness = float(thickness)
        if not (
            float(minimum_length) <= frame.length <= float(maximum_length)
            and 0.0 < thickness <= float(maximum_thickness)
        ):
            continue
        # A segment that itself hosts a detected opening has strong evidence
        # of being structural, even if it is short.
        if str(identifier) in hosted_openings:
            continue

        best = None
        for anchor in opening_anchors:
            host_id = str(anchor.get("host_wall", ""))
            host_axis = anchor.get("host_axis")
            if host_axis is None or host_id == str(identifier):
                continue
            try:
                host_frame = segment_frame(host_axis)
            except (ValueError, TypeError):
                continue
            if host_frame.length < max(1.50, 1.50 * frame.length):
                continue
            opening_type = str(anchor.get("type", "")).lower()
            if opening_type not in {"door", "window"}:
                continue
            width = float(anchor.get("width", 0.0))
            if width <= 0:
                continue
            ratio = frame.length / width
            ratio_limits = ((0.55, 1.45) if opening_type == "door"
                            else (0.25, 1.55))
            if not (ratio_limits[0] <= ratio <= ratio_limits[1]):
                continue

            center = np.asarray(anchor.get("center"), dtype=float)[:2]
            jambs = [
                center - host_frame.u * width * 0.5,
                center + host_frame.u * width * 0.5,
            ]
            endpoints = np.asarray(frame.segment, dtype=float)
            matches = [
                (float(np.linalg.norm(endpoint - jamb)), endpoint_index, jamb)
                for endpoint_index, endpoint in enumerate(endpoints)
                for jamb in jambs
            ]
            hinge_distance, hinge_index, jamb = min(
                matches, key=lambda value: value[0])
            if hinge_distance > float(hinge_tolerance):
                continue

            angle = angle_difference_deg(frame, host_frame)
            if angle < float(minimum_open_angle):
                continue
            hinge = endpoints[hinge_index]
            free = endpoints[1 - hinge_index]
            host_free_distance = _point_to_segment_distance(
                free, host_frame.segment)
            if host_free_distance < float(minimum_free_clearance):
                continue

            other_distances = []
            host_index = wall_index.get(host_id)
            for other_index, other_frame in enumerate(frames):
                if other_index in {index, host_index}:
                    continue
                other_distances.append(_point_to_segment_distance(
                    free, other_frame.segment))
            nearest_other = min(other_distances, default=float("inf"))
            if nearest_other < float(minimum_free_clearance):
                continue

            profile = _leaf_vertical_profile(
                points,
                hinge,
                free,
                thickness,
                z_floor,
                z_ceiling,
            )
            if profile["height"] is None:
                vertical_match = False
            elif opening_type == "door":
                vertical_match = (
                    profile["bottom_offset"] <= 0.40
                    and 1.40 <= profile["height"] <= 2.60
                    and profile["top_gap"] >= 0.25
                )
            else:
                vertical_match = (
                    profile["bottom_offset"] >= 0.20
                    and 0.35 <= profile["height"] <= 2.30
                    and profile["top_gap"] >= 0.15
                )

            hinge_score = max(
                0.0, 1.0 - hinge_distance / max(float(hinge_tolerance), _EPS))
            ratio_center = 1.0 if opening_type == "door" else 0.75
            ratio_score = max(0.0, 1.0 - abs(ratio - ratio_center))
            non_orthogonality = min(angle, abs(90.0 - angle))
            angle_score = min(
                1.0,
                non_orthogonality / 30.0,
            )
            geometry_score = (
                0.40 * hinge_score
                + 0.25 * ratio_score
                + 0.35 * max(0.0, angle_score)
            )
            score = 0.72 * geometry_score + 0.28 * float(vertical_match)
            # An articulated panel is often scanned at a clearly
            # non-structural angle (e.g. 30-60 degrees).  Because this stage
            # only prepares a PNG proposal, a strong hinge+jamb match at such
            # an angle may be suggested even when background points contaminate
            # the vertical profile.  Orthogonal/full-height partitions remain
            # review-only and are never silently removed.
            suppression_proposal = (
                geometry_score >= 0.55
                and (
                    vertical_match
                    or non_orthogonality >= 18.0
                )
            )
            candidate = {
                "wall_id": str(identifier),
                "wall_index": int(index),
                "opening_id": str(anchor.get("id", "")),
                "host_wall": host_id,
                "type": opening_type,
                "hinge": [float(value) for value in hinge],
                "free_edge": [float(value) for value in free],
                "matched_jamb": [float(value) for value in jamb],
                "hinge_distance": float(hinge_distance),
                "length": float(frame.length),
                "opening_width": float(width),
                "length_width_ratio": float(ratio),
                "thickness": float(thickness),
                "open_angle_deg": float(angle),
                "non_orthogonality_deg": float(non_orthogonality),
                "free_edge_clearance": float(nearest_other),
                "vertical_match": bool(vertical_match),
                "profile": profile,
                "geometry_score": float(geometry_score),
                "score": float(score),
                "status": "proposed" if suppression_proposal else "review",
                "suppress": bool(suppression_proposal),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        if best is not None:
            results.append(best)
            continue

        if not allow_unanchored or frame.length > float(
                unanchored_maximum_length):
            continue
        if wall_labels is not None and str(wall_labels[index]).lower() != "interior":
            continue

        # The opening raster can be incomplete while the hinge remains clear in
        # plan.  A leaf has a compact axis at a non-structural angle to a much
        # longer host.  Its outer edge is either free, or it floats close to an
        # opposite jamb without forming precise wall-to-wall topology.
        for host_index, host_frame in enumerate(frames):
            if host_index == index:
                continue
            if host_frame.length < max(1.50, 1.50 * frame.length):
                continue
            angle = angle_difference_deg(frame, host_frame)
            non_orthogonality = min(angle, abs(90.0 - angle))
            if non_orthogonality < 20.0:
                continue
            endpoints = np.asarray(frame.segment, dtype=float)
            host_distances = [
                _point_to_segment_distance(endpoint, host_frame.segment)
                for endpoint in endpoints
            ]
            hinge_index = int(np.argmin(host_distances))
            hinge_distance = float(host_distances[hinge_index])
            if hinge_distance > float(unanchored_hinge_tolerance):
                continue
            hinge = endpoints[hinge_index]
            free = endpoints[1 - hinge_index]

            endpoint_connections = []
            for endpoint in endpoints:
                distances = [
                    _point_to_segment_distance(endpoint, other_frame.segment)
                    for other_index, other_frame in enumerate(frames)
                    if other_index != index
                ]
                endpoint_connections.append(min(distances, default=float("inf")))
            other_distances = [
                _point_to_segment_distance(free, other_frame.segment)
                for other_index, other_frame in enumerate(frames)
                if other_index not in {index, host_index}
            ]
            nearest_other = min(other_distances, default=float("inf"))
            free_edge = nearest_other >= float(minimum_free_clearance)
            floating_between_jambs = (
                max(endpoint_connections) <= float(unanchored_hinge_tolerance)
                and min(endpoint_connections) > float(exact_connection_tolerance)
            )
            profile = _leaf_vertical_profile(
                points, hinge, free, thickness, z_floor, z_ceiling)
            coverage = profile.get("layer_coverage") or []
            lower_layers = coverage[:2]
            lower_layer_support = (
                max(lower_layers) if lower_layers else 1.0
            )
            attached_both_ends = (
                max(endpoint_connections) <= float(exact_connection_tolerance)
            )
            window_leaf_between_hosts = (
                attached_both_ends
                and non_orthogonality >= 25.0
                and profile.get("bottom_offset") is not None
                and float(profile["bottom_offset"]) >= 0.20
                and float(lower_layer_support) <= 0.25
            )
            if not (
                free_edge
                or floating_between_jambs
                or window_leaf_between_hosts
            ):
                continue
            hinge_score = max(
                0.0,
                1.0 - hinge_distance /
                max(float(unanchored_hinge_tolerance), _EPS),
            )
            length_score = max(
                0.0, 1.0 - abs(frame.length - 0.85) / 0.85)
            angle_score = min(1.0, non_orthogonality / 30.0)
            geometry_score = (
                0.35 * hinge_score
                + 0.25 * length_score
                + 0.40 * angle_score
            )
            suppression_proposal = geometry_score >= (
                0.64 if window_leaf_between_hosts else 0.62
            )
            candidate = {
                "wall_id": str(identifier),
                "wall_index": int(index),
                "opening_id": "",
                "host_wall": str(wall_ids[host_index]),
                "type": "articulated_panel",
                "source": (
                    "window_leaf_between_hosts"
                    if window_leaf_between_hosts
                    else (
                        "floating_hinged_panel"
                        if floating_between_jambs and not free_edge
                        else "wall_hinge_geometry"
                    )
                ),
                "hinge": [float(value) for value in hinge],
                "free_edge": [float(value) for value in free],
                "matched_jamb": None,
                "hinge_distance": hinge_distance,
                "length": float(frame.length),
                "opening_width": None,
                "length_width_ratio": None,
                "thickness": float(thickness),
                "open_angle_deg": float(angle),
                "non_orthogonality_deg": float(non_orthogonality),
                "free_edge_clearance": float(nearest_other),
                "vertical_match": bool(window_leaf_between_hosts),
                "profile": profile,
                "geometry_score": float(geometry_score),
                "score": float(geometry_score),
                "status": "proposed" if suppression_proposal else "review",
                "suppress": bool(suppression_proposal),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        if best is not None:
            results.append(best)
    return sorted(results, key=lambda result: result["wall_id"])


def keep_non_leaf_wall_indices(count: int, leaf_results):
    """Return wall indices left after approved leaf suggestions are removed."""
    suppressed = {
        int(result["wall_index"])
        for result in leaf_results
        if result.get("suppress")
    }
    return [index for index in range(int(count)) if index not in suppressed]


def keep_quality_wall_indices(thicknesses, wall_labels, diagnostics, *,
                              wall_axes=None,
                              thick_ratio: float = 2.0,
                              thick_minimum: float = 0.30):
    """Reject double-line candidates with several independent red flags.

    Absolute thickness alone is valid for old masonry.  A candidate is removed
    only when a floor-relative thickness outlier also has weak two-face
    evidence, or when score, coherence and paired persistence are all poor.
    A long facade may legitimately have low upper-run coverage because of
    windows, so thickness outlier rejection is limited to compact candidates.
    Conversely, a compact exterior appendage with uniformly weak paired-face
    evidence is rejected by the same quality contract as an interior object.
    """
    if not (len(thicknesses) == len(wall_labels) == len(diagnostics)):
        raise ValueError("wall quality inputs must have equal size")
    if wall_axes is not None and len(wall_axes) != len(thicknesses):
        raise ValueError("wall axes and quality inputs must have equal size")
    interior_thicknesses = [
        float(thickness)
        for thickness, label in zip(thicknesses, wall_labels)
        if str(label).lower() == "interior"
        and np.isfinite(float(thickness))
        and float(thickness) > 0.0
    ]
    typical = float(np.median(interior_thicknesses)) \
        if interior_thicknesses else 0.0
    kept = []
    decisions = []
    for index, (thickness, label, metrics) in enumerate(zip(
            thicknesses, wall_labels, diagnostics)):
        thickness = float(thickness)
        length = (
            float(np.linalg.norm(
                np.asarray(wall_axes[index], dtype=float)[1]
                - np.asarray(wall_axes[index], dtype=float)[0]
            ))
            if wall_axes is not None else 0.0
        )
        compact_candidate = wall_axes is None or length <= 4.0
        score = float(metrics.get("detection_score", 0.0))
        coherence = float(metrics.get("pair_coherence", 0.0))
        paired = float(metrics.get("paired_persistent_coverage", 0.0))
        upper_run = float(metrics.get("upper_run_coverage", 0.0))
        thick_outlier = (
            str(label).lower() == "interior"
            and compact_candidate
            and typical > 0.0
            and thickness >= max(
                float(thick_minimum), float(thick_ratio) * typical)
            and score < 0.75
            and coherence < 0.65
            and paired < 0.50
            and upper_run < 0.35
        )
        weak_pair = (
            compact_candidate
            and score < 0.50
            and coherence < 0.40
            and paired < 0.15
        )
        if thick_outlier or weak_pair:
            decisions.append({
                "wall_index": int(index),
                "reason": (
                    "thickness_outlier_with_weak_faces"
                    if thick_outlier else "weak_two_face_evidence"
                ),
                "typical_thickness": typical,
                "length": length,
                "wall_label": str(label),
                "thickness": thickness,
                "detection_score": score,
                "pair_coherence": coherence,
                "paired_persistent_coverage": paired,
                "upper_run_coverage": upper_run,
            })
        else:
            kept.append(index)
    return kept, decisions


def deduplicate_overlapping_wall_axes(axes, thicknesses, quality_scores=None, *,
                                      angle_tolerance: float = 3.0,
                                      minimum_overlap_ratio: float = 0.60,
                                      minimum_overlap: float = 0.30,
                                      clearance: float = 0.05):
    """Return indices of unique wall volumes, preferring stronger evidence.

    Face pairing is one-to-one, but two independent pairs can still describe
    the same physical wall.  This happens especially when a slab-derived
    facade face pairs once with the true wall and again with a nearby contour.
    Two candidates are duplicates only when their axes are parallel, overlap
    substantially along their length and their thickness ribbons overlap.
    """
    frames = []
    for axis in axes:
        try:
            frames.append(segment_frame(axis))
        except ValueError:
            frames.append(None)
    thicknesses = [float(value) for value in thicknesses]
    if quality_scores is None:
        quality_scores = [0.0] * len(frames)
    quality_scores = [float(value) for value in quality_scores]

    def duplicates(i, j):
        a, b = frames[i], frames[j]
        if a is None or b is None:
            return False
        if angle_difference_deg(a, b) > float(angle_tolerance):
            return False
        _reference, interval_a, interval_b = _intervals_in_frame(a, b)
        overlap = max(
            0.0,
            min(interval_a[1], interval_b[1])
            - max(interval_a[0], interval_b[0]),
        )
        shorter = min(a.length, b.length)
        if overlap < max(
            float(minimum_overlap),
            float(minimum_overlap_ratio) * shorter,
        ):
            return False
        separation = 0.5 * (
            float(_distance_to_infinite_line(np.asarray(a.segment), b).mean())
            + float(_distance_to_infinite_line(np.asarray(b.segment), a).mean())
        )
        overlap_distance = (
            0.5 * thicknesses[i] + 0.5 * thicknesses[j] + float(clearance)
        )
        return separation <= overlap_distance

    # Select the strongest candidate first.  For an evidence tie, prefer the
    # thinner and then the longer wall: duplicated facade pairs commonly have
    # an inflated thickness because they used the wrong opposite face.
    order = sorted(
        range(len(frames)),
        key=lambda index: (
            quality_scores[index],
            -thicknesses[index],
            frames[index].length if frames[index] is not None else 0.0,
        ),
        reverse=True,
    )
    kept = []
    for candidate in order:
        if frames[candidate] is None:
            continue
        if any(duplicates(candidate, previous) for previous in kept):
            continue
        kept.append(candidate)
    return sorted(kept)


def single_conflicts_with_paired_wall(segment, paired_groups, *,
                                      single_thickness: float,
                                      pixel_size: float,
                                      angle_tolerance: float = 3.0,
                                      minimum_overlap: float = 0.20,
                                      clearance: float = 0.20) -> bool:
    """Whether a one-face candidate is already explained by a two-face wall.

    A third contour beside a confirmed wall often comes from wall-face
    thickness in the raster, furniture touching it or a duplicated contour.
    It must overlap longitudinally, be parallel and lie inside a local buffer
    around the paired wall.  A distant parallel partition is therefore kept.
    """
    try:
        candidate = segment_frame(segment)
    except ValueError:
        return True
    metric_clearance = max(float(clearance), 1.5 * float(pixel_size))
    for pair in paired_groups:
        try:
            axis, paired_thickness = wall_axis_from_pair(pair)
            paired = segment_frame(axis)
        except (ValueError, IndexError, TypeError):
            continue
        if angle_difference_deg(candidate, paired) > angle_tolerance:
            continue
        _ref, candidate_interval, paired_interval = _intervals_in_frame(
            candidate, paired)
        overlap = max(
            0.0,
            min(candidate_interval[1], paired_interval[1]) -
            max(candidate_interval[0], paired_interval[0]),
        )
        shorter = min(candidate.length, paired.length)
        required_overlap = max(
            float(minimum_overlap), min(0.75, 0.25 * shorter))
        if overlap < required_overlap:
            continue
        separation = float(_distance_to_infinite_line(
            np.asarray(candidate.segment), paired).mean())
        exclusion_distance = (
            0.5 * float(paired_thickness) +
            0.5 * float(single_thickness) + metric_clearance
        )
        if separation <= exclusion_distance:
            return True
    return False


def _line_intersection(a, b):
    p, r = np.asarray(a[0], float), np.asarray(a[1], float) - np.asarray(a[0], float)
    q, s = np.asarray(b[0], float), np.asarray(b[1], float) - np.asarray(b[0], float)
    cross = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross) <= _EPS:
        return None
    qp = q - p
    t = float((qp[0] * s[1] - qp[1] * s[0]) / cross)
    return p + t * r


def adjust_axis_intersections(axes, thicknesses, pixel_size: float,
                              maximum_snap: float = 0.15):
    """Snap corners/T-junctions with a local, bounded metric tolerance."""
    result = [[list(map(float, axis[0])), list(map(float, axis[1]))]
              for axis in axes]
    proposals = [[[] for _ in range(2)] for _ in result]
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            intersection = _line_intersection(result[i], result[j])
            if intersection is None:
                continue
            for wall_index in (i, j):
                tolerance = min(
                    maximum_snap,
                    0.5 * float(thicknesses[wall_index]) + 2.0 * float(pixel_size),
                )
                for endpoint in range(2):
                    distance = float(np.linalg.norm(
                        np.asarray(result[wall_index][endpoint]) - intersection))
                    if distance <= tolerance:
                        proposals[wall_index][endpoint].append((distance, intersection.copy()))
    for i in range(len(result)):
        for endpoint in range(2):
            if proposals[i][endpoint]:
                _distance, point = min(proposals[i][endpoint], key=lambda item: item[0])
                result[i][endpoint] = point.tolist()
    return result


def make_single_line_group(segment, thickness: float, grid=None, centroid=None):
    """Create the hidden face on the locally less-supported side."""
    frame = segment_frame(segment)
    plus = [[float(p[0] + thickness * frame.n[0]),
             float(p[1] + thickness * frame.n[1])] for p in frame.segment]
    minus = [[float(p[0] - thickness * frame.n[0]),
              float(p[1] - thickness * frame.n[1])] for p in frame.segment]
    plus_support = float(_grid_signature(plus, grid).mean()) if grid else 0.0
    minus_support = float(_grid_signature(minus, grid).mean()) if grid else 0.0
    if abs(plus_support - minus_support) > 0.02:
        synthetic = plus if plus_support < minus_support else minus
    else:
        # Ambiguous local evidence: retain the old centroid heuristic only as
        # a deterministic fallback, not as the primary decision rule.
        center = np.mean(np.asarray(frame.segment), axis=0)
        target = np.asarray(centroid if centroid is not None else (0.0, 0.0))
        synthetic = plus if np.dot(frame.n, center - target) >= 0 else minus
    return [frame.segment, synthetic]
