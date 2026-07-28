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


def face_pair_vertical_metrics(pair, grid, *, corridor_pixels: int = 1,
                               persistent_slices: int = 4):
    """Measure height persistence on each proposed wall face independently.

    The old final check counted all points assigned to a broad wall corridor.
    A false parallel axis could therefore borrow support from a nearby real
    wall, and furniture visible in only the lower half could satisfy a simple
    "3 of 6 slices" count.  Here every face retains a longitudinal hit matrix:
    rows are height slices and columns are positions along the face.
    """
    if len(pair) != 2:
        raise ValueError("a wall candidate must contain exactly two faces")
    face_hits = [
        _grid_hits(face, grid, corridor_pixels=corridor_pixels)
        for face in pair
    ]
    n_slices = max((hits.shape[0] for hits in face_hits), default=0)
    if n_slices == 0:
        return {
            "accepted_face": -1,
            "face_coverages": np.zeros((2, 0), dtype=float),
            "bottom_coverage": 0.0,
            "top_coverage": 0.0,
            "persistent_coverage": 0.0,
            "score": 0.0,
        }

    coverages = []
    persistences = []
    for hits in face_hits:
        if hits.shape[1] == 0:
            coverages.append(np.zeros(n_slices, dtype=float))
            persistences.append(0.0)
            continue
        coverage = hits.mean(axis=1)
        coverages.append(coverage)
        required = min(max(2, int(persistent_slices)), hits.shape[0])
        persistences.append(float((hits.sum(axis=0) >= required).mean()))
    coverages = np.asarray(coverages, dtype=float)
    edge_count = max(1, min(2, n_slices // 2))
    bottom = coverages[:, :edge_count].mean(axis=1)
    top = coverages[:, -edge_count:].mean(axis=1)
    # One coherent observed face is enough for an exterior or single-line wall;
    # its synthetic hidden face is expected to have little or no cloud support.
    face_scores = 0.40 * top + 0.25 * bottom + 0.35 * np.asarray(persistences)
    accepted_face = int(np.argmax(face_scores))
    return {
        "accepted_face": accepted_face,
        "face_coverages": coverages,
        "bottom_coverage": float(bottom[accepted_face]),
        "top_coverage": float(top[accepted_face]),
        "persistent_coverage": float(persistences[accepted_face]),
        "score": float(face_scores[accepted_face]),
    }


def wall_pair_has_vertical_support(pair, grid, *, corridor_pixels: int = 1,
                                   persistent_slices: int = 4,
                                   minimum_bottom_coverage: float = 0.12,
                                   minimum_top_coverage: float = 0.25,
                                   minimum_persistent_coverage: float = 0.10):
    """Return whether one physical face behaves like a vertically tall wall."""
    metrics = face_pair_vertical_metrics(
        pair,
        grid,
        corridor_pixels=corridor_pixels,
        persistent_slices=persistent_slices,
    )
    accepted = (
        metrics["bottom_coverage"] >= float(minimum_bottom_coverage)
        and metrics["top_coverage"] >= float(minimum_top_coverage)
        and metrics["persistent_coverage"] >= float(minimum_persistent_coverage)
    )
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
    candidates.sort(reverse=True, key=lambda item: item[0])
    used = set()
    pairs = []
    diagnostics = []
    for _score, i, j, candidate in candidates:
        if i in used or j in used:
            continue
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
    return RefinementContext(points=array, tree=cKDTree(array[:, :2]))


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
        minimum_length: float = 0.45,
        maximum_length: float = 2.10,
        maximum_thickness: float = 0.23,
        hinge_tolerance: float = 0.30,
        minimum_open_angle: float = 15.0,
        minimum_free_clearance: float = 0.24):
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
    frames = [segment_frame(axis) for axis in axes]
    wall_index = {str(identifier): index
                  for index, identifier in enumerate(wall_ids)}
    hosted_openings = {
        str(anchor.get("host_wall"))
        for anchor in opening_anchors
        if anchor.get("host_wall")
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
    return sorted(results, key=lambda result: result["wall_id"])


def keep_non_leaf_wall_indices(count: int, leaf_results):
    """Return wall indices left after approved leaf suggestions are removed."""
    suppressed = {
        int(result["wall_index"])
        for result in leaf_results
        if result.get("suppress")
    }
    return [index for index in range(int(count)) if index not in suppressed]


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
