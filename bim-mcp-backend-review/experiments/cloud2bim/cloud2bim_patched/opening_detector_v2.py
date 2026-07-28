"""Wall-local opening proposals from a rectified X-Z occupancy grid.

The detector never authors IFC entities.  It returns reviewable candidates with
stable IDs, evidence metrics and the host-wall offset expected by the existing
Cloud-to-BIM manual-opening contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter, gaussian_filter1d, label, maximum_filter1d
from scipy.signal import find_peaks


@dataclass(frozen=True)
class OpeningCandidate:
    id: str
    host_wall: str
    type: str
    s_center: float
    width: float
    z_min: float
    z_max: float
    height: float
    confidence: float
    status: str
    evidence: dict[str, float | int | bool | str]

    def to_dict(self):
        return asdict(self)


@dataclass
class WallOpeningResult:
    wall_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float
    floor_z: float
    ceiling_z: float
    point_count: int
    grid_cell: float
    candidates: list[OpeningCandidate]
    counts: np.ndarray
    x_edges: np.ndarray
    z_edges: np.ndarray

    def to_dict(self):
        return {
            "wall_id": self.wall_id,
            "start": list(self.start),
            "end": list(self.end),
            "thickness": self.thickness,
            "floor_z": self.floor_z,
            "ceiling_z": self.ceiling_z,
            "point_count": self.point_count,
            "grid_cell": self.grid_cell,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class TopologyOpeningCandidate:
    id: str
    host_wall: str
    type: str
    between: tuple[str, str]
    s_center: float
    global_center: tuple[float, float]
    width: float
    z_min: float
    z_max: float
    height: float
    confidence: float
    status: str
    evidence: dict[str, float | int | bool | str]

    def to_dict(self):
        payload = asdict(self)
        payload["between"] = list(self.between)
        payload["global_center"] = list(self.global_center)
        return payload


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    tagged, count = label(np.asarray(mask, dtype=bool))
    result = []
    for tag in range(1, count + 1):
        indices = np.flatnonzero(tagged == tag)
        if len(indices):
            result.append((int(indices[0]), int(indices[-1]) + 1))
    return result


def _rectified_points(points, start, end, thickness, floor_z, ceiling_z, face_band):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 0.05:
        return np.empty((0, 3)), length
    u = direction / length
    normal = np.asarray([-u[1], u[0]])
    rel = points[:, :2] - start
    s = rel @ u
    d = rel @ normal
    z = points[:, 2] - float(floor_z)
    corridor = 0.5 * max(float(thickness), 0.10) + float(face_band)
    mask = ((s >= 0.0) & (s <= length) & (np.abs(d) <= corridor)
            & (z >= 0.0) & (points[:, 2] <= float(ceiling_z)))
    return np.column_stack([s[mask], d[mask], z[mask]]), length


def _face_deficit(local, s0, s1, reference_width, thickness):
    values = []
    half = 0.5 * max(float(thickness), 0.10)
    for sign in (-1.0, 1.0):
        face = local[np.abs(local[:, 1] - sign * half) <= 0.12]
        if len(face) < 80:
            continue
        inside = np.count_nonzero((face[:, 0] >= s0) & (face[:, 0] <= s1)) / max(s1 - s0, 0.05)
        ref = np.count_nonzero(
            ((face[:, 0] >= s0-reference_width) & (face[:, 0] < s0))
            | ((face[:, 0] > s1) & (face[:, 0] <= s1+reference_width))) / max(2*reference_width, 0.05)
        if ref > 0:
            values.append(float(np.clip(1.0-inside/ref, 0.0, 1.0)))
    return values


def _frame_window_proposals(
    counts: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    existing: list[dict],
    grid_cell: float,
) -> list[dict]:
    """Find repeated framed windows that do not look like empty wall voids.

    A framed window commonly produces three long vertical ridges (left jamb,
    mullion and right jamb) plus a sill and a header.  This complements the
    deficit detector when glass, furniture or a second surface is visible
    through the opening.  Requiring a repeated family keeps isolated wall
    edges and door frames out of the automatic proposal set.
    """
    if any(item["type"] == "door" and item["confidence"] >= .60 for item in existing):
        return []
    if sum(item["type"] == "window" and item["confidence"] >= .60 for item in existing) >= 2:
        return []
    if counts.size == 0 or counts.shape[0] < 8 or counts.shape[1] < 12:
        return []
    image = gaussian_filter(np.log1p(counts.astype(float)), sigma=(.7, .7))
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    wall_height = float(z_edges[-1] - z_edges[0])
    vertical_band = (
        (z_centers >= max(.55, .22 * wall_height))
        & (z_centers <= min(wall_height - .05, .98 * wall_height))
    )
    if np.count_nonzero(vertical_band) < 5:
        return []
    vertical_profile = gaussian_filter1d(
        np.quantile(image[vertical_band], .70, axis=0)
        + .40 * np.mean(image[vertical_band], axis=0),
        sigma=1,
    )
    peak_indices, peak_data = find_peaks(
        vertical_profile,
        distance=max(2, int(round(.12 / grid_cell))),
        prominence=max(.05, .03 * float(np.std(vertical_profile))),
    )
    prominence = {
        int(index): float(peak_data["prominences"][offset])
        for offset, index in enumerate(peak_indices)
    }
    peak_indices = [
        int(index) for index in peak_indices
        if .25 <= x_centers[index] <= x_edges[-1] - .25
    ]
    seeds = []
    for middle_offset, middle in enumerate(peak_indices):
        if prominence.get(middle, 0.0) < .16:
            continue
        options = []
        for left in peak_indices[:middle_offset]:
            left_gap = float(x_centers[middle] - x_centers[left])
            if not .30 <= left_gap <= .72:
                continue
            for right in peak_indices[middle_offset + 1:]:
                right_gap = float(x_centers[right] - x_centers[middle])
                if not .30 <= right_gap <= .72:
                    continue
                span = float(x_centers[right] - x_centers[left])
                symmetry = 1.0 - abs(left_gap - right_gap) / max(left_gap, right_gap)
                if .72 <= span <= 1.45 and symmetry >= .62:
                    score = (
                        symmetry
                        + .20 * min(prominence.get(middle, 0.0), 3.0)
                        + .02 * (
                            prominence.get(left, 0.0)
                            + prominence.get(right, 0.0)
                        )
                    )
                    options.append((score, left, right, symmetry))
        if options:
            _, left, right, symmetry = max(options)
            seeds.append((middle, left, right, symmetry))
    chosen = []
    for seed in sorted(seeds, key=lambda value: value[3], reverse=True):
        if all(abs(x_centers[seed[0]] - x_centers[item[0]]) > .70 for item in chosen):
            chosen.append(seed)
    if not chosen:
        return []
    family_width = float(np.median([
        x_centers[right] - x_centers[left]
        for _, left, right, _ in chosen
    ]))
    if not .70 <= family_width <= 1.50:
        return []

    horizontal_profile = gaussian_filter1d(
        np.quantile(image, .75, axis=1) + .40 * np.mean(image, axis=1),
        sigma=1,
    )
    horizontal_peaks, horizontal_data = find_peaks(
        horizontal_profile,
        distance=max(2, int(round(.15 / grid_cell))),
        prominence=.08,
    )
    bottom_options = [
        (float(horizontal_data["prominences"][offset]), int(index))
        for offset, index in enumerate(horizontal_peaks)
        if max(.45, .20 * wall_height) <= z_centers[index] <= min(1.45, .50 * wall_height)
    ]
    top_options = [
        (float(horizontal_data["prominences"][offset]), int(index))
        for offset, index in enumerate(horizontal_peaks)
        if max(1.55, .75 * wall_height) <= z_centers[index] <= wall_height - .03
    ]
    if not bottom_options or not top_options:
        return []
    bottom_index = max(bottom_options)[1]
    top_index = max(top_options)[1]
    z_min = float(z_centers[bottom_index])
    z_max = float(z_centers[top_index])
    if z_max - z_min < 1.00:
        return []

    centers = [middle for middle, _, _, _ in chosen]
    weak_window_intervals = [
        (float(item["s0"]), float(item["s1"]))
        for item in existing
        if item["type"] == "window"
    ]
    strong_prominence = max(.35, .20 * float(np.std(vertical_profile)))
    for middle in peak_indices:
        if prominence.get(middle, 0.0) < strong_prominence:
            continue
        if any(abs(x_centers[middle] - x_centers[item]) < .75 * family_width for item in centers):
            continue
        if not any(
                start - .75 <= x_centers[middle] <= end + .75
                for start, end in weak_window_intervals):
            continue
        i0 = int(np.searchsorted(x_centers, x_centers[middle] - family_width / 2))
        i1 = int(np.searchsorted(x_centers, x_centers[middle] + family_width / 2))
        band = image[:, max(0, i0):min(len(x_centers), i1 + 1)]
        if band.shape[1] < 3:
            continue
        local_horizontal = gaussian_filter1d(
            np.quantile(band, .75, axis=1) + .40 * np.mean(band, axis=1),
            sigma=1,
        )
        baseline = float(np.median(local_horizontal))
        spread = float(np.std(local_horizontal)) + 1e-6
        sill_support = (float(local_horizontal[bottom_index]) - baseline) / spread
        header_support = (float(local_horizontal[top_index]) - baseline) / spread
        if sill_support >= .18 and header_support >= .18:
            centers.append(middle)
    centers = sorted(set(centers), key=lambda index: x_centers[index])
    # Two overlapping frame rectangles cannot be separate openings.  This
    # occurs when the same three-part frame is interpreted once from each
    # side of its central mullion.  Collapse the overlap onto the strongest
    # vertical ridge between the two provisional centres.
    merged_centers = []
    offset = 0
    while offset < len(centers):
        group = [centers[offset]]
        while (
            offset + 1 < len(centers)
            and x_centers[centers[offset + 1]] - x_centers[group[-1]]
                < .98 * family_width
        ):
            offset += 1
            group.append(centers[offset])
        if len(group) == 1:
            merged_centers.append(group[0])
        else:
            between = [
                index for index in peak_indices
                if x_centers[group[0]] <= x_centers[index] <= x_centers[group[-1]]
            ]
            overlap_midpoint = (
                x_centers[group[0]] + x_centers[group[-1]]) / 2
            merged_centers.append(min(
                between or group,
                key=lambda index: abs(x_centers[index] - overlap_midpoint),
            ))
        offset += 1
    centers = sorted(set(merged_centers), key=lambda index: x_centers[index])
    if len(centers) < 2:
        return []

    family_symmetry = float(np.mean([seed[3] for seed in chosen]))
    confidence = float(np.clip(.62 + .14 * family_symmetry + .03 * min(len(centers), 3), 0.0, .90))
    result = []
    for center_index in centers:
        center = float(x_centers[center_index])
        result.append({
            "type": "window",
            "s0": center - family_width / 2,
            "s1": center + family_width / 2,
            "z0": z_min,
            "z1": z_max,
            "horizontal_deficit": 0.0,
            "vertical_deficit": 0.0,
            "face_score": 0.0,
            "observed_faces": 0,
            "touches_floor": False,
            "edge_distance": min(center - family_width / 2, x_edges[-1] - center - family_width / 2),
            "confidence": confidence,
            "evidence_extra": {
                "detector_mode": "repeated_frame_family",
                "frame_family": True,
                "family_size": len(centers),
                "frame_symmetry": round(family_symmetry, 4),
                "sill_z": round(z_min, 4),
                "header_z": round(z_max, 4),
            },
        })
    return result


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def detect_topology_openings(
    walls: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    min_gap: float = .65,
    max_gap: float = 1.45,
    min_angle_degrees: float = 45.0,
    target_tolerance: float = .08,
    alignment_tolerance: float = .98,
    door_height: float = 2.10,
) -> list[TopologyOpeningCandidate]:
    """Detect door-sized gaps where one wall axis stops before another wall."""
    raw = []
    for host_wall in sorted(walls):
        start, end = (np.asarray(value, dtype=float) for value in walls[host_wall])
        host_vector = end - start
        host_length = float(np.linalg.norm(host_vector))
        if host_length <= .05:
            continue
        host_direction = host_vector / host_length
        for endpoint, outward in ((start, -host_direction), (end, host_direction)):
            for target_wall in sorted(walls):
                if target_wall == host_wall:
                    continue
                target_start, target_end = (
                    np.asarray(value, dtype=float) for value in walls[target_wall])
                target_vector = target_end - target_start
                target_length = float(np.linalg.norm(target_vector))
                if target_length <= .05:
                    continue
                denominator = _cross_2d(host_vector, target_vector)
                if abs(denominator) < 1e-8:
                    continue
                delta = target_start - start
                host_parameter = _cross_2d(delta, target_vector) / denominator
                target_parameter = _cross_2d(delta, host_vector) / denominator
                intersection = start + host_parameter * host_vector
                gap_vector = intersection - endpoint
                gap = float(np.linalg.norm(gap_vector))
                if not min_gap <= gap <= max_gap:
                    continue
                alignment = float(np.dot(gap_vector / gap, outward))
                if alignment < alignment_tolerance:
                    continue
                if not -target_tolerance <= target_parameter <= 1.0 + target_tolerance:
                    continue
                target_direction = target_vector / target_length
                angle = float(np.degrees(np.arccos(np.clip(
                    abs(np.dot(host_direction, target_direction)), 0.0, 1.0))))
                if angle < min_angle_degrees:
                    continue
                center = (endpoint + intersection) / 2
                s_center = float(np.dot(center - start, host_direction))
                angle_score = float(np.sin(np.radians(angle)))
                middle = (min_gap + max_gap) / 2
                half_range = max((max_gap - min_gap) / 2, .01)
                range_score = float(np.clip(1.0 - abs(gap - middle) / half_range, 0.0, 1.0))
                confidence = float(np.clip(.62 + .22 * angle_score + .08 * range_score, 0.0, .95))
                raw.append({
                    "host_wall": host_wall,
                    "target_wall": target_wall,
                    "s_center": s_center,
                    "center": center,
                    "gap": gap,
                    "angle": angle,
                    "alignment": alignment,
                    "target_parameter": target_parameter,
                    "confidence": confidence,
                })
    unique = []
    for item in sorted(raw, key=lambda value: (
            value["host_wall"], value["s_center"], value["target_wall"])):
        if any(np.linalg.norm(item["center"] - kept["center"]) < .15 for kept in unique):
            continue
        unique.append(item)
    result = []
    for index, item in enumerate(unique, 1):
        result.append(TopologyOpeningCandidate(
            id=f"D-GAP-{index:02d}",
            host_wall=item["host_wall"],
            type="door",
            between=(item["host_wall"], item["target_wall"]),
            s_center=round(item["s_center"], 4),
            global_center=(
                round(float(item["center"][0]), 4),
                round(float(item["center"][1]), 4),
            ),
            width=round(item["gap"], 4),
            z_min=0.0,
            z_max=round(float(door_height), 4),
            height=round(float(door_height), 4),
            confidence=round(item["confidence"], 4),
            status="proposed" if item["confidence"] >= .60 else "review",
            evidence={
                "detector_mode": "wall_axis_topology",
                "gap_to_intersection": round(item["gap"], 4),
                "intersection_angle_degrees": round(item["angle"], 2),
                "extension_alignment": round(item["alignment"], 4),
                "target_parameter": round(item["target_parameter"], 4),
            },
        ))
    return result


def detect_wall_openings(
    points: np.ndarray,
    *,
    wall_id: str,
    start: Iterable[float],
    end: Iterable[float],
    thickness: float,
    floor_z: float,
    ceiling_z: float,
    grid_cell: float = 0.06,
    face_band: float = 0.12,
    min_width: float = 0.40,
    max_width: float = 3.20,
    min_height: float = 0.60,
    door_floor_tolerance: float = 0.25,
    door_min_height: float = 1.60,
    opening_min_top: float = 1.55,
) -> WallOpeningResult:
    points = np.asarray(points, dtype=float)
    local, length = _rectified_points(
        points, start, end, thickness, floor_z, ceiling_z, face_band)
    height = max(0.05, float(ceiling_z)-float(floor_z))
    nx = max(2, int(np.ceil(length/grid_cell)))
    nz = max(2, int(np.ceil(height/grid_cell)))
    x_edges = np.linspace(0.0, length, nx+1)
    z_edges = np.linspace(0.0, height, nz+1)
    if len(local):
        counts, _, _ = np.histogram2d(local[:, 2], local[:, 0], bins=(z_edges, x_edges))
    else:
        counts = np.zeros((nz, nx), dtype=float)
    density = gaussian_filter(counts.astype(float), sigma=(1.0, 1.0))
    column = density.sum(axis=0)
    window_bins = max(5, int(round(1.20/grid_cell)))
    baseline = maximum_filter1d(column, size=window_bins, mode="nearest")
    ratio = np.divide(column, baseline, out=np.ones_like(column), where=baseline > 1e-9)
    # Windows retain points above and below the void, so their column density
    # commonly stays near 60% of a solid wall.  Use a permissive proposal
    # threshold; the 2-D vertical evidence and confidence score filter it.
    low = ratio < 0.72
    low[:max(1, int(.12/grid_cell))] = False
    low[-max(1, int(.12/grid_cell)):] = False
    low = binary_closing(low, structure=np.ones(max(2, int(.18/grid_cell)), dtype=bool))
    low = binary_opening(low, structure=np.ones(max(1, int(.08/grid_cell)), dtype=bool))

    raw = []
    for i0, i1 in _runs(low):
        s0, s1 = float(x_edges[i0]), float(x_edges[i1])
        width = s1-s0
        if not (min_width <= width <= max_width):
            continue
        side_bins = max(2, int(round(.35/grid_cell)))
        left = density[:, max(0, i0-side_bins):i0]
        right = density[:, i1:min(nx, i1+side_bins)]
        references = [part.mean(axis=1) for part in (left, right) if part.shape[1]]
        if not references:
            continue
        reference = np.maximum.reduce(references)
        inside = density[:, i0:i1].mean(axis=1)
        deficit = np.divide(reference-inside, reference+1e-6)
        vertical = deficit > .42
        vertical[:max(1, int(.03/grid_cell))] = vertical[:max(1, int(.03/grid_cell))]
        vertical = binary_closing(vertical, structure=np.ones(max(2, int(.15/grid_cell)), dtype=bool))
        vertical = binary_opening(vertical, structure=np.ones(max(1, int(.08/grid_cell)), dtype=bool))
        vertical_runs = [(a,b) for a,b in _runs(vertical) if z_edges[b]-z_edges[a] >= min_height]
        if not vertical_runs:
            continue
        z0i, z1i = max(vertical_runs, key=lambda ab: z_edges[ab[1]]-z_edges[ab[0]])
        z0, z1 = float(z_edges[z0i]), float(z_edges[z1i])
        opening_height = z1-z0
        if z1 < opening_min_top:
            continue
        touches_floor = z0 <= door_floor_tolerance
        if touches_floor and opening_height >= door_min_height:
            kind = "door"
        elif z0 >= .30:
            kind = "window"
        else:
            kind = "unknown"
        horizontal_deficit = float(np.clip(1.0-np.mean(ratio[i0:i1]), 0.0, 1.0))
        vertical_deficit = float(np.clip(np.mean(deficit[z0i:z1i]), 0.0, 1.0))
        face_deficits = _face_deficit(local, s0, s1, .35, thickness)
        face_score = float(np.mean(face_deficits)) if face_deficits else 0.0
        plausibility = 1.0 if kind != "unknown" else .45
        confidence = float(np.clip(.38*horizontal_deficit + .38*vertical_deficit + .14*face_score + .10*plausibility, 0.0, 1.0))
        edge_distance = min(s0, length-s1)
        if width < .65:
            confidence *= .65
        if edge_distance < .25:
            confidence *= .45
        raw.append({
            "type": kind, "s0": s0, "s1": s1, "z0": z0, "z1": z1,
            "horizontal_deficit": horizontal_deficit,
            "vertical_deficit": vertical_deficit,
            "face_score": face_score,
            "observed_faces": len(face_deficits),
            "touches_floor": touches_floor,
            "edge_distance": edge_distance,
            "confidence": confidence,
        })
    raw = [
        item for item in raw
        if not (
            item["edge_distance"] < .18
            and item["confidence"] < .35
            and item["z1"] >= height - .15
        )
    ]
    frame_windows = _frame_window_proposals(
        counts, x_edges, z_edges, raw, float(grid_cell))
    if frame_windows:
        raw = [item for item in raw if item["type"] != "window"] + frame_windows
    candidates = []
    counters = {"door": 0, "window": 0, "unknown": 0}
    prefixes = {"door": "D", "window": "J", "unknown": "C"}
    for item in sorted(raw, key=lambda value: value["s0"]):
        kind = item["type"]
        counters[kind] += 1
        candidate_id = f"{prefixes[kind]}-{wall_id}-{counters[kind]:02d}"
        evidence = {
            "touches_floor": bool(item["touches_floor"]),
            "horizontal_deficit": round(item["horizontal_deficit"], 4),
            "vertical_deficit": round(item["vertical_deficit"], 4),
            "two_face_support": round(item["face_score"], 4),
            "observed_faces": int(item["observed_faces"]),
            "edge_distance": round(item["edge_distance"], 4),
        }
        evidence.update(item.get("evidence_extra", {}))
        candidates.append(OpeningCandidate(
            id=candidate_id,
            host_wall=wall_id,
            type=kind,
            s_center=round((item["s0"]+item["s1"])/2, 4),
            width=round(item["s1"]-item["s0"], 4),
            z_min=round(item["z0"], 4),
            z_max=round(item["z1"], 4),
            height=round(item["z1"]-item["z0"], 4),
            confidence=round(item["confidence"], 4),
            status="proposed" if item["confidence"] >= .60 else "review",
            evidence=evidence,
        ))
    return WallOpeningResult(
        wall_id=wall_id,
        start=tuple(map(float, start)), end=tuple(map(float, end)),
        thickness=float(thickness), floor_z=float(floor_z), ceiling_z=float(ceiling_z),
        point_count=int(len(local)), grid_cell=float(grid_cell), candidates=candidates,
        counts=counts, x_edges=x_edges, z_edges=z_edges)


def render_wall_result(result: WallOpeningResult, output: Path):
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 7))
    image = np.log1p(result.counts)
    ax.imshow(image, origin="lower", aspect="auto", cmap="gray_r",
              extent=(result.x_edges[0], result.x_edges[-1], result.z_edges[0], result.z_edges[-1]))
    colors = {"door": "#dc2626", "window": "#16a34a", "unknown": "#f59e0b"}
    for candidate in result.candidates:
        color = colors[candidate.type] if candidate.status == "proposed" else "#f59e0b"
        ax.add_patch(plt.Rectangle(
            (candidate.s_center-candidate.width/2, candidate.z_min),
            candidate.width, candidate.height, fill=False, ec=color, lw=2.5))
        ax.text(candidate.s_center, candidate.z_max+.05,
                f"{candidate.id}  {candidate.confidence:.2f}",
                ha="center", va="bottom", color=color, fontsize=10, fontweight="bold")
    ax.set_title(f"{result.wall_id} — grade retificada X-Z | {len(result.candidates)} candidatos")
    ax.set_xlabel("distância ao longo da parede (m)")
    ax.set_ylabel("altura acima do piso (m)")
    ax.set_ylim(0, max(result.z_edges[-1], .1))
    fig.tight_layout(); fig.savefig(output, dpi=180, bbox_inches="tight"); plt.close(fig)
