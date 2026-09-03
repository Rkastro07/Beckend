"""Dependency-light 2D geometry used by the BIM revision engine.

The editing layer deliberately avoids depending on a language model or a
geometry service.  The functions in this module are deterministic and operate
in metres.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable, Sequence


EPS = 1e-9
Point = tuple[float, float]
Segment = tuple[Point, Point]


def point(value: Sequence[float]) -> Point:
    if len(value) < 2:
        raise ValueError("um ponto precisa de duas coordenadas")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(component) for component in result):
        raise ValueError("coordenada não finita")
    return result


def add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def scale(a: Point, factor: float) -> Point:
    return a[0] * factor, a[1] * factor


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def norm(vector: Point) -> float:
    return math.hypot(vector[0], vector[1])


def distance(a: Point, b: Point) -> float:
    return norm(sub(a, b))


def unit(vector: Point) -> Point:
    length = norm(vector)
    if length <= EPS:
        raise ValueError("vetor degenerado")
    return vector[0] / length, vector[1] / length


def left_normal(vector: Point) -> Point:
    direction = unit(vector)
    return -direction[1], direction[0]


def segment_length(segment: Segment) -> float:
    return distance(segment[0], segment[1])


def line_parameters(a: Segment, b: Segment) -> tuple[Point, float, float] | None:
    """Return the infinite-line intersection and parameters on A and B."""
    p, r = a[0], sub(a[1], a[0])
    q, s = b[0], sub(b[1], b[0])
    denominator = cross(r, s)
    if abs(denominator) <= EPS:
        return None
    qp = sub(q, p)
    ta = cross(qp, s) / denominator
    tb = cross(qp, r) / denominator
    return add(p, scale(r, ta)), ta, tb


def line_intersection(a: Segment, b: Segment) -> Point | None:
    result = line_parameters(a, b)
    return result[0] if result is not None else None


def segment_intersection(
    a: Segment,
    b: Segment,
    tolerance: float = 1e-8,
) -> tuple[Point, float, float] | None:
    result = line_parameters(a, b)
    if result is None:
        return None
    where, ta, tb = result
    if -tolerance <= ta <= 1.0 + tolerance and -tolerance <= tb <= 1.0 + tolerance:
        return where, min(1.0, max(0.0, ta)), min(1.0, max(0.0, tb))
    return None


def project_to_line(value: Point, segment: Segment, clamp: bool = False) -> Point:
    vector = sub(segment[1], segment[0])
    denominator = dot(vector, vector)
    if denominator <= EPS:
        return segment[0]
    parameter = dot(sub(value, segment[0]), vector) / denominator
    if clamp:
        parameter = min(1.0, max(0.0, parameter))
    return add(segment[0], scale(vector, parameter))


def parameter_on_line(value: Point, segment: Segment) -> float:
    vector = sub(segment[1], segment[0])
    denominator = dot(vector, vector)
    if denominator <= EPS:
        return 0.0
    return dot(sub(value, segment[0]), vector) / math.sqrt(denominator)


def point_on_segment(value: Point, segment: Segment, tolerance: float = 1e-8) -> bool:
    projection = project_to_line(value, segment, clamp=True)
    return distance(value, projection) <= tolerance


def wall_axis(wall: dict) -> Segment:
    return (
        (float(wall["ax"]), float(wall["ay"])),
        (float(wall["bx"]), float(wall["by"])),
    )


def wall_arc_geometry(wall: dict) -> dict | None:
    curve = wall.get("curva")
    if wall.get("geometria") != "arco" or not isinstance(curve, dict):
        return None
    ax, ay = float(wall["ax"]), float(wall["ay"])
    bx, by = float(wall["bx"]), float(wall["by"])
    cx, cy = float(curve["x"]), float(curve["y"])
    denominator = 2.0 * (
        ax * (cy - by) + cx * (by - ay) + bx * (ay - cy)
    )
    chord = math.hypot(bx - ax, by - ay)
    if chord <= EPS or abs(denominator) < chord * chord * 1e-7:
        return None
    a2, c2, b2 = ax * ax + ay * ay, cx * cx + cy * cy, bx * bx + by * by
    ox = (a2 * (cy - by) + c2 * (by - ay) + b2 * (ay - cy)) / denominator
    oy = (a2 * (bx - cx) + c2 * (ax - bx) + b2 * (cx - ax)) / denominator
    radius = math.hypot(ax - ox, ay - oy)
    start = math.atan2(ay - oy, ax - ox)
    control = math.atan2(cy - oy, cx - ox)
    end = math.atan2(by - oy, bx - ox)
    ccw_sweep = (end - start) % (2.0 * math.pi)
    ccw_control = (control - start) % (2.0 * math.pi)
    sweep = ccw_sweep if ccw_control <= ccw_sweep + 1e-7 else ccw_sweep - 2.0 * math.pi
    length = abs(sweep) * radius
    if not math.isfinite(length) or length <= EPS:
        return None
    return {
        "center": (ox, oy),
        "radius": radius,
        "start": start,
        "sweep": sweep,
        "length": length,
    }


def wall_length(wall: dict) -> float:
    arc = wall_arc_geometry(wall)
    return float(arc["length"] if arc else distance(*wall_axis(wall)))


def wall_frame(wall: dict, longitudinal: float) -> tuple[Point, Point, Point]:
    arc = wall_arc_geometry(wall)
    length = wall_length(wall)
    longitudinal = max(0.0, min(length, float(longitudinal)))
    if not arc:
        a, b = wall_axis(wall)
        tangent = unit(sub(b, a))
        value = add(a, scale(tangent, longitudinal))
    else:
        angle = arc["start"] + arc["sweep"] * longitudinal / length
        direction = 1.0 if arc["sweep"] >= 0 else -1.0
        tangent = (-math.sin(angle) * direction, math.cos(angle) * direction)
        value = (
            arc["center"][0] + arc["radius"] * math.cos(angle),
            arc["center"][1] + arc["radius"] * math.sin(angle),
        )
    normal = (-tangent[1], tangent[0])
    return value, tangent, normal


def wall_axis_segments(wall: dict, max_segment: float = 0.12) -> list[Segment]:
    arc = wall_arc_geometry(wall)
    if not arc:
        return [wall_axis(wall)]
    count = max(2, min(512, int(math.ceil(arc["length"] / max_segment))))
    points = [wall_frame(wall, arc["length"] * index / count)[0]
              for index in range(count + 1)]
    return list(zip(points, points[1:]))


def wall_corners(wall: dict) -> list[Point]:
    if wall_arc_geometry(wall):
        half = float(wall.get("espessura", 0.15)) * 0.5
        frames = [wall_frame(wall, wall_length(wall) * index / max(
            2, min(512, int(math.ceil(wall_length(wall) / 0.12)))
        )) for index in range(max(
            2, min(512, int(math.ceil(wall_length(wall) / 0.12)))
        ) + 1)]
        return [
            add(value, scale(normal, side * half))
            for value, _tangent, normal in frames
            for side in (-1.0, 1.0)
        ]
    a, b = wall_axis(wall)
    normal = left_normal(sub(b, a))
    half = float(wall.get("espessura", 0.15)) * 0.5
    offset = scale(normal, half)
    return [add(a, offset), add(b, offset), sub(b, offset), sub(a, offset)]


def polygon_area(vertices: Sequence[Point]) -> float:
    if len(vertices) < 3:
        return 0.0
    return 0.5 * sum(
        cross(vertices[index], vertices[(index + 1) % len(vertices)])
        for index in range(len(vertices))
    )


def polygon_perimeter(vertices: Sequence[Point]) -> float:
    return sum(
        distance(vertices[index], vertices[(index + 1) % len(vertices)])
        for index in range(len(vertices))
    ) if len(vertices) >= 2 else 0.0


def convex_hull(values: Iterable[Point]) -> list[Point]:
    points = sorted(set(point(value) for value in values))
    if len(points) <= 1:
        return points

    def half(sequence):
        result: list[Point] = []
        for candidate in sequence:
            while len(result) >= 2 and cross(
                sub(result[-1], result[-2]),
                sub(candidate, result[-1]),
            ) <= EPS:
                result.pop()
            result.append(candidate)
        return result

    lower = half(points)
    upper = half(reversed(points))
    return lower[:-1] + upper[:-1]


def point_in_polygon(value: Point, vertices: Sequence[Point], tolerance: float = 1e-8) -> bool:
    if len(vertices) < 3:
        return False
    for index in range(len(vertices)):
        edge = (vertices[index], vertices[(index + 1) % len(vertices)])
        if point_on_segment(value, edge, tolerance=tolerance):
            return True
    inside = False
    x, y = value
    previous = vertices[-1]
    for current in vertices:
        x1, y1 = previous
        x2, y2 = current
        if ((y1 > y) != (y2 > y)):
            hit_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < hit_x:
                inside = not inside
        previous = current
    return inside


class _NodeStore:
    def __init__(self, tolerance: float):
        self.tolerance = max(float(tolerance), 1e-7)
        self.points: list[Point] = []
        self.grid: dict[tuple[int, int], list[int]] = defaultdict(list)

    def _cell(self, value: Point) -> tuple[int, int]:
        return (
            int(math.floor(value[0] / self.tolerance)),
            int(math.floor(value[1] / self.tolerance)),
        )

    def add(self, value: Point) -> int:
        cell = self._cell(value)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for index in self.grid.get((cell[0] + dx, cell[1] + dy), ()):
                    if distance(self.points[index], value) <= self.tolerance:
                        return index
        index = len(self.points)
        self.points.append(value)
        self.grid[cell].append(index)
        return index


def _clean_face(vertices: Sequence[Point], tolerance: float) -> list[Point]:
    cleaned: list[Point] = []
    for value in vertices:
        if not cleaned or distance(cleaned[-1], value) > tolerance:
            cleaned.append(value)
    if len(cleaned) > 1 and distance(cleaned[0], cleaned[-1]) <= tolerance:
        cleaned.pop()
    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        result: list[Point] = []
        for index, current in enumerate(cleaned):
            before = cleaned[index - 1]
            after = cleaned[(index + 1) % len(cleaned)]
            if abs(cross(sub(current, before), sub(after, current))) <= tolerance * 0.1:
                changed = True
                continue
            result.append(current)
        cleaned = result
    return cleaned


def planar_faces(
    walls: Sequence[dict],
    snap_tolerance: float = 0.03,
    min_area: float = 0.5,
) -> tuple[list[list[Point]], dict]:
    """Polygonize wall axes with a small, graph-only endpoint tolerance.

    Wall coordinates are never mutated.  The tolerance only decides whether
    graph nodes represent the same junction.
    """
    segments = [
        (str(wall["id"]), segment)
        for wall in walls
        for segment in wall_axis_segments(wall)
    ]
    split_parameters = [[0.0, 1.0] for _ in segments]
    intersection_count = 0

    for left in range(len(segments)):
        _, segment_a = segments[left]
        for right in range(left + 1, len(segments)):
            _, segment_b = segments[right]
            hit = segment_intersection(segment_a, segment_b)
            if hit is not None:
                _, ta, tb = hit
                split_parameters[left].append(ta)
                split_parameters[right].append(tb)
                intersection_count += 1
                continue
            # T-junctions in scans can be numerically near-collinear.
            for value, owner, target in (
                (segment_a[0], left, right),
                (segment_a[1], left, right),
                (segment_b[0], right, left),
                (segment_b[1], right, left),
            ):
                target_segment = segments[target][1]
                if point_on_segment(value, target_segment, tolerance=snap_tolerance):
                    vector = sub(target_segment[1], target_segment[0])
                    denominator = dot(vector, vector)
                    if denominator > EPS:
                        parameter = dot(sub(value, target_segment[0]), vector) / denominator
                        if snap_tolerance < parameter * norm(vector) < norm(vector) - snap_tolerance:
                            split_parameters[target].append(min(1.0, max(0.0, parameter)))

    nodes = _NodeStore(snap_tolerance)
    edges: set[tuple[int, int]] = set()
    edge_walls: dict[tuple[int, int], set[str]] = defaultdict(set)
    for (wall_id, segment), parameters in zip(segments, split_parameters):
        values = sorted(set(round(value, 10) for value in parameters))
        vector = sub(segment[1], segment[0])
        for start, end in zip(values, values[1:]):
            if end - start <= EPS:
                continue
            a = add(segment[0], scale(vector, start))
            b = add(segment[0], scale(vector, end))
            ia, ib = nodes.add(a), nodes.add(b)
            if ia == ib:
                continue
            edge = (min(ia, ib), max(ia, ib))
            edges.add(edge)
            edge_walls[edge].add(wall_id)

    adjacency: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    ordered = {
        index: sorted(
            neighbours,
            key=lambda neighbour: math.atan2(
                nodes.points[neighbour][1] - nodes.points[index][1],
                nodes.points[neighbour][0] - nodes.points[index][0],
            ),
        )
        for index, neighbours in adjacency.items()
    }

    used: set[tuple[int, int]] = set()
    faces: list[list[Point]] = []
    for edge in sorted(edges):
        for start in (edge, (edge[1], edge[0])):
            if start in used:
                continue
            current = start
            face_indices: list[int] = []
            local: set[tuple[int, int]] = set()
            closed = False
            for _ in range(max(8, len(edges) * 2 + 4)):
                if current in local:
                    closed = current == start
                    break
                local.add(current)
                used.add(current)
                source, target = current
                face_indices.append(source)
                neighbours = ordered.get(target, [])
                if source not in neighbours or not neighbours:
                    break
                incoming_index = neighbours.index(source)
                following = neighbours[(incoming_index - 1) % len(neighbours)]
                current = (target, following)
                if current == start:
                    closed = True
                    break
            if not closed:
                continue
            vertices = _clean_face(
                [nodes.points[index] for index in face_indices],
                snap_tolerance,
            )
            area = polygon_area(vertices)
            if len(vertices) >= 3 and area >= float(min_area):
                faces.append(vertices)

    topology = {
        "node_count": len(nodes.points),
        "edge_count": len(edges),
        "intersection_count": intersection_count,
        "endpoint_nodes": [
            {"x": nodes.points[index][0], "y": nodes.points[index][1]}
            for index, neighbours in adjacency.items()
            if len(neighbours) == 1
        ],
        "junction_nodes": [
            {
                "x": nodes.points[index][0],
                "y": nodes.points[index][1],
                "degree": len(neighbours),
            }
            for index, neighbours in adjacency.items()
            if len(neighbours) >= 3
        ],
        "closed_face_count": len(faces),
        "snap_tolerance": float(snap_tolerance),
    }
    return faces, topology
