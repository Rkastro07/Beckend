"""Recover an editable Scan-to-BIM snapshot from a generated IFC.

The primary recovery path is the JSON sidecar saved beside new IFC outputs.
This module exists for older outputs, created before editor snapshots were
persisted.  It intentionally recovers only the elements understood by the
focused editor: walls, hosted doors/windows, floor/ceiling contours and spaces.
"""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any


def _rounded(value: float) -> float:
    return round(float(value), 4)


def recover_editor_model(ifc_path, *, force_ceiling: bool = False) -> dict[str, Any]:
    """Return ``ModeloPlanta`` and ``PlantaConfig`` reconstructed from an IFC."""

    import ifcopenshell
    import ifcopenshell.geom
    from shapely.geometry import MultiPoint, Polygon
    from shapely.ops import unary_union

    model = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    def shape_data(entity):
        shape = ifcopenshell.geom.create_shape(settings, entity)
        raw = list(shape.geometry.verts)
        points = [
            (float(raw[index]), float(raw[index + 1]), float(raw[index + 2]))
            for index in range(0, len(raw), 3)
        ]
        faces = list(getattr(shape.geometry, "faces", ()))
        return points, faces

    def product_bounds(entity):
        points, _ = shape_data(entity)
        if not points:
            raise ValueError(f"{entity.is_a()} #{entity.id()} sem geometria")
        xs, ys, zs = zip(*points)
        return points, (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

    def footprint(entity):
        points, faces = shape_data(entity)
        triangles = []
        for index in range(0, len(faces), 3):
            try:
                polygon = Polygon([
                    points[faces[index]][:2],
                    points[faces[index + 1]][:2],
                    points[faces[index + 2]][:2],
                ])
            except (IndexError, TypeError):
                continue
            if polygon.is_valid and polygon.area > 1e-8:
                triangles.append(polygon)
        merged = unary_union(triangles) if triangles else MultiPoint(
            [(x, y) for x, y, _ in points]
        ).convex_hull
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda value: value.area)
        if merged.geom_type != "Polygon":
            merged = MultiPoint([(x, y) for x, y, _ in points]).convex_hull
        merged = merged.simplify(0.005, preserve_topology=True)
        return [[_rounded(x), _rounded(y)] for x, y in list(merged.exterior.coords)[:-1]]

    walls_ifc = list(model.by_type("IfcWall"))
    if not walls_ifc:
        raise ValueError("IFC sem IfcWall para recuperar")

    recovered_walls = []
    wall_lookup = {}
    wall_bases = []
    for index, wall in enumerate(walls_ifc, 1):
        points, bounds = product_bounds(wall)
        xy = list(dict.fromkeys((_rounded(x), _rounded(y)) for x, y, _ in points))
        rectangle = MultiPoint(xy).convex_hull.minimum_rotated_rectangle
        corners = list(rectangle.exterior.coords)[:-1]
        if len(corners) != 4:
            raise ValueError(f"parede {wall.Name or wall.id()} sem retangulo recuperavel")
        edges = []
        for corner_index, start in enumerate(corners):
            end = corners[(corner_index + 1) % 4]
            length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
            edges.append((length, start, end))
        edges.sort(key=lambda value: value[0], reverse=True)
        axis_length, edge_start, edge_end = edges[0]
        if axis_length <= 1e-6:
            continue
        ux = (edge_end[0] - edge_start[0]) / axis_length
        uy = (edge_end[1] - edge_start[1]) / axis_length
        cx = sum(value[0] for value in corners) / 4.0
        cy = sum(value[1] for value in corners) / 4.0
        start = (cx - ux * axis_length / 2.0, cy - uy * axis_length / 2.0)
        end = (cx + ux * axis_length / 2.0, cy + uy * axis_length / 2.0)
        identifier = str(wall.Name or f"W-REC-{index:03d}")
        item = {
            "id": identifier,
            "ax": _rounded(start[0]),
            "ay": _rounded(start[1]),
            "bx": _rounded(end[0]),
            "by": _rounded(end[1]),
            "espessura": _rounded(min(value[0] for value in edges)),
            "altura": _rounded(bounds[5] - bounds[2]),
            "elevacao": 0.0,
            "layer": "IFC-Recovery",
        }
        recovered_walls.append(item)
        wall_lookup[wall.id()] = {
            "editor": item,
            "origin_z": bounds[2],
            "direction": (ux, uy),
            "length": axis_length,
        }
        wall_bases.append(bounds[2])

    # A Scan-to-BIM storey uses one visual ground plane.  Older IFCs could put
    # a manually drawn wall at absolute Z=0 while every detected wall used the
    # cloud's negative datum.  An isolated base cluster is therefore realigned
    # to the dominant base instead of faithfully recovering the old bug.
    base_clusters = Counter(round(value, 2) for value in wall_bases)
    dominant_key = base_clusters.most_common(1)[0][0]
    dominant_base = float(median(
        value for value in wall_bases if round(value, 2) == dominant_key
    ))
    warnings = []
    for wall, original_base in zip(recovered_walls, wall_bases):
        relative = float(original_base) - dominant_base
        cluster_size = base_clusters[round(original_base, 2)]
        if abs(relative) > 0.20 and cluster_size < max(2, len(wall_bases) * 0.20):
            warnings.append(
                f"{wall['id']}: base isolada {original_base:.3f} m alinhada a "
                f"{dominant_base:.3f} m"
            )
            relative = 0.0
        wall["elevacao"] = _rounded(relative)

    recovered_openings = []
    for opening_index, opening in enumerate(model.by_type("IfcOpeningElement"), 1):
        voids = [
            relation for relation in model.get_inverse(opening)
            if relation.is_a("IfcRelVoidsElement")
        ]
        fills = [
            relation for relation in model.get_inverse(opening)
            if relation.is_a("IfcRelFillsElement")
        ]
        if not voids or not fills:
            continue
        host = voids[0].RelatingBuildingElement
        host_data = wall_lookup.get(host.id())
        if host_data is None:
            continue
        filling = fills[0].RelatedBuildingElement
        opening_type = "window" if filling.is_a("IfcWindow") else "door"
        points, bounds = product_bounds(opening)
        wall = host_data["editor"]
        ux, uy = host_data["direction"]
        projections = [
            (x - wall["ax"]) * ux + (y - wall["ay"]) * uy
            for x, y, _ in points
        ]
        width = max(projections) - min(projections)
        center = (max(projections) + min(projections)) / 2.0
        item = {
            "id": f"O-REC-{opening_index:03d}",
            "parede_id": wall["id"],
            "tipo": opening_type,
            "s_centro": _rounded(center),
            "largura": _rounded(width),
            "altura": _rounded(bounds[5] - bounds[2]),
            "origem": "ifc-recovery",
        }
        if opening_type == "window":
            item["peitoril"] = _rounded(max(0.0, bounds[2] - host_data["origin_z"]))
        else:
            item["peitoril"] = 0.0
        recovered_openings.append(item)

    slabs = []
    for slab in model.by_type("IfcSlab"):
        try:
            _, bounds = product_bounds(slab)
            slabs.append((bounds[2], bounds[5], slab))
        except Exception:
            continue
    slabs.sort(key=lambda value: value[0])
    floor_slab = slabs[0] if slabs else None
    ceiling_slab = slabs[-1] if len(slabs) > 1 else None
    slab_contour = footprint(floor_slab[2]) if floor_slab else []

    spaces = []
    for index, space in enumerate(model.by_type("IfcSpace"), 1):
        try:
            contour = footprint(space)
        except Exception:
            continue
        if len(contour) >= 3:
            polygon = Polygon(contour)
            spaces.append({
                "id": str(space.Name or f"SPACE-{index:03d}"),
                "contorno": contour,
                "area": _rounded(polygon.area),
                "perimetro": _rounded(polygon.length),
                "origem": "ifc-recovery",
            })

    coordinates = [
        (value[key_x], value[key_y])
        for value in recovered_walls
        for key_x, key_y in (("ax", "ay"), ("bx", "by"))
    ] + [(value[0], value[1]) for value in slab_contour]
    xs = [value[0] for value in coordinates]
    ys = [value[1] for value in coordinates]
    wall_heights = [float(value["altura"]) for value in recovered_walls]
    default_height = float(median(wall_heights)) if wall_heights else 2.8
    coverings = list(model.by_type("IfcCovering"))
    ceiling_enabled = bool(coverings) or bool(force_ceiling)

    editor_model = {
        "revision": "R00-IFC-RECOVERY",
        "escala": 1,
        "single_line": False,
        "nome": "scan-recuperado",
        "bbox": {
            "xmin": _rounded(min(xs)),
            "ymin": _rounded(min(ys)),
            "xmax": _rounded(max(xs)),
            "ymax": _rounded(max(ys)),
        },
        "diagnostico": {
            "sobras": 0,
            "cantos_costurados": 0,
            "blocos_esquadria": len(recovered_openings),
            "recuperado_ifc": True,
        },
        "paredes": recovered_walls,
        "aberturas": recovered_openings,
        "spaces": spaces,
        "laje": {
            "contorno": slab_contour,
            "piso": {
                "ativo": floor_slab is not None,
                "espessura": _rounded(floor_slab[1] - floor_slab[0]) if floor_slab else 0.15,
            },
            "teto": {
                "ativo": ceiling_slab is not None,
                "espessura": _rounded(ceiling_slab[1] - ceiling_slab[0]) if ceiling_slab else 0.15,
            },
        },
    }
    config = {
        "altura": _rounded(default_height),
        "porta_altura": 2.1,
        "janela_altura": 1.2,
        "janela_peitoril": 1.0,
        "esquadria_detalhada": False,
        "cobertura": True,
        "forro": {
            "ativo": ceiling_enabled,
            "altura": _rounded(max(0.1, default_height - 0.1)),
            "espessura": 0.03,
        },
    }
    return {
        "modelo": editor_model,
        "config": config,
        "warnings": warnings,
        "counts": {
            "walls": len(recovered_walls),
            "openings": len(recovered_openings),
            "spaces": len(spaces),
            "slabs": len(slabs),
        },
    }
