"""Gera a matriz com/sem escada e com/sem Space de um IFC detectado."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.geom
import ifcopenshell.guid
from ifcopenshell.util.placement import get_local_placement
from shapely.geometry import Polygon
from shapely.validation import explain_validity


PATCHED_DIR = Path(__file__).resolve().parents[1] / "cloud2bim_patched"
if str(PATCHED_DIR) not in sys.path:
    sys.path.insert(0, str(PATCHED_DIR))

from space_generator import identify_zones  # noqa: E402


def _body_representation(product):
    representation = getattr(product, "Representation", None)
    if not representation:
        return None
    return next(
        (
            item
            for item in representation.Representations
            if item.RepresentationIdentifier == "Body"
        ),
        None,
    )


def _axis_representation(product):
    representation = getattr(product, "Representation", None)
    if not representation:
        return None
    return next(
        (
            item
            for item in representation.Representations
            if item.RepresentationIdentifier == "Axis"
        ),
        None,
    )


def _wall_dictionaries(model):
    storeys = list(model.by_type("IfcBuildingStorey"))
    walls = []
    for wall in model.by_type("IfcWall"):
        axis = _axis_representation(wall)
        body = _body_representation(wall)
        if not axis or not body or not axis.Items or not body.Items:
            continue
        polyline = axis.Items[0]
        solid = body.Items[0]
        if not polyline.is_a("IfcPolyline") or not solid.is_a("IfcExtrudedAreaSolid"):
            continue
        points = polyline.Points
        profile = solid.SweptArea
        thickness = 0.15
        if profile.is_a("IfcRectangleProfileDef"):
            thickness = min(float(profile.XDim), float(profile.YDim))
        placement = get_local_placement(wall.ObjectPlacement)
        relation = next(iter(getattr(wall, "ContainedInStructure", ()) or ()), None)
        storey = 1
        if relation and relation.RelatingStructure in storeys:
            storey += storeys.index(relation.RelatingStructure)
        walls.append(
            {
                "start_point": tuple(float(v) for v in points[0].Coordinates[:2]),
                "end_point": tuple(float(v) for v in points[-1].Coordinates[:2]),
                "thickness": thickness,
                "material": "Wall",
                "z_placement": float(placement[2, 3]),
                "height": float(solid.Depth),
                "storey": storey,
                "wall_id": wall.id(),
            }
        )
    return walls


def _slab_levels(model):
    levels = []
    for slab in model.by_type("IfcSlab"):
        body = _body_representation(slab)
        if not body or not body.Items:
            continue
        solid = body.Items[0]
        if not solid.is_a("IfcExtrudedAreaSolid"):
            continue
        z = float(get_local_placement(slab.ObjectPlacement)[2, 3])
        levels.append((z, z + float(solid.Depth), slab))
    return sorted(levels, key=lambda item: item[0])


def _body_context(model):
    contexts = model.by_type("IfcGeometricRepresentationSubContext")
    return next(
        (ctx for ctx in contexts if ctx.ContextIdentifier == "Body"),
        contexts[0] if contexts else model.by_type("IfcGeometricRepresentationContext")[0],
    )


def _create_local_placement(model, z):
    location = model.create_entity(
        "IfcCartesianPoint", Coordinates=(0.0, 0.0, float(z))
    )
    placement = model.create_entity("IfcAxis2Placement3D", Location=location)
    return model.create_entity("IfcLocalPlacement", RelativePlacement=placement)


def remove_spaces(model):
    for space in list(model.by_type("IfcSpace")):
        ifcopenshell.api.run("root.remove_product", model, product=space)


def remove_stair(model):
    stair_openings = [
        opening
        for opening in model.by_type("IfcOpeningElement")
        if (opening.Name or "").strip().lower() == "vao de escada"
    ]
    for stair in list(model.by_type("IfcStair")):
        ifcopenshell.api.run("root.remove_product", model, product=stair)
    for opening in stair_openings:
        ifcopenshell.api.run("root.remove_product", model, product=opening)


def add_detected_spaces(model, maximum_snap=0.15):
    remove_spaces(model)
    walls = _wall_dictionaries(model)
    zones = identify_zones(
        walls, snapping_distance=float(maximum_snap), plot_zones=False
    )
    slabs = _slab_levels(model)
    if len(slabs) < 2:
        raise RuntimeError("Sao necessarias duas lajes para definir a altura do Space.")
    floor_bottom, floor_top, floor_slab = slabs[0]
    upper_bottom = slabs[1][0]
    clear_height = upper_bottom - floor_top
    if clear_height <= 0.0:
        raise RuntimeError("Altura livre do Space invalida.")

    relation = next(iter(getattr(floor_slab, "ContainedInStructure", ()) or ()), None)
    if relation:
        storey = relation.RelatingStructure
    else:
        storey = min(
            model.by_type("IfcBuildingStorey"),
            key=lambda item: float(get_local_placement(item.ObjectPlacement)[2, 3]),
        )

    context = _body_context(model)
    owner_history = model.by_type("IfcOwnerHistory")[0]
    material = next(
        (
            item
            for item in model.by_type("IfcMaterial")
            if (item.Name or "") == "Space volume"
        ),
        None,
    )

    spaces = []
    for number, (_, zone) in enumerate(zones.items(), start=1):
        coordinates = list(zone["vertices"])
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
        points = [
            model.create_entity(
                "IfcCartesianPoint", Coordinates=(float(x), float(y))
            )
            for x, y in coordinates
        ]
        polyline = model.create_entity("IfcPolyline", Points=points)
        profile = model.create_entity(
            "IfcArbitraryClosedProfileDef",
            ProfileType="AREA",
            OuterCurve=polyline,
        )
        solid_position = model.create_entity(
            "IfcAxis2Placement3D",
            Location=model.create_entity(
                "IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)
            ),
        )
        solid = model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile,
            Position=solid_position,
            ExtrudedDirection=model.create_entity(
                "IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)
            ),
            Depth=float(clear_height),
        )
        shape = model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=context,
            RepresentationIdentifier="Body",
            RepresentationType="SweptSolid",
            Items=[solid],
        )
        product_shape = model.create_entity(
            "IfcProductDefinitionShape", Representations=[shape]
        )
        name = f"1.{number}"
        space = model.create_entity(
            "IfcSpace",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner_history,
            Name=name,
            ObjectPlacement=_create_local_placement(model, floor_top),
            Representation=product_shape,
            LongName=f"Room No. {name}",
            CompositionType="ELEMENT",
            PredefinedType="INTERNAL",
        )
        model.create_entity(
            "IfcRelContainedInSpatialStructure",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner_history,
            RelatedElements=[space],
            RelatingStructure=storey,
        )
        if material:
            model.create_entity(
                "IfcRelAssociatesMaterial",
                GlobalId=ifcopenshell.guid.new(),
                OwnerHistory=owner_history,
                RelatedObjects=[space],
                RelatingMaterial=material,
            )
        spaces.append(space)
    return spaces


def audit(path):
    model = ifcopenshell.open(path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    geometry_errors = []
    profile_errors = []
    for ifc_class in (
        "IfcWall",
        "IfcSlab",
        "IfcWindow",
        "IfcDoor",
        "IfcStair",
        "IfcSpace",
    ):
        for product in model.by_type(ifc_class):
            try:
                ifcopenshell.geom.create_shape(settings, product)
            except Exception as exc:  # pragma: no cover - diagnostic output
                geometry_errors.append(f"{ifc_class} #{product.id()}: {exc}")
    for slab in model.by_type("IfcSlab"):
        body = _body_representation(slab)
        if not body:
            profile_errors.append(f"IfcSlab #{slab.id()}: no Body representation")
            continue
        for item_index, solid in enumerate(body.Items, start=1):
            if not solid.is_a("IfcExtrudedAreaSolid"):
                profile_errors.append(
                    f"IfcSlab #{slab.id()} item {item_index}: "
                    f"unexpected {solid.is_a()}"
                )
                continue
            profile = solid.SweptArea
            if not profile.is_a("IfcArbitraryClosedProfileDef"):
                continue
            curve = profile.OuterCurve
            if not curve.is_a("IfcPolyline"):
                continue
            coordinates = [
                tuple(float(value) for value in point.Coordinates[:2])
                for point in curve.Points
            ]
            polygon = Polygon(coordinates)
            if polygon.is_empty or not polygon.is_valid:
                profile_errors.append(
                    f"IfcSlab #{slab.id()} item {item_index}: "
                    f"{explain_validity(polygon)}"
                )
    stair_openings = [
        item
        for item in model.by_type("IfcOpeningElement")
        if (item.Name or "").strip().lower() == "vao de escada"
    ]
    return {
        "walls": len(model.by_type("IfcWall")),
        "windows": len(model.by_type("IfcWindow")),
        "doors": len(model.by_type("IfcDoor")),
        "slabs": len(model.by_type("IfcSlab")),
        "stairs": len(model.by_type("IfcStair")),
        "spaces": len(model.by_type("IfcSpace")),
        "openings": len(model.by_type("IfcOpeningElement")),
        "stair_openings": len(stair_openings),
        "geometry_errors": geometry_errors,
        "profile_errors": profile_errors,
    }


def generate(source, output_dir, stem, maximum_snap=0.15):
    combinations = (
        (True, True, "COM_ESCADA_COM_SPACE"),
        (True, False, "COM_ESCADA_SEM_SPACE"),
        (False, True, "SEM_ESCADA_COM_SPACE"),
        (False, False, "SEM_ESCADA_SEM_SPACE"),
    )
    results = {}
    for with_stair, with_space, suffix in combinations:
        model = ifcopenshell.open(source)
        remove_spaces(model)
        if not with_stair:
            remove_stair(model)
        if with_space:
            add_detected_spaces(model, maximum_snap=maximum_snap)
        output = output_dir / f"{stem}_{suffix}.ifc"
        model.write(str(output))
        results[str(output)] = audit(str(output))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stem", default=None)
    parser.add_argument(
        "--space-max-snap",
        type=float,
        default=float(os.getenv("SPACE_MAX_SNAP", "0.15")),
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.source.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or args.source.stem
    results = generate(
        args.source, output_dir, stem, maximum_snap=args.space_max_snap
    )
    for path, result in results.items():
        print(Path(path).name, result)
        if result["geometry_errors"] or result["profile_errors"]:
            raise RuntimeError(
                f"Falha de geometria em {path}: "
                f"{result['geometry_errors'] + result['profile_errors']}"
            )


if __name__ == "__main__":
    main()
