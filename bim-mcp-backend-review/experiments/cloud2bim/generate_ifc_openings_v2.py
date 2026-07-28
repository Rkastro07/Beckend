"""Materialise an approved Opening Detector V2 PNG/JSON as IFC elements.

The detector's confidence is preserved as audit metadata, but it is not used
as an authoring filter.  Every candidate present in the approved JSON becomes:

    IfcWall -> IfcOpeningElement -> IfcDoor / IfcWindow

Existing Cloud2BIM openings and fillings are replaced to avoid duplicates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ifcopenshell
import numpy as np
from ifcopenshell.api.feature import add_feature, add_filling
from ifcopenshell.api.geometry import (
    add_wall_representation,
    edit_object_placement,
)
from ifcopenshell.api.pset import add_pset, edit_pset
from ifcopenshell.api.root import create_entity, remove_product
from ifcopenshell.api.spatial import assign_container
from ifcopenshell.util.element import get_container


def body_context(model):
    contexts = [
        context
        for context in model.by_type("IfcGeometricRepresentationSubContext")
        if context.ContextIdentifier == "Body"
    ]
    if not contexts:
        raise RuntimeError("IfcGeometricRepresentationSubContext Body ausente")
    return next(
        (
            context
            for context in contexts
            if getattr(context, "TargetView", None) == "MODEL_VIEW"
        ),
        contexts[0],
    )


def remove_existing_openings(model):
    removed_fillings = 0
    removed_openings = 0
    for opening in list(model.by_type("IfcOpeningElement")):
        for relationship in list(opening.HasFillings):
            remove_product(
                model,
                product=relationship.RelatedBuildingElement,
            )
            removed_fillings += 1
        remove_product(model, product=opening)
        removed_openings += 1
    # Defensive cleanup for malformed IFCs with unhosted fillings.
    for filling in list(model.by_type("IfcDoor")) + list(model.by_type("IfcWindow")):
        remove_product(model, product=filling)
        removed_fillings += 1
    return removed_openings, removed_fillings


def axis_frame(start, end):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= .05:
        raise ValueError("eixo de parede degenerado")
    direction = vector / length
    normal = np.asarray([-direction[1], direction[0]])
    return start, end, direction, normal


def placement_matrix(origin_xy, direction, normal, elevation):
    matrix = np.eye(4)
    matrix[:3, 0] = (direction[0], direction[1], 0.0)
    matrix[:3, 1] = (normal[0], normal[1], 0.0)
    matrix[:3, 2] = (0.0, 0.0, 1.0)
    matrix[:3, 3] = (
        float(origin_xy[0]),
        float(origin_xy[1]),
        float(elevation),
    )
    return matrix


def add_box_representation(
    model,
    product,
    *,
    context,
    matrix,
    width,
    height,
    thickness,
):
    representation = add_wall_representation(
        model,
        context=context,
        length=float(width),
        height=float(height),
        thickness=float(thickness),
        offset=-float(thickness) / 2,
    )
    product.Representation = model.create_entity(
        "IfcProductDefinitionShape",
        Representations=[representation],
    )
    edit_object_placement(
        model,
        product=product,
        matrix=matrix,
        is_si=True,
    )


def add_audit_properties(
    model,
    product,
    *,
    candidate,
    host_wall,
    topology,
):
    evidence = candidate.get("evidence") or {}
    pset = add_pset(
        model,
        product=product,
        name="Pset_Cloud2BIMOpeningV2",
    )
    edit_pset(
        model,
        pset=pset,
        properties={
            "SourceCandidateId": str(candidate["id"]),
            "HostWallId": str(host_wall),
            "DetectorStatus": str(candidate.get("status", "approved_png")),
            "DetectorConfidence": float(candidate.get("confidence", 0.0)),
            "DetectorMode": str(
                evidence.get("detector_mode", "wall_local_density_deficit")
            ),
            "IsTopologyGap": bool(topology),
            "ApprovalContract": "PNG_APPROVED_ALL_CANDIDATES",
        },
    )


def author_candidate(
    model,
    *,
    context,
    wall,
    wall_record,
    candidate,
    floor_z,
    topology=False,
):
    start, _, direction, normal = axis_frame(
        wall_record["start"],
        wall_record["end"],
    )
    width = float(candidate["width"])
    height = float(candidate["height"])
    if width <= .10 or height <= .10:
        raise ValueError(f"dimensao invalida em {candidate['id']}")
    if topology:
        center = np.asarray(candidate["global_center"], dtype=float)
    else:
        center = start + direction * float(candidate["s_center"])
    origin = center - direction * width / 2
    elevation = float(floor_z) + float(candidate["z_min"])
    matrix = placement_matrix(origin, direction, normal, elevation)
    wall_thickness = max(float(wall_record.get("thickness", .15)), .08)

    opening = create_entity(
        model,
        ifc_class="IfcOpeningElement",
        predefined_type="OPENING",
        name=f"OPENING-{candidate['id']}",
    )
    opening.Description = (
        f"Cloud2BIM Opening V2; host={wall.Name}; "
        f"source={candidate['id']}"
    )
    add_box_representation(
        model,
        opening,
        context=context,
        matrix=matrix,
        width=width,
        height=height,
        thickness=wall_thickness + .20,
    )
    add_feature(model, feature=opening, element=wall)

    opening_type = str(candidate["type"])
    if opening_type == "door":
        ifc_class = "IfcDoor"
        predefined_type = "DOOR"
    elif opening_type == "window":
        ifc_class = "IfcWindow"
        predefined_type = "WINDOW"
    else:
        raise ValueError(
            f"tipo de abertura nao suportado em {candidate['id']}: "
            f"{opening_type}"
        )
    filling = create_entity(
        model,
        ifc_class=ifc_class,
        predefined_type=predefined_type,
        name=str(candidate["id"]),
    )
    filling.Description = (
        "Gerado pelo Opening Detector V2 apos aprovacao visual do PNG"
    )
    filling.Tag = str(candidate["id"])
    filling.OverallHeight = height
    filling.OverallWidth = width
    filling_thickness = min(max(.04, wall_thickness * .30), .08)
    add_box_representation(
        model,
        filling,
        context=context,
        matrix=matrix,
        width=width,
        height=height,
        thickness=filling_thickness,
    )
    add_filling(model, opening=opening, element=filling)
    container = get_container(wall)
    if container is not None:
        assign_container(
            model,
            products=[filling],
            relating_structure=container,
        )
    add_audit_properties(
        model,
        filling,
        candidate=candidate,
        host_wall=wall.Name,
        topology=topology,
    )
    return opening, filling


def validate(model, expected):
    counts = {
        "IfcOpeningElement": len(model.by_type("IfcOpeningElement")),
        "IfcDoor": len(model.by_type("IfcDoor")),
        "IfcWindow": len(model.by_type("IfcWindow")),
        "IfcRelVoidsElement": len(model.by_type("IfcRelVoidsElement")),
        "IfcRelFillsElement": len(model.by_type("IfcRelFillsElement")),
    }
    if counts != expected:
        raise RuntimeError(
            f"contagens IFC invalidas: obtido={counts}, esperado={expected}"
        )
    for opening in model.by_type("IfcOpeningElement"):
        if len(opening.VoidsElements) != 1 or len(opening.HasFillings) != 1:
            raise RuntimeError(
                f"abertura #{opening.id()} sem host/filling unico"
            )
    for filling in model.by_type("IfcDoor") + model.by_type("IfcWindow"):
        if len(filling.FillsVoids) != 1:
            raise RuntimeError(
                f"filling #{filling.id()} sem IfcRelFillsElement unico"
            )
        if float(filling.OverallWidth or 0) <= 0:
            raise RuntimeError(f"filling #{filling.id()} sem largura valida")
        if float(filling.OverallHeight or 0) <= 0:
            raise RuntimeError(f"filling #{filling.id()} sem altura valida")
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Gera IFC V2 a partir do JSON/PNG aprovado",
    )
    parser.add_argument("base_ifc", type=Path)
    parser.add_argument("proposals_json", type=Path)
    parser.add_argument("output_ifc", type=Path)
    args = parser.parse_args()

    model = ifcopenshell.open(str(args.base_ifc))
    payload = json.loads(args.proposals_json.read_text(encoding="utf-8"))
    wall_records = {
        wall["wall_id"]: wall
        for wall in payload["walls"]
    }
    ifc_walls = {
        wall.Name: wall
        for wall in model.by_type("IfcWall")
        if wall.Name
    }
    missing = sorted(set(wall_records) - set(ifc_walls))
    if missing:
        raise RuntimeError(f"paredes do JSON ausentes no IFC: {missing}")

    old_openings, old_fillings = remove_existing_openings(model)
    context = body_context(model)
    authored = []
    for wall_record in payload["walls"]:
        wall = ifc_walls[wall_record["wall_id"]]
        for candidate in wall_record.get("candidates", []):
            authored.append(author_candidate(
                model,
                context=context,
                wall=wall,
                wall_record=wall_record,
                candidate=candidate,
                floor_z=payload["floor_z"],
                topology=False,
            ))
    for candidate in payload.get("topology_candidates", []):
        wall_id = candidate["host_wall"]
        wall = ifc_walls[wall_id]
        authored.append(author_candidate(
            model,
            context=context,
            wall=wall,
            wall_record=wall_records[wall_id],
            candidate=candidate,
            floor_z=payload["floor_z"],
            topology=True,
        ))

    candidates = [
        candidate
        for wall in payload["walls"]
        for candidate in wall.get("candidates", [])
    ] + list(payload.get("topology_candidates", []))
    door_count = sum(
        candidate["type"] == "door"
        for candidate in candidates
    )
    window_count = sum(
        candidate["type"] == "window"
        for candidate in candidates
    )
    expected = {
        "IfcOpeningElement": len(candidates),
        "IfcDoor": door_count,
        "IfcWindow": window_count,
        "IfcRelVoidsElement": len(candidates),
        "IfcRelFillsElement": len(candidates),
    }
    counts = validate(model, expected)
    args.output_ifc.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(args.output_ifc))
    summary = {
        "schema": "cloud2bim.ifc-openings-v2-result.v1",
        "base_ifc": str(args.base_ifc),
        "proposals_json": str(args.proposals_json),
        "output_ifc": str(args.output_ifc),
        "confidence_used_as_filter": False,
        "old_openings_removed": old_openings,
        "old_fillings_removed": old_fillings,
        "local_candidates": sum(
            len(wall.get("candidates", []))
            for wall in payload["walls"]
        ),
        "topology_candidates": len(payload.get("topology_candidates", [])),
        "counts": counts,
    }
    summary_path = args.output_ifc.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
