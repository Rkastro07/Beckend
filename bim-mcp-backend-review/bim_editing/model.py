"""Canonical editable-model helpers."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from .geometry import convex_hull, wall_corners, wall_length


MODEL_SCHEMA = "bim.editable-model.v1"
REVISION_SCHEMA = "bim.edit-operations.v1"
ENDPOINT_ORDER = "lexicographic_xy_at_import_then_persistent"


def refresh_derived(model: dict) -> dict:
    walls = model.get("paredes", [])
    for wall in walls:
        wall_id = str(wall["id"])
        wall["id"] = wall_id
        wall["parts"] = {
            "P1": {
                "selector": f"{wall_id}.P1",
                "x": round(float(wall["ax"]), 6),
                "y": round(float(wall["ay"]), 6),
            },
            "P2": {
                "selector": f"{wall_id}.P2",
                "x": round(float(wall["bx"]), 6),
                "y": round(float(wall["by"]), 6),
            },
            "AXIS": {"selector": f"{wall_id}.AXIS"},
        }
        wall["comprimento"] = round(wall_length(wall), 6)

    physical_points = [
        value
        for wall in walls
        for value in wall_corners(wall)
    ]
    if physical_points:
        xs = [value[0] for value in physical_points]
        ys = [value[1] for value in physical_points]
        model["bbox"] = {
            "xmin": min(xs),
            "ymin": min(ys),
            "xmax": max(xs),
            "ymax": max(ys),
        }
    else:
        model["bbox"] = {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0}
    model["schema"] = MODEL_SCHEMA
    model["endpoint_order"] = ENDPOINT_ORDER
    return model


def normalize_model(payload: dict[str, Any], canonicalize_initial_order: bool = True) -> dict:
    model = deepcopy(payload)
    model.setdefault("paredes", [])
    model.setdefault("aberturas", [])
    model.setdefault(
        "laje",
        {
            "contorno": [],
            "piso": {"ativo": True, "espessura": 0.12},
            "teto": {"ativo": True, "espessura": 0.12},
        },
    )
    model.setdefault("spaces", [])
    model.setdefault("edit_history", [])
    model.setdefault("revision", "R00")
    model.setdefault("warnings", [])
    model.setdefault("single_line", False)
    model.setdefault("escala", 1.0)
    model.setdefault("nome", "modelo-bim")

    identifiers: set[str] = set()
    reversed_walls: dict[str, float] = {}
    for index, wall in enumerate(model["paredes"]):
        wall.setdefault("id", f"W-{index + 1:03d}")
        wall["id"] = str(wall["id"])
        if wall["id"] in identifiers:
            raise ValueError(f"ID de parede duplicado: {wall['id']}")
        identifiers.add(wall["id"])
        for key in ("ax", "ay", "bx", "by", "espessura"):
            wall[key] = float(wall[key])
            if not math.isfinite(wall[key]):
                raise ValueError(f"{wall['id']}: valor não finito em {key}")
        if wall["espessura"] <= 0:
            raise ValueError(f"{wall['id']}: espessura deve ser positiva")
        if wall.get("geometria") == "arco":
            curve = wall.get("curva")
            if not isinstance(curve, dict):
                raise ValueError(f"{wall['id']}: parede curva sem ponto C")
            for key in ("x", "y"):
                curve[key] = float(curve[key])
                if not math.isfinite(curve[key]):
                    raise ValueError(f"{wall['id']}: valor não finito em curva.{key}")
        length = wall_length(wall)
        if length <= 1e-4:
            raise ValueError(f"{wall['id']}: parede degenerada")
        if canonicalize_initial_order:
            p1 = (round(wall["ax"], 9), round(wall["ay"], 9))
            p2 = (round(wall["bx"], 9), round(wall["by"], 9))
            if p2 < p1:
                wall["ax"], wall["bx"] = wall["bx"], wall["ax"]
                wall["ay"], wall["by"] = wall["by"], wall["ay"]
                reversed_walls[wall["id"]] = length

    opening_ids: set[str] = set()
    for index, opening in enumerate(model["aberturas"]):
        opening.setdefault("id", f"O-{index + 1:03d}")
        opening["id"] = str(opening["id"])
        if opening["id"] in opening_ids or opening["id"] in identifiers:
            raise ValueError(f"ID de elemento duplicado: {opening['id']}")
        opening_ids.add(opening["id"])
        opening["parede_id"] = str(opening["parede_id"])
        opening["s_centro"] = float(opening["s_centro"])
        opening["largura"] = float(opening["largura"])
        if opening["parede_id"] in reversed_walls:
            opening["s_centro"] = (
                reversed_walls[opening["parede_id"]] - opening["s_centro"]
            )

    if not model["laje"].get("contorno") and model["paredes"]:
        model["laje"]["contorno"] = [
            list(value)
            for value in convex_hull(
                corner
                for wall in model["paredes"]
                for corner in wall_corners(wall)
            )
        ]
    return refresh_derived(model)


def strip_derived(model: dict) -> dict:
    """Return a JSON-friendly copy without redundant endpoint coordinates."""
    result = deepcopy(model)
    for wall in result.get("paredes", []):
        wall.pop("parts", None)
        wall.pop("comprimento", None)
    return result
