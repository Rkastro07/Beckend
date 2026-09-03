"""Adapters between detector artifacts and the canonical editable model."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .model import normalize_model


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def model_from_cloud2bim(
    diagnostics_csv: str | Path,
    openings_json: str | Path | None = None,
    vertical_levels_json: str | Path | None = None,
    *,
    revision: str = "R00",
) -> dict:
    diagnostics_path = Path(diagnostics_csv)
    vertical_source = (
        load_json(vertical_levels_json)
        if vertical_levels_json is not None
        and Path(vertical_levels_json).exists()
        else None
    )
    vertical_storey = (
        vertical_source.get("storeys", [None])[0]
        if vertical_source and vertical_source.get("storeys")
        else None
    )
    wall_height = (
        float(vertical_storey["wall_height_from_floor"])
        if vertical_storey
        and vertical_storey.get("wall_height_from_floor") is not None
        else None
    )
    walls = []
    with diagnostics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = str(row.get("reference") or "").strip()
            if not identifier:
                continue
            wall = {
                "id": identifier,
                "ax": float(row["start_x"]),
                "ay": float(row["start_y"]),
                "bx": float(row["end_x"]),
                "by": float(row["end_y"]),
                "espessura": float(row["thickness"]),
                "layer": f"Cloud2BIM::{row.get('storey') or 'S01'}",
                "origem": "cloud2bim.wall_detector_v2",
                "detector": row.get("detector"),
                "confidence": row.get("confidence"),
                "review_status": row.get("review_status"),
                "evidence_type": row.get("evidence_type"),
            }
            if row.get("height_band_max") and row.get("height_band_min"):
                band_min = float(row["height_band_min"])
                band_max = float(row["height_band_max"])
                wall["faixa_vertical_evidencia"] = {
                    "z_min": band_min,
                    "z_max": band_max,
                    "altura": band_max - band_min,
                    "uso": "wall_detection_support_only",
                }
            if wall_height is not None:
                wall["altura"] = wall_height
                wall["elevacao"] = 0.0
            walls.append(wall)

    openings = []
    opening_source = None
    topology_candidates = []
    if openings_json is not None:
        opening_source = load_json(openings_json)
        for wall in opening_source.get("walls", []):
            host = str(wall.get("wall_id") or "")
            for candidate in wall.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                identifier = str(candidate.get("id") or "")
                if not identifier or not host:
                    continue
                opening = {
                    "id": identifier,
                    "parede_id": host,
                    "tipo": str(candidate.get("type") or "door"),
                    "s_centro": float(candidate.get("s_center", 0.0)),
                    "largura": float(candidate.get("width", 0.8)),
                    "altura": float(candidate.get("height", 0.0)),
                    "peitoril": float(candidate.get("z_min", 0.0)),
                    "origem": "cloud2bim.opening_detector_v2",
                    "confidence": candidate.get("confidence"),
                    "review_status": candidate.get("status"),
                    "detector_mode": candidate.get("detector_mode"),
                }
                openings.append(opening)
        topology_candidates = list(opening_source.get("topology_candidates", []))

    warnings = []
    if vertical_storey is None:
        warnings.append(
            "niveis verticais nao fornecidos; altura estrutural e forro "
            "precisam de revisao"
        )
    suspended = (
        vertical_storey.get("suspended_ceiling", {})
        if vertical_storey
        else {}
    )
    if vertical_storey and suspended.get("status") != "detected":
        warnings.append(
            "forro suspenso nao detectado; nenhum IfcCovering sera assumido"
        )

    floor_thickness = (
        float(vertical_storey["floor"]["thickness"])
        if vertical_storey
        else 0.12
    )
    structural_thickness = (
        float(vertical_storey["structural_ceiling"]["thickness"])
        if vertical_storey
        else 0.12
    )
    ifc_config = {}
    if wall_height is not None:
        ifc_config["altura"] = wall_height
    if suspended.get("status") == "detected":
        ifc_config["forro"] = {
            "ativo": True,
            "altura": float(suspended["height_from_floor"]),
            "espessura": float(suspended.get("thickness", 0.03)),
            "confidence": suspended.get("confidence"),
            "detector": suspended.get("detector"),
            "requires_visual_review": bool(
                suspended.get("requires_visual_review", True)
            ),
        }

    model = {
        "schema": "bim.editable-model.v1",
        "revision": revision,
        "escala": 1.0,
        "single_line": False,
        "nome": diagnostics_path.stem,
        "source": {
            "format": "cloud2bim-diagnostics",
            "family": "point-cloud",
            "mode": "detected",
            "semantic_level": "reviewable",
            "diagnostics": str(diagnostics_path),
            "openings": str(openings_json) if openings_json else None,
            "vertical_levels": (
                str(vertical_levels_json)
                if vertical_levels_json is not None
                else None
            ),
        },
        "warnings": warnings,
        "diagnostico": {
            "paredes_detectadas": len(walls),
            "aberturas_propostas": len(openings),
            "aberturas_topologicas_propostas": len(topology_candidates),
            "forro_detectado": suspended.get("status") == "detected",
        },
        "paredes": walls,
        "aberturas": openings,
        "topology_opening_candidates": topology_candidates,
        "laje": {
            "contorno": [],
            "piso": {
                "ativo": True,
                "espessura": floor_thickness,
                "categoria": "structural_slab",
            },
            "teto": {
                "ativo": True,
                "espessura": structural_thickness,
                "categoria": "structural_slab",
            },
        },
        "vertical_levels": vertical_source,
        "ifc_config": ifc_config,
        "spaces": [],
        "edit_history": [],
    }
    return normalize_model(model)


def parts_index(model: dict) -> dict:
    return {
        "schema": "bim.element-parts.v1",
        "revision": model.get("revision"),
        "walls": {
            wall["id"]: {
                "P1": dict(wall["parts"]["P1"]),
                "P2": dict(wall["parts"]["P2"]),
                "AXIS": dict(wall["parts"]["AXIS"]),
                "length": wall.get("comprimento"),
                "thickness": wall.get("espessura"),
            }
            for wall in model.get("paredes", [])
        },
    }
