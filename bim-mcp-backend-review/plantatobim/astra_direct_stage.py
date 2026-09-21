"""Astra-only image-to-editor stage for Plan-to-BIM.

This path deliberately receives no wall/opening candidates and calls no local
geometry detector. The only geometric proposal comes from the Astra response;
the backend merely validates and normalizes it for the existing manual editor.
"""

from __future__ import annotations

import base64
import math
import os
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .astra_semantic_stage import ASTRA_MODEL
from .gpt_plan_assistant import (
    GptPlanError,
    OpenAIResponsesClient,
    _image_data_url_from_path,
    build_candidate_model,
)
from .astra_visual_geometry import VISUAL_INSTRUCTIONS, VISUAL_SCHEMA, prepare_visual_inputs, visual_to_review


DIRECT_SCHEMA_VERSION = "2.0"
DIRECT_PIPELINE_VERSION = "astra-direct-v2"

DIRECT_INSTRUCTIONS = VISUAL_INSTRUCTIONS


def _image_info(image_path: Path) -> tuple[int, int, str, str]:
    path = Path(image_path)
    try:
        with Image.open(path) as image:
            width_px, height_px = image.size
            image_format = (image.format or path.suffix.lstrip(".") or "png").lower()
    except Exception as exc:
        raise GptPlanError("Não foi possível abrir a imagem preparada da planta.") from exc
    if width_px <= 0 or height_px <= 0:
        raise GptPlanError("A imagem preparada possui dimensões inválidas.")
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(image_format, "image/png")
    return width_px, height_px, image_format, mime


def build_direct_editor_model(
    image_path: Path,
    analysis: dict[str, Any],
    *,
    canvas_width_m: float,
    original_name: str,
    auto_review_threshold: float = 0.85,
) -> dict[str, Any]:
    """Validate Astra-authored geometry and adapt it to the manual editor."""
    width_px, height_px, image_format, mime = _image_info(image_path)
    width_m = float(canvas_width_m)
    if not math.isfinite(width_m) or not 1 <= width_m <= 500:
        raise GptPlanError("A largura métrica do canvas é inválida.")
    height_m = width_m * height_px / width_px
    if analysis.get("changed") is not True:
        raise GptPlanError("O Astra não devolveu o modelo geométrico completo.")
    if not analysis.get("walls"):
        raise GptPlanError("O Astra não encontrou paredes suficientes para abrir o editor.")
    if analysis.get("slab_contour") and len(analysis["slab_contour"]) < 3:
        raise GptPlanError("O Astra devolveu um contorno de laje incompleto.")

    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    base_model: dict[str, Any] = {
        "ok": True,
        "nome": Path(original_name).stem,
        "escala": width_m / width_px,
        "single_line": False,
        "bbox": {"xmin": 0.0, "ymin": 0.0, "xmax": width_m, "ymax": height_m},
        "diagnostico": {
            "sobras": 0,
            "cantos_costurados": 0,
            "blocos_esquadria": 0,
            "elementos_lidos": 0,
            "geometrias_aproximadas": 0,
        },
        "source": {
            "format": image_format,
            "family": "raster",
            "mode": "astra-direct",
            "semantic_level": "astra-authored-geometry",
            "scale_source": "user-building-width",
            "astra_model": str(analysis.get("_model") or ASTRA_MODEL),
            "geometry_source": "astra-only",
            "heuristic_detector_used": False,
        },
        "reference": {
            "kind": "raster2seq",
            "engine": "astra-direct",
            "label": "Planta original enviada diretamente ao Astra",
            "bounds": [0.0, 0.0, width_m, height_m],
            "image_mime": mime,
            "image_base64": encoded,
            "canvas_size": [width_px, height_px],
            "canvas_width_m": width_m,
            "rooms": [],
            "openings": [],
            "dimensions": [],
        },
        "warnings": [],
        "paredes": [],
        "aberturas": [],
        "laje": {
            "contorno": [],
            "piso": {"ativo": True, "espessura": 0.12},
            "teto": {"ativo": False, "espessura": 0.12},
        },
        "spaces": [],
    }
    candidate = build_candidate_model(base_model, analysis)
    candidate["ok"] = True
    candidate["nome"] = Path(original_name).stem
    candidate["engine"] = DIRECT_PIPELINE_VERSION
    candidate["source"] = {
        **base_model["source"],
        "astra_model": str(analysis.get("_model") or ASTRA_MODEL),
    }
    if analysis.get("_visual_manifest"):
        candidate["source"]["visual_protocol"] = "astra-visual-seven-v2"
        candidate["source"]["image_framing_used"] = True
        candidate["source"]["geometry_candidates_used"] = False
    if not analysis.get("slab_contour"):
        candidate["laje"]["piso"]["ativo"] = False

    review_walls: list[str] = []
    for wall in candidate["paredes"]:
        confidence = float(wall.get("confidence") or 0)
        status = "review" if confidence < auto_review_threshold else "keep"
        wall["origem"] = "gpt-6-astra-direct"
        wall["astra_status"] = status
        wall["astra_semantic"] = {
            "candidate_id": str(wall["id"]),
            "action": status,
            "classification": str(wall.get("tipo") or "wall"),
            "suggested_name": wall.get("nome"),
            "confidence": confidence,
            "reason": str(wall.get("semantic_reason") or "interpretação visual Astra"),
            "visual_evidence": [],
        }
        if status == "review":
            wall["ml_status"] = "uncertain"
            review_walls.append(str(wall["id"]))

    review_openings: list[str] = []
    for opening in candidate["aberturas"]:
        confidence = float(opening.get("confidence") or 0)
        status = "review" if confidence < auto_review_threshold else "keep"
        opening["origem"] = "gpt-6-astra-direct"
        opening["astra_status"] = status
        opening["astra_semantic"] = {
            "candidate_id": str(opening["id"]),
            "action": status,
            "classification": str(opening.get("tipo") or "door"),
            "suggested_name": opening.get("nome"),
            "confidence": confidence,
            "reason": str(opening.get("semantic_reason") or "interpretação visual Astra"),
            "visual_evidence": [],
        }
        if status == "review":
            review_openings.append(str(opening["id"]))

    unresolved = [str(item) for item in analysis.get("unresolved") or []]
    assumptions = [str(item) for item in analysis.get("assumptions") or []]
    slab_review = not analysis.get("slab_contour") or float(analysis.get("confidence") or 0) < auto_review_threshold
    candidate["laje"]["astra_status"] = "review" if slab_review else "keep"
    candidate["astra_editor"] = {
        "status": "ready-for-manual-review",
        "pipeline_version": DIRECT_PIPELINE_VERSION,
        "geometry_source": "astra-only",
        "heuristic_detector_used": False,
        "active": {
            "walls": len(candidate["paredes"]),
            "openings": len(candidate["aberturas"]),
        },
        "pending_review": {
            "walls": review_walls,
            "openings": review_openings,
            "slab": slab_review,
        },
        "excluded": {"walls": [], "openings": []},
        "missing_elements": [],
        "needs_human_review": bool(
            review_walls or review_openings or slab_review or unresolved or assumptions
        ),
    }
    candidate["astra_direct"] = dict(analysis)
    candidate["astra_semantic"] = dict(analysis)
    candidate["warnings"] = [
        "Modelo geométrico criado exclusivamente pelo Astra; revise antes de gerar IFC/DXF.",
        *(f"PENDÊNCIA ASTRA: {item}" for item in unresolved),
        *(f"HIPÓTESE ASTRA: {item}" for item in assumptions),
    ]
    return candidate


class AstraDirectPlanStage:
    def __init__(self, client: Any | None = None, *, model: str = ASTRA_MODEL) -> None:
        self.client = client or OpenAIResponsesClient(
            model=model,
            timeout_seconds=float(os.environ.get("OPENAI_PLAN_TIMEOUT_SECONDS", "1200")),
        )

    def analyze(
        self,
        image_path: Path,
        *,
        canvas_width_m: float,
        original_name: str,
        user_message: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        artifact_dir = image_path.parent / "astra_visual"
        manifest = prepare_visual_inputs(image_path, artifact_dir)
        prompt = (
            "Reconstruct this plan independently. No wall/opening candidates are supplied.\n"
            f"Arquivo: {Path(original_name).name}\n"
            f"Crop size: {manifest['crop_size_px']} pixels.\n"
            "The first image and six detail images share the GLOBAL 0..1000 frame.\n"
            + "\n".join(f"Image {i+1}: {item['path']}, global extent {item['extent']}"
                        for i, item in enumerate(manifest["inputs"])) + "\n"
            f"Objetivo adicional: {user_message[:3000]}\n"
            "Return all visible geometry in compact JSON, no metric or height inference."
        )
        effort = os.environ.get("OPENAI_PLAN_VISUAL_REASONING_EFFORT", "high")
        (artifact_dir / "request_protocol.json").write_text(json.dumps({
            "instructions": VISUAL_INSTRUCTIONS, "user_text": prompt,
            "schema": VISUAL_SCHEMA, "reasoning_effort": effort,
            "geometry_candidates_used": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        preparation_seconds = time.perf_counter() - started
        raw, metadata = self.client.structured(
            schema_name="plan_bim_visual_geometry_v2",
            schema=VISUAL_SCHEMA,
            instructions=VISUAL_INSTRUCTIONS,
            user_text=prompt,
            images=[_image_data_url_from_path(artifact_dir / item["path"]) for item in manifest["inputs"]],
            reasoning_effort=effort,
            artifact_dir=artifact_dir,
        )
        (artifact_dir / "geometry_raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        adaptation_started = time.perf_counter()
        analysis = visual_to_review(raw, manifest, canvas_width_m,
            wall_height=float(os.environ.get("PLAN_BIM_WALL_HEIGHT_M", "2.8")),
            door_height=float(os.environ.get("PLAN_BIM_DOOR_HEIGHT_M", "2.1")),
            window_height=float(os.environ.get("PLAN_BIM_WINDOW_HEIGHT_M", "1.2")),
            window_sill=float(os.environ.get("PLAN_BIM_WINDOW_SILL_M", "1.0")))
        analysis["_provider"] = metadata.get("provider") or "openai"
        analysis["_model"] = metadata.get("model") or ASTRA_MODEL
        candidate = build_direct_editor_model(
            image_path,
            analysis,
            canvas_width_m=canvas_width_m,
            original_name=original_name,
        )
        metadata.update({"visual_preparation_seconds": round(preparation_seconds, 3),
                         "adaptation_seconds": round(time.perf_counter()-adaptation_started, 3),
                         "stage_seconds": round(time.perf_counter()-started, 3),
                         "artifact_dir": str(artifact_dir)})
        return candidate, analysis, metadata
