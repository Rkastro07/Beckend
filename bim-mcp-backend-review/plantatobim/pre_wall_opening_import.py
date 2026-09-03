"""Adapter from the pre-wall YOLO experiment to the editable plan model."""

from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

from .raster_2d_import import Raster2DError, detect_slab_contour_2d
from .raster_room_ocr import analyze_room_ocr


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCRIPT = BASE_DIR / "experiments" / "plant2bim" / "run_pre_wall_opening_pipeline.py"
DEFAULT_WEIGHTS = (
    BASE_DIR
    / "artifacts"
    / "cubicasa_yolo_aligned_v2"
    / "runs"
    / "doors_windows_aligned_v2_960_finetune"
    / "weights"
    / "best.pt"
)
DEFAULT_WALL_WEIGHTS = (
    BASE_DIR
    / "artifacts"
    / "cubicasa_wall_seg_v1"
    / "runs"
    / "walls_seg_v1_960_m4"
    / "weights"
    / "best.pt"
)
DEFAULT_YOLO_PYTHON = (
    BASE_DIR
    / ".codex_tmp"
    / "yolo_world_zero_shot"
    / "venv"
    / "Scripts"
    / "python.exe"
)


class PreWallOpeningError(RuntimeError):
    pass


def _runtime_path(environment_name: str, fallback: Path) -> Path:
    configured = str(os.environ.get(environment_name) or "").strip()
    return Path(configured) if configured else fallback


def run_pre_wall_pipeline(
    image_path: Path,
    output_dir: Path,
    *,
    canvas_width_m: float,
    metric_refinement: bool = True,
    timeout_seconds: int = 240,
) -> dict[str, Any]:
    if not math.isfinite(canvas_width_m) or canvas_width_m <= 0:
        raise PreWallOpeningError("A largura geométrica precisa ser positiva.")
    python_path = _runtime_path("PLANT2BIM_YOLO_PYTHON", DEFAULT_YOLO_PYTHON)
    weights_path = _runtime_path("PLANT2BIM_YOLO_WEIGHTS", DEFAULT_WEIGHTS)
    wall_weights_path = _runtime_path(
        "PLANT2BIM_YOLO_WALL_WEIGHTS",
        DEFAULT_WALL_WEIGHTS,
    )
    script_path = _runtime_path("PLANT2BIM_PRE_WALL_SCRIPT", DEFAULT_SCRIPT)
    if not python_path.exists():
        raise PreWallOpeningError(
            "Ambiente YOLO não encontrado. Configure PLANT2BIM_YOLO_PYTHON."
        )
    if not weights_path.exists():
        raise PreWallOpeningError(
            "Pesos do detector não encontrados. Configure PLANT2BIM_YOLO_WEIGHTS."
        )
    if not script_path.exists():
        raise PreWallOpeningError("Script do detector pré-paredes não encontrado.")

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_path),
        str(script_path),
        "--image",
        str(image_path),
        "--weights",
        str(weights_path),
        "--output-dir",
        str(output_dir),
        "--canvas-width-m",
        str(canvas_width_m),
    ]
    yolo_device = str(os.environ.get("PLANT2BIM_YOLO_DEVICE") or "").strip()
    if yolo_device:
        command.extend(["--device", yolo_device])
    if not metric_refinement:
        command.append("--disable-metric-refinement")
    wall_source = str(os.environ.get("PLANT2BIM_YOLO_WALL_SOURCE") or "hybrid").strip().lower()
    if wall_source not in {"geometry", "yolo", "hybrid"}:
        wall_source = "hybrid"
    command.extend(["--wall-source", wall_source])
    if wall_weights_path.exists():
        command.extend(["--wall-weights", str(wall_weights_path)])
    try:
        completed = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PreWallOpeningError(
            f"O detector YOLO excedeu {timeout_seconds} segundos."
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "erro desconhecido").strip()
        raise PreWallOpeningError(
            f"Falha no detector YOLO pré-paredes: {detail[-1500:]}"
        )
    result_path = output_dir / "result.json"
    if not result_path.exists():
        raise PreWallOpeningError("O detector terminou sem produzir result.json.")
    return json.loads(result_path.read_text(encoding="utf-8"))


def pre_wall_result_to_editor_model(
    result: dict[str, Any],
    crop_path: Path,
    *,
    source_name: str,
) -> dict[str, Any]:
    color = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if color is None:
        raise PreWallOpeningError("A imagem recortada do detector não foi encontrada.")
    image_height, image_width = color.shape[:2]
    requested_canvas_width_m = float(result["canvas_width_m"])
    fallback_pixel_m = requested_canvas_width_m / float(image_width)
    detection_scale = result.get("detection_scale") or {}
    visual_pixel_m = float(detection_scale.get("detection_pixel_m") or fallback_pixel_m)
    visual_canvas_extent_m = float(
        detection_scale.get("detection_canvas_extent_m") or requested_canvas_width_m
    )
    axes = list(result.get("walls") or [])
    detected_openings = list(result.get("openings") or [])
    if not axes:
        raise PreWallOpeningError("O detector não produziu paredes para abrir no editor.")

    room_ocr = analyze_room_ocr(
        crop_path,
        axes,
        color.shape,
        fallback_pixel_m=fallback_pixel_m,
    )
    room_scale = room_ocr["scale"]
    ocr_scale_applied = bool(room_scale.get("applied"))
    pixel_m = float(room_scale.get("pixel_m") or fallback_pixel_m)
    canvas_width_m = image_width * pixel_m
    canvas_height_m = image_height * pixel_m
    object_size_pixel_m = pixel_m if ocr_scale_applied else visual_pixel_m

    def world_point(x: float, y: float) -> tuple[float, float]:
        return (
            round(float(x) * pixel_m, 5),
            round((image_height - float(y)) * pixel_m, 5),
        )

    reference_rooms: list[dict[str, Any]] = []
    reference_dimensions: list[dict[str, Any]] = []
    for room_index, room in enumerate(room_ocr.get("rooms") or [], 1):
        world_polygon = [world_point(point[0], point[1]) for point in room["points_px"]]
        declared_areas = list(room.get("areas") or [])
        declared_pairs = list(room.get("dimension_pairs") or [])
        reference_rooms.append({
            "id": str(room.get("id") or f"ROOM-OCR-{room_index:03d}"),
            "category_id": int(room.get("category_id", 8)),
            "label": str(room.get("label") or f"Ambiente {room_index}"),
            "points": world_polygon,
            "area": round(float(room.get("area_px", 0.0)) * pixel_m * pixel_m, 4),
            "ocr_text": [str(line.get("text") or "") for line in room.get("ocr_lines") or []],
            "declared_area_m2": (
                float(declared_areas[0]["value_m2"])
                if declared_areas
                else None
            ),
            "declared_dimensions_m": (
                list(declared_pairs[0]["values_m"])
                if declared_pairs
                else []
            ),
        })
        for pair_index, pair in enumerate(declared_pairs, 1):
            bbox_px = pair["bbox_px"]
            position_px = pair["position_px"]
            bbox = {
                "xmin": round(float(bbox_px["xmin"]) * pixel_m, 5),
                "ymin": round((image_height - float(bbox_px["ymax"])) * pixel_m, 5),
                "xmax": round(float(bbox_px["xmax"]) * pixel_m, 5),
                "ymax": round((image_height - float(bbox_px["ymin"])) * pixel_m, 5),
            }
            position = {
                "x": round(float(position_px[0]) * pixel_m, 5),
                "y": round((image_height - float(position_px[1])) * pixel_m, 5),
            }
            for value_index, value_m in enumerate(pair["values_m"], 1):
                reference_dimensions.append({
                    "id": f"ROOM-DIM-{room_index:03d}-{pair_index}-{value_index}",
                    "text": str(pair["text"]),
                    "line_text": str(pair["line_text"]),
                    "value_m": round(float(value_m), 5),
                    "confidence": max(0.5, float(room_scale.get("confidence", 0.0))),
                    "assumption": "room-dimension-pair",
                    "kind": "linear",
                    "position": position,
                    "bbox": bbox,
                })

    walls: list[dict[str, Any]] = []
    for index, axis in enumerate(axes):
        if axis["orientation"] == "vertical":
            ax, ay = world_point(float(axis["fixed"]), float(axis["end"]))
            bx, by = world_point(float(axis["fixed"]), float(axis["start"]))
        else:
            ax, ay = world_point(float(axis["start"]), float(axis["fixed"]))
            bx, by = world_point(float(axis["end"]), float(axis["fixed"]))
        walls.append({
            "id": f"W-YOLO-{index + 1:03d}",
            "ax": ax,
            "ay": ay,
            "bx": bx,
            "by": by,
            "espessura": round(
                max(
                    0.06,
                    min(0.45, float(axis.get("thickness", 6.0)) * object_size_pixel_m),
                ),
                4,
            ),
            "altura": 2.8,
            "elevacao": 0.0,
            "layer": "Wall-Pre-Wall-YOLO",
            "nome": f"Parede YOLO {index + 1}",
            "tipo": "wall",
            "ifc_class": "IfcWall",
            "origem": "pre-wall-yolo",
            "confidence": 0.72 if result.get("wall_geometry_source") == "thick-wall-2d" else 0.58,
        })

    editor_openings: list[dict[str, Any]] = []
    for index, opening in enumerate(detected_openings):
        host_index = int(opening.get("wall_index", -1))
        if host_index < 0 or host_index >= len(axes):
            continue
        axis = axes[host_index]
        wall = walls[host_index]
        start_px = np.asarray(opening["start_px"], dtype=float)
        end_px = np.asarray(opening["end_px"], dtype=float)
        opening_width = float(
            np.linalg.norm(end_px - start_px) * object_size_pixel_m
            if ocr_scale_applied
            else opening.get("classification_width_m")
            or np.linalg.norm(end_px - start_px) * visual_pixel_m
        )
        if axis["orientation"] == "vertical":
            opening_center_px = (float(start_px[1]) + float(end_px[1])) / 2.0
            center_along = (float(axis["end"]) - opening_center_px) * pixel_m
        else:
            opening_center_px = (float(start_px[0]) + float(end_px[0])) / 2.0
            center_along = (opening_center_px - float(axis["start"])) * pixel_m
        wall_length = math.hypot(
            float(wall["bx"]) - float(wall["ax"]),
            float(wall["by"]) - float(wall["ay"]),
        )
        opening_width = min(
            max(0.30, opening_width),
            max(0.30, wall_length - 0.04),
        )
        center_along = max(
            opening_width / 2.0,
            min(wall_length - opening_width / 2.0, center_along),
        )
        kind = "door" if opening.get("type") == "door" else "window"
        editor_openings.append({
            "id": f"O-YOLO-{index + 1:03d}",
            "parede_id": wall["id"],
            "tipo": kind,
            "s_centro": round(center_along, 5),
            "largura": round(opening_width, 4),
            "nome": f"{'Porta' if kind == 'door' else 'Janela'} YOLO {index + 1}",
            "altura": 2.1 if kind == "door" else 1.2,
            "peitoril": 0.0 if kind == "door" else 1.0,
            "origem": "pre-wall-yolo",
            "confidence": round(float(opening.get("confidence", 0.5)), 4),
            "semantic_reason": str(
                opening.get("classification") or "consenso YOLO multiescala"
            ),
        })

    slab_pixels, slab_diagnostics = detect_slab_contour_2d(
        axes,
        color.shape,
        canvas_width_m=visual_canvas_extent_m,
    )
    slab_contour = [world_point(point[0], point[1]) for point in slab_pixels]
    slab_active = len(slab_contour) >= 3
    slab_area_m2 = (
        abs(float(cv2.contourArea(np.asarray(slab_pixels, dtype=np.float32))))
        * pixel_m
        * pixel_m
        if len(slab_pixels) >= 3
        else 0.0
    )
    ok, encoded = cv2.imencode(".png", color)
    if not ok:
        raise PreWallOpeningError("Não foi possível codificar o fundo da planta.")

    doors = sum(item["tipo"] == "door" for item in editor_openings)
    windows = sum(item["tipo"] == "window" for item in editor_openings)
    raw_count = int(result.get("raw_detection_count", 0))
    candidate_count = int(result.get("consensus_candidate_count", 0))
    rejected = max(0, candidate_count - len(editor_openings))
    diagnostics = {
        "source_wall_segment_count": int(result.get("raster_wall_count", len(axes))),
        "canonical_wall_count": len(axes),
        "wall_segments_absorbed": 0,
        "wall_gap_count": candidate_count,
        "classified_wall_gaps": len(editor_openings),
        "unclassified_wall_gaps": rejected,
        "unmatched_openings": rejected,
        "slab_method": slab_diagnostics.get("slab_method"),
        "slab_area_m2": round(slab_area_m2, 4),
        "pre_wall_raw_detections": raw_count,
        "pre_wall_consensus_candidates": candidate_count,
        "pre_wall_wall_source": result.get("wall_geometry_source"),
        "pre_wall_visual_pixel_m": round(visual_pixel_m, 8),
        "pre_wall_metric_pixel_m": round(pixel_m, 8),
        "pre_wall_object_sizes_stable": bool(detection_scale),
        "pre_wall_scales": result.get("scales", []),
        "requested_canvas_width_m": round(requested_canvas_width_m, 5),
        "effective_canvas_width_m": round(canvas_width_m, 5),
        "room_count": len(reference_rooms),
        "room_ocr_status": room_ocr.get("ocr", {}).get("status"),
        "room_ocr_lines": int(room_ocr.get("ocr", {}).get("line_count", 0)),
        "room_ocr_matched_lines": int(room_ocr.get("ocr", {}).get("matched_line_count", 0)),
        "room_scale_candidates": int(room_scale.get("candidate_count", 0)),
        "room_scale_inliers": int(room_scale.get("inlier_count", 0)),
        "room_scale_applied": ocr_scale_applied,
        "room_scale_confidence": float(room_scale.get("confidence", 0.0)),
    }
    return {
        "ok": True,
        "escala": pixel_m,
        "single_line": str(result.get("wall_geometry_source", "")).startswith("single-line"),
        "nome": Path(source_name).stem,
        "bbox": {
            "xmin": 0.0,
            "ymin": 0.0,
            "xmax": canvas_width_m,
            "ymax": canvas_height_m,
        },
        "diagnostico": {
            "sobras": rejected,
            "cantos_costurados": 0,
            "blocos_esquadria": len(editor_openings),
            "elementos_lidos": len(walls) + len(editor_openings) + int(slab_active),
            "geometrias_aproximadas": rejected,
        },
        "source": {
            "format": Path(source_name).suffix.lower().lstrip(".") or "image",
            "family": "raster",
            "mode": "pre-wall-yolo",
            "semantic_level": "yolo-multiscale+geometric-hosting",
            "scale_source": (
                "room-ocr-consensus"
                if ocr_scale_applied
                else "user-building-width"
            ),
            "object_size_source": (
                "room-ocr-consensus"
                if ocr_scale_applied
                else "visual-wall-thickness-calibration"
                if detection_scale
                else "user-building-width"
            ),
        },
        "reference": {
            "kind": "raster2seq",
            "engine": "pre-wall-yolo",
            "label": "YOLO multiescala antes das paredes",
            "bounds": [0.0, 0.0, canvas_width_m, canvas_height_m],
            "image_mime": "image/png",
            "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "canvas_size": [image_width, image_height],
            "canvas_width_m": canvas_width_m,
            "rooms": reference_rooms,
            "openings": [],
            "dimensions": reference_dimensions,
        },
        "warnings": [
            "YOLO experimental: revise manualmente falsos positivos em mobiliário e linhas de cota.",
            f"Consenso multiescala: {raw_count} caixas brutas, {candidate_count} candidatos e {len(editor_openings)} aberturas hospedadas.",
            f"Resultado editável: {len(walls)} paredes, {doors} portas e {windows} janelas.",
            *(
                [
                    "Escala automática por OCR de ambientes aplicada com "
                    f"{float(room_scale.get('confidence', 0.0)) * 100:.0f}% de confiança: "
                    f"largura útil {canvas_width_m:.2f} m."
                ]
                if ocr_scale_applied
                else []
            ),
            *(
                [
                    f"OCR associou textos a {len(reference_rooms)} ambiente(s), mas não houve "
                    "concordância suficiente para substituir a largura informada."
                ]
                if reference_rooms and not ocr_scale_applied
                else []
            ),
            *(
                ["Paredes finas foram reconstruídas pelo fallback de linhas; confira espessuras e encontros."]
                if str(result.get("wall_geometry_source", "")).startswith("single-line")
                else []
            ),
        ],
        "paredes": walls,
        "aberturas": editor_openings,
        "laje": {
            "contorno": slab_contour,
            "piso": {"ativo": slab_active, "espessura": 0.12},
            "teto": {"ativo": False, "espessura": 0.12},
        },
        "spaces": [],
        "raster_2d": diagnostics,
        "room_ocr": room_ocr,
        "pre_wall": result,
    }


def pre_wall_image_to_editor_model(
    image_path: Path,
    output_dir: Path,
    *,
    canvas_width_m: float,
    metric_refinement: bool = True,
) -> dict[str, Any]:
    result = run_pre_wall_pipeline(
        image_path,
        output_dir,
        canvas_width_m=canvas_width_m,
        metric_refinement=metric_refinement,
    )
    return pre_wall_result_to_editor_model(
        result,
        output_dir / "building_crop.png",
        source_name=image_path.name,
    )
