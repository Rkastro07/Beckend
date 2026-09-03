"""OCR local de rasters vinculados ao CAD.

O OCR é evidência auxiliar: preserva textos e cotas e, quando encontra uma
expressão arquitetônica forte, entrega uma pista à gramática geométrica. A
geometria vetorial continua sendo a autoridade para posição e dimensões.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np

try:
    from .cad_object_grammar_v3 import (
        classify_architectural_text,
        dimensions_from_text,
        normalize_text,
    )
except ImportError:  # suporte aos scripts legados executados pelo diretório
    from cad_object_grammar_v3 import (
        classify_architectural_text,
        dimensions_from_text,
        normalize_text,
    )


OCR_ENGINE = "windows-media-ocr"
_SCRIPT = Path(__file__).resolve().parents[1] / "runtime" / "windows_ocr.ps1"


class RasterOcrError(RuntimeError):
    """Falha controlada ao executar ou alinhar o OCR."""


_DIMENSION_TOKEN = re.compile(
    r"(?<![\d])(?P<number>\d{1,5}(?:[.,]\d{1,3})?)\s*"
    r"(?P<unit>mm|cm|m)?(?![\d])",
    re.IGNORECASE,
)
_NON_DIMENSION_CONTEXT = re.compile(
    r"(?:escala|scale|folha|sheet|rev(?:is[aã]o)?|data|date|"
    r"área|area|m²|m2|sqm|n[º°o]\.?\s*|\d+\s*:)",
    re.IGNORECASE,
)
_OBJECT_SIZE_CONTEXT = re.compile(
    r"(?:\d\s*[x×]\s*\d|\d\s*[wv]{1,2}\s*x\s*\d|"
    r"w\W{0,2}h\s*\d|(?:^|\s)\.?h\s*\d|w\s*x\s*\d|"
    r"window|janela|door|porta|bed|cama|table|mesa)",
    re.IGNORECASE,
)


def _ocr_cache_path(image_path: Path) -> Path:
    digest = hashlib.sha256()
    digest.update(b"windows-media-ocr-v1\0")
    with image_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    default_root = Path(__file__).resolve().parents[1] / ".runtime" / "cache" / "raster_ocr"
    cache_root = Path(os.environ.get("RASTER_OCR_CACHE_ROOT", default_root))
    return cache_root / f"{digest.hexdigest()}.json"


def run_windows_ocr(image_path: str | Path, timeout: float = 60.0) -> dict:
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise RasterOcrError(f"Imagem vinculada não encontrada: {image_path}")
    cache_path = _ocr_cache_path(image_path)
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and isinstance(cached.get("lines"), list):
                cached["cache_hit"] = True
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    if not _SCRIPT.is_file():
        raise RasterOcrError(f"Script do Windows OCR ausente: {_SCRIPT}")
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(_SCRIPT),
                "-ImagePath",
                str(image_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise RasterOcrError(
            f"OCR da imagem excedeu {timeout:.0f} segundos."
        ) from exc
    stdout = completed.stdout.decode("utf-8-sig", errors="replace").strip()
    stderr = completed.stderr.decode("utf-8-sig", errors="replace").strip()
    if completed.returncode != 0:
        raise RasterOcrError(stderr or stdout or "Falha no Windows OCR.")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RasterOcrError(
            f"Windows OCR retornou JSON inválido: {stdout[-500:]}"
        ) from exc
    if isinstance(result, dict) and isinstance(result.get("lines"), list):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(result, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
    return result


def _dimension_value_to_meters(
    number_text: str,
    unit: str | None,
    *,
    integer_mode: str | None = None,
) -> tuple[float, float, str] | None:
    normalized = number_text.replace(",", ".")
    try:
        number = float(normalized)
    except ValueError:
        return None
    if not math.isfinite(number) or number <= 0:
        return None

    explicit_unit = (unit or "").lower()
    if explicit_unit == "mm":
        meters, confidence, assumption = number / 1000.0, 0.98, "explicit-mm"
    elif explicit_unit == "cm":
        meters, confidence, assumption = number / 100.0, 0.98, "explicit-cm"
    elif explicit_unit == "m":
        meters, confidence, assumption = number, 0.98, "explicit-m"
    elif "," in number_text or "." in number_text:
        meters, confidence, assumption = number, 0.86, "decimal-meters"
    elif integer_mode == "mm" and number >= 10:
        meters, confidence, assumption = number / 1000.0, 0.64, "integer-mm-context"
    elif number >= 1000:
        meters, confidence, assumption = number / 1000.0, 0.66, "integer-mm"
    elif number >= 100:
        meters, confidence, assumption = number / 100.0, 0.54, "integer-cm"
    elif number >= 10:
        meters, confidence, assumption = number / 100.0, 0.30, "short-integer-cm"
    else:
        return None
    if not 0.05 <= meters <= 50.0:
        return None
    return float(meters), confidence, assumption


def _dimension_kind(line_text: str, value_m: float) -> str:
    """Separa cotas lineares de medidas que não devem calibrar a planta."""
    if _OBJECT_SIZE_CONTEXT.search(line_text):
        return "object-size"
    if value_m <= 0.35:
        return "thickness"
    return "linear"


def dimension_candidates_from_ocr(
    ocr_result: dict,
    *,
    canvas_width_m: float,
    canvas_size: int = 256,
) -> list[dict]:
    """Extrai cotas candidatas e as alinha ao canvas quadrado do Raster2Seq."""
    raster_width = float(ocr_result.get("width") or 0)
    raster_height = float(ocr_result.get("height") or 0)
    if min(raster_width, raster_height, canvas_width_m, canvas_size) <= 0:
        raise RasterOcrError("Dimensões inválidas ao alinhar cotas OCR.")

    resize_scale = min(canvas_size / raster_width, canvas_size / raster_height)
    padded_width = raster_width * resize_scale
    padded_height = raster_height * resize_scale
    pad_x = (canvas_size - padded_width) / 2.0
    pad_y = (canvas_size - padded_height) / 2.0
    world_scale = canvas_width_m / float(canvas_size)

    def aligned_box(item: dict) -> tuple[dict, dict]:
        x0 = pad_x + float(item.get("x") or 0) * resize_scale
        y0 = pad_y + float(item.get("y") or 0) * resize_scale
        x1 = x0 + float(item.get("width") or 0) * resize_scale
        y1 = y0 + float(item.get("height") or 0) * resize_scale
        bbox = {
            "xmin": round(x0 * world_scale, 5),
            "ymin": round((canvas_size - y1) * world_scale, 5),
            "xmax": round(x1 * world_scale, 5),
            "ymax": round((canvas_size - y0) * world_scale, 5),
        }
        position = {
            "x": round((bbox["xmin"] + bbox["xmax"]) / 2.0, 5),
            "y": round((bbox["ymin"] + bbox["ymax"]) / 2.0, 5),
        }
        return bbox, position

    # Plantas técnicas normalmente usam uma unidade única. A presença de uma
    # cota inteira >= 1000 é uma pista forte de que outros inteiros, como 800,
    # também estão em milímetros (0,80 m), e não em centímetros (8,00 m).
    integer_mode = None
    for line in ocr_result.get("lines") or []:
        line_text = str(line.get("text") or "").strip()
        if not line_text or _NON_DIMENSION_CONTEXT.search(line_text):
            continue
        for word in line.get("words") or [line]:
            for match in _DIMENSION_TOKEN.finditer(str(word.get("text") or "")):
                number_text = match.group("number")
                if (
                    not match.group("unit")
                    and "," not in number_text
                    and "." not in number_text
                    and float(number_text) >= 1000
                ):
                    integer_mode = "mm"
                    break
            if integer_mode:
                break
        if integer_mode:
            break

    candidates = []
    for line_index, line in enumerate(ocr_result.get("lines") or []):
        line_text = str(line.get("text") or "").strip()
        if not line_text or _NON_DIMENSION_CONTEXT.search(line_text):
            continue
        words = line.get("words") or [line]
        for word_index, word in enumerate(words):
            word_text = str(word.get("text") or "").strip()
            for match_index, match in enumerate(_DIMENSION_TOKEN.finditer(word_text)):
                parsed = _dimension_value_to_meters(
                    match.group("number"),
                    match.group("unit"),
                    integer_mode=integer_mode,
                )
                if parsed is None:
                    continue
                value_m, confidence, assumption = parsed
                bbox, position = aligned_box(word)
                candidates.append({
                    "id": f"OCR-DIM-{line_index + 1}-{word_index + 1}-{match_index + 1}",
                    "text": match.group(0).strip(),
                    "line_text": line_text,
                    "value_m": round(value_m, 5),
                    "confidence": confidence,
                    "assumption": assumption,
                    "kind": _dimension_kind(line_text, value_m),
                    "position": position,
                    "bbox": bbox,
                })
    return candidates


def extract_raster_dimensions(
    image_path: str | Path,
    *,
    canvas_width_m: float,
    canvas_size: int = 256,
    runner=run_windows_ocr,
) -> tuple[list[dict], dict]:
    """Executa OCR sem tornar uma falha de leitura fatal para o importador."""
    try:
        result = runner(image_path)
        candidates = dimension_candidates_from_ocr(
            result,
            canvas_width_m=canvas_width_m,
            canvas_size=canvas_size,
        )
        return candidates, {
            "status": "ok",
            "engine": result.get("engine", OCR_ENGINE),
            "language": result.get("language"),
            "cache_hit": bool(result.get("cache_hit")),
            "line_count": len(result.get("lines") or []),
            "dimension_count": len(candidates),
        }
    except Exception as exc:
        return [], {
            "status": "failed",
            "engine": OCR_ENGINE,
            "error": str(exc),
            "line_count": 0,
            "dimension_count": 0,
        }


def pixel_to_cad(
    x: float,
    y: float,
    image_record: dict,
    *,
    raster_width: float,
    raster_height: float,
) -> np.ndarray:
    """Converte pixel top-left para coordenada CAD usando a entidade IMAGE."""
    declared_width, declared_height = image_record["image_size"]
    if min(raster_width, raster_height, declared_width, declared_height) <= 0:
        raise RasterOcrError("Dimensões inválidas para alinhamento raster/CAD.")
    px = float(x) * float(declared_width) / float(raster_width)
    py_from_bottom = (
        float(raster_height) - float(y)
    ) * float(declared_height) / float(raster_height)
    insert = np.asarray(image_record["insert_raw"], dtype=float)
    u_pixel = np.asarray(image_record["u_pixel_raw"], dtype=float)
    v_pixel = np.asarray(image_record["v_pixel_raw"], dtype=float)
    return insert + u_pixel * px + v_pixel * py_from_bottom


def _aligned_box(line: dict, image_record: dict, ocr_result: dict) -> dict:
    x0 = float(line["x"])
    y0 = float(line["y"])
    x1 = x0 + float(line["width"])
    y1 = y0 + float(line["height"])
    corners = np.asarray([
        pixel_to_cad(
            x0, y0, image_record,
            raster_width=ocr_result["width"],
            raster_height=ocr_result["height"],
        ),
        pixel_to_cad(
            x1, y0, image_record,
            raster_width=ocr_result["width"],
            raster_height=ocr_result["height"],
        ),
        pixel_to_cad(
            x1, y1, image_record,
            raster_width=ocr_result["width"],
            raster_height=ocr_result["height"],
        ),
        pixel_to_cad(
            x0, y1, image_record,
            raster_width=ocr_result["width"],
            raster_height=ocr_result["height"],
        ),
    ])
    center = np.mean(corners, axis=0)
    return {
        "position_raw": center,
        "cad_bbox_raw": {
            "xmin": float(np.min(corners[:, 0])),
            "ymin": float(np.min(corners[:, 1])),
            "xmax": float(np.max(corners[:, 0])),
            "ymax": float(np.max(corners[:, 1])),
        },
    }


def aligned_ocr_evidence(
    image_record: dict,
    *,
    runner=run_windows_ocr,
) -> tuple[list[dict], dict]:
    """Executa OCR e devolve evidências em coordenadas CAD brutas."""
    image_path = image_record.get("resolved_path")
    try:
        result = runner(image_path)
        lines = []
        semantic_records = []
        for index, line in enumerate(result.get("lines") or []):
            text = str(line.get("text") or "").strip()
            if not text:
                continue
            aligned = _aligned_box(line, image_record, result)
            meaning = classify_architectural_text(text)
            evidence = {
                "text": text,
                "normalized_text": normalize_text(text),
                "engine": result.get("engine", OCR_ENGINE),
                "language": result.get("language"),
                "image_handle": image_record.get("handle", ""),
                "image_path": str(image_path or ""),
                "pixel_bbox": {
                    key: float(line[key])
                    for key in ("x", "y", "width", "height")
                },
                **aligned,
            }
            lines.append(evidence)
            if meaning is None:
                continue
            record = {
                "text": text,
                "normalized_text": normalize_text(text),
                "rotation": float(result.get("text_angle") or 0.0),
                "layer": str(image_record.get("layer") or "Linked image"),
                "entity_type": "RASTER_OCR",
                "handle": (
                    f"{image_record.get('handle', '')}:ocr:{index}"
                ),
                "source_kind": "raster-ocr",
                **aligned,
                **meaning,
                **dimensions_from_text(text, meaning["subtype"]),
            }
            semantic_records.append(record)
        return semantic_records, {
            "status": "ok",
            "engine": result.get("engine", OCR_ENGINE),
            "language": result.get("language"),
            "width": int(result.get("width") or 0),
            "height": int(result.get("height") or 0),
            "lines": lines,
            "line_count": len(lines),
            "semantic_cue_count": len(semantic_records),
        }
    except Exception as exc:
        return [], {
            "status": "failed",
            "engine": OCR_ENGINE,
            "error": str(exc),
            "lines": [],
            "line_count": 0,
            "semantic_cue_count": 0,
        }


def serialize_ocr_diagnostic(diagnostic: dict, scale: float) -> dict:
    """Remove arrays NumPy e expõe as coordenadas alinhadas em metros."""
    serialized = dict(diagnostic)
    serialized_lines = []
    for line in diagnostic.get("lines") or []:
        item = dict(line)
        position = np.asarray(item.pop("position_raw"), dtype=float) * scale
        raw_box = item.pop("cad_bbox_raw")
        item["cad_position"] = {
            "x": round(float(position[0]), 4),
            "y": round(float(position[1]), 4),
        }
        item["cad_bbox"] = {
            key: round(float(value) * scale, 4)
            for key, value in raw_box.items()
        }
        serialized_lines.append(item)
    serialized["lines"] = serialized_lines
    return serialized
