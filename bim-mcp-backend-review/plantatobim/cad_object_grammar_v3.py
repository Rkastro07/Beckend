"""Gramática determinística de objetos arquitetônicos em desenhos CAD.

Este módulo não gera IFC e não depende de LLM. Ele transforma evidências
preservadas do CAD (texto, bloco, arco e segmentos) em candidatos semânticos
que o detector geométrico consegue hospedar nas paredes.

A primeira versão cobre:

* portas e janelas indicadas por texto/cota próximos da geometria;
* portões e portas seccionais largas;
* blocos genéricos com arco de abertura;
* diagnóstico explícito das evidências utilizadas.
"""
from __future__ import annotations

import math
import re
import unicodedata

import numpy as np


GRAMMAR_VERSION = "cad-object-grammar-v3.0"

_DOOR_TOKENS = (
    "door", "doors", "porta", "portas", "puerta", "puertas", "deur",
)
_GARAGE_DOOR_TOKENS = (
    "garage door", "garage doors", "garagedoor", "sectional door",
    "sectional overhead door", "overhead door", "roller door",
    "rolling door", "porta seccional", "portao", "portao de garagem",
    "porta de garagem",
)
_WINDOW_TOKENS = (
    "window", "windows", "janela", "janelas", "ventana", "ventanas",
    "fenster", "glazing", "glazed", "vidro",
)
_NON_ELEMENT_CONTEXT = (
    "schedule", "legend", "legenda", "lista", "table", "tabela",
    "detail", "detalhe", "note", "nota", "typical", "tipico",
)


def normalize_text(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("\n", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.,x×/ -]+", " ", text)).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized = normalize_text(phrase)
    return f" {normalized} " in f" {text} "


def classify_architectural_text(value: str | None) -> dict | None:
    """Retorna o significado arquitetônico de um TEXT/MTEXT relevante."""
    text = normalize_text(value)
    if not text or any(_contains_phrase(text, token) for token in _NON_ELEMENT_CONTEXT):
        return None
    if any(_contains_phrase(text, token) for token in _GARAGE_DOOR_TOKENS):
        return {
            "role": "door",
            "subtype": "garage_door",
            "confidence": 0.96,
            "reason": "garage-door-text",
        }
    if any(_contains_phrase(text, token) for token in _DOOR_TOKENS):
        return {
            "role": "door",
            "subtype": "door",
            "confidence": 0.86,
            "reason": "door-text",
        }
    if any(_contains_phrase(text, token) for token in _WINDOW_TOKENS):
        return {
            "role": "window",
            "subtype": "window",
            "confidence": 0.86,
            "reason": "window-text",
        }
    return None


def _dimension_to_meters(value: str) -> float | None:
    try:
        number = float(value.replace(",", "."))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    # Convenções frequentes em anotações: 4800 mm, 480 cm ou 4.800 m.
    if number >= 1000:
        number /= 1000.0
    elif number >= 20:
        number /= 100.0
    return number if 0.05 <= number <= 30.0 else None


def dimensions_from_text(value: str | None, subtype: str | None = None) -> dict:
    """Extrai largura/altura de notações como 2,180 x 4,800 ou 820/2100."""
    text = normalize_text(value)
    matches = re.findall(
        r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(?:x|×|/)\s*(\d+(?:[.,]\d+)?)(?!\d)",
        text,
    )
    values = []
    for pair in matches:
        converted = [_dimension_to_meters(item) for item in pair]
        values.extend(item for item in converted if item is not None)
    if len(values) < 2:
        return {}

    first, second = values[0], values[1]
    if subtype == "garage_door":
        width, height = max(first, second), min(first, second)
    else:
        # Em esquadrias comuns a altura costuma ser a maior das duas medidas.
        width, height = min(first, second), max(first, second)
    return {
        "declared_width": float(width),
        "declared_height": float(height),
    }


def text_record(entity, layer: str) -> dict | None:
    """Materializa texto CAD sem depender do tipo concreto do ezdxf."""
    entity_type = entity.dxftype()
    if entity_type not in ("TEXT", "MTEXT", "ATTRIB", "ATTDEF"):
        return None
    try:
        if entity_type == "MTEXT":
            content = entity.plain_text()
            insert = entity.dxf.insert
        else:
            content = str(entity.dxf.text)
            insert = entity.dxf.insert
    except Exception:
        return None
    meaning = classify_architectural_text(content)
    if meaning is None:
        return None
    try:
        rotation = float(getattr(entity.dxf, "rotation", 0.0) or 0.0)
    except Exception:
        rotation = 0.0
    dimensions = dimensions_from_text(content, meaning["subtype"])
    return {
        "text": str(content),
        "normalized_text": normalize_text(content),
        "position_raw": np.array([float(insert.x), float(insert.y)], dtype=float),
        "rotation": rotation,
        "layer": str(layer or "0"),
        "entity_type": entity_type,
        "handle": str(getattr(entity.dxf, "handle", "") or ""),
        **meaning,
        **dimensions,
    }


def generic_insert_candidate(record: dict, scale: float) -> dict | None:
    """Reconhece bloco anônimo de porta pela gramática folha + arco."""
    if not record.get("has_arc"):
        return None
    points = np.asarray(record.get("pts_raw", []), dtype=float) * float(scale)
    if len(points) < 2:
        return None
    spans = np.ptp(points, axis=0)
    extent = float(max(spans))
    if not 0.35 <= extent <= 3.20:
        return None
    return {
        "tipo": "door",
        "pts": points,
        "origem": "cad-block-geometry-v3",
        "confidence": 0.78,
        "source_layer": record.get("layer", ""),
        "block_name": record.get("block_name", ""),
        "semantic_subtype": "swing_door",
        "semantic_reason": "generic-block-with-swing-arc",
        "max_width": 3.20,
    }


def _point_segment_distance(point, start, end) -> float:
    vector = end - start
    denominator = float(vector @ vector)
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start))
    position = float(np.clip(((point - start) @ vector) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + vector * position)))


def text_geometry_candidates(
    text_records: list[dict],
    raw_by_layer: dict[str, list[tuple]],
    selected_layers: set[str],
    scale: float,
) -> tuple[list[dict], list[dict]]:
    """Liga textos arquitetônicos a segmentos de dimensões compatíveis.

    Retorna candidatos prontos para hospedagem e diagnósticos serializáveis.
    A associação exige uma palavra arquitetônica forte e, para portas/janelas
    comuns, uma dimensão explícita. Portões podem usar apenas o texto forte.
    """
    segments = []
    for layer in sorted(selected_layers):
        for raw in raw_by_layer.get(layer, []):
            x1, y1, x2, y2, _ = raw
            start = np.array([x1, y1], dtype=float) * scale
            end = np.array([x2, y2], dtype=float) * scale
            length = float(np.linalg.norm(end - start))
            if length >= 0.25:
                segments.append({
                    "start": start,
                    "end": end,
                    "length": length,
                    "layer": layer,
                })

    candidates = []
    diagnostics = []
    used = set()
    for cue in text_records:
        declared_width = cue.get("declared_width")
        is_garage = cue.get("subtype") == "garage_door"
        if declared_width is None and not is_garage:
            diagnostics.append({
                "text": cue["text"],
                "role": cue["role"],
                "status": "insufficient-dimensions",
                "reason": cue["reason"],
            })
            continue

        position = np.asarray(cue["position_raw"], dtype=float) * scale
        search_radius = max(
            1.50,
            min(3.50, float(declared_width or 3.0) * 0.75),
        )
        ranked = []
        for index, segment in enumerate(segments):
            length = segment["length"]
            if declared_width is not None:
                tolerance = max(0.15, float(declared_width) * 0.22)
                length_delta = abs(length - float(declared_width))
                if length_delta > tolerance:
                    continue
            elif not 1.80 <= length <= 6.50:
                continue
            distance = _point_segment_distance(
                position, segment["start"], segment["end"],
            )
            if distance > search_radius:
                continue
            relative_delta = (
                abs(length - float(declared_width)) / max(float(declared_width), 1e-9)
                if declared_width is not None else 0.25
            )
            ranked.append((
                relative_delta + distance / max(search_radius, 1e-9),
                distance,
                index,
                segment,
            ))

        if not ranked:
            diagnostics.append({
                "text": cue["text"],
                "role": cue["role"],
                "status": "no-compatible-geometry",
                "reason": cue["reason"],
            })
            continue

        _, distance, segment_index, segment = min(ranked, key=lambda item: item[:3])
        identity = (
            segment["layer"],
            tuple(np.round(segment["start"], 6)),
            tuple(np.round(segment["end"], 6)),
            cue["role"],
        )
        if identity in used:
            continue
        used.add(identity)

        maximum = 6.50 if is_garage else 3.20
        candidate = {
            "tipo": cue["role"],
            "pts": np.vstack([segment["start"], segment["end"]]),
            "origem": (
                "cad-raster-ocr-geometry-v3"
                if cue.get("source_kind") == "raster-ocr"
                else "cad-text-geometry-v3"
            ),
            "confidence": float(cue["confidence"]),
            "source_layer": segment["layer"],
            "block_name": "",
            "source_text": cue["text"],
            "semantic_subtype": cue["subtype"],
            "semantic_reason": cue["reason"],
            "max_width": maximum,
        }
        for field in ("declared_width", "declared_height"):
            if cue.get(field) is not None:
                candidate[field] = float(cue[field])
        candidates.append(candidate)
        diagnostics.append({
            "text": cue["text"],
            "role": cue["role"],
            "subtype": cue["subtype"],
            "status": "matched",
            "source_layer": segment["layer"],
            "segment_length": round(segment["length"], 4),
            "distance": round(float(distance), 4),
            "declared_width": cue.get("declared_width"),
            "declared_height": cue.get("declared_height"),
            "reason": cue["reason"],
            "source_kind": cue.get("source_kind", "cad-text"),
        })
    return candidates, diagnostics
