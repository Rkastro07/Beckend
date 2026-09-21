"""Detector CAD V2 para o modelador Planta-to-BIM.

O DXF preserva muito mais informacao do que apenas linhas: unidades, layers,
nomes de blocos e tipos de entidades. Este modulo combina essas evidencias com
pareamento geometrico e mantem um diagnostico editavel para o frontend.

DWG usa exatamente o mesmo detector depois da conversao local para DXF.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import math
from pathlib import Path
import re
import unicodedata

import ezdxf
import numpy as np

import planta_to_ifc_v1 as pl
from cad_object_grammar_v3 import (
    GRAMMAR_VERSION,
    generic_insert_candidate,
    text_geometry_candidates,
    text_record,
)
from cad_raster_ocr import aligned_ocr_evidence, serialize_ocr_diagnostic


CAD_ROLES = ("wall", "door", "window", "opening", "ignore")

_ROLE_TOKENS = {
    "door": (
        "door", "doors", "porta", "portas", "puerta", "puertas",
        "deur", "dorr", "garage door", "garagedoor",
    ),
    "window": (
        "window", "windows", "wndw", "janela", "janelas", "ventana",
        "ventanas", "fenster", "glaz", "vidro", "vitre",
    ),
    "opening": (
        "opening", "openings", "abertura", "aberturas", "esquadria",
        "esquadrias", "pueryventa", "puerta y ventana", "vano", "vaos",
    ),
    "wall": (
        "wall", "walls", "parede", "paredes", "pared", "muro", "muros",
        "vegg", "partition", "partitions", "tabique", "masonry", "casco",
        "stem wall", "stemwall",
    ),
}

_IGNORE_TOKENS = (
    "annot", "annotation", "anot", "note", "nota", "text", "texto",
    "dim", "cota", "dimension", "title", "carimbo", "legend", "legenda",
    "hatch", "hachura", "pattern", "grid", "axis", "eixo", "level",
    "room", "space", "area", "furniture", "furn", "mob", "mueble",
    "mobiliario", "fixture", "equip", "casework", "case", "cabinet",
    "toilet", "sanitary", "plumb", "elect", "lighting", "symbol",
    "detail", "section", "sect", "header", "elevation", "roof", "ceiling",
    "floor", "flor", "slab", "gulv", "beam", "column", "coluna",
    "footing", "footer", "foundation", "insulation", "membrane",
    "landscape", "terrain", "site", "viewport", "defpoints",
)

_INSUNITS_TO_METERS = {
    1: 0.0254,       # inch
    2: 0.3048,       # foot
    3: 1609.344,     # mile
    4: 0.001,        # millimeter
    5: 0.01,         # centimeter
    6: 1.0,          # meter
    7: 1000.0,       # kilometer
    8: 2.54e-8,      # microinch
    9: 2.54e-5,      # mil
    10: 0.9144,      # yard
    11: 1e-10,       # angstrom
    12: 1e-9,        # nanometer
    13: 1e-6,        # micron
    14: 0.1,         # decimeter
    15: 10.0,        # decameter
    16: 100.0,       # hectometer
}


def normalize_cad_name(value: str | None) -> str:
    """Normaliza nomes de layer/bloco sem perder prefixos de XREF."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("$0$", " ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _contains_token(name: str, token: str) -> bool:
    normalized = normalize_cad_name(token)
    if not normalized:
        return False
    padded = f" {name} "
    return f" {normalized} " in padded or normalized.replace(" ", "") in name.replace(" ", "")


def classify_name(value: str | None) -> str | None:
    """Classifica um nome CAD isolado."""
    name = normalize_cad_name(value)
    if not name:
        return None
    # Esquadrias sao mais especificas que parede: A-GARAGE-DOOR nao pode
    # virar parede apenas porque o XREF contem a palavra WALL.
    for role in ("door", "window", "opening", "wall"):
        if any(_contains_token(name, token) for token in _ROLE_TOKENS[role]):
            return role
    if any(_contains_token(name, token) for token in _IGNORE_TOKENS):
        return "ignore"
    return None


def _override_role(layer: str, layer_map: dict | None) -> str | None:
    if not layer_map:
        return None
    normalized = normalize_cad_name(layer)
    for key in (layer, normalized):
        value = layer_map.get(key)
        if value is None:
            continue
        role = str(value).strip().lower()
        if role in ("", "auto", "none", "null"):
            return None
        if role in CAD_ROLES:
            return role
    return None


def classify_layer(layer: str, layer_map: dict | None = None) -> tuple[str | None, str]:
    override = _override_role(layer, layer_map)
    if override:
        return override, "manual"
    role = classify_name(layer)
    return role, ("layer-name" if role else "unclassified")


def classify_entity(
    layer: str,
    block_name: str | None = None,
    layer_map: dict | None = None,
) -> tuple[str | None, str]:
    """Classifica INSERT pelo bloco antes do layer."""
    override = _override_role(layer, layer_map)
    if override:
        return override, "manual"
    block_role = classify_name(block_name)
    if block_role:
        return block_role, "block-name"
    return classify_layer(layer)


def _effective_layer(entity, inherited_layer: str | None) -> str:
    own = str(getattr(entity.dxf, "layer", "") or "")
    if own in ("", "0") and inherited_layer:
        return inherited_layer
    return own or inherited_layer or "0"


def _safe_virtual_entities(entity):
    try:
        return list(entity.virtual_entities())
    except Exception:
        return []


def _opening_points(entity):
    points = []
    has_arc = False

    def visit(item, depth=0):
        nonlocal has_arc
        if depth > 8:
            return
        if item.dxftype() == "INSERT":
            children = _safe_virtual_entities(item)
            if not children:
                try:
                    insert = item.dxf.insert
                    points.append((float(insert.x), float(insert.y)))
                except Exception:
                    pass
                return
            for child in children:
                visit(child, depth + 1)
            return
        if item.dxftype() == "ARC":
            has_arc = True
        # O arco da folha aberta nao define a largura do vao; fica apenas como
        # evidencia de porta.
        if item.dxftype() == "ARC":
            try:
                center = item.dxf.center
                radius = float(item.dxf.radius)
                a0 = math.radians(float(item.dxf.start_angle))
                a1 = math.radians(float(item.dxf.end_angle))
                points.extend([
                    (float(center.x), float(center.y)),
                    (float(center.x + radius * math.cos(a0)),
                     float(center.y + radius * math.sin(a0))),
                    (float(center.x + radius * math.cos(a1)),
                     float(center.y + radius * math.sin(a1))),
                ])
            except Exception:
                pass
            return
        for x1, y1, x2, y2 in pl._segmentos_entidade_dxf(item):
            points.extend(((x1, y1), (x2, y2)))

    visit(entity)
    if not points and entity.dxftype() == "INSERT":
        try:
            insert = entity.dxf.insert
            points.append((float(insert.x), float(insert.y)))
        except Exception:
            pass
    if not points:
        return np.empty((0, 2), dtype=float), has_arc
    return np.unique(np.asarray(points, dtype=float), axis=0), has_arc


def _new_layer_stats(name: str) -> dict:
    return {
        "name": name,
        "entities": 0,
        "segments": 0,
        "blocks": 0,
        "entity_types": Counter(),
        "block_names": Counter(),
    }


def _collect_document(
    doc,
    layer_map,
    source_dir=None,
    linked_image_override=None,
):
    stats = {
        str(layer.dxf.name): _new_layer_stats(str(layer.dxf.name))
        for layer in doc.layers
    }
    explicit_by_layer = defaultdict(list)
    unknown_by_layer = defaultdict(list)
    opening_inserts = []
    opening_loose = []
    generic_inserts = []
    semantic_texts = []
    linked_images = []

    def visit(entity, inherited_layer=None, depth=0):
        if depth > 8:
            return
        layer = _effective_layer(entity, inherited_layer)
        info = stats.setdefault(layer, _new_layer_stats(layer))
        info["entities"] += 1
        info["entity_types"][entity.dxftype()] += 1
        block_name = (
            str(getattr(entity.dxf, "name", "") or "")
            if entity.dxftype() == "INSERT"
            else ""
        )
        role, reason = classify_entity(layer, block_name, layer_map)

        if entity.dxftype() == "IMAGE":
            try:
                stored_path = str(entity.image_def.dxf.filename or "")
            except Exception:
                stored_path = ""
            resolved = Path(stored_path)
            if stored_path and not resolved.is_absolute() and source_dir is not None:
                resolved = Path(source_dir) / resolved
            resolution_source = "cad-reference"
            override = (
                Path(linked_image_override).resolve()
                if linked_image_override else None
            )
            if (not stored_path or not resolved.is_file()) and override is not None:
                resolved = override
                resolution_source = "uploaded-companion"
            try:
                insert = entity.dxf.insert
                u_pixel = entity.dxf.u_pixel
                v_pixel = entity.dxf.v_pixel
                image_size = entity.dxf.image_size
                transform = {
                    "insert_raw": [float(insert.x), float(insert.y)],
                    "u_pixel_raw": [float(u_pixel.x), float(u_pixel.y)],
                    "v_pixel_raw": [float(v_pixel.x), float(v_pixel.y)],
                    "image_size": [
                        float(image_size.x), float(image_size.y),
                    ],
                }
            except Exception:
                transform = {
                    "insert_raw": [0.0, 0.0],
                    "u_pixel_raw": [1.0, 0.0],
                    "v_pixel_raw": [0.0, 1.0],
                    "image_size": [0.0, 0.0],
                }
            linked_images.append({
                "layer": layer,
                "stored_path": stored_path,
                "resolved_path": (
                    str(resolved) if (stored_path or override is not None) else ""
                ),
                "available": bool(resolved.is_file()),
                "handle": str(getattr(entity.dxf, "handle", "") or ""),
                "resolution_source": resolution_source,
                **transform,
            })
            return

        semantic_text = text_record(entity, layer)
        if semantic_text is not None:
            semantic_texts.append(semantic_text)
        if entity.dxftype() in ("TEXT", "MTEXT", "ATTRIB", "ATTDEF"):
            return

        if entity.dxftype() == "INSERT":
            info["blocks"] += 1
            if block_name:
                info["block_names"][block_name] += 1
            for attribute in getattr(entity, "attribs", []):
                semantic_attribute = text_record(attribute, layer)
                if semantic_attribute is not None:
                    semantic_texts.append(semantic_attribute)
            if role in ("door", "window", "opening"):
                points, has_arc = _opening_points(entity)
                if len(points):
                    opening_inserts.append({
                        "role": role,
                        "pts_raw": points,
                        "has_arc": has_arc,
                        "layer": layer,
                        "block_name": block_name,
                        "reason": reason,
                    })
                return
            if role == "ignore":
                return
            points, has_arc = _opening_points(entity)
            layer_role, _ = classify_layer(layer, layer_map)
            if has_arc and len(points) >= 2 and layer_role != "ignore":
                # Blocos anônimos de porta frequentemente perdem o nome
                # semântico, mas preservam a folha e o arco. Não explodimos
                # essa geometria no conjunto de paredes.
                generic_inserts.append({
                    "pts_raw": points,
                    "has_arc": True,
                    "layer": layer,
                    "block_name": block_name,
                })
                return
            children = _safe_virtual_entities(entity)
            for child in children:
                visit(child, layer, depth + 1)
            return

        segments = pl._segmentos_entidade_dxf(entity)
        info["segments"] += len(segments)
        if not segments:
            return
        raw = [(x1, y1, x2, y2, layer) for x1, y1, x2, y2 in segments]
        if role == "wall":
            explicit_by_layer[layer].extend(raw)
        elif role in ("door", "window", "opening"):
            points = np.asarray(
                [(value[0], value[1]) for value in segments]
                + [(value[2], value[3]) for value in segments],
                dtype=float,
            )
            opening_loose.append({
                "role": role,
                "pts_raw": np.unique(points, axis=0),
                "has_arc": entity.dxftype() == "ARC",
                "layer": layer,
                "block_name": "",
                "reason": reason,
            })
        elif role != "ignore":
            unknown_by_layer[layer].extend(raw)

    for entity in doc.modelspace():
        visit(entity)
    return (
        stats,
        explicit_by_layer,
        unknown_by_layer,
        opening_inserts,
        opening_loose,
        generic_inserts,
        semantic_texts,
        linked_images,
    )


def _raw_extent(raw_segments) -> float:
    if not raw_segments:
        return 0.0
    xs = [value for segment in raw_segments for value in (segment[0], segment[2])]
    ys = [value for segment in raw_segments for value in (segment[1], segment[3])]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def _detect_scale(doc, raw_segments, forced):
    if forced is not None:
        return float(forced), "forced"
    extent = _raw_extent(raw_segments)
    unit_code = int(doc.header.get("$INSUNITS", 0) or 0)
    unit_scale = _INSUNITS_TO_METERS.get(unit_code)
    # Muitos DXFs declaram milimetros no header embora as coordenadas tenham
    # sido exportadas em centimetros/unidades de plotagem. Uma planta inteira
    # abaixo de 2 m e um forte sinal de header incorreto.
    if unit_scale and 2.0 <= extent * unit_scale <= 5000.0:
        return unit_scale, f"dxf-insunits-{unit_code}"
    return pl.detectar_escala_auto(raw_segments, extent), "geometry-auto"


def _orientation_score(segments) -> float:
    angles = []
    weights = []
    for a, b, _ in segments:
        vector = b - a
        length = float(np.linalg.norm(vector))
        if length < 0.25:
            continue
        angles.append(math.atan2(vector[1], vector[0]))
        weights.append(length)
    if not angles:
        return 0.0
    values = np.exp(4j * np.asarray(angles))
    return float(abs(np.average(values, weights=np.asarray(weights))))


def pair_wall_faces_v2(segments):
    """Pareia faces de paredes por faixas longitudinais.

    O pareador legado usa cada segmento apenas uma vez. Isso perde paredes em
    plantas arquitetonicas reais, porque uma face longa frequentemente encontra
    varias faces menores depois de encontros T, portas e janelas. Aqui a linha e
    dividida nos intervalos onde o conjunto de faces ativas muda; em cada faixa,
    faces vizinhas sao casadas sem cruzamento. Assim, uma face pode participar
    de varios pares em trechos diferentes, mas nunca gera duas paredes no mesmo
    trecho.
    """
    if not segments:
        return [], []

    angles = []
    for start, end, _ in segments:
        vector = end - start
        angles.append(math.atan2(vector[1], vector[0]) % math.pi)

    groups = []
    grouped = [False] * len(segments)
    for index in range(len(segments)):
        if grouped[index]:
            continue
        group = [index]
        grouped[index] = True
        for other in range(index + 1, len(segments)):
            if grouped[other]:
                continue
            delta = abs(angles[index] - angles[other])
            delta = min(delta, math.pi - delta)
            if delta <= pl.ANG_TOL:
                group.append(other)
                grouped[other] = True
        groups.append(group)

    wall_spans = []
    paired_intervals = defaultdict(list)
    thickness_epsilon = 1e-6

    for group in groups:
        if len(group) < 2:
            continue
        reference = max(
            group,
            key=lambda item: np.linalg.norm(
                segments[item][1] - segments[item][0]
            ),
        )
        direction = segments[reference][1] - segments[reference][0]
        unit = direction / np.linalg.norm(direction)
        # Orientacao canonica deixa os resultados deterministas mesmo quando a
        # LINE original foi desenhada no sentido oposto.
        if unit[0] < -1e-9 or (abs(unit[0]) <= 1e-9 and unit[1] < 0):
            unit = -unit
        normal = np.array([-unit[1], unit[0]])

        info = {}
        breakpoints = []
        for item in group:
            start, end, _ = segments[item]
            offset = float(normal @ ((start + end) / 2))
            first, second = sorted((
                float(unit @ start),
                float(unit @ end),
            ))
            info[item] = (offset, first, second)
            breakpoints.extend((first, second))
        breakpoints = sorted(set(round(value, 9) for value in breakpoints))

        atom_pairs = []
        for first, second in zip(breakpoints, breakpoints[1:]):
            if second - first <= 1e-8:
                continue
            midpoint = (first + second) / 2
            active = sorted(
                (
                    (info[item][0], item)
                    for item in group
                    if info[item][1] <= midpoint + 1e-8
                    and info[item][2] >= midpoint - 1e-8
                ),
                key=lambda value: (value[0], value[1]),
            )
            if len(active) < 2:
                continue

            @lru_cache(maxsize=None)
            def solve(left, right):
                if left >= right:
                    return 0, 0.0, ()
                # Primeiro maximiza a quantidade de paredes; no empate escolhe
                # o conjunto de faces mais proximas (espessura menor).
                best = solve(left + 1, right)
                best_key = (best[0], -best[1])
                for partner in range(left + 1, right + 1):
                    thickness = abs(active[partner][0] - active[left][0])
                    if not (
                        pl.ESP_MIN - thickness_epsilon
                        <= thickness
                        <= pl.ESP_MAX + thickness_epsilon
                    ):
                        continue
                    inside = solve(left + 1, partner - 1)
                    outside = solve(partner + 1, right)
                    candidate = (
                        1 + inside[0] + outside[0],
                        thickness + inside[1] + outside[1],
                        ((active[left][1], active[partner][1]),)
                        + inside[2] + outside[2],
                    )
                    key = (candidate[0], -candidate[1])
                    if key > best_key:
                        best = candidate
                        best_key = key
                return best

            _, _, pairs = solve(0, len(active) - 1)
            for face_a, face_b in pairs:
                pair = tuple(sorted((face_a, face_b)))
                atom_pairs.append((pair, first, second))

        # Junta atomos consecutivos do mesmo par antes de aplicar o comprimento
        # minimo. Isso preserva uma parede longa ainda que outras faces comecem
        # ou terminem no meio dela.
        by_pair = defaultdict(list)
        for pair, first, second in atom_pairs:
            by_pair[pair].append((first, second))
        for pair, intervals in by_pair.items():
            for first, second in _merge_intervals(intervals, tolerance=1e-7):
                length = second - first
                if length < pl.OVERLAP_MIN - 1e-7:
                    continue
                face_a, face_b = pair
                offset_a = info[face_a][0]
                offset_b = info[face_b][0]
                thickness = abs(offset_a - offset_b)
                axis_offset = (offset_a + offset_b) / 2
                start = unit * first + normal * axis_offset
                end = unit * second + normal * axis_offset
                wall_spans.append({
                    "eixo": (start, end),
                    "espessura": float(thickness),
                    "comprimento": float(length),
                    "layer": segments[face_a][2],
                })
                paired_intervals[face_a].append((first, second))
                paired_intervals[face_b].append((first, second))

    leftovers = []
    for index, (start, end, layer) in enumerate(segments):
        length = float(np.linalg.norm(end - start))
        covered = sum(
            second - first
            for first, second in _merge_intervals(
                paired_intervals.get(index, []),
                tolerance=1e-7,
            )
        )
        if length - covered >= pl.LEFTOVER_WARN:
            leftovers.append((start, end, layer))
    return wall_spans, leftovers


def _score_wall_layer(raw_segments, scale):
    segments = pl._segs_em_escala(raw_segments, scale)
    segments = [
        segment for segment in segments
        if float(np.linalg.norm(segment[1] - segment[0])) >= 0.25
    ]
    if len(segments) < 2:
        return {
            "confidence": 0.0, "pairs": 0, "paired_length": 0.0,
            "coverage": 0.0, "orientation": 0.0,
        }
    merged = pl.mesclar_colineares(segments, gap_max=0.30)
    walls, _ = pair_wall_faces_v2(merged)
    total_length = sum(float(np.linalg.norm(b - a)) for a, b, _ in segments)
    paired_length = sum(float(wall["comprimento"]) for wall in walls)
    coverage = min(1.0, 2.0 * paired_length / max(total_length, 1e-9))
    orientation = _orientation_score(segments)
    pair_factor = min(1.0, len(walls) / 6.0)
    confidence = 0.62 * coverage + 0.23 * pair_factor + 0.15 * orientation
    return {
        "confidence": round(float(confidence), 4),
        "pairs": len(walls),
        "paired_length": round(float(paired_length), 4),
        "coverage": round(float(coverage), 4),
        "orientation": round(float(orientation), 4),
    }


def _cluster_opening_records(records, scale, radius=0.45):
    if not records:
        return []
    scaled = []
    for record in records:
        points = np.asarray(record["pts_raw"], dtype=float) * scale
        if len(points):
            scaled.append({**record, "pts": points})
    parent = list(range(len(scaled)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(scaled)):
        for j in range(i + 1, len(scaled)):
            if scaled[i]["layer"] != scaled[j]["layer"]:
                continue
            distance = np.min(np.linalg.norm(
                scaled[i]["pts"][:, None, :] - scaled[j]["pts"][None, :, :],
                axis=2,
            ))
            if distance <= radius:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for index in range(len(scaled)):
        groups[find(index)].append(index)

    clustered = []
    for indexes in groups.values():
        items = [scaled[index] for index in indexes]
        roles = [item["role"] for item in items]
        direct = next((role for role in roles if role in ("door", "window")), None)
        points = np.vstack([item["pts"] for item in items])
        has_arc = any(item["has_arc"] for item in items)
        role = direct or _infer_generic_opening_role(points, has_arc)
        clustered.append({
            "tipo": role,
            "pts": points,
            "origem": "cad-layer-geometry",
            "confidence": 0.82 if direct else (0.72 if has_arc else 0.58),
            "source_layer": items[0]["layer"],
            "block_name": "",
        })
    return clustered


def _infer_generic_opening_role(points, has_arc):
    if has_arc:
        return "door"
    if len(points) < 2:
        return "door"
    centered = points - points.mean(axis=0)
    values = np.linalg.eigvalsh(centered.T @ centered)
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    thinness = float(values[0] / max(values[-1], 1e-9))
    # Linhas finas e compridas no vao sao tipicas de caixilho.
    return "window" if extent >= 1.20 and thinness < 0.15 else "door"


def _insert_opening_candidates(records, scale):
    candidates = []
    for record in records:
        points = np.asarray(record["pts_raw"], dtype=float) * scale
        role = record["role"]
        if role == "opening":
            role = _infer_generic_opening_role(points, record["has_arc"])
        candidates.append({
            "tipo": role,
            "pts": points,
            "origem": "cad-block",
            "confidence": (
                0.97 if record["reason"] == "block-name"
                else 0.88 if record["role"] in ("door", "window")
                else 0.65
            ),
            "source_layer": record["layer"],
            "block_name": record["block_name"],
        })
    return candidates


def _merge_intervals(intervals, tolerance=0.08):
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]


def _internal_gaps(intervals, length, minimum=0.45, maximum=3.00):
    merged = _merge_intervals(intervals)
    gaps = []
    for left, right in zip(merged, merged[1:]):
        start, end = left[1], right[0]
        width = end - start
        if minimum <= width <= maximum and start > 0.10 and end < length - 0.10:
            gaps.append((start, end))
    return gaps


def _near_wall_junction(point, wall_index, walls, radius):
    for index, wall in enumerate(walls):
        if index == wall_index:
            continue
        for endpoint in wall["eixo"]:
            if float(np.linalg.norm(endpoint - point)) <= radius:
                return True
    return False


def infer_gap_openings(raw_by_layer, walls, existing):
    """Encontra vaos presentes nas duas faces da mesma parede."""
    generated = []
    existing_by_wall = defaultdict(list)
    for opening in existing:
        existing_by_wall[opening["parede_idx"]].append(opening)

    for wall_index, wall in enumerate(walls):
        if wall.get("_cad_mode") != "paired":
            continue
        raw_segments = raw_by_layer.get(wall.get("layer"), [])
        if len(raw_segments) < 4:
            continue
        a, b = wall["eixo"]
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 1.0:
            continue
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        thickness = float(wall["espessura"])
        side_intervals = {1: [], -1: []}

        for p1, p2, _ in raw_segments:
            vector = p2 - p1
            segment_length = float(np.linalg.norm(vector))
            if segment_length < 0.10:
                continue
            segment_unit = vector / segment_length
            cross_2d = unit[0] * segment_unit[1] - unit[1] * segment_unit[0]
            if abs(float(cross_2d)) > math.sin(math.radians(3.0)):
                continue
            midpoint = (p1 + p2) / 2
            signed = float((midpoint - a) @ normal)
            if abs(abs(signed) - thickness / 2) > max(0.06, thickness * 0.35):
                continue
            start, end = sorted((float((p1 - a) @ unit), float((p2 - a) @ unit)))
            start, end = max(0.0, start), min(length, end)
            if end - start >= 0.10:
                side_intervals[1 if signed >= 0 else -1].append((start, end))

        positive = _internal_gaps(side_intervals[1], length)
        negative = _internal_gaps(side_intervals[-1], length)
        for first in positive:
            best = None
            for second in negative:
                overlap = min(first[1], second[1]) - max(first[0], second[0])
                if overlap < 0.30:
                    continue
                center_delta = abs(sum(first) / 2 - sum(second) / 2)
                candidate = (overlap - center_delta, second)
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is None:
                continue
            second = best[1]
            start = (first[0] + second[0]) / 2
            end = (first[1] + second[1]) / 2
            center = (start + end) / 2
            width = end - start
            if any(abs(center - item["s_centro"]) <= max(0.25, width * 0.35)
                   for item in existing_by_wall[wall_index]):
                continue
            point = a + unit * center
            if _near_wall_junction(
                point, wall_index, walls, max(0.30, width * 0.45)
            ):
                continue
            generated.append({
                "parede_idx": wall_index,
                "tipo": "door" if width <= 1.20 else "window",
                "s_centro": float(center),
                "largura": float(np.clip(width, 0.40, 3.00)),
                "origem": "cad-gap-paired-faces",
                "confidence": 0.58,
            })
            existing_by_wall[wall_index].append(generated[-1])
    return generated


def _dedupe_openings(openings):
    ordered = sorted(
        openings,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )
    kept = []
    for candidate in ordered:
        duplicate = any(
            existing["parede_idx"] == candidate["parede_idx"]
            and abs(existing["s_centro"] - candidate["s_centro"])
            <= max(0.20, min(existing["largura"], candidate["largura"]) * 0.30)
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda item: (item["parede_idx"], item["s_centro"]))


def _point_segment_distance(point, start, end):
    vector = end - start
    denominator = float(vector @ vector)
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start))
    position = float(np.clip(((point - start) @ vector) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + vector * position)))


def _segments_intersect(a1, a2, b1, b2):
    def orientation(p, q, r):
        return float((q[0] - p[0]) * (r[1] - p[1])
                     - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)
    if all(abs(value) <= 1e-9 for value in (o1, o2, o3, o4)):
        return (
            max(min(a1[0], a2[0]), min(b1[0], b2[0]))
            <= min(max(a1[0], a2[0]), max(b1[0], b2[0])) + 1e-9
            and max(min(a1[1], a2[1]), min(b1[1], b2[1]))
            <= min(max(a1[1], a2[1]), max(b1[1], b2[1])) + 1e-9
        )
    return o1 * o2 <= 1e-9 and o3 * o4 <= 1e-9


def _wall_distance(first, second):
    a1, a2 = first["eixo"]
    b1, b2 = second["eixo"]
    if _segments_intersect(a1, a2, b1, b2):
        return 0.0
    return min(
        _point_segment_distance(a1, b1, b2),
        _point_segment_distance(a2, b1, b2),
        _point_segment_distance(b1, a1, a2),
        _point_segment_distance(b2, a1, a2),
    )


def split_wall_regions(walls, connection_tolerance=2.50):
    """Agrupa plantas desconectadas desenhadas lado a lado no modelspace."""
    if not walls:
        return []
    parent = list(range(len(walls)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            tolerance = max(
                connection_tolerance,
                float(walls[i].get("espessura", 0.15)),
                float(walls[j].get("espessura", 0.15)),
            )
            if _wall_distance(walls[i], walls[j]) <= tolerance:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for index in range(len(walls)):
        groups[find(index)].append(index)

    raw_regions = []
    for indexes in groups.values():
        xs = [
            float(point[0])
            for index in indexes for point in walls[index]["eixo"]
        ]
        ys = [
            float(point[1])
            for index in indexes for point in walls[index]["eixo"]
        ]
        total_length = sum(
            float(walls[index]["comprimento"]) for index in indexes
        )
        raw_regions.append({
            "indexes": indexes,
            "n_walls": len(indexes),
            "total_length": total_length,
            "bbox": {
                "xmin": min(xs), "ymin": min(ys),
                "xmax": max(xs), "ymax": max(ys),
            },
        })

    substantial = [
        region for region in raw_regions
        if region["n_walls"] >= 2 and region["total_length"] >= 2.0
    ]
    if not substantial:
        substantial = raw_regions
    substantial.sort(key=lambda item: (
        item["bbox"]["xmin"], item["bbox"]["ymin"],
    ))
    for number, region in enumerate(substantial, 1):
        region["id"] = f"cad-region-{number}"
        region["name"] = f"Planta CAD {number}"
    return substantial


def parse_dxf_v2(
    dxf_path,
    *,
    escala_forcada=None,
    esp_default=0.15,
    layer_map=None,
    cad_region=None,
    linked_image=None,
):
    """DXF -> modelo editavel com diagnostico de classificacao CAD."""
    dxf_path = Path(dxf_path)
    doc = ezdxf.readfile(str(dxf_path))
    (
        stats,
        explicit_by_layer,
        unknown_by_layer,
        opening_inserts,
        opening_loose,
        generic_inserts,
        semantic_texts,
        linked_images,
    ) = _collect_document(
        doc,
        layer_map or {},
        dxf_path.parent,
        linked_image_override=linked_image,
    )

    all_raw = [
        segment
        for group in list(explicit_by_layer.values()) + list(unknown_by_layer.values())
        for segment in group
    ]
    if not all_raw:
        raise SystemExit("Nenhuma geometria linear CAD utilizavel foi encontrada.")
    scale, scale_source = _detect_scale(doc, all_raw, escala_forcada)
    reference_raw = []

    def collect_reference(entity, inherited_layer=None, depth=0):
        if depth > 8:
            return
        layer = _effective_layer(entity, inherited_layer)
        if entity.dxftype() == "INSERT":
            for child in _safe_virtual_entities(entity):
                collect_reference(child, layer, depth + 1)
            return
        for x1, y1, x2, y2 in pl._segmentos_entidade_dxf(entity):
            reference_raw.append((x1, y1, x2, y2, layer))

    for entity in doc.modelspace():
        collect_reference(entity)
    reference_scaled = pl._segs_em_escala(reference_raw, scale)

    ocr_diagnostics = []
    ocr_semantic_texts = []
    for image in linked_images:
        if not image["available"]:
            continue
        semantic_records, diagnostic = aligned_ocr_evidence(image)
        ocr_semantic_texts.extend(semantic_records)
        diagnostic.update({
            "image_handle": image.get("handle", ""),
            "image_path": image.get("resolved_path", ""),
            "resolution_source": image.get("resolution_source", ""),
        })
        ocr_diagnostics.append(serialize_ocr_diagnostic(diagnostic, scale))
    semantic_texts.extend(ocr_semantic_texts)

    explicit_layers = set(explicit_by_layer)
    inferred_layers = set()
    scores = {}
    for layer, raw in unknown_by_layer.items():
        scores[layer] = _score_wall_layer(raw, scale)
        score = scores[layer]
        threshold = 0.72 if explicit_layers else 0.42
        minimum_pairs = 3 if explicit_layers else 2
        minimum_length = 4.0 if explicit_layers else 2.0
        if (
            score["confidence"] >= threshold
            and score["pairs"] >= minimum_pairs
            and score["paired_length"] >= minimum_length
        ):
            inferred_layers.add(layer)

    selected_layers = explicit_layers | inferred_layers
    if not selected_layers and scores:
        best_layer = max(scores, key=lambda key: scores[key]["confidence"])
        if scores[best_layer]["pairs"] >= 2:
            selected_layers.add(best_layer)
            inferred_layers.add(best_layer)
    if not selected_layers:
        raise SystemExit(
            "Nenhum layer de parede foi reconhecido. Use o mapeamento CAD "
            "para marcar ao menos um layer como Parede."
        )

    selected_raw_by_layer = {
        layer: (
            explicit_by_layer.get(layer)
            or unknown_by_layer.get(layer)
            or []
        )
        for layer in selected_layers
    }
    text_candidates, grammar_diagnostics = text_geometry_candidates(
        semantic_texts,
        selected_raw_by_layer,
        selected_layers,
        scale,
    )
    generic_block_candidates = [
        candidate
        for record in generic_inserts
        if (candidate := generic_insert_candidate(record, scale)) is not None
    ]

    walls = []
    leftovers_count = 0
    raw_scaled_by_layer = {}
    layer_modes = {}
    for layer in sorted(selected_layers):
        raw = explicit_by_layer.get(layer) or unknown_by_layer.get(layer) or []
        scaled = pl._segs_em_escala(raw, scale)
        raw_scaled_by_layer[layer] = scaled
        merged = pl.mesclar_colineares(scaled)
        paired, leftovers = pair_wall_faces_v2(merged)
        fraction = pl.fracao_pareada(paired, merged)
        if fraction < pl.SINGLE_LINE_FRAC:
            layer_walls = pl.paredes_single_line(merged, esp_default)
            mode = "single-line"
            leftovers = []
        else:
            layer_walls = paired
            mode = "paired"
        confidence = (
            scores.get(layer, {}).get("confidence", 0.78)
            if layer in inferred_layers else 0.98
        )
        for wall in layer_walls:
            wall["origem"] = (
                "cad-geometry-inferred" if layer in inferred_layers
                else "cad-layer"
            )
            wall["confidence"] = float(confidence)
            wall["_cad_mode"] = mode
        walls.extend(layer_walls)
        leftovers_count += len(leftovers)
        layer_modes[layer] = mode

    walls, sewn_count = pl.costurar_cantos(walls)
    if not walls:
        raise SystemExit("Os layers selecionados nao produziram paredes validas.")

    regions = split_wall_regions(walls)
    selected_region = None
    if regions:
        selected_region = next(
            (region for region in regions if region["id"] == cad_region),
            None,
        )
        if selected_region is None:
            selected_region = max(
                regions,
                key=lambda item: (item["n_walls"], item["total_length"]),
            )
        walls = [walls[index] for index in selected_region["indexes"]]

    block_candidates = _insert_opening_candidates(opening_inserts, scale)
    loose_candidates = _cluster_opening_records(opening_loose, scale)
    semantic_candidates = (
        block_candidates
        + loose_candidates
        + generic_block_candidates
        + text_candidates
    )
    openings = pl.casar_esquadrias_com_paredes(semantic_candidates, walls)
    gap_openings = infer_gap_openings(raw_scaled_by_layer, walls, openings)
    openings = _dedupe_openings(openings + gap_openings)

    layer_diagnostics = []
    ignored_entities = 0
    for layer in sorted(stats, key=lambda name: (-stats[name]["entities"], name.lower())):
        info = stats[layer]
        explicit_role, reason = classify_layer(layer, layer_map)
        score = scores.get(layer, {
            "confidence": 0.0, "pairs": 0, "paired_length": 0.0,
            "coverage": 0.0, "orientation": 0.0,
        })
        if layer in inferred_layers:
            effective_role = "wall"
            reason = "geometry"
            confidence = score["confidence"]
        else:
            effective_role = explicit_role
            confidence = 1.0 if reason == "manual" else (0.98 if explicit_role else 0.0)
        included = layer in selected_layers or effective_role in ("door", "window", "opening")
        if not included:
            ignored_entities += int(info["entities"])
        layer_diagnostics.append({
            "name": layer,
            "entities": int(info["entities"]),
            "segments": int(info["segments"]),
            "blocks": int(info["blocks"]),
            "block_names": [
                {"name": name, "count": int(count)}
                for name, count in info["block_names"].most_common(8)
            ],
            "entity_types": {
                name: int(count) for name, count in info["entity_types"].most_common()
            },
            "detected_role": effective_role,
            "reason": reason,
            "confidence": round(float(confidence), 4),
            "included": bool(included),
            "wall_score": score,
            "wall_mode": layer_modes.get(layer),
        })

    warnings = []
    if inferred_layers:
        warnings.append(
            f"{len(inferred_layers)} layer(s) de parede inferido(s) pela geometria: "
            + ", ".join(sorted(inferred_layers))
        )
    if gap_openings:
        warnings.append(
            f"{len(gap_openings)} abertura(s) candidata(s) inferida(s) por gaps "
            "nas duas faces; revise porta/janela antes do IFC."
        )
    if text_candidates:
        warnings.append(
            f"{len(text_candidates)} esquadria(s) associada(s) por texto, "
            "dimensão e geometria pelo CAD Object Grammar V3."
        )
    successful_ocr = [
        diagnostic for diagnostic in ocr_diagnostics
        if diagnostic.get("status") == "ok"
    ]
    failed_ocr = [
        diagnostic for diagnostic in ocr_diagnostics
        if diagnostic.get("status") != "ok"
    ]
    if successful_ocr:
        ocr_line_count = sum(
            int(diagnostic.get("line_count") or 0)
            for diagnostic in successful_ocr
        )
        warnings.append(
            f"OCR local alinhou {ocr_line_count} linha(s) de texto do raster "
            "às coordenadas CAD; o texto é evidência auxiliar."
        )
    if failed_ocr:
        warnings.append(
            f"OCR local falhou em {len(failed_ocr)} imagem(ns) vinculada(s); "
            "a geometria CAD continuou sendo processada."
        )
    if len(regions) > 1:
        warnings.append(
            f"{len(regions)} plantas desconectadas foram encontradas no CAD; "
            f"mostrando {selected_region['name']}. Selecione a outra regiao "
            "no editor para revisar o outro pavimento."
        )
    if ignored_entities:
        warnings.append(
            f"{ignored_entities} entidade(s) CAD permaneceram fora da "
            "classificacao; use o mapa de layers se alguma pertencer ao BIM."
        )
    missing_linked_images = [
        image for image in linked_images if not image["available"]
    ]
    if missing_linked_images:
        warnings.append(
            f"{len(missing_linked_images)} imagem(ns) vinculada(s) ao CAD não "
            "estão disponíveis. Símbolos e textos presentes somente no raster "
            "não podem ser interpretados; envie o pacote com as referências."
        )

    summary = {
        "entities": int(sum(item["entities"] for item in stats.values())),
        "layers": len(stats),
        "wall_layers": len(selected_layers),
        "inferred_wall_layers": len(inferred_layers),
        "semantic_opening_candidates": len(semantic_candidates),
        "grammar_opening_candidates": len(
            generic_block_candidates + text_candidates
        ),
        "semantic_text_cues": len(semantic_texts),
        "linked_images": len(linked_images),
        "missing_linked_images": len(missing_linked_images),
        "ocr_images": len(successful_ocr),
        "ocr_lines": sum(
            int(diagnostic.get("line_count") or 0)
            for diagnostic in successful_ocr
        ),
        "ocr_semantic_cues": len(ocr_semantic_texts),
        "ocr_failures": len(failed_ocr),
        "gap_opening_candidates": len(gap_openings),
        "ignored_entities": ignored_entities,
        "units_code": int(doc.header.get("$INSUNITS", 0) or 0),
        "scale_source": scale_source,
        "regions": len(regions),
    }
    region_diagnostics = [
        {
            "id": region["id"],
            "name": region["name"],
            "n_walls": region["n_walls"],
            "total_length": round(float(region["total_length"]), 4),
            "bbox": {
                key: round(float(value), 4)
                for key, value in region["bbox"].items()
            },
            "selected": region is selected_region,
        }
        for region in regions
    ]
    return {
        "paredes": walls,
        "aberturas": openings,
        "escala": scale,
        "single_line": all(mode == "single-line" for mode in layer_modes.values()),
        "n_sobras": leftovers_count,
        "n_cantos": sewn_count,
        "n_blocos_esq": len(semantic_candidates),
        "laje_contorno": pl.contorno_laje(walls),
        "reference": pl.referencia_vetorial(
            reference_scaled,
            crop_bbox=(selected_region["bbox"] if selected_region else None),
        ),
        "warnings": warnings,
        "source": {
            "format": "dxf",
            "family": "cad",
            "mode": "cad-v2",
            "semantic_level": "layer-block-text-geometry",
            "grammar_version": GRAMMAR_VERSION,
            "scale_source": scale_source,
            "cad_layers": layer_diagnostics,
            "cad_summary": summary,
            "cad_region": (
                {
                    "id": selected_region["id"],
                    "name": selected_region["name"],
                }
                if selected_region else None
            ),
            "cad_regions": region_diagnostics,
            "layer_map": dict(layer_map or {}),
            "cad_semantic_cues": grammar_diagnostics,
            "cad_linked_images": linked_images,
            "cad_raster_ocr": ocr_diagnostics,
        },
    }
