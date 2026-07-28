# -*- coding: utf-8 -*-
"""Importadores geometricos para o modelador Planta -> BIM.

O contrato de saida e o mesmo usado por ``planta_to_ifc_v1``:

    {
        "paredes": [{"eixo": (A, B), "espessura": ..., ...}],
        "aberturas": [{"parede_idx": ..., "tipo": ..., ...}],
        "laje_contorno": [...],
        "source": {...},
    }

IFC e importado semanticamente. SVG/DXF passam pelo reconhecedor vetorial.
Malhas e nuvens aparecem no catalogo, mas sao encaminhadas ao Cloud-to-BIM:
uma superficie triangulada nao deve virar uma parede BIM por adivinhacao.
"""
from __future__ import annotations

import math
import re
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


class GeometryImportError(ValueError):
    """Erro de entrada com mensagem apropriada para API/usuario."""


DIRECT_EDIT_FORMATS = (".dxf", ".svg", ".ifc", ".ifczip")

FORMAT_CAPABILITIES = {
    ".ifc": {
        "familia": "bim",
        "modo": "semantico",
        "rota": "modelador",
        "status": "disponivel",
        "endpoint": "/api/planta/importar",
        "preserva": ["paredes", "portas", "janelas", "lajes", "pavimentos", "GUIDs"],
    },
    ".ifczip": {
        "familia": "bim",
        "modo": "semantico",
        "rota": "modelador",
        "status": "disponivel",
        "endpoint": "/api/planta/importar",
        "preserva": ["paredes", "portas", "janelas", "lajes", "pavimentos", "GUIDs"],
    },
    ".dxf": {
        "familia": "cad",
        "modo": "vetorial",
        "rota": "modelador",
        "status": "disponivel",
        "endpoint": "/api/planta/importar",
        "preserva": ["linhas", "polylines", "blocos", "layers", "arcos tessellados"],
    },
    ".svg": {
        "familia": "vetor",
        "modo": "vetorial",
        "rota": "modelador",
        "status": "disponivel",
        "endpoint": "/api/planta/importar",
        "preserva": ["paths", "linhas", "polylines", "poligonos", "layers/grupos"],
    },
    ".dwg": {
        "familia": "cad",
        "modo": "conversao",
        "rota": "conversor-dwg-dxf",
        "status": "requer_conversor_local",
        "preserva": ["geometria CAD apos conversao confiavel"],
    },
    ".dgn": {
        "familia": "cad",
        "modo": "conversao",
        "rota": "conversor-dgn-dxf",
        "status": "planejado",
        "preserva": ["geometria CAD apos conversao confiavel"],
    },
    ".rvt": {
        "familia": "bim_proprietario",
        "modo": "conversao",
        "rota": "exportar-ifc",
        "status": "requer_revit_ou_aps",
        "preserva": ["semantica BIM quando exportada corretamente para IFC"],
    },
    ".ifcxml": {
        "familia": "bim",
        "modo": "semantico",
        "rota": "modelador",
        "status": "planejado",
        "preserva": ["objetos e relacoes IFC"],
    },
    ".obj": {
        "familia": "malha",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "disponivel_via_obj-to-ply",
        "endpoint": "/api/tools/obj-to-ply",
        "preserva": ["vertices", "faces"],
    },
    ".usdz": {
        "familia": "malha",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "disponivel_via_usdz-to-ply",
        "endpoint": "/api/tools/usdz-to-ply",
        "preserva": ["malhas", "hierarquia USD"],
    },
    ".stl": {
        "familia": "malha",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "planejado",
        "preserva": ["vertices", "faces"],
    },
    ".glb": {
        "familia": "malha",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "planejado",
        "preserva": ["malhas", "nos", "transformacoes"],
    },
    ".gltf": {
        "familia": "malha",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "planejado",
        "preserva": ["malhas", "nos", "transformacoes"],
    },
    ".fbx": {
        "familia": "malha",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "planejado",
        "preserva": ["malhas", "nos", "transformacoes"],
    },
    ".dae": {
        "familia": "malha",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "planejado",
        "preserva": ["malhas", "nos", "transformacoes"],
    },
    ".skp": {
        "familia": "modelo_proprietario",
        "modo": "conversao",
        "rota": "cloud-to-bim",
        "status": "requer_conversor_skp",
        "preserva": ["malhas e grupos apos conversao"],
    },
    ".3dm": {
        "familia": "nurbs_brep",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "planejado",
        "preserva": ["curvas", "superficies", "breps", "malhas"],
    },
    ".step": {
        "familia": "brep",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "planejado",
        "preserva": ["solidos e superficies"],
    },
    ".stp": {
        "familia": "brep",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "planejado",
        "preserva": ["solidos e superficies"],
    },
    ".iges": {
        "familia": "brep",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "planejado",
        "preserva": ["curvas e superficies"],
    },
    ".igs": {
        "familia": "brep",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "planejado",
        "preserva": ["curvas e superficies"],
    },
    ".ply": {
        "familia": "nuvem_ou_malha",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "disponivel",
        "endpoint": "/api/scan/upload",
        "preserva": ["pontos", "cores", "faces quando presentes"],
    },
    ".e57": {
        "familia": "nuvem",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "disponivel",
        "endpoint": "/api/scan/upload",
        "preserva": ["pontos", "intensidade", "cores quando presentes"],
    },
    ".asc": {
        "familia": "nuvem",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "disponivel_via_asc-to-ply",
        "endpoint": "/api/tools/asc-to-ply",
        "preserva": ["pontos"],
    },
    ".xyz": {
        "familia": "nuvem",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "disponivel_via_asc-to-ply",
        "endpoint": "/api/tools/asc-to-ply",
        "preserva": ["pontos"],
    },
    ".las": {
        "familia": "nuvem",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "planejado",
        "preserva": ["pontos", "classificacao LAS"],
    },
    ".laz": {
        "familia": "nuvem",
        "modo": "geometria_3d",
        "rota": "cloud-to-bim",
        "status": "planejado",
        "preserva": ["pontos", "classificacao LAS"],
    },
}


def format_capabilities():
    """Catalogo serializavel para front-end e futuro MCP resource."""
    return {
        "entrada_editavel": list(DIRECT_EDIT_FORMATS),
        "formatos": FORMAT_CAPABILITIES,
        "regra": (
            "BIM preserva semantica; CAD/SVG preservam vetores; "
            "malhas e nuvens seguem para Cloud-to-BIM."
        ),
    }


def importar_geometria(path, escala_forcada=None, esp_default=0.15, pavimento=None):
    """Despacha uma entrada geometrica para o importador correto."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".dxf":
        import planta_to_ifc_v1 as pl

        modelo = pl.parse_dxf(path, escala_forcada=escala_forcada,
                              esp_default=esp_default)
        modelo["source"] = {
            "format": "dxf",
            "family": "cad",
            "mode": "vector",
            "semantic_level": "inferred",
        }
        return modelo
    if ext == ".svg":
        return importar_svg(path, escala_forcada=escala_forcada,
                            esp_default=esp_default)
    if ext == ".ifc":
        return importar_ifc(path, pavimento=pavimento,
                            esp_default=esp_default)
    if ext == ".ifczip":
        return importar_ifczip(path, pavimento=pavimento,
                               esp_default=esp_default)

    cap = FORMAT_CAPABILITIES.get(ext)
    if cap:
        raise GeometryImportError(
            f"{ext} contem geometria, mas sua rota correta e "
            f"'{cap['rota']}' ({cap['status']}); nao e uma planta vetorial editavel."
        )
    raise GeometryImportError(
        f"Formato {ext or '(sem extensao)'} nao reconhecido. "
        f"Entradas editaveis: {', '.join(DIRECT_EDIT_FORMATS)}."
    )


# ---------------------------------------------------------------------------
# SVG vetorial
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_PATH_TOKEN_RE = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
)
_TRANSFORM_RE = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")
_WALL_TOKENS = ("wall", "parede", "pared", "muro", "vegg")
_DOOR_TOKENS = ("door", "porta", "puerta")
_WINDOW_TOKENS = ("window", "janela", "ventana", "glaz")


def _local_name(tag):
    return tag.rsplit("}", 1)[-1].lower()


def _geometry_role(label):
    value = (label or "").lower()
    if any(token in value for token in _WALL_TOKENS):
        return "wall"
    if any(token in value for token in _DOOR_TOKENS):
        return "door"
    if any(token in value for token in _WINDOW_TOKENS):
        return "window"
    return None


def _svg_label(element, inherited=""):
    pieces = [inherited]
    for key, value in element.attrib.items():
        if key.rsplit("}", 1)[-1].lower() in ("id", "class", "label"):
            pieces.append(value)
    return " ".join(p for p in pieces if p).strip()


def _transform_matrix(spec):
    matrix = np.eye(3, dtype=float)
    for name, raw in _TRANSFORM_RE.findall(spec or ""):
        values = [float(v) for v in _NUMBER_RE.findall(raw)]
        op = np.eye(3, dtype=float)
        name = name.lower()
        if name == "matrix" and len(values) >= 6:
            a, b, c, d, e, f = values[:6]
            op = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])
        elif name == "translate" and values:
            op[0, 2] = values[0]
            op[1, 2] = values[1] if len(values) > 1 else 0.0
        elif name == "scale" and values:
            op[0, 0] = values[0]
            op[1, 1] = values[1] if len(values) > 1 else values[0]
        elif name == "rotate" and values:
            angle = math.radians(values[0])
            c, s = math.cos(angle), math.sin(angle)
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            if len(values) >= 3:
                cx, cy = values[1:3]
                t1 = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
                t2 = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
                op = t1 @ rot @ t2
            else:
                op = rot
        elif name == "skewx" and values:
            op[0, 1] = math.tan(math.radians(values[0]))
        elif name == "skewy" and values:
            op[1, 0] = math.tan(math.radians(values[0]))
        matrix = matrix @ op
    return matrix


def _apply_matrix(point, matrix):
    value = matrix @ np.array([point[0], point[1], 1.0], dtype=float)
    return np.array(value[:2], dtype=float)


def _segments_from_points(points, closed=False):
    points = [np.asarray(p, dtype=float) for p in points]
    segments = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    if closed and len(points) > 2 and np.linalg.norm(points[-1] - points[0]) > 1e-9:
        segments.append((points[-1], points[0]))
    return segments


def _path_segments(path_data, curve_steps=12):
    """Tessela os comandos SVG mais comuns em segmentos 2D.

    Curvas Bezier sao amostradas; arcos ``A`` usam a corda e geram um aviso.
    A aproximacao fica explicita nos diagnosticos do importador.
    """
    tokens = _PATH_TOKEN_RE.findall(path_data or "")
    i = 0
    command = None
    current = np.zeros(2)
    start = np.zeros(2)
    previous_control = None
    result = []
    arc_chords = 0
    counts = {
        "M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4,
        "Q": 4, "T": 2, "A": 7, "Z": 0,
    }

    def is_command(token):
        return len(token) == 1 and token.isalpha()

    while i < len(tokens):
        if is_command(tokens[i]):
            command = tokens[i]
            i += 1
            if command.upper() == "Z":
                if np.linalg.norm(current - start) > 1e-9:
                    result.append((current.copy(), start.copy()))
                current = start.copy()
                previous_control = None
                continue
        if command is None:
            break
        upper = command.upper()
        needed = counts[upper]
        if i + needed > len(tokens) or any(is_command(t) for t in tokens[i:i + needed]):
            command = None
            continue
        values = [float(v) for v in tokens[i:i + needed]]
        i += needed
        relative = command.islower()

        def point(x, y):
            p = np.array([x, y], dtype=float)
            return current + p if relative else p

        if upper == "M":
            current = point(values[0], values[1])
            start = current.copy()
            command = "l" if relative else "L"
            previous_control = None
        elif upper == "L":
            end = point(values[0], values[1])
            result.append((current.copy(), end.copy()))
            current = end
            previous_control = None
        elif upper == "H":
            end = current.copy()
            end[0] = current[0] + values[0] if relative else values[0]
            result.append((current.copy(), end.copy()))
            current = end
            previous_control = None
        elif upper == "V":
            end = current.copy()
            end[1] = current[1] + values[0] if relative else values[0]
            result.append((current.copy(), end.copy()))
            current = end
            previous_control = None
        elif upper == "C":
            c1 = point(values[0], values[1])
            c2 = point(values[2], values[3])
            end = point(values[4], values[5])
            last = current.copy()
            for step in range(1, curve_steps + 1):
                t = step / curve_steps
                p = ((1 - t) ** 3 * current + 3 * (1 - t) ** 2 * t * c1
                     + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * end)
                result.append((last, p))
                last = p
            current, previous_control = end, c2
        elif upper == "S":
            c1 = (2 * current - previous_control
                  if previous_control is not None else current.copy())
            c2 = point(values[0], values[1])
            end = point(values[2], values[3])
            last = current.copy()
            for step in range(1, curve_steps + 1):
                t = step / curve_steps
                p = ((1 - t) ** 3 * current + 3 * (1 - t) ** 2 * t * c1
                     + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * end)
                result.append((last, p))
                last = p
            current, previous_control = end, c2
        elif upper == "Q":
            control = point(values[0], values[1])
            end = point(values[2], values[3])
            last = current.copy()
            for step in range(1, curve_steps + 1):
                t = step / curve_steps
                p = (1 - t) ** 2 * current + 2 * (1 - t) * t * control + t ** 2 * end
                result.append((last, p))
                last = p
            current, previous_control = end, control
        elif upper == "T":
            control = (2 * current - previous_control
                       if previous_control is not None else current.copy())
            end = point(values[0], values[1])
            last = current.copy()
            for step in range(1, curve_steps + 1):
                t = step / curve_steps
                p = (1 - t) ** 2 * current + 2 * (1 - t) * t * control + t ** 2 * end
                result.append((last, p))
                last = p
            current, previous_control = end, control
        elif upper == "A":
            end = point(values[5], values[6])
            result.append((current.copy(), end.copy()))
            current = end
            previous_control = None
            arc_chords += 1
    return result, arc_chords


def _element_segments(element):
    tag = _local_name(element.tag)
    a = element.attrib
    if tag == "line":
        return [(np.array([float(a.get("x1", 0)), float(a.get("y1", 0))]),
                 np.array([float(a.get("x2", 0)), float(a.get("y2", 0))]))], 0
    if tag in ("polyline", "polygon"):
        values = [float(v) for v in _NUMBER_RE.findall(a.get("points", ""))]
        points = [np.array(values[j:j + 2]) for j in range(0, len(values) - 1, 2)]
        return _segments_from_points(points, closed=(tag == "polygon")), 0
    if tag == "rect":
        x, y = float(a.get("x", 0)), float(a.get("y", 0))
        w, h = float(a.get("width", 0)), float(a.get("height", 0))
        points = [np.array([x, y]), np.array([x + w, y]),
                  np.array([x + w, y + h]), np.array([x, y + h])]
        return _segments_from_points(points, closed=True), 0
    if tag in ("circle", "ellipse"):
        cx, cy = float(a.get("cx", 0)), float(a.get("cy", 0))
        rx = float(a.get("r", a.get("rx", 0)))
        ry = float(a.get("r", a.get("ry", 0)))
        points = [
            np.array([cx + rx * math.cos(2 * math.pi * j / 32),
                      cy + ry * math.sin(2 * math.pi * j / 32)])
            for j in range(32)
        ]
        return _segments_from_points(points, closed=True), 0
    if tag == "path":
        return _path_segments(a.get("d", ""))
    return [], 0


def _length_to_m(value):
    match = re.fullmatch(
        r"\s*([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)\s*([A-Za-z]*)\s*",
        value or "",
    )
    if not match:
        return None
    number, unit = float(match.group(1)), match.group(2).lower()
    factors = {
        "m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254,
        "ft": 0.3048, "px": 0.0254 / 96.0,
    }
    if unit not in factors:
        return None
    return number * factors[unit]


def read_svg_geometry(svg_path, escala_forcada=None):
    """Le primitivas SVG e devolve segmentos classificados, ja em metros."""
    root = ET.parse(str(svg_path)).getroot()
    records = []
    arc_chords = 0

    def visit(element, inherited_matrix, inherited_label):
        nonlocal arc_chords
        label = _svg_label(element, inherited_label)
        matrix = inherited_matrix @ _transform_matrix(element.attrib.get("transform", ""))
        segments, n_arcs = _element_segments(element)
        arc_chords += n_arcs
        if segments:
            transformed = [(_apply_matrix(a, matrix), _apply_matrix(b, matrix))
                           for a, b in segments]
            records.append({
                "segments": transformed,
                "role": _geometry_role(label),
                "label": label or "SVG-Geometry",
            })
        for child in list(element):
            visit(child, matrix, label)

    visit(root, np.eye(3), "")
    all_segments = [segment for record in records for segment in record["segments"]]
    if not all_segments:
        raise GeometryImportError("SVG sem primitivas geometricas reconheciveis.")

    all_points = np.vstack([point for segment in all_segments for point in segment])
    extent = float(np.ptp(all_points, axis=0).max())
    if escala_forcada is not None:
        scale = float(escala_forcada)
        scale_source = "forced"
    else:
        viewbox_values = [float(v) for v in _NUMBER_RE.findall(root.attrib.get("viewBox", ""))]
        physical_width = _length_to_m(root.attrib.get("width", ""))
        if physical_width is not None and len(viewbox_values) == 4 and viewbox_values[2] > 0:
            scale = physical_width / viewbox_values[2]
            scale_source = "svg-width/viewBox"
        else:
            scale = 0.001 if extent > 1000 else (0.01 if extent > 100 else 1.0)
            scale_source = "auto-extent"

    explicit_walls = [record for record in records if record["role"] == "wall"]
    wall_records = explicit_walls or [record for record in records if record["role"] is None]
    wall_segments = [
        (a * scale, b * scale, record["label"])
        for record in wall_records for a, b in record["segments"]
        if np.linalg.norm((b - a) * scale) >= 0.05
    ]
    openings = []
    for record in records:
        if record["role"] not in ("door", "window"):
            continue
        points = np.vstack([point for segment in record["segments"] for point in segment])
        openings.append({"tipo": record["role"], "pts": points * scale})
    warnings = []
    if not explicit_walls:
        warnings.append(
            "SVG sem grupo/layer de parede; geometria nao classificada foi tratada como parede."
        )
    if arc_chords:
        warnings.append(f"{arc_chords} arco(s) de path SVG representado(s) pela corda.")
    return {
        "segments": wall_segments,
        "openings": openings,
        "scale": scale,
        "scale_source": scale_source,
        "warnings": warnings,
        "records": len(records),
    }


def importar_svg(svg_path, escala_forcada=None, esp_default=0.15):
    import planta_to_ifc_v1 as pl

    data = read_svg_geometry(svg_path, escala_forcada=escala_forcada)
    segs = pl.mesclar_colineares(data["segments"])
    paredes, sobras = pl.parear_paredes(segs)
    frac = pl.fracao_pareada(paredes, segs)
    single = frac < pl.SINGLE_LINE_FRAC
    if single:
        paredes = pl.paredes_single_line(segs, esp_default)
        sobras = []
    paredes, n_cost = pl.costurar_cantos(paredes)
    aberturas = pl.casar_esquadrias_com_paredes(data["openings"], paredes)
    return {
        "paredes": paredes,
        "aberturas": aberturas,
        "escala": data["scale"],
        "single_line": single,
        "n_sobras": len(sobras),
        "n_cantos": n_cost,
        "n_blocos_esq": len(data["openings"]),
        "laje_contorno": pl.contorno_laje(paredes),
        "warnings": data["warnings"],
        "source": {
            "format": "svg",
            "family": "vector",
            "mode": "vector",
            "semantic_level": "layer-inferred" if not data["warnings"] else "geometry-inferred",
            "scale_source": data["scale_source"],
            "vector_records": data["records"],
        },
    }


# ---------------------------------------------------------------------------
# IFC semantico
# ---------------------------------------------------------------------------
def _convex_hull(points):
    values = sorted({(round(float(p[0]), 6), round(float(p[1]), 6)) for p in points})
    if len(values) < 3:
        return [np.array(p) for p in values]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return [np.array(p) for p in lower[:-1] + upper[:-1]]


def _entity_name(entity, fallback):
    value = getattr(entity, "Name", None)
    return str(value).strip() if value else fallback


def _storey_members(storey):
    result = []
    for relation in getattr(storey, "ContainsElements", ()) or ():
        result.extend(getattr(relation, "RelatedElements", ()) or ())
    return result


def _ifc_shape_vertices(product, ifcopenshell_geom):
    settings = ifcopenshell_geom.settings()
    world_coords = False
    try:
        settings.set(settings.USE_WORLD_COORDS, True)
        world_coords = True
    except Exception:
        pass
    shape = ifcopenshell_geom.create_shape(settings, product)
    vertices = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
    if not world_coords:
        try:
            from ifcopenshell.util.placement import get_local_placement

            matrix = get_local_placement(product.ObjectPlacement)
            vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        except Exception:
            pass
    return vertices


def _fit_wall_from_vertices(vertices, esp_default=0.15):
    """Ajusta eixo/espessura/altura a uma geometria de parede em coordenadas mundo."""
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) < 2:
        raise GeometryImportError("Parede IFC sem vertices suficientes.")
    xy = vertices[:, :2]
    center = xy.mean(axis=0)
    centered = xy - center
    covariance = centered.T @ centered / max(1, len(centered))
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    if direction[0] < 0 or (abs(direction[0]) < 1e-9 and direction[1] < 0):
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    long_values = centered @ direction
    perp_values = centered @ normal
    long_min, long_max = float(long_values.min()), float(long_values.max())
    perp_min, perp_max = float(perp_values.min()), float(perp_values.max())
    length = long_max - long_min
    measured_thickness = perp_max - perp_min
    thickness = measured_thickness
    approximated = False
    if thickness < 0.02 or thickness > 1.5:
        thickness = float(esp_default)
        approximated = True
    if length < 0.05:
        raise GeometryImportError("Parede IFC degenerada na projecao XY.")
    axis_offset = (perp_min + perp_max) / 2
    axis_center = center + normal * axis_offset
    a = axis_center + direction * long_min
    b = axis_center + direction * long_max
    zmin, zmax = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    return {
        "eixo": (a, b),
        "espessura": float(thickness),
        "comprimento": float(length),
        "altura": max(0.01, zmax - zmin),
        "elevacao": zmin,
        "geometry_approximated": approximated,
    }


def _select_storey(storeys, requested):
    if not storeys:
        return None
    ordered = sorted(
        storeys,
        key=lambda s: (
            float(getattr(s, "Elevation", 0.0) or 0.0),
            _entity_name(s, ""),
        ),
    )
    if requested is None or str(requested).strip() == "":
        return ordered[0]
    key = str(requested).strip().casefold()
    for index, storey in enumerate(ordered):
        candidates = {
            str(index),
            str(getattr(storey, "GlobalId", "")),
            _entity_name(storey, ""),
        }
        if key in {candidate.casefold() for candidate in candidates}:
            return storey
    available = ", ".join(_entity_name(s, f"Pavimento {i}") for i, s in enumerate(ordered))
    raise GeometryImportError(f"Pavimento '{requested}' nao encontrado. Disponiveis: {available}.")


def importar_ifc(ifc_path, pavimento=None, esp_default=0.15):
    """IFC -> modelo 2D editavel, preservando semantica e proveniencia."""
    try:
        import ifcopenshell
        import ifcopenshell.geom as ifc_geom
    except ImportError as exc:
        raise GeometryImportError(
            "IfcOpenShell nao esta instalado; instale as dependencias do backend."
        ) from exc

    model = ifcopenshell.open(str(ifc_path))
    try:
        from ifcopenshell.util.unit import calculate_unit_scale

        unit_scale = float(calculate_unit_scale(model))
    except Exception:
        unit_scale = 1.0
    storeys = list(model.by_type("IfcBuildingStorey"))
    selected = _select_storey(storeys, pavimento)
    all_walls = list(model.by_type("IfcWall"))
    if pavimento is None and selected is not None:
        wall_ids = {wall.id() for wall in all_walls}
        ordered_storeys = sorted(
            storeys,
            key=lambda s: float(getattr(s, "Elevation", 0.0) or 0.0),
        )
        selected = next(
            (storey for storey in ordered_storeys
             if any(entity.id() in wall_ids for entity in _storey_members(storey))),
            selected,
        )
    selected_members = _storey_members(selected) if selected is not None else []
    member_ids = {entity.id() for entity in selected_members}

    if selected is not None and member_ids:
        walls = [wall for wall in all_walls if wall.id() in member_ids]
    else:
        walls = all_walls
    if not walls:
        raise GeometryImportError("IFC sem IfcWall no pavimento selecionado.")

    warnings = []
    internal_walls = []
    wall_index = {}
    wall_world_base = {}
    approximated = 0
    for wall in walls:
        try:
            vertices = _ifc_shape_vertices(wall, ifc_geom)
            fitted = _fit_wall_from_vertices(vertices, esp_default=esp_default)
        except Exception as exc:
            warnings.append(
                f"Parede {_entity_name(wall, str(wall.id()))} ignorada: {exc}"
            )
            continue
        guid = str(getattr(wall, "GlobalId", "") or "")
        fitted.update({
            "id": f"ifc-{guid or wall.id()}",
            "guid": guid or None,
            "nome": _entity_name(wall, f"Parede-{wall.id()}"),
            "ifc_class": wall.is_a(),
            "tipo": str(getattr(wall, "PredefinedType", "") or ""),
            "layer": f"IFC::{_entity_name(selected, 'Sem pavimento')}",
            "nivel": _entity_name(selected, "Sem pavimento"),
            "origem": "ifc",
        })
        if fitted["geometry_approximated"]:
            approximated += 1
            warnings.append(
                f"Espessura de {_entity_name(wall, str(wall.id()))} "
                f"nao era confiavel; usado {esp_default:.3f} m."
            )
        wall_index[wall.id()] = len(internal_walls)
        wall_world_base[wall.id()] = fitted["elevacao"]
        internal_walls.append(fitted)
    if not internal_walls:
        raise GeometryImportError("Nenhuma parede IFC produziu geometria 2D valida.")
    # O editor trabalha em coordenadas locais do pavimento. A elevacao absoluta
    # continua registrada em source.pavimento, sem deixar paredes "flutuando".
    floor_base = min(wall["elevacao"] for wall in internal_walls)
    for wall in internal_walls:
        wall["elevacao"] -= floor_base

    openings = []
    for wall in walls:
        if wall.id() not in wall_index:
            continue
        wall_data = internal_walls[wall_index[wall.id()]]
        a, b = wall_data["eixo"]
        direction = (b - a) / np.linalg.norm(b - a)
        for void_relation in getattr(wall, "HasOpenings", ()) or ():
            opening = getattr(void_relation, "RelatedOpeningElement", None)
            if opening is None:
                continue
            fillings = list(getattr(opening, "HasFillings", ()) or ())
            fill = getattr(fillings[0], "RelatedBuildingElement", None) if fillings else None
            if fill is not None and fill.is_a("IfcWindow"):
                kind = "window"
            elif fill is not None and fill.is_a("IfcDoor"):
                kind = "door"
            else:
                kind = "door"
            geometry_source = fill or opening
            try:
                vertices = _ifc_shape_vertices(geometry_source, ifc_geom)
                projected = (vertices[:, :2] - a) @ direction
                width = float(projected.max() - projected.min())
                center = float((projected.max() + projected.min()) / 2)
                zmin, zmax = float(vertices[:, 2].min()), float(vertices[:, 2].max())
                geometry_available = True
            except Exception:
                width = float(getattr(fill, "OverallWidth", 0.0) or 0.0) if fill else 0.0
                center = wall_data["comprimento"] / 2
                zmin = zmax = wall_world_base[wall.id()]
                geometry_available = False
            if width < 0.1:
                width = 0.80 if kind == "door" else 1.00
            element_height = max(0.0, zmax - zmin)
            if element_height < 0.1:
                element_height = 2.10 if kind == "door" else 1.20
            sill = (max(0.0, zmin - wall_world_base[wall.id()])
                    if geometry_available else (0.0 if kind == "door" else 1.0))
            fill_guid = str(getattr(fill, "GlobalId", "") or "") if fill else ""
            openings.append({
                "id": f"ifc-{fill_guid or opening.id()}",
                "parede_idx": wall_index[wall.id()],
                "tipo": kind,
                "s_centro": center,
                "largura": width,
                "altura": element_height,
                "peitoril": sill,
                "guid": fill_guid or None,
                "nome": _entity_name(fill, f"Vao-{opening.id()}") if fill else f"Vao-{opening.id()}",
                "origem": "ifc",
            })

    slab_points = []
    all_slabs = list(model.by_type("IfcSlab"))
    slabs = ([slab for slab in all_slabs if slab.id() in member_ids]
             if selected is not None and member_ids else all_slabs)
    slab_thicknesses = []
    for slab in slabs:
        try:
            vertices = _ifc_shape_vertices(slab, ifc_geom)
            slab_points.extend(vertices[:, :2])
            slab_thicknesses.append(float(np.ptp(vertices[:, 2])))
        except Exception as exc:
            warnings.append(f"Laje {_entity_name(slab, str(slab.id()))} ignorada: {exc}")
    if slab_points:
        contour = _convex_hull(slab_points)
    else:
        contour = _convex_hull(
            [point for wall in internal_walls for point in wall["eixo"]]
        )
    slab_thickness = (float(np.median(slab_thicknesses))
                      if slab_thicknesses else 0.12)

    ordered_storeys = sorted(
        storeys,
        key=lambda s: float(getattr(s, "Elevation", 0.0) or 0.0),
    )
    available_storeys = []
    for index, storey in enumerate(ordered_storeys):
        ids = {entity.id() for entity in _storey_members(storey)}
        available_storeys.append({
            "id": str(getattr(storey, "GlobalId", "") or index),
            "nome": _entity_name(storey, f"Pavimento {index + 1}"),
            "elevacao": float(getattr(storey, "Elevation", 0.0) or 0.0) * unit_scale,
            "n_paredes": sum(1 for wall in all_walls if wall.id() in ids),
        })

    return {
        "paredes": internal_walls,
        "aberturas": openings,
        "escala": 1.0,
        "single_line": False,
        "n_sobras": 0,
        "n_cantos": 0,
        "n_blocos_esq": len(openings),
        "n_elementos": len(internal_walls) + len(openings) + len(slabs),
        "n_aproximados": approximated,
        "laje_contorno": contour,
        "laje_faces": {
            "piso": {"ativo": bool(slabs), "espessura": slab_thickness},
            "teto": {"ativo": False, "espessura": slab_thickness},
        },
        "warnings": warnings,
        "source": {
            "format": "ifc",
            "family": "bim",
            "mode": "semantic",
            "semantic_level": "native",
            "pavimento": ({
                "id": str(getattr(selected, "GlobalId", "") or ""),
                "nome": _entity_name(selected, "Sem pavimento"),
                "elevacao": float(getattr(selected, "Elevation", 0.0) or 0.0) * unit_scale,
            } if selected is not None else None),
            "pavimentos_disponiveis": available_storeys,
        },
    }


def importar_ifczip(ifczip_path, pavimento=None, esp_default=0.15):
    with zipfile.ZipFile(str(ifczip_path)) as archive:
        candidates = [
            info for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".ifc")
        ]
        if not candidates:
            raise GeometryImportError("IFCZIP sem arquivo .ifc.")
        candidate = min(candidates, key=lambda info: len(Path(info.filename).parts))
        if candidate.file_size > 500 * 1024 * 1024:
            raise GeometryImportError("IFC dentro do IFCZIP excede 500 MB.")
        with tempfile.TemporaryDirectory(prefix="planta_ifczip_") as temp_dir:
            target = Path(temp_dir) / "modelo.ifc"
            with archive.open(candidate) as source, target.open("wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            model = importar_ifc(target, pavimento=pavimento,
                                 esp_default=esp_default)
            model["source"]["format"] = "ifczip"
            return model
