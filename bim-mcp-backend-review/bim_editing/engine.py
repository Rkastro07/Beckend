"""Declarative, deterministic editing operations for the canonical BIM model."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Any

from .geometry import (
    EPS,
    Point,
    Segment,
    add,
    convex_hull,
    distance,
    dot,
    left_normal,
    line_intersection,
    line_parameters,
    norm,
    parameter_on_line,
    planar_faces,
    point,
    point_in_polygon,
    polygon_area,
    polygon_perimeter,
    project_to_line,
    scale,
    sub,
    unit,
    wall_axis,
    wall_arc_geometry,
    wall_corners,
    wall_length,
)
from .model import REVISION_SCHEMA, normalize_model, refresh_derived


class RevisionError(ValueError):
    pass


def _next_revision(value: str) -> str:
    text = str(value or "R00")
    if text.startswith("R") and text[1:].isdigit():
        return f"R{int(text[1:]) + 1:02d}"
    return f"{text}.1"


class RevisionEngine:
    """Apply edit operations without changing the base-model object."""

    def __init__(self, model: dict[str, Any]):
        already_canonical = model.get("endpoint_order") is not None
        self.base = normalize_model(
            model,
            canonicalize_initial_order=not already_canonical,
        )
        self.model: dict[str, Any] = {}
        self.report: dict[str, Any] = {}

    def _wall_index(self) -> dict[str, dict]:
        return {str(wall["id"]): wall for wall in self.model["paredes"]}

    def _opening_index(self) -> dict[str, dict]:
        return {str(opening["id"]): opening for opening in self.model["aberturas"]}

    def _wall(self, identifier: str) -> dict:
        try:
            return self._wall_index()[str(identifier)]
        except KeyError as exc:
            raise RevisionError(f"parede não encontrada: {identifier}") from exc

    @staticmethod
    def _selector_parts(selector: str) -> tuple[str, str | None]:
        text = str(selector)
        if "." not in text:
            return text, None
        element, suffix = text.rsplit(".", 1)
        if suffix.upper() in {"P1", "P2", "AXIS"}:
            return element, suffix.upper()
        return text, None

    def resolve_selector(self, selector: str) -> dict[str, Any]:
        element, part = self._selector_parts(selector)
        wall = self._wall(element)
        if part is None:
            return {"kind": "wall", "element": element, "value": wall}
        if part == "P1":
            value = (float(wall["ax"]), float(wall["ay"]))
            return {"kind": "point", "element": element, "part": part, "value": value}
        if part == "P2":
            value = (float(wall["bx"]), float(wall["by"]))
            return {"kind": "point", "element": element, "part": part, "value": value}
        return {
            "kind": "axis",
            "element": element,
            "part": "AXIS",
            "value": wall_axis(wall),
        }

    def _target_axis(self, target: Any) -> Segment:
        if isinstance(target, str):
            resolved = self.resolve_selector(target)
            if resolved["kind"] == "axis":
                return resolved["value"]
            if resolved["kind"] == "wall":
                return wall_axis(resolved["value"])
            raise RevisionError(f"seletor não representa eixo: {target}")
        if isinstance(target, dict):
            identifier = (
                target.get("element")
                or target.get("wall_id")
                or target.get("selector")
            )
            if not identifier:
                raise RevisionError("alvo sem elemento")
            return self._target_axis(str(identifier))
        raise RevisionError(f"alvo de eixo inválido: {target!r}")

    def _resolve_point(
        self,
        value: Any,
        *,
        source_axis: Segment | None = None,
        from_point: Point | None = None,
    ) -> Point:
        if isinstance(value, str):
            resolved = self.resolve_selector(value)
            if resolved["kind"] != "point":
                raise RevisionError(f"seletor não representa ponto: {value}")
            return resolved["value"]
        if isinstance(value, (list, tuple)):
            return point(value)
        if not isinstance(value, dict):
            raise RevisionError(f"ponto inválido: {value!r}")
        if "point" in value:
            return point(value["point"])
        if "selector" in value and not value.get("mode"):
            return self._resolve_point(value["selector"])

        mode = str(value.get("mode", "")).lower()
        axis = self._target_axis(value)
        reference = from_point
        if reference is None and source_axis is not None:
            reference = source_axis[1]
        if mode in {"nearest_point", "nearest_on_segment"}:
            if reference is None:
                raise RevisionError("nearest_point exige um ponto de referência")
            return project_to_line(reference, axis, clamp=True)
        if mode in {"perpendicular_projection", "project"}:
            if reference is None:
                raise RevisionError("perpendicular_projection exige referência")
            return project_to_line(reference, axis, clamp=False)
        if mode in {"axis_intersection", "intersection"}:
            if source_axis is None:
                raise RevisionError("axis_intersection exige um eixo de origem")
            result = line_intersection(source_axis, axis)
            if result is None:
                raise RevisionError("os eixos informados são paralelos")
            return result
        if mode in {"p1", "p2"}:
            return axis[0] if mode == "p1" else axis[1]
        raise RevisionError(f"modo de ponto desconhecido: {mode or '<vazio>'}")

    def _direction(self, value: Any) -> Point:
        if isinstance(value, (list, tuple)):
            return unit(point(value))
        if isinstance(value, str):
            return unit(sub(*reversed(self._target_axis(value))))
        if not isinstance(value, dict):
            raise RevisionError("direção inválida")
        if "vector" in value:
            return unit(point(value["vector"]))
        if "parallel_to" in value:
            axis = self._target_axis(value["parallel_to"])
            return unit(sub(axis[1], axis[0]))
        if "perpendicular_to" in value:
            axis = self._target_axis(value["perpendicular_to"])
            return left_normal(sub(axis[1], axis[0]))
        raise RevisionError("direção precisa de vector, parallel_to ou perpendicular_to")

    def apply(self, specification: dict[str, Any]) -> tuple[dict, dict]:
        if specification.get("schema", REVISION_SCHEMA) != REVISION_SCHEMA:
            raise RevisionError(
                f"schema de operação incompatível: {specification.get('schema')}"
            )
        self.model = deepcopy(self.base)
        base_revision = str(self.model.get("revision", "R00"))
        target_revision = str(
            specification.get("revision") or _next_revision(base_revision)
        )
        self.report = {
            "schema": "bim.edit-report.v1",
            "base_revision": base_revision,
            "revision": target_revision,
            "operation_results": [],
            "warnings": [],
            "recalculated": [],
            "validation": {},
        }

        for index, operation in enumerate(specification.get("operations", []), 1):
            if not isinstance(operation, dict) or not operation.get("op"):
                raise RevisionError(f"operação {index} sem campo op")
            name = str(operation["op"]).lower()
            handler = getattr(self, f"_op_{name}", None)
            if handler is None:
                raise RevisionError(f"operação desconhecida: {name}")
            result = handler(operation)
            self.report["operation_results"].append(
                {"index": index, "op": name, **(result or {})}
            )
            refresh_derived(self.model)

        requested = specification.get(
            "recalculate",
            ["openings", "topology", "spaces", "slabs", "validation"],
        )
        requested = [str(value).lower() for value in requested]
        policies = dict(specification.get("policies", {}))
        if "openings" in requested:
            self._recalculate_openings(policies)
        if "topology" in requested or "spaces" in requested:
            self._recalculate_topology(policies)
        if "spaces" in requested:
            self._recalculate_spaces(policies)
        if "slabs" in requested:
            self._recalculate_slabs(policies)
        refresh_derived(self.model)
        if "validation" in requested:
            self.report["validation"] = self._validate(policies)
            self.report["recalculated"].append("validation")
            if not self.report["validation"]["valid"]:
                raise RevisionError(
                    "revisão inválida: "
                    + "; ".join(self.report["validation"]["errors"])
                )

        self.model["revision"] = target_revision
        history = {
            "revision": target_revision,
            "base_revision": base_revision,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "operations": deepcopy(specification.get("operations", [])),
            "recalculated": list(self.report["recalculated"]),
            "warnings": list(self.report["warnings"]),
        }
        self.model.setdefault("edit_history", []).append(history)
        self.model["last_revision_report"] = deepcopy(self.report)
        return deepcopy(self.model), deepcopy(self.report)

    def _op_delete_elements(self, operation: dict) -> dict:
        identifiers = [str(value) for value in operation.get("ids", [])]
        if not identifiers and operation.get("id"):
            identifiers = [str(operation["id"])]
        wall_ids = {wall["id"] for wall in self.model["paredes"]}
        opening_ids = {opening["id"] for opening in self.model["aberturas"]}
        unknown = [
            identifier
            for identifier in identifiers
            if identifier not in wall_ids and identifier not in opening_ids
        ]
        if unknown and operation.get("strict", True):
            raise RevisionError("elementos não encontrados: " + ", ".join(unknown))
        deleted_walls = [identifier for identifier in identifiers if identifier in wall_ids]
        deleted_openings = [
            identifier for identifier in identifiers if identifier in opening_ids
        ]
        hosted = [
            opening["id"]
            for opening in self.model["aberturas"]
            if opening["parede_id"] in deleted_walls
        ]
        deleted_openings.extend(hosted)
        self.model["paredes"] = [
            wall for wall in self.model["paredes"] if wall["id"] not in deleted_walls
        ]
        self.model["aberturas"] = [
            opening
            for opening in self.model["aberturas"]
            if opening["id"] not in set(deleted_openings)
            and opening["parede_id"] not in deleted_walls
        ]
        return {
            "deleted_walls": sorted(deleted_walls),
            "deleted_openings": sorted(set(deleted_openings)),
            "ignored": unknown,
        }

    def _op_add_wall(self, operation: dict) -> dict:
        identifier = str(operation.get("id") or "")
        if not identifier:
            raise RevisionError("add_wall exige id")
        if identifier in self._wall_index() or identifier in self._opening_index():
            raise RevisionError(f"ID já existe: {identifier}")
        start_value = operation.get("p1", operation.get("from"))
        if start_value is None:
            raise RevisionError("add_wall exige p1/from")
        p1 = self._resolve_point(start_value)
        if operation.get("p2") is not None:
            p2 = self._resolve_point(operation["p2"], from_point=p1)
        elif operation.get("direction") is not None and operation.get("until") is not None:
            direction = self._direction(operation["direction"])
            target = self._target_axis(operation["until"])
            ray = (p1, add(p1, direction))
            result = line_parameters(ray, target)
            if result is None:
                raise RevisionError("a nova parede não intercepta o alvo")
            p2, along, _ = result
            if operation.get("forward_only", False) and along < -EPS:
                raise RevisionError("a interseção está atrás da direção informada")
        elif operation.get("direction") is not None and operation.get("length") is not None:
            p2 = add(
                p1,
                scale(self._direction(operation["direction"]), float(operation["length"])),
            )
        else:
            raise RevisionError("add_wall exige p2 ou direction+until/length")
        if distance(p1, p2) <= 1e-4:
            raise RevisionError("nova parede degenerada")
        metadata = dict(operation.get("metadata", {}))
        wall = {
            "id": identifier,
            "ax": p1[0],
            "ay": p1[1],
            "bx": p2[0],
            "by": p2[1],
            "espessura": float(operation.get("thickness", 0.15)),
            "layer": str(operation.get("layer", "Wall-Edit")),
            "origem": "revision_engine",
            **metadata,
        }
        self.model["paredes"].append(wall)
        return {
            "wall_id": identifier,
            "p1": list(p1),
            "p2": list(p2),
            "length": distance(p1, p2),
        }

    def _op_move_wall_endpoint(self, operation: dict) -> dict:
        identifier = str(
            operation.get("element")
            or operation.get("wall_id")
            or operation.get("id")
            or ""
        )
        endpoint = str(operation.get("endpoint", "")).upper()
        if not identifier and operation.get("selector"):
            identifier, selector_part = self._selector_parts(operation["selector"])
            endpoint = endpoint or str(selector_part or "")
        if endpoint not in {"P1", "P2"}:
            raise RevisionError("move_wall_endpoint exige endpoint P1 ou P2")
        wall = self._wall(identifier)
        axis = wall_axis(wall)
        current = axis[0] if endpoint == "P1" else axis[1]
        if operation.get("delta") is not None:
            delta = point(operation["delta"])
            target = add(current, delta)
        elif operation.get("distance") is not None:
            direction = unit(sub(current, axis[1] if endpoint == "P1" else axis[0]))
            target = add(current, scale(direction, float(operation["distance"])))
        elif operation.get("target") is not None:
            target = self._resolve_point(
                operation["target"],
                source_axis=axis,
                from_point=current,
            )
        else:
            raise RevisionError("move_wall_endpoint exige target, delta ou distance")
        other = axis[1] if endpoint == "P1" else axis[0]
        if distance(target, other) <= 1e-4:
            raise RevisionError("a operação deixaria a parede degenerada")
        if endpoint == "P1":
            wall["ax"], wall["ay"] = target
        else:
            wall["bx"], wall["by"] = target
        return {
            "wall_id": identifier,
            "endpoint": endpoint,
            "before": list(current),
            "after": list(target),
        }

    def _op_connect_endpoint(self, operation: dict) -> dict:
        proxy = dict(operation)
        proxy["op"] = "move_wall_endpoint"
        if proxy.get("selector"):
            identifier, endpoint = self._selector_parts(proxy["selector"])
            proxy.setdefault("element", identifier)
            proxy.setdefault("endpoint", endpoint)
        return self._op_move_wall_endpoint(proxy)

    def _op_move_wall(self, operation: dict) -> dict:
        identifier = str(operation.get("id") or operation.get("element") or "")
        wall = self._wall(identifier)
        delta = point(operation.get("delta", (0.0, 0.0)))
        before = wall_axis(wall)
        wall["ax"] += delta[0]
        wall["ay"] += delta[1]
        wall["bx"] += delta[0]
        wall["by"] += delta[1]
        return {
            "wall_id": identifier,
            "delta": list(delta),
            "before": [list(before[0]), list(before[1])],
        }

    def _op_set_wall_thickness(self, operation: dict) -> dict:
        identifier = str(operation.get("id") or operation.get("element") or "")
        wall = self._wall(identifier)
        value = float(operation.get("thickness"))
        if value <= 0:
            raise RevisionError("espessura deve ser positiva")
        before = float(wall["espessura"])
        wall["espessura"] = value
        return {"wall_id": identifier, "before": before, "after": value}

    def _op_merge_walls(self, operation: dict) -> dict:
        identifiers = [str(value) for value in operation.get("ids", [])]
        if len(identifiers) < 2:
            raise RevisionError("merge_walls exige pelo menos duas paredes")
        walls = [self._wall(identifier) for identifier in identifiers]
        target_id = str(operation.get("target_id") or identifiers[0])
        if target_id not in identifiers and target_id in self._wall_index():
            raise RevisionError(f"target_id já existe: {target_id}")

        reference = max(walls, key=lambda wall: norm(sub(*reversed(wall_axis(wall)))))
        ref_axis = wall_axis(reference)
        direction = unit(sub(ref_axis[1], ref_axis[0]))
        if direction[0] < -EPS or (abs(direction[0]) <= EPS and direction[1] < 0):
            direction = scale(direction, -1.0)
        normal = (-direction[1], direction[0])
        angles = []
        for wall in walls:
            current = unit(sub(wall_axis(wall)[1], wall_axis(wall)[0]))
            angles.append(math.degrees(math.acos(min(1.0, abs(dot(direction, current))))))
        tolerance = float(operation.get("angle_tolerance_deg", 12.0))
        if max(angles) > tolerance:
            raise RevisionError(
                f"paredes não colineares o suficiente para merge ({max(angles):.2f}°)"
            )

        all_points = [value for wall in walls for value in wall_axis(wall)]
        t_values = [dot(value, direction) for value in all_points]
        rho_min = math.inf
        rho_max = -math.inf
        for wall in walls:
            axis = wall_axis(wall)
            rho = 0.5 * (dot(axis[0], normal) + dot(axis[1], normal))
            half = float(wall["espessura"]) * 0.5
            rho_min = min(rho_min, rho - half)
            rho_max = max(rho_max, rho + half)
        rho_mid = 0.5 * (rho_min + rho_max)
        p1 = add(scale(direction, min(t_values)), scale(normal, rho_mid))
        p2 = add(scale(direction, max(t_values)), scale(normal, rho_mid))
        thickness = rho_max - rho_min

        opening_centres: list[tuple[dict, Point]] = []
        old_by_id = {wall["id"]: wall for wall in walls}
        for opening in self.model["aberturas"]:
            if opening["parede_id"] in old_by_id:
                old_axis = wall_axis(old_by_id[opening["parede_id"]])
                old_direction = unit(sub(old_axis[1], old_axis[0]))
                centre = add(
                    old_axis[0],
                    scale(old_direction, float(opening["s_centro"])),
                )
                opening_centres.append((opening, centre))

        template = deepcopy(walls[0])
        template.update(
            {
                "id": target_id,
                "ax": p1[0],
                "ay": p1[1],
                "bx": p2[0],
                "by": p2[1],
                "espessura": thickness,
                "origem": "revision_engine.merge",
                "merged_from": identifiers,
            }
        )
        self.model["paredes"] = [
            wall for wall in self.model["paredes"] if wall["id"] not in identifiers
        ]
        self.model["paredes"].append(template)
        new_direction = unit(sub(p2, p1))
        for opening, centre in opening_centres:
            opening["parede_id"] = target_id
            opening["s_centro"] = dot(sub(centre, p1), new_direction)
        return {
            "merged": identifiers,
            "wall_id": target_id,
            "thickness": thickness,
            "p1": list(p1),
            "p2": list(p2),
        }

    def _op_add_opening(self, operation: dict) -> dict:
        identifier = str(operation.get("id") or "")
        host = str(operation.get("wall_id") or operation.get("host_wall") or "")
        if not identifier or not host:
            raise RevisionError("add_opening exige id e wall_id")
        if identifier in self._opening_index() or identifier in self._wall_index():
            raise RevisionError(f"ID já existe: {identifier}")
        wall = self._wall(host)
        axis = wall_axis(wall)
        if operation.get("s_center") is not None:
            centre = float(operation["s_center"])
        elif operation.get("s_centro") is not None:
            centre = float(operation["s_centro"])
        elif operation.get("at") is not None:
            location = self._resolve_point(operation["at"], from_point=axis[0])
            centre = parameter_on_line(project_to_line(location, axis, clamp=True), axis)
        else:
            centre = 0.5 * norm(sub(axis[1], axis[0]))
        opening = {
            "id": identifier,
            "parede_id": host,
            "tipo": str(operation.get("type") or operation.get("tipo") or "door"),
            "s_centro": centre,
            "largura": float(operation.get("width") or operation.get("largura") or 0.8),
            "origem": "revision_engine",
        }
        for field in ("altura", "peitoril", "nome"):
            if operation.get(field) is not None:
                opening[field] = operation[field]
        self.model["aberturas"].append(opening)
        return {"opening_id": identifier, "host_wall": host}

    def _op_insert_opening(self, operation: dict) -> dict:
        return self._op_add_opening(operation)

    def _op_move_opening(self, operation: dict) -> dict:
        identifier = str(operation.get("id") or operation.get("element") or "")
        try:
            opening = self._opening_index()[identifier]
        except KeyError as exc:
            raise RevisionError(f"abertura não encontrada: {identifier}") from exc
        before = float(opening["s_centro"])
        if operation.get("s_center") is not None:
            opening["s_centro"] = float(operation["s_center"])
        elif operation.get("s_centro") is not None:
            opening["s_centro"] = float(operation["s_centro"])
        elif operation.get("delta") is not None:
            opening["s_centro"] += float(operation["delta"])
        else:
            raise RevisionError("move_opening exige s_center ou delta")
        return {"opening_id": identifier, "before": before, "after": opening["s_centro"]}

    def _op_resize_opening(self, operation: dict) -> dict:
        identifier = str(operation.get("id") or operation.get("element") or "")
        try:
            opening = self._opening_index()[identifier]
        except KeyError as exc:
            raise RevisionError(f"abertura não encontrada: {identifier}") from exc
        before = float(opening["largura"])
        opening["largura"] = float(operation.get("width") or operation.get("largura"))
        return {"opening_id": identifier, "before": before, "after": opening["largura"]}

    def _op_set_opening_type(self, operation: dict) -> dict:
        identifier = str(operation.get("id") or operation.get("element") or "")
        try:
            opening = self._opening_index()[identifier]
        except KeyError as exc:
            raise RevisionError(f"abertura não encontrada: {identifier}") from exc

        target_type = str(
            operation.get("type") or operation.get("tipo") or ""
        ).lower()
        if target_type not in {"door", "window"}:
            raise RevisionError("set_opening_type exige type door ou window")

        before = {
            "id": identifier,
            "type": str(opening.get("tipo", "")),
            "height": opening.get("altura"),
            "sill": opening.get("peitoril"),
        }
        preserve_head = bool(operation.get("preserve_head", True))
        if (
            before["type"] == "window"
            and target_type == "door"
            and preserve_head
            and opening.get("altura") is not None
        ):
            opening["altura"] = float(opening["altura"]) + float(
                opening.get("peitoril", 0.0)
            )
            opening["peitoril"] = 0.0

        opening["tipo"] = target_type
        if operation.get("height") is not None:
            opening["altura"] = float(operation["height"])
        elif operation.get("altura") is not None:
            opening["altura"] = float(operation["altura"])
        if operation.get("sill") is not None:
            opening["peitoril"] = float(operation["sill"])
        elif operation.get("peitoril") is not None:
            opening["peitoril"] = float(operation["peitoril"])

        new_id = str(operation.get("new_id") or identifier)
        if new_id != identifier:
            if new_id in self._opening_index() or new_id in self._wall_index():
                raise RevisionError(f"ID já existe: {new_id}")
            opening["renamed_from"] = identifier
            opening["id"] = new_id

        return {
            "opening_id": new_id,
            "renamed_from": identifier if new_id != identifier else None,
            "before": before,
            "after": {
                "type": target_type,
                "height": opening.get("altura"),
                "sill": opening.get("peitoril"),
            },
        }

    def _op_change_opening_type(self, operation: dict) -> dict:
        return self._op_set_opening_type(operation)

    def _op_copy_opening_pattern(self, operation: dict) -> dict:
        source_id = str(
            operation.get("source_wall_id") or operation.get("source") or ""
        )
        target_id = str(
            operation.get("target_wall_id") or operation.get("target") or ""
        )
        if not source_id or not target_id:
            raise RevisionError(
                "copy_opening_pattern exige source_wall_id e target_wall_id"
            )
        source_wall = self._wall(source_id)
        target_wall = self._wall(target_id)
        source_axis = wall_axis(source_wall)
        target_axis = wall_axis(target_wall)
        target_length = distance(*target_axis)

        selected_ids = {
            str(value) for value in operation.get("source_opening_ids", [])
        }
        excluded_ids = {
            str(value) for value in operation.get("exclude_source_opening_ids", [])
        }
        source_openings = sorted(
            (
                opening
                for opening in self.model["aberturas"]
                if opening["parede_id"] == source_id
                and (not selected_ids or opening["id"] in selected_ids)
                and opening["id"] not in excluded_ids
            ),
            key=lambda opening: (float(opening["s_centro"]), opening["id"]),
        )
        if not source_openings:
            raise RevisionError(
                f"nenhuma abertura selecionada para copiar de {source_id}"
            )

        removed = []
        if operation.get("replace_target", True):
            removed = [
                opening["id"]
                for opening in self.model["aberturas"]
                if opening["parede_id"] == target_id
            ]
            self.model["aberturas"] = [
                opening
                for opening in self.model["aberturas"]
                if opening["parede_id"] != target_id
            ]

        type_override = operation.get("type") or operation.get("tipo")
        prefix = str(operation.get("id_prefix") or "")
        copied = []
        source_direction = unit(sub(source_axis[1], source_axis[0]))
        existing_ids = set(self._opening_index())
        for index, source_opening in enumerate(source_openings, 1):
            source_centre = add(
                source_axis[0],
                scale(source_direction, float(source_opening["s_centro"])),
            )
            target_centre = project_to_line(source_centre, target_axis, clamp=False)
            target_s = parameter_on_line(target_centre, target_axis)
            width = float(source_opening["largura"])
            if target_s + width * 0.5 < -EPS or target_s - width * 0.5 > target_length + EPS:
                raise RevisionError(
                    f"{source_opening['id']}: projeção cai fora de {target_id}"
                )
            target_type = str(type_override or source_opening.get("tipo") or "door")
            if prefix:
                identifier = f"{prefix}-{index:02d}"
            else:
                marker = "D" if target_type == "door" else "J"
                identifier = f"{marker}-{target_id}-{index:02d}"
            if identifier in existing_ids:
                raise RevisionError(f"ID já existe: {identifier}")
            existing_ids.add(identifier)
            opening = {
                "id": identifier,
                "parede_id": target_id,
                "tipo": target_type,
                "s_centro": target_s,
                "largura": width,
                "origem": "revision_engine.copy_opening_pattern",
                "aligned_from": source_opening["id"],
                "alignment": "perpendicular_projection",
            }
            for field in ("altura", "peitoril", "nome"):
                if source_opening.get(field) is not None:
                    opening[field] = source_opening[field]
            self.model["aberturas"].append(opening)
            copied.append(
                {
                    "source_opening": source_opening["id"],
                    "opening_id": identifier,
                    "s_center": target_s,
                }
            )
        return {
            "source_wall": source_id,
            "target_wall": target_id,
            "removed_target_openings": sorted(removed),
            "copied": copied,
        }

    def _op_close_wall_junctions(self, operation: dict) -> dict:
        max_distance = float(operation.get("max_distance", 0.35))
        min_angle_deg = float(operation.get("min_angle_deg", 25.0))
        segment_tolerance = float(operation.get("segment_tolerance", 0.03))
        opening_clearance = float(operation.get("opening_clearance", 0.08))
        protect_openings = bool(operation.get("protect_openings", True))
        iterations = max(1, int(operation.get("iterations", 2)))
        allowed_kinds = {
            str(value).upper()
            for value in operation.get("kinds", ["L", "T"])
        }
        exclusions = {
            frozenset(map(str, pair))
            for pair in operation.get("exclude_pairs", [])
        }
        moved: dict[str, dict] = {}
        joined: list[dict] = []
        blocked: list[dict] = []

        def opening_at(wall: dict, where: Point) -> str | None:
            axis = wall_axis(wall)
            along = parameter_on_line(where, axis)
            for opening in self.model["aberturas"]:
                if opening["parede_id"] != wall["id"]:
                    continue
                half = float(opening["largura"]) * 0.5 + opening_clearance
                if abs(along - float(opening["s_centro"])) <= half:
                    return str(opening["id"])
            return None

        def movement_crosses_opening(
            wall: dict,
            endpoint: str,
            where: Point,
        ) -> str | None:
            axis = wall_axis(wall)
            length = distance(*axis)
            before = 0.0 if endpoint == "P1" else length
            after = parameter_on_line(where, axis)
            low, high = sorted((before, after))
            if high - low <= EPS:
                return None
            for opening in self.model["aberturas"]:
                if opening["parede_id"] != wall["id"]:
                    continue
                half = float(opening["largura"]) * 0.5 + opening_clearance
                opening_low = float(opening["s_centro"]) - half
                opening_high = float(opening["s_centro"]) + half
                if max(low, opening_low) < min(high, opening_high):
                    return str(opening["id"])
            return None

        for iteration in range(iterations):
            # O fechamento automático atual trabalha com interseções de retas.
            # Curvas permanecem intocadas até a operação ganhar junções arco/linha.
            walls = [
                wall for wall in self.model["paredes"]
                if wall_arc_geometry(wall) is None
            ]
            proposals: dict[
                tuple[str, str],
                tuple[float, Point, str, tuple[str, str]],
            ] = {}
            round_joined: list[dict] = []

            def propose(
                wall: dict,
                endpoint: str,
                gap: float,
                where: Point,
                kind: str,
                pair: tuple[str, str],
            ) -> bool:
                if protect_openings:
                    obstruction = movement_crosses_opening(wall, endpoint, where)
                    if obstruction:
                        blocked.append(
                            {
                                "walls": list(pair),
                                "kind": kind,
                                "reason": "source_opening",
                                "opening_id": obstruction,
                            }
                        )
                        return False
                key = (wall["id"], endpoint)
                current = proposals.get(key)
                if current is None or gap < current[0]:
                    proposals[key] = (gap, where, kind, pair)
                return True

            for left in range(len(walls)):
                wall_a = walls[left]
                axis_a = wall_axis(wall_a)
                direction_a = unit(sub(axis_a[1], axis_a[0]))
                for right in range(left + 1, len(walls)):
                    wall_b = walls[right]
                    pair_key = frozenset((wall_a["id"], wall_b["id"]))
                    if pair_key in exclusions:
                        continue
                    axis_b = wall_axis(wall_b)
                    direction_b = unit(sub(axis_b[1], axis_b[0]))
                    angle = math.degrees(
                        math.acos(min(1.0, abs(dot(direction_a, direction_b))))
                    )
                    if angle < min_angle_deg:
                        continue
                    result = line_parameters(axis_a, axis_b)
                    if result is None:
                        continue
                    intersection, _, _ = result
                    distances_a = [
                        distance(intersection, axis_a[0]),
                        distance(intersection, axis_a[1]),
                    ]
                    distances_b = [
                        distance(intersection, axis_b[0]),
                        distance(intersection, axis_b[1]),
                    ]
                    endpoint_a = "P1" if distances_a[0] <= distances_a[1] else "P2"
                    endpoint_b = "P1" if distances_b[0] <= distances_b[1] else "P2"
                    gap_a, gap_b = min(distances_a), min(distances_b)
                    on_a = (
                        distance(
                            intersection,
                            project_to_line(intersection, axis_a, clamp=True),
                        )
                        <= segment_tolerance
                    )
                    on_b = (
                        distance(
                            intersection,
                            project_to_line(intersection, axis_b, clamp=True),
                        )
                        <= segment_tolerance
                    )
                    pair = (wall_a["id"], wall_b["id"])

                    if "L" in allowed_kinds and gap_a <= max_distance and gap_b <= max_distance:
                        blocked_a = opening_at(wall_a, intersection) if protect_openings else None
                        blocked_b = opening_at(wall_b, intersection) if protect_openings else None
                        if blocked_a or blocked_b:
                            blocked.append(
                                {
                                    "walls": list(pair),
                                    "kind": "L",
                                    "reason": "junction_opening",
                                    "opening_id": blocked_a or blocked_b,
                                }
                            )
                            continue
                        accepted_a = propose(
                            wall_a, endpoint_a, gap_a, intersection, "L", pair
                        )
                        accepted_b = propose(
                            wall_b, endpoint_b, gap_b, intersection, "L", pair
                        )
                        if accepted_a and accepted_b:
                            round_joined.append(
                                {
                                    "walls": list(pair),
                                    "kind": "L",
                                    "point": list(intersection),
                                }
                            )
                        continue

                    if "T" not in allowed_kinds:
                        continue
                    if gap_a <= max_distance and on_b:
                        obstruction = opening_at(wall_b, intersection) if protect_openings else None
                        if obstruction:
                            blocked.append(
                                {
                                    "walls": list(pair),
                                    "kind": "T",
                                    "reason": "target_opening",
                                    "opening_id": obstruction,
                                }
                            )
                        elif propose(
                            wall_a, endpoint_a, gap_a, intersection, "T", pair
                        ):
                            round_joined.append(
                                {
                                    "walls": list(pair),
                                    "kind": "T",
                                    "point": list(intersection),
                                }
                            )
                    if gap_b <= max_distance and on_a:
                        obstruction = opening_at(wall_a, intersection) if protect_openings else None
                        if obstruction:
                            blocked.append(
                                {
                                    "walls": list(pair),
                                    "kind": "T",
                                    "reason": "target_opening",
                                    "opening_id": obstruction,
                                }
                            )
                        elif propose(
                            wall_b, endpoint_b, gap_b, intersection, "T", pair
                        ):
                            round_joined.append(
                                {
                                    "walls": list(pair),
                                    "kind": "T",
                                    "point": list(intersection),
                                }
                            )

            if not proposals:
                break
            for (wall_id, endpoint), (gap, where, kind, pair) in proposals.items():
                wall = self._wall(wall_id)
                before = wall_axis(wall)[0 if endpoint == "P1" else 1]
                if endpoint == "P1":
                    wall["ax"], wall["ay"] = where
                else:
                    wall["bx"], wall["by"] = where
                moved[f"{wall_id}.{endpoint}"] = {
                    "before": list(before),
                    "after": list(where),
                    "distance": gap,
                    "kind": kind,
                    "walls": list(pair),
                    "iteration": iteration + 1,
                }
            joined.extend(round_joined)

        unique_joined = []
        seen_joined = set()
        for item in joined:
            key = (
                tuple(sorted(item["walls"])),
                item["kind"],
                tuple(round(value, 6) for value in item["point"]),
            )
            if key not in seen_joined:
                seen_joined.add(key)
                unique_joined.append(item)
        return {
            "joined": unique_joined,
            "moved_endpoints": moved,
            "blocked_by_openings": blocked,
        }

    def _op_close_small_gaps(self, operation: dict) -> dict:
        max_distance = float(operation.get("max_distance", 0.25))
        max_parallel_cos = float(operation.get("max_parallel_cos", 0.30))
        exclusions = {
            frozenset(map(str, pair))
            for pair in operation.get("exclude_pairs", [])
        }
        walls = self.model["paredes"]
        proposals: dict[tuple[str, str], tuple[float, Point]] = {}
        joined_pairs = []
        for left in range(len(walls)):
            wall_a = walls[left]
            axis_a = wall_axis(wall_a)
            ua = unit(sub(axis_a[1], axis_a[0]))
            for right in range(left + 1, len(walls)):
                wall_b = walls[right]
                if frozenset((wall_a["id"], wall_b["id"])) in exclusions:
                    continue
                axis_b = wall_axis(wall_b)
                ub = unit(sub(axis_b[1], axis_b[0]))
                if abs(dot(ua, ub)) > max_parallel_cos:
                    continue
                intersection = line_intersection(axis_a, axis_b)
                if intersection is None:
                    continue
                candidates = []
                for wall, axis in ((wall_a, axis_a), (wall_b, axis_b)):
                    distances = [
                        distance(intersection, axis[0]),
                        distance(intersection, axis[1]),
                    ]
                    endpoint = "P1" if distances[0] <= distances[1] else "P2"
                    candidates.append((wall["id"], endpoint, min(distances)))
                if any(candidate[2] > max_distance for candidate in candidates):
                    continue
                for wall_id, endpoint, gap in candidates:
                    key = (wall_id, endpoint)
                    if key not in proposals or gap < proposals[key][0]:
                        proposals[key] = (gap, intersection)
                joined_pairs.append([wall_a["id"], wall_b["id"]])
        for (wall_id, endpoint), (_, target) in proposals.items():
            wall = self._wall(wall_id)
            if endpoint == "P1":
                wall["ax"], wall["ay"] = target
            else:
                wall["bx"], wall["by"] = target
        return {
            "joined_pairs": joined_pairs,
            "moved_endpoints": [
                f"{wall_id}.{endpoint}" for wall_id, endpoint in sorted(proposals)
            ],
        }

    def _recalculate_openings(self, policies: dict) -> None:
        wall_index = self._wall_index()
        valid = []
        policy = str(policies.get("opening_out_of_bounds", "clamp")).lower()
        for opening in self.model["aberturas"]:
            wall = wall_index.get(opening["parede_id"])
            if wall is None:
                self.report["warnings"].append(
                    f"{opening['id']}: removida porque a parede hospedeira não existe"
                )
                continue
            length = wall_length(wall)
            width = float(opening["largura"])
            centre = float(opening["s_centro"])
            if width <= 0:
                raise RevisionError(f"{opening['id']}: largura deve ser positiva")
            if policy == "clamp":
                new_width = min(width, length)
                new_centre = min(
                    length - new_width * 0.5,
                    max(new_width * 0.5, centre),
                )
                if abs(new_width - width) > EPS or abs(new_centre - centre) > EPS:
                    self.report["warnings"].append(
                        f"{opening['id']}: ajustada aos limites de {wall['id']}"
                    )
                opening["largura"] = new_width
                opening["s_centro"] = new_centre
            elif centre - width * 0.5 < -EPS or centre + width * 0.5 > length + EPS:
                raise RevisionError(
                    f"{opening['id']}: abertura ultrapassa {wall['id']}"
                )

            wall_height_value = wall.get("altura") or wall.get("altura_observada")
            opening_height_value = opening.get("altura")
            if wall_height_value is not None and opening_height_value is not None:
                wall_height = float(wall_height_value)
                opening_height = float(opening_height_value)
                sill = float(opening.get("peitoril", 0.0) or 0.0)
                if wall_height <= 0 or opening_height <= 0:
                    raise RevisionError(
                        f"{opening['id']}: dimensao vertical invalida"
                    )
                outside_vertical = (
                    sill < -EPS
                    or sill + opening_height > wall_height + EPS
                )
                if outside_vertical and policy == "clamp":
                    opening.setdefault(
                        "vertical_detectado",
                        {
                            "peitoril": sill,
                            "altura": opening_height,
                            "topo": sill + opening_height,
                        },
                    )
                    new_sill = max(0.0, sill)
                    new_height = min(
                        opening_height,
                        max(0.0, wall_height - new_sill),
                    )
                    if new_height <= 0.05:
                        self.report["warnings"].append(
                            f"{opening['id']}: removida porque fica fora da "
                            f"altura de {wall['id']}"
                        )
                        continue
                    opening["peitoril"] = new_sill
                    opening["altura"] = new_height
                    self.report["warnings"].append(
                        f"{opening['id']}: ajustada verticalmente aos limites "
                        f"de {wall['id']}"
                    )
                elif outside_vertical:
                    raise RevisionError(
                        f"{opening['id']}: abertura ultrapassa verticalmente "
                        f"{wall['id']}"
                    )
            valid.append(opening)
        self.model["aberturas"] = valid
        self.report["recalculated"].append("openings")

    def _recalculate_topology(self, policies: dict) -> None:
        tolerance = float(policies.get("topology_snap_tolerance", 0.03))
        min_area = float(policies.get("minimum_space_area", 0.5))
        faces, topology = planar_faces(
            self.model["paredes"],
            snap_tolerance=tolerance,
            min_area=min_area,
        )
        self.model["topology"] = topology
        self.model["_calculated_faces"] = faces
        self.report["recalculated"].append("topology")

    def _recalculate_spaces(self, policies: dict) -> None:
        faces = self.model.pop("_calculated_faces", None)
        if faces is None:
            faces, topology = planar_faces(
                self.model["paredes"],
                snap_tolerance=float(policies.get("topology_snap_tolerance", 0.03)),
                min_area=float(policies.get("minimum_space_area", 0.5)),
            )
            self.model["topology"] = topology
        self.model["spaces"] = [
            {
                "id": f"SPACE-{index:03d}",
                "contorno": [[value[0], value[1]] for value in vertices],
                "area": abs(polygon_area(vertices)),
                "perimetro": polygon_perimeter(vertices),
                "origem": "wall_axis_topology",
            }
            for index, vertices in enumerate(
                sorted(faces, key=lambda values: (-abs(polygon_area(values)), values)),
                1,
            )
        ]
        self.report["recalculated"].append("spaces")

    def _recalculate_slabs(self, policies: dict) -> None:
        mode = str(policies.get("slab_fit_mode", "outer_faces_hull"))
        if mode != "outer_faces_hull":
            raise RevisionError(f"modo de slab ainda não implementado: {mode}")
        hull = convex_hull(
            corner
            for wall in self.model["paredes"]
            for corner in wall_corners(wall)
        )
        self.model.setdefault("laje", {})
        self.model["laje"]["contorno"] = [list(value) for value in hull]
        self.model["laje"].setdefault("piso", {"ativo": True, "espessura": 0.12})
        self.model["laje"].setdefault("teto", {"ativo": True, "espessura": 0.12})
        self.model["laje"]["fit_mode"] = mode
        self.report["recalculated"].append("slabs")

    def _validate(self, policies: dict | None = None) -> dict:
        policies = policies or {}
        errors: list[str] = []
        warnings: list[str] = []
        wall_outside_slab = str(
            policies.get("wall_outside_slab", "error")
        ).lower()
        if wall_outside_slab not in {"error", "warning", "ignore"}:
            raise RevisionError(
                "wall_outside_slab precisa ser error, warning ou ignore"
            )
        wall_ids = [wall["id"] for wall in self.model["paredes"]]
        opening_ids = [opening["id"] for opening in self.model["aberturas"]]
        all_ids = wall_ids + opening_ids
        if len(all_ids) != len(set(all_ids)):
            errors.append("existem IDs duplicados")
        wall_index = self._wall_index()
        for wall in self.model["paredes"]:
            if wall_length(wall) <= 1e-4:
                errors.append(f"{wall['id']}: parede degenerada")
            if float(wall["espessura"]) <= 0:
                errors.append(f"{wall['id']}: espessura inválida")
        for opening in self.model["aberturas"]:
            wall = wall_index.get(opening["parede_id"])
            if wall is None:
                errors.append(f"{opening['id']}: abertura órfã")
                continue
            length = wall_length(wall)
            half = float(opening["largura"]) * 0.5
            centre = float(opening["s_centro"])
            if centre - half < -EPS or centre + half > length + EPS:
                errors.append(f"{opening['id']}: fora dos limites da parede")
        slab = [point(value) for value in self.model.get("laje", {}).get("contorno", [])]
        if self.model["paredes"] and len(slab) < 3:
            errors.append("slab sem contorno válido")
        elif slab:
            for wall in self.model["paredes"]:
                if any(
                    not point_in_polygon(corner, slab, tolerance=1e-6)
                    for corner in wall_corners(wall)
                ):
                    message = f"{wall['id']}: face externa fora do slab"
                    if wall_outside_slab == "error":
                        errors.append(message)
                    elif wall_outside_slab == "warning":
                        warnings.append(message)
        topology = self.model.get("topology", {})
        if topology.get("endpoint_nodes") and not self.model.get("spaces"):
            warnings.append(
                "a rede possui extremidades abertas e não formou nenhum space"
            )
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "wall_count": len(self.model["paredes"]),
            "opening_count": len(self.model["aberturas"]),
            "space_count": len(self.model.get("spaces", [])),
            "slab_vertex_count": len(slab),
        }
