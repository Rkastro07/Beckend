"""Executor das primeiras receitas BIM sobre a API de alto nivel IfcOpenShell."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .geometry import (
    GeometryRuleError,
    offset_frame,
    point3,
    validate_hosted_opening,
    wall_frame,
)


class AuthoringError(ValueError):
    """Falha de pre-condicao ou de execucao de uma receita."""


ApiRunner = Callable[..., Any]


def _default_api_runner(usecase: str, model: Any, **kwargs: Any) -> Any:
    try:
        import ifcopenshell.api as api
    except ImportError as exc:
        raise AuthoringError(
            "IfcOpenShell nao esta instalado. Instale requirements-cloud2bim.txt."
        ) from exc
    return api.run(usecase, model, **kwargs)


@dataclass
class WallHandle:
    entity: Any
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    matrix: np.ndarray
    length: float
    height: float
    thickness: float
    elevation: float
    storey: Any | None
    body_context: Any


@dataclass
class AssemblyResult:
    host: Any
    opening: Any
    filling: Any
    void_relation: Any
    fill_relation: Any
    recipe_id: str


class IfcAuthoringEngine:
    """Operacoes geometricas com pre-condicoes mensuraveis.

    Coordenadas e dimensoes recebidas por esta classe sao sempre SI (metros).
    O runner e injetavel para testes; em producao usa ``ifcopenshell.api.run``.
    """

    def __init__(self, model: Any, api_runner: ApiRunner | None = None):
        self.model = model
        self.api_runner = api_runner or _default_api_runner

    def _run(self, usecase: str, **kwargs: Any) -> Any:
        try:
            return self.api_runner(usecase, self.model, **kwargs)
        except AuthoringError:
            raise
        except Exception as exc:
            raise AuthoringError(f"{usecase} falhou: {exc}") from exc

    @staticmethod
    def _set_ifc_attributes(entity: Any, **attributes: Any) -> None:
        for name, value in attributes.items():
            if value is None:
                continue
            try:
                setattr(entity, name, value)
            except Exception:
                # Diferencas de schema podem tornar um atributo indisponivel.
                pass

    def create_wall(
        self,
        *,
        start: Sequence[float],
        end: Sequence[float],
        height: float,
        thickness: float,
        body_context: Any,
        storey: Any | None = None,
        elevation: float = 0.0,
        name: str = "Parede",
        predefined_type: str = "STANDARD",
    ) -> WallHandle:
        if height <= 0:
            raise AuthoringError("Parede precisa de altura positiva")
        if thickness <= 0:
            raise AuthoringError("Parede precisa de espessura positiva")
        try:
            matrix, length = wall_frame(start, end, elevation=elevation)
        except GeometryRuleError as exc:
            raise AuthoringError(str(exc)) from exc

        wall = self._run(
            "root.create_entity",
            ifc_class="IfcWall",
            name=name,
            predefined_type=predefined_type,
        )
        representation = self._run(
            "geometry.add_wall_representation",
            context=body_context,
            length=float(length),
            height=float(height),
            thickness=float(thickness),
        )
        self._run(
            "geometry.assign_representation",
            product=wall,
            representation=representation,
        )
        if storey is not None:
            self._run(
                "spatial.assign_container",
                products=[wall],
                relating_structure=storey,
            )
        self._run(
            "geometry.edit_object_placement",
            product=wall,
            matrix=matrix,
            is_si=True,
        )
        a = point3(start, name="start")
        b = point3(end, name="end")
        a[2] = b[2] = float(elevation)
        return WallHandle(
            entity=wall,
            start=tuple(float(value) for value in a),
            end=tuple(float(value) for value in b),
            matrix=matrix,
            length=length,
            height=float(height),
            thickness=float(thickness),
            elevation=float(elevation),
            storey=storey,
            body_context=body_context,
        )

    def insert_window(
        self,
        host: WallHandle,
        *,
        offset_from_start: float,
        width: float,
        height: float,
        sill_height: float,
        name: str = "Janela",
        partition_type: str = "SINGLE_PANEL",
        lining_properties: dict[str, Any] | None = None,
        panel_properties: list[dict[str, Any]] | None = None,
        boolean_margin: float = 0.05,
        end_clearance: float = 0.0,
    ) -> AssemblyResult:
        representation_kwargs: dict[str, Any] = {
            "context": host.body_context,
            "overall_height": float(height),
            "overall_width": float(width),
            "partition_type": partition_type,
        }
        if lining_properties is not None:
            representation_kwargs["lining_properties"] = lining_properties
        if panel_properties is not None:
            representation_kwargs["panel_properties"] = panel_properties
        return self._insert_filling(
            host,
            recipe_id="assembly.window-in-wall",
            ifc_class="IfcWindow",
            predefined_type="WINDOW",
            geometry_usecase="geometry.add_window_representation",
            geometry_kwargs=representation_kwargs,
            offset_from_start=offset_from_start,
            width=width,
            height=height,
            sill_height=sill_height,
            name=name,
            boolean_margin=boolean_margin,
            end_clearance=end_clearance,
        )

    def insert_door(
        self,
        host: WallHandle,
        *,
        offset_from_start: float,
        width: float,
        height: float,
        name: str = "Porta",
        operation_type: str = "SINGLE_SWING_LEFT",
        lining_properties: dict[str, Any] | None = None,
        panel_properties: dict[str, Any] | None = None,
        boolean_margin: float = 0.05,
        end_clearance: float = 0.0,
    ) -> AssemblyResult:
        representation_kwargs: dict[str, Any] = {
            "context": host.body_context,
            "overall_height": float(height),
            "overall_width": float(width),
            "operation_type": operation_type,
        }
        if lining_properties is not None:
            representation_kwargs["lining_properties"] = lining_properties
        if panel_properties is not None:
            representation_kwargs["panel_properties"] = panel_properties
        return self._insert_filling(
            host,
            recipe_id="assembly.door-in-wall",
            ifc_class="IfcDoor",
            predefined_type="DOOR",
            geometry_usecase="geometry.add_door_representation",
            geometry_kwargs=representation_kwargs,
            offset_from_start=offset_from_start,
            width=width,
            height=height,
            sill_height=0.0,
            name=name,
            boolean_margin=boolean_margin,
            end_clearance=end_clearance,
        )

    def _insert_filling(
        self,
        host: WallHandle,
        *,
        recipe_id: str,
        ifc_class: str,
        predefined_type: str,
        geometry_usecase: str,
        geometry_kwargs: dict[str, Any],
        offset_from_start: float,
        width: float,
        height: float,
        sill_height: float,
        name: str,
        boolean_margin: float,
        end_clearance: float,
    ) -> AssemblyResult:
        if boolean_margin <= 0:
            raise AuthoringError("boolean_margin precisa ser positivo")
        try:
            validate_hosted_opening(
                host_length=host.length,
                host_height=host.height,
                offset_from_start=float(offset_from_start),
                width=float(width),
                height=float(height),
                sill_height=float(sill_height),
                end_clearance=float(end_clearance),
            )
        except GeometryRuleError as exc:
            raise AuthoringError(str(exc)) from exc

        opening = self._run(
            "root.create_entity",
            ifc_class="IfcOpeningElement",
            name=f"Vao-{name}",
        )
        opening_representation = self._run(
            "geometry.add_wall_representation",
            context=host.body_context,
            length=float(width),
            height=float(height),
            thickness=float(host.thickness + 2 * boolean_margin),
        )
        self._run(
            "geometry.assign_representation",
            product=opening,
            representation=opening_representation,
        )
        void_relation = self._run(
            "feature.add_feature",
            feature=opening,
            element=host.entity,
        )
        opening_matrix = offset_frame(
            host.matrix,
            along=float(offset_from_start),
            normal=-float(boolean_margin),
            vertical=float(sill_height),
        )
        self._run(
            "geometry.edit_object_placement",
            product=opening,
            matrix=opening_matrix,
            is_si=True,
        )

        filling = self._run(
            "root.create_entity",
            ifc_class=ifc_class,
            name=name,
            predefined_type=predefined_type,
        )
        filling_representation = self._run(
            geometry_usecase,
            **geometry_kwargs,
        )
        self._run(
            "geometry.assign_representation",
            product=filling,
            representation=filling_representation,
        )
        self._set_ifc_attributes(
            filling,
            OverallHeight=float(height),
            OverallWidth=float(width),
        )
        fill_relation = self._run(
            "feature.add_filling",
            opening=opening,
            element=filling,
        )
        if host.storey is not None:
            self._run(
                "spatial.assign_container",
                products=[filling],
                relating_structure=host.storey,
            )
        filling_matrix = offset_frame(
            host.matrix,
            along=float(offset_from_start),
            normal=float(host.thickness / 2.0),
            vertical=float(sill_height),
        )
        # Executado depois de add_filling para o IfcOpenShell poder escolher o
        # IfcOpeningElement como PlacementRelTo e manter a matriz de mundo.
        self._run(
            "geometry.edit_object_placement",
            product=filling,
            matrix=filling_matrix,
            is_si=True,
        )
        return AssemblyResult(
            host=host.entity,
            opening=opening,
            filling=filling,
            void_relation=void_relation,
            fill_relation=fill_relation,
            recipe_id=recipe_id,
        )

