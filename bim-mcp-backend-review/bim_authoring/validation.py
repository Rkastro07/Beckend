"""Validacoes semanticas pequenas para objetos hospedados em aberturas IFC."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: str
    message: str
    entity_id: str | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "entity_id": self.entity_id,
        }


@dataclass
class ValidationReport:
    recipe_id: str
    issues: list[ValidationIssue] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add(
        self,
        code: str,
        severity: str,
        message: str,
        entity: Any | None = None,
    ) -> None:
        entity_id = None
        if entity is not None:
            try:
                entity_id = entity.id()
            except Exception:
                entity_id = getattr(entity, "GlobalId", None)
        self.issues.append(
            ValidationIssue(code, severity, message, entity_id)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "ok": self.ok,
            "facts": self.facts,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _ifc_class(entity: Any) -> str:
    try:
        return str(entity.is_a())
    except Exception:
        return type(entity).__name__


def validate_filling(
    element: Any,
    *,
    expected_class: str | None = None,
    expected_host_classes: tuple[str, ...] = ("IfcWall", "IfcSlab", "IfcRoof"),
    require_spatial_containment: bool = True,
    require_relative_placement: bool = True,
) -> ValidationReport:
    """Valida janela/porta/skylight que preenche um IfcOpeningElement."""
    actual_class = _ifc_class(element)
    report = ValidationReport(
        recipe_id=(
            "assembly.window-in-wall"
            if actual_class == "IfcWindow"
            else "assembly.door-in-wall"
        )
    )
    report.facts["element_class"] = actual_class
    if expected_class and actual_class != expected_class:
        report.add(
            "FILLING_CLASS",
            "error",
            f"Esperado {expected_class}, recebido {actual_class}.",
            element,
        )

    fills = list(getattr(element, "FillsVoids", ()) or ())
    report.facts["fills_voids_count"] = len(fills)
    if len(fills) != 1:
        report.add(
            "FILL_RELATION_COUNT",
            "error",
            "O elemento deve preencher exatamente uma abertura.",
            element,
        )
        return report
    opening = getattr(fills[0], "RelatingOpeningElement", None)
    if opening is None or _ifc_class(opening) != "IfcOpeningElement":
        report.add(
            "OPENING_MISSING",
            "error",
            "IfcRelFillsElement nao aponta para um IfcOpeningElement.",
            element,
        )
        return report

    voids = list(getattr(opening, "VoidsElements", ()) or ())
    report.facts["voids_elements_count"] = len(voids)
    if len(voids) != 1:
        report.add(
            "VOID_RELATION_COUNT",
            "error",
            "A abertura deve cortar exatamente um elemento hospedeiro.",
            opening,
        )
    else:
        host = getattr(voids[0], "RelatingBuildingElement", None)
        host_class = _ifc_class(host) if host is not None else "None"
        report.facts["host_class"] = host_class
        if host_class not in expected_host_classes:
            report.add(
                "HOST_CLASS",
                "error",
                f"Hospedeiro {host_class} nao pertence a {expected_host_classes}.",
                host,
            )

    if getattr(element, "Representation", None) is None:
        report.add(
            "FILLING_GEOMETRY",
            "warning",
            "Elemento de preenchimento nao possui representacao geometrica.",
            element,
        )
    if getattr(opening, "Representation", None) is None:
        report.add(
            "OPENING_GEOMETRY",
            "error",
            "Abertura sem geometria nao consegue cortar o hospedeiro.",
            opening,
        )

    placement = getattr(element, "ObjectPlacement", None)
    opening_placement = getattr(opening, "ObjectPlacement", None)
    if placement is None:
        report.add(
            "FILLING_PLACEMENT",
            "error",
            "Elemento com geometria precisa de ObjectPlacement.",
            element,
        )
    elif (
        require_relative_placement
        and opening_placement is not None
        and getattr(placement, "PlacementRelTo", None) is not opening_placement
    ):
        report.add(
            "PLACEMENT_RELATIVE_TO_OPENING",
            "warning",
            "PlacementRelTo deveria apontar para o placement da abertura.",
            element,
        )

    if require_spatial_containment:
        contained = list(getattr(element, "ContainedInStructure", ()) or ())
        report.facts["spatial_containment_count"] = len(contained)
        if len(contained) != 1:
            report.add(
                "SPATIAL_CONTAINMENT",
                "error",
                "Janela/porta deve estar contida em exatamente uma estrutura espacial.",
                element,
            )

    for attribute in ("OverallWidth", "OverallHeight"):
        value = getattr(element, attribute, None)
        if value is not None and float(value) <= 0:
            report.add(
                f"{attribute.upper()}_POSITIVE",
                "error",
                f"{attribute} deve ser positivo.",
                element,
            )
    return report

