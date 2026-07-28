"""Geometria pequena e deterministica usada pelas receitas de autoria."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


class GeometryRuleError(ValueError):
    """Dimensao ou posicionamento incompativel com a receita."""


def point3(value: Sequence[float], *, name: str) -> np.ndarray:
    if len(value) not in (2, 3):
        raise GeometryRuleError(f"{name} deve ter 2 ou 3 coordenadas")
    result = np.asarray(
        (value[0], value[1], value[2] if len(value) == 3 else 0.0),
        dtype=float,
    )
    if not np.isfinite(result).all():
        raise GeometryRuleError(f"{name} contem coordenada nao finita")
    return result


def wall_frame(
    start: Sequence[float],
    end: Sequence[float],
    *,
    elevation: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Matriz mundo onde X segue o eixo da parede, Y e a normal e Z sobe."""
    a = point3(start, name="start")
    b = point3(end, name="end")
    a[2] = float(elevation)
    b[2] = float(elevation)
    direction = b - a
    direction[2] = 0.0
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        raise GeometryRuleError("Parede precisa ter comprimento positivo")
    x_axis = direction / length
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    matrix = np.eye(4)
    matrix[:3, 0] = x_axis
    matrix[:3, 1] = y_axis
    matrix[:3, 2] = z_axis
    matrix[:3, 3] = a
    return matrix, length


def offset_frame(
    host_matrix: np.ndarray,
    *,
    along: float,
    normal: float = 0.0,
    vertical: float = 0.0,
) -> np.ndarray:
    matrix = np.asarray(host_matrix, dtype=float).copy()
    if matrix.shape != (4, 4):
        raise GeometryRuleError("host_matrix deve ser 4x4")
    matrix[:3, 3] = (
        host_matrix[:3, 3]
        + host_matrix[:3, 0] * float(along)
        + host_matrix[:3, 1] * float(normal)
        + host_matrix[:3, 2] * float(vertical)
    )
    return matrix


def validate_hosted_opening(
    *,
    host_length: float,
    host_height: float,
    offset_from_start: float,
    width: float,
    height: float,
    sill_height: float,
    end_clearance: float = 0.0,
) -> None:
    values = {
        "host_length": host_length,
        "host_height": host_height,
        "offset_from_start": offset_from_start,
        "width": width,
        "height": height,
        "sill_height": sill_height,
        "end_clearance": end_clearance,
    }
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise GeometryRuleError("Dimensoes devem ser finitas")
    if width <= 0 or height <= 0:
        raise GeometryRuleError("Abertura precisa de largura e altura positivas")
    if offset_from_start < end_clearance:
        raise GeometryRuleError("Abertura invade a folga inicial da parede")
    if offset_from_start + width > host_length - end_clearance + 1e-9:
        raise GeometryRuleError("Abertura ultrapassa o fim da parede")
    if sill_height < 0:
        raise GeometryRuleError("Peitoril nao pode ser negativo")
    if sill_height + height > host_height + 1e-9:
        raise GeometryRuleError("Abertura ultrapassa o topo da parede")

