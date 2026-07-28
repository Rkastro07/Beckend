"""Aplica a revisao solicitada ao caso Door Ground Floor.

Revisao:
- W-011 e W-013 recebem o mesmo alcance longitudinal de W-010;
- W-018 fecha W-010 -> W-013 no alinhamento traseiro;
- W-019 fecha W-010 -> W-013 no alinhamento frontal;
- piso e cobertura passam a usar o retangulo das faces externas de todas as
  paredes, cobrindo o modelo de ponta a ponta.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from planta_to_ifc_v1 import dict_para_modelo, gerar_ifc_do_modelo, ifc_para_ply


def wall_by_code(model: dict[str, Any], code: str) -> dict[str, Any]:
    index = int(code.rsplit("-", 1)[-1]) - 1
    wall_id = f"w{index}"
    return next(wall for wall in model["paredes"] if wall["id"] == wall_id)


def set_vertical_range(
    wall: dict[str, Any],
    *,
    ymin: float,
    ymax: float,
) -> None:
    if not math.isclose(float(wall["ax"]), float(wall["bx"]), abs_tol=1e-4):
        raise ValueError(f"{wall['id']} nao e vertical neste caso")
    x = (float(wall["ax"]) + float(wall["bx"])) / 2
    wall.update({"ax": x, "ay": ymin, "bx": x, "by": ymax})


def wall_footprint_points(wall: dict[str, Any]) -> list[tuple[float, float]]:
    ax, ay = float(wall["ax"]), float(wall["ay"])
    bx, by = float(wall["bx"]), float(wall["by"])
    thickness = float(wall["espessura"])
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return []
    nx, ny = -dy / length * thickness / 2, dx / length * thickness / 2
    return [
        (ax + nx, ay + ny),
        (ax - nx, ay - ny),
        (bx + nx, by + ny),
        (bx - nx, by - ny),
    ]


def rebuild_slab_rectangle(model: dict[str, Any]) -> dict[str, float]:
    points = [
        point
        for wall in model["paredes"]
        for point in wall_footprint_points(wall)
    ]
    xmin = min(point[0] for point in points)
    ymin = min(point[1] for point in points)
    xmax = max(point[0] for point in points)
    ymax = max(point[1] for point in points)
    model["laje"] = {
        "contorno": [
            [round(xmin, 4), round(ymin, 4)],
            [round(xmax, 4), round(ymin, 4)],
            [round(xmax, 4), round(ymax, 4)],
            [round(xmin, 4), round(ymax, 4)],
        ],
        "piso": {"ativo": True, "espessura": 0.12},
        "teto": {"ativo": True, "espessura": 0.12},
    }
    model["bbox"] = {
        "xmin": round(xmin, 4),
        "ymin": round(ymin, 4),
        "xmax": round(xmax, 4),
        "ymax": round(ymax, 4),
    }
    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "width": xmax - xmin,
        "depth": ymax - ymin,
        "area": (xmax - xmin) * (ymax - ymin),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model = json.loads(args.model_json.read_text(encoding="utf-8"))
    w010 = wall_by_code(model, "W-010")
    w011 = wall_by_code(model, "W-011")
    w013 = wall_by_code(model, "W-013")

    reference_y = sorted((float(w010["ay"]), float(w010["by"])))
    front_y, back_y = reference_y
    set_vertical_range(w011, ymin=front_y, ymax=back_y)
    set_vertical_range(w013, ymin=front_y, ymax=back_y)

    x010 = (float(w010["ax"]) + float(w010["bx"])) / 2
    x013 = (float(w013["ax"]) + float(w013["bx"])) / 2
    closure_thickness = max(
        float(w010["espessura"]),
        float(w013["espessura"]),
    )
    model["paredes"].extend(
        [
            {
                "id": "w17",
                "ax": x010,
                "ay": back_y,
                "bx": x013,
                "by": back_y,
                "espessura": closure_thickness,
                "layer": "Wall-Ext",
                "nome": "Fechamento-Traseiro",
            },
            {
                "id": "w18",
                "ax": x010,
                "ay": front_y,
                "bx": x013,
                "by": front_y,
                "espessura": closure_thickness,
                "layer": "Wall-Ext",
                "nome": "Fechamento-Frontal",
            },
        ]
    )
    slab = rebuild_slab_rectangle(model)
    model["edit_history"] = [
        {
            "revision": "R01",
            "operations": [
                "W-011 igualada longitudinalmente a W-010",
                "W-013 igualada longitudinalmente a W-010",
                "W-018 adicionada entre W-010 e W-013 no fundo",
                "W-019 adicionada entre W-010 e W-013 na frente",
                "piso e teto reconstruidos pelas faces externas de todas as paredes",
            ],
        }
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    edited_json = args.output_dir / "doorgrondfloor_edited_r01_model.json"
    edited_ifc = args.output_dir / "doorgrondfloor_edited_r01.ifc"
    edited_ply = args.output_dir / "doorgrondfloor_edited_r01_preview.ply"
    summary_path = args.output_dir / "doorgrondfloor_edited_r01_summary.json"
    edited_json.write_text(
        json.dumps(model, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    internal = dict_para_modelo(model)
    config = {
        "altura": 2.80,
        "esp_laje": 0.12,
        "pavimento": "Terreo",
        "projeto": "Door Ground Floor - Edited R01",
        "cobertura": True,
        "porta_altura": 2.10,
        "janela_altura": 1.20,
        "janela_peitoril": 1.00,
        "esquadria_detalhada": True,
    }
    gerar_ifc_do_modelo(
        internal["paredes"],
        internal["aberturas"],
        edited_ifc,
        config,
        laje=internal["laje"],
    )
    mesh_elements, triangles = ifc_para_ply(edited_ifc, edited_ply)
    summary = {
        "revision": "R01",
        "ifc": str(edited_ifc),
        "model_json": str(edited_json),
        "walls": len(model["paredes"]),
        "doors": len(model["aberturas"]),
        "slabs": 2,
        "slab": slab,
        "mesh_elements": mesh_elements,
        "triangles": triangles,
        "equalized_wall_length": back_y - front_y,
        "new_wall_codes": ["W-018", "W-019"],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
