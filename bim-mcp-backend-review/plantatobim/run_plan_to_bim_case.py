"""Executa o mesmo nucleo do app Planta-to-BIM com saida isolada.

Serve para testes reproduziveis sem sobrescrever o DXF ou os IFCs originais.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from planta_to_ifc_v1 import (
    gerar_ifc_do_modelo,
    ifc_para_ply,
    modelo_para_dict,
    parse_dxf,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dxf", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--altura", type=float, default=2.80)
    parser.add_argument("--esp-laje", type=float, default=0.12)
    parser.add_argument("--esp-parede", type=float, default=0.15)
    parser.add_argument("--escala", type=float, default=None)
    parser.add_argument("--esquadria-detalhada", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dxf.stem}_current_app"
    ifc_path = args.output_dir / f"{stem}.ifc"
    ply_path = args.output_dir / f"{stem}_preview.ply"
    model_path = args.output_dir / f"{stem}_model.json"

    model = parse_dxf(
        args.dxf,
        escala_forcada=args.escala,
        esp_default=args.esp_parede,
    )
    config = {
        "altura": args.altura,
        "esp_laje": args.esp_laje,
        "pavimento": "Terreo",
        "projeto": "Door Ground Floor - Current App",
        "cobertura": True,
        "porta_altura": 2.10,
        "janela_altura": 1.20,
        "janela_peitoril": 1.00,
        "esquadria_detalhada": args.esquadria_detalhada,
    }
    gerar_ifc_do_modelo(
        model["paredes"],
        model["aberturas"],
        ifc_path,
        config,
    )
    mesh_elements, triangles = ifc_para_ply(ifc_path, ply_path)
    serializable = modelo_para_dict(model)
    serializable["source"] = {
        "path": str(args.dxf),
        "generator": "plantatobim.planta_to_ifc_v1",
        "case": "current_app",
    }
    model_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "source": str(args.dxf),
        "ifc": str(ifc_path),
        "ply": str(ply_path),
        "model_json": str(model_path),
        "scale": model["escala"],
        "single_line": model["single_line"],
        "walls": len(model["paredes"]),
        "doors": sum(
            1 for opening in model["aberturas"] if opening["tipo"] == "door"
        ),
        "windows": sum(
            1 for opening in model["aberturas"] if opening["tipo"] == "window"
        ),
        "slabs": 2,
        "mesh_elements": mesh_elements,
        "triangles": triangles,
        "slab_vertices": len(model["laje_contorno"]),
    }
    (args.output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
