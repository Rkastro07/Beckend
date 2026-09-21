"""Command-line interface for import, revision and layered rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import load_json, model_from_cloud2bim, parts_index, save_json
from .cloud_review import build_cloud_review
from .engine import RevisionEngine
from .render import render_revision_set


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _selected(values: list[str] | None) -> list[str]:
    result = []
    for value in values or []:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def import_cloud(args) -> dict:
    model = model_from_cloud2bim(
        args.diagnostics,
        args.openings,
        args.vertical_levels,
        revision=args.revision,
    )
    save_json(args.output, model)
    return {
        "model": str(Path(args.output).resolve()),
        "walls": len(model["paredes"]),
        "openings": len(model["aberturas"]),
    }


def review_cloud(args) -> dict:
    result = build_cloud_review(
        args.diagnostics,
        args.openings,
        args.output,
        vertical_levels_json=args.vertical_levels,
        revision=args.revision,
    )
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in result.items()
    }


def apply_revision(args) -> dict:
    model = load_json(args.model)
    specification = load_json(args.operations)
    revised, report = RevisionEngine(model).apply(specification)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_json(output_dir / "revision_model.json", revised)
    report_path = save_json(output_dir / "revision_report.json", report)
    parts_path = save_json(output_dir / "element_parts.json", parts_index(revised))
    selected = _selected(args.select)
    if not selected:
        render_config = specification.get("render", {})
        selected = [str(value) for value in render_config.get("selected", [])]
    images = render_revision_set(revised, output_dir, selected=selected)

    if args.export_ifc:
        from plantatobim.planta_to_ifc_v1 import (  # imported only on demand
            dict_para_modelo,
            gerar_ifc_do_modelo,
        )

        internal = dict_para_modelo(revised)
        ifc_config = _deep_merge(
            revised.get("ifc_config", {}),
            specification.get("ifc_config", {}),
        )
        gerar_ifc_do_modelo(
            internal["paredes"],
            internal["aberturas"],
            args.export_ifc,
            config=ifc_config,
            laje=internal["laje"],
            spaces=internal["spaces"],
        )

    return {
        "model": str(model_path.resolve()),
        "report": str(report_path.resolve()),
        "parts": str(parts_path.resolve()),
        "overview": str(images["overview"].resolve()),
        "edit": str(images["edit"].resolve()),
        "ifc": str(Path(args.export_ifc).resolve()) if args.export_ifc else None,
        "walls": len(revised["paredes"]),
        "openings": len(revised["aberturas"]),
        "spaces": len(revised.get("spaces", [])),
        "valid": report.get("validation", {}).get("valid"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Motor determinístico de revisões BIM com seletores P1/P2"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    importer = subparsers.add_parser(
        "import-cloud",
        help="converte diagnósticos Cloud2BIM no modelo editável",
    )
    importer.add_argument("diagnostics", type=Path)
    importer.add_argument("output", type=Path)
    importer.add_argument("--openings", type=Path)
    importer.add_argument("--vertical-levels", type=Path)
    importer.add_argument("--revision", default="R00")
    importer.set_defaults(handler=import_cloud)

    reviewer = subparsers.add_parser(
        "review-cloud",
        help="gera o PNG numerado que deve ser aprovado antes do IFC final",
    )
    reviewer.add_argument("diagnostics", type=Path)
    reviewer.add_argument("openings", type=Path)
    reviewer.add_argument("output", type=Path)
    reviewer.add_argument("--vertical-levels", type=Path)
    reviewer.add_argument("--revision", default="R00-detection")
    reviewer.set_defaults(handler=review_cloud)

    editor = subparsers.add_parser(
        "apply",
        help="aplica uma lista de operações e gera uma nova revisão",
    )
    editor.add_argument("model", type=Path)
    editor.add_argument("operations", type=Path)
    editor.add_argument("output", type=Path)
    editor.add_argument(
        "--select",
        action="append",
        help="elementos/partes destacados no modo de edição, separados por vírgula",
    )
    editor.add_argument("--export-ifc", type=Path)
    editor.set_defaults(handler=apply_revision)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
