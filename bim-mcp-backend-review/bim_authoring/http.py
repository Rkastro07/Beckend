"""Rotas somente-leitura do catalogo BIM para o backend local atual."""
from __future__ import annotations

from typing import Any

from .catalog import RecipeCatalogError, load_default_catalog
from .mcp_adapter import describe_mcp_surface


def _tags_from_args(args: Any) -> list[str]:
    tags = []
    for raw in args.getlist("tag"):
        tags.extend(value.strip() for value in raw.split(",") if value.strip())
    return tags


def register_bim_authoring_routes(app: Any) -> None:
    """Registra descoberta HTTP sem importar IfcOpenShell nem executar escrita."""
    catalog = load_default_catalog()

    @app.get("/api/bim-authoring/recipes")
    def _bim_authoring_recipes():
        from flask import jsonify, request

        query = (request.args.get("q") or "").strip()
        tags = _tags_from_args(request.args)
        status = request.args.get("status") or None
        if query:
            recipes = catalog.search(
                query,
                tags=tags,
                status=status,
                limit=min(max(request.args.get("limit", 20, type=int), 1), 100),
            )
        else:
            recipes = catalog.list(
                tags=tags,
                status=status,
                ifc_version=request.args.get("ifc_version") or None,
            )
        return jsonify(
            {
                "count": len(recipes),
                "tags": catalog.tags(),
                "recipes": [recipe.to_summary() for recipe in recipes],
            }
        )

    @app.get("/api/bim-authoring/recipes/<recipe_id>")
    def _bim_authoring_recipe(recipe_id: str):
        from flask import jsonify

        try:
            return jsonify(catalog.get(recipe_id).data)
        except RecipeCatalogError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.get("/api/bim-authoring/mcp-surface")
    def _bim_authoring_mcp_surface():
        from flask import jsonify

        return jsonify(describe_mcp_surface())
