"""Superficie serializavel da biblioteca para HTTP e um futuro servidor MCP.

Este modulo nao depende do SDK MCP. Assim o mesmo contrato pode ser exposto
agora pela API Flask e, depois, por Resources e Tools sem duplicar o catalogo.
"""
from __future__ import annotations

from typing import Any, Iterable

from .catalog import RecipeCatalog, load_default_catalog


RESOURCE_INDEX_URI = "bim://authoring/recipes"
RESOURCE_RECIPE_PREFIX = "bim://authoring/recipe/"


def _catalog_or_default(catalog: RecipeCatalog | None) -> RecipeCatalog:
    return catalog if catalog is not None else load_default_catalog()


def list_recipes(
    *,
    tags: Iterable[str] | None = None,
    status: str | None = None,
    ifc_version: str | None = None,
    catalog: RecipeCatalog | None = None,
) -> dict[str, Any]:
    selected = _catalog_or_default(catalog).list(
        tags=tags,
        status=status,
        ifc_version=ifc_version,
    )
    return {
        "count": len(selected),
        "recipes": [recipe.to_summary() for recipe in selected],
    }


def search_recipes(
    query: str,
    *,
    tags: Iterable[str] | None = None,
    status: str | None = None,
    limit: int = 20,
    catalog: RecipeCatalog | None = None,
) -> dict[str, Any]:
    selected = _catalog_or_default(catalog).search(
        query,
        tags=tags,
        status=status,
        limit=limit,
    )
    return {
        "query": query,
        "count": len(selected),
        "recipes": [recipe.to_summary() for recipe in selected],
    }


def get_recipe(
    recipe_id: str,
    *,
    catalog: RecipeCatalog | None = None,
) -> dict[str, Any]:
    return _catalog_or_default(catalog).get(recipe_id).data


def get_resource(
    uri: str,
    *,
    catalog: RecipeCatalog | None = None,
) -> dict[str, Any]:
    selected_catalog = _catalog_or_default(catalog)
    if uri == RESOURCE_INDEX_URI:
        return selected_catalog.as_resource_index()
    if uri.startswith(RESOURCE_RECIPE_PREFIX):
        recipe_id = uri.removeprefix(RESOURCE_RECIPE_PREFIX)
        return selected_catalog.get(recipe_id).data
    raise ValueError(f"Recurso BIM desconhecido: {uri}")


def describe_mcp_surface() -> dict[str, Any]:
    """Declaracao neutra que orienta o adaptador MCP a ser criado."""
    return {
        "resources": [
            {
                "uri": RESOURCE_INDEX_URI,
                "name": "Indice de receitas de autoria BIM",
                "mime_type": "application/json",
            },
            {
                "uri_template": f"{RESOURCE_RECIPE_PREFIX}{{recipe_id}}",
                "name": "Receita de autoria BIM",
                "mime_type": "application/json",
            },
        ],
        "read_tools": [
            "bim_authoring_list_recipes",
            "bim_authoring_search_recipes",
            "bim_authoring_get_recipe",
        ],
        "write_tools_planned": [
            "bim_authoring_create_wall",
            "bim_authoring_insert_window",
            "bim_authoring_insert_door",
            "bim_authoring_validate_assembly",
        ],
    }
