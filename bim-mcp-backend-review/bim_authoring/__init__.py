"""Biblioteca de engenharia de autoria BIM para o futuro servidor MCP."""

from .catalog import Recipe, RecipeCatalog, load_default_catalog
from .engine import (
    AssemblyResult,
    AuthoringError,
    IfcAuthoringEngine,
    WallHandle,
)
from .mcp_adapter import (
    describe_mcp_surface,
    get_recipe,
    get_resource,
    list_recipes,
    search_recipes,
)
from .validation import ValidationIssue, ValidationReport, validate_filling

__all__ = [
    "AssemblyResult",
    "AuthoringError",
    "IfcAuthoringEngine",
    "Recipe",
    "RecipeCatalog",
    "ValidationIssue",
    "ValidationReport",
    "WallHandle",
    "describe_mcp_surface",
    "get_recipe",
    "get_resource",
    "list_recipes",
    "load_default_catalog",
    "search_recipes",
    "validate_filling",
]
