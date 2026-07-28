"""Catalogo pesquisavel das receitas de autoria BIM.

Os arquivos de receita sao JSON de proposito: o catalogo funciona apenas com a
biblioteca padrao, mesmo antes de instalar IfcOpenShell, Flask ou PyYAML.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class RecipeCatalogError(ValueError):
    """Receita ausente, duplicada ou estruturalmente invalida."""


@dataclass(frozen=True)
class Recipe:
    id: str
    version: str
    title: str
    summary: str
    status: str
    ifc_versions: tuple[str, ...]
    tags: tuple[str, ...]
    executor: str | None
    data: dict[str, Any]
    path: Path

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path) -> "Recipe":
        required = (
            "schema_version",
            "id",
            "version",
            "title",
            "summary",
            "status",
            "ifc_versions",
            "tags",
            "inputs",
            "entities",
            "relationships",
            "geometry",
            "steps",
            "checks",
            "failure_modes",
            "references",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise RecipeCatalogError(
                f"{path}: campos obrigatorios ausentes: {', '.join(missing)}"
            )
        recipe_id = str(data["id"])
        if not re.fullmatch(r"[a-z][a-z0-9_.-]+", recipe_id):
            raise RecipeCatalogError(f"{path}: id de receita invalido: {recipe_id}")
        if data["status"] not in ("implemented", "documented", "experimental"):
            raise RecipeCatalogError(
                f"{path}: status deve ser implemented, documented ou experimental"
            )
        if not isinstance(data["inputs"], list) or not isinstance(data["steps"], list):
            raise RecipeCatalogError(f"{path}: inputs e steps devem ser listas")
        if not data["steps"]:
            raise RecipeCatalogError(f"{path}: receita sem passos")
        orders = [step.get("order") for step in data["steps"]]
        if orders != sorted(orders) or len(set(orders)) != len(orders):
            raise RecipeCatalogError(
                f"{path}: steps precisam de order unico e crescente"
            )
        return cls(
            id=recipe_id,
            version=str(data["version"]),
            title=str(data["title"]),
            summary=str(data["summary"]),
            status=str(data["status"]),
            ifc_versions=tuple(str(v) for v in data["ifc_versions"]),
            tags=tuple(str(tag).lower() for tag in data["tags"]),
            executor=(str(data["executor"]) if data.get("executor") else None),
            data=data,
            path=path,
        )

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "ifc_versions": list(self.ifc_versions),
            "tags": list(self.tags),
            "executor": self.executor,
        }


class RecipeCatalog:
    def __init__(self, recipes: Iterable[Recipe]):
        self._recipes: dict[str, Recipe] = {}
        for recipe in recipes:
            if recipe.id in self._recipes:
                other = self._recipes[recipe.id]
                raise RecipeCatalogError(
                    f"Receita duplicada {recipe.id}: {other.path} e {recipe.path}"
                )
            self._recipes[recipe.id] = recipe

    @classmethod
    def from_directory(cls, root: str | Path) -> "RecipeCatalog":
        root = Path(root)
        if not root.is_dir():
            raise RecipeCatalogError(f"Diretorio de receitas inexistente: {root}")
        recipes = []
        for path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RecipeCatalogError(f"Falha ao ler {path}: {exc}") from exc
            recipes.append(Recipe.from_dict(data, path))
        if not recipes:
            raise RecipeCatalogError(f"Nenhuma receita JSON em {root}")
        return cls(recipes)

    def list(
        self,
        *,
        tags: Iterable[str] | None = None,
        status: str | None = None,
        ifc_version: str | None = None,
    ) -> list[Recipe]:
        required_tags = {tag.lower() for tag in (tags or ())}
        result = []
        for recipe in self._recipes.values():
            if required_tags and not required_tags.issubset(set(recipe.tags)):
                continue
            if status and recipe.status != status:
                continue
            if ifc_version and ifc_version not in recipe.ifc_versions:
                continue
            result.append(recipe)
        return sorted(result, key=lambda recipe: recipe.id)

    def get(self, recipe_id: str) -> Recipe:
        try:
            return self._recipes[recipe_id]
        except KeyError as exc:
            suggestions = [
                recipe.id for recipe in self.search(recipe_id, limit=5)
            ]
            suffix = f" Sugestoes: {', '.join(suggestions)}." if suggestions else ""
            raise RecipeCatalogError(
                f"Receita '{recipe_id}' nao encontrada.{suffix}"
            ) from exc

    def search(
        self,
        query: str,
        *,
        tags: Iterable[str] | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[Recipe]:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if len(token) > 1
        ]
        candidates = self.list(tags=tags, status=status)
        scored = []
        for recipe in candidates:
            title = recipe.title.lower()
            recipe_id = recipe.id.lower()
            tags_text = " ".join(recipe.tags)
            haystack = " ".join(
                (recipe_id, title, recipe.summary.lower(), tags_text)
            )
            if tokens and not all(token in haystack for token in tokens):
                continue
            score = sum(
                8 if token in recipe_id else
                5 if token in title else
                3 if token in tags_text else 1
                for token in tokens
            )
            scored.append((score, recipe.id, recipe))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored[: max(1, limit)]]

    def tags(self) -> list[str]:
        return sorted({tag for recipe in self._recipes.values() for tag in recipe.tags})

    def as_resource_index(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "resource_uri": "bim://authoring/recipes",
            "count": len(self._recipes),
            "tags": self.tags(),
            "recipes": [recipe.to_summary() for recipe in self.list()],
        }

    def __len__(self) -> int:
        return len(self._recipes)


def load_default_catalog() -> RecipeCatalog:
    return RecipeCatalog.from_directory(Path(__file__).with_name("recipes"))
