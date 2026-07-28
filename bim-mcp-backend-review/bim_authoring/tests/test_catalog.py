import json
import unittest

from bim_authoring.catalog import RecipeCatalogError, load_default_catalog
from bim_authoring.mcp_adapter import (
    RESOURCE_INDEX_URI,
    describe_mcp_surface,
    get_resource,
    list_recipes,
    search_recipes,
)


class RecipeCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_default_catalog()

    def test_loads_complete_catalog(self):
        self.assertEqual(len(self.catalog), 7)
        self.assertEqual(len(self.catalog.list(status="implemented")), 3)

    def test_search_in_portuguese(self):
        result = self.catalog.search("janela parede")
        self.assertEqual([recipe.id for recipe in result], ["assembly.window-in-wall"])

    def test_unknown_recipe_has_clear_error(self):
        with self.assertRaisesRegex(RecipeCatalogError, "nao encontrada"):
            self.catalog.get("assembly.missing")

    def test_mcp_payloads_are_json_serializable(self):
        payloads = [
            list_recipes(status="implemented", catalog=self.catalog),
            search_recipes("porta", catalog=self.catalog),
            get_resource(RESOURCE_INDEX_URI, catalog=self.catalog),
            describe_mcp_surface(),
        ]
        for payload in payloads:
            json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
