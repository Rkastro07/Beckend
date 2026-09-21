from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import ifcopenshell
import ifcopenshell.geom
import numpy as np

from plantatobim.planta_to_ifc_v1 import (
    dict_para_modelo,
    gerar_ifc_do_modelo,
)


class IfcSpaceExportTests(unittest.TestCase):
    def test_exports_curved_wall_as_one_ifc_wall_with_hosted_window(self):
        editable = {
            "paredes": [{
                "id": "W-ARC",
                "ax": 0.0,
                "ay": 0.0,
                "bx": 4.0,
                "by": 0.0,
                "espessura": 0.15,
                "layer": "Wall",
                "geometria": "arco",
                "curva": {"x": 2.0, "y": 2.0},
            }],
            "aberturas": [{
                "id": "J-ARC-01",
                "parede_id": "W-ARC",
                "tipo": "window",
                "s_centro": 3.14159265,
                "largura": 1.0,
                "altura": 1.2,
                "peitoril": 1.0,
            }],
            "laje": {
                "contorno": [[-1, -1], [5, -1], [5, 3], [-1, 3]],
                "piso": {"ativo": True, "espessura": 0.12},
                "teto": {"ativo": False, "espessura": 0.12},
            },
        }
        internal = dict_para_modelo(editable)
        self.assertAlmostEqual(internal["paredes"][0]["comprimento"], 2 * np.pi)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "curved.ifc"
            gerar_ifc_do_modelo(
                internal["paredes"],
                internal["aberturas"],
                output,
                config={"altura": 2.8},
                laje=internal["laje"],
            )
            model = ifcopenshell.open(str(output))
            wall = model.by_type("IfcWall")[0]
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)
            shape = ifcopenshell.geom.create_shape(settings, wall)
            vertices = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)

        self.assertEqual(len(model.by_type("IfcWall")), 1)
        self.assertEqual(len(model.by_type("IfcWindow")), 1)
        self.assertEqual(len(model.by_type("IfcRelVoidsElement")), 1)
        self.assertGreater(float(vertices[:, 1].max()), 1.9)

    def test_exports_editor_spaces_to_ifc(self):
        editable = {
            "paredes": [
                {
                    "id": "W-001",
                    "ax": 0.0,
                    "ay": 0.0,
                    "bx": 4.0,
                    "by": 0.0,
                    "espessura": 0.15,
                    "layer": "Wall",
                }
            ],
            "aberturas": [],
            "spaces": [
                {
                    "id": "SPACE-001",
                    "contorno": [[0, 0], [4, 0], [4, 3], [0, 3]],
                    "area": 12.0,
                }
            ],
            "laje": {
                "contorno": [[0, 0], [4, 0], [4, 3], [0, 3]],
                "piso": {"ativo": True, "espessura": 0.12},
                "teto": {"ativo": True, "espessura": 0.12},
            },
        }
        internal = dict_para_modelo(editable)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "spaces.ifc"
            gerar_ifc_do_modelo(
                internal["paredes"],
                internal["aberturas"],
                output,
                config={"altura": 3.0},
                laje=internal["laje"],
                spaces=internal["spaces"],
            )
            model = ifcopenshell.open(str(output))

        spaces = model.by_type("IfcSpace")
        self.assertEqual(len(spaces), 1)
        self.assertEqual(spaces[0].Name, "SPACE-001")
        self.assertEqual(spaces[0].PredefinedType, "INTERNAL")

    def test_exports_ceiling_inside_space_and_detailed_door_relations(self):
        editable = {
            "paredes": [
                {
                    "id": "W-001",
                    "ax": 0.0,
                    "ay": 0.0,
                    "bx": 4.0,
                    "by": 0.0,
                    "espessura": 0.15,
                    "layer": "Wall",
                }
            ],
            "aberturas": [
                {
                    "id": "D-W-001-01",
                    "parede_id": "W-001",
                    "tipo": "door",
                    "s_centro": 2.0,
                    "largura": 0.8,
                    "altura": 2.1,
                    "peitoril": 0.0,
                }
            ],
            "spaces": [
                {
                    "id": "SPACE-001",
                    "contorno": [[0, 0], [4, 0], [4, 3], [0, 3]],
                    "area": 12.0,
                }
            ],
            "laje": {
                "contorno": [[0, 0], [4, 0], [4, 3], [0, 3]],
                "piso": {"ativo": True, "espessura": 0.12},
                "teto": {"ativo": True, "espessura": 0.12},
            },
        }
        internal = dict_para_modelo(editable)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ceiling_and_door.ifc"
            gerar_ifc_do_modelo(
                internal["paredes"],
                internal["aberturas"],
                output,
                config={
                    "altura": 3.0,
                    "esquadria_detalhada": True,
                    "esquadria_sobreposicao": 0.02,
                    "forro": {
                        "ativo": True,
                        "altura": 2.7,
                        "espessura": 0.03,
                    },
                },
                laje=internal["laje"],
                spaces=internal["spaces"],
            )
            model = ifcopenshell.open(str(output))

            ceiling = model.by_type("IfcCovering")[0]
            space = model.by_type("IfcSpace")[0]
            self.assertEqual(ceiling.PredefinedType, "CEILING")
            self.assertEqual(
                ceiling.ContainedInStructure[0].RelatingStructure,
                space,
            )
            self.assertEqual(len(model.by_type("IfcRelVoidsElement")), 1)
            self.assertEqual(len(model.by_type("IfcRelFillsElement")), 1)
            self.assertEqual(
                model.by_type("IfcRelVoidsElement")[0]
                .RelatingBuildingElement.Name,
                "W-001",
            )
            self.assertEqual(
                model.by_type("IfcRelFillsElement")[0]
                .RelatedBuildingElement.Name,
                "D-W-001-01",
            )

            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)
            ceiling_shape = ifcopenshell.geom.create_shape(settings, ceiling)
            ceiling_vertices = np.asarray(
                ceiling_shape.geometry.verts,
                dtype=float,
            ).reshape(-1, 3)
            self.assertAlmostEqual(float(ceiling_vertices[:, 2].min()), 2.7)
            self.assertAlmostEqual(float(ceiling_vertices[:, 2].max()), 2.73)


if __name__ == "__main__":
    unittest.main()
