from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import ezdxf

from plantatobim.export_editor_model_dxf import export_model_to_dxf


class DxfEditorExportTests(unittest.TestCase):
    def test_exports_curved_wall_and_opening_as_curved_polylines(self):
        model = {
            "paredes": [{
                "id": "W-ARC",
                "ax": 0.0,
                "ay": 0.0,
                "bx": 4.0,
                "by": 0.0,
                "espessura": 0.15,
                "altura": 2.8,
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
            }],
            "laje": {"contorno": []},
            "spaces": [],
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "curved.dxf"
            result = export_model_to_dxf(model, output)
            drawing = ezdxf.readfile(output)
            host_axes = [
                entity for entity in drawing.modelspace()
                if entity.dxf.layer == "A-WALL-HOST"
            ]
            window_lines = [
                entity for entity in drawing.modelspace()
                if entity.dxf.layer == "A-WINDOW"
                and entity.dxftype() == "LWPOLYLINE"
            ]

        self.assertEqual(result["walls"], 1)
        self.assertEqual(len(host_axes), 1)
        self.assertGreater(len(host_axes[0]), 20)
        self.assertEqual(len(window_lines), 2)
        self.assertTrue(all(len(entity) > 2 for entity in window_lines))


if __name__ == "__main__":
    unittest.main()
