from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from bim_editing.adapters import model_from_cloud2bim


class CloudAdapterVerticalTests(unittest.TestCase):
    def test_vertical_levels_define_wall_slab_and_ceiling_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnostics = root / "wall_diagnostics.csv"
            with diagnostics.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "reference",
                        "start_x",
                        "start_y",
                        "end_x",
                        "end_y",
                        "thickness",
                        "storey",
                        "detector",
                        "confidence",
                        "review_status",
                        "evidence_type",
                        "height_band_min",
                        "height_band_max",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "reference": "W-S01-001",
                    "start_x": 0,
                    "start_y": 0,
                    "end_x": 4,
                    "end_y": 0,
                    "thickness": 0.15,
                    "storey": "S01",
                    "detector": "v2",
                    "confidence": "HIGH",
                    "review_status": "AUTO_ACCEPTED",
                    "evidence_type": "interior",
                    "height_band_min": 0.46,
                    "height_band_max": 3.53,
                })

            openings = root / "openings.json"
            openings.write_text(json.dumps({
                "walls": [],
                "topology_candidates": [],
            }), encoding="utf-8")
            vertical = root / "vertical_levels.json"
            vertical.write_text(json.dumps({
                "schema": "cloud2bim.vertical-levels.v1",
                "storeys": [{
                    "id": "S01",
                    "floor": {
                        "bottom_z": -0.122,
                        "top_z": 0.178,
                        "thickness": 0.3,
                    },
                    "structural_ceiling": {
                        "bottom_z": 3.815,
                        "top_z": 4.324,
                        "thickness": 0.509,
                    },
                    "wall_height_from_floor": 3.637,
                    "suspended_ceiling": {
                        "status": "detected",
                        "height_from_floor": 3.0135,
                        "thickness": 0.03,
                        "confidence": "HIGH",
                        "detector": "regional_horizontal_normals_v1",
                    },
                }],
            }), encoding="utf-8")

            model = model_from_cloud2bim(
                diagnostics,
                openings,
                vertical,
            )

        wall = model["paredes"][0]
        self.assertAlmostEqual(wall["altura"], 3.637)
        self.assertNotIn("altura_observada", wall)
        self.assertEqual(
            wall["faixa_vertical_evidencia"]["uso"],
            "wall_detection_support_only",
        )
        self.assertAlmostEqual(
            model["laje"]["piso"]["espessura"],
            0.3,
        )
        self.assertAlmostEqual(
            model["laje"]["teto"]["espessura"],
            0.509,
        )
        self.assertAlmostEqual(model["ifc_config"]["altura"], 3.637)
        self.assertAlmostEqual(
            model["ifc_config"]["forro"]["altura"],
            3.0135,
        )
        self.assertTrue(model["diagnostico"]["forro_detectado"])


if __name__ == "__main__":
    unittest.main()
