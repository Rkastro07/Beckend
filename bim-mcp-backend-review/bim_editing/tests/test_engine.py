from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from bim_editing.engine import RevisionEngine
from bim_editing.model import normalize_model
from bim_editing.render import render_model
from bim_editing.cloud_review import build_cloud_review, PRIMARY_REVIEW_NAME


def wall(identifier, p1, p2, thickness=0.15):
    return {
        "id": identifier,
        "ax": p1[0],
        "ay": p1[1],
        "bx": p2[0],
        "by": p2[1],
        "espessura": thickness,
        "layer": "Wall-Test",
    }


def model(walls, openings=None):
    return {
        "revision": "R00",
        "paredes": walls,
        "aberturas": openings or [],
        "laje": {
            "contorno": [],
            "piso": {"ativo": True, "espessura": 0.12},
            "teto": {"ativo": True, "espessura": 0.12},
        },
    }


class RevisionEngineTests(unittest.TestCase):
    def test_curved_wall_keeps_arc_length_and_opening_during_revision(self):
        curved = wall("W-ARC", (0, 0), (4, 0))
        curved.update({
            "geometria": "arco",
            "curva": {"x": 2, "y": 2},
        })
        payload = model(
            [curved],
            [{
                "id": "J-ARC",
                "parede_id": "W-ARC",
                "tipo": "window",
                "s_centro": 4.5,
                "largura": 1.0,
            }],
        )

        revised, report = RevisionEngine(payload).apply({
            "operations": [],
            "recalculate": ["openings", "validation"],
        })

        self.assertAlmostEqual(revised["paredes"][0]["comprimento"], 2 * 3.14159265, places=5)
        self.assertAlmostEqual(revised["aberturas"][0]["s_centro"], 4.5)
        self.assertTrue(report["validation"]["valid"])

    def test_initial_endpoint_order_remaps_opening_offset(self):
        payload = model(
            [wall("W-001", (4, 0), (0, 0))],
            [{
                "id": "D-001",
                "parede_id": "W-001",
                "tipo": "door",
                "s_centro": 1.0,
                "largura": 0.8,
            }],
        )
        result = normalize_model(payload)
        current = result["paredes"][0]
        self.assertEqual((current["ax"], current["ay"]), (0.0, 0.0))
        self.assertEqual(current["parts"]["P1"]["selector"], "W-001.P1")
        self.assertAlmostEqual(result["aberturas"][0]["s_centro"], 3.0)

    def test_move_endpoint_keeps_p1_identity(self):
        payload = model([wall("W-001", (0, 0), (4, 0))])
        revised, report = RevisionEngine(payload).apply({
            "operations": [{
                "op": "move_wall_endpoint",
                "selector": "W-001.P1",
                "target": [0, -1],
            }],
        })
        current = revised["paredes"][0]
        self.assertEqual(
            (current["parts"]["P1"]["x"], current["parts"]["P1"]["y"]),
            (0.0, -1.0),
        )
        self.assertTrue(report["validation"]["valid"])

    def test_add_wall_from_p1_perpendicular_until_target(self):
        payload = model([
            wall("W-001", (0, 0), (4, 0)),
            wall("W-002", (0, 3), (4, 3)),
        ])
        revised, _ = RevisionEngine(payload).apply({
            "operations": [{
                "op": "add_wall",
                "id": "W-003",
                "from": "W-001.P1",
                "direction": {"perpendicular_to": "W-001"},
                "until": "W-002",
                "thickness": 0.20,
            }],
        })
        added = next(wall for wall in revised["paredes"] if wall["id"] == "W-003")
        self.assertAlmostEqual(added["ax"], 0.0)
        self.assertAlmostEqual(added["ay"], 0.0)
        self.assertAlmostEqual(added["bx"], 0.0)
        self.assertAlmostEqual(added["by"], 3.0)
        self.assertAlmostEqual(added["espessura"], 0.20)

    def test_connect_endpoint_to_axis_intersection_preserves_wall_axis(self):
        payload = model([
            wall("W-001", (0, 0), (4, 0)),
            wall("W-002", (5, -2), (5, 2)),
        ])
        revised, _ = RevisionEngine(payload).apply({
            "operations": [{
                "op": "connect_endpoint",
                "selector": "W-001.P2",
                "target": {
                    "element": "W-002",
                    "mode": "axis_intersection",
                },
            }],
        })
        current = next(wall for wall in revised["paredes"] if wall["id"] == "W-001")
        self.assertAlmostEqual(current["bx"], 5.0)
        self.assertAlmostEqual(current["by"], 0.0)

    def test_merge_walls_uses_full_physical_thickness(self):
        payload = model([
            wall("W-001", (0, 0), (4, 0), 0.10),
            wall("W-002", (0, 0.20), (4, 0.20), 0.10),
        ])
        revised, _ = RevisionEngine(payload).apply({
            "operations": [{
                "op": "merge_walls",
                "ids": ["W-001", "W-002"],
                "target_id": "W-MERGED",
            }],
        })
        self.assertEqual(len(revised["paredes"]), 1)
        self.assertAlmostEqual(revised["paredes"][0]["espessura"], 0.30)

    def test_delete_wall_also_deletes_hosted_opening(self):
        payload = model(
            [
                wall("W-001", (0, 0), (4, 0)),
                wall("W-002", (0, 2), (4, 2)),
            ],
            [{
                "id": "D-001",
                "parede_id": "W-002",
                "tipo": "door",
                "s_centro": 2.0,
                "largura": 0.8,
            }],
        )
        revised, _ = RevisionEngine(payload).apply({
            "operations": [{"op": "delete_elements", "ids": ["W-002"]}],
        })
        self.assertEqual([wall["id"] for wall in revised["paredes"]], ["W-001"])
        self.assertEqual(revised["aberturas"], [])

    def test_square_recalculates_space_and_slab(self):
        payload = model([
            wall("W-001", (0, 0), (4, 0)),
            wall("W-002", (4, 0), (4, 3)),
            wall("W-003", (4, 3), (0, 3)),
            wall("W-004", (0, 3), (0, 0)),
        ])
        revised, report = RevisionEngine(payload).apply({"operations": []})
        self.assertEqual(len(revised["spaces"]), 1)
        self.assertAlmostEqual(revised["spaces"][0]["area"], 12.0)
        self.assertGreaterEqual(len(revised["laje"]["contorno"]), 4)
        self.assertTrue(report["validation"]["valid"])

    def test_wall_outside_slab_can_be_advisory(self):
        payload = model([wall("W-OUTSIDE", (0, 0), (4, 0), 0.20)])
        payload["laje"]["contorno"] = [
            [-0.20, -0.20],
            [2.00, -0.20],
            [2.00, 0.20],
            [-0.20, 0.20],
        ]

        revised, report = RevisionEngine(payload).apply({
            "operations": [],
            "recalculate": ["validation"],
            "policies": {"wall_outside_slab": "warning"},
        })

        self.assertEqual(revised["laje"]["contorno"], payload["laje"]["contorno"])
        self.assertTrue(report["validation"]["valid"])
        self.assertEqual(report["validation"]["errors"], [])
        self.assertIn(
            "W-OUTSIDE: face externa fora do slab",
            report["validation"]["warnings"],
        )

    def test_opening_is_clamped_after_wall_is_shortened(self):
        payload = model(
            [wall("W-001", (0, 0), (4, 0))],
            [{
                "id": "D-001",
                "parede_id": "W-001",
                "tipo": "door",
                "s_centro": 3.4,
                "largura": 1.0,
            }],
        )
        revised, report = RevisionEngine(payload).apply({
            "operations": [{
                "op": "move_wall_endpoint",
                "selector": "W-001.P2",
                "target": [3, 0],
            }],
        })
        self.assertAlmostEqual(revised["aberturas"][0]["s_centro"], 2.5)
        self.assertTrue(any("ajustada" in warning for warning in report["warnings"]))

    def test_opening_is_clamped_to_observed_wall_height(self):
        host = wall("W-001", (0, 0), (4, 0))
        host["altura_observada"] = 3.0
        payload = model(
            [host],
            [{
                "id": "J-001",
                "parede_id": "W-001",
                "tipo": "window",
                "s_centro": 2.0,
                "largura": 1.0,
                "peitoril": 2.2,
                "altura": 1.2,
            }],
        )
        revised, report = RevisionEngine(payload).apply({"operations": []})
        opening = revised["aberturas"][0]
        self.assertAlmostEqual(opening["peitoril"], 2.2)
        self.assertAlmostEqual(opening["altura"], 0.8)
        self.assertAlmostEqual(opening["vertical_detectado"]["topo"], 3.4)
        self.assertTrue(any(
            "ajustada verticalmente" in warning
            for warning in report["warnings"]
        ))

    def test_window_to_door_preserves_observed_head_height(self):
        payload = model(
            [wall("W-001", (0, 0), (4, 0))],
            [{
                "id": "J-001",
                "parede_id": "W-001",
                "tipo": "window",
                "s_centro": 2.0,
                "largura": 0.8,
                "altura": 1.20,
                "peitoril": 0.90,
            }],
        )
        revised, report = RevisionEngine(payload).apply({
            "operations": [{
                "op": "set_opening_type",
                "id": "J-001",
                "new_id": "D-001",
                "type": "door",
            }],
        })
        opening = revised["aberturas"][0]
        self.assertEqual(opening["id"], "D-001")
        self.assertEqual(opening["tipo"], "door")
        self.assertAlmostEqual(opening["altura"], 2.10)
        self.assertAlmostEqual(opening["peitoril"], 0.0)
        self.assertEqual(opening["renamed_from"], "J-001")
        self.assertTrue(report["validation"]["valid"])

    def test_copy_opening_pattern_projects_between_parallel_walls(self):
        payload = model(
            [
                wall("W-SOURCE", (0, 0), (10, 0)),
                wall("W-TARGET", (0, 3), (10, 3)),
            ],
            [
                {
                    "id": "D-SOURCE-01",
                    "parede_id": "W-SOURCE",
                    "tipo": "door",
                    "s_centro": 2.0,
                    "largura": 0.8,
                    "altura": 2.1,
                    "peitoril": 0.0,
                },
                {
                    "id": "D-SOURCE-02",
                    "parede_id": "W-SOURCE",
                    "tipo": "door",
                    "s_centro": 7.0,
                    "largura": 1.0,
                    "altura": 2.1,
                    "peitoril": 0.0,
                },
            ],
        )
        revised, _ = RevisionEngine(payload).apply({
            "operations": [{
                "op": "copy_opening_pattern",
                "source_wall_id": "W-SOURCE",
                "target_wall_id": "W-TARGET",
                "id_prefix": "D-TARGET",
            }],
        })
        copied = [
            opening
            for opening in revised["aberturas"]
            if opening["parede_id"] == "W-TARGET"
        ]
        self.assertEqual([opening["id"] for opening in copied], [
            "D-TARGET-01",
            "D-TARGET-02",
        ])
        self.assertEqual([opening["s_centro"] for opening in copied], [2.0, 7.0])

    def test_close_wall_junctions_connects_t_without_rotating_axis(self):
        payload = model([
            wall("W-HOST", (0, 0), (6, 0)),
            wall("W-BRANCH", (3, 0.25), (3, 3)),
        ])
        revised, report = RevisionEngine(payload).apply({
            "operations": [{
                "op": "close_wall_junctions",
                "max_distance": 0.30,
            }],
        })
        branch = next(
            wall for wall in revised["paredes"] if wall["id"] == "W-BRANCH"
        )
        self.assertAlmostEqual(branch["ax"], 3.0)
        self.assertAlmostEqual(branch["ay"], 0.0)
        result = report["operation_results"][0]
        self.assertEqual(
            result["moved_endpoints"]["W-BRANCH.P1"]["kind"],
            "T",
        )

    def test_close_wall_junctions_protects_target_opening(self):
        payload = model(
            [
                wall("W-HOST", (0, 0), (6, 0)),
                wall("W-BRANCH", (3, 0.25), (3, 3)),
            ],
            [{
                "id": "D-HOST-01",
                "parede_id": "W-HOST",
                "tipo": "door",
                "s_centro": 3.0,
                "largura": 0.9,
            }],
        )
        revised, report = RevisionEngine(payload).apply({
            "operations": [{
                "op": "close_wall_junctions",
                "max_distance": 0.30,
            }],
        })
        branch = next(
            wall for wall in revised["paredes"] if wall["id"] == "W-BRANCH"
        )
        self.assertAlmostEqual(branch["ay"], 0.25)
        result = report["operation_results"][0]
        self.assertEqual(
            result["blocked_by_openings"][0]["opening_id"],
            "D-HOST-01",
        )

    def test_edit_renderer_labels_only_selected_endpoints(self):
        payload = normalize_model(model([
            wall("W-001", (0, 0), (4, 0)),
            wall("W-002", (0, 2), (4, 2)),
        ]))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "edit.png"
            render_model(
                payload,
                output,
                mode="edit",
                selected=["W-001"],
                width=1000,
                height=700,
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 10_000)

    def test_cloud_review_writes_primary_approval_png(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnostics = root / "wall_diagnostics.csv"
            diagnostics.write_text(
                "reference,start_x,start_y,end_x,end_y,thickness,storey,"
                "detector,confidence,review_status,evidence_type\n"
                "W-S01-001,0,0,4,0,0.2,1,v2,HIGH,AUTO_ACCEPTED,interior\n",
                encoding="utf-8",
            )
            openings = root / "opening_candidates_v2.json"
            openings.write_text(
                '{"walls":[{"wall_id":"W-S01-001","candidates":[]}]}',
                encoding="utf-8",
            )
            result = build_cloud_review(diagnostics, openings, root / "output")
            self.assertEqual(result["primary_png"].name, PRIMARY_REVIEW_NAME)
            self.assertTrue(result["primary_png"].exists())
            self.assertTrue(result["status"].exists())


if __name__ == "__main__":
    unittest.main()
