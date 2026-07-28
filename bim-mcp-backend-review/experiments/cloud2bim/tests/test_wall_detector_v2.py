import math
import sys
import unittest
from pathlib import Path

import numpy as np


PATCHED = Path(__file__).resolve().parents[1] / "cloud2bim_patched"
sys.path.insert(0, str(PATCHED))

from wall_detector_v2 import (  # noqa: E402
    build_multislice_wall_grid,
    build_refinement_context,
    deduplicate_overlapping_wall_axes,
    detect_articulated_leaf_walls,
    face_pair_vertical_metrics,
    keep_non_leaf_wall_indices,
    make_single_line_group,
    merge_collinear_fragments,
    pair_wall_faces,
    plausible_wall_geometry,
    refine_face_pair,
    single_conflicts_with_paired_wall,
    wall_pair_has_vertical_support,
    wall_axis_from_pair,
)


def occupancy_grid(slices=4, pixel=0.1):
    return {
        "slice_masks": np.zeros((slices, 30, 80), dtype=bool),
        "pixel_size": pixel,
        "x_min": -1.0,
        "y_min": -1.0,
    }


class WallDetectorV2Tests(unittest.TestCase):
    @staticmethod
    def panel_points(start, end, z_min, z_max):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        result = []
        for parameter in np.linspace(0.0, 1.0, 31):
            xy = start + parameter * (end - start)
            for z in np.linspace(z_min, z_max, 31):
                result.append([xy[0], xy[1], z])
        return np.asarray(result)

    def test_open_door_leaf_is_suggested_for_suppression(self):
        host = [[0.0, 0.0], [5.0, 0.0]]
        leaf = [[3.0, 0.0], [3.5, 0.866]]
        points = self.panel_points(leaf[0], leaf[1], 0.02, 2.10)
        results = detect_articulated_leaf_walls(
            axes=[host, leaf],
            thicknesses=[0.20, 0.05],
            wall_ids=["W-HOST", "W-LEAF"],
            opening_anchors=[{
                "id": "D-01",
                "host_wall": "W-HOST",
                "type": "door",
                "width": 1.0,
                "center": [2.5, 0.0],
                "host_axis": host,
            }],
            points=points,
            z_floor=0.0,
            z_ceiling=3.0,
        )

        self.assertEqual([result["wall_id"] for result in results], ["W-LEAF"])
        self.assertTrue(results[0]["vertical_match"])
        self.assertTrue(results[0]["suppress"])
        self.assertGreater(results[0]["open_angle_deg"], 50.0)
        self.assertEqual(keep_non_leaf_wall_indices(2, results), [0])

    def test_full_height_partition_at_jamb_is_not_suppressed(self):
        host = [[0.0, 0.0], [5.0, 0.0]]
        partition = [[3.0, 0.0], [3.0, 1.0]]
        points = self.panel_points(partition[0], partition[1], 0.0, 3.0)
        results = detect_articulated_leaf_walls(
            axes=[host, partition],
            thicknesses=[0.20, 0.10],
            wall_ids=["W-HOST", "W-PARTITION"],
            opening_anchors=[{
                "id": "D-01",
                "host_wall": "W-HOST",
                "type": "door",
                "width": 1.0,
                "center": [2.5, 0.0],
                "host_axis": host,
            }],
            points=points,
            z_floor=0.0,
            z_ceiling=3.0,
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["vertical_match"])
        self.assertFalse(results[0]["suppress"])

    def test_panel_away_from_opening_is_not_a_leaf_candidate(self):
        host = [[0.0, 0.0], [5.0, 0.0]]
        panel = [[4.5, 1.0], [5.0, 1.866]]
        points = self.panel_points(panel[0], panel[1], 0.0, 2.1)
        results = detect_articulated_leaf_walls(
            axes=[host, panel],
            thicknesses=[0.20, 0.05],
            wall_ids=["W-HOST", "W-PANEL"],
            opening_anchors=[{
                "id": "D-01",
                "host_wall": "W-HOST",
                "type": "door",
                "width": 1.0,
                "center": [2.5, 0.0],
                "host_axis": host,
            }],
            points=points,
            z_floor=0.0,
            z_ceiling=3.0,
        )

        self.assertEqual(results, [])

    def test_single_line_parallel_to_double_line_is_excluded(self):
        paired = [[[[0.0, 0.0], [5.0, 0.0]],
                   [[0.0, 0.20], [5.0, 0.20]]]]
        self.assertTrue(single_conflicts_with_paired_wall(
            [[0.5, 0.35], [4.5, 0.35]], paired,
            single_thickness=0.15, pixel_size=0.10))

    def test_distant_or_perpendicular_single_line_is_not_excluded(self):
        paired = [[[[0.0, 0.0], [5.0, 0.0]],
                   [[0.0, 0.20], [5.0, 0.20]]]]
        self.assertFalse(single_conflicts_with_paired_wall(
            [[0.5, 1.20], [4.5, 1.20]], paired,
            single_thickness=0.15, pixel_size=0.10))
        self.assertFalse(single_conflicts_with_paired_wall(
            [[2.5, -1.0], [2.5, 1.0]], paired,
            single_thickness=0.15, pixel_size=0.10))
        self.assertFalse(single_conflicts_with_paired_wall(
            [[6.0, 0.25], [8.0, 0.25]], paired,
            single_thickness=0.15, pixel_size=0.10))

    def test_compact_two_face_object_is_not_a_wall(self):
        self.assertFalse(plausible_wall_geometry(
            [[0.0, 0.0], [0.55, 0.0]], 0.61, 0.30, 2.5))
        self.assertTrue(plausible_wall_geometry(
            [[0.0, 0.0], [5.00, 0.0]], 0.61, 0.30, 2.5))

    def test_vertical_persistence_rejects_single_slice_clutter(self):
        points = []
        for slice_index in range(6):
            z = (slice_index + 0.5) / 6.0
            for x in np.linspace(0.0, 1.0, 21):
                points.append([x, 0.0, z])       # persistent wall
        for x in np.linspace(0.0, 1.0, 21):
            points.append([x, 0.8, 0.08])        # low, one-slice clutter

        _density, binary, _xe, _ye, grid, required = (
            build_multislice_wall_grid(points, 0.0, 1.0, 0.1, 6, 0.30))
        wall_y = int(math.floor((0.0 - grid["y_min"]) / 0.1))
        clutter_y = min(binary.shape[0] - 1, int(math.floor(
            (0.8 - grid["y_min"]) / 0.1)))

        self.assertEqual(required, 2)
        self.assertGreater(np.count_nonzero(binary[wall_y]), 0)
        self.assertEqual(np.count_nonzero(binary[clutter_y]), 0)

    def test_collinear_gap_is_joined_when_a_height_slice_supports_it(self):
        grid = occupancy_grid()
        # Gap x=2..3, y=0.  One height band sees a lintel/sill there.
        grid["slice_masks"][2, 9:12, 29:42] = True
        segments = [[[0.0, 0.0], [2.0, 0.0]],
                    [[3.0, 0.0], [5.0, 0.0]]]

        merged = merge_collinear_fragments(
            segments, grid, max_gap=1.2, max_unseen_gap=0.2,
            minimum_slice_support=0.5)

        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(np.linalg.norm(
            np.asarray(merged[0][1]) - np.asarray(merged[0][0])), 5.0, places=6)

    def test_collinear_gap_stays_split_without_point_support(self):
        segments = [[[0.0, 0.0], [2.0, 0.0]],
                    [[3.0, 0.0], [5.0, 0.0]]]

        merged = merge_collinear_fragments(
            segments, occupancy_grid(), max_gap=1.2, max_unseen_gap=0.2,
            minimum_slice_support=0.1)

        self.assertEqual(len(merged), 2)

    def test_face_pairing_is_strictly_one_to_one(self):
        faces = [
            [[0.0, 0.00], [5.0, 0.00]],
            [[0.0, 0.15], [5.0, 0.15]],
            [[0.0, 0.50], [5.0, 0.50]],
        ]

        pairs, leftovers, diagnostics = pair_wall_faces(
            faces, 0.10, 0.40, minimum_overlap=0.2)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(pairs[0]), 2)
        self.assertEqual(len(leftovers), 1)
        self.assertAlmostEqual(diagnostics[0]["thickness"], 0.15, places=6)

    def test_original_points_refine_angle_and_thickness(self):
        rng = np.random.default_rng(7)
        actual_angle = math.radians(30.0)
        actual_u = np.array([math.cos(actual_angle), math.sin(actual_angle)])
        actual_n = np.array([-actual_u[1], actual_u[0]])
        t = np.linspace(-2.5, 2.5, 400)
        xyz = []
        for rho in (-0.10, 0.10):
            xy = (t[:, None] * actual_u + rho * actual_n +
                  rng.normal(0.0, 0.003, (len(t), 1)) * actual_n)
            z = np.linspace(0.1, 2.9, len(t))[:, None]
            xyz.append(np.column_stack([xy, z]))
        points = np.vstack(xyz)

        proposed_angle = math.radians(32.0)
        proposed_u = np.array([math.cos(proposed_angle), math.sin(proposed_angle)])
        proposed_n = np.array([-proposed_u[1], proposed_u[0]])
        pair = []
        for rho in (-0.11, 0.11):
            pair.append([(-2.5 * proposed_u + rho * proposed_n).tolist(),
                         (2.5 * proposed_u + rho * proposed_n).tolist()])

        context = build_refinement_context(points, 0.0, 3.0)
        refined = refine_face_pair(pair, context, 0.04, 0.08, 0.40)
        axis, thickness = wall_axis_from_pair(refined)
        vector = np.asarray(axis[1]) - np.asarray(axis[0])
        measured_angle = math.degrees(math.atan2(vector[1], vector[0])) % 180.0

        self.assertAlmostEqual(measured_angle, 30.0, delta=0.35)
        self.assertAlmostEqual(thickness, 0.20, delta=0.015)

    def test_single_face_expands_toward_locally_empty_side(self):
        grid = occupancy_grid()
        # The observed segment is y=0.  Occupancy on y=+0.15 means the hidden
        # face must be synthesized on y=-0.15.
        grid["slice_masks"][:, 11:14, 9:62] = True
        group = make_single_line_group(
            [[0.0, 0.0], [5.0, 0.0]], 0.15, grid, centroid=(2.5, 1.0))

        synthetic_y = np.mean(np.asarray(group[1])[:, 1])
        self.assertAlmostEqual(synthetic_y, -0.15, places=6)

    def test_face_profile_keeps_wall_with_localised_opening(self):
        grid = occupancy_grid(slices=6)
        # One observed face persists over the full height. A door-sized gap is
        # local along X in the lower four layers; the lintel remains above it.
        face_y = int(math.floor((0.0 - grid["y_min"]) / 0.1))
        grid["slice_masks"][:, face_y, 10:61] = True
        grid["slice_masks"][:4, face_y, 30:40] = False
        pair = [
            [[0.0, 0.0], [5.0, 0.0]],
            [[0.0, 0.2], [5.0, 0.2]],
        ]

        accepted, metrics = wall_pair_has_vertical_support(
            pair, grid, corridor_pixels=0, persistent_slices=4,
            minimum_bottom_coverage=0.12,
            minimum_top_coverage=0.15,
            minimum_persistent_coverage=0.10)

        self.assertTrue(accepted)
        self.assertGreater(metrics["top_coverage"], 0.90)
        self.assertGreater(metrics["persistent_coverage"], 0.70)

    def test_face_profile_rejects_lower_three_layer_furniture(self):
        grid = occupancy_grid(slices=6)
        face0_y = int(math.floor((0.0 - grid["y_min"]) / 0.1))
        face1_y = int(math.floor((0.2 - grid["y_min"]) / 0.1))
        grid["slice_masks"][:3, face0_y, 10:61] = True
        grid["slice_masks"][:3, face1_y, 10:61] = True
        pair = [
            [[0.0, 0.0], [5.0, 0.0]],
            [[0.0, 0.2], [5.0, 0.2]],
        ]

        accepted, metrics = wall_pair_has_vertical_support(
            pair, grid, corridor_pixels=0, persistent_slices=4)

        self.assertFalse(accepted)
        self.assertEqual(metrics["top_coverage"], 0.0)
        self.assertEqual(metrics["persistent_coverage"], 0.0)

    def test_each_face_is_measured_without_borrowing_broad_axis_support(self):
        grid = occupancy_grid(slices=6)
        real_wall_y = int(math.floor((0.0 - grid["y_min"]) / 0.1))
        grid["slice_masks"][:, real_wall_y, 10:61] = True
        false_pair = [
            [[0.0, 0.4], [5.0, 0.4]],
            [[0.0, 1.0], [5.0, 1.0]],
        ]

        accepted, metrics = wall_pair_has_vertical_support(
            false_pair, grid, corridor_pixels=0)

        self.assertFalse(accepted)
        self.assertLess(metrics["score"], 0.05)
        self.assertEqual(
            face_pair_vertical_metrics(
                false_pair, grid, corridor_pixels=0)["accepted_face"],
            0,
        )

    def test_overlapping_parallel_wall_axes_are_deduplicated_by_quality(self):
        axes = [
            [[0.0, 0.0], [8.0, 0.0]],   # correct wall
            [[0.0, 0.4], [8.0, 0.4]],   # overlapping false parallel
            [[0.0, 2.0], [8.0, 2.0]],   # separate parallel wall
        ]
        thicknesses = [0.50, 0.67, 0.20]

        kept = deduplicate_overlapping_wall_axes(
            axes, thicknesses, quality_scores=[0.90, 0.20, 0.80])

        self.assertEqual(kept, [0, 2])


if __name__ == "__main__":
    unittest.main()
