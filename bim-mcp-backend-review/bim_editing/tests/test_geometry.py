from __future__ import annotations

import unittest

from bim_editing.geometry import planar_faces, polygon_area


class PlanarTopologyTests(unittest.TestCase):
    def test_t_junction_splits_edge_without_creating_false_space(self):
        walls = [
            {"id": "A", "ax": 0, "ay": 0, "bx": 4, "by": 0},
            {"id": "B", "ax": 2, "ay": 0, "bx": 2, "by": 2},
        ]
        faces, topology = planar_faces(walls)
        self.assertEqual(faces, [])
        self.assertEqual(topology["junction_nodes"][0]["degree"], 3)

    def test_internal_wall_divides_closed_enclosure(self):
        walls = [
            {"id": "A", "ax": 0, "ay": 0, "bx": 4, "by": 0},
            {"id": "B", "ax": 4, "ay": 0, "bx": 4, "by": 3},
            {"id": "C", "ax": 4, "ay": 3, "bx": 0, "by": 3},
            {"id": "D", "ax": 0, "ay": 3, "bx": 0, "by": 0},
            {"id": "E", "ax": 2, "ay": 0, "bx": 2, "by": 3},
        ]
        faces, topology = planar_faces(walls)
        self.assertEqual(topology["closed_face_count"], 2)
        self.assertEqual(sorted(round(polygon_area(face), 6) for face in faces), [6.0, 6.0])


if __name__ == "__main__":
    unittest.main()
