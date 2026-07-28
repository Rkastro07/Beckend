import unittest

import numpy as np

from bim_authoring.geometry import (
    GeometryRuleError,
    offset_frame,
    validate_hosted_opening,
    wall_frame,
)


class GeometryTests(unittest.TestCase):
    def test_wall_frame_is_metric_and_orthonormal(self):
        matrix, length = wall_frame((1, 2), (4, 6), elevation=3)
        self.assertAlmostEqual(length, 5.0)
        np.testing.assert_allclose(matrix[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(matrix[:3, 0], [0.6, 0.8, 0.0])
        np.testing.assert_allclose(
            matrix[:3, :3].T @ matrix[:3, :3],
            np.eye(3),
            atol=1e-12,
        )

    def test_offset_uses_host_axes(self):
        matrix, _ = wall_frame((0, 0), (0, 10))
        shifted = offset_frame(matrix, along=2, normal=0.5, vertical=1)
        np.testing.assert_allclose(shifted[:3, 3], [-0.5, 2.0, 1.0])

    def test_opening_must_fit_host(self):
        with self.assertRaisesRegex(GeometryRuleError, "fim da parede"):
            validate_hosted_opening(
                host_length=3,
                host_height=2.8,
                offset_from_start=2.5,
                width=1,
                height=1.2,
                sill_height=1,
            )
        with self.assertRaisesRegex(GeometryRuleError, "topo"):
            validate_hosted_opening(
                host_length=3,
                host_height=2.8,
                offset_from_start=0.5,
                width=1,
                height=2,
                sill_height=1,
            )


if __name__ == "__main__":
    unittest.main()
