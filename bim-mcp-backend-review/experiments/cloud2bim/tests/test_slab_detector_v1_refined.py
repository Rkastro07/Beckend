import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon


PATCHED = Path(__file__).resolve().parents[1] / "cloud2bim_patched"
sys.path.insert(0, str(PATCHED))

from aux_functions import refine_slab_contour_v1  # noqa: E402


def test_v1_refined_repairs_zero_width_connections_as_multiple_components():
    # Two real regions encoded by the V1 contour as one ring connected by a
    # retraced, zero-width path. Such a ring is invalid as a single IFC profile.
    contour = np.array(
        [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
            (6.0, 1.0),
            (8.0, 1.0),
            (8.0, 3.0),
            (6.0, 3.0),
            (6.0, 1.0),
            (0.0, 0.0),
        ]
    )

    components = refine_slab_contour_v1(
        contour,
        pixel_size=0.05,
        simplify_tolerance=0.05,
        minimum_component_area=0.5,
    )

    assert len(components) == 2
    assert all(Polygon(component).is_valid for component in components)
    assert sorted(round(Polygon(component).area, 2) for component in components) == [
        4.0,
        16.0,
    ]


def test_v1_refined_simplifies_raster_noise_with_bounded_displacement():
    contour = np.array(
        [
            (0.00, 0.00),
            (1.00, 0.03),
            (2.00, -0.02),
            (3.00, 0.02),
            (4.00, 0.00),
            (4.03, 1.00),
            (3.98, 2.00),
            (4.00, 3.00),
            (3.00, 2.98),
            (2.00, 3.03),
            (1.00, 2.98),
            (0.00, 3.00),
            (-0.03, 2.00),
            (0.02, 1.00),
        ]
    )
    raw = Polygon(contour)

    components = refine_slab_contour_v1(
        contour,
        pixel_size=0.05,
        simplify_tolerance=0.12,
        minimum_component_area=0.5,
    )

    assert len(components) == 1
    refined = Polygon(components[0])
    assert refined.is_valid
    assert len(components[0]) < len(contour)
    assert raw.hausdorff_distance(refined) <= 0.15
    assert abs(refined.area - raw.area) / raw.area < 0.03


def test_v1_refined_discards_only_subthreshold_islands():
    contour = np.array(
        [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
            (6.0, 1.0),
            (6.2, 1.0),
            (6.2, 1.2),
            (6.0, 1.2),
            (6.0, 1.0),
            (0.0, 0.0),
        ]
    )

    components = refine_slab_contour_v1(
        contour,
        pixel_size=0.05,
        simplify_tolerance=0.05,
        minimum_component_area=0.5,
    )

    assert len(components) == 1
    assert round(Polygon(components[0]).area, 2) == 16.0
