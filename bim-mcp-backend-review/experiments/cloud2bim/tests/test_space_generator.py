from cloud2bim_patched.space_generator import (
    convert_to_dictionary,
    identify_zones,
)


def _segment(start, end):
    return {
        'start_point': start,
        'end_point': end,
        'height': 2.8,
        'storey': 1,
    }


def _wall(start, end, thickness=0.2):
    return {
        'start_point': start,
        'end_point': end,
        'thickness': thickness,
        'material': 'Concrete',
        'z_placement': 0.0,
        'height': 2.8,
        'storey': 1,
    }


def test_convert_to_dictionary_orders_a_closed_boundary():
    spaces = {
        'A': [
            _segment((0.0, 0.0), (2.0, 0.0)),
            _segment((2.0, 1.0), (0.0, 1.0)),
            _segment((2.0, 0.0), (2.0, 1.0)),
            _segment((0.0, 1.0), (0.0, 0.0)),
        ]
    }

    result = convert_to_dictionary(spaces)

    assert set(result) == {'A'}
    assert result['A']['vertices'][0] == result['A']['vertices'][-1]
    assert len(result['A']['vertices']) == 5


def test_convert_to_dictionary_rejects_a_missing_wall_edge():
    spaces = {
        'A': [
            _segment((0.0, 0.0), (2.0, 0.0)),
            _segment((2.0, 0.0), (2.0, 1.0)),
            _segment((2.0, 1.0), (0.0, 1.0)),
            # This does not reach the first point: a 40 cm gap is a real gap.
            _segment((0.0, 1.0), (0.0, 0.4)),
        ]
    }

    assert convert_to_dictionary(spaces, connection_tolerance=0.15) == {}


def test_convert_to_dictionary_snaps_only_a_small_joint_error():
    spaces = {
        'A': [
            _segment((0.0, 0.0), (2.0, 0.0)),
            _segment((2.05, 0.0), (2.0, 1.0)),
            _segment((2.0, 1.0), (0.0, 1.0)),
            _segment((0.0, 1.0), (0.0, 0.0)),
        ]
    }

    result = convert_to_dictionary(spaces, connection_tolerance=0.15)

    assert set(result) == {'A'}
    assert result['A']['vertices'][0] == result['A']['vertices'][-1]


def test_identify_zones_uses_only_closed_wall_axis_polygons():
    closed = [
        _wall((0.0, 0.0), (4.0, 0.0)),
        _wall((4.0, 0.0), (4.0, 3.0)),
        _wall((4.0, 3.0), (0.0, 3.0)),
        _wall((0.0, 3.0), (0.0, 0.0)),
    ]
    opened = closed[:-1]

    zones = identify_zones(closed, snapping_distance=0.15, plot_zones=False)

    assert len(zones) == 1
    assert identify_zones(opened, snapping_distance=0.15, plot_zones=False) == {}
    xs = [point[0] for point in next(iter(zones.values()))['vertices']]
    ys = [point[1] for point in next(iter(zones.values()))['vertices']]
    assert round(min(xs), 6) == 0.1
    assert round(max(xs), 6) == 3.9
    assert round(min(ys), 6) == 0.1
    assert round(max(ys), 6) == 2.9
