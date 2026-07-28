import sys
from pathlib import Path

import numpy as np

PATCHED=Path(__file__).resolve().parents[1]/"cloud2bim_patched"
sys.path.insert(0,str(PATCHED))

from opening_detector_v2 import detect_topology_openings, detect_wall_openings  # noqa: E402


def synthetic_wall():
    points=[]
    for x in np.arange(0.0,8.01,.04):
        for z in np.arange(0.0,3.01,.04):
            door=1.0<=x<=2.0 and z<=2.12
            window=4.0<=x<=5.2 and 1.0<=z<=2.2
            if door or window:continue
            points.append((x,-.10,z)); points.append((x,.10,z))
    return np.asarray(points)


def test_rectified_detector_separates_door_and_window():
    result=detect_wall_openings(synthetic_wall(),wall_id="W-TEST-001",start=(0.,0.),end=(8.,0.),thickness=.20,floor_z=0.,ceiling_z=3.,grid_cell=.05)
    kinds=[candidate.type for candidate in result.candidates]
    assert "door" in kinds
    assert "window" in kinds
    door=next(candidate for candidate in result.candidates if candidate.type=="door")
    window=next(candidate for candidate in result.candidates if candidate.type=="window")
    assert abs(door.s_center-1.5)<.25
    assert abs(window.s_center-4.6)<.30
    assert door.evidence["touches_floor"] is True
    assert window.evidence["touches_floor"] is False


def synthetic_framed_windows():
    generator = np.random.default_rng(42)
    points = []
    for face in (-.10, .10):
        for x, z in zip(
                generator.uniform(0.0, 8.0, 16000),
                generator.uniform(0.0, 3.0, 16000)):
            points.append((x, face, z))
    for center in (2.0, 5.2):
        for frame_x in (center - .50, center, center + .50):
            for z in np.arange(.95, 2.81, .02):
                points.append((frame_x, -.10, z))
                points.append((frame_x, .10, z))
        for frame_z in (.95, 2.80):
            for x in np.arange(center - .50, center + .51, .02):
                points.append((x, -.10, frame_z))
                points.append((x, .10, frame_z))
    return np.asarray(points)


def test_repeated_frame_family_recovers_windows_with_background_returns():
    result = detect_wall_openings(
        synthetic_framed_windows(),
        wall_id="W-TEST-FRAMES",
        start=(0., 0.),
        end=(8., 0.),
        thickness=.20,
        floor_z=0.,
        ceiling_z=3.,
        grid_cell=.05,
    )
    windows = [
        candidate
        for candidate in result.candidates
        if candidate.type == "window" and candidate.status == "proposed"
    ]
    assert len(windows) == 2
    assert abs(windows[0].s_center - 2.0) < .20
    assert abs(windows[1].s_center - 5.2) < .20
    assert all(
        candidate.evidence.get("detector_mode") == "repeated_frame_family"
        for candidate in windows
    )


def test_topology_detector_finds_door_sized_wall_extension():
    walls = {
        "W-A": (np.asarray((0., 0.)), np.asarray((1., 0.))),
        "W-B": (np.asarray((2., -1.)), np.asarray((2., 1.))),
    }
    candidates = detect_topology_openings(walls)
    assert len(candidates) == 1
    assert candidates[0].host_wall == "W-A"
    assert candidates[0].between == ("W-A", "W-B")
    assert abs(candidates[0].width - 1.0) < .01


if __name__ == "__main__":
    test_rectified_detector_separates_door_and_window()
    test_repeated_frame_family_recovers_windows_with_background_returns()
    test_topology_detector_finds_door_sized_wall_extension()
    print("opening_detector_v2: 3 testes aprovados")
