import sys
from pathlib import Path

import ezdxf
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_detection_v2 import pair_wall_faces_v2, parse_dxf_v2
from cad_object_grammar_v3 import (
    GRAMMAR_VERSION,
    classify_architectural_text,
    dimensions_from_text,
)


def _new_doc(units=6):
    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = units
    return doc


def _add_double_wall(msp, layer, y=0.0, start=0.0, end=5.0, thickness=0.20):
    msp.add_line((start, y), (end, y), dxfattribs={"layer": layer})
    msp.add_line(
        (start, y + thickness),
        (end, y + thickness),
        dxfattribs={"layer": layer},
    )


def _add_room(msp, layer, x0):
    _add_double_wall(msp, layer, y=0.0, start=x0, end=x0 + 5.0)
    _add_double_wall(msp, layer, y=3.0, start=x0, end=x0 + 5.0)
    msp.add_line((x0, 0), (x0, 3.2), dxfattribs={"layer": layer})
    msp.add_line((x0 + 0.2, 0), (x0 + 0.2, 3.2), dxfattribs={"layer": layer})
    msp.add_line((x0 + 5.0, 0), (x0 + 5.0, 3.2), dxfattribs={"layer": layer})
    msp.add_line((x0 + 5.2, 0), (x0 + 5.2, 3.2), dxfattribs={"layer": layer})


def test_block_names_classify_door_and_window_on_generic_layer(tmp_path):
    doc = _new_doc()
    doc.layers.add("A-WALL")
    modelspace = doc.modelspace()
    _add_double_wall(modelspace, "A-WALL")

    door = doc.blocks.new("PORTA_TIPO_090")
    door.add_line((0, 0), (0.90, 0))
    modelspace.add_blockref("PORTA_TIPO_090", (1.0, 0.10))

    window = doc.blocks.new("JANELA_TIPO_120")
    window.add_line((0, 0), (1.20, 0))
    modelspace.add_blockref("JANELA_TIPO_120", (3.0, 0.10))

    path = tmp_path / "semantic_blocks.dxf"
    doc.saveas(path)
    model = parse_dxf_v2(path)

    assert {opening["tipo"] for opening in model["aberturas"]} == {
        "door", "window",
    }
    assert all(
        opening["origem"] == "cad-block"
        for opening in model["aberturas"]
    )
    assert model["source"]["cad_summary"]["semantic_opening_candidates"] == 2
    assert model["reference"]["kind"] == "vector"
    assert model["reference"]["source_count"] >= 4
    assert len(model["reference"]["segments"]) == model["reference"]["source_count"]


def test_geometry_infers_unknown_wall_layer_and_ignores_dimensions(tmp_path):
    doc = _new_doc()
    doc.layers.add("ARQ_01")
    doc.layers.add("A-DIMS")
    modelspace = doc.modelspace()
    _add_double_wall(modelspace, "ARQ_01", y=0.0)
    _add_double_wall(modelspace, "ARQ_01", y=2.0)
    _add_double_wall(modelspace, "ARQ_01", y=4.0)
    for index in range(8):
        modelspace.add_line(
            (0, 10 + index * 0.03),
            (5, 10 + index * 0.03),
            dxfattribs={"layer": "A-DIMS"},
        )

    path = tmp_path / "inferred_layers.dxf"
    doc.saveas(path)
    model = parse_dxf_v2(path)
    layers = {
        item["name"]: item for item in model["source"]["cad_layers"]
    }

    assert layers["ARQ_01"]["detected_role"] == "wall"
    assert layers["ARQ_01"]["reason"] == "geometry"
    assert layers["A-DIMS"]["detected_role"] == "ignore"
    assert model["source"]["cad_summary"]["inferred_wall_layers"] == 1
    assert len(model["paredes"]) == 3


def test_manual_layer_map_overrides_unknown_convention(tmp_path):
    doc = _new_doc()
    doc.layers.add("X-101")
    modelspace = doc.modelspace()
    _add_double_wall(modelspace, "X-101")
    path = tmp_path / "manual_map.dxf"
    doc.saveas(path)

    model = parse_dxf_v2(path, layer_map={"X-101": "wall"})
    layer = next(
        item for item in model["source"]["cad_layers"]
        if item["name"] == "X-101"
    )
    assert layer["detected_role"] == "wall"
    assert layer["reason"] == "manual"
    assert model["source"]["layer_map"] == {"X-101": "wall"}


def test_incorrect_insunits_falls_back_to_geometry_scale(tmp_path):
    doc = _new_doc(units=4)  # declara mm, mas coordenadas abaixo estao em cm
    doc.layers.add("A-WALL")
    modelspace = doc.modelspace()
    _add_double_wall(
        modelspace,
        "A-WALL",
        start=0.0,
        end=800.0,
        thickness=20.0,
    )
    path = tmp_path / "wrong_units.dxf"
    doc.saveas(path)

    model = parse_dxf_v2(path)
    assert model["escala"] == pytest.approx(0.01)
    assert model["source"]["cad_summary"]["scale_source"] == "geometry-auto"
    assert model["paredes"][0]["espessura"] == pytest.approx(0.20)


def test_gap_on_both_wall_faces_creates_reviewable_opening(tmp_path):
    doc = _new_doc()
    doc.layers.add("A-WALL")
    modelspace = doc.modelspace()
    for y in (0.0, 0.20):
        modelspace.add_line((0, y), (2.0, y), dxfattribs={"layer": "A-WALL"})
        modelspace.add_line((2.9, y), (5.0, y), dxfattribs={"layer": "A-WALL"})
    path = tmp_path / "wall_gap.dxf"
    doc.saveas(path)

    model = parse_dxf_v2(path)
    gap_openings = [
        opening for opening in model["aberturas"]
        if opening.get("origem") == "cad-gap-paired-faces"
    ]
    assert len(gap_openings) == 1
    assert gap_openings[0]["tipo"] == "door"
    assert gap_openings[0]["largura"] == pytest.approx(0.90)


def test_disconnected_floorplans_become_selectable_cad_regions(tmp_path):
    doc = _new_doc()
    doc.layers.add("A-WALL")
    modelspace = doc.modelspace()
    _add_room(modelspace, "A-WALL", 0.0)
    _add_room(modelspace, "A-WALL", 20.0)
    # Ruido isolado nao deve virar uma terceira planta selecionavel.
    _add_double_wall(modelspace, "A-WALL", y=20.0, start=50.0, end=51.0)
    path = tmp_path / "two_floorplans.dxf"
    doc.saveas(path)

    first = parse_dxf_v2(path, cad_region="cad-region-1")
    second = parse_dxf_v2(path, cad_region="cad-region-2")

    assert len(first["source"]["cad_regions"]) == 2
    assert first["source"]["cad_region"]["id"] == "cad-region-1"
    assert second["source"]["cad_region"]["id"] == "cad-region-2"
    assert max(point[0] for wall in first["paredes"] for point in wall["eixo"]) < 10
    assert min(point[0] for wall in second["paredes"] for point in wall["eixo"]) > 10
    assert first["reference"]["bounds"][2] < 10
    assert second["reference"]["bounds"][0] > 10


def test_long_wall_face_can_pair_with_multiple_shorter_faces():
    segments = [
        (np.array([0.0, 0.0]), np.array([6.0, 0.0]), "Wall-Int"),
        (np.array([0.0, 0.05]), np.array([2.0, 0.05]), "Wall-Int"),
        (np.array([3.0, 0.05]), np.array([6.0, 0.05]), "Wall-Int"),
    ]

    walls, leftovers = pair_wall_faces_v2(segments)

    assert len(walls) == 2
    assert sum(wall["comprimento"] for wall in walls) == pytest.approx(5.0)
    assert [wall["espessura"] for wall in walls] == pytest.approx([0.05, 0.05])
    assert len(leftovers) == 1  # trecho de 1 m sem segunda face continua revisavel


def test_pairer_prefers_neighboring_faces_over_false_thick_pair():
    segments = [
        (np.array([0.0, 0.0]), np.array([4.0, 0.0]), "Wall-Int"),
        (np.array([0.0, 0.05]), np.array([4.0, 0.05]), "Wall-Int"),
        (np.array([0.0, 0.38]), np.array([4.0, 0.38]), "Wall-Int"),
    ]

    walls, leftovers = pair_wall_faces_v2(segments)

    assert len(walls) == 1
    assert walls[0]["espessura"] == pytest.approx(0.05)
    assert len(leftovers) == 1


def test_text_grammar_understands_garage_door_dimensions():
    meaning = classify_architectural_text(
        "2,180 x 4,800 SECTIONAL GARAGE DOOR",
    )
    dimensions = dimensions_from_text(
        "2,180 x 4,800 SECTIONAL GARAGE DOOR",
        meaning["subtype"],
    )

    assert meaning["role"] == "door"
    assert meaning["subtype"] == "garage_door"
    assert dimensions["declared_width"] == pytest.approx(4.80)
    assert dimensions["declared_height"] == pytest.approx(2.18)
    assert classify_architectural_text("GARAGE DOOR SCHEDULE") is None


def test_wide_garage_door_text_is_linked_to_single_line_geometry(tmp_path):
    doc = _new_doc()
    doc.layers.add("A-WALL")
    modelspace = doc.modelspace()
    # Parede inferior desenhada em três linhas; o trecho central representa
    # o painel do portão fechado e antes virava apenas parede contínua.
    modelspace.add_line((0.0, 0.0), (0.6, 0.0), dxfattribs={"layer": "A-WALL"})
    modelspace.add_line((0.6, 0.0), (5.4, 0.0), dxfattribs={"layer": "A-WALL"})
    modelspace.add_line((5.4, 0.0), (6.0, 0.0), dxfattribs={"layer": "A-WALL"})
    modelspace.add_line((0.0, 0.0), (0.0, 4.0), dxfattribs={"layer": "A-WALL"})
    modelspace.add_line((0.0, 4.0), (6.0, 4.0), dxfattribs={"layer": "A-WALL"})
    modelspace.add_line((6.0, 4.0), (6.0, 0.0), dxfattribs={"layer": "A-WALL"})
    modelspace.add_mtext(
        "2.180 x 4.800 SECTIONAL GARAGE DOOR",
        dxfattribs={"layer": "0", "insert": (3.0, -0.4)},
    )
    path = tmp_path / "garage_door_text.dxf"
    doc.saveas(path)

    model = parse_dxf_v2(path)
    garage_doors = [
        opening for opening in model["aberturas"]
        if opening.get("semantic_subtype") == "garage_door"
    ]

    assert len(garage_doors) == 1
    assert garage_doors[0]["tipo"] == "door"
    assert garage_doors[0]["largura"] == pytest.approx(4.80)
    assert garage_doors[0]["origem"] == "cad-text-geometry-v3"
    assert model["source"]["grammar_version"] == GRAMMAR_VERSION
    assert model["source"]["cad_summary"]["grammar_opening_candidates"] == 1
    assert any(
        cue["status"] == "matched"
        for cue in model["source"]["cad_semantic_cues"]
    )


def test_generic_block_with_swing_arc_becomes_door_without_semantic_name(tmp_path):
    doc = _new_doc()
    doc.layers.add("A-WALL")
    modelspace = doc.modelspace()
    _add_double_wall(modelspace, "A-WALL")

    anonymous = doc.blocks.new("ANON_17")
    anonymous.add_line((0.0, 0.0), (0.90, 0.0))
    anonymous.add_arc((0.0, 0.0), radius=0.90, start_angle=0, end_angle=90)
    modelspace.add_blockref("ANON_17", (1.0, 0.10))
    path = tmp_path / "anonymous_swing_door.dxf"
    doc.saveas(path)

    model = parse_dxf_v2(path)
    inferred = [
        opening for opening in model["aberturas"]
        if opening.get("origem") == "cad-block-geometry-v3"
    ]

    assert len(inferred) == 1
    assert inferred[0]["tipo"] == "door"
    assert inferred[0]["semantic_subtype"] == "swing_door"


def test_arc_inside_ignored_sanitary_layer_does_not_become_door(tmp_path):
    doc = _new_doc()
    doc.layers.add("A-WALL")
    doc.layers.add("Sanitary")
    modelspace = doc.modelspace()
    _add_double_wall(modelspace, "A-WALL")

    toilet = doc.blocks.new("Wall-mounted WC - Standard")
    toilet.add_line((0.0, 0.0), (0.70, 0.0))
    toilet.add_arc((0.35, 0.30), radius=0.30, start_angle=0, end_angle=180)
    modelspace.add_blockref(
        "Wall-mounted WC - Standard",
        (1.0, 0.10),
        dxfattribs={"layer": "Sanitary"},
    )
    path = tmp_path / "sanitary_arc.dxf"
    doc.saveas(path)

    model = parse_dxf_v2(path)

    assert not any(
        opening.get("origem") == "cad-block-geometry-v3"
        for opening in model["aberturas"]
    )


def test_missing_linked_raster_is_reported_as_lost_semantic_evidence(tmp_path):
    doc = _new_doc()
    doc.layers.add("A-WALL")
    doc.layers.add("Linked image")
    modelspace = doc.modelspace()
    _add_double_wall(modelspace, "A-WALL")
    image_def = doc.add_image_def(
        filename="missing_floor_plan.png",
        size_in_pixel=(100, 100),
    )
    modelspace.add_image(
        image_def,
        insert=(0, 0),
        size_in_units=(10, 10),
        dxfattribs={"layer": "Linked image"},
    )
    path = tmp_path / "missing_linked_image.dxf"
    doc.saveas(path)

    model = parse_dxf_v2(path)

    assert model["source"]["cad_summary"]["linked_images"] == 1
    assert model["source"]["cad_summary"]["missing_linked_images"] == 1
    assert model["source"]["cad_linked_images"][0]["available"] is False
    assert any(
        "imagem(ns) vinculada(s)" in warning
        for warning in model["warnings"]
    )


def test_uploaded_companion_raster_is_aligned_by_ocr(tmp_path, monkeypatch):
    doc = _new_doc()
    doc.layers.add("A-WALL")
    doc.layers.add("Linked image")
    modelspace = doc.modelspace()
    _add_double_wall(modelspace, "A-WALL")
    image_def = doc.add_image_def(
        filename="missing_floor_plan.png",
        size_in_pixel=(100, 100),
    )
    modelspace.add_image(
        image_def,
        insert=(0, 0),
        size_in_units=(10, 10),
        dxfattribs={"layer": "Linked image"},
    )
    path = tmp_path / "linked_image_override.dxf"
    doc.saveas(path)
    companion = tmp_path / "uploaded.png"
    companion.write_bytes(b"test")

    def fake_ocr(image):
        assert Path(image["resolved_path"]) == companion
        return [], {
            "status": "ok",
            "engine": "fake-ocr",
            "language": "pt-BR",
            "lines": [{
                "text": "GARAGE",
                "normalized_text": "garage",
                "position_raw": np.array([1.0, 2.0]),
                "cad_bbox_raw": {
                    "xmin": 0.5, "ymin": 1.5,
                    "xmax": 1.5, "ymax": 2.5,
                },
            }],
            "line_count": 1,
            "semantic_cue_count": 0,
        }

    monkeypatch.setattr("cad_detection_v2.aligned_ocr_evidence", fake_ocr)
    model = parse_dxf_v2(path, linked_image=companion)

    summary = model["source"]["cad_summary"]
    assert summary["missing_linked_images"] == 0
    assert summary["ocr_images"] == 1
    assert summary["ocr_lines"] == 1
    assert model["source"]["cad_linked_images"][0]["resolution_source"] == (
        "uploaded-companion"
    )
    assert model["source"]["cad_raster_ocr"][0]["lines"][0][
        "cad_position"
    ] == {"x": 1.0, "y": 2.0}
