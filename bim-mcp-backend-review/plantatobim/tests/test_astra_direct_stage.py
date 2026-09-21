from __future__ import annotations

from PIL import Image
import pytest

from plantatobim.astra_direct_stage import (
    AstraDirectPlanStage,
    build_direct_editor_model,
)
from plantatobim.gpt_plan_assistant import GptPlanError
from plantatobim.astra_visual_geometry import visual_to_review, prepare_visual_inputs


def _analysis() -> dict:
    return {
        "message": "Modelo geométrico direto.",
        "changed": True,
        "confidence": 0.92,
        "observations": ["Perímetro visível."],
        "assumptions": [],
        "walls": [
            {
                "id": "W-ASTRA-001",
                "ax": 1,
                "ay": 1,
                "bx": 9,
                "by": 1,
                "thickness": 0.15,
                "height": 2.8,
                "kind": "wall",
                "name": "Parede externa",
                "confidence": 0.96,
                "reason": "Duas linhas paralelas contínuas.",
            },
            {
                "id": "W-ASTRA-002",
                "ax": 1,
                "ay": 4,
                "bx": 9,
                "by": 4,
                "thickness": 0.12,
                "height": 2.8,
                "kind": "wall",
                "name": "Parede interna",
                "confidence": 0.72,
                "reason": "Trecho parcialmente encoberto.",
            },
        ],
        "openings": [
            {
                "id": "D-ASTRA-001",
                "wall_id": "W-ASTRA-001",
                "type": "door",
                "s_center": 4,
                "width": 0.9,
                "height": 2.1,
                "sill": 0,
                "name": "Porta",
                "confidence": 0.93,
                "reason": "Folha e arco visíveis.",
            }
        ],
        "slab_contour": [
            {"x": 0.5, "y": 0.5},
            {"x": 9.5, "y": 0.5},
            {"x": 9.5, "y": 4.5},
            {"x": 0.5, "y": 4.5},
        ],
        "unresolved": [],
        "_model": "gpt-6-astra",
    }


def _image(tmp_path):
    path = tmp_path / "planta.png"
    Image.new("RGB", (200, 100), "white").save(path)
    return path


def test_direct_model_contains_only_astra_authored_geometry(tmp_path):
    model = build_direct_editor_model(
        _image(tmp_path),
        _analysis(),
        canvas_width_m=10,
        original_name="planta.png",
    )
    assert model["engine"] == "astra-direct-v2"
    assert model["bbox"]["ymax"] == 5
    assert model["source"]["mode"] == "astra-direct"
    assert model["source"]["heuristic_detector_used"] is False
    assert model["reference"]["engine"] == "astra-direct"
    assert len(model["paredes"]) == 2
    assert model["paredes"][0]["origem"] == "gpt-6-astra-direct"
    assert model["paredes"][1]["astra_status"] == "review"
    assert model["astra_editor"]["geometry_source"] == "astra-only"
    assert model["astra_editor"]["heuristic_detector_used"] is False
    assert model["astra_editor"]["active"] == {"walls": 2, "openings": 1}
    assert model["astra_editor"]["excluded"] == {"walls": [], "openings": []}
    assert "detector" not in model


@pytest.mark.parametrize("missing", ["walls"])
def test_direct_model_rejects_incomplete_astra_geometry(tmp_path, missing):
    analysis = _analysis()
    analysis[missing] = []
    with pytest.raises(GptPlanError):
        build_direct_editor_model(
            _image(tmp_path),
            analysis,
            canvas_width_m=10,
            original_name="planta.png",
        )


def _visual():
    return {
        "walls": [["W1", 100, 200, 900, 200, 15, .95, "wall"]],
        "openings": [["O1", "W1", "door", 500, 200, 90, .9]],
        "dimensions": [], "notes": "", "unresolved": [],
        "slab_contour": [[50, 50], [950, 50], [950, 950], [50, 950]],
    }


def test_stage_sends_seven_views_and_no_detector_payload(tmp_path):
    captured = {}

    class FakeClient:
        def structured(self, **kwargs):
            captured.update(kwargs)
            return _visual(), {
                "provider": "openai",
                "model": "gpt-6-astra",
                "response_id": "resp_direct",
                "usage": {},
            }

    model, analysis, metadata = AstraDirectPlanStage(client=FakeClient()).analyze(
        _image(tmp_path),
        canvas_width_m=10,
        original_name="planta.png",
    )
    assert captured["schema_name"] == "plan_bim_visual_geometry_v2"
    assert len(captured["images"]) == 7
    assert "No wall/opening candidates" in captured["user_text"]
    assert "Include angled walls" in captured["instructions"]
    assert captured["reasoning_effort"] == "high"
    assert "detector_model" not in captured["user_text"]
    assert model["source"]["heuristic_detector_used"] is False
    assert analysis["_model"] == "gpt-6-astra"
    assert metadata["response_id"] == "resp_direct"


def test_crop_mapping_preserves_diagonal_and_opening_position():
    raw = _visual()
    raw["walls"][0][1:5] = [0, 0, 1000, 1000]
    raw["openings"][0][3:5] = [500, 500]
    manifest = {"source_size_px": [1000, 2000], "crop_bbox_original_px": [100, 400, 600, 1400]}
    result = visual_to_review(raw, manifest, 20)
    wall = result["walls"][0]
    assert [wall[k] for k in ("ax", "ay", "bx", "by")] == [2, 32, 12, 12]
    assert wall["thickness"] == pytest.approx(.15)
    assert result["openings"][0]["s_center"] == pytest.approx(500**.5 / 2)
    assert result["openings"][0]["width"] == pytest.approx(.9)
    assert raw["walls"][0][1:5] == [0, 0, 1000, 1000]


def test_missing_slab_does_not_discard_recognized_walls(tmp_path):
    analysis = _analysis()
    analysis["slab_contour"] = []
    model = build_direct_editor_model(_image(tmp_path), analysis, canvas_width_m=10, original_name="x.png")
    assert len(model["paredes"]) == 2
    assert model["laje"]["contorno"] == []
    assert model["laje"]["piso"]["ativo"] is False
    assert model["astra_editor"]["pending_review"]["slab"] is True


def test_off_wall_opening_is_not_silently_moved():
    raw = _visual()
    raw["openings"][0][4] = 800
    with pytest.raises(GptPlanError, match="não está sobre"):
        visual_to_review(raw, {"source_size_px": [1000, 1000], "crop_bbox_original_px": [0, 0, 1000, 1000]}, 20)


def test_views_cover_crop_and_use_global_extents(tmp_path):
    manifest = prepare_visual_inputs(_image(tmp_path), tmp_path / "views")
    assert manifest["crop_bbox_original_px"] == [0, 0, 200, 100]
    assert len(manifest["inputs"]) == 7
    assert manifest["inputs"][1]["extent"] == [0, 0, 560, 380]
    assert manifest["inputs"][-1]["extent"] == [440, 620, 1000, 1000]
