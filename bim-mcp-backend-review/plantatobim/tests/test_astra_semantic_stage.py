from __future__ import annotations

from copy import deepcopy

import pytest

from plantatobim.astra_semantic_stage import (
    AstraSemanticStage,
    apply_semantic_review,
    validate_semantic_review,
)
from plantatobim.gpt_plan_assistant import GptPlanError


def _model():
    return {
        "bbox": {"xmin": 0, "ymin": 0, "xmax": 8, "ymax": 6},
        "source": {"mode": "2d+yolo"},
        "warnings": [],
        "paredes": [
            {"id": "W-001", "tipo": "wall", "ifc_class": "IfcWall"},
            {"id": "W-002", "tipo": "wall", "ifc_class": "IfcWall"},
        ],
        "aberturas": [{"id": "O-001", "tipo": "door", "parede_id": "W-001"}],
        "laje": {"contorno": [[0, 0], [8, 0], [8, 6], [0, 6]]},
    }


def _decision(candidate_id, classification, *, action="keep", confidence=0.95):
    return {
        "candidate_id": candidate_id,
        "action": action,
        "classification": classification,
        "suggested_name": None,
        "confidence": confidence,
        "reason": "evidência visual",
        "visual_evidence": ["traço compatível"],
    }


def _review():
    return {
        "schema_version": "1.0",
        "document": {
            "drawing_id": "A-01",
            "floor_label": "Térreo",
            "discipline": "architecture",
            "scale_text": "1:50",
            "scale_denominator": 50,
            "title": "Planta",
            "notes": [],
        },
        "walls": [
            _decision("W-001", "structural-wall"),
            _decision("W-002", "not-building-element", action="reject"),
        ],
        "openings": [_decision("O-001", "window")],
        "slab": {
            "action": "keep",
            "classification": "floor-slab",
            "confidence": 0.9,
            "reason": "perímetro fechado",
        },
        "missing_elements": [],
        "needs_human_review": True,
        "unresolved": ["confirmar W-002"],
        "summary": "Classificação concluída.",
        "_model": "gpt-6-astra",
    }


def test_semantic_stage_compiles_only_active_geometry_for_the_editor():
    candidate = apply_semantic_review(_model(), _review())

    assert len(candidate["paredes"]) == 1
    assert candidate["paredes"][0]["tipo"] == "structural-wall"
    assert candidate["aberturas"][0]["tipo"] == "window"
    assert candidate["source"]["mode"] == "2d+yolo+astra-compiled"
    assert candidate["source"]["astra_model"] == "gpt-6-astra"
    assert candidate["astra_editor"]["status"] == "ready-for-manual-review"
    assert candidate["astra_editor"]["active"] == {"walls": 1, "openings": 1}
    assert len(candidate["astra_editor"]["excluded"]["walls"]) == 1
    assert candidate["astra_editor"]["excluded"]["walls"][0]["candidate"]["id"] == "W-002"


def test_semantic_stage_keeps_review_items_visible_and_removes_orphan_openings():
    model = _model()
    model["aberturas"].append(
        {"id": "O-002", "tipo": "door", "parede_id": "W-002"}
    )
    review = _review()
    review["walls"][0] = _decision(
        "W-001", "uncertain", action="review", confidence=0.55
    )
    review["openings"].append(_decision("O-002", "door"))

    candidate = apply_semantic_review(model, review)

    assert candidate["paredes"][0]["astra_status"] == "review"
    assert candidate["paredes"][0]["ml_status"] == "uncertain"
    assert [item["id"] for item in candidate["aberturas"]] == ["O-001"]
    assert candidate["astra_editor"]["pending_review"]["walls"] == ["W-001"]
    excluded = candidate["astra_editor"]["excluded"]["openings"]
    assert excluded[0]["candidate"]["id"] == "O-002"
    assert excluded[0]["compiler_reason"] == "host-wall-rejected"


def test_semantic_stage_rejects_unknown_candidate_id():
    review = _review()
    review["walls"][0]["candidate_id"] = "W-999"

    with pytest.raises(GptPlanError, match="desconhecido"):
        validate_semantic_review(review, _model())


def test_semantic_stage_requires_a_decision_for_every_candidate():
    review = deepcopy(_review())
    review["walls"] = review["walls"][:1]

    with pytest.raises(GptPlanError, match="não classificou"):
        validate_semantic_review(review, _model())


def test_astra_stage_does_not_inherit_a_different_model_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_PLAN_MODEL", "gpt-5.6-terra")

    stage = AstraSemanticStage()

    assert stage.client.model == "gpt-6-astra"
