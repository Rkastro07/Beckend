from __future__ import annotations

import json

import pytest

from plantatobim.gpt_plan_assistant import (
    DEEPSEEK_BETA_CHAT_URL,
    DeepSeekPlanClient,
    GptPlanAssistant,
    GptPlanError,
    GptPlanUnavailable,
    OpenAIResponsesClient,
    build_candidate_model,
    validate_calibration,
)


def _model():
    return {
        "nome": "teste",
        "bbox": {"xmin": 0.0, "ymin": 0.0, "xmax": 10.0, "ymax": 10.0},
        "diagnostico": {"blocos_esquadria": 0},
        "source": {"mode": "morphology-2d", "semantic_level": "geometric"},
        "warnings": [],
        "paredes": [{
            "id": "W-001", "ax": 0.0, "ay": 0.0, "bx": 5.0, "by": 0.0,
            "espessura": 0.15, "altura": 2.8, "layer": "Wall",
        }],
        "aberturas": [],
        "laje": {
            "contorno": [[0, 0], [5, 0], [5, 4], [0, 4]],
            "piso": {"ativo": True, "espessura": 0.12},
            "teto": {"ativo": False, "espessura": 0.12},
        },
        "spaces": [],
    }


def _review():
    return {
        "message": "Parede e porta revisadas.",
        "changed": True,
        "confidence": 0.9,
        "observations": ["vão com arco"],
        "assumptions": [],
        "unresolved": [],
        "walls": [{
            "id": "W-001", "ax": 0.0, "ay": 0.0, "bx": 5.0, "by": 0.0,
            "thickness": 0.18, "height": 2.8, "kind": "wall",
            "name": "Parede mãe", "confidence": 0.93, "reason": "massa contínua",
        }],
        "openings": [{
            "id": "D-001", "wall_id": "W-001", "type": "door",
            "s_center": 2.0, "width": 0.8, "height": 2.1, "sill": 0.2,
            "name": "Porta", "confidence": 0.91, "reason": "arco e folha",
        }],
        "slab_contour": [
            {"x": 0, "y": 0}, {"x": 5, "y": 0},
            {"x": 5, "y": 4}, {"x": 0, "y": 4},
        ],
    }


def test_candidate_keeps_parent_wall_and_hosts_opening():
    candidate = build_candidate_model(_model(), _review())

    assert len(candidate["paredes"]) == 1
    assert candidate["paredes"][0]["id"] == "W-001"
    assert candidate["paredes"][0]["origem"] == "gpt-6-astra-visual-review"
    assert candidate["aberturas"][0]["parede_id"] == "W-001"
    assert candidate["aberturas"][0]["peitoril"] == 0.0
    assert candidate["source"]["mode"] == "morphology-2d+gpt-6-astra-vision"


def test_candidate_rejects_orphan_opening():
    review = _review()
    review["openings"][0]["wall_id"] = "W-INEXISTENTE"

    with pytest.raises(GptPlanError, match="hospedeira"):
        build_candidate_model(_model(), review)


def test_candidate_rejects_opening_outside_parent_wall():
    review = _review()
    review["openings"][0]["s_center"] = 4.9

    with pytest.raises(GptPlanError, match="saiu"):
        build_candidate_model(_model(), review)


def test_candidate_locks_wall_axis_bound_to_printed_dimension():
    model = _model()
    model["metric_constraints"] = [{
        "id": "DIM-001",
        "value_m": 3.08,
        "axis": "vertical",
        "wall_bindings": [{
            "role": "p1",
            "wall_id": "W-001",
            "orientation": "horizontal",
            "locked_axis_m": 0.0,
        }],
    }]
    review = _review()
    review["walls"][0]["ay"] = 0.8
    review["walls"][0]["by"] = 0.8

    candidate = build_candidate_model(model, review)

    assert candidate["paredes"][0]["ay"] == 0.0
    assert candidate["paredes"][0]["by"] == 0.0
    enforcement = candidate["gpt_plan"]["_metric_constraint_enforcement"]
    assert enforcement["protected_wall_axes"] == 1


def test_calibration_requires_non_parallel_dimensions_to_apply():
    calibration = {
        "message": "ok",
        "needs_rectification": True,
        "source_quad_px": [
            {"x": 10, "y": 10}, {"x": 190, "y": 10},
            {"x": 190, "y": 90}, {"x": 10, "y": 90},
        ],
        "main_width_m": 8.3,
        "main_height_m": 5.3,
        "right_extra_m": 0.0,
        "dimensions": [{
            "text": "3.08", "value_m": 3.08, "axis": "horizontal",
            "p1_px": {"x": 20, "y": 80},
            "p2_px": {"x": 120, "y": 80},
            "confidence": 0.9, "reason": "legível",
        }],
        "confidence": 0.9,
        "assumptions": [],
    }

    validated = validate_calibration(calibration, (100, 200, 3))

    assert validated["validation"]["valid_quad"] is True
    assert validated["validation"]["non_parallel_dimensions"] is False
    assert validated["applied"] is False


def test_calibration_ignores_model_confidence_when_geometry_is_valid():
    calibration = {
        "message": "cotas encontradas",
        "needs_rectification": True,
        "source_quad_px": [
            {"x": 10, "y": 10}, {"x": 190, "y": 10},
            {"x": 190, "y": 90}, {"x": 10, "y": 90},
        ],
        "main_width_m": 8.4,
        "main_height_m": 4.9,
        "right_extra_m": 1.0,
        "dimensions": [
            {
                "text": "8.40", "value_m": 8.4, "axis": "horizontal",
                "p1_px": {"x": 10, "y": 90},
                "p2_px": {"x": 190, "y": 90},
                "confidence": 0.1, "reason": "legível",
            },
            {
                "text": "4.90", "value_m": 4.9, "axis": "vertical",
                "p1_px": {"x": 10, "y": 10},
                "p2_px": {"x": 10, "y": 90},
                "confidence": 0.1, "reason": "legível",
            },
        ],
        "confidence": 0.01,
        "assumptions": [],
    }

    validated = validate_calibration(calibration, (100, 200, 3))

    assert validated["applied"] is True
    assert validated["main_width_m"] == pytest.approx(8.4)
    assert validated["main_height_m"] == pytest.approx(4.9)
    assert validated["validation"]["model_confidence_advisory_only"] is True


def test_calibration_replaces_guessed_extent_with_anchored_dimensions():
    calibration = {
        "message": "cotas ancoradas",
        "needs_rectification": True,
        "source_quad_px": [
            {"x": 10, "y": 10}, {"x": 190, "y": 10},
            {"x": 190, "y": 90}, {"x": 10, "y": 90},
        ],
        "main_width_m": 7.37,
        "main_height_m": 4.85,
        "right_extra_m": 0.0,
        "dimensions": [
            {
                "text": "3.00", "value_m": 3.0, "axis": "horizontal",
                "p1_px": {"x": 55, "y": 80},
                "p2_px": {"x": 145, "y": 80},
                "confidence": 0.01, "reason": "extremos visíveis",
            },
            {
                "text": "2.00", "value_m": 2.0, "axis": "vertical",
                "p1_px": {"x": 30, "y": 30},
                "p2_px": {"x": 30, "y": 70},
                "confidence": 0.01, "reason": "extremos visíveis",
            },
        ],
        "confidence": 0.01,
        "assumptions": [],
    }

    validated = validate_calibration(calibration, (100, 200, 3))

    assert validated["applied"] is True
    assert validated["model_main_width_m"] == pytest.approx(7.37)
    assert validated["main_width_m"] == pytest.approx(6.0)
    assert validated["model_main_height_m"] == pytest.approx(4.85)
    assert validated["main_height_m"] == pytest.approx(4.0)
    assert validated["validation"]["extent_source"] == "continuous-dimension-chain-piecewise-1d"


def test_continuous_chain_preserves_each_printed_dimension_and_rejects_other_line():
    calibration = {
        "message": "cadeia inferior",
        "needs_rectification": True,
        "source_quad_px": [
            {"x": 0, "y": 0}, {"x": 1000, "y": 0},
            {"x": 1000, "y": 500}, {"x": 0, "y": 500},
        ],
        "main_width_m": 9.9,
        "main_height_m": 9.9,
        "right_extra_m": 0.0,
        "dimensions": [
            {
                "text": "3.08", "value_m": 3.08, "axis": "horizontal",
                "p1_px": {"x": 0, "y": 490}, "p2_px": {"x": 400, "y": 490},
                "confidence": 0.1, "reason": "cadeia",
            },
            {
                "text": "1.20", "value_m": 1.20, "axis": "horizontal",
                "p1_px": {"x": 400, "y": 490}, "p2_px": {"x": 540, "y": 490},
                "confidence": 0.1, "reason": "cadeia",
            },
            {
                "text": "3.29", "value_m": 3.29, "axis": "horizontal",
                "p1_px": {"x": 540, "y": 490}, "p2_px": {"x": 1000, "y": 490},
                "confidence": 0.1, "reason": "cadeia",
            },
            {
                "text": "3.50", "value_m": 3.5, "axis": "vertical",
                "p1_px": {"x": 10, "y": 230}, "p2_px": {"x": 10, "y": 500},
                "confidence": 0.1, "reason": "linha conflitante",
            },
            {
                "text": "2.40", "value_m": 2.4, "axis": "vertical",
                "p1_px": {"x": 990, "y": 0}, "p2_px": {"x": 990, "y": 250},
                "confidence": 0.1, "reason": "cadeia direita",
            },
            {
                "text": "2.40", "value_m": 2.4, "axis": "vertical",
                "p1_px": {"x": 990, "y": 250}, "p2_px": {"x": 990, "y": 500},
                "confidence": 0.1, "reason": "cadeia direita",
            },
        ],
        "confidence": 0.01,
        "assumptions": [],
    }

    validated = validate_calibration(calibration, (500, 1000, 3))

    assert validated["main_width_m"] == pytest.approx(7.57)
    assert validated["main_height_m"] == pytest.approx(4.8)
    by_text = {item["text"]: item for item in validated["dimensions"]}
    assert by_text["3.08"]["derived_value_m"] == pytest.approx(3.08)
    assert by_text["1.20"]["derived_value_m"] == pytest.approx(1.20)
    assert by_text["3.29"]["derived_value_m"] == pytest.approx(3.29)
    assert by_text["3.50"]["used_for_scale"] is False


def test_same_dimension_line_keeps_hard_anchors_when_middle_cota_is_missing():
    calibration = {
        "message": "cadeia com lacuna",
        "needs_rectification": True,
        "source_quad_px": [
            {"x": 0, "y": 0}, {"x": 1000, "y": 0},
            {"x": 1000, "y": 500}, {"x": 0, "y": 500},
        ],
        "main_width_m": 0,
        "main_height_m": 0,
        "right_extra_m": 0.0,
        "dimensions": [
            {
                "text": "3.08", "value_m": 3.08, "axis": "horizontal",
                "p1_px": {"x": 0, "y": 490}, "p2_px": {"x": 400, "y": 490},
                "confidence": 0.1, "reason": "primeiro trecho",
            },
            {
                "text": "3.29", "value_m": 3.29, "axis": "horizontal",
                "p1_px": {"x": 590, "y": 490}, "p2_px": {"x": 1000, "y": 490},
                "confidence": 0.1, "reason": "último trecho",
            },
            {
                "text": "2.40", "value_m": 2.4, "axis": "vertical",
                "p1_px": {"x": 990, "y": 0}, "p2_px": {"x": 990, "y": 250},
                "confidence": 0.1, "reason": "metade superior",
            },
            {
                "text": "2.40", "value_m": 2.4, "axis": "vertical",
                "p1_px": {"x": 990, "y": 250}, "p2_px": {"x": 990, "y": 500},
                "confidence": 0.1, "reason": "metade inferior",
            },
        ],
        "confidence": 0.01,
        "assumptions": [],
    }

    validated = validate_calibration(calibration, (500, 1000, 3))

    horizontal = [
        item for item in validated["dimensions"] if item["axis"] == "horizontal"
    ]
    assert all(item["used_for_scale"] for item in horizontal)
    assert horizontal[0]["derived_value_m"] == pytest.approx(3.08)
    assert horizontal[1]["derived_value_m"] == pytest.approx(3.29)


class _FakeResponse:
    def __init__(self, result, *, status_code=200):
        self.result = result
        self.status_code = status_code

    def json(self):
        return {
            "id": "resp_test",
            "model": "gpt-5.5-2026-04-23",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(self.result)}],
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }


class _FakeSession:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _FakeResponse(self.result)


class _FakeDeepSeekResponse:
    status_code = 200

    def __init__(self, result, model):
        self.result = result
        self.model = model

    def json(self):
        return {
            "id": "ds_test",
            "model": self.model,
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {"arguments": json.dumps(self.result)},
                    }],
                },
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        }


class _FakeDeepSeekSession:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _FakeDeepSeekResponse(self.result, kwargs["json"]["model"])


def test_responses_request_keeps_key_server_side_and_disables_storage():
    session = _FakeSession({"ok": True})
    client = OpenAIResponsesClient(api_key="server-secret", session=session)

    result, metadata = client.structured(
        schema_name="test_schema",
        schema={
            "type": "object", "additionalProperties": False,
            "required": ["ok"], "properties": {"ok": {"type": "boolean"}},
        },
        instructions="teste",
        user_text="analise",
        images=["data:image/png;base64,AA=="],
    )

    _args, kwargs = session.calls[0]
    assert result == {"ok": True}
    assert metadata["response_id"] == "resp_test"
    assert kwargs["headers"]["Authorization"] == "Bearer server-secret"
    assert kwargs["json"]["model"] == "gpt-6-astra"
    assert kwargs["json"]["max_output_tokens"] == 48000
    assert kwargs["json"]["store"] is False
    assert kwargs["json"]["input"][0]["content"][1]["detail"] == "original"
    assert kwargs["json"]["text"]["format"]["strict"] is True


def test_deepseek_uses_vision_model_and_strict_tool_for_images():
    session = _FakeDeepSeekSession({"ok": True})
    client = DeepSeekPlanClient(api_key="server-secret", session=session)

    result, metadata = client.structured(
        schema_name="test_schema",
        schema={
            "type": "object", "additionalProperties": False,
            "required": ["ok"], "properties": {"ok": {"type": "boolean"}},
        },
        instructions="teste",
        user_text="analise",
        images=["data:image/png;base64,AA=="],
    )

    args, kwargs = session.calls[0]
    assert args[0] == DEEPSEEK_BETA_CHAT_URL
    assert result == {"ok": True}
    assert metadata["provider"] == "deepseek"
    assert kwargs["headers"]["Authorization"] == "Bearer server-secret"
    assert kwargs["json"]["model"] == "deepseek-v4-flash-vision-exp"
    assert kwargs["json"]["tools"][0]["function"]["strict"] is True
    assert kwargs["json"]["messages"][1]["content"][1]["type"] == "image_url"


def test_deepseek_uses_reasoning_model_for_text_audit():
    session = _FakeDeepSeekSession({"ok": True})
    client = DeepSeekPlanClient(api_key="server-secret", session=session)

    client.structured(
        schema_name="test_schema",
        schema={
            "type": "object", "additionalProperties": False,
            "required": ["ok"], "properties": {"ok": {"type": "boolean"}},
        },
        instructions="audite",
        user_text="modelo",
        images=[],
    )

    _args, kwargs = session.calls[0]
    assert kwargs["json"]["model"] == "deepseek-v4-pro"


def test_deepseek_text_audit_can_be_disabled(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_TEXT_AUDIT", "false")

    client = DeepSeekPlanClient(api_key="server-secret")

    assert client.requires_text_audit is False


def test_assistant_can_select_provider_per_request(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")

    assistant = GptPlanAssistant(provider="openai")
    status = assistant.status()

    assert assistant.client.provider == "openai"
    assert status["default_provider"] == "openai"
    assert status["providers"]["openai"]["configured"] is True
    assert status["providers"]["deepseek"]["configured"] is True
    assert status["providers"]["openai"]["model"] == "gpt-6-astra"


def test_assistant_rejects_unknown_provider():
    with pytest.raises(GptPlanUnavailable, match="Provedor"):
        GptPlanAssistant(provider="browser-secret")
