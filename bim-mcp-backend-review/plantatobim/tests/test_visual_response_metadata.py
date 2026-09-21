import json
import pytest
from plantatobim.gpt_plan_assistant import OpenAIResponsesClient, GptPlanError


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_response_is_recorded_even_if_truncated_and_never_contains_key(tmp_path, status):
    captured = {}
    class Response:
        status_code = 200
        def json(self):
            return {"id": "test", "status": status, "model": "gpt-6-astra",
                    "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                    "output_text": '{"ok":true}'}
    class Session:
        def post(self, url, **kwargs):
            captured.update(kwargs)
            return Response()
    client = OpenAIResponsesClient(api_key="test-private-secret", session=Session())
    def call():
        return client.structured(schema_name="test", schema={}, instructions="test",
                                 user_text="test", images=["data:image/png;base64,AA=="]*7,
                                 reasoning_effort="high", artifact_dir=tmp_path)
    if status == "incomplete":
        with pytest.raises(GptPlanError, match="não concluída"):
            call()
    else:
        assert call()[0] == {"ok": True}
    metadata = json.loads((tmp_path / "api_metadata.json").read_text())
    assert metadata["image_count"] == 7
    assert metadata["reasoning_effort"] == "high"
    assert metadata["api_seconds"] >= 0
    assert captured["json"]["reasoning"] == {"effort": "high"}
    assert "test-private-secret" not in (tmp_path / "response.json").read_text()
    assert "test-private-secret" not in (tmp_path / "api_metadata.json").read_text()
