from __future__ import annotations

import json
from pathlib import Path

from supabase_training_archive import SupabaseTrainingArchive


def test_archive_keeps_initial_and_final_files_in_same_job(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, str, bytes | None, str | None]] = []

    def fake_call(
        self,
        method,
        endpoint,
        *,
        data=None,
        content_type=None,
        extra_headers=None,
        expected=(200,),
    ):
        calls.append((method, endpoint, data, content_type))
        if method == "GET" and "/bucket/" in endpoint:
            return json.dumps({"id": self.bucket, "public": False}).encode()
        return b"{}"

    monkeypatch.setattr(SupabaseTrainingArchive, "_call", fake_call)
    archive = SupabaseTrainingArchive(
        project_url="https://example.supabase.co",
        secret_key="sb_secret_test-only",
    )
    original = tmp_path / "source.png"
    original.write_bytes(b"png")
    ifc = tmp_path / "result.ifc"
    ifc.write_bytes(b"ISO-10303-21;")

    archive.archive_initial(
        job="0123456789",
        original_path=original,
        rendered_image_path=original,
        model={"paredes": [], "aberturas": []},
        canvas_width_m=20,
        engine="test-engine",
    )
    archive.archive_final(
        job="0123456789",
        ifc_path=ifc,
        model={"paredes": [{"id": "w1"}], "aberturas": []},
        config={"altura": 2.8},
        engine="test-engine",
    )

    endpoints = [endpoint for _, endpoint, _, _ in calls]
    assert any(endpoint.endswith("/jobs/0123456789/original.png") for endpoint in endpoints)
    assert any(endpoint.endswith("/jobs/0123456789/model-initial.json") for endpoint in endpoints)
    assert any(endpoint.endswith("/jobs/0123456789/model-final.json") for endpoint in endpoints)
    assert any(endpoint.endswith("/jobs/0123456789/result.ifc") for endpoint in endpoints)
    assert sum(endpoint.endswith("/jobs/0123456789/manifest.json") for endpoint in endpoints) == 2


def test_secret_key_is_sent_only_in_apikey_header(monkeypatch):
    captured_headers: dict[str, str] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout):
        captured_headers.update(dict(req.header_items()))
        return FakeResponse()

    monkeypatch.setattr("supabase_training_archive.request.urlopen", fake_urlopen)
    archive = SupabaseTrainingArchive(
        project_url="https://example.supabase.co",
        secret_key="sb_secret_test-only",
    )
    archive._call("GET", "/storage/v1/bucket/test")

    normalized = {key.lower(): value for key, value in captured_headers.items()}
    assert normalized["apikey"] == "sb_secret_test-only"
    assert "authorization" not in normalized
