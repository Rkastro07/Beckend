from __future__ import annotations

import io

import plan_to_bim_free_app as free_app


class FakeArchive:
    def __init__(self):
        self.initial_calls = []
        self.final_calls = []

    def archive_initial(self, **kwargs):
        self.initial_calls.append(kwargs)
        return {"status": "converted"}

    def archive_final(self, **kwargs):
        self.final_calls.append(kwargs)
        return {"status": "ifc-exported"}


def test_consented_conversion_archives_original_and_model(monkeypatch):
    from plantatobim import pre_wall_opening_import

    fake_archive = FakeArchive()
    monkeypatch.setattr(free_app, "TRAINING_ARCHIVE", fake_archive)
    monkeypatch.setattr(
        pre_wall_opening_import,
        "pre_wall_image_to_editor_model",
        lambda *_args, **_kwargs: {
            "paredes": [{"id": "w1"}],
            "aberturas": [],
            "laje": {"contorno": []},
        },
    )

    response = free_app.app.test_client().post(
        "/api/plan-to-bim",
        data={
            "file": (io.BytesIO(b"not-a-real-png"), "sample.png"),
            "canvas_width_m": "20",
            "archive_consent": "true",
        },
        content_type="multipart/form-data",
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["archive_consent"] is True
    assert payload["archive_status"] == "saved"
    assert len(fake_archive.initial_calls) == 1
    assert fake_archive.initial_calls[0]["job"] == payload["job"]
    assert fake_archive.initial_calls[0]["original_path"].name == "sample.png"


def test_ifc_export_uses_same_consented_job(monkeypatch):
    from plantatobim import planta_to_ifc_v1

    fake_archive = FakeArchive()
    monkeypatch.setattr(free_app, "TRAINING_ARCHIVE", fake_archive)
    monkeypatch.setattr(
        planta_to_ifc_v1,
        "dict_para_modelo",
        lambda _payload: {
            "paredes": [{"id": "w1"}],
            "aberturas": [],
            "laje": None,
            "spaces": [],
        },
    )

    def fake_generate(_walls, _openings, ifc_path, _config, **_kwargs):
        ifc_path.write_text("ISO-10303-21;", encoding="utf-8")

    monkeypatch.setattr(planta_to_ifc_v1, "gerar_ifc_do_modelo", fake_generate)

    response = free_app.app.test_client().post(
        "/api/referencia/finalizar",
        json={
            "modelo": {"paredes": [{"id": "w1"}], "aberturas": []},
            "config": {"altura": 2.8},
            "nome": "sample",
            "exigir_aprovacao_cliente": True,
            "aprovacao_cliente": {"confirmado": True},
            "source_job": "0123456789",
            "archive_consent": True,
        },
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["archive_status"] == "saved"
    assert len(fake_archive.final_calls) == 1
    assert fake_archive.final_calls[0]["job"] == "0123456789"
    assert fake_archive.final_calls[0]["ifc_path"].read_text() == "ISO-10303-21;"
