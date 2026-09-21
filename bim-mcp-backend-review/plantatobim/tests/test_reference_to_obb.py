import io
import os
from pathlib import Path
import sys

from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "cloud2bim"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / ".runtime" / "cache" / "matplotlib"),
)

import app_obb


def test_pdf_reference_can_be_finalized_and_compared(tmp_path):
    pdf_path = tmp_path / "reference.pdf"
    drawing = canvas.Canvas(str(pdf_path), pagesize=(720, 360))
    drawing.rect(72, 72, 360, 180, stroke=1, fill=0)
    drawing.save()

    client = app_obb.app.test_client()
    with pdf_path.open("rb") as source:
        imported = client.post(
            "/api/referencia/importar",
            data={
                "file": (io.BytesIO(source.read()), pdf_path.name),
                "pdf_scale": "100",
                "esp_default": "0.15",
            },
            content_type="multipart/form-data",
        )
    assert imported.status_code == 200, imported.get_json()
    model = imported.get_json()
    assert model["source"]["format"] == "pdf"
    assert len(model["paredes"]) == 4

    finalized = client.post(
        "/api/referencia/finalizar",
        json={
            "modelo": {
                "paredes": model["paredes"],
                "aberturas": model["aberturas"],
                "laje": model["laje"],
            },
            "config": {
                "altura": 2.8,
                "pavimento": "Térreo",
                "projeto": "Teste referência PDF",
            },
            "nome": "reference",
        },
    )
    assert finalized.status_code == 200, finalized.get_json()
    reference = finalized.get_json()
    assert reference["ifc_token"]
    assert reference["ready_for_comparison"] is True
    assert reference["pavimentos"]

    preview = client.get(reference["preview_url"])
    assert preview.status_code == 200
    compared = client.post(
        "/api/analisar_ai",
        data={
            "ifc_token": reference["ifc_token"],
            "pavimento": reference["pavimentos"][0],
            "ply_file": (io.BytesIO(preview.data), "reference_preview.ply"),
        },
        content_type="multipart/form-data",
    )
    assert compared.status_code == 200, compared.get_json()
    analysis = compared.get_json()
    assert analysis["resultados"]
    assert analysis["pavimento"] == reference["pavimentos"][0]
