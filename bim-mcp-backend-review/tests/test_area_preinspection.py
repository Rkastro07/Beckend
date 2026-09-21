import pytest

from plantatobim.area_preinspection import validate_inspection, DeepSeekAreaInspector, first_page_text
from plantatobim.astra_local_flow import quote_for_document, AstraLocalFlowManager
from test_astra_local_flow import FakePaymentGateway, _create_job, _inspection


def raw_area(**overrides):
    return {"area_m2": 200, "confidence": .97, "basis": "printed_gross_floor_area",
            "scope": "single_floor_first_page", "page": 1, "ambiguous": False,
            "evidence": "Área construída do pavimento: 200,00 m²", **overrides}


def test_explicit_area_and_text_evidence():
    raw = raw_area()
    result = validate_inspection(raw, "Título\n" + raw["evidence"] + "\nRodapé")
    assert result["area_m2"] == 200
    assert result["verification"] == "pdf_text_and_vision"


def test_pdfium_resource_lifecycle(tmp_path):
    import pypdfium2 as pdfium
    path = tmp_path / "blank.pdf"
    doc = pdfium.PdfDocument.new()
    page = doc.new_page(200, 200)
    page.close()
    doc.save(str(path))
    doc.close()
    assert first_page_text(path) == ""


@pytest.mark.parametrize("overrides", [
    {"area_m2": float("nan")}, {"area_m2": float("inf")}, {"area_m2": -1},
    {"area_m2": 250}, {"confidence": .5}, {"confidence": float("nan")},
    {"basis": "estimated_from_canvas"}, {"scope": "whole_building"},
    {"ambiguous": True}, {"page": 2}, {"evidence": "Área do terreno: 200 m²"},
    {"evidence": "Área da sala: 200 m²"}, {"evidence": "Área construída: 200"},
    {"evidence": "Pavimento 200. Área construída: 100 m²"},
    {"evidence": "Área construída: 1.2.3 m²"},
])
def test_uncertain_area_does_not_block_fixed_page_quote(overrides):
    result = validate_inspection(raw_area(**overrides))
    quote = quote_for_document({}, result)
    assert result["status"] == "needs_area"
    assert quote["ready"] is True
    assert quote["customer_price"] == 59.9


def test_model_cannot_invent_pdf_evidence():
    assert validate_inspection(raw_area(), "Área construída: 500 m²")["status"] == "needs_area"


def test_every_first_page_uses_the_configured_minimum(monkeypatch):
    monkeypatch.setenv("ASTRA_TEST_FIXED_COST_BRL", "0")
    monkeypatch.setenv("ASTRA_TEST_USD_BRL", "1")
    assert quote_for_document({}, _inspection(20))["customer_price"] == 59.9
    assert quote_for_document({}, _inspection(200))["customer_price"] == 59.9
    assert quote_for_document({}, _inspection(1000))["customer_price"] == 59.9


def test_area_reference_is_separate_from_launch_price(monkeypatch):
    monkeypatch.setenv("PLAN_BIM_MIN_PRICE_BRL", "59.90")
    monkeypatch.setenv("PLAN_BIM_PRICE_PER_M2", "0.50")
    quote = quote_for_document({}, {
        "status": "estimated",
        "estimated_area_m2": 386.4,
        "scale_source": "door-wall-consensus",
        "scale_confidence": 0.88,
    })
    assert quote["calculated_price_brl"] == 193.2
    assert quote["applied_price_brl"] == 59.9
    assert quote["customer_price"] == 59.9
    assert quote["pricing_mode"] == "launch-fixed"
    assert quote["scale_source"] == "door-wall-consensus"


def test_no_area_from_profile_or_user_width():
    quote = quote_for_document({"area_m2": 200, "canvas_width_m": 20})
    assert quote["customer_price"] == 59.9
    assert quote["ready"] is True


def test_quote_skips_area_inspection_and_freezes_checkout(monkeypatch, tmp_path):
    calls = []
    def inspect(self, **kwargs):
        calls.append(kwargs)
        return {**_inspection(), "internal": {"provider": "secret-provider", "raw": {}}}
    monkeypatch.setattr(DeepSeekAreaInspector, "inspect", inspect)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    gateway = FakePaymentGateway()
    root = tmp_path / "jobs"
    manager = AstraLocalFlowManager(root, enabled=True, payment_gateway=gateway)
    created = _create_job(manager, tmp_path)
    job, token = created["job"], created["access_token"]
    assert "secret-provider" not in str(created)
    assert "area_inspection" not in created
    manager.create_checkout(job, token)
    before = manager.public_status(job, token)["quote"]["customer_price"]
    manager.close()
    monkeypatch.setenv("PLAN_BIM_MIN_PRICE_BRL", "99.90")
    manager = AstraLocalFlowManager(root, enabled=True, payment_gateway=gateway)
    assert manager.public_status(job, token)["quote"]["customer_price"] == before
    manager.create_checkout(job, token)
    assert calls == []
    assert len(gateway.preferences) == 1
    assert manager._read(job)["payment"]["accepted_quote"]["customer_price"] == before
    manager.close()


def test_area_unknown_does_not_block_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(DeepSeekAreaInspector, "inspect", lambda self, **kwargs: validate_inspection({}))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    gateway = FakePaymentGateway()
    manager = AstraLocalFlowManager(tmp_path / "jobs", enabled=True, payment_gateway=gateway)
    created = _create_job(manager, tmp_path)
    assert created["quote"]["checkout_available"] is True
    manager.create_checkout(created["job"], created["access_token"])
    assert len(gateway.preferences) == 1
    assert manager._read(created["job"])["status"] == "awaiting_payment"
    manager.close()


def test_multiple_pages_dont_multiply_the_bill():
    quote = quote_for_document({"page_count": 12}, _inspection(200))
    assert quote["page_count"] == 12
    assert quote["processed_pages"] == 1
    assert quote["price_per_page"] == 59.9
    assert quote["customer_price"] == 59.9
    assert quote["basis"]["processed_pages"] == 1
