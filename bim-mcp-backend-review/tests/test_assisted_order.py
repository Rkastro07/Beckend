from __future__ import annotations

import io

import pytest

import plan_to_bim_free_app as free_app
from plantatobim.assisted_order import AssistedOrderManager


class FakeGateway:
    configured = True
    sandbox = True

    def __init__(self):
        self.preferences = []
        self.payments = []

    def create_preference(self, **payload):
        self.preferences.append(payload)
        return {"preference_id": "pref-test", "checkout_url": "https://sandbox.mercadopago.com.br/checkout/test", "sandbox": True}

    def search_payments(self, reference):
        return [item for item in self.payments if item["external_reference"] == reference]

    def get_payment(self, payment_id):
        return next(item for item in self.payments if str(item["id"]) == str(payment_id))

    def verify_webhook(self, **_kwargs):
        return True


class FakeMailer:
    configured = True

    def __init__(self):
        self.sent = []

    def send(self, *, order, attachment_path):
        self.sent.append((order, attachment_path.read_bytes()))
        return {"status": "sent", "test": False}


def make_order(manager, tmp_path):
    source = tmp_path / "obra.pdf"
    source.write_bytes(b"%PDF-test")
    return manager.create(job="0123456789", original_name="obra.pdf", original_path=source, area_m2="123.4", contact_email="cliente@example.com")


def approved(manager):
    state = manager._read("0123456789")
    return {"id": "554433", "status": "approved", "external_reference": state["payment"]["external_reference"], "transaction_amount": state["quote"]["customer_price"], "currency_id": "BRL"}


def test_order_charges_fifty_cents_and_delivers_only_after_match(tmp_path):
    gateway, mailer = FakeGateway(), FakeMailer()
    manager = AssistedOrderManager(tmp_path / "orders", payment_gateway=gateway, mailer=mailer)
    created = make_order(manager, tmp_path)
    assert created["quote"]["customer_price"] == 61.70
    assert "cliente@example.com" not in str(created)
    checkout = manager.create_checkout(created["job"], created["access_token"])
    assert checkout["checkout_url"].startswith("https://sandbox.mercadopago")
    assert gateway.preferences[0]["amount"] == 61.70
    gateway.payments.append(approved(manager))
    status = manager.sync_payment(created["job"], created["access_token"])
    assert status["status"] == "delivered"
    assert status["delivery"]["status"] == "sent"
    assert len(mailer.sent) == 1
    assert mailer.sent[0][1] == b"%PDF-test"
    # Further status polls must not create a second email or change delivery state.
    again = manager.sync_payment(created["job"], created["access_token"])
    assert again["status"] == "delivered"
    assert len(mailer.sent) == 1


def test_wrong_value_never_delivers(tmp_path):
    gateway, mailer = FakeGateway(), FakeMailer()
    manager = AssistedOrderManager(tmp_path / "orders", payment_gateway=gateway, mailer=mailer)
    created = make_order(manager, tmp_path)
    manager.create_checkout(created["job"], created["access_token"])
    payment = approved(manager)
    payment["transaction_amount"] = 1
    gateway.payments.append(payment)
    status = manager.sync_payment(created["job"], created["access_token"])
    assert status["payment"]["status"] == "review_required"
    assert status["delivery"]["status"] == "not_started"
    assert not mailer.sent


@pytest.mark.parametrize("area", ["0", "-1", "500001", "nope"])
def test_invalid_area_is_rejected(tmp_path, area):
    manager = AssistedOrderManager(tmp_path / "orders", payment_gateway=FakeGateway(), mailer=FakeMailer())
    source = tmp_path / "obra.pdf"
    source.write_bytes(b"x")
    with pytest.raises(ValueError):
        manager.create(job="0123456789", original_name="obra.pdf", original_path=source, area_m2=area, contact_email="cliente@example.com")


def test_http_upload_and_token_protection(monkeypatch, tmp_path):
    manager = AssistedOrderManager(tmp_path / "orders", payment_gateway=FakeGateway(), mailer=FakeMailer())
    monkeypatch.setattr(free_app, "ASSISTED_ORDER_MANAGER", manager)
    client = free_app.app.test_client()
    response = client.post("/api/assisted-orders", data={"file": (io.BytesIO(b"%PDF-test"), "obra.pdf"), "area_m2": "50", "contact_email": "cliente@example.com"}, content_type="multipart/form-data")
    assert response.status_code == 201
    payload = response.get_json()
    assert payload["quote"]["customer_price"] == 25
    assert client.get(f"/api/assisted-orders/{payload['job']}?token=wrong").status_code == 404


def test_disabled_assisted_order_cannot_create_or_receive_payment(monkeypatch):
    monkeypatch.setattr(free_app, "ASSISTED_ORDER_ENABLED", False)
    client = free_app.app.test_client()

    health = client.get("/api/health").get_json()
    assert "assisted-order" not in health["capabilities"]
    assert health["assisted_order"]["enabled"] is False
    assert health["assisted_order"]["payment_configured"] is False
    assert client.post(
        "/api/assisted-orders",
        data={"file": (io.BytesIO(b"%PDF-test"), "obra.pdf")},
        content_type="multipart/form-data",
    ).status_code == 404
    assert client.post("/api/assisted-orders/0123456789/checkout", json={}).status_code == 404
