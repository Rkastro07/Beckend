from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac

from plantatobim.mercadopago_checkout import (
    MercadoPagoCheckout,
    verify_webhook_signature,
)


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def test_preference_keeps_secret_in_backend_and_omits_local_callbacks():
    session = FakeSession(
        [
            FakeResponse(
                201,
                {
                    "id": "pref-test",
                    "sandbox_init_point": (
                        "https://sandbox.mercadopago.com.br/checkout/v1/redirect"
                    ),
                    "init_point": (
                        "https://www.mercadopago.com.br/checkout/v1/redirect"
                    ),
                },
            )
        ]
    )
    checkout = MercadoPagoCheckout(
        access_token="backend-test-token",
        enabled=True,
        sandbox=True,
        public_frontend_url="http://127.0.0.1:3001",
        public_backend_url="http://127.0.0.1:8082",
        session=session,
    )

    created = checkout.create_preference(
        job="0123456789",
        external_reference="plan2bim-0123456789-random",
        title="Conversão Pro - sample",
        amount="59.90",
    )

    assert created["preference_id"] == "pref-test"
    assert created["checkout_url"].startswith("https://www.mercadopago.com.br/")
    method, url, request = session.calls[0]
    assert method == "POST"
    assert url.endswith("/checkout/preferences")
    assert request["headers"]["Authorization"] == "Bearer backend-test-token"
    assert "back_urls" not in request["json"]
    assert "notification_url" not in request["json"]
    assert request["json"]["items"][0]["unit_price"] == 59.90
    assert request["json"]["external_reference"] == "plan2bim-0123456789-random"


def test_preference_adds_only_public_https_callbacks():
    session = FakeSession(
        [
            FakeResponse(
                201,
                {
                    "id": "pref-live",
                    "init_point": "https://www.mercadopago.com.br/checkout/v1/redirect",
                },
            )
        ]
    )
    checkout = MercadoPagoCheckout(
        access_token="backend-live-token",
        webhook_secret="webhook-secret",
        enabled=True,
        sandbox=False,
        public_frontend_url="https://plan2bim.example",
        public_backend_url="https://api.plan2bim.example",
        session=session,
    )

    checkout.create_preference(
        job="0123456789",
        external_reference="plan2bim-0123456789-random",
        title="Conversão Pro",
        amount=59.9,
    )

    payload = session.calls[0][2]["json"]
    assert payload["back_urls"]["success"] == "https://plan2bim.example/?payment=success"
    assert payload["notification_url"] == (
        "https://api.plan2bim.example/api/payments/mercadopago/webhook"
    )
    assert payload["auto_return"] == "approved"


def test_first_preview_checkout_expires_with_retained_files():
    session = FakeSession([FakeResponse(201, {
        "id": "pref-preview",
        "init_point": "https://www.mercadopago.com.br/checkout/v1/redirect",
    })])
    checkout = MercadoPagoCheckout(
        access_token="backend-test-token", enabled=True, sandbox=True,
        session=session,
    )
    expires_at = (datetime.now(timezone.utc) + timedelta(days=20)).isoformat()
    checkout.create_preference(
        job="0123456789", external_reference="plan2bim-preview",
        title="Desbloquear prévia", amount="59.90", expires_at=expires_at,
    )
    payload = session.calls[0][2]["json"]
    assert payload["expires"] is True
    assert payload["expiration_date_to"] == expires_at
    assert payload["date_of_expiration"] == expires_at


def test_webhook_signature_uses_documented_manifest_and_constant_time_hash():
    timestamp = "1742505638683"
    data_id = "123456"
    request_id = "request-abc"
    secret = "test-webhook-secret"
    manifest = f"id:{data_id};request-id:{request_id};ts:{timestamp};"
    digest = hmac.new(
        secret.encode("utf-8"), manifest.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    assert verify_webhook_signature(
        x_signature=f"ts={timestamp},v1={digest}",
        x_request_id=request_id,
        data_id=data_id,
        secret=secret,
    )
    assert not verify_webhook_signature(
        x_signature=f"ts={timestamp},v1={'0' * 64}",
        x_request_id=request_id,
        data_id=data_id,
        secret=secret,
    )


def test_from_env_selects_test_webhook_secret(monkeypatch):
    monkeypatch.setenv("MERCADOPAGO_CHECKOUT_ENABLED", "true")
    monkeypatch.setenv("MERCADOPAGO_ACCESS_TOKEN", "APP_USR-test-token")
    monkeypatch.setenv("MERCADOPAGO_USE_SANDBOX", "true")
    monkeypatch.setenv("MERCADOPAGO_WEBHOOK_SECRET_TEST", "test-signature")
    monkeypatch.setenv(
        "MERCADOPAGO_WEBHOOK_SECRET_PRODUCTION", "production-signature"
    )
    gateway = MercadoPagoCheckout.from_env()
    assert gateway.sandbox is True
    assert gateway.webhook_secret == "test-signature"


def test_from_env_selects_production_webhook_secret(monkeypatch):
    monkeypatch.setenv("MERCADOPAGO_CHECKOUT_ENABLED", "true")
    monkeypatch.setenv("MERCADOPAGO_ACCESS_TOKEN", "APP_USR-production-token")
    monkeypatch.setenv("MERCADOPAGO_USE_SANDBOX", "false")
    monkeypatch.setenv("MERCADOPAGO_WEBHOOK_SECRET_TEST", "test-signature")
    monkeypatch.setenv(
        "MERCADOPAGO_WEBHOOK_SECRET_PRODUCTION", "production-signature"
    )
    gateway = MercadoPagoCheckout.from_env()
    assert gateway.sandbox is False
    assert gateway.webhook_secret == "production-signature"
