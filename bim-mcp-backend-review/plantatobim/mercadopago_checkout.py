"""Backend-only Mercado Pago Checkout Pro integration.

The browser receives only the hosted checkout URL. Credentials, preference IDs,
payment IDs and validation details stay on the backend.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlparse

import requests


API_BASE_URL = "https://api.mercadopago.com"


class MercadoPagoError(RuntimeError):
    """Base error for safe handling at the HTTP boundary."""


class MercadoPagoConfigurationError(MercadoPagoError):
    """Raised when checkout credentials are not configured."""


class MercadoPagoRequestError(MercadoPagoError):
    """Raised when Mercado Pago rejects or cannot complete a request."""


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _public_https_url(value: str | None) -> str | None:
    candidate = str(value or "").strip().rstrip("/")
    if not candidate:
        return None
    parsed = urlparse(candidate)
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    if parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}:
        return None
    return candidate


def _money(value: Any) -> Decimal:
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise MercadoPagoRequestError("O valor do pagamento é inválido.") from exc


def verify_webhook_signature(
    *,
    x_signature: str | None,
    x_request_id: str | None,
    data_id: str | None,
    secret: str | None,
) -> bool:
    """Validate Mercado Pago's documented HMAC-SHA256 webhook signature."""
    if not all((x_signature, x_request_id, data_id, secret)):
        return False
    parts: dict[str, str] = {}
    for raw_part in str(x_signature).split(","):
        key, separator, value = raw_part.strip().partition("=")
        if separator and key and value:
            parts[key] = value
    timestamp = parts.get("ts")
    received = parts.get("v1")
    if not timestamp or not received:
        return False
    manifest = f"id:{data_id};request-id:{x_request_id};ts:{timestamp};"
    expected = hmac.new(
        str(secret).encode("utf-8"),
        manifest.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, received)


class MercadoPagoCheckout:
    """Small REST client for the Checkout Pro operations used by Plan2BIM."""

    def __init__(
        self,
        *,
        access_token: str = "",
        webhook_secret: str = "",
        enabled: bool = False,
        sandbox: bool = True,
        public_frontend_url: str = "",
        public_backend_url: str = "",
        timeout_seconds: float = 15.0,
        session: Any | None = None,
    ) -> None:
        self.access_token = access_token.strip()
        self.webhook_secret = webhook_secret.strip()
        self.enabled = bool(enabled)
        self.sandbox = bool(sandbox)
        self.public_frontend_url = _public_https_url(public_frontend_url)
        self.public_backend_url = _public_https_url(public_backend_url)
        self.timeout_seconds = max(3.0, min(float(timeout_seconds), 30.0))
        self._session = session or requests.Session()

    @classmethod
    def from_env(cls) -> "MercadoPagoCheckout":
        sandbox = _enabled(os.environ.get("MERCADOPAGO_USE_SANDBOX", "true"))
        mode_secret_name = (
            "MERCADOPAGO_WEBHOOK_SECRET_TEST"
            if sandbox
            else "MERCADOPAGO_WEBHOOK_SECRET_PRODUCTION"
        )
        # The legacy single variable remains as a local-development fallback.
        # Cloud Run binds both mode-specific values from Secret Manager.
        webhook_secret = (
            os.environ.get(mode_secret_name, "").strip()
            or os.environ.get("MERCADOPAGO_WEBHOOK_SECRET", "").strip()
        )
        return cls(
            access_token=os.environ.get("MERCADOPAGO_ACCESS_TOKEN", ""),
            webhook_secret=webhook_secret,
            enabled=_enabled(os.environ.get("MERCADOPAGO_CHECKOUT_ENABLED")),
            sandbox=sandbox,
            public_frontend_url=os.environ.get(
                "MERCADOPAGO_PUBLIC_FRONTEND_URL", ""
            ),
            public_backend_url=os.environ.get(
                "MERCADOPAGO_PUBLIC_BACKEND_URL", ""
            ),
            timeout_seconds=float(
                os.environ.get("MERCADOPAGO_TIMEOUT_SECONDS", "15")
            ),
        )

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.access_token)

    @property
    def webhook_configured(self) -> bool:
        return self.configured and bool(self.webhook_secret)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if not self.configured:
            raise MercadoPagoConfigurationError(
                "O pagamento ainda não está configurado no servidor."
            )
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if idempotency_key:
            headers["X-Idempotency-Key"] = idempotency_key
        try:
            response = self._session.request(
                method,
                f"{API_BASE_URL}{path}",
                headers=headers,
                json=json_body,
                params=params,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise MercadoPagoRequestError(
                "Não foi possível falar com o serviço de pagamento. Tente novamente."
            ) from exc
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.status_code < 200 or response.status_code >= 300:
            raise MercadoPagoRequestError(
                "O serviço de pagamento não aceitou a solicitação. Tente novamente."
            )
        if not isinstance(payload, dict):
            raise MercadoPagoRequestError(
                "O serviço de pagamento devolveu uma resposta inválida."
            )
        return payload

    def create_preference(
        self,
        *,
        job: str,
        external_reference: str,
        title: str,
        amount: Any,
    ) -> dict[str, Any]:
        price = _money(amount)
        if price <= 0:
            raise MercadoPagoRequestError("O valor do pagamento é inválido.")
        body: dict[str, Any] = {
            "items": [
                {
                    "id": job,
                    "title": str(title).strip()[:120] or "Conversão Pro",
                    "description": "Conversão de planta para modelo BIM editável",
                    "quantity": 1,
                    "currency_id": "BRL",
                    "unit_price": float(price),
                }
            ],
            "external_reference": external_reference,
            "metadata": {"plan2bim_job": job},
            "binary_mode": False,
        }
        if self.public_frontend_url:
            body["back_urls"] = {
                "success": f"{self.public_frontend_url}/?payment=success",
                "pending": f"{self.public_frontend_url}/?payment=pending",
                "failure": f"{self.public_frontend_url}/?payment=failure",
            }
            body["auto_return"] = "approved"
        if self.public_backend_url:
            body["notification_url"] = (
                f"{self.public_backend_url}/api/payments/mercadopago/webhook"
            )
        payload = self._request(
            "POST",
            "/checkout/preferences",
            json_body=body,
            idempotency_key=f"plan2bim-{job}",
        )
        preference_id = str(payload.get("id") or "").strip()
        # Checkout Pro's current API documents ``init_point`` as the hosted
        # checkout URL for both live and test credentials.  The older
        # ``sandbox_init_point`` host can enter a redirect loop when a current
        # test-buyer session reaches it, so it must not be preferred here.
        checkout_url = str(payload.get("init_point") or "").strip()
        parsed = urlparse(checkout_url)
        host = (parsed.hostname or "").lower()
        allowed_host = (
            host == "mercadopago.com"
            or host.endswith(".mercadopago.com")
            or host == "mercadopago.com.br"
            or host.endswith(".mercadopago.com.br")
        )
        if (
            not preference_id
            or parsed.scheme != "https"
            or not allowed_host
        ):
            raise MercadoPagoRequestError(
                "O serviço de pagamento não devolveu um checkout válido."
            )
        return {
            "preference_id": preference_id,
            "checkout_url": checkout_url,
            "sandbox": self.sandbox,
        }

    def search_payments(self, external_reference: str) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/v1/payments/search",
            params={
                "external_reference": external_reference,
                "sort": "date_created",
                "criteria": "desc",
                "range": "date_created",
                "begin_date": "NOW-30DAYS",
                "end_date": "NOW",
                "limit": 20,
            },
        )
        results = payload.get("results")
        if not isinstance(results, list):
            return []
        return [item for item in results if isinstance(item, dict)]

    def get_payment(self, payment_id: str) -> dict[str, Any]:
        candidate = str(payment_id or "").strip()
        if not candidate.isdigit():
            raise MercadoPagoRequestError("A notificação de pagamento é inválida.")
        return self._request("GET", f"/v1/payments/{candidate}")

    def verify_webhook(
        self,
        *,
        x_signature: str | None,
        x_request_id: str | None,
        data_id: str | None,
    ) -> bool:
        return verify_webhook_signature(
            x_signature=x_signature,
            x_request_id=x_request_id,
            data_id=data_id,
            secret=self.webhook_secret,
        )
