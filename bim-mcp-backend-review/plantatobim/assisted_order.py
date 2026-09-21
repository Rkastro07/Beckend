"""Paid assisted-order intake: payment confirmation precedes email delivery."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from email.message import EmailMessage
import hashlib
import hmac
import mimetypes
import os
from pathlib import Path
import secrets
import smtplib
import threading
from typing import Any

from .mercadopago_checkout import MercadoPagoCheckout


RECIPIENT_DEFAULT = "agenciaduobrotherdigital@gmail.com"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _price_for_area(value: Any) -> tuple[Decimal, Decimal]:
    try:
        area = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Informe a área da obra em m².") from exc
    if not area.is_finite() or not Decimal("1") <= area <= Decimal("500000"):
        raise ValueError("A área deve ficar entre 1 e 500.000 m².")
    return area, (area * Decimal("0.50")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _valid_email(value: str) -> str:
    candidate = str(value or "").strip().lower()
    if len(candidate) > 254 or "@" not in candidate:
        raise ValueError("Informe um e-mail de contato válido.")
    local, _, domain = candidate.rpartition("@")
    if not local or "." not in domain or any(char.isspace() for char in candidate):
        raise ValueError("Informe um e-mail de contato válido.")
    return candidate


class AssistedOrderMailer:
    """SMTP sender with an explicit local dry-run, never browser-side credentials."""

    def __init__(self) -> None:
        self.recipient = os.environ.get("PLAN_BIM_ASSISTED_RECIPIENT", RECIPIENT_DEFAULT).strip()
        self.host = os.environ.get("PLAN_BIM_ASSISTED_SMTP_HOST", "").strip()
        self.port = int(os.environ.get("PLAN_BIM_ASSISTED_SMTP_PORT", "587"))
        self.username = os.environ.get("PLAN_BIM_ASSISTED_SMTP_USERNAME", "").strip()
        self.password = os.environ.get("PLAN_BIM_ASSISTED_SMTP_PASSWORD", "")
        self.sender = os.environ.get("PLAN_BIM_ASSISTED_FROM", "").strip()
        self.use_ssl = _enabled(os.environ.get("PLAN_BIM_ASSISTED_SMTP_SSL"))
        self.dry_run = _enabled(os.environ.get("PLAN_BIM_ASSISTED_EMAIL_DRY_RUN"))

    @property
    def configured(self) -> bool:
        return self.dry_run or bool(self.recipient and self.host and self.sender)

    def send(self, *, order: dict[str, Any], attachment_path: Path) -> dict[str, Any]:
        if not self.configured:
            raise RuntimeError("O envio de pedidos assistidos ainda não está configurado.")
        message = EmailMessage()
        message["Subject"] = f"Novo pedido Plan to BIM — {order['job']}"
        message["From"] = self.sender or "Plan to BIM local <no-reply@localhost>"
        message["To"] = self.recipient
        message.set_content(
            "Novo pedido pago de conversão assistida.\n\n"
            f"Pedido: {order['job']}\n"
            f"Contato: {order['contact']['email']}\n"
            f"Área declarada: {order['quote']['area_m2']} m²\n"
            f"Valor pago: R$ {order['quote']['customer_price']:.2f}\n"
            f"Arquivo: {order['source']['original_name']}\n"
        )
        content_type, _ = mimetypes.guess_type(attachment_path.name)
        maintype, _, subtype = (content_type or "application/octet-stream").partition("/")
        message.add_attachment(
            attachment_path.read_bytes(), maintype=maintype, subtype=subtype or "octet-stream",
            filename=attachment_path.name,
        )
        if self.dry_run:
            output = attachment_path.parent.parent / "delivery-preview.eml"
            output.write_bytes(bytes(message))
            return {"status": "saved", "test": True}
        try:
            if self.use_ssl:
                with smtplib.SMTP_SSL(self.host, self.port, timeout=20) as client:
                    if self.username:
                        client.login(self.username, self.password)
                    client.send_message(message)
            else:
                with smtplib.SMTP(self.host, self.port, timeout=20) as client:
                    client.ehlo()
                    client.starttls()
                    client.ehlo()
                    if self.username:
                        client.login(self.username, self.password)
                    client.send_message(message)
        except (OSError, smtplib.SMTPException) as exc:
            raise RuntimeError("Não foi possível encaminhar a planta à equipe agora.") from exc
        return {"status": "sent", "test": False}


class AssistedOrderManager:
    def __init__(self, root_dir: Path, *, payment_gateway: Any | None = None, mailer: Any | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.payment_gateway = payment_gateway or MercadoPagoCheckout.from_env()
        self.mailer = mailer or AssistedOrderMailer()
        self._lock = threading.RLock()

    def _path(self, job: str) -> Path:
        return self.root_dir / job / "order.json"

    def _read(self, job: str) -> dict[str, Any]:
        try:
            return __import__("json").loads(self._path(job).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LookupError("Pedido não encontrado.") from exc

    def _write(self, state: dict[str, Any]) -> None:
        import json
        state["updated_at"] = _now()
        path = self._path(state["job"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _authorized(self, job: str, token: str) -> dict[str, Any]:
        state = self._read(job)
        if not hmac.compare_digest(self._hash(str(token or "")), state.get("access_token_hash", "")):
            raise PermissionError("Pedido não encontrado.")
        return state

    def create(
        self,
        *,
        job: str,
        original_name: str,
        original_path: Path,
        area_m2: Any,
        contact_email: str,
        owner_id: str = "",
        owner_email: str = "",
    ) -> dict[str, Any]:
        area, price = _price_for_area(area_m2)
        token = secrets.token_urlsafe(32)
        state = {
            "job": job, "created_at": _now(), "access_token_hash": self._hash(token),
            "owner": {"id": str(owner_id or "").strip(), "email": str(owner_email or "").strip().lower()},
            "status": "awaiting_payment", "source": {"original_name": original_name, "original_path": str(original_path)},
            "contact": {"email": _valid_email(contact_email)},
            "quote": {"currency": "BRL", "area_m2": float(area), "price_per_m2": 0.50,
                      "customer_price": float(price), "scope": "Pedido assistido; área declarada no envio."},
            "payment": {"status": "not_started", "confirmed": False},
            "delivery": {"status": "not_started"},
        }
        with self._lock:
            self._write(state)
        response = self.public_status(job, token)
        response["access_token"] = token
        return response

    def public_status(self, job: str, token: str) -> dict[str, Any]:
        with self._lock:
            state = self._authorized(job, token)
        return self._public_status_from_state(state)

    def _public_status_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        payment = state.get("payment") or {}
        delivery = state.get("delivery") or {}
        return {
            "ok": True, "job": state["job"], "status": state["status"],
            "created_at": state.get("created_at"), "updated_at": state.get("updated_at"),
            "quote": deepcopy(state["quote"]),
            "payment": {"status": payment.get("status", "not_started"), "test": bool(payment.get("sandbox", self.payment_gateway.sandbox))},
            "delivery": {"status": delivery.get("status", "not_started"), "test": bool(delivery.get("test", False))},
            "checkout_available": bool(self.payment_gateway.configured and self.mailer.configured),
        }

    def public_status_for_owner(self, job: str, owner_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._read(job)
            if str((state.get("owner") or {}).get("id") or "") != str(owner_id or ""):
                raise PermissionError("Pedido não encontrado.")
        return self._public_status_from_state(state)

    def list_for_owner(self, owner_id: str) -> list[dict[str, Any]]:
        import json
        projects: list[dict[str, Any]] = []
        with self._lock:
            for path in self.root_dir.glob("*/order.json"):
                try:
                    state = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if str((state.get("owner") or {}).get("id") or "") != str(owner_id or ""):
                    continue
                projects.append({
                    **self._public_status_from_state(state),
                    "kind": "assisted",
                    "source_name": str((state.get("source") or {}).get("original_name") or "Planta"),
                })
        return projects

    def resume_for_owner(self, job: str, owner_id: str) -> dict[str, Any]:
        token = secrets.token_urlsafe(32)
        with self._lock:
            state = self._read(job)
            if str((state.get("owner") or {}).get("id") or "") != str(owner_id or ""):
                raise PermissionError("Pedido não encontrado.")
            state["access_token_hash"] = self._hash(token)
            self._write(state)
        response = self._public_status_from_state(state)
        response["access_token"] = token
        response["source_name"] = str((state.get("source") or {}).get("original_name") or "Planta")
        return response

    def create_checkout(self, job: str, token: str) -> dict[str, Any]:
        if not self.payment_gateway.configured:
            raise RuntimeError("O pagamento ainda não está configurado no servidor.")
        if not self.mailer.configured:
            raise RuntimeError("O encaminhamento da planta ainda não está configurado no servidor.")
        with self._lock:
            state = self._authorized(job, token)
            if state.get("status") != "awaiting_payment":
                return self.public_status(job, token)
            payment = state["payment"]
            if payment.get("checkout_url"):
                response = self.public_status(job, token)
                response["checkout_url"] = payment["checkout_url"]
                return response
            reference = f"plan2bim-assisted-{job}-{secrets.token_hex(8)}"
            checkout = self.payment_gateway.create_preference(
                job=job, external_reference=reference,
                title=f"Pedido assistido — {state['quote']['area_m2']} m²",
                amount=state["quote"]["customer_price"],
            )
            state["payment"] = {"status": "checkout_created", "confirmed": False,
                "external_reference": reference, "preference_id": checkout["preference_id"],
                "checkout_url": checkout["checkout_url"], "sandbox": bool(checkout.get("sandbox")),
                "accepted_quote": deepcopy(state["quote"])}
            self._write(state)
        response = self.public_status(job, token)
        response["checkout_url"] = checkout["checkout_url"]
        return response

    def _find_by_reference(self, reference: str) -> dict[str, Any] | None:
        import json
        for path in self.root_dir.glob("*/order.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if hmac.compare_digest(str((state.get("payment") or {}).get("external_reference") or ""), reference):
                return state
        return None

    @staticmethod
    def _payment_matches(state: dict[str, Any], payment: dict[str, Any]) -> bool:
        try:
            amount = Decimal(str(payment.get("transaction_amount"))).quantize(Decimal("0.01"))
            expected = Decimal(str(state["quote"]["customer_price"])).quantize(Decimal("0.01"))
        except (InvalidOperation, TypeError, ValueError):
            return False
        return amount == expected and str(payment.get("currency_id") or "").upper() == "BRL"

    def _deliver(self, job: str) -> None:
        with self._lock:
            state = self._read(job)
            if state.get("delivery", {}).get("status") != "pending":
                return
            state["delivery"] = {"status": "sending"}
            self._write(state)
        try:
            result = self.mailer.send(order=state, attachment_path=Path(state["source"]["original_path"]))
        except RuntimeError:
            with self._lock:
                state = self._read(job)
                state["delivery"] = {"status": "failed"}
                self._write(state)
            return
        with self._lock:
            state = self._read(job)
            state["delivery"] = {"status": result["status"], "test": bool(result["test"]), "sent_at": _now()}
            state["status"] = "delivered" if result["status"] in {"sent", "saved"} else "approved"
            self._write(state)

    def _apply_payment(self, payment: dict[str, Any]) -> bool:
        reference = str(payment.get("external_reference") or "")
        if not reference:
            return False
        with self._lock:
            state = self._find_by_reference(reference)
            if state is None:
                return False
            current = deepcopy(state["payment"])
            provider_status = str(payment.get("status") or "").lower()
            current["provider_status"] = provider_status
            current["last_checked_at"] = _now()
            if provider_status == "approved":
                # A later poll must not downgrade a delivered order or resend it.
                if current.get("confirmed") and state.get("delivery", {}).get("status") in {"sent", "saved"}:
                    return True
                if not self._payment_matches(state, payment):
                    current["status"] = "review_required"
                    state["payment"] = current
                    self._write(state)
                    return True
                current.update({"status": "approved", "confirmed": True, "confirmed_at": _now()})
                state["payment"] = current
                if state.get("delivery", {}).get("status") == "not_started":
                    state["delivery"] = {"status": "pending"}
                state["status"] = "approved"
                self._write(state)
                job = state["job"]
            else:
                if state.get("status") == "awaiting_payment":
                    current["status"] = "pending" if provider_status in {"pending", "in_process", "authorized"} else "not_approved"
                    state["payment"] = current
                    self._write(state)
                return True
        self._deliver(job)
        return True

    def sync_payment(self, job: str, token: str) -> dict[str, Any]:
        with self._lock:
            state = self._authorized(job, token)
            reference = str((state.get("payment") or {}).get("external_reference") or "")
        if not reference:
            raise ValueError("Inicie o pagamento antes de consultar a aprovação.")
        payments = self.payment_gateway.search_payments(reference)
        selected = next((item for item in payments if str(item.get("external_reference") or "") == reference and str(item.get("status") or "").lower() == "approved"), None)
        if selected is None:
            selected = next((item for item in payments if str(item.get("external_reference") or "") == reference), None)
        if selected is not None:
            self._apply_payment(selected)
        return self.public_status(job, token)

    def handle_payment_notification(self, payment_id: str) -> bool:
        return self._apply_payment(self.payment_gateway.get_payment(payment_id))
