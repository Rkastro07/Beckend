"""Abuse guard for the public Plan2BIM endpoints.

Production counters are consumed atomically in Supabase so the limits remain
global when Cloud Run scales horizontally. Local development keeps a small
in-process implementation and raw IP addresses are never stored in either mode.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import hmac
import os
import threading
import time


class RateLimitExceeded(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: int) -> None:
        super().__init__(message)
        self.retry_after_seconds = max(1, int(retry_after_seconds))


@dataclass(frozen=True)
class LimitRule:
    scope: str
    maximum: int
    window_seconds: int


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


class OperationalGuard:
    """Rate-limit upload preparation and checkout creation by user and IP."""

    def __init__(
        self,
        *,
        secret: str = "",
        clock=time.time,
        durable_store=None,
    ) -> None:
        signing_secret = str(secret or "").strip()
        if not signing_secret:
            signing_secret = os.environ.get("PLAN_BIM_RATE_LIMIT_SECRET", "").strip()
        if not signing_secret:
            signing_secret = os.environ.get("PLAN_BIM_SESSION_SECRET", "").strip()
        if not signing_secret:
            signing_secret = os.environ.get("SUPABASE_SECRET_KEY", "").strip()
        # Local development still gets stable counters without shipping a default
        # production secret. Production already has PLAN_BIM_SESSION_SECRET.
        self._secret = (signing_secret or os.urandom(32).hex()).encode("utf-8")
        self._clock = clock
        self._durable_store = durable_store
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self.upload_user = LimitRule(
            "upload-user",
            _positive_int("PLAN_BIM_UPLOADS_PER_USER_HOUR", 8),
            60 * 60,
        )
        self.upload_ip = LimitRule(
            "upload-ip",
            _positive_int("PLAN_BIM_UPLOADS_PER_IP_HOUR", 20),
            60 * 60,
        )
        self.order_user = LimitRule(
            "order-user",
            _positive_int("PLAN_BIM_ORDERS_PER_USER_DAY", 3),
            24 * 60 * 60,
        )
        self.order_ip = LimitRule(
            "order-ip",
            _positive_int("PLAN_BIM_ORDERS_PER_IP_DAY", 10),
            24 * 60 * 60,
        )
        self.preview_global = LimitRule(
            "first-preview-global",
            _positive_int("PLAN_BIM_FIRST_PREVIEWS_PER_DAY", 15),
            24 * 60 * 60,
        )
        self.preview_ip = LimitRule(
            "first-preview-ip",
            _positive_int("PLAN_BIM_FIRST_PREVIEWS_PER_IP_DAY", 2),
            24 * 60 * 60,
        )

    def fingerprint(self, value: str) -> str:
        normalized = str(value or "unknown").strip().lower() or "unknown"
        return hmac.new(
            self._secret,
            normalized.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _consume(self, rule: LimitRule, identity: str, message: str) -> None:
        now = float(self._clock())
        identity_hash = self.fingerprint(identity)
        if self._durable_store is not None:
            allowed, retry_after, _ = self._durable_store.consume_rate_limit(
                scope=rule.scope,
                identity_hash=identity_hash,
                maximum=rule.maximum,
                window_seconds=rule.window_seconds,
            )
            if not allowed:
                raise RateLimitExceeded(message, retry_after)
            return

        key = (rule.scope, identity_hash)
        cutoff = now - rule.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= rule.maximum:
                retry_after = int(max(1, events[0] + rule.window_seconds - now))
                raise RateLimitExceeded(message, retry_after)
            events.append(now)

    def check_upload(self, *, owner_id: str, client_ip: str) -> str:
        if owner_id:
            self._consume(
                self.upload_user,
                f"user:{owner_id}",
                "Você atingiu o limite de envios por hora. Aguarde antes de enviar outra planta.",
            )
        self._consume(
            self.upload_ip,
            f"ip:{client_ip}",
            "Muitos arquivos foram enviados desta conexão. Aguarde e tente novamente.",
        )
        return self.fingerprint(f"ip:{client_ip}")

    def check_order(self, *, owner_id: str, client_ip: str) -> None:
        if owner_id:
            self._consume(
                self.order_user,
                f"user:{owner_id}",
                "Você atingiu o limite diário de novos pedidos. Os pedidos existentes continuam disponíveis.",
            )
        self._consume(
            self.order_ip,
            f"ip:{client_ip}",
            "Muitos pedidos foram iniciados desta conexão hoje. Tente novamente mais tarde.",
        )

    def check_first_preview(self, *, client_ip: str) -> None:
        self._consume(
            self.preview_ip,
            f"ip:{client_ip}",
            "O limite de prévias desta conexão foi atingido hoje. Tente novamente amanhã.",
        )
        self._consume(
            self.preview_global,
            "plan2bim:first-preview",
            "As prévias de hoje se esgotaram. Tente novamente amanhã.",
        )
