from __future__ import annotations

import io
import pytest

import plan_to_bim_free_app as free_app
from plantatobim.operational_guard import OperationalGuard, RateLimitExceeded


class FakeDurableStore:
    def __init__(self) -> None:
        self.calls = []

    def consume_rate_limit(self, **payload):
        self.calls.append(payload)
        return (len(self.calls) == 1, 77, len(self.calls))


def test_upload_limit_is_enforced_without_storing_raw_ip(monkeypatch):
    monkeypatch.setenv("PLAN_BIM_UPLOADS_PER_USER_HOUR", "2")
    monkeypatch.setenv("PLAN_BIM_UPLOADS_PER_IP_HOUR", "5")
    now = [1000.0]
    guard = OperationalGuard(secret="test-secret", clock=lambda: now[0])

    fingerprint = guard.check_upload(owner_id="user-1", client_ip="203.0.113.8")
    guard.check_upload(owner_id="user-1", client_ip="203.0.113.8")

    assert fingerprint != "203.0.113.8"
    with pytest.raises(RateLimitExceeded) as caught:
        guard.check_upload(owner_id="user-1", client_ip="203.0.113.8")
    assert caught.value.retry_after_seconds == 3600

    now[0] += 3601
    guard.check_upload(owner_id="user-1", client_ip="203.0.113.8")


def test_order_limit_is_separate_from_upload_limit(monkeypatch):
    monkeypatch.setenv("PLAN_BIM_ORDERS_PER_USER_DAY", "1")
    monkeypatch.setenv("PLAN_BIM_ORDERS_PER_IP_DAY", "10")
    guard = OperationalGuard(secret="test-secret", clock=lambda: 1000.0)

    guard.check_upload(owner_id="user-1", client_ip="203.0.113.8")
    guard.check_order(owner_id="user-1", client_ip="203.0.113.8")
    with pytest.raises(RateLimitExceeded):
        guard.check_order(owner_id="user-1", client_ip="203.0.113.8")


def test_durable_store_receives_only_hashed_identity(monkeypatch):
    monkeypatch.setenv("PLAN_BIM_UPLOADS_PER_IP_HOUR", "1")
    store = FakeDurableStore()
    guard = OperationalGuard(secret="test-secret", durable_store=store)

    fingerprint = guard.check_upload(owner_id="", client_ip="203.0.113.8")
    assert store.calls[0]["identity_hash"] == fingerprint
    assert "203.0.113.8" not in repr(store.calls)

    with pytest.raises(RateLimitExceeded) as caught:
        guard.check_upload(owner_id="", client_ip="203.0.113.8")
    assert caught.value.retry_after_seconds == 77


def test_upload_endpoint_returns_429_and_retry_after(monkeypatch):
    class RejectingGuard:
        def check_upload(self, **_kwargs):
            raise RateLimitExceeded("Limite de teste atingido.", 123)

    monkeypatch.setattr(free_app, "OPERATIONAL_GUARD", RejectingGuard())
    response = free_app.app.test_client().post(
        "/api/plan-to-bim",
        data={"file": (io.BytesIO(b"not-read"), "test.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "123"
    assert response.get_json() == {
        "error": "Limite de teste atingido.",
        "retry_after_seconds": 123,
    }
