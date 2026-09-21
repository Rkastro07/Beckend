from __future__ import annotations

import io

import plan_to_bim_free_app as free_app
from plantatobim.assisted_order import AssistedOrderManager
from plantatobim.supabase_auth import AuthUser, SupabaseAuth


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeSession:
    def get(self, _url, *, headers, timeout):
        assert timeout == 10
        token = headers["Authorization"].removeprefix("Bearer ")
        return FakeResponse({
            "id": f"user-{token}",
            "email": f"{token}@example.com",
            "user_metadata": {"full_name": f"Pessoa {token}"},
        })


class FakeGateway:
    configured = True
    sandbox = True


class FakeMailer:
    configured = True


def configured_auth():
    return SupabaseAuth(
        url="https://project.supabase.co",
        secret_key="server-secret",
        session_secret="cookie-secret",
        session=FakeSession(),
    )


def test_verified_supabase_identity_becomes_signed_cookie():
    auth = configured_auth()
    user = auth.verify_access_token("alice")
    assert user == AuthUser(
        id="user-alice",
        email="alice@example.com",
        name="Pessoa alice",
    )
    cookie = auth.issue_cookie(user)
    assert auth.read_cookie(cookie) == user
    assert auth.read_cookie(f"{cookie}tampered") is None


def test_projects_are_bound_to_logged_in_owner(monkeypatch, tmp_path):
    auth = configured_auth()
    orders = AssistedOrderManager(
        tmp_path / "orders",
        payment_gateway=FakeGateway(),
        mailer=FakeMailer(),
    )
    monkeypatch.setattr(free_app, "AUTH", auth)
    monkeypatch.setattr(free_app, "ASSISTED_ORDER_MANAGER", orders)
    client = free_app.app.test_client()

    login = client.post("/api/auth/session", json={"access_token": "alice"})
    assert login.status_code == 200
    assert "HttpOnly" in login.headers["Set-Cookie"]
    assert "SameSite=Lax" in login.headers["Set-Cookie"]

    created = client.post(
        "/api/assisted-orders",
        data={
            "file": (io.BytesIO(b"%PDF-test"), "obra.pdf"),
            "area_m2": "100",
            "contact_email": "alice@example.com",
        },
        content_type="multipart/form-data",
    )
    assert created.status_code == 201
    job = created.get_json()["job"]

    listing = client.get("/api/projects")
    assert listing.status_code == 200
    assert [(item["job"], item["kind"]) for item in listing.get_json()["projects"]] == [
        (job, "assisted")
    ]

    stranger = free_app.app.test_client()
    assert stranger.post("/api/auth/session", json={"access_token": "bob"}).status_code == 200
    assert stranger.get("/api/projects").get_json()["projects"] == []
    denied = stranger.post(f"/api/projects/assisted/{job}/resume")
    assert denied.status_code == 404
