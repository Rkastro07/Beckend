"""Supabase Google sign-in with a backend-owned, HTTP-only session cookie.

The Supabase secret key never reaches the browser.  The browser completes the
provider redirect, sends the short-lived access token once, and receives an
signed cookie that only contains the minimum user identity needed to
link Plan2BIM jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any
from urllib.parse import urlencode, urlparse

import requests
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer


COOKIE_NAME = "plan2bim_session"


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str
    name: str = ""
    avatar_url: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "avatar_url": self.avatar_url,
        }


class SupabaseAuth:
    def __init__(
        self,
        *,
        url: str = "",
        secret_key: str = "",
        session_secret: str = "",
        session: Any | None = None,
        max_age_seconds: int = 60 * 60 * 24 * 30,
    ) -> None:
        self.url = str(url or "").strip().rstrip("/")
        self.secret_key = str(secret_key or "").strip()
        signing_secret = str(session_secret or self.secret_key).strip()
        self._serializer = (
            URLSafeTimedSerializer(signing_secret, salt="plan2bim-user-session-v1")
            if signing_secret
            else None
        )
        self._session = session or requests.Session()
        self.max_age_seconds = max(300, int(max_age_seconds))

    @classmethod
    def from_env(cls) -> "SupabaseAuth":
        return cls(
            url=os.environ.get("SUPABASE_URL", ""),
            secret_key=os.environ.get("SUPABASE_SECRET_KEY", ""),
            session_secret=os.environ.get("PLAN_BIM_SESSION_SECRET", ""),
            max_age_seconds=int(
                os.environ.get("PLAN_BIM_SESSION_MAX_AGE_SECONDS", "2592000")
            ),
        )

    @property
    def configured(self) -> bool:
        parsed = urlparse(self.url)
        return bool(
            parsed.scheme == "https"
            and parsed.hostname
            and self.secret_key
            and self._serializer
        )

    @property
    def required(self) -> bool:
        return _enabled(os.environ.get("PLAN_BIM_AUTH_REQUIRED"))

    def oauth_url(self, redirect_to: str) -> str:
        if not self.configured:
            raise RuntimeError("O login ainda não está configurado no servidor.")
        return f"{self.url}/auth/v1/authorize?{urlencode({'provider': 'google', 'redirect_to': redirect_to})}"

    def verify_access_token(self, access_token: str) -> AuthUser:
        token = str(access_token or "").strip()
        if not self.configured or not token:
            raise PermissionError("Sessão inválida.")
        try:
            response = self._session.get(
                f"{self.url}/auth/v1/user",
                headers={
                    "apikey": self.secret_key,
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                timeout=10,
            )
        except requests.RequestException as exc:
            raise RuntimeError("Não foi possível validar o login agora.") from exc
        if response.status_code != 200:
            raise PermissionError("Sessão inválida ou expirada.")
        try:
            payload = response.json()
        except ValueError as exc:
            raise PermissionError("Sessão inválida.") from exc
        metadata = payload.get("user_metadata") or {}
        user_id = str(payload.get("id") or "").strip()
        email = str(payload.get("email") or "").strip().lower()
        if not user_id or not email:
            raise PermissionError("A conta Google não devolveu uma identidade válida.")
        return AuthUser(
            id=user_id,
            email=email,
            name=str(metadata.get("full_name") or metadata.get("name") or "").strip(),
            avatar_url=str(metadata.get("avatar_url") or metadata.get("picture") or "").strip(),
        )

    def issue_cookie(self, user: AuthUser) -> str:
        if self._serializer is None:
            raise RuntimeError("O login ainda não está configurado no servidor.")
        return self._serializer.dumps(user.as_dict())

    def read_cookie(self, value: str | None) -> AuthUser | None:
        if self._serializer is None or not value:
            return None
        try:
            payload = self._serializer.loads(value, max_age=self.max_age_seconds)
        except (BadSignature, SignatureExpired):
            return None
        if not isinstance(payload, dict):
            return None
        user_id = str(payload.get("id") or "").strip()
        email = str(payload.get("email") or "").strip().lower()
        if not user_id or not email:
            return None
        return AuthUser(
            id=user_id,
            email=email,
            name=str(payload.get("name") or "").strip(),
            avatar_url=str(payload.get("avatar_url") or "").strip(),
        )


def cookie_options(*, secure: bool) -> dict[str, Any]:
    return {
        "httponly": True,
        "secure": bool(secure),
        "samesite": "Lax",
        "path": "/",
    }
