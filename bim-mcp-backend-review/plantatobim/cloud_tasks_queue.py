"""Cloud Tasks dispatcher used by the paid Astra worker."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import os
from typing import Any

class CloudTasksError(RuntimeError):
    """Raised when a paid job cannot be placed on the durable queue."""


@dataclass(frozen=True)
class CloudTasksDispatcher:
    project: str
    location: str
    queue: str
    worker_base_url: str
    service_account_email: str

    @classmethod
    def from_env(cls) -> "CloudTasksDispatcher | None":
        queue = os.environ.get("PLAN_BIM_CLOUD_TASKS_QUEUE", "").strip()
        if not queue:
            return None
        project = (
            os.environ.get("PLAN_BIM_CLOUD_TASKS_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or ""
        ).strip()
        location = os.environ.get(
            "PLAN_BIM_CLOUD_TASKS_LOCATION", "us-central1"
        ).strip()
        base_url = (
            os.environ.get("PLAN_BIM_WORKER_BASE_URL")
            or os.environ.get("MERCADOPAGO_PUBLIC_BACKEND_URL")
            or ""
        ).strip().rstrip("/")
        service_account = os.environ.get(
            "PLAN_BIM_CLOUD_TASKS_SERVICE_ACCOUNT", ""
        ).strip()
        if not all((project, location, base_url, service_account)):
            raise CloudTasksError(
                "Cloud Tasks habilitado com configuração incompleta."
            )
        return cls(project, location, queue, base_url, service_account)

    @property
    def audience(self) -> str:
        return self.worker_base_url

    def enqueue(self, job: str, attempt: int = 0) -> str:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        session = AuthorizedSession(credentials)
        parent = (
            f"projects/{self.project}/locations/{self.location}/queues/{self.queue}"
        )
        task_name = f"{parent}/tasks/astra-{job}-{max(0, int(attempt))}"
        payload = base64.b64encode(
            json.dumps({"job": job, "attempt": int(attempt)}).encode("utf-8")
        ).decode("ascii")
        task: dict[str, Any] = {
            "name": task_name,
            "dispatchDeadline": "1800s",
            "httpRequest": {
                "httpMethod": "POST",
                "url": f"{self.worker_base_url}/api/internal/astra-flow/jobs/{job}/run",
                "headers": {"Content-Type": "application/json"},
                "body": payload,
                "oidcToken": {
                    "serviceAccountEmail": self.service_account_email,
                    "audience": self.audience,
                },
            },
        }
        response = session.post(
            f"https://cloudtasks.googleapis.com/v2/{parent}/tasks",
            json={"task": task},
            timeout=45,
        )
        if response.status_code == 409:
            return task_name
        if response.status_code not in {200, 201}:
            raise CloudTasksError(
                f"Cloud Tasks respondeu com HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        return str((response.json() or {}).get("name") or task_name)

    def verify_request(self, authorization: str) -> dict[str, Any]:
        from google.auth.transport.requests import Request
        from google.oauth2 import id_token

        scheme, _, token = str(authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise PermissionError("Autenticação do worker ausente.")
        claims = id_token.verify_oauth2_token(token, Request(), self.audience)
        email = str(claims.get("email") or "")
        if not claims.get("email_verified") or email != self.service_account_email:
            raise PermissionError("Identidade do worker inválida.")
        return claims
