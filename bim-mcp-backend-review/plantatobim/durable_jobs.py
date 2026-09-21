"""Durable persistence for paid Plan2BIM jobs.

Job metadata lives in Supabase Postgres and private artifacts live in
Supabase Storage.  The browser never receives the service key or a Storage
object URL; every read goes through the authenticated Flask API.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import mimetypes
import os
from pathlib import Path
import threading
from typing import Any
from urllib import error, parse, request


DEFAULT_BUCKET = "plan2bim-jobs"
DEFAULT_TABLE = "plan_bim_jobs"


class DurableJobError(RuntimeError):
    """Raised when durable state or an artifact cannot be persisted."""


def _enabled(name: str, default: str = "false") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {
        "1", "true", "yes", "on",
    }


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


@dataclass
class SupabaseJobStore:
    project_url: str
    secret_key: str
    bucket: str = DEFAULT_BUCKET
    table: str = DEFAULT_TABLE
    timeout_seconds: float = 45.0

    def __post_init__(self) -> None:
        self.project_url = self.project_url.rstrip("/")
        self.bucket = self.bucket.strip() or DEFAULT_BUCKET
        self.table = self.table.strip() or DEFAULT_TABLE
        self._bucket_ready = False
        self._bucket_lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "SupabaseJobStore | None":
        if not _enabled("PLAN_BIM_DURABLE_JOBS_ENABLED"):
            return None
        project_url = os.environ.get("SUPABASE_URL", "").strip()
        secret_key = os.environ.get("SUPABASE_SECRET_KEY", "").strip()
        if not project_url or not secret_key:
            raise DurableJobError(
                "Persistência durável habilitada sem SUPABASE_URL/SUPABASE_SECRET_KEY."
            )
        if not project_url.startswith("https://"):
            raise DurableJobError("SUPABASE_URL precisa usar HTTPS.")
        return cls(
            project_url=project_url,
            secret_key=secret_key,
            bucket=os.environ.get("PLAN_BIM_JOBS_BUCKET", DEFAULT_BUCKET),
            table=os.environ.get("PLAN_BIM_JOBS_TABLE", DEFAULT_TABLE),
            timeout_seconds=float(
                os.environ.get("PLAN_BIM_DURABLE_TIMEOUT_SECONDS", "45")
            ),
        )

    def _call(
        self,
        method: str,
        endpoint: str,
        *,
        data: bytes | None = None,
        content_type: str | None = None,
        headers: dict[str, str] | None = None,
        expected: tuple[int, ...] = (200,),
    ) -> bytes:
        request_headers = {
            "apikey": self.secret_key,
            "Authorization": f"Bearer {self.secret_key}",
            "User-Agent": "Plan2BIM-CloudRun/2.0",
        }
        if content_type:
            request_headers["Content-Type"] = content_type
        if headers:
            request_headers.update(headers)
        req = request.Request(
            f"{self.project_url}{endpoint}",
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read()
                if response.status not in expected:
                    raise DurableJobError(
                        f"Supabase respondeu com HTTP {response.status}."
                    )
                return body
        except error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            raise DurableJobError(
                f"Supabase respondeu com HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise DurableJobError(
                f"Não foi possível alcançar o Supabase: {exc.reason}"
            ) from exc

    def ensure_private_bucket(self) -> None:
        if self._bucket_ready:
            return
        with self._bucket_lock:
            if self._bucket_ready:
                return
            bucket_id = parse.quote(self.bucket, safe="")
            try:
                payload = json.loads(
                    self._call("GET", f"/storage/v1/bucket/{bucket_id}") or b"{}"
                )
                if payload.get("public") is True:
                    raise DurableJobError(
                        f"O bucket {self.bucket!r} existe, mas está público."
                    )
            except DurableJobError as exc:
                if "HTTP 404" not in str(exc) and "NoSuchBucket" not in str(exc):
                    raise
                self._call(
                    "POST",
                    "/storage/v1/bucket",
                    data=_json_bytes({
                        "id": self.bucket,
                        "name": self.bucket,
                        "public": False,
                        "file_size_limit": 30 * 1024 * 1024,
                    }),
                    content_type="application/json",
                    expected=(200,),
                )
            self._bucket_ready = True

    def save(self, state: dict[str, Any]) -> None:
        owner = state.get("owner") or {}
        payment = state.get("payment") or {}
        row = {
            "job_id": str(state["job"]),
            "owner_id": str(owner.get("id") or ""),
            "owner_email": str(owner.get("email") or ""),
            "status": str(state.get("status") or ""),
            "external_reference": str(payment.get("external_reference") or "") or None,
            "state": deepcopy(state),
            "created_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
        }
        table = parse.quote(self.table, safe="")
        self._call(
            "POST",
            f"/rest/v1/{table}?on_conflict=job_id",
            data=_json_bytes(row),
            content_type="application/json",
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            expected=(200, 201, 204),
        )

    def load(self, job: str) -> dict[str, Any] | None:
        table = parse.quote(self.table, safe="")
        job_filter = parse.quote(f"eq.{job}", safe=".")
        body = self._call(
            "GET",
            f"/rest/v1/{table}?select=state&job_id={job_filter}&limit=1",
        )
        rows = json.loads(body or b"[]")
        return deepcopy(rows[0]["state"]) if rows else None

    def list_for_owner(self, owner_id: str) -> list[dict[str, Any]]:
        table = parse.quote(self.table, safe="")
        owner_filter = parse.quote(f"eq.{owner_id}", safe=".")
        body = self._call(
            "GET",
            f"/rest/v1/{table}?select=state&owner_id={owner_filter}"
            "&order=updated_at.desc&limit=200",
        )
        return [deepcopy(row["state"]) for row in json.loads(body or b"[]")]

    def find_by_external_reference(
        self, external_reference: str
    ) -> dict[str, Any] | None:
        table = parse.quote(self.table, safe="")
        reference_filter = parse.quote(f"eq.{external_reference}", safe=".")
        body = self._call(
            "GET",
            f"/rest/v1/{table}?select=state"
            f"&external_reference={reference_filter}&limit=1",
        )
        rows = json.loads(body or b"[]")
        return deepcopy(rows[0]["state"]) if rows else None

    def consume_rate_limit(
        self,
        *,
        scope: str,
        identity_hash: str,
        maximum: int,
        window_seconds: int,
    ) -> tuple[bool, int, int]:
        """Atomically consume one fixed-window allowance in Postgres."""
        body = self._call(
            "POST",
            "/rest/v1/rpc/consume_plan_bim_rate_limit",
            data=_json_bytes({
                "p_scope": str(scope),
                "p_identity_hash": str(identity_hash),
                "p_limit": max(1, int(maximum)),
                "p_window_seconds": max(1, int(window_seconds)),
            }),
            content_type="application/json",
            expected=(200,),
        )
        rows = json.loads(body or b"[]")
        if not rows:
            raise DurableJobError("Supabase não retornou o resultado do limite.")
        row = rows[0]
        return (
            bool(row.get("allowed")),
            max(1, int(row.get("retry_after_seconds") or 1)),
            max(0, int(row.get("current_count") or 0)),
        )

    def upload_bytes(
        self, object_path: str, payload: bytes, *, content_type: str
    ) -> str:
        self.ensure_private_bucket()
        bucket = parse.quote(self.bucket, safe="")
        path = parse.quote(object_path.strip("/"), safe="/")
        self._call(
            "POST",
            f"/storage/v1/object/{bucket}/{path}",
            data=payload,
            content_type=content_type,
            headers={"x-upsert": "true"},
            expected=(200,),
        )
        return object_path

    def upload_file(self, object_path: str, local_path: Path) -> str:
        content_type = mimetypes.guess_type(local_path.name)[0]
        return self.upload_bytes(
            object_path,
            local_path.read_bytes(),
            content_type=content_type or "application/octet-stream",
        )

    def download_bytes(self, object_path: str) -> bytes:
        self.ensure_private_bucket()
        bucket = parse.quote(self.bucket, safe="")
        path = parse.quote(object_path.strip("/"), safe="/")
        return self._call(
            "GET", f"/storage/v1/object/authenticated/{bucket}/{path}"
        )

    def persist_sources(
        self, job: str, original_path: Path, image_path: Path
    ) -> dict[str, str]:
        original_object = f"jobs/{job}/input/original{original_path.suffix.lower()}"
        image_object = f"jobs/{job}/input/page-1{image_path.suffix.lower()}"
        self.upload_file(original_object, original_path)
        if original_path.resolve() == image_path.resolve():
            image_object = original_object
        else:
            self.upload_file(image_object, image_path)
        return {
            "original_object": original_object,
            "image_object": image_object,
        }

    def materialize_sources(
        self, job: str, state: dict[str, Any], root_dir: Path
    ) -> dict[str, Any]:
        source = state.setdefault("source", {})
        job_dir = root_dir / job
        job_dir.mkdir(parents=True, exist_ok=True)
        mapping = (
            ("original_path", "original_object", "original"),
            ("image_path", "image_object", "page-1"),
        )
        for path_key, object_key, fallback_name in mapping:
            current = Path(str(source.get(path_key) or ""))
            if current.is_file():
                continue
            object_path = str(source.get(object_key) or "")
            if not object_path:
                raise DurableJobError(
                    f"Arquivo durável ausente para {path_key} na tarefa {job}."
                )
            suffix = Path(object_path).suffix or ".bin"
            local_path = job_dir / f"{fallback_name}{suffix}"
            local_path.write_bytes(self.download_bytes(object_path))
            source[path_key] = str(local_path)
        return state

    def save_result(
        self, job: str, editor_model: dict[str, Any], analysis: dict[str, Any]
    ) -> dict[str, str]:
        result_object = f"jobs/{job}/result/editor-model.json"
        analysis_object = f"jobs/{job}/result/analysis.json"
        self.upload_bytes(
            result_object,
            _json_bytes(editor_model),
            content_type="application/json; charset=utf-8",
        )
        self.upload_bytes(
            analysis_object,
            _json_bytes(analysis),
            content_type="application/json; charset=utf-8",
        )
        return {
            "result_object": result_object,
            "analysis_object": analysis_object,
        }

    def load_json(self, object_path: str) -> dict[str, Any]:
        return json.loads(self.download_bytes(object_path).decode("utf-8"))

    def delete_objects(self, object_paths: list[str]) -> None:
        prefixes = sorted({str(path).strip("/") for path in object_paths if path})
        if not prefixes:
            return
        self.ensure_private_bucket()
        bucket = parse.quote(self.bucket, safe="")
        self._call(
            "DELETE",
            f"/storage/v1/object/{bucket}",
            data=_json_bytes({"prefixes": prefixes}),
            content_type="application/json",
            expected=(200,),
        )

    def purge_expired_files(self, *, retention_days: int, limit: int = 100) -> dict[str, int]:
        """Delete private artifacts after the published retention period.

        The order row remains as minimal operational/payment history, while file
        paths and Storage object references are removed from its state.
        """
        days = max(1, int(retention_days))
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        table = parse.quote(self.table, safe="")
        # Query-string '+' is decoded as a space by PostgREST unless percent-
        # encoded, which would turn the UTC offset into an invalid timestamp.
        created_filter = parse.quote(f"lt.{cutoff.isoformat()}", safe=".:-")
        body = self._call(
            "GET",
            f"/rest/v1/{table}?select=state&created_at={created_filter}"
            f"&order=created_at.asc&limit={max(1, min(int(limit), 500))}",
        )
        states = [row.get("state") for row in json.loads(body or b"[]")]
        purged = 0
        skipped = 0
        for state in states:
            if not isinstance(state, dict) or state.get("files_purged_at"):
                skipped += 1
                continue
            source = state.get("source") if isinstance(state.get("source"), dict) else {}
            object_paths = [
                source.get("original_object"),
                source.get("image_object"),
                state.get("result_object"),
                state.get("analysis_object"),
            ]
            self.delete_objects([str(path) for path in object_paths if path])
            for key in ("original_path", "image_path", "original_object", "image_object"):
                source.pop(key, None)
            for key in (
                "result_path", "analysis_path", "result_object", "analysis_object",
            ):
                state.pop(key, None)
            state["source"] = source
            state["files_purged_at"] = datetime.now(timezone.utc).isoformat()
            state["updated_at"] = state["files_purged_at"]
            state.setdefault("retention", {})["files_available"] = False
            self.save(state)
            purged += 1
        return {"purged_jobs": purged, "skipped_jobs": skipped}
