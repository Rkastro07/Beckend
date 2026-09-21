"""Private Supabase Storage archive for consented Plan-to-BIM samples.

The module deliberately uses the Storage HTTP API directly.  This keeps the
Cloud Run image small and, more importantly, ensures the server-only secret is
never needed by the browser or the Vercel application.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import mimetypes
import os
from pathlib import Path
import threading
from typing import Any
from urllib import error, parse, request


ARCHIVE_SCHEMA = "plan2bim.training-sample.v1"
CONSENT_VERSION = "2026-09-04.v1"
DEFAULT_BUCKET = "plan2bim-training"


class SupabaseArchiveError(RuntimeError):
    """Raised when a private archive operation cannot be completed."""


def load_local_env(path: Path) -> None:
    """Load a simple local .env file without overriding real environment vars."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _safe_extension(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf"}:
        return suffix
    return ".bin"


@dataclass(frozen=True)
class SupabaseTrainingArchive:
    project_url: str
    secret_key: str
    bucket: str = DEFAULT_BUCKET
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_url", self.project_url.rstrip("/"))
        object.__setattr__(self, "bucket", self.bucket.strip() or DEFAULT_BUCKET)
        object.__setattr__(self, "_bucket_ready", False)
        object.__setattr__(self, "_bucket_lock", threading.Lock())

    @classmethod
    def from_env(cls) -> SupabaseTrainingArchive | None:
        project_url = os.environ.get("SUPABASE_URL", "").strip()
        secret_key = os.environ.get("SUPABASE_SECRET_KEY", "").strip()
        bucket = os.environ.get("SUPABASE_STORAGE_BUCKET", DEFAULT_BUCKET).strip()
        if not project_url or not secret_key:
            return None
        if not project_url.startswith("https://"):
            raise SupabaseArchiveError("SUPABASE_URL precisa usar HTTPS.")
        if not secret_key.startswith(("sb_secret_", "eyJ")):
            raise SupabaseArchiveError("SUPABASE_SECRET_KEY possui formato inválido.")
        return cls(project_url=project_url, secret_key=secret_key, bucket=bucket)

    def _call(
        self,
        method: str,
        endpoint: str,
        *,
        data: bytes | None = None,
        content_type: str | None = None,
        extra_headers: dict[str, str] | None = None,
        expected: tuple[int, ...] = (200,),
    ) -> bytes:
        headers = {
            "apikey": self.secret_key,
            "User-Agent": "Plan2BIM-CloudRun/1.0",
        }
        if content_type:
            headers["Content-Type"] = content_type
        if extra_headers:
            headers.update(extra_headers)
        req = request.Request(
            f"{self.project_url}{endpoint}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read()
                if response.status not in expected:
                    raise SupabaseArchiveError(
                        f"Supabase Storage respondeu com HTTP {response.status}."
                    )
                return body
        except error.HTTPError as exc:
            detail = exc.read(512).decode("utf-8", errors="replace")
            raise SupabaseArchiveError(
                f"Supabase Storage respondeu com HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise SupabaseArchiveError(
                f"Não foi possível alcançar o Supabase Storage: {exc.reason}"
            ) from exc

    def ensure_private_bucket(self) -> None:
        if self._bucket_ready:
            return
        with self._bucket_lock:
            if self._bucket_ready:
                return
            bucket_id = parse.quote(self.bucket, safe="")
            try:
                body = self._call(
                    "GET",
                    f"/storage/v1/bucket/{bucket_id}",
                    expected=(200,),
                )
                bucket_data = json.loads(body or b"{}")
                if bucket_data.get("public") is True:
                    raise SupabaseArchiveError(
                        f"O bucket {self.bucket!r} existe, mas está público."
                    )
            except SupabaseArchiveError as exc:
                error_text = str(exc)
                if not any(
                    marker in error_text
                    for marker in ("HTTP 404", '"statusCode":"404"', "NoSuchBucket")
                ):
                    raise
                self._call(
                    "POST",
                    "/storage/v1/bucket",
                    data=_json_bytes({
                        "id": self.bucket,
                        "name": self.bucket,
                        "public": False,
                    }),
                    content_type="application/json",
                    expected=(200,),
                )
            object.__setattr__(self, "_bucket_ready", True)

    def upload_bytes(
        self,
        object_path: str,
        payload: bytes,
        *,
        content_type: str,
    ) -> str:
        self.ensure_private_bucket()
        encoded_bucket = parse.quote(self.bucket, safe="")
        encoded_path = parse.quote(object_path.strip("/"), safe="/")
        self._call(
            "POST",
            f"/storage/v1/object/{encoded_bucket}/{encoded_path}",
            data=payload,
            content_type=content_type,
            extra_headers={"x-upsert": "true"},
            expected=(200,),
        )
        return object_path

    def upload_file(self, object_path: str, local_path: Path) -> str:
        content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        return self.upload_bytes(
            object_path,
            local_path.read_bytes(),
            content_type=content_type,
        )

    def archive_initial(
        self,
        *,
        job: str,
        original_path: Path,
        rendered_image_path: Path,
        model: dict[str, Any],
        canvas_width_m: float,
        engine: str,
    ) -> dict[str, Any]:
        prefix = f"jobs/{job}"
        original_extension = _safe_extension(original_path)
        original_object = f"{prefix}/original{original_extension}"
        rendered_object = (
            f"{prefix}/rendered-page-1.png"
            if rendered_image_path.resolve() != original_path.resolve()
            else None
        )
        model_object = f"{prefix}/model-initial.json"
        consent_object = f"{prefix}/consent.json"
        accepted_at = utc_now()
        consent = {
            "accepted": True,
            "version": CONSENT_VERSION,
            "accepted_at": accepted_at,
        }
        self.ensure_private_bucket()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(self.upload_file, original_object, original_path),
                pool.submit(
                    self.upload_bytes,
                    model_object,
                    _json_bytes(model),
                    content_type="application/json; charset=utf-8",
                ),
                pool.submit(
                    self.upload_bytes,
                    consent_object,
                    _json_bytes(consent),
                    content_type="application/json; charset=utf-8",
                ),
            ]
            if rendered_object:
                futures.append(
                    pool.submit(self.upload_file, rendered_object, rendered_image_path)
                )
            for future in futures:
                future.result()
        manifest = {
            "schema": ARCHIVE_SCHEMA,
            "job": job,
            "status": "converted",
            "consent": consent,
            "pipeline": {
                "engine": engine,
                "canvas_width_m": canvas_width_m,
            },
            "objects": {
                "original": original_object,
                "rendered_image": rendered_object,
                "model_initial": model_object,
                "model_final": None,
                "ifc": None,
                "consent": consent_object,
            },
            "updated_at": utc_now(),
        }
        self.upload_bytes(
            f"{prefix}/manifest.json",
            _json_bytes(manifest),
            content_type="application/json; charset=utf-8",
        )
        return manifest

    def archive_final(
        self,
        *,
        job: str,
        ifc_path: Path,
        model: dict[str, Any],
        config: dict[str, Any],
        engine: str,
    ) -> dict[str, Any]:
        prefix = f"jobs/{job}"
        model_object = f"{prefix}/model-final.json"
        ifc_object = f"{prefix}/result.ifc"
        self.ensure_private_bucket()
        with ThreadPoolExecutor(max_workers=2) as pool:
            model_future = pool.submit(
                self.upload_bytes,
                model_object,
                _json_bytes({"modelo": model, "config": config}),
                content_type="application/json; charset=utf-8",
            )
            ifc_future = pool.submit(self.upload_file, ifc_object, ifc_path)
            model_future.result()
            ifc_future.result()
        manifest = {
            "schema": ARCHIVE_SCHEMA,
            "job": job,
            "status": "ifc-exported",
            "consent": {
                "accepted": True,
                "version": CONSENT_VERSION,
            },
            "pipeline": {"engine": engine},
            "objects": {
                "original": "stored-during-conversion",
                "rendered_image": "stored-when-source-is-pdf",
                "model_initial": f"{prefix}/model-initial.json",
                "model_final": model_object,
                "ifc": ifc_object,
                "consent": f"{prefix}/consent.json",
            },
            "updated_at": utc_now(),
        }
        self.upload_bytes(
            f"{prefix}/manifest.json",
            _json_bytes(manifest),
            content_type="application/json; charset=utf-8",
        )
        return manifest
