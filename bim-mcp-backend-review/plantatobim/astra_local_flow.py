"""Paid Plan-to-BIM flow with local measurement and Astra-authored geometry.

Preflight may use a local detector to estimate scale and area for pricing. Its
coordinates are never supplied to Astra. After approved payment, Astra receives
the plan image directly and authors the complete editable geometry.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Callable

from PIL import Image

from .astra_direct_stage import AstraDirectPlanStage, DIRECT_PIPELINE_VERSION
from .cloud_tasks_queue import CloudTasksDispatcher
from .durable_jobs import SupabaseJobStore
from .mercadopago_checkout import MercadoPagoCheckout
from .astra_semantic_stage import (
    ASTRA_MODEL,
    estimate_astra_cost,
)


TERMINAL_STATUSES = {"completed", "failed"}
ACTIVE_STATUSES = {"queued", "running"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_canvas_width(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 1 <= number <= 500 else None


def _cost_reference(profile: dict[str, Any]) -> dict[str, Any]:
    """Quote from file/image metadata only; no geometry detector is involved."""
    page_count = max(1, int(profile.get("page_count") or 1))
    processed_pages = 1
    width_px = max(1, int(profile.get("image_width_px") or 1))
    height_px = max(1, int(profile.get("image_height_px") or 1))
    file_size_bytes = max(0, int(profile.get("file_size_bytes") or 0))
    megapixels = width_px * height_px / 1_000_000
    file_size_mb = file_size_bytes / (1024 * 1024)
    expected_usd = min(2.25, max(1.05, 1.00 + min(megapixels, 12) * 0.035))
    low_usd = max(0.65, expected_usd * 0.80)
    high_usd = min(3.25, expected_usd * 1.40)
    usd_brl = float(os.environ.get("ASTRA_TEST_USD_BRL", "5.50"))
    retry_reserve = float(os.environ.get("ASTRA_TEST_RETRY_RESERVE", "1.50"))
    fixed_cost_brl = float(os.environ.get("ASTRA_TEST_FIXED_COST_BRL", "2.50"))
    multiplier = float(os.environ.get("ASTRA_TEST_PRICE_MULTIPLIER", "3.0"))
    minimum_brl = float(os.environ.get("ASTRA_TEST_MIN_PRICE_BRL", "49.90"))
    protected_cost_brl = high_usd * usd_brl * retry_reserve + fixed_cost_brl
    calculated_brl = max(minimum_brl, protected_cost_brl * multiplier)
    customer_brl = max(minimum_brl, math.ceil((calculated_brl + 0.10) / 10) * 10 - 0.10)
    return {
        "currency": "BRL",
        "customer_price": round(customer_brl, 2),
        "pricing_version": "astra-direct-beta-v1",
        "payment_mode": "mercado-pago",
        "api_cost_estimate_usd": {
            "low": round(low_usd, 2),
            "expected": round(expected_usd, 2),
            "high": round(high_usd, 2),
        },
        "estimated_processing_seconds": {"min": 300, "max": 900},
        "basis": {
            "pricing_input": "document-metadata-only",
            "page_count": page_count,
            "processed_pages": processed_pages,
            "image_width_px": width_px,
            "image_height_px": height_px,
            "image_megapixels": round(megapixels, 2),
            "file_size_mb": round(file_size_mb, 2),
            "heuristic_detector_used": False,
            "pilot": (
                "Astra direct geometry beta; first page only; "
                "no detector candidates"
            ),
            "usd_brl_safety_rate": usd_brl,
            "retry_reserve": retry_reserve,
            "fixed_cost_brl": fixed_cost_brl,
            "price_multiplier": multiplier,
            "protected_cost_brl": round(protected_cost_brl, 2),
        },
        "disclaimer": "Preço calculado antes do início do processamento.",
    }


def quote_for_document(profile: dict[str, Any], inspection: dict | None = None) -> dict[str, Any]:
    """Show an area reference while freezing the launch price charged at checkout."""
    quote = _cost_reference(profile)
    applied = Decimal(os.environ.get("PLAN_BIM_MIN_PRICE_BRL", "59.90"))
    rate = Decimal(os.environ.get("PLAN_BIM_PRICE_PER_M2", "0.50"))
    if not applied.is_finite() or applied <= 0 or not rate.is_finite() or rate <= 0:
        raise RuntimeError("Configuração de preço inválida.")
    page_count = max(1, int(profile.get("page_count") or 1))
    price = applied.quantize(Decimal("0.01"), rounding="ROUND_CEILING")
    inspection = inspection if isinstance(inspection, dict) else {}
    area = None
    try:
        raw_area = inspection.get("estimated_area_m2")
        if raw_area is None and inspection.get("status") in {"estimated", "verified"}:
            raw_area = inspection.get("area_m2")
        candidate = Decimal(str(raw_area))
        if candidate.is_finite() and candidate > 0:
            area = candidate
    except (InvalidOperation, TypeError, ValueError):
        area = None
    calculated = (
        (area * rate).quantize(Decimal("0.01"), rounding="ROUND_CEILING")
        if area is not None
        else None
    )
    quote.update({
        "pricing_version": "launch-fixed-area-reference-v1",
        "pricing_mode": "launch-fixed",
        "page_count": page_count,
        "processed_pages": 1,
        "price_per_page": float(price),
        "minimum_price": float(price),
        "applied_price_brl": float(price),
        "price_per_m2": float(rate),
        "calculated_price_brl": float(calculated) if calculated is not None else None,
        "estimated_area_m2": float(area) if area is not None else None,
        "scale_source": inspection.get("scale_source"),
        "scale_confidence": inspection.get("scale_confidence"),
        "scope": (
            "Uma página por pedido. A área e o valor por m² são uma referência; "
            "nesta fase, a cobrança usa o valor fixo informado abaixo."
        ),
        "ready": True,
        "customer_price": float(price),
        "minimum_applied": True,
    })
    quote["basis"]["pricing_input"] = "local-area-reference-fixed-launch-price"
    quote["basis"]["measurement_only"] = True
    return quote


def _customer_quote(quote: dict[str, Any]) -> dict[str, Any]:
    """Return only information required for the customer's decision."""
    basis = quote.get("basis") if isinstance(quote.get("basis"), dict) else {}
    page_count = max(1, int(quote.get("page_count") or basis.get("page_count") or 1))
    processed_pages = max(
        1,
        int(quote.get("processed_pages") or basis.get("processed_pages") or 1),
    )
    price_per_page = quote.get("price_per_page")
    if price_per_page is None:
        price_per_page = quote.get("customer_price")
    return {
        "currency": quote.get("currency", "BRL"),
        "customer_price": quote.get("customer_price"),
        "page_count": page_count,
        "processed_pages": processed_pages,
        "price_per_page": deepcopy(price_per_page),
        **{
            key: deepcopy(quote.get(key))
            for key in (
                "minimum_price", "minimum_applied", "scope", "ready",
                "pricing_mode", "applied_price_brl", "price_per_m2",
                "calculated_price_brl", "estimated_area_m2", "scale_source",
                "scale_confidence",
            )
        },
        "payment_mode": quote.get("payment_mode", "mercado-pago"),
        "estimated_processing_seconds": deepcopy(
            quote.get("estimated_processing_seconds") or {"min": 180, "max": 480}
        ),
        "disclaimer": "O processamento começa após a aprovação do pagamento.",
    }


def _customer_result(model: dict[str, Any]) -> dict[str, Any]:
    """Remove provider, pipeline and cost details from the browser payload."""
    result = deepcopy(model)
    result["engine"] = "pro-analysis-v1"
    source = result.get("source")
    if isinstance(source, dict):
        for key in ("astra_model", "geometry_source", "heuristic_detector_used",
                    "visual_protocol", "image_framing_used", "geometry_candidates_used"):
            source.pop(key, None)
        source["mode"] = "pro-analysis"
        source["semantic_level"] = "review-ready"
    reference = result.get("reference")
    if isinstance(reference, dict) and reference.get("kind") == "raster2seq":
        reference["engine"] = "pro-analysis"
        reference["label"] = "Planta de referência da Conversão Pro"
    editor = result.get("astra_editor")
    if isinstance(editor, dict):
        for key in ("pipeline_version", "geometry_source", "heuristic_detector_used"):
            editor.pop(key, None)
    result.pop("astra_direct", None)
    result.pop("astra_semantic", None)
    result.pop("gpt_plan", None)
    for element in [*(result.get("paredes") or []), *(result.get("aberturas") or [])]:
        if isinstance(element, dict) and str(element.get("origem") or "").startswith(
            "gpt-6-astra"
        ):
            element["origem"] = "conversao-pro"
    return result


class AstraLocalFlowManager:
    def __init__(
        self,
        root_dir: Path,
        *,
        enabled: bool | None = None,
        stage_factory: Callable[[], Any] = AstraDirectPlanStage,
        payment_gateway: Any | None = None,
        max_workers: int | None = None,
        inspector_factory: Callable[[], Any] | None = None,
        job_store: Any | None = None,
        dispatcher: Any | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = (
            str(os.environ.get("ASTRA_LOCAL_FLOW_ENABLED", "false")).lower()
            in {"1", "true", "yes", "on"}
            if enabled is None
            else enabled
        )
        self.job_store = job_store if job_store is not None else SupabaseJobStore.from_env()
        self.dispatcher = (
            dispatcher if dispatcher is not None else CloudTasksDispatcher.from_env()
        )
        workers = max_workers or int(os.environ.get("ASTRA_LOCAL_MAX_WORKERS", "2"))
        self._executor = (
            None
            if self.dispatcher is not None
            else ThreadPoolExecutor(max_workers=max(1, min(workers, 4)))
        )
        self._stage_factory = stage_factory
        # Kept as an optional compatibility argument for older callers. Pricing
        # no longer invokes a vision model or depends on a printed area.
        self._inspector_factory = inspector_factory
        self.payment_gateway = payment_gateway or MercadoPagoCheckout.from_env()
        self._lock = threading.RLock()
        self._inflight: set[str] = set()
        self._recover_orphaned_jobs()
        self._refresh_pending_quotes()

    def job_dir(self, job: str) -> Path:
        return self.root_dir / job

    def _state_path(self, job: str) -> Path:
        return self.job_dir(job) / "job.json"

    def _read(self, job: str) -> dict[str, Any]:
        if self.job_store is not None:
            state = self.job_store.load(job)
            if state is not None:
                self._write_local(state)
                return state
        path = self._state_path(job)
        if not path.is_file():
            raise LookupError("Tarefa não encontrada.")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_local(self, state: dict[str, Any]) -> None:
        path = self._state_path(state["job"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)

    def _write(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _now()
        if self.job_store is not None:
            self.job_store.save(state)
        self._write_local(state)

    def _recover_orphaned_jobs(self) -> None:
        """Fail jobs whose in-process executor disappeared after a local restart."""
        if self.job_store is not None and self.dispatcher is not None:
            # Cloud Tasks retries requests that lose their Cloud Run instance.
            return
        with self._lock:
            for path in self.root_dir.glob("*/job.json"):
                try:
                    state = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if state.get("status") not in {"queued", "running"}:
                    continue
                state.update({
                    "status": "failed",
                    "stage": "backend_restarted",
                    "error": (
                        "O backend local foi reiniciado durante o processamento. "
                        "A tarefa não será reenviada automaticamente; confirme uma nova "
                        "tentativa somente após verificar o consumo da API."
                    ),
                    "failed_at": _now(),
                    "retry_requires_confirmation": True,
                    "ifc_generated": False,
                })
                self._write(state)

    def _document_profile_from_state(self, state: dict[str, Any]) -> dict[str, Any] | None:
        existing = state.get("document")
        if isinstance(existing, dict) and existing.get("image_width_px"):
            return deepcopy(existing)
        source = state.get("source") or {}
        image_path = Path(str(source.get("image_path") or ""))
        original_path = Path(str(source.get("original_path") or ""))
        if not image_path.is_file():
            return None
        try:
            with Image.open(image_path) as image:
                width_px, height_px = image.size
        except Exception:
            return None
        page_count = 1
        render_path = image_path.parent / "pdf_render.json"
        if render_path.is_file():
            try:
                page_count = max(
                    1,
                    int(json.loads(render_path.read_text(encoding="utf-8")).get("page_count") or 1),
                )
            except (OSError, ValueError, TypeError):
                page_count = 1
        return {
            "page_count": page_count,
            "processed_pages": 1,
            "image_width_px": width_px,
            "image_height_px": height_px,
            "file_size_bytes": original_path.stat().st_size if original_path.is_file() else 0,
            "heuristic_detector_used": False,
        }

    @staticmethod
    def _preanalysis(
        profile: dict[str, Any],
        preparation_seconds: float = 0.0,
        inspection: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        inspection = inspection if isinstance(inspection, dict) else {}
        return {
            "page_count": max(1, int(profile.get("page_count") or 1)),
            "processed_pages": 1,
            "image_width_px": max(1, int(profile.get("image_width_px") or 1)),
            "image_height_px": max(1, int(profile.get("image_height_px") or 1)),
            "file_size_bytes": max(0, int(profile.get("file_size_bytes") or 0)),
            "preparation_seconds": round(float(preparation_seconds), 3),
            "measurement_completed": inspection.get("status") == "estimated",
            "estimated_area_m2": inspection.get("estimated_area_m2"),
            "scale_source": inspection.get("scale_source"),
            "scale_confidence": inspection.get("scale_confidence"),
            "geometry_detector_used": False,
            "message": (
                "Primeira página medida localmente para estimar escala e área. "
                "A geometria final será criada separadamente durante o processamento."
                if inspection.get("status") == "estimated"
                else "Primeira página preparada. A cobrança continua disponível pelo valor fixo."
            ),
        }

    def _refresh_pending_quotes(self) -> None:
        """Migrate unpaid local-simulation jobs to the provider-backed checkout."""
        with self._lock:
            for path in self.root_dir.glob("*/job.json"):
                try:
                    state = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if state.get("status") not in {"awaiting_confirmation", "awaiting_payment"}:
                    continue
                profile = self._document_profile_from_state(state)
                if profile is None:
                    continue
                legacy_payment = state.get("payment") or {}
                # A provider preference is a committed amount. Never reprice it on restart.
                if legacy_payment.get("preference_id") or legacy_payment.get("confirmed"):
                    continue
                if (state.get("quote") or {}).get("pricing_version") in {
                    "single-page-minimum-v1", "launch-fixed-area-reference-v1",
                }:
                    continue
                state["document"] = profile
                state["processing_mode"] = "astra-direct"
                state["status"] = "awaiting_payment"
                state["stage"] = "quote_ready"
                state["preanalysis"] = self._preanalysis(profile)
                state["quote"] = quote_for_document(profile, state.get("area_inspection"))
                if legacy_payment.get("mode") == "local-simulation":
                    state["payment"] = {
                        "mode": "mercado-pago",
                        "status": "not_started",
                        "confirmed": False,
                    }
                if (path.parent / "detector_model.json").is_file():
                    state["legacy_detector_ignored"] = True
                self._write(state)

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _authorized(self, job: str, token: str) -> dict[str, Any]:
        state = self._read(job)
        supplied = self._token_hash(str(token or ""))
        if not hmac.compare_digest(supplied, state.get("access_token_hash", "")):
            raise PermissionError("Tarefa não encontrada.")
        return state

    def create_job(
        self,
        *,
        job: str,
        original_name: str,
        original_path: Path,
        image_path: Path,
        document_profile: dict[str, Any],
        canvas_width_m: float,
        archive_consent: bool,
        preparation_seconds: float,
        owner_id: str = "",
        owner_email: str = "",
        client_ip_hash: str = "",
        area_inspection: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("O fluxo Astra local está desativado.")
        token = secrets.token_urlsafe(32)
        directory = self.job_dir(job)
        directory.mkdir(parents=True, exist_ok=True)
        profile = deepcopy(document_profile)
        profile["heuristic_detector_used"] = False
        inspection = deepcopy(area_inspection) if isinstance(area_inspection, dict) else {
            "status": "unavailable",
            "estimated_area_m2": None,
        }
        requested_canvas_width_m = canvas_width_m
        estimated_width = _finite_canvas_width(inspection.get("effective_canvas_width_m"))
        if estimated_width is not None and float(inspection.get("scale_confidence") or 0) >= 0.60:
            canvas_width_m = estimated_width
        (directory / "document_profile.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        durable_sources: dict[str, str] = {}
        if self.job_store is not None:
            durable_sources = self.job_store.persist_sources(
                job, Path(original_path), Path(image_path)
            )
        state = {
            "job": job,
            "status": "awaiting_payment",
            "stage": "quote_ready",
            "progress": 25,
            "processing_mode": "astra-direct",
            "created_at": _now(),
            "updated_at": _now(),
            "access_token_hash": self._token_hash(token),
            "owner": {
                "id": str(owner_id or "").strip(),
                "email": str(owner_email or "").strip().lower(),
            },
            "security": {
                "client_ip_hash": str(client_ip_hash or ""),
            },
            "source": {
                "original_name": original_name,
                "original_path": str(original_path),
                "image_path": str(image_path),
                "canvas_width_m": canvas_width_m,
                "requested_canvas_width_m": requested_canvas_width_m,
                "scale_source": inspection.get("scale_source"),
                "scale_confidence": inspection.get("scale_confidence"),
                "archive_consent": archive_consent,
                **durable_sources,
            },
            "document": profile,
            "preanalysis": self._preanalysis(profile, preparation_seconds, inspection),
            "area_inspection": inspection,
            "quote": quote_for_document(profile, inspection),
            "payment": {
                "mode": "mercado-pago",
                "status": "not_started",
                "confirmed": False,
            },
            "ifc_generated": False,
            "retention": {
                "file_days": max(
                    1, int(os.environ.get("PLAN_BIM_FILE_RETENTION_DAYS", "30"))
                ),
                "files_delete_after": (
                    datetime.now(timezone.utc)
                    + timedelta(
                        days=max(
                            1,
                            int(os.environ.get("PLAN_BIM_FILE_RETENTION_DAYS", "30")),
                        )
                    )
                ).isoformat(),
                "files_available": True,
            },
        }
        with self._lock:
            self._write(state)
        response = self.public_status(job, token, include_result=False)
        response["access_token"] = token
        return response

    def _states_for_owner(self, owner_id: str) -> list[dict[str, Any]]:
        owner = str(owner_id or "").strip()
        if not owner:
            return []
        if self.job_store is not None:
            return self.job_store.list_for_owner(owner)
        states: list[dict[str, Any]] = []
        for path in self.root_dir.glob("*/job.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str((state.get("owner") or {}).get("id") or "") == owner:
                states.append(state)
        return states

    def _active_for_owner(
        self, owner_id: str, *, exclude_job: str = ""
    ) -> dict[str, Any] | None:
        for state in self._states_for_owner(owner_id):
            if state.get("job") == exclude_job:
                continue
            if state.get("status") in ACTIVE_STATUSES:
                return state
        return None

    def assert_owner_idle(self, owner_id: str, *, exclude_job: str = "") -> None:
        if owner_id and self._active_for_owner(owner_id, exclude_job=exclude_job):
            raise ValueError(
                "Você já possui uma Conversão Pro em andamento. Aguarde a conclusão antes de iniciar outra."
            )

    def public_status(
        self, job: str, token: str, *, include_result: bool = True
    ) -> dict[str, Any]:
        with self._lock:
            state = self._authorized(job, token)
        return self._public_status_from_state(state, include_result=include_result)

    def _public_status_from_state(
        self, state: dict[str, Any], *, include_result: bool = True
    ) -> dict[str, Any]:
        public = {
            key: deepcopy(value)
            for key, value in state.items()
            if key not in {
                "api",
                "access_token_hash",
                "document",
                "legacy_detector_ignored",
                "processing_mode",
                "source",
                "result_path",
                "result_object",
                "review_path",
                "analysis_path",
                "analysis_object",
                "queue",
                "payment",
                "area_inspection",
                "owner",
            }
        }
        public["stage"] = {
            "astra_direct_analysis": "analysis",
            "astra_semantic_review": "analysis",
            "validating_astra_model": "validation",
            "validating_semantics": "validation",
        }.get(str(state.get("stage") or ""), state.get("stage"))
        if isinstance(state.get("quote"), dict):
            public["quote"] = _customer_quote(state["quote"])
            public["quote"]["checkout_available"] = bool(
                self.payment_gateway.configured and
                (state["quote"].get("ready") or (state.get("payment") or {}).get("preference_id"))
            )
            public["quote"]["test_payment"] = bool(
                self.payment_gateway.sandbox
            )
        payment = state.get("payment") or {}
        public["payment"] = {
            "status": str(payment.get("status") or "not_started"),
            "test": bool(payment.get("sandbox", self.payment_gateway.sandbox)),
        }
        retention = state.get("retention") or {}
        public["retention"] = {
            "file_days": int(retention.get("file_days") or 30),
            "files_delete_after": retention.get("files_delete_after"),
            "files_available": bool(
                retention.get("files_available", not state.get("files_purged_at"))
            ),
        }
        if isinstance(state.get("preanalysis"), dict):
            public["preanalysis"] = {
                "prepared": True,
                "page_count": int(state["preanalysis"].get("page_count") or 1),
                "message": str(state["preanalysis"].get("message") or "Primeira página preparada."),
                "processed_pages": 1,
            }
        if isinstance(state.get("semantic"), dict):
            public["analysis"] = {
                "summary": state["semantic"].get("summary") or "",
                "needs_human_review": bool(
                    state["semantic"].get("needs_human_review")
                ),
                "geometry": deepcopy(state["semantic"].get("geometry") or {}),
            }
            public.pop("semantic", None)
        if state.get("status") == "failed":
            public["error"] = (
                "Não foi possível concluir a Conversão Pro. "
                "Você pode iniciar uma nova tentativa sem pagar novamente."
            )
        public["ok"] = state["status"] != "failed"
        if (
            include_result
            and state["status"] == "completed"
            and not state.get("files_purged_at")
        ):
            result_path = Path(str(state.get("result_path") or ""))
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            elif self.job_store is not None and state.get("result_object"):
                result = self.job_store.load_json(str(state["result_object"]))
            else:
                raise LookupError("Resultado da tarefa não está disponível.")
            public["result"] = _customer_result(result)
        return public

    def public_status_for_owner(
        self, job: str, owner_id: str, *, include_result: bool = True
    ) -> dict[str, Any]:
        with self._lock:
            state = self._read(job)
            if str((state.get("owner") or {}).get("id") or "") != str(owner_id or ""):
                raise PermissionError("Tarefa não encontrada.")
        return self._public_status_from_state(state, include_result=include_result)

    def list_for_owner(self, owner_id: str) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        with self._lock:
            if self.job_store is not None:
                states = self.job_store.list_for_owner(str(owner_id or ""))
            else:
                states = []
                for path in self.root_dir.glob("*/job.json"):
                    try:
                        states.append(json.loads(path.read_text(encoding="utf-8")))
                    except (OSError, ValueError):
                        continue
            for state in states:
                if str((state.get("owner") or {}).get("id") or "") != str(owner_id or ""):
                    continue
                public = self._public_status_from_state(state, include_result=False)
                projects.append({
                    **public,
                    "kind": "astra",
                    "source_name": str((state.get("source") or {}).get("original_name") or "Planta"),
                })
        return projects

    def resume_for_owner(self, job: str, owner_id: str) -> dict[str, Any]:
        """Issue a new browser token after account ownership is verified."""
        token = secrets.token_urlsafe(32)
        with self._lock:
            state = self._read(job)
            if str((state.get("owner") or {}).get("id") or "") != str(owner_id or ""):
                raise PermissionError("Tarefa não encontrada.")
            state["access_token_hash"] = self._token_hash(token)
            self._write(state)
        response = self._public_status_from_state(state, include_result=True)
        response["access_token"] = token
        response["source_name"] = str((state.get("source") or {}).get("original_name") or "Planta")
        return response

    @staticmethod
    def _payment_matches_quote(
        state: dict[str, Any], payment: dict[str, Any]
    ) -> bool:
        try:
            expected = Decimal(str((state.get("quote") or {}).get("customer_price"))).quantize(
                Decimal("0.01")
            )
            received = Decimal(str(payment.get("transaction_amount"))).quantize(
                Decimal("0.01")
            )
        except (InvalidOperation, TypeError, ValueError):
            return False
        return (
            received == expected
            and str(payment.get("currency_id") or "").upper() == "BRL"
        )

    def _state_for_external_reference(
        self, external_reference: str
    ) -> dict[str, Any] | None:
        if self.job_store is not None:
            return self.job_store.find_by_external_reference(external_reference)
        for path in self.root_dir.glob("*/job.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            payment = state.get("payment") or {}
            if hmac.compare_digest(
                str(payment.get("external_reference") or ""),
                str(external_reference or ""),
            ):
                return state
        return None

    def _enqueue(self, job: str) -> None:
        if self.dispatcher is not None:
            state = self._read(job)
            task_name = self.dispatcher.enqueue(
                job, attempt=int(state.get("retry_count") or 0)
            )
            state["queue"] = {
                "provider": "cloud-tasks",
                "task_name": task_name,
                "enqueued_at": _now(),
            }
            self._write(state)
            return
        if job not in self._inflight:
            self._inflight.add(job)
            if self._executor is None:
                raise RuntimeError("Executor local indisponível.")
            self._executor.submit(self._run, job)

    def create_checkout(self, job: str, token: str) -> dict[str, Any]:
        """Create one hosted checkout preference for an unpaid job."""
        if not self.enabled:
            raise RuntimeError("A Conversão Pro está desativada.")
        if not self.payment_gateway.configured:
            raise RuntimeError("O pagamento ainda não está configurado no servidor.")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("O processamento ainda não está configurado no servidor.")
        with self._lock:
            state = self._authorized(job, token)
            if state.get("status") != "awaiting_payment":
                if state.get("status") in {"queued", "running", "completed"}:
                    return self.public_status(job, token, include_result=True)
                raise ValueError("Esta tarefa não pode iniciar um novo pagamento.")
            payment = deepcopy(state.get("payment") or {})
            if payment.get("checkout_url") and payment.get("preference_id"):
                response = self.public_status(job, token, include_result=False)
                response["checkout_url"] = payment["checkout_url"]
                return response
            if not (state.get("quote") or {}).get("ready"):
                raise ValueError("O orçamento ainda não está disponível.")
            self.assert_owner_idle(
                str((state.get("owner") or {}).get("id") or ""),
                exclude_job=job,
            )
            external_reference = str(
                payment.get("external_reference")
                or f"plan2bim-{job}-{secrets.token_hex(8)}"
            )
            source_name = Path(
                str((state.get("source") or {}).get("original_name") or "planta")
            ).stem
            checkout = self.payment_gateway.create_preference(
                job=job,
                external_reference=external_reference,
                title=f"Conversão Pro - {source_name}",
                amount=(state.get("quote") or {}).get("customer_price"),
            )
            state["payment"] = {
                "mode": "mercado-pago",
                "status": "checkout_created",
                "confirmed": False,
                "external_reference": external_reference,
                "preference_id": checkout["preference_id"],
                "checkout_url": checkout["checkout_url"],
                "sandbox": bool(checkout.get("sandbox")),
                "created_at": _now(),
                "accepted_quote": deepcopy(state["quote"]),
            }
            state["stage"] = "awaiting_payment"
            self._write(state)
        response = self.public_status(job, token, include_result=False)
        response["checkout_url"] = checkout["checkout_url"]
        return response

    def _apply_provider_payment(self, payment: dict[str, Any]) -> str | None:
        external_reference = str(payment.get("external_reference") or "").strip()
        if not external_reference:
            return None
        with self._lock:
            state = self._state_for_external_reference(external_reference)
            if state is None:
                return None
            job = str(state["job"])
            current = deepcopy(state.get("payment") or {})
            provider_status = str(payment.get("status") or "").lower()
            current.update({
                "provider_status": provider_status or "unknown",
                "last_checked_at": _now(),
            })
            payment_id = str(payment.get("id") or "").strip()
            if payment_id:
                current["provider_payment_id"] = payment_id
            if provider_status == "approved":
                if not self._payment_matches_quote(state, payment):
                    current["status"] = "review_required"
                    state["payment"] = current
                    state["stage"] = "payment_review_required"
                    self._write(state)
                    return job
                current.update({
                    "status": "approved",
                    "confirmed": True,
                    "confirmed_at": current.get("confirmed_at") or _now(),
                })
                state["payment"] = current
                if state.get("status") == "awaiting_payment":
                    owner_id = str((state.get("owner") or {}).get("id") or "")
                    blocked = self._active_for_owner(owner_id, exclude_job=job)
                    state["status"] = "waiting_for_slot" if blocked else "queued"
                    state["stage"] = (
                        "waiting_for_previous_job" if blocked else "payment_approved"
                    )
                    state["progress"] = 30
                    self._write(state)
                    if not blocked:
                        self._enqueue(job)
                else:
                    self._write(state)
                return job
            if state.get("status") == "awaiting_payment":
                if provider_status in {"pending", "in_process", "authorized"}:
                    current["status"] = "pending"
                    state["stage"] = "payment_pending"
                elif provider_status in {"rejected", "cancelled", "refunded", "charged_back"}:
                    current["status"] = "not_approved"
                    state["stage"] = "payment_not_approved"
                state["payment"] = current
                self._write(state)
            return job

    def sync_payment(self, job: str, token: str) -> dict[str, Any]:
        """Reconcile a local checkout by querying Mercado Pago server-to-server."""
        with self._lock:
            state = self._authorized(job, token)
            if state.get("status") != "awaiting_payment":
                if (
                    state.get("status") == "queued"
                    and self.dispatcher is not None
                    and not (state.get("queue") or {}).get("task_name")
                ):
                    self._enqueue(job)
                return self.public_status(job, token, include_result=True)
            external_reference = str(
                (state.get("payment") or {}).get("external_reference") or ""
            )
        if not external_reference:
            raise ValueError("Inicie o pagamento antes de consultar a aprovação.")
        payments = self.payment_gateway.search_payments(external_reference)
        matching = [
            item
            for item in payments
            if str(item.get("external_reference") or "") == external_reference
        ]
        approved = next(
            (item for item in matching if str(item.get("status") or "").lower() == "approved"),
            None,
        )
        selected = approved or (matching[0] if matching else None)
        if selected is not None:
            self._apply_provider_payment(selected)
        return self.public_status(job, token, include_result=True)

    def handle_payment_notification(self, payment_id: str) -> bool:
        """Fetch and apply a provider payment referenced by a verified webhook."""
        payment = self.payment_gateway.get_payment(payment_id)
        return self._apply_provider_payment(payment) is not None

    def retry(self, job: str, token: str) -> dict[str, Any]:
        """Retry a failed paid conversion without charging the customer again."""
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("O processamento ainda não está configurado no servidor.")
        with self._lock:
            state = self._authorized(job, token)
            if state.get("status") in {"queued", "running", "completed"}:
                return self.public_status(job, token, include_result=True)
            if state.get("status") != "failed":
                raise ValueError("Esta tarefa não está disponível para nova tentativa.")
            if (state.get("payment") or {}).get("status") != "approved":
                raise ValueError("O pagamento desta tarefa ainda não foi aprovado.")
            self.assert_owner_idle(
                str((state.get("owner") or {}).get("id") or ""),
                exclude_job=job,
            )
            state.pop("error", None)
            state.pop("failed_at", None)
            state["status"] = "queued"
            state["stage"] = "retry_confirmed"
            state["progress"] = 30
            state["retry_count"] = int(state.get("retry_count") or 0) + 1
            state.pop("queue", None)
            self._write(state)
            self._enqueue(job)
        return self.public_status(job, token, include_result=True)

    def _update(self, job: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            state = self._read(job)
            state.update(changes)
            self._write(state)
            return state

    def _promote_next_for_owner(self, owner_id: str, *, finished_job: str) -> None:
        if not owner_id or self._active_for_owner(owner_id, exclude_job=finished_job):
            return
        waiting = [
            state
            for state in self._states_for_owner(owner_id)
            if state.get("status") == "waiting_for_slot"
            and (state.get("payment") or {}).get("status") == "approved"
        ]
        if not waiting:
            return
        next_state = sorted(
            waiting, key=lambda item: str(item.get("created_at") or "")
        )[0]
        next_state["status"] = "queued"
        next_state["stage"] = "previous_job_completed"
        next_state["progress"] = 30
        self._write(next_state)
        self._enqueue(str(next_state["job"]))

    def purge_expired_files(self) -> dict[str, int]:
        if self.job_store is None:
            return {"purged_jobs": 0, "skipped_jobs": 0}
        return self.job_store.purge_expired_files(
            retention_days=max(
                1, int(os.environ.get("PLAN_BIM_FILE_RETENTION_DAYS", "30"))
            )
        )

    def run_worker(self, job: str) -> dict[str, Any]:
        """Run one already-paid job from a trusted Cloud Tasks request."""
        with self._lock:
            state = self._read(job)
            if state.get("status") == "completed":
                return state
            if (state.get("payment") or {}).get("status") != "approved":
                raise PermissionError("A tarefa ainda não possui pagamento aprovado.")
            if state.get("status") not in {"queued", "running"}:
                raise ValueError("A tarefa não está disponível para processamento.")
        self._run(job)
        return self._read(job)

    def verify_worker_request(self, authorization: str) -> None:
        if self.dispatcher is None:
            raise PermissionError("Worker durável não configurado.")
        self.dispatcher.verify_request(authorization)

    def _run(self, job: str) -> bool:
        started = time.perf_counter()
        owner_id = ""
        try:
            with self._lock:
                state = self._read(job)
                owner_id = str((state.get("owner") or {}).get("id") or "")
                if state.get("status") == "completed":
                    return True
                if self.job_store is not None:
                    state = self.job_store.materialize_sources(
                        job, state, self.root_dir
                    )
                state.update({
                    "status": "running",
                    "stage": "astra_direct_analysis",
                    "progress": 45,
                    "started_at": _now(),
                })
                self._write(state)
            source = state.get("source") or {}
            stage = self._stage_factory()
            api_started = time.perf_counter()
            editor_model, analysis, metadata = stage.analyze(
                Path(str(source.get("image_path") or "")),
                canvas_width_m=float(source.get("canvas_width_m") or 0),
                original_name=str(source.get("original_name") or "planta.png"),
                user_message=(
                    "Crie diretamente toda a geometria editável desta planta. "
                    "Não use detector heurístico e não gere IFC."
                ),
            )
            api_seconds = metadata.get("api_seconds", round(time.perf_counter() - api_started, 3))
            returned_model = str(
                metadata.get("model") or analysis.get("_model") or ""
            )
            if not returned_model.startswith(ASTRA_MODEL):
                raise RuntimeError(
                    f"Era esperado {ASTRA_MODEL}, mas a API devolveu "
                    f"{returned_model or 'modelo desconhecido'}."
                )
            self._update(job, stage="validating_astra_model", progress=88)
            if not editor_model.get("paredes"):
                raise RuntimeError("O modelo Astra não contém paredes editáveis.")
            editor_model["job"] = job
            editor_model["engine"] = DIRECT_PIPELINE_VERSION
            editor_model["archive_consent"] = bool(
                source.get("archive_consent")
            )
            result_path = self.job_dir(job) / "editor_model_astra_direct.json"
            analysis_path = self.job_dir(job) / "astra_direct_analysis.json"
            result_path.write_text(
                json.dumps(editor_model, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            analysis_path.write_text(
                json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            durable_results: dict[str, str] = {}
            if self.job_store is not None:
                durable_results = self.job_store.save_result(
                    job, editor_model, analysis
                )
            usage = metadata.get("usage") or {}
            astra_editor = editor_model.get("astra_editor") or {}
            self._update(
                job,
                status="completed",
                stage="editor_ready",
                progress=100,
                completed_at=_now(),
                result_path=str(result_path),
                analysis_path=str(analysis_path),
                **durable_results,
                semantic={
                    "model": returned_model,
                    "geometry_source": "astra-only",
                    "heuristic_detector_used": False,
                    "needs_human_review": bool(
                        astra_editor.get("needs_human_review")
                    ),
                    "summary": analysis.get("message") or "",
                    "unresolved": analysis.get("unresolved") or [],
                    "geometry": {
                        "walls": len(editor_model.get("paredes") or []),
                        "openings": len(editor_model.get("aberturas") or []),
                    },
                },
                api={
                    "response_id": metadata.get("response_id"),
                    "seconds": api_seconds,
                    "visual_preparation_seconds": metadata.get("visual_preparation_seconds"),
                    "adaptation_seconds": metadata.get("adaptation_seconds"),
                    "reasoning_effort": metadata.get("reasoning_effort"),
                    "image_count": metadata.get("image_count"),
                    "usage": usage,
                    "cost": estimate_astra_cost(usage),
                },
                total_seconds=round(time.perf_counter() - started, 3),
                ifc_generated=False,
            )
            return True
        except Exception as exc:
            self._update(
                job,
                status="failed",
                stage="failed",
                error=str(exc)[:1000],
                failed_at=_now(),
                total_seconds=round(time.perf_counter() - started, 3),
                ifc_generated=False,
            )
            return False
        finally:
            with self._lock:
                self._inflight.discard(job)
                self._promote_next_for_owner(owner_id, finished_job=job)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
