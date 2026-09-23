# -*- coding: utf-8 -*-
"""Plan-to-BIM service with a free path and an opt-in local Astra test flow."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
import traceback
import uuid
from urllib.parse import urlparse

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

from plantatobim.astra_local_flow import AstraLocalFlowManager
from plantatobim.sol_low_review_stage import SolLowReviewStage
from plantatobim.assisted_order import AssistedOrderManager
from plantatobim.mercadopago_checkout import MercadoPagoCheckout, MercadoPagoError
from plantatobim.operational_guard import OperationalGuard, RateLimitExceeded
from plantatobim.local_area_estimator import estimate_local_area
from plantatobim.supabase_auth import COOKIE_NAME, SupabaseAuth, cookie_options

from supabase_training_archive import (
    SupabaseArchiveError,
    SupabaseTrainingArchive,
    load_local_env,
)


RUNTIME_DIR = Path(tempfile.gettempdir())
OUTPUT_FOLDER = RUNTIME_DIR / "plan_to_bim_outputs"
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf")
ENGINE_NAME = "2d+yolo-walls+yolo-openings"
JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{10}$")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
load_local_env(Path(__file__).with_name(".env"))
LOW_COST_LOCAL_PILOT = str(os.environ.get("PLAN_BIM_SOL_LOW_PILOT", "false")).lower() in {
    "1", "true", "yes", "on"
}
FIRST_PREVIEW_ENABLED = str(os.environ.get("PLAN_BIM_FIRST_PREVIEW_ENABLED", "false")).lower() in {
    "1", "true", "yes", "on"
}
PAID_ONLY_FLOW = LOW_COST_LOCAL_PILOT or FIRST_PREVIEW_ENABLED
TRAINING_ARCHIVE = SupabaseTrainingArchive.from_env()
PILOT_PAYMENT_GATEWAY = None
if LOW_COST_LOCAL_PILOT:
    PILOT_PAYMENT_GATEWAY = MercadoPagoCheckout.from_env()
    # The generic credential may be live. Never use it for this sandbox-only pilot.
    PILOT_PAYMENT_GATEWAY.access_token = os.environ.get(
        "MERCADOPAGO_ACCESS_TOKEN_TEST", ""
    ).strip()
    PILOT_PAYMENT_GATEWAY.webhook_secret = os.environ.get(
        "MERCADOPAGO_WEBHOOK_SECRET_TEST", ""
    ).strip()
    # Local orders must not redirect to, or notify, the deployed production site.
    PILOT_PAYMENT_GATEWAY.public_frontend_url = None
    PILOT_PAYMENT_GATEWAY.public_backend_url = None
ASTRA_FLOW_MANAGER = AstraLocalFlowManager(
    OUTPUT_FOLDER / ("sol_low_pilot_jobs" if LOW_COST_LOCAL_PILOT else "astra_jobs"),
    first_preview_enabled=FIRST_PREVIEW_ENABLED,
    **({"sol_stage_factory": SolLowReviewStage, "low_cost_pilot": True,
        "payment_gateway": PILOT_PAYMENT_GATEWAY}
       if LOW_COST_LOCAL_PILOT else {}),
)
ASSISTED_ORDER_MANAGER = AssistedOrderManager(
    OUTPUT_FOLDER / "assisted_orders",
    payment_gateway=ASTRA_FLOW_MANAGER.payment_gateway,
)
AUTH = SupabaseAuth.from_env()
OPERATIONAL_GUARD = OperationalGuard(durable_store=ASTRA_FLOW_MANAGER.job_store)


def _valid_upload(file_storage) -> bool:
    filename = secure_filename(str(getattr(file_storage, "filename", "") or ""))
    return bool(filename) and Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def _form_number(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(request.form.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} precisa ser numérico.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} deve ficar entre {minimum:g} e {maximum:g}.")
    return value


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


ASSISTED_ORDER_ENABLED = _truthy(
    os.environ.get("PLAN_BIM_ASSISTED_ORDER_ENABLED", "true")
) and not LOW_COST_LOCAL_PILOT


def _safe_source_job(value) -> str | None:
    candidate = str(value or "").strip().lower()
    return candidate if JOB_ID_PATTERN.fullmatch(candidate) else None


def _current_user():
    return AUTH.read_cookie(request.cookies.get(COOKIE_NAME))


def _client_ip() -> str:
    """Best available client address behind Vercel and Cloud Run proxies."""
    for header in ("X-Real-IP", "X-Vercel-Forwarded-For", "X-Forwarded-For"):
        value = str(request.headers.get(header) or "").strip()
        if value:
            return value.split(",", 1)[0].strip()
    return str(request.remote_addr or "unknown")


def _rate_limited(exc: RateLimitExceeded):
    response = jsonify({
        "error": str(exc),
        "retry_after_seconds": exc.retry_after_seconds,
    })
    response.headers["Retry-After"] = str(exc.retry_after_seconds)
    return response, 429


def _allowed_auth_redirect(value: str) -> str:
    candidate = str(value or "").strip()
    parsed = urlparse(candidate)
    configured = str(os.environ.get("PLAN_BIM_PUBLIC_FRONTEND_URL") or "").strip().rstrip("/")
    configured_parsed = urlparse(configured) if configured else None
    configured_origin = (
        f"{configured_parsed.scheme}://{configured_parsed.netloc}"
        if configured_parsed and configured_parsed.scheme and configured_parsed.netloc
        else ""
    )
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""
    local = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}
    if parsed.path != "/auth/callback" or (not local and origin != configured_origin):
        raise ValueError("O retorno do login não é permitido.")
    return candidate


def _save_as_image(file_storage, job_dir: Path) -> tuple[Path, Path, str]:
    """Save an image or render the first page of a PDF to a PNG."""
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    original_name = secure_filename(file_storage.filename) or "planta.png"
    source_path = input_dir / original_name
    file_storage.save(str(source_path))
    if source_path.suffix.lower() != ".pdf":
        return source_path, source_path, original_name

    try:
        from plantatobim.pdf_raster import render_first_page

        image_path = input_dir / f"{source_path.stem}_pagina_1.png"
        diagnostic = render_first_page(source_path, image_path)
        (input_dir / "pdf_render.json").write_text(
            json.dumps(diagnostic, indent=2), encoding="utf-8"
        )
        return image_path, source_path, original_name
    except Exception as exc:
        raise ValueError("Não foi possível renderizar a primeira página do PDF.") from exc


def _save_assisted_order_upload(file_storage, order_dir: Path) -> tuple[Path, str]:
    """Keep the original private file intact for the paid human-assisted request."""
    input_dir = order_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    original_name = secure_filename(file_storage.filename) or "planta.pdf"
    source_path = input_dir / original_name
    file_storage.save(str(source_path))
    return source_path, original_name


def _astra_document_profile(image_path: Path, original_path: Path) -> dict:
    """Read document metadata for pricing without detecting any geometry."""
    try:
        with Image.open(image_path) as image:
            width_px, height_px = image.size
    except Exception as exc:
        raise ValueError("Não foi possível abrir a imagem preparada da planta.") from exc
    page_count = 1
    render_path = image_path.parent / "pdf_render.json"
    if render_path.is_file():
        try:
            page_count = max(
                1,
                int(json.loads(render_path.read_text(encoding="utf-8")).get("page_count") or 1),
            )
        except (OSError, TypeError, ValueError):
            page_count = 1
    return {
        "page_count": page_count,
        "processed_pages": 1,
        "image_width_px": width_px,
        "image_height_px": height_px,
        "file_size_bytes": original_path.stat().st_size,
        "heuristic_detector_used": False,
    }


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

    @app.after_request
    def _security_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/")
    @app.get("/api/health")
    def health():
        capabilities = ["editor-2d-3d", "ifc4", "dxf"]
        if not PAID_ONLY_FLOW:
            capabilities.insert(0, "plan-to-bim")
        if ASTRA_FLOW_MANAGER.enabled:
            capabilities.append("plan-to-bim-paid" if LOW_COST_LOCAL_PILOT else "plan-to-bim-pro")
        if ASSISTED_ORDER_ENABLED and ASSISTED_ORDER_MANAGER.payment_gateway.configured:
            capabilities.append("assisted-order")
        return jsonify({
            "status": "online",
            "service": "Plan-to-BIM Local Pilot" if LOW_COST_LOCAL_PILOT else "Plan-to-BIM Free",
            "version": "2.1.0",
            "capabilities": capabilities,
            "pro_flow": {
                "enabled": ASTRA_FLOW_MANAGER.enabled,
                "payment_mode": "mercado-pago",
                "payment_configured": bool(
                    ASTRA_FLOW_MANAGER.payment_gateway.configured
                ),
                "creates_ifc": False,
            },
            "assisted_order": {
                "enabled": ASSISTED_ORDER_ENABLED,
                "price_per_m2_brl": 0.50,
                "payment_configured": bool(
                    ASSISTED_ORDER_ENABLED
                    and ASSISTED_ORDER_MANAGER.payment_gateway.configured
                ),
                "delivery_configured": bool(
                    ASSISTED_ORDER_ENABLED and ASSISTED_ORDER_MANAGER.mailer.configured
                ),
            },
            "auth": {
                "configured": AUTH.configured,
                "provider": "google",
                "required_for_pro": AUTH.required or PAID_ONLY_FLOW,
            },
        })

    @app.get("/api/auth/google")
    def auth_google():
        try:
            redirect_to = _allowed_auth_redirect(str(request.args.get("redirect_to") or ""))
            return jsonify({"ok": True, "url": AUTH.oauth_url(redirect_to)})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503

    @app.post("/api/auth/session")
    def auth_session():
        body = request.get_json(silent=True) or {}
        try:
            user = AUTH.verify_access_token(str(body.get("access_token") or ""))
            response = jsonify({"ok": True, "user": user.as_dict()})
            response.set_cookie(
                COOKIE_NAME,
                AUTH.issue_cookie(user),
                max_age=AUTH.max_age_seconds,
                **cookie_options(
                    secure=(
                        request.is_secure
                        or request.headers.get("X-Forwarded-Proto", "").lower() == "https"
                        or str(os.environ.get("PLAN_BIM_PUBLIC_FRONTEND_URL") or "").startswith("https://")
                    )
                ),
            )
            return response
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 401
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503

    @app.get("/api/auth/me")
    def auth_me():
        user = _current_user()
        return jsonify({
            "ok": True,
            "configured": AUTH.configured,
            "authenticated": user is not None,
            "user": user.as_dict() if user else None,
        })

    @app.post("/api/auth/logout")
    def auth_logout():
        response = jsonify({"ok": True})
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    @app.get("/api/projects")
    def projects():
        user = _current_user()
        if user is None:
            return jsonify({"error": "Entre com o Google para ver seus projetos."}), 401
        items = [*ASTRA_FLOW_MANAGER.list_for_owner(user.id)]
        if ASSISTED_ORDER_ENABLED:
            items.extend(ASSISTED_ORDER_MANAGER.list_for_owner(user.id))
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return jsonify({"ok": True, "projects": items})

    @app.post("/api/projects/<kind>/<job>/resume")
    def resume_project(kind: str, job: str):
        user = _current_user()
        if user is None:
            return jsonify({"error": "Entre com o Google para retomar este projeto."}), 401
        if not JOB_ID_PATTERN.fullmatch(job):
            return jsonify({"error": "Projeto não encontrado."}), 404
        try:
            if kind == "astra":
                return jsonify(ASTRA_FLOW_MANAGER.resume_for_owner(job, user.id))
            if kind == "assisted" and ASSISTED_ORDER_ENABLED:
                return jsonify(ASSISTED_ORDER_MANAGER.resume_for_owner(job, user.id))
            return jsonify({"error": "Projeto não encontrado."}), 404
        except (LookupError, PermissionError):
            return jsonify({"error": "Projeto não encontrado."}), 404

    @app.get("/api/referencia/formatos")
    def formats():
        return jsonify({
            "ok": True,
            "entrada": list(SUPPORTED_EXTENSIONS),
            "pdf": "A primeira página é usada na conversão.",
        })

    @app.post("/api/plan-to-bim")
    @app.post("/api/referencia/pre-wall-yolo")
    def plan_to_bim():
        if PAID_ONLY_FLOW:
            return jsonify({"error": "Entre na sua conta para iniciar sua primeira prévia ou contratar uma nova conversão."}), 403
        image_file = request.files.get("file")
        if not image_file or not _valid_upload(image_file):
            return jsonify({"error": "Envie uma planta em PDF, PNG ou JPG."}), 400
        try:
            owner = _current_user()
            OPERATIONAL_GUARD.check_upload(
                owner_id=owner.id if owner else "",
                client_ip=_client_ip(),
            )
            canvas_width_m = _form_number("canvas_width_m", 20.0, 1.0, 500.0)
            metric_refinement = str(
                request.form.get("metric_refinement", "true")
            ).strip().lower() not in {"0", "false", "no"}
            archive_consent = _truthy(request.form.get("archive_consent"))

            sid = uuid.uuid4().hex[:10]
            job_dir = OUTPUT_FOLDER / f"plan_to_bim_{sid}"
            image_path, original_path, original_name = _save_as_image(image_file, job_dir)

            from plantatobim.pre_wall_opening_import import (
                PreWallOpeningError,
                pre_wall_image_to_editor_model,
            )

            try:
                model = pre_wall_image_to_editor_model(
                    image_path,
                    job_dir / "detector",
                    canvas_width_m=canvas_width_m,
                    metric_refinement=metric_refinement,
                )
            except PreWallOpeningError as exc:
                app.logger.error("Conversão %s falhou: %s", sid, exc)
                return jsonify({"error": str(exc)}), 422

            model["nome"] = Path(original_name).stem
            model["job"] = sid
            model["engine"] = ENGINE_NAME
            model["archive_consent"] = archive_consent
            if archive_consent and TRAINING_ARCHIVE is not None:
                try:
                    TRAINING_ARCHIVE.archive_initial(
                        job=sid,
                        original_path=original_path,
                        rendered_image_path=image_path,
                        model=model,
                        canvas_width_m=canvas_width_m,
                        engine=ENGINE_NAME,
                    )
                    model["archive_status"] = "saved"
                except SupabaseArchiveError as exc:
                    app.logger.warning("Falha ao arquivar conversão %s: %s", sid, exc)
                    model["archive_status"] = "failed"
            elif archive_consent:
                app.logger.warning(
                    "Arquivamento autorizado para %s, mas o Supabase não está configurado.",
                    sid,
                )
                model["archive_status"] = "unavailable"
            else:
                model["archive_status"] = "not-requested"
            return jsonify(model)
        except RateLimitExceeded as exc:
            return _rate_limited(exc)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Falha na conversão Plan-to-BIM: {exc}"}), 500

    @app.post("/api/astra-flow/preflight")
    def astra_preflight():
        if not ASTRA_FLOW_MANAGER.enabled:
            return jsonify({"error": "A Conversão Pro está indisponível."}), 404
        image_file = request.files.get("file")
        if not image_file or not _valid_upload(image_file):
            return jsonify({"error": "Envie uma planta em PDF, PNG ou JPG."}), 400
        try:
            owner = _current_user()
            if (AUTH.required or PAID_ONLY_FLOW) and owner is None:
                return jsonify({"error": "Entre com o Google para iniciar a Conversão Pro."}), 401
            ASTRA_FLOW_MANAGER.assert_owner_idle(owner.id if owner else "")
            client_ip_hash = OPERATIONAL_GUARD.check_upload(
                owner_id=owner.id if owner else "",
                client_ip=_client_ip(),
            )
            canvas_width_m = _form_number("canvas_width_m", 20.0, 1.0, 500.0)
            archive_consent = _truthy(request.form.get("archive_consent"))
            service_tier = str(request.form.get("service_tier") or "essential").strip().lower()
            if LOW_COST_LOCAL_PILOT and service_tier not in {"essential", "advanced"}:
                return jsonify({"error": "Escolha uma versão de conversão válida."}), 400
            if not LOW_COST_LOCAL_PILOT:
                service_tier = "advanced"
            if FIRST_PREVIEW_ENABLED:
                service_tier = "advanced"
            first_preview = bool(
                FIRST_PREVIEW_ENABLED and owner
                and not ASTRA_FLOW_MANAGER._states_for_owner(owner.id)
            )
            if first_preview:
                OPERATIONAL_GUARD.check_first_preview(client_ip=_client_ip())
            sid = uuid.uuid4().hex[:10]
            job_dir = ASTRA_FLOW_MANAGER.job_dir(sid)
            preparation_started = time.perf_counter()
            image_path, original_path, original_name = _save_as_image(image_file, job_dir)
            document_profile = _astra_document_profile(image_path, original_path)
            if LOW_COST_LOCAL_PILOT and service_tier == "essential":
                area_inspection = {
                    "status": "unavailable",
                    "estimated_area_m2": None,
                    "message": "Arquivo preparado. Preço fixo por uma página; análise após o pagamento.",
                }
            else:
                try:
                    area_inspection = estimate_local_area(
                        original_path=original_path,
                        image_path=image_path,
                        output_dir=job_dir / "preanalysis",
                        fallback_canvas_width_m=canvas_width_m,
                    )
                except Exception as exc:
                    app.logger.warning(
                        "Pré-análise local indisponível para %s: %s", sid, exc
                    )
                    area_inspection = {
                        "status": "unavailable",
                        "estimated_area_m2": None,
                        "message": (
                            "Não foi possível estimar a área com confiança; "
                            "o valor aplicado permanece fixo."
                        ),
                    }
            preparation_seconds = time.perf_counter() - preparation_started
            response = ASTRA_FLOW_MANAGER.create_job(
                job=sid,
                original_name=original_name,
                original_path=original_path,
                image_path=image_path,
                document_profile=document_profile,
                canvas_width_m=canvas_width_m,
                archive_consent=archive_consent,
                preparation_seconds=preparation_seconds,
                owner_id=owner.id if owner else "",
                owner_email=owner.email if owner else "",
                client_ip_hash=client_ip_hash,
                area_inspection=area_inspection,
                service_tier=service_tier,
                first_preview=first_preview,
            )
            return jsonify(response), 201
        except RateLimitExceeded as exc:
            return _rate_limited(exc)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            app.logger.warning("Conversão Pro indisponível: %s", exc)
            return jsonify({"error": "Conversão Pro temporariamente indisponível."}), 503
        except Exception as exc:
            traceback.print_exc()
            app.logger.error("Falha interna no preparo da Conversão Pro: %s", exc)
            return jsonify({"error": "Não foi possível preparar esta conversão."}), 500

    @app.post("/api/assisted-orders")
    def create_assisted_order():
        if not ASSISTED_ORDER_ENABLED:
            return jsonify({"error": "Serviço indisponível."}), 404
        image_file = request.files.get("file")
        if not image_file or not _valid_upload(image_file):
            return jsonify({"error": "Envie uma planta em PDF, PNG ou JPG."}), 400
        try:
            owner = _current_user()
            if AUTH.required and owner is None:
                return jsonify({"error": "Entre com o Google para criar seu pedido."}), 401
            sid = uuid.uuid4().hex[:10]
            order_dir = ASSISTED_ORDER_MANAGER.root_dir / sid
            original_path, original_name = _save_assisted_order_upload(image_file, order_dir)
            response = ASSISTED_ORDER_MANAGER.create(
                job=sid,
                original_name=original_name,
                original_path=original_path,
                area_m2=request.form.get("area_m2"),
                contact_email=str(request.form.get("contact_email") or ""),
                owner_id=owner.id if owner else "",
                owner_email=owner.email if owner else "",
            )
            return jsonify(response), 201
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            app.logger.exception("Falha ao criar pedido assistido: %s", exc)
            return jsonify({"error": "Não foi possível criar o pedido assistido."}), 500

    @app.post("/api/assisted-orders/<job>/checkout")
    def assisted_order_checkout(job: str):
        if not ASSISTED_ORDER_ENABLED:
            return jsonify({"error": "Serviço indisponível."}), 404
        if not JOB_ID_PATTERN.fullmatch(job):
            return jsonify({"error": "Pedido não encontrado."}), 404
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(ASSISTED_ORDER_MANAGER.create_checkout(job, str(body.get("access_token") or ""))), 201
        except (LookupError, PermissionError):
            return jsonify({"error": "Pedido não encontrado."}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        except MercadoPagoError as exc:
            app.logger.warning("Checkout assistido indisponível: %s", exc)
            return jsonify({"error": "Não foi possível abrir o pagamento agora."}), 503
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503

    @app.post("/api/assisted-orders/<job>/payment/sync")
    def assisted_order_payment_sync(job: str):
        if not ASSISTED_ORDER_ENABLED:
            return jsonify({"error": "Serviço indisponível."}), 404
        if not JOB_ID_PATTERN.fullmatch(job):
            return jsonify({"error": "Pedido não encontrado."}), 404
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(ASSISTED_ORDER_MANAGER.sync_payment(job, str(body.get("access_token") or "")))
        except (LookupError, PermissionError):
            return jsonify({"error": "Pedido não encontrado."}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        except MercadoPagoError:
            return jsonify({"error": "Não foi possível confirmar o pagamento agora."}), 503

    @app.get("/api/assisted-orders/<job>")
    def assisted_order_status(job: str):
        if not ASSISTED_ORDER_ENABLED:
            return jsonify({"error": "Serviço indisponível."}), 404
        if not JOB_ID_PATTERN.fullmatch(job):
            return jsonify({"error": "Pedido não encontrado."}), 404
        try:
            return jsonify(ASSISTED_ORDER_MANAGER.public_status(job, str(request.args.get("token") or "")))
        except (LookupError, PermissionError):
            return jsonify({"error": "Pedido não encontrado."}), 404

    @app.post("/api/astra-flow/jobs/<job>/checkout")
    def astra_checkout(job: str):
        if not JOB_ID_PATTERN.fullmatch(job):
            return jsonify({"error": "Tarefa não encontrada."}), 404
        body = request.get_json(silent=True) or {}
        try:
            owner = _current_user()
            if (AUTH.required or FIRST_PREVIEW_ENABLED) and owner is None:
                return jsonify({"error": "Entre com o Google para abrir o pagamento."}), 401
            token = str(body.get("access_token") or "")
            if PAID_ONLY_FLOW:
                ASTRA_FLOW_MANAGER.authorize_owner(job, token, owner.id if owner else "")
            current = ASTRA_FLOW_MANAGER.public_status(
                job, token, include_result=False
            )
            # Reopening an already-created checkout is idempotent and must not
            # consume another daily order allowance.
            if (current.get("payment") or {}).get("status") == "not_started":
                OPERATIONAL_GUARD.check_order(
                    owner_id=owner.id if owner else "",
                    client_ip=_client_ip(),
                )
            response = ASTRA_FLOW_MANAGER.create_checkout(
                job, token
            )
            return jsonify(response), 201
        except RateLimitExceeded as exc:
            return _rate_limited(exc)
        except (LookupError, PermissionError):
            return jsonify({"error": "Tarefa não encontrada."}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        except (MercadoPagoError, RuntimeError) as exc:
            app.logger.warning("Checkout indisponível: %s", exc)
            return jsonify({"error": str(exc)}), 503

    @app.post("/api/astra-flow/jobs/<job>/payment/sync")
    def astra_payment_sync(job: str):
        if not JOB_ID_PATTERN.fullmatch(job):
            return jsonify({"error": "Tarefa não encontrada."}), 404
        body = request.get_json(silent=True) or {}
        try:
            if PAID_ONLY_FLOW:
                owner = _current_user()
                ASTRA_FLOW_MANAGER.authorize_owner(
                    job, str(body.get("access_token") or ""), owner.id if owner else ""
                )
            response = ASTRA_FLOW_MANAGER.sync_payment(
                job, str(body.get("access_token") or "")
            )
            return jsonify(response)
        except (LookupError, PermissionError):
            return jsonify({"error": "Tarefa não encontrada."}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        except (MercadoPagoError, RuntimeError) as exc:
            app.logger.warning("Falha ao consultar pagamento: %s", exc)
            return jsonify({"error": "Não foi possível confirmar o pagamento agora."}), 503

    @app.post("/api/astra-flow/jobs/<job>/retry")
    def astra_retry(job: str):
        if not JOB_ID_PATTERN.fullmatch(job):
            return jsonify({"error": "Tarefa não encontrada."}), 404
        body = request.get_json(silent=True) or {}
        try:
            if PAID_ONLY_FLOW:
                owner = _current_user()
                ASTRA_FLOW_MANAGER.authorize_owner(
                    job, str(body.get("access_token") or ""), owner.id if owner else ""
                )
            response = ASTRA_FLOW_MANAGER.retry(
                job, str(body.get("access_token") or "")
            )
            return jsonify(response), 202 if response["status"] in {"queued", "running"} else 200
        except (LookupError, PermissionError):
            return jsonify({"error": "Tarefa não encontrada."}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        except RuntimeError as exc:
            app.logger.warning("Nova tentativa indisponível: %s", exc)
            return jsonify({"error": "Conversão Pro temporariamente indisponível."}), 503

    @app.post("/api/payments/mercadopago/webhook")
    def mercadopago_webhook():
        gateway = ASTRA_FLOW_MANAGER.payment_gateway
        if not gateway.webhook_configured:
            return "", 503
        body = request.get_json(silent=True) or {}
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        data_id = str(request.args.get("data.id") or data.get("id") or "").strip()
        event_type = str(request.args.get("type") or body.get("type") or "").strip()
        if event_type != "payment" or not data_id:
            return "", 200
        if not gateway.verify_webhook(
            x_signature=request.headers.get("x-signature"),
            x_request_id=request.headers.get("x-request-id"),
            data_id=data_id,
        ):
            return "", 401
        try:
            ASTRA_FLOW_MANAGER.handle_payment_notification(data_id)
            if ASSISTED_ORDER_ENABLED:
                ASSISTED_ORDER_MANAGER.handle_payment_notification(data_id)
            return "", 200
        except MercadoPagoError as exc:
            app.logger.warning("Falha ao processar webhook de pagamento: %s", exc)
            return "", 502

    @app.post("/api/internal/astra-flow/jobs/<job>/run")
    def astra_durable_worker(job: str):
        """Trusted Cloud Tasks target; never called by the customer browser."""
        if not JOB_ID_PATTERN.fullmatch(job):
            return jsonify({"error": "Tarefa não encontrada."}), 404
        try:
            ASTRA_FLOW_MANAGER.verify_worker_request(
                str(request.headers.get("Authorization") or "")
            )
        except PermissionError:
            return jsonify({"error": "Worker não autorizado."}), 401
        try:
            state = ASTRA_FLOW_MANAGER.run_worker(job)
            return jsonify({
                "ok": state.get("status") == "completed",
                "job": job,
                "status": state.get("status"),
            })
        except LookupError:
            return jsonify({"error": "Tarefa não encontrada."}), 404
        except (PermissionError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 409
        except Exception as exc:
            app.logger.exception("Falha de infraestrutura no worker %s: %s", job, exc)
            # Non-2xx asks Cloud Tasks to retry infrastructure failures.
            return jsonify({"error": "Falha temporária do worker."}), 503

    @app.post("/api/internal/maintenance/retention")
    def retention_maintenance():
        """Daily authenticated cleanup target for Cloud Scheduler."""
        try:
            ASTRA_FLOW_MANAGER.verify_worker_request(
                str(request.headers.get("Authorization") or "")
            )
        except PermissionError:
            return jsonify({"error": "Manutenção não autorizada."}), 401
        try:
            return jsonify({"ok": True, **ASTRA_FLOW_MANAGER.purge_expired_files()})
        except Exception as exc:
            app.logger.exception("Falha na limpeza de retenção: %s", exc)
            return jsonify({"error": "Falha temporária na limpeza de arquivos."}), 503

    @app.get("/api/astra-flow/jobs/<job>")
    def astra_job_status(job: str):
        if not JOB_ID_PATTERN.fullmatch(job):
            return jsonify({"error": "Tarefa não encontrada."}), 404
        try:
            if PAID_ONLY_FLOW:
                owner = _current_user()
                ASTRA_FLOW_MANAGER.authorize_owner(
                    job, str(request.args.get("token") or ""), owner.id if owner else ""
                )
            response = ASTRA_FLOW_MANAGER.public_status(
                job, str(request.args.get("token") or ""), include_result=True
            )
            return jsonify(response)
        except (LookupError, PermissionError):
            return jsonify({"error": "Tarefa não encontrada."}), 404

    @app.post("/api/bim-editing/apply")
    def apply_revision():
        payload = request.get_json(silent=True) or {}
        try:
            from bim_editing.adapters import parts_index
            from bim_editing.engine import RevisionEngine, RevisionError

            revised, report = RevisionEngine(payload["model"]).apply(payload["revision"])
            return jsonify({
                "model": revised,
                "report": report,
                "parts": parts_index(revised),
            })
        except (KeyError, TypeError, ValueError, RevisionError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/referencia/finalizar")
    def finalize():
        try:
            body = request.get_json(force=True, silent=True) or {}
            if PAID_ONLY_FLOW:
                source_job = _safe_source_job(body.get("source_job"))
                owner = _current_user()
                if not source_job:
                    return jsonify({"error": "Pedido pago obrigatório para exportar IFC."}), 403
                try:
                    ASTRA_FLOW_MANAGER.authorize_paid_export(
                        source_job, str(body.get("access_token") or ""),
                        owner.id if owner else "",
                    )
                except (LookupError, PermissionError):
                    return jsonify({"error": "Pedido pago não encontrado ou ainda indisponível."}), 403
            model_payload = body.get("modelo") or body
            config = body.get("config", {})
            name = secure_filename(body.get("nome", "planta")) or "planta"
            archive_consent = _truthy(body.get("archive_consent"))
            source_job = _safe_source_job(body.get("source_job"))
            if body.get("exigir_aprovacao_cliente"):
                approval = body.get("aprovacao_cliente") or {}
                if approval.get("confirmado") is not True:
                    return jsonify({
                        "error": "Confirme a revisão antes de exportar o IFC.",
                        "requires_approval": True,
                    }), 409
            if not model_payload.get("paredes"):
                return jsonify({"error": "Modelo sem paredes."}), 400

            from plantatobim import planta_to_ifc_v1 as planta_module

            internal = planta_module.dict_para_modelo(model_payload)
            if not internal["paredes"]:
                return jsonify({"error": "Nenhuma parede válida no modelo."}), 400

            sid = uuid.uuid4().hex[:10]
            ifc_name = f"{source_job}_{sid}_{name}.ifc" if PAID_ONLY_FLOW else f"{sid}_{name}.ifc"
            ifc_path = OUTPUT_FOLDER / ifc_name
            with contextlib.redirect_stdout(io.StringIO()):
                planta_module.gerar_ifc_do_modelo(
                    internal["paredes"],
                    internal["aberturas"],
                    ifc_path,
                    config,
                    laje=internal.get("laje"),
                    spaces=internal.get("spaces"),
                )
            archive_status = "not-requested"
            if archive_consent and source_job and TRAINING_ARCHIVE is not None:
                try:
                    TRAINING_ARCHIVE.archive_final(
                        job=source_job,
                        ifc_path=ifc_path,
                        model=model_payload,
                        config=config,
                        engine=ENGINE_NAME,
                    )
                    archive_status = "saved"
                except SupabaseArchiveError as exc:
                    app.logger.warning("Falha ao arquivar IFC %s: %s", source_job, exc)
                    archive_status = "failed"
            elif archive_consent:
                archive_status = "unavailable" if TRAINING_ARCHIVE is None else "missing-job"
            return jsonify({
                "ok": True,
                "ifc_url": (
                    f"/api/backend/outputs/{ifc_name}" if PAID_ONLY_FLOW
                    else f"/outputs/{ifc_name}"
                ),
                "preview_url": None,
                "ifc_token": None,
                "ready_for_comparison": False,
                "requires_approval": False,
                "n_paredes": len(internal["paredes"]),
                "n_aberturas": len(internal["aberturas"]),
                "archive_status": archive_status,
            })
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Falha ao gerar IFC: {exc}"}), 500

    @app.post("/api/referencia/exportar-dxf")
    def export_dxf():
        try:
            body = request.get_json(force=True, silent=True) or {}
            source_job = _safe_source_job(body.get("source_job"))
            if PAID_ONLY_FLOW:
                owner = _current_user()
                if not source_job:
                    return jsonify({"error": "Pedido pago obrigatório para exportar DXF."}), 403
                try:
                    ASTRA_FLOW_MANAGER.authorize_paid_export(
                        source_job, str(body.get("access_token") or ""),
                        owner.id if owner else "",
                    )
                except (LookupError, PermissionError):
                    return jsonify({"error": "Pedido pago não encontrado ou ainda indisponível."}), 403
            approval = body.get("aprovacao_cliente") or {}
            if approval.get("confirmado") is not True:
                return jsonify({
                    "error": "Confirme a revisão antes de exportar o DXF.",
                    "requires_approval": True,
                }), 409
            model_payload = body.get("modelo") or {}
            if not model_payload.get("paredes"):
                return jsonify({"error": "Modelo sem paredes."}), 400

            from plantatobim.export_editor_model_dxf import export_model_to_dxf

            name = secure_filename(body.get("nome", "planta")) or "planta"
            sid = uuid.uuid4().hex[:10]
            dxf_name = f"{source_job}_{sid}_{name}.dxf" if PAID_ONLY_FLOW else f"{sid}_{name}.dxf"
            report = export_model_to_dxf(model_payload, OUTPUT_FOLDER / dxf_name)
            return jsonify({
                "ok": True,
                "dxf_url": (
                    f"/api/backend/outputs/{dxf_name}" if PAID_ONLY_FLOW
                    else f"/outputs/{dxf_name}"
                ),
                **{key: value for key, value in report.items() if key != "output"},
            })
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Falha ao gerar DXF: {exc}"}), 500

    @app.get("/outputs/<path:filename>")
    def download_output(filename: str):
        if PAID_ONLY_FLOW:
            job = _safe_source_job(filename.split("_", 1)[0])
            owner = _current_user()
            if not job or not filename.lower().endswith((".ifc", ".dxf")):
                return jsonify({"error": "Arquivo não encontrado."}), 404
            try:
                ASTRA_FLOW_MANAGER.authorize_paid_file(job, owner.id if owner else "")
            except (LookupError, PermissionError):
                return jsonify({"error": "Arquivo não encontrado."}), 404
        return send_from_directory(
            str(OUTPUT_FOLDER),
            filename,
            as_attachment=str(request.args.get("download", "")).lower() in {
                "1", "true", "yes",
            },
        )

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "O arquivo ultrapassa o limite de 25 MB."}), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="127.0.0.1" if LOW_COST_LOCAL_PILOT else "0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
    )
