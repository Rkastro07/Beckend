# -*- coding: utf-8 -*-
"""Public Plan-to-BIM service with no LLM providers or routes."""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import tempfile
import traceback
import uuid

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename


RUNTIME_DIR = Path(tempfile.gettempdir())
OUTPUT_FOLDER = RUNTIME_DIR / "plan_to_bim_outputs"
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


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


def _save_as_image(file_storage, job_dir: Path) -> tuple[Path, str]:
    """Save an image or render the first page of a PDF to a PNG."""
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    original_name = secure_filename(file_storage.filename) or "planta.png"
    source_path = input_dir / original_name
    file_storage.save(str(source_path))
    if source_path.suffix.lower() != ".pdf":
        return source_path, original_name

    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(source_path))
        if len(document) == 0:
            raise ValueError("PDF sem páginas.")
        page = document[0]
        rendered = page.render(scale=3.0).to_pil().convert("RGB")
        image_path = input_dir / f"{source_path.stem}_pagina_1.png"
        rendered.save(image_path, format="PNG", optimize=True)
        page.close()
        document.close()
        return image_path, original_name
    except Exception as exc:
        raise ValueError("Não foi possível renderizar a primeira página do PDF.") from exc


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
        return jsonify({
            "status": "online",
            "service": "Plan-to-BIM Free",
            "version": "2.0.0",
            "engine": "2d+yolo-walls+yolo-openings",
            "capabilities": ["plan-to-bim", "editor-2d-3d", "ifc4", "dxf"],
            "llm": False,
        })

    @app.get("/api/referencia/formatos")
    def formats():
        return jsonify({
            "ok": True,
            "entrada": list(SUPPORTED_EXTENSIONS),
            "pdf": "A primeira página é convertida pelo mesmo motor híbrido.",
            "engine": "2d+yolo-walls+yolo-openings",
        })

    @app.post("/api/plan-to-bim")
    @app.post("/api/referencia/pre-wall-yolo")
    def plan_to_bim():
        image_file = request.files.get("file")
        if not image_file or not _valid_upload(image_file):
            return jsonify({"error": "Envie uma planta em PDF, PNG ou JPG."}), 400
        try:
            canvas_width_m = _form_number("canvas_width_m", 20.0, 1.0, 500.0)
            metric_refinement = str(
                request.form.get("metric_refinement", "true")
            ).strip().lower() not in {"0", "false", "no"}

            sid = uuid.uuid4().hex[:10]
            job_dir = OUTPUT_FOLDER / f"plan_to_bim_{sid}"
            image_path, original_name = _save_as_image(image_file, job_dir)

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
                return jsonify({"error": str(exc)}), 422

            model["nome"] = Path(original_name).stem
            model["job"] = sid
            model["engine"] = "2d+yolo-walls+yolo-openings"
            return jsonify(model)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Falha na conversão Plan-to-BIM: {exc}"}), 500

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
            model_payload = body.get("modelo") or body
            config = body.get("config", {})
            name = secure_filename(body.get("nome", "planta")) or "planta"
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
            ifc_name = f"{sid}_{name}.ifc"
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
            return jsonify({
                "ok": True,
                "ifc_url": f"/outputs/{ifc_name}",
                "preview_url": None,
                "ifc_token": None,
                "ready_for_comparison": False,
                "requires_approval": False,
                "n_paredes": len(internal["paredes"]),
                "n_aberturas": len(internal["aberturas"]),
            })
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Falha ao gerar IFC: {exc}"}), 500

    @app.post("/api/referencia/exportar-dxf")
    def export_dxf():
        try:
            body = request.get_json(force=True, silent=True) or {}
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
            dxf_name = f"{uuid.uuid4().hex[:10]}_{name}.dxf"
            report = export_model_to_dxf(model_payload, OUTPUT_FOLDER / dxf_name)
            return jsonify({
                "ok": True,
                "dxf_url": f"/outputs/{dxf_name}",
                **{key: value for key, value in report.items() if key != "output"},
            })
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Falha ao gerar DXF: {exc}"}), 500

    @app.get("/outputs/<path:filename>")
    def download_output(filename: str):
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
