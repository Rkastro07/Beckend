"""HTTP adapter for the revision engine used by the current local app."""

from __future__ import annotations

from typing import Any

from .adapters import parts_index
from .engine import RevisionEngine, RevisionError
from .mcp_adapter import SUPPORTED_OPERATIONS, describe_mcp_surface


def register_bim_editing_routes(app: Any) -> None:
    @app.get("/api/bim-editing/operations")
    def _editing_operations():
        from flask import jsonify

        return jsonify(
            {
                "schema": "bim.edit-operations-catalog.v1",
                "operations": SUPPORTED_OPERATIONS,
            }
        )

    @app.get("/api/bim-editing/mcp-surface")
    def _editing_mcp_surface():
        from flask import jsonify

        return jsonify(describe_mcp_surface())

    @app.post("/api/bim-editing/resolve")
    def _resolve_part():
        from flask import jsonify, request

        payload = request.get_json(silent=True) or {}
        try:
            engine = RevisionEngine(payload["model"])
            engine.model = engine.base
            resolved = engine.resolve_selector(str(payload["selector"]))
            if resolved["kind"] == "wall":
                resolved["value"] = {
                    "id": resolved["value"]["id"],
                    "parts": resolved["value"]["parts"],
                }
            return jsonify(resolved)
        except (KeyError, TypeError, ValueError, RevisionError) as exc:
            return jsonify({"error": str(exc)}), 400
    @app.post("/api/bim-editing/apply")
    def _apply_revision():
        from flask import jsonify, request

        payload = request.get_json(silent=True) or {}
        try:
            revised, report = RevisionEngine(payload["model"]).apply(
                payload["revision"]
            )
            return jsonify(
                {
                    "model": revised,
                    "report": report,
                    "parts": parts_index(revised),
                }
            )
        except (KeyError, TypeError, ValueError, RevisionError) as exc:
            return jsonify({"error": str(exc)}), 400
