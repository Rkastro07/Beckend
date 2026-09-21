"""Build the approval-first PNG from Cloud2BIM detector artifacts."""

from __future__ import annotations

from pathlib import Path

from .adapters import model_from_cloud2bim, parts_index, save_json
from .render import render_model


PRIMARY_REVIEW_NAME = "00_ABRIR_PRIMEIRO_REVISAO.png"


def build_cloud_review(
    diagnostics_csv: str | Path,
    openings_json: str | Path,
    output_dir: str | Path,
    *,
    vertical_levels_json: str | Path | None = None,
    revision: str = "R00-detection",
) -> dict:
    output = Path(output_dir)
    review_dir = output / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    model = model_from_cloud2bim(
        diagnostics_csv,
        openings_json,
        vertical_levels_json,
        revision=revision,
    )
    model_path = save_json(review_dir / "review_model.json", model)
    parts_path = save_json(review_dir / "element_parts.json", parts_index(model))
    primary_png = output / PRIMARY_REVIEW_NAME
    render_model(
        model,
        primary_png,
        mode="overview",
        width=3600,
        height=2400,
    )
    status_path = save_json(
        review_dir / "approval_status.json",
        {
            "schema": "bim.approval-status.v1",
            "revision": revision,
            "approval_required": True,
            "approved": False,
            "primary_png": str(primary_png.resolve()),
            "model": str(model_path.resolve()),
            "parts": str(parts_path.resolve()),
            "wall_count": len(model["paredes"]),
            "opening_count": len(model["aberturas"]),
            "vertical_levels": (
                str(Path(vertical_levels_json).resolve())
                if vertical_levels_json is not None
                else None
            ),
            "suspended_ceiling_detected": bool(
                model.get("diagnostico", {}).get("forro_detectado")
            ),
            "ifc_status": "base_internal_not_approved",
            "next_action": "review_png_and_submit_edit_operations",
        },
    )
    return {
        "primary_png": primary_png,
        "model": model_path,
        "parts": parts_path,
        "status": status_path,
        "walls": len(model["paredes"]),
        "openings": len(model["aberturas"]),
        "suspended_ceiling_detected": bool(
            model.get("diagnostico", {}).get("forro_detectado")
        ),
    }
