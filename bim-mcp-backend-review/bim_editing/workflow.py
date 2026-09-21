"""File-based workflow shared by the BIM MCP server and local automation.

This module only orchestrates existing deterministic components. It does not
contain geometry rules: IFC recovery, revisions, rendering and IFC authoring
remain owned by their canonical modules.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
import uuid

from .adapters import load_json, parts_index, save_json
from .engine import RevisionEngine
from .ifc_recovery import recover_editor_model
from .model import normalize_model
from .render import render_revision_set


class BimWorkflowError(RuntimeError):
    """Controlled failure in the artifact workflow."""


def _absolute_file(path: str | Path, *, suffix: str | None = None) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise BimWorkflowError(f"arquivo nao encontrado: {resolved}")
    if suffix and resolved.suffix.lower() != suffix.lower():
        raise BimWorkflowError(f"arquivo precisa ter extensao {suffix}: {resolved}")
    return resolved


def _output_directory(output_dir: str | Path | None, *, prefix: str) -> Path:
    if output_dir is None:
        root = Path(os.environ.get(
            "BIM_MCP_WORKSPACE",
            Path(tempfile.gettempdir()) / "bim_mcp_workspace",
        ))
        directory = root / f"{prefix}_{uuid.uuid4().hex[:10]}"
    else:
        directory = Path(output_dir).expanduser()
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _selected(values: Iterable[str] | None, revision: dict | None = None) -> list[str]:
    explicit = [str(value) for value in (values or []) if str(value).strip()]
    if explicit:
        return explicit
    render = (revision or {}).get("render", {})
    return [str(value) for value in render.get("selected", [])]


def _paths(payload: dict[str, Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, path in payload.items():
        resolved = path.resolve()
        result[name] = str(resolved)
        result[f"{name}_uri"] = resolved.as_uri()
    return result


def _manifest(directory: Path, stage: str, artifacts: dict[str, Path], **extra) -> Path:
    return save_json(directory / "artifact_manifest.json", {
        "schema": "bim.mcp-artifact-manifest.v1",
        "stage": stage,
        "approved": stage == "exported",
        "artifacts": _paths(artifacts),
        **extra,
    })


def recover_ifc_artifacts(
    ifc_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    force_ceiling: bool = False,
    selected: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Recover one IFC into the canonical JSON editor and review PNGs."""

    source = _absolute_file(ifc_path, suffix=".ifc")
    directory = _output_directory(output_dir, prefix=source.stem)
    recovered = recover_editor_model(source, force_ceiling=force_ceiling)
    model = dict(recovered["modelo"])
    # File-based clients must not lose the vertical/export configuration.
    model["ifc_config"] = dict(recovered["config"])
    model = normalize_model(model)

    model_path = save_json(directory / "base_model.json", model)
    config_path = save_json(directory / "ifc_config.json", recovered["config"])
    parts_path = save_json(directory / "element_parts.json", parts_index(model))
    recovery_path = save_json(directory / "recovery_report.json", {
        "source_ifc": str(source),
        "warnings": recovered.get("warnings", []),
        "counts": recovered.get("counts", {}),
    })
    images = render_revision_set(model, directory, selected=selected or [])
    artifacts = {
        "model": model_path,
        "config": config_path,
        "parts": parts_path,
        "report": recovery_path,
        "overview_png": images["overview"],
        "edit_png": images["edit"],
    }
    manifest_path = _manifest(
        directory,
        "recovered",
        artifacts,
        source_ifc=str(source),
        counts=recovered.get("counts", {}),
    )
    artifacts["manifest"] = manifest_path
    return {
        "schema": "bim.mcp-recovery-result.v1",
        "output_dir": str(directory),
        "counts": recovered.get("counts", {}),
        "warnings": recovered.get("warnings", []),
        "artifacts": _paths(artifacts),
        "next_action": "apply_bim_revision",
    }


def render_model_artifacts(
    model_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    selected: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Render the existing canonical JSON without changing it or creating IFC."""

    source = _absolute_file(model_path, suffix=".json")
    directory = _output_directory(output_dir, prefix=source.stem)
    model = load_json(source)
    images = render_revision_set(model, directory, selected=selected or [])
    return {
        "schema": "bim.mcp-render-result.v1",
        "model": str(source),
        "output_dir": str(directory),
        "artifacts": _paths({
            "overview_png": images["overview"],
            "edit_png": images["edit"],
        }),
        "next_action": "review_png_before_export",
    }


def apply_revision_artifacts(
    model_path: str | Path,
    revision: dict[str, Any] | str | Path,
    output_dir: str | Path | None = None,
    *,
    selected: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Apply declarative edits and produce JSON/report/PNGs, never an IFC."""

    source = _absolute_file(model_path, suffix=".json")
    specification = (
        load_json(_absolute_file(revision, suffix=".json"))
        if isinstance(revision, (str, Path))
        else dict(revision)
    )
    directory = _output_directory(output_dir, prefix="revision")
    revised, report = RevisionEngine(load_json(source)).apply(specification)
    model_output = save_json(directory / "revision_model.json", revised)
    report_output = save_json(directory / "revision_report.json", report)
    parts_output = save_json(directory / "element_parts.json", parts_index(revised))
    revision_output = save_json(directory / "revision_operations.json", specification)
    images = render_revision_set(
        revised,
        directory,
        selected=_selected(selected, specification),
    )
    artifacts = {
        "model": model_output,
        "revision": revision_output,
        "report": report_output,
        "parts": parts_output,
        "overview_png": images["overview"],
        "edit_png": images["edit"],
    }
    manifest_path = _manifest(
        directory,
        "revised-awaiting-approval",
        artifacts,
        base_model=str(source),
        validation=report.get("validation", {}),
    )
    artifacts["manifest"] = manifest_path
    return {
        "schema": "bim.mcp-revision-result.v1",
        "output_dir": str(directory),
        "counts": {
            "walls": len(revised.get("paredes", [])),
            "openings": len(revised.get("aberturas", [])),
            "spaces": len(revised.get("spaces", [])),
        },
        "validation": report.get("validation", {}),
        "artifacts": _paths(artifacts),
        "next_action": "review_png_before_export",
    }


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def export_approved_ifc(
    model_path: str | Path,
    output_ifc: str | Path,
    *,
    approved: bool,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export an approved JSON revision through the existing IFC authoring engine."""

    if approved is not True:
        raise BimWorkflowError(
            "exportacao bloqueada: aprove visualmente a PNG e envie approved=true"
        )
    source = _absolute_file(model_path, suffix=".json")
    destination = Path(output_ifc).expanduser().resolve()
    if destination.suffix.lower() != ".ifc":
        raise BimWorkflowError("output_ifc precisa terminar em .ifc")
    if destination.exists() and not overwrite:
        raise BimWorkflowError(
            f"IFC de destino ja existe; escolha outro caminho: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = load_json(source)
    final_config = dict(model.get("ifc_config", {}))
    if config_path is not None:
        final_config = _deep_merge(
            final_config,
            load_json(_absolute_file(config_path, suffix=".json")),
        )
    if config:
        final_config = _deep_merge(final_config, config)

    from plantatobim.planta_to_ifc_v1 import dict_para_modelo, gerar_ifc_do_modelo

    internal = dict_para_modelo(model)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        gerar_ifc_do_modelo(
            internal["paredes"],
            internal["aberturas"],
            destination,
            config=final_config,
            laje=internal["laje"],
            spaces=internal["spaces"],
        )
    if not destination.is_file() or destination.stat().st_size < 100:
        raise BimWorkflowError("gerador nao produziu um IFC valido")

    import ifcopenshell

    ifc = ifcopenshell.open(str(destination))
    counts = {
        "walls": len(ifc.by_type("IfcWall")),
        "doors": len(ifc.by_type("IfcDoor")),
        "windows": len(ifc.by_type("IfcWindow")),
        "slabs": len(ifc.by_type("IfcSlab")),
        "spaces": len(ifc.by_type("IfcSpace")),
        "coverings": len(ifc.by_type("IfcCovering")),
    }
    manifest_path = _manifest(
        destination.parent,
        "exported",
        {"model": source, "ifc": destination},
        counts=counts,
    )
    return {
        "schema": "bim.mcp-ifc-export-result.v1",
        "approved": True,
        "counts": counts,
        "artifacts": _paths({"ifc": destination, "manifest": manifest_path}),
        "next_action": "download_or_compare_ifc",
    }
