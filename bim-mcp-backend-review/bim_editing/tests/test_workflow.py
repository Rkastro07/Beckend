from __future__ import annotations

import json
from pathlib import Path

import ifcopenshell
import pytest

from bim_editing.mcp_adapter import describe_mcp_surface
from bim_editing.workflow import (
    BimWorkflowError,
    apply_revision_artifacts,
    export_approved_ifc,
    recover_ifc_artifacts,
)
from plantatobim.planta_to_ifc_v1 import dict_para_modelo, gerar_ifc_do_modelo


def _editable_rectangle():
    return {
        "revision": "R00",
        "paredes": [
            {"id": "W-001", "ax": 0, "ay": 0, "bx": 4, "by": 0,
             "espessura": 0.15, "layer": "Wall"},
            {"id": "W-002", "ax": 4, "ay": 0, "bx": 4, "by": 3,
             "espessura": 0.15, "layer": "Wall"},
            {"id": "W-003", "ax": 4, "ay": 3, "bx": 0, "by": 3,
             "espessura": 0.15, "layer": "Wall"},
            {"id": "W-004", "ax": 0, "ay": 3, "bx": 0, "by": 0,
             "espessura": 0.15, "layer": "Wall"},
        ],
        "aberturas": [],
        "spaces": [{
            "id": "SPACE-001",
            "contorno": [[0, 0], [4, 0], [4, 3], [0, 3]],
            "area": 12.0,
        }],
        "laje": {
            "contorno": [[0, 0], [4, 0], [4, 3], [0, 3]],
            "piso": {"ativo": True, "espessura": 0.12},
            "teto": {"ativo": True, "espessura": 0.12},
        },
    }


def _source_ifc(path: Path):
    editable = _editable_rectangle()
    internal = dict_para_modelo(editable)
    gerar_ifc_do_modelo(
        internal["paredes"],
        internal["aberturas"],
        path,
        config={"altura": 3.0},
        laje=internal["laje"],
        spaces=internal["spaces"],
    )


def test_file_workflow_requires_png_approval_before_ifc(tmp_path):
    source = tmp_path / "source.ifc"
    _source_ifc(source)

    recovered = recover_ifc_artifacts(
        source,
        tmp_path / "recovered",
        selected=["W-001.P2"],
    )
    base_model = Path(recovered["artifacts"]["model"])
    assert base_model.is_file()
    assert Path(recovered["artifacts"]["overview_png"]).is_file()
    assert Path(recovered["artifacts"]["edit_png"]).is_file()
    assert recovered["counts"]["walls"] == 4

    revised = apply_revision_artifacts(
        base_model,
        {
            "schema": "bim.edit-operations.v1",
            "revision": "R01",
            "operations": [{
                "op": "set_wall_thickness",
                "id": "W-001",
                "thickness": 0.20,
            }],
            "render": {"selected": ["W-001.P1", "W-001.P2"]},
        },
        tmp_path / "revision",
    )
    revised_model = Path(revised["artifacts"]["model"])
    assert revised_model.is_file()
    assert Path(revised["artifacts"]["edit_png"]).is_file()
    assert revised["next_action"] == "review_png_before_export"
    model_data = json.loads(revised_model.read_text(encoding="utf-8"))
    wall = next(item for item in model_data["paredes"] if item["id"] == "W-001")
    assert wall["espessura"] == 0.20

    output_ifc = tmp_path / "revision" / "approved.ifc"
    with pytest.raises(BimWorkflowError, match="approved=true"):
        export_approved_ifc(revised_model, output_ifc, approved=False)
    assert not output_ifc.exists()

    exported = export_approved_ifc(revised_model, output_ifc, approved=True)
    assert exported["approved"] is True
    assert output_ifc.is_file()
    model = ifcopenshell.open(str(output_ifc))
    assert len(model.by_type("IfcWall")) == 4
    assert len(model.by_type("IfcSpace")) == 1


def test_mcp_surface_exposes_complete_artifact_lifecycle():
    surface = describe_mcp_surface()
    names = {item["name"] for item in surface["tools"]}
    assert names == {
        "recover_ifc_for_editing",
        "resolve_bim_part",
        "apply_bim_revision",
        "render_bim_revision",
        "export_approved_bim_revision",
    }
    assert surface["one_off_code_allowed"] is False
    export_tool = next(
        item for item in surface["tools"]
        if item["name"] == "export_approved_bim_revision"
    )
    assert export_tool["input_schema"]["properties"]["approved"] == {"const": True}
