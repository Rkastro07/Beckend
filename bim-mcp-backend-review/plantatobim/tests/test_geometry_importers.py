import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry_importers import (
    DIRECT_EDIT_FORMATS,
    _fit_wall_from_vertices,
    format_capabilities,
    importar_pdf,
    read_pdf_geometry,
    read_svg_geometry,
)


def _write(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    return path


def test_catalog_separates_modeler_from_cloud_to_bim():
    catalog = format_capabilities()
    assert set(DIRECT_EDIT_FORMATS) == {
        ".dxf", ".svg", ".pdf", ".dwg", ".ifc", ".ifczip",
    }
    assert catalog["formatos"][".ifc"]["modo"] == "semantico"
    assert catalog["formatos"][".obj"]["rota"] == "cloud-to-bim"
    assert "converter_available" in catalog["formatos"][".dwg"]
    assert catalog["formatos"][".dwf"]["rota"] == "exportar-dwg-dxf-pdf"


def test_svg_reads_layers_transform_and_metric_width(tmp_path):
    path = _write(
        tmp_path / "planta.svg",
        """<svg xmlns="http://www.w3.org/2000/svg"
             width="10m" height="5m" viewBox="0 0 1000 500">
             <g id="A-WALL" transform="translate(100 50)">
               <line x1="0" y1="0" x2="300" y2="0"/>
             </g>
             <g id="A-DOOR">
               <rect x="200" y="-10" width="80" height="20"/>
             </g>
           </svg>""",
    )
    data = read_svg_geometry(path)
    assert data["scale"] == 0.01
    assert len(data["segments"]) == 1
    assert len(data["reference_segments"]) == 5
    a, b, layer = data["segments"][0]
    assert layer.endswith("A-WALL")
    assert np.allclose(a, [1.0, 0.5])
    assert np.allclose(b, [4.0, 0.5])
    assert len(data["openings"]) == 1
    assert data["openings"][0]["tipo"] == "door"


def test_svg_without_wall_layer_uses_unclassified_geometry_with_warning(tmp_path):
    path = _write(
        tmp_path / "simple.svg",
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
             <polyline points="0,0 4,0 4,3"/>
           </svg>""",
    )
    data = read_svg_geometry(path, escala_forcada=1.0)
    assert len(data["segments"]) == 2
    assert any("sem grupo/layer" in warning for warning in data["warnings"])


def test_pdf_vector_reads_edges_and_applies_paper_scale(tmp_path):
    from reportlab.pdfgen import canvas

    path = tmp_path / "planta.pdf"
    drawing = canvas.Canvas(str(path), pagesize=(720, 360))
    drawing.rect(72, 72, 360, 180, stroke=1, fill=0)
    drawing.save()

    data = read_pdf_geometry(path, pdf_scale=100)
    assert data["page_count"] == 1
    assert data["vector_records"] == 4
    assert np.isclose(data["scale"], (0.0254 / 72.0) * 100)

    model = importar_pdf(path, pdf_scale=100)
    assert len(model["paredes"]) == 4
    assert model["single_line"] is True
    assert model["source"]["format"] == "pdf"
    assert model["source"]["paper_scale_denominator"] == 100
    assert model["reference"]["kind"] == "vector"
    assert len(model["reference"]["segments"]) == 4


def test_ifc_wall_fit_recovers_axis_thickness_and_height():
    vertices = []
    for x in (0.0, 5.0):
        for y in (-0.10, 0.10):
            for z in (1.5, 4.3):
                vertices.append([x, y, z])
    fitted = _fit_wall_from_vertices(np.asarray(vertices))
    a, b = fitted["eixo"]
    assert np.isclose(np.linalg.norm(b - a), 5.0)
    assert np.isclose(fitted["espessura"], 0.20)
    assert np.isclose(fitted["altura"], 2.8)
    assert np.isclose(fitted["elevacao"], 1.5)
    assert fitted["geometry_approximated"] is False
