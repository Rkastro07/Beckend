import subprocess
import sys
from pathlib import Path

import ezdxf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cad_converters


def _write_valid_dxf(path: Path):
    document = ezdxf.new("R2018")
    document.modelspace().add_line((0, 0), (5, 0))
    document.saveas(str(path))


def test_dwg_conversion_retries_without_isolate(monkeypatch, tmp_path):
    executable = tmp_path / "accoreconsole.exe"
    executable.write_bytes(b"fake")
    source = tmp_path / "drawing.dwg"
    source.write_bytes(b"dwg")
    target = tmp_path / "drawing.dxf"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "/isolate" in command:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout=b"ERROR: Failed to create registry keys for /isolate",
            )
        _write_valid_dxf(target)
        return subprocess.CompletedProcess(command, 0, stdout=b"DXF created")

    monkeypatch.setattr(cad_converters, "find_accoreconsole", lambda: executable)
    monkeypatch.setattr(cad_converters, "find_libredwg", lambda: None)
    monkeypatch.setattr(cad_converters.subprocess, "run", fake_run)
    monkeypatch.setattr(cad_converters, "_AUTOCAD_ISOLATE_SUPPORTED", None)

    result = cad_converters.convert_dwg_to_dxf(source, target)

    assert result == target
    assert len(calls) == 2
    assert "/isolate" in calls[0]
    assert "/isolate" not in calls[1]
    assert not target.with_suffix(".dwg_to_dxf.scr").exists()


def test_known_bad_isolate_mode_is_skipped(monkeypatch, tmp_path):
    executable = tmp_path / "accoreconsole.exe"
    executable.write_bytes(b"fake")
    source = tmp_path / "drawing.dwg"
    source.write_bytes(b"dwg")
    target = tmp_path / "drawing.dxf"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        _write_valid_dxf(target)
        return subprocess.CompletedProcess(command, 0, stdout=b"DXF created")

    monkeypatch.setattr(cad_converters, "find_accoreconsole", lambda: executable)
    monkeypatch.setattr(cad_converters, "find_libredwg", lambda: None)
    monkeypatch.setattr(cad_converters.subprocess, "run", fake_run)
    monkeypatch.setattr(cad_converters, "_AUTOCAD_ISOLATE_SUPPORTED", False)

    cad_converters.convert_dwg_to_dxf(source, target)

    assert len(calls) == 1
    assert "/isolate" not in calls[0]


def test_libredwg_is_used_when_autocad_is_unavailable(monkeypatch, tmp_path):
    executable = tmp_path / "dwgread"
    executable.write_bytes(b"fake")
    source = tmp_path / "drawing.dwg"
    source.write_bytes(b"dwg")
    target = tmp_path / "drawing.dxf"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        _write_valid_dxf(target)
        return subprocess.CompletedProcess(command, 0, stdout=b"DXF created")

    monkeypatch.setattr(cad_converters, "find_accoreconsole", lambda: None)
    monkeypatch.setattr(cad_converters, "find_libredwg", lambda: executable)
    monkeypatch.setattr(cad_converters.subprocess, "run", fake_run)

    result, details = cad_converters.convert_dwg_to_dxf_with_details(
        source, target,
    )

    assert result == target
    assert details["engine"] == "gnu_libredwg"
    assert details["validation"]["geometric_entities"] == 1
    assert calls[0][1:5] == ["-v1", "-O", "DXF", "-o"]


def test_libredwg_is_preferred_over_autocad(monkeypatch, tmp_path):
    libredwg = tmp_path / "dwgread"
    autocad = tmp_path / "accoreconsole.exe"
    libredwg.write_bytes(b"fake")
    autocad.write_bytes(b"fake")
    source = tmp_path / "drawing.dwg"
    source.write_bytes(b"dwg")
    target = tmp_path / "drawing.dxf"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        _write_valid_dxf(target)
        return subprocess.CompletedProcess(command, 0, stdout=b"DXF created")

    monkeypatch.setattr(cad_converters, "find_accoreconsole", lambda: autocad)
    monkeypatch.setattr(cad_converters, "find_libredwg", lambda: libredwg)
    monkeypatch.setattr(cad_converters.subprocess, "run", fake_run)

    _result, details = cad_converters.convert_dwg_to_dxf_with_details(
        source, target,
    )

    assert details["engine"] == "gnu_libredwg"
    assert calls[0][0] == str(libredwg)


def test_libredwg_normalizes_windows_double_carriage_returns(
    monkeypatch, tmp_path,
):
    executable = tmp_path / "dwgread.exe"
    executable.write_bytes(b"fake")
    source = tmp_path / "drawing.dwg"
    source.write_bytes(b"dwg")
    target = tmp_path / "drawing.dxf"

    def fake_run(command, **kwargs):
        _write_valid_dxf(target)
        raw = target.read_bytes().replace(b"\r\n", b"\n")
        target.write_bytes(raw.replace(b"\n", b"\r\r\n"))
        return subprocess.CompletedProcess(command, 0, stdout=b"DXF created")

    monkeypatch.setattr(cad_converters, "find_accoreconsole", lambda: None)
    monkeypatch.setattr(cad_converters, "find_libredwg", lambda: executable)
    monkeypatch.setattr(cad_converters.subprocess, "run", fake_run)

    result, details = cad_converters.convert_dwg_to_dxf_with_details(
        source, target,
    )

    assert result == target
    assert b"\r\r\n" not in target.read_bytes()
    assert details["validation"]["geometric_entities"] == 1


def test_invalid_libredwg_output_is_rejected(monkeypatch, tmp_path):
    executable = tmp_path / "dwgread"
    executable.write_bytes(b"fake")
    source = tmp_path / "drawing.dwg"
    source.write_bytes(b"dwg")
    target = tmp_path / "drawing.dxf"

    def fake_run(command, **kwargs):
        target.write_bytes(b"not-a-dxf" * 50)
        return subprocess.CompletedProcess(command, 0, stdout=b"done")

    monkeypatch.setattr(cad_converters, "find_accoreconsole", lambda: None)
    monkeypatch.setattr(cad_converters, "find_libredwg", lambda: executable)
    monkeypatch.setattr(cad_converters.subprocess, "run", fake_run)

    try:
        cad_converters.convert_dwg_to_dxf(source, target)
    except cad_converters.CadConversionError as exc:
        assert "DXF válido" in str(exc)
    else:
        raise AssertionError("conversão inválida deveria ter sido bloqueada")
    assert not target.exists()


def test_missing_dwg_engines_has_actionable_error(monkeypatch, tmp_path):
    source = tmp_path / "drawing.dwg"
    source.write_bytes(b"dwg")
    monkeypatch.setattr(cad_converters, "find_accoreconsole", lambda: None)
    monkeypatch.setattr(cad_converters, "find_libredwg", lambda: None)

    try:
        cad_converters.convert_dwg_to_dxf(source, tmp_path / "drawing.dxf")
    except cad_converters.CadConversionError as exc:
        assert "ACCORECONSOLE_PATH" in str(exc)
        assert "dwgread" in str(exc)
    else:
        raise AssertionError("ausência de conversores deveria falhar")
