"""Conversores CAD locais usados antes do reconhecedor Planta-to-BIM.

O reconhecedor trabalha com DXF. Formatos proprietários são materializados em
um DXF temporário por um conversor instalado na própria máquina. Nenhum
download ou serviço externo é executado durante a conversão.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import uuid


class CadConversionError(RuntimeError):
    """Falha controlada ao converter um formato CAD proprietário."""


_AUTOCAD_LOCK = threading.Lock()
_AUTOCAD_ISOLATE_USER = "bim_platform"
_AUTOCAD_ISOLATE_SUPPORTED: bool | None = None
_REPO_ROOT = Path(__file__).resolve().parents[1]
_AUTOCAD_CANDIDATES = (
    Path(r"C:\Program Files\Autodesk\AutoCAD 2026\accoreconsole.exe"),
    Path(r"C:\Program Files\Autodesk\AutoCAD 2025\accoreconsole.exe"),
    Path(r"C:\Program Files\Autodesk\AutoCAD 2024\accoreconsole.exe"),
)
_LIBREDWG_CANDIDATES = (
    _REPO_ROOT / ".runtime" / "tools" / "libredwg-0.13.4-win64" / "dwgread.exe",
    Path("/usr/local/bin/dwgread"),
    Path("/usr/bin/dwgread"),
)


def find_accoreconsole() -> Path | None:
    configured = os.environ.get("ACCORECONSOLE_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    discovered = shutil.which("accoreconsole.exe")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend(_AUTOCAD_CANDIDATES)
    return next((path for path in candidates if path.is_file()), None)


def find_libredwg() -> Path | None:
    """Localiza o CLI oficial ``dwgread`` do GNU LibreDWG."""
    configured = os.environ.get("LIBREDWG_DWGREAD_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    discovered = shutil.which("dwgread")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend(_LIBREDWG_CANDIDATES)
    return next((path for path in candidates if path.is_file()), None)


def converter_status() -> dict:
    executable = find_accoreconsole()
    libredwg = find_libredwg()
    available = executable is not None or libredwg is not None
    return {
        "autocad_core_console": {
            "available": executable is not None,
            "path": str(executable) if executable else None,
            "formats": [".dwg"] if executable else [],
        },
        "libredwg": {
            "available": libredwg is not None,
            "path": str(libredwg) if libredwg else None,
            "formats": [".dwg"] if libredwg else [],
        },
        "dwg": {
            "available": available,
            "preferred": (
                "GNU LibreDWG" if libredwg
                else "AutoCAD Core Console" if executable
                else None
            ),
        },
        "dwf": {
            "available": False,
            "reason": (
                "DWF clássico armazena a geometria em W2D e o AutoCAD o "
                "expõe apenas como underlay; exporte para DWG, DXF ou PDF "
                "vetorial antes da importação."
            ),
        },
    }


def _decode_autocad_output(raw: bytes) -> str:
    if not raw:
        return ""
    if b"\x00" in raw[:200]:
        return raw.decode("utf-16-le", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _script_path(value: Path) -> str:
    return str(value.resolve()).replace("\\", "/")


def _autocad_isolated_profile() -> Path:
    """Reserva um caminho inexistente para o perfil isolado do AutoCAD."""
    configured_root = os.environ.get("ACCORECONSOLE_PROFILE_DIR")
    root = (
        Path(configured_root).expanduser()
        if configured_root
        else Path(tempfile.gettempdir())
    )
    root.mkdir(parents=True, exist_ok=True)
    # O /isolate cria a pasta e falha quando ela já existe vazia.
    return (root / f"bim_accoreconsole_profile_{uuid.uuid4().hex}").resolve()


def _validate_dxf(target: Path) -> dict:
    """Confirma que a conversao produziu um DXF parseavel e com geometria."""
    if not target.is_file() or target.stat().st_size < 100:
        raise CadConversionError("O conversor não produziu um DXF útil.")
    try:
        import ezdxf

        document = ezdxf.readfile(str(target))
        counts: dict[str, int] = {}
        for entity in document.modelspace():
            kind = entity.dxftype()
            counts[kind] = counts.get(kind, 0) + 1
    except Exception as exc:
        raise CadConversionError(
            f"O conversor produziu um DXF inválido: {exc}"
        ) from exc
    geometric_types = {
        "LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE",
        "SPLINE", "INSERT", "HATCH", "SOLID", "TRACE", "3DFACE",
        "MLINE", "POINT", "TEXT", "MTEXT", "DIMENSION", "LEADER",
        "MULTILEADER",
    }
    geometric_count = sum(
        count for kind, count in counts.items() if kind in geometric_types
    )
    if geometric_count == 0:
        raise CadConversionError(
            "O DXF convertido não contém entidades geométricas reconhecíveis."
        )
    return {
        "entities": sum(counts.values()),
        "geometric_entities": geometric_count,
        "entity_types": counts,
        "bytes": target.stat().st_size,
    }


def _convert_dwg_with_autocad(
    source: str | Path,
    target: str | Path,
    *,
    timeout: float = 180.0,
) -> Path:
    """Converte DWG em DXF 2018 ASCII usando o AutoCAD Core Console."""
    source = Path(source).resolve()
    target = Path(target).resolve()
    if source.suffix.lower() != ".dwg":
        raise CadConversionError("A entrada do conversor precisa ser .dwg.")
    if not source.is_file():
        raise CadConversionError(f"DWG não encontrado: {source}")

    executable = find_accoreconsole()
    if executable is None:
        raise CadConversionError(
            "AutoCAD Core Console não encontrado. Configure "
            "ACCORECONSOLE_PATH ou exporte o DWG para DXF."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    isolated_profile = _autocad_isolated_profile()
    script = target.with_suffix(".dwg_to_dxf.scr")
    script.write_text(
        "\n".join((
            "_.FILEDIA",
            "0",
            "_.CMDDIA",
            "0",
            "_.SAVEAS",
            "_DXF",
            "16",
            f'"{_script_path(target)}"',
            "_.QUIT",
            "_Y",
            "",
        )),
        encoding="utf-8",
    )

    def run_console(*, isolated: bool):
        command = [str(executable)]
        if isolated:
            command.extend((
                "/isolate",
                _AUTOCAD_ISOLATE_USER,
                str(isolated_profile),
            ))
        command.extend(("/i", str(source), "/s", str(script)))
        return subprocess.run(
            command,
            cwd=str(target.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    attempts: list[tuple[str, subprocess.CompletedProcess]] = []
    global _AUTOCAD_ISOLATE_SUPPORTED
    try:
        with _AUTOCAD_LOCK:
            if _AUTOCAD_ISOLATE_SUPPORTED is not False:
                isolated_result = run_console(isolated=True)
                attempts.append(("perfil isolado", isolated_result))
                isolated_ok = (
                    isolated_result.returncode == 0
                    and target.is_file()
                    and target.stat().st_size >= 100
                )
                if isolated_ok:
                    _AUTOCAD_ISOLATE_SUPPORTED = True
                else:
                    # /isolate pode falhar ao criar a colmeia HKCU temporaria.
                    # O lock ja serializa as conversoes, portanto o perfil
                    # normal e um fallback seguro para a aplicacao local.
                    _AUTOCAD_ISOLATE_SUPPORTED = False
                    target.unlink(missing_ok=True)

            if not target.is_file() or target.stat().st_size < 100:
                normal_result = run_console(isolated=False)
                attempts.append(("perfil normal", normal_result))
    except subprocess.TimeoutExpired as exc:
        raise CadConversionError(
            f"Conversão DWG excedeu {timeout:.0f} segundos."
        ) from exc
    finally:
        script.unlink(missing_ok=True)
        shutil.rmtree(isolated_profile, ignore_errors=True)

    completed = attempts[-1][1]
    output = _decode_autocad_output(completed.stdout)
    if completed.returncode != 0 or not target.is_file() or target.stat().st_size < 100:
        details = []
        for label, result in attempts:
            decoded = _decode_autocad_output(result.stdout)
            tail = "\n".join(decoded.strip().splitlines()[-8:])
            if tail:
                details.append(f"[{label}]\n{tail}")
        detail = "\n".join(details)
        raise CadConversionError(
            "AutoCAD não produziu o DXF esperado."
            + (f"\n{detail}" if detail else "")
        )
    return target


def _convert_dwg_with_libredwg(
    source: str | Path,
    target: str | Path,
    *,
    timeout: float = 180.0,
) -> Path:
    """Converte DWG para DXF ASCII com ``dwgread`` do GNU LibreDWG."""
    source = Path(source).resolve()
    target = Path(target).resolve()
    executable = find_libredwg()
    if executable is None:
        raise CadConversionError("GNU LibreDWG (dwgread) não encontrado.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    command = [
        str(executable),
        "-v1",
        "-O", "DXF",
        "-o", str(target),
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(target.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env={**os.environ, "LC_ALL": "C.UTF-8"},
        )
    except subprocess.TimeoutExpired as exc:
        raise CadConversionError(
            f"Conversão DWG com LibreDWG excedeu {timeout:.0f} segundos."
        ) from exc
    if not target.is_file() or target.stat().st_size < 100:
        output = _decode_autocad_output(completed.stdout)
        tail = "\n".join(output.strip().splitlines()[-10:])
        raise CadConversionError(
            "LibreDWG não produziu o DXF esperado."
            + (f"\n{tail}" if tail else "")
        )
    # O pacote oficial win64 0.13.4 escreve CR-CR-LF no DXF ASCII. Isso
    # representa linhas em branco para leitores estritos como o ezdxf. A
    # versao Linux usada no Cloud Run nao apresenta o problema; normalizar
    # aqui mantem o mesmo contrato nas duas plataformas sem alterar geometria.
    raw = target.read_bytes()
    if b"\r\r\n" in raw:
        target.write_bytes(raw.replace(b"\r\r\n", b"\r\n"))
    return target


def convert_dwg_to_dxf_with_details(
    source: str | Path,
    target: str | Path,
    *,
    timeout: float = 180.0,
) -> tuple[Path, dict]:
    """Converte por AutoCAD ou LibreDWG e devolve auditoria da conversão."""
    source = Path(source).resolve()
    target = Path(target).resolve()
    if source.suffix.lower() != ".dwg":
        raise CadConversionError("A entrada do conversor precisa ser .dwg.")
    if not source.is_file():
        raise CadConversionError(f"DWG não encontrado: {source}")

    attempts = []
    engines = []
    if find_libredwg() is not None:
        engines.append(("gnu_libredwg", "GNU LibreDWG",
                        _convert_dwg_with_libredwg))
    if find_accoreconsole() is not None:
        engines.append(("autocad_core_console", "AutoCAD Core Console",
                        _convert_dwg_with_autocad))

    if not engines:
        raise CadConversionError(
            "Nenhum conversor DWG disponível. Configure ACCORECONSOLE_PATH "
            "no Windows ou instale GNU LibreDWG (dwgread) no Linux."
        )

    for engine, label, converter in engines:
        target.unlink(missing_ok=True)
        try:
            converted = converter(source, target, timeout=timeout)
            validation = _validate_dxf(converted)
            return converted, {
                "engine": engine,
                "label": label,
                "validation": validation,
            }
        except CadConversionError as exc:
            attempts.append(f"{label}: {exc}")

    target.unlink(missing_ok=True)
    raise CadConversionError(
        "Nenhum conversor conseguiu materializar um DXF válido.\n"
        + "\n".join(attempts)
    )


def convert_dwg_to_dxf(
    source: str | Path,
    target: str | Path,
    *,
    timeout: float = 180.0,
) -> Path:
    """API compatível: converte DWG e devolve o caminho do DXF validado."""
    converted, _details = convert_dwg_to_dxf_with_details(
        source, target, timeout=timeout,
    )
    return converted
