"""Experimental Raster2Seq bridge for the local Planta-to-BIM editor.

The neural runtime intentionally stays outside the production Python runtime.
On the Windows development machine it is executed inside WSL and returns the
official Raster2Seq JSON plus the exact square image seen by the model.  The
functions that translate that JSON into the editor contract are pure Python so
they remain testable without PyTorch, CUDA or WSL.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import posixpath
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

try:
    from .cad_raster_ocr import extract_raster_dimensions
except ImportError:  # suporte a execução direta pelos scripts legados
    from cad_raster_ocr import extract_raster_dimensions


RASTER2SEQ_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
RASTER2SEQ_CANVAS_SIZE = 256
RASTER2SEQ_CACHE_VERSION = b"cubicasa5k-poly2seq-v1\0"

CC5K_LABELS = {
    0: "Exterior",
    1: "Cozinha",
    2: "Sala",
    3: "Quarto",
    4: "Banheiro",
    5: "Entrada",
    6: "Depósito",
    7: "Garagem",
    8: "Indefinido",
    9: "Janela",
    10: "Porta",
}


class Raster2SeqUnavailable(RuntimeError):
    """Raised when the optional local WSL runtime cannot be used."""


def _raster2seq_cache_paths(image_path: Path) -> tuple[str, Path, Path]:
    digest = hashlib.sha256()
    digest.update(RASTER2SEQ_CACHE_VERSION)
    with image_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    cache_key = digest.hexdigest()
    default_root = Path(__file__).resolve().parents[1] / ".runtime" / "cache" / "raster2seq"
    cache_root = Path(os.environ.get("RASTER2SEQ_CACHE_ROOT", default_root))
    cache_dir = cache_root / cache_key
    return cache_key, cache_dir / "predictions.json", cache_dir / "processed.png"


def _load_raster2seq_cache(
    image_path: Path,
) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]] | None:
    cache_key, json_path, processed_image_path = _raster2seq_cache_paths(image_path)
    if not json_path.is_file() or not processed_image_path.is_file():
        return None
    try:
        predictions = json.loads(json_path.read_text(encoding="utf-8"))
        processed_image = processed_image_path.read_bytes()
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(predictions, list) or not processed_image:
        return None
    return predictions, processed_image, {
        "cache_hit": True,
        "cache_key": cache_key,
        "json_path": str(json_path),
        "preview_path": None,
        "processed_image_path": str(processed_image_path),
        "stdout_tail": "Raster2Seq reutilizado do cache local.",
    }


def _polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2.0


def _clean_points(raw_points: Any) -> list[list[float]]:
    if not isinstance(raw_points, list):
        return []
    points: list[list[float]] = []
    for item in raw_points:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        point = [x, y]
        if not points or point != points[-1]:
            points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def _world_points(
    points: Iterable[Iterable[float]],
    *,
    canvas_size: int,
    canvas_width_m: float,
) -> list[list[float]]:
    scale = canvas_width_m / float(canvas_size)
    return [
        [round(float(x) * scale, 5), round((canvas_size - float(y)) * scale, 5)]
        for x, y in points
    ]


def predictions_to_editor_model(
    predictions: list[dict[str, Any]],
    *,
    image_base64: str,
    source_name: str,
    canvas_width_m: float = 20.0,
    canvas_size: int = RASTER2SEQ_CANVAS_SIZE,
) -> dict[str, Any]:
    """Translate official Raster2Seq JSON to a non-destructive editor overlay."""
    if not image_base64:
        raise ValueError("A imagem processada pelo Raster2Seq está vazia.")
    if not math.isfinite(canvas_width_m) or canvas_width_m <= 0:
        raise ValueError("A largura de referência precisa ser positiva.")
    if canvas_size <= 0:
        raise ValueError("O tamanho do canvas precisa ser positivo.")

    rooms: list[dict[str, Any]] = []
    openings: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        try:
            category_id = int(prediction.get("category_id", 8))
        except (TypeError, ValueError):
            category_id = 8
        raw_points = _clean_points(prediction.get("segmentation"))
        minimum = 2 if category_id in (9, 10) else 3
        if len(raw_points) < minimum:
            continue
        world = _world_points(
            raw_points,
            canvas_size=canvas_size,
            canvas_width_m=canvas_width_m,
        )
        item = {
            "id": str(prediction.get("id", index)),
            "category_id": category_id,
            "label": CC5K_LABELS.get(category_id, f"Classe {category_id}"),
            "points": world,
        }
        if category_id in (9, 10):
            item["kind"] = "window" if category_id == 9 else "door"
            openings.append(item)
        else:
            item["area"] = round(_polygon_area(world), 4)
            rooms.append(item)

    bounds = [0.0, 0.0, float(canvas_width_m), float(canvas_width_m)]
    return {
        "ok": True,
        "escala": canvas_width_m / float(canvas_size),
        "single_line": False,
        "nome": Path(source_name).stem,
        "bbox": {
            "xmin": bounds[0],
            "ymin": bounds[1],
            "xmax": bounds[2],
            "ymax": bounds[3],
        },
        "diagnostico": {
            "sobras": 0,
            "cantos_costurados": 0,
            "blocos_esquadria": len(openings),
            "elementos_lidos": len(rooms) + len(openings),
            "geometrias_aproximadas": len(rooms),
        },
        "source": {
            "format": Path(source_name).suffix.lower().lstrip(".") or "image",
            "family": "raster",
            "mode": "raster2seq-overlay",
            "semantic_level": "experimental",
            "scale_source": "user-canvas-width",
        },
        "reference": {
            "kind": "raster2seq",
            "label": "Previsão Raster2Seq",
            "bounds": bounds,
            "image_mime": "image/png",
            "image_base64": image_base64,
            "canvas_size": [canvas_size, canvas_size],
            "canvas_width_m": canvas_width_m,
            "rooms": rooms,
            "openings": openings,
        },
        "warnings": [
            "A IA propôs ambientes e esquadrias como referência visual; "
            "as paredes BIM continuam editáveis e precisam ser confirmadas.",
            "A largura informada escala o canvas completo, incluindo margens brancas.",
        ],
        "paredes": [],
        "aberturas": [],
        "laje": {
            "contorno": [],
            "piso": {"ativo": False, "espessura": 0.12},
            "teto": {"ativo": False, "espessura": 0.12},
        },
        "spaces": [],
    }


def _wsl_path(path: Path, *, wsl_executable: str) -> str:
    # Calling ``wsl.exe wslpath C:\...`` directly loses backslashes while WSL
    # translates argv. Quoting the path inside bash preserves the Windows path.
    command = f"wslpath -a {shlex.quote(str(path.resolve()))}"
    result = subprocess.run(
        [wsl_executable, "bash", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def run_raster2seq_wsl(
    image_path: Path,
    output_dir: Path,
    *,
    timeout_seconds: int = 600,
) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    """Run the official CubiCasa5K checkpoint in the configured local WSL."""
    cached = _load_raster2seq_cache(image_path)
    if cached is not None:
        return cached
    if os.name != "nt":
        raise Raster2SeqUnavailable(
            "O protótipo Raster2Seq local está configurado para Windows + WSL."
        )
    wsl_executable = shutil.which("wsl") or shutil.which("wsl.exe")
    if not wsl_executable:
        raise Raster2SeqUnavailable("WSL não encontrado nesta máquina.")

    runtime_root = os.environ.get("RASTER2SEQ_WSL_ROOT", "/home/rafael/.cache/codex-r2s-01")
    python_path = os.environ.get("RASTER2SEQ_WSL_PYTHON", "/home/rafael/mask3d_env/bin/python")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_wsl = _wsl_path(image_path, wsl_executable=wsl_executable)
    # ``image_wsl`` is a Linux path even though this function runs on Windows.
    # pathlib.Path would turn it back into Windows separators here.
    input_wsl = posixpath.dirname(image_wsl)
    output_wsl = _wsl_path(output_dir, wsl_executable=wsl_executable)

    python_args = [
        python_path,
        "predict.py",
        "--dataset_name=cubicasa",
        f"--dataset_root={input_wsl}",
        "--checkpoint=hf:cubicasa5k",
        f"--output_dir={output_wsl}",
        "--semantic_classes=12",
        "--input_channels=3",
        "--poly2seq",
        "--seq_len=512",
        "--num_bins=32",
        "--disable_poly_refine",
        "--dec_attn_concat_src",
        "--per_token_sem_loss",
        "--use_anchor",
        "--ema4eval",
        "--save_pred",
        "--batch_size=1",
        "--num_workers=0",
    ]
    environment = {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": (
            "/home/rafael/mask3d_env/lib/python3.10/site-packages/torch/lib:"
            "/usr/local/cuda/lib64"
        ),
        "PYTHONPATH": f"{runtime_root}/deps:{runtime_root}/models/ops:.",
    }
    exports = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in environment.items()
    )
    command = (
        f"cd {shlex.quote(runtime_root)} && {exports} "
        + " ".join(shlex.quote(value) for value in python_args)
    )
    result = subprocess.run(
        [wsl_executable, "bash", "-lc", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "erro desconhecido").strip()
        raise Raster2SeqUnavailable(f"Falha no Raster2Seq local: {detail[-2000:]}")

    result_root = output_dir / "cubicasa5k"
    json_path = result_root / "jsons" / f"{image_path.stem}.json"
    processed_image_path = result_root / f"{image_path.stem}.png"
    preview_path = result_root / f"{image_path.stem}_pred_floorplan.png"
    if not json_path.is_file() or not processed_image_path.is_file():
        detail = (result.stderr or result.stdout or "sem saída do processo").strip()
        raise Raster2SeqUnavailable(
            "O Raster2Seq terminou sem produzir o JSON ou a imagem alinhada esperada: "
            f"{detail[-2000:]}"
        )

    predictions = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise Raster2SeqUnavailable("O JSON do Raster2Seq não contém uma lista de previsões.")
    processed_image = processed_image_path.read_bytes()
    cache_key, cache_json_path, cache_image_path = _raster2seq_cache_paths(image_path)
    cache_json_path.parent.mkdir(parents=True, exist_ok=True)
    cache_json_path.write_text(
        json.dumps(predictions, ensure_ascii=False),
        encoding="utf-8",
    )
    cache_image_path.write_bytes(processed_image)
    metadata = {
        "cache_hit": False,
        "cache_key": cache_key,
        "stdout_tail": result.stdout[-2000:],
        "json_path": str(json_path),
        "preview_path": str(preview_path) if preview_path.is_file() else None,
        "processed_image_path": str(processed_image_path),
    }
    return predictions, processed_image, metadata


def raster2seq_image_to_editor_model(
    image_path: Path,
    output_dir: Path,
    *,
    canvas_width_m: float = 20.0,
) -> dict[str, Any]:
    predictions, processed_image, metadata = run_raster2seq_wsl(
        image_path,
        output_dir,
    )
    model = predictions_to_editor_model(
        predictions,
        image_base64=base64.b64encode(processed_image).decode("ascii"),
        source_name=image_path.name,
        canvas_width_m=canvas_width_m,
    )
    dimensions, ocr_diagnostic = extract_raster_dimensions(
        image_path,
        canvas_width_m=canvas_width_m,
        canvas_size=RASTER2SEQ_CANVAS_SIZE,
    )
    model["reference"]["dimensions"] = dimensions
    model["raster_ocr"] = ocr_diagnostic
    if dimensions:
        model["warnings"].append(
            f"OCR encontrou {len(dimensions)} cota(s) candidata(s); confirme cada valor "
            "contra uma parede antes de recalibrar a planta."
        )
    model["raster2seq_runtime"] = metadata
    return model
