"""bbox-features runner: chama experiments/sonata/bbox_features.py via subprocess.

Padrao igual ao sonata_runner — o backend nao importa torch/sonata, chama o
script standalone que vive no venv_sonata.

Output: JSON gerado em disco (mesmo formato que bbox_features.py escreve).
"""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional


_BASE = Path(__file__).resolve().parent.parent
SONATA_DIR = _BASE / "experiments" / "sonata"
SONATA_PY = SONATA_DIR / "venv_sonata" / "Scripts" / "python.exe"
SONATA_REPO = SONATA_DIR / "repo"
BBOX_FEATURES = SONATA_DIR / "bbox_features.py"


class BBoxRunnerError(Exception):
    pass


def _sanity_check_env() -> None:
    if not SONATA_PY.exists():
        raise BBoxRunnerError(f"Sonata venv nao encontrado: {SONATA_PY}")
    if not BBOX_FEATURES.exists():
        raise BBoxRunnerError(f"bbox_features.py nao encontrado: {BBOX_FEATURES}")
    if not SONATA_REPO.exists():
        raise BBoxRunnerError(f"sonata repo nao encontrado: {SONATA_REPO}")


def run_bbox_features(
    ply_path: str,
    ifc_path: str,
    pavimento: Optional[str] = None,
    output_dir: Optional[str] = None,
    timeout: float = 300.0,
    verbose: bool = False,
    semantic_align: bool = False,
) -> dict:
    """Roda bbox_features.py e retorna o JSON parseado.

    Args:
        ply_path: caminho do PLY de entrada
        ifc_path: caminho do IFC
        pavimento: nome do pavimento, None ou "__TODOS__" pra todos
        output_dir: pasta de saida (JSON + PLY/OBJs de viz). None = temp dir
        timeout: segundos maximos pro subprocess
        verbose: log de progresso
        semantic_align: ativa ICP por classe Sonata (T por wall/floor/ceiling)

    Returns: dict com a estrutura do JSON gerado por bbox_features.py
    """
    _sanity_check_env()

    ply_path = str(Path(ply_path).resolve())
    ifc_path = str(Path(ifc_path).resolve())
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="bbox_features_")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    pav = pavimento or "__TODOS__"

    cmd = [
        str(SONATA_PY),
        str(BBOX_FEATURES),
        ply_path,
        ifc_path,
        str(output_dir),
        pav,
    ]
    if semantic_align:
        cmd.append("--semantic-align")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SONATA_REPO)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if verbose:
        print(f"[bbox_runner] subprocess: {' '.join(cmd)}")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, env=env, timeout=timeout,
            capture_output=True, text=True, errors="replace",
        )
    except subprocess.TimeoutExpired:
        raise BBoxRunnerError(
            f"bbox_features.py estourou timeout de {timeout}s"
        )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        raise BBoxRunnerError(
            f"bbox_features.py exit={proc.returncode}\n"
            f"STDOUT:\n{proc.stdout[-2000:]}\n"
            f"STDERR:\n{proc.stderr[-2000:]}"
        )

    # JSON gerado no output_dir com nome derivado do PLY
    stem = Path(ply_path).stem
    json_path = Path(output_dir) / f"{stem}_sonata_bbox.json"
    if not json_path.exists():
        raise BBoxRunnerError(
            f"subprocess terminou OK mas JSON nao foi criado: {json_path}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    result["_subprocess_seconds"] = elapsed
    result["_output_dir"] = str(output_dir)
    result["_json_path"] = str(json_path)

    if verbose:
        n = result["stats_globais"]["n_elementos"]
        print(f"[bbox_runner] OK  {n} elementos  {elapsed:.1f}s")

    return result


def is_available() -> bool:
    try:
        _sanity_check_env()
        return True
    except BBoxRunnerError:
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Uso: python -m pipeline_v2.bbox_runner <ply> <ifc> [pavimento] [out_dir]")
        print(f"Disponivel: {is_available()}")
        sys.exit(0)
    r = run_bbox_features(
        sys.argv[1], sys.argv[2],
        pavimento=sys.argv[3] if len(sys.argv) > 3 else None,
        output_dir=sys.argv[4] if len(sys.argv) > 4 else None,
        verbose=True,
    )
    print(f"Elementos: {r['stats_globais']['n_elementos']}")
    print(f"Verdicts: {r['stats_globais']['verdicts']}")
