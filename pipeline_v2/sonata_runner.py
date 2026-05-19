"""Sonata runner: chama o serviço Sonata via subprocess e gerencia cache.

O Sonata vive em venv próprio (`experiments/sonata/venv_sonata/`) com PyTorch
2.5 + CUDA 12.4 + spconv. O backend principal (`app_obb.py`) NÃO importa
sonata diretamente — chama via subprocess e troca dados via pickle em disco.

Padrão:
    result = run_sonata(ply_path, voxel=0.15)
    # result tem o mesmo schema que sonata_serve.py grava
"""

import os
import pickle
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from . import sonata_cache


# Resolve paths relativos à raiz do projeto Beckend
_BASE = Path(__file__).resolve().parent.parent
SONATA_DIR = _BASE / "experiments" / "sonata"
SONATA_PY = SONATA_DIR / "venv_sonata" / "Scripts" / "python.exe"
SONATA_REPO = SONATA_DIR / "repo"
SONATA_SERVE = SONATA_DIR / "sonata_serve.py"


class SonataError(Exception):
    """Falha no subprocess Sonata."""
    pass


def _sanity_check_env() -> None:
    """Valida que o venv do Sonata e os arquivos existem antes de chamar."""
    if not SONATA_PY.exists():
        raise SonataError(
            f"Sonata venv não encontrado: {SONATA_PY}\n"
            "Execute o setup em experiments/sonata/ antes de usar."
        )
    if not SONATA_SERVE.exists():
        raise SonataError(f"sonata_serve.py não encontrado: {SONATA_SERVE}")
    if not SONATA_REPO.exists():
        raise SonataError(f"sonata repo não encontrado: {SONATA_REPO}")


def run_sonata(
    ply_path: str,
    voxel: float = 0.15,
    dbscan_eps: Optional[float] = None,
    dbscan_min: int = 15,
    timeout: float = 180.0,
    use_cache: bool = True,
    verbose: bool = False,
) -> dict:
    """Roda Sonata sobre `ply_path` e retorna dict com pred/conf/instances.

    Args:
        ply_path: caminho do PLY
        voxel: voxel size pro downsample (default 0.15m; Faro denso usa 0.3)
        dbscan_eps: distância DBSCAN. None = 2.5×voxel automaticamente.
        dbscan_min: pts mínimos pra cluster
        timeout: segundos máximos pro subprocess (default 3min)
        use_cache: se True, retorna cache hit imediatamente quando disponível
        verbose: print de progresso

    Returns:
        dict com keys: pts_voxel, pred, confidence, instances, voxel_size,
        class_names, time_seconds, dbscan_eps, dbscan_min, [from_cache]

    Raises:
        SonataError: se subprocess falhar, timeout, ou pickle corrompido
    """
    ply_path = str(Path(ply_path).resolve())

    # 1. Cache hit?
    if use_cache:
        cached = sonata_cache.get(ply_path, voxel)
        if cached is not None:
            cached["from_cache"] = True
            if verbose:
                print(f"[sonata_runner] cache HIT pra {Path(ply_path).name} voxel={voxel}m")
            return cached

    # 2. Cache miss → roda subprocess
    _sanity_check_env()

    if verbose:
        print(f"[sonata_runner] cache MISS — invocando subprocess")

    out_pkl = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pkl", prefix="sonata_out_"
    ).name

    try:
        cmd = [
            str(SONATA_PY),
            str(SONATA_SERVE),
            "--ply", ply_path,
            "--out", out_pkl,
            "--voxel", str(voxel),
            "--dbscan_min", str(dbscan_min),
            "--quiet",  # subprocess silencioso; logs vão pra runner se quiser
        ]
        if dbscan_eps is not None:
            cmd += ["--dbscan_eps", str(dbscan_eps)]

        # Ambiente do subprocess: PYTHONPATH precisa apontar pro repo do Sonata
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SONATA_REPO)
        env["PYTHONIOENCODING"] = "utf-8"
        # Mitiga fragmentação de VRAM (RTX 3050 6GB é apertado)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        t0 = time.time()
        proc = subprocess.run(
            cmd, env=env, timeout=timeout,
            capture_output=True, text=True, errors="replace",
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            raise SonataError(
                f"sonata_serve.py exit code {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout[-2000:]}\n"
                f"STDERR:\n{proc.stderr[-2000:]}"
            )

        if not Path(out_pkl).exists():
            raise SonataError(
                f"subprocess terminou OK mas pickle não foi criado: {out_pkl}"
            )

        # 3. Lê resultado
        with open(out_pkl, "rb") as f:
            result = pickle.load(f)

        result["from_cache"] = False
        result["subprocess_seconds"] = elapsed

        # 4. Salva cache
        if use_cache:
            sonata_cache.set(ply_path, voxel, result)

        if verbose:
            n_pts = len(result.get("pts_voxel", []))
            n_inst = len(result.get("instances", []))
            print(f"[sonata_runner] OK  {n_pts:,} pts  {n_inst} instâncias  {elapsed:.1f}s")

        return result

    except subprocess.TimeoutExpired:
        raise SonataError(
            f"subprocess Sonata estourou timeout de {timeout}s pra {Path(ply_path).name}"
        )
    finally:
        # Limpa arquivo temporário (cache já guarda em outro lugar)
        try:
            Path(out_pkl).unlink(missing_ok=True)
        except OSError:
            pass


def is_available() -> bool:
    """Verifica se Sonata pode ser executado. Útil pra fallback gracioso."""
    try:
        _sanity_check_env()
        return True
    except SonataError:
        return False


if __name__ == "__main__":
    # Smoke test
    import sys
    if len(sys.argv) < 2:
        print("Uso: python -m pipeline_v2.sonata_runner <ply_path> [voxel]")
        print(f"\nSonata disponível: {is_available()}")
        print(f"Cache stats: {sonata_cache.stats()}")
        sys.exit(0)

    ply = sys.argv[1]
    voxel = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    result = run_sonata(ply, voxel=voxel, verbose=True)
    print(f"\nfrom_cache: {result.get('from_cache')}")
    print(f"pts: {len(result['pts_voxel']):,}")
    print(f"instâncias: {len(result['instances'])}")
