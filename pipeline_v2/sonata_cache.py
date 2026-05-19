"""Cache do resultado do Sonata por hash do PLY + voxel.

Sonata leva 5-15s por inferência. Cachear evita rodar 2× na mesma sessão
(ex: re-análise depois de alterar parâmetros downstream, ou múltiplas
chamadas do endpoint pro mesmo PLY).

Chave: sha1(ply_bytes[:1MB] + filesize + voxel_size + tag_versao)
Storage: pickles em OUTPUT_FOLDER/_sonata_cache/{hash}.pkl

Limpeza: por idade (TTL 7 dias) — chamar `cleanup_old()` periodicamente.
"""

import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Optional


# Versão do cache — incrementar quando schema do pickle mudar.
# Evita ler cache antigo com formato incompatível.
CACHE_VERSION = "v1"

# Onde guardar. Resolve em runtime contra OUTPUT_FOLDER do backend.
# Pra não importar de app_obb (ciclo), aceitamos override por env var.
_DEFAULT_CACHE_DIR = Path(
    os.environ.get("BIM_SONATA_CACHE")
    or (Path(os.environ.get("TEMP", "/tmp")) / "bim_outputs" / "_sonata_cache")
)


def _cache_dir() -> Path:
    d = _DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def compute_key(ply_path: str, voxel_size: float, sample_bytes: int = 1024 * 1024) -> str:
    """Hash determinístico do PLY + voxel.

    Lê só os primeiros `sample_bytes` (1MB default) por velocidade — basta
    pra discriminar arquivos diferentes; colisão é improvável pra PLYs reais.
    """
    p = Path(ply_path)
    if not p.exists():
        raise FileNotFoundError(f"PLY não existe: {ply_path}")

    h = hashlib.sha1()
    h.update(CACHE_VERSION.encode())
    h.update(str(round(voxel_size, 4)).encode())
    h.update(str(p.stat().st_size).encode())

    with open(p, "rb") as f:
        h.update(f.read(sample_bytes))

    return h.hexdigest()[:16]   # 64-bit é suficiente


def get(ply_path: str, voxel_size: float) -> Optional[dict]:
    """Tenta carregar cache pra (ply, voxel). Retorna None se miss."""
    try:
        key = compute_key(ply_path, voxel_size)
    except FileNotFoundError:
        return None

    cache_file = _cache_dir() / f"{key}.pkl"
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        # Cache corrompido — remove e considera miss
        print(f"[sonata_cache] cache corrompido em {cache_file.name}: {e}, removendo")
        cache_file.unlink(missing_ok=True)
        return None


def set(ply_path: str, voxel_size: float, result: dict) -> str:
    """Salva resultado no cache. Retorna o path do arquivo gravado."""
    key = compute_key(ply_path, voxel_size)
    cache_file = _cache_dir() / f"{key}.pkl"

    # Escrita atômica: grava em .tmp e renomeia
    tmp_file = cache_file.with_suffix(".pkl.tmp")
    with open(tmp_file, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_file.replace(cache_file)

    return str(cache_file)


def cleanup_old(max_age_days: float = 7.0) -> int:
    """Remove caches mais velhos que `max_age_days`. Retorna nº removido."""
    cutoff = time.time() - max_age_days * 86400
    d = _cache_dir()
    removed = 0
    for p in d.glob("*.pkl"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def stats() -> dict:
    """Diagnóstico: quantos arquivos, tamanho total, mais antigo."""
    d = _cache_dir()
    files = list(d.glob("*.pkl"))
    if not files:
        return {"count": 0, "total_mb": 0.0, "oldest_age_days": None}
    sizes = [f.stat().st_size for f in files]
    mtimes = [f.stat().st_mtime for f in files]
    now = time.time()
    return {
        "count": len(files),
        "total_mb": round(sum(sizes) / 1024 / 1024, 2),
        "oldest_age_days": round((now - min(mtimes)) / 86400, 1),
        "newest_age_days": round((now - max(mtimes)) / 86400, 1),
    }


if __name__ == "__main__":
    # Smoke test / inspeção do cache atual
    print(f"Cache dir: {_cache_dir()}")
    print(f"Stats: {stats()}")
