"""Roteador RF v1 ↔ v2.

Carrega ambos modelos (se disponíveis) e roteia a inferência conforme o que
o caller consegue fornecer:

  - features_v2 disponível + modelo v2 carregado → usa v2
  - senão                                        → cai pro v1 (sempre presente)

Schema do pickle (v1 já existente, v2 segue mesmo formato):
    {
        'model':  sklearn.ensemble.RandomForestClassifier,
        'scaler': sklearn.preprocessing.StandardScaler,  # opcional (pode ser None)
        # extras pra v2:
        'feature_version': 'v1' | 'v2',
        'n_features':      11 | 17,
    }

Status codes (espelha _ML_STATUS de app_obb.py):
    0 = COMPLETO
    1 = PARCIAL
    2 = AUSENTE
"""

from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np


# ============================================================
# Paths default — alinhados com app_obb._ML_MODEL_PATH
# ============================================================
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_DEFAULT_V1_PATH = _REPO_ROOT / "ml" / "models" / "random_forest.pkl"
_DEFAULT_V2_PATH = _REPO_ROOT / "ml" / "models" / "random_forest_v2.pkl"

# Pode-se sobrescrever via env (CI / experimentos).
V1_PATH = Path(os.environ.get("PIPELINE_V2_RF_V1", str(_DEFAULT_V1_PATH)))
V2_PATH = Path(os.environ.get("PIPELINE_V2_RF_V2", str(_DEFAULT_V2_PATH)))


# Espelho do _ML_STATUS pra log / debug; o app_obb tem a versão "oficial"
# com cor/texto. Aqui mantemos só code↔int.
STATUS_INT_TO_CODE = {
    0: "COMPLETO",
    1: "PARCIAL",
    2: "AUSENTE",
}


# ============================================================
# Router singleton (thread-safe load lazy)
# ============================================================
class RFRouter:
    """Roteador stateful que carrega v1 e (se existir) v2 sob demanda.

    Use `get_router()` em vez de instanciar diretamente.
    """

    def __init__(self,
                 v1_path: Path = V1_PATH,
                 v2_path: Path = V2_PATH,
                 verbose: bool = False):
        self.v1_path = Path(v1_path)
        self.v2_path = Path(v2_path)
        self.verbose = verbose
        self._v1: Optional[dict] = None
        self._v2: Optional[dict] = None
        self._lock = threading.Lock()
        self._loaded = False

    # ----- carregamento -----
    def _load_once(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._v1 = self._try_load(self.v1_path, label="v1")
            self._v2 = self._try_load(self.v2_path, label="v2")
            self._loaded = True

    def _try_load(self, path: Path, label: str) -> Optional[dict]:
        if not path.exists():
            if self.verbose:
                print(f"RFRouter: {label} ausente em {path}")
            return None
        try:
            with open(path, "rb") as f:
                saved = pickle.load(f)
            if "model" not in saved:
                if self.verbose:
                    print(f"RFRouter: {label} sem chave 'model' — ignorado")
                return None
            if self.verbose:
                n_feat = saved.get("n_features", "?")
                ver    = saved.get("feature_version", "?")
                print(f"RFRouter: {label} carregado ({n_feat} feats, version={ver})")
            return saved
        except Exception as e:
            if self.verbose:
                print(f"RFRouter: falha ao carregar {label} ({path}): {e}")
            return None

    # ----- introspecção -----
    @property
    def has_v1(self) -> bool:
        self._load_once()
        return self._v1 is not None

    @property
    def has_v2(self) -> bool:
        self._load_once()
        return self._v2 is not None

    @property
    def info(self) -> dict:
        self._load_once()
        out = {"v1_available": self.has_v1, "v2_available": self.has_v2}
        if self._v1 is not None:
            out["v1_n_features"] = self._v1.get("n_features", 11)
        if self._v2 is not None:
            out["v2_n_features"] = self._v2.get("n_features", 17)
        return out

    # ----- inferência -----
    def predict(
        self,
        features_v1: np.ndarray,
        features_v2: Optional[np.ndarray] = None,
        prefer: str = "auto",
    ) -> tuple[int, str]:
        """Retorna (status_int, version_used).

        Args:
            features_v1: shape (11,) — sempre obrigatório (fallback).
            features_v2: shape (17,) — opcional; só usado se v2 carregado.
            prefer: "auto" (default; usa v2 se possível) | "v1" (força v1)
                | "v2" (força v2 — erro se indisponível).

        Returns:
            (status_int ∈ {0,1,2}, version_used ∈ {"v1", "v2"})
        """
        self._load_once()
        if self._v1 is None and self._v2 is None:
            raise RuntimeError(
                "RFRouter: nenhum modelo carregado (v1 e v2 ausentes)"
            )

        # ----- decide rota -----
        if prefer == "v2":
            if self._v2 is None:
                raise RuntimeError("RFRouter: prefer='v2' mas modelo v2 indisponível")
            return self._predict_with(self._v2, features_v2, label="v2")

        if prefer == "v1":
            if self._v1 is None:
                raise RuntimeError("RFRouter: prefer='v1' mas modelo v1 indisponível")
            return self._predict_with(self._v1, features_v1, label="v1")

        # auto
        if self._v2 is not None and features_v2 is not None:
            return self._predict_with(self._v2, features_v2, label="v2")
        if self._v1 is not None:
            return self._predict_with(self._v1, features_v1, label="v1")
        # v1 ausente, v2 sem features → erro
        raise RuntimeError(
            "RFRouter: v1 ausente e features_v2=None — sem como inferir"
        )

    def predict_proba(
        self,
        features_v1: np.ndarray,
        features_v2: Optional[np.ndarray] = None,
        prefer: str = "auto",
    ) -> tuple[np.ndarray, str]:
        """Igual a `predict` mas retorna distribuição de probabilidades.

        Returns:
            (proba shape (3,) float, version_used)
        """
        self._load_once()
        if prefer == "v2" or (prefer == "auto" and self._v2 is not None and features_v2 is not None):
            saved, feats, label = self._v2, features_v2, "v2"
        else:
            saved, feats, label = self._v1, features_v1, "v1"
        if saved is None:
            raise RuntimeError(f"RFRouter.predict_proba: modelo {label} indisponível")

        feats = self._prepare_features(saved, feats, label)
        model = saved["model"]
        if not hasattr(model, "predict_proba"):
            raise RuntimeError(f"RFRouter: modelo {label} não suporta predict_proba")
        proba = model.predict_proba(feats.reshape(1, -1))[0]
        return proba.astype(np.float32), label

    # ----- internos -----
    def _predict_with(self, saved: dict, feats: Optional[np.ndarray],
                       label: str) -> tuple[int, str]:
        feats = self._prepare_features(saved, feats, label)
        model = saved["model"]
        pred = int(model.predict(feats.reshape(1, -1))[0])
        return pred, label

    def _prepare_features(self, saved: dict, feats: Optional[np.ndarray],
                           label: str) -> np.ndarray:
        if feats is None:
            raise ValueError(f"RFRouter: features para {label} não fornecidas")
        expected = saved.get("n_features")
        if expected is not None and feats.shape != (expected,):
            raise ValueError(
                f"RFRouter ({label}): esperava shape ({expected},), got {feats.shape}"
            )
        scaler = saved.get("scaler")
        x = feats.astype(np.float32, copy=False)
        if scaler is not None:
            x = scaler.transform(x.reshape(1, -1))[0]
        return x


# ============================================================
# Singleton
# ============================================================
_ROUTER: Optional[RFRouter] = None
_ROUTER_LOCK = threading.Lock()


def get_router(verbose: bool = False) -> RFRouter:
    """Retorna o router compartilhado (carrega na primeira chamada)."""
    global _ROUTER
    if _ROUTER is None:
        with _ROUTER_LOCK:
            if _ROUTER is None:
                _ROUTER = RFRouter(verbose=verbose)
    return _ROUTER


def reset_router() -> None:
    """Força recarga (útil em testes / após retrain)."""
    global _ROUTER
    with _ROUTER_LOCK:
        _ROUTER = None


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    router = get_router(verbose=True)
    print()
    print("Info:", router.info)

    if router.has_v1:
        # Tenta uma predição com features dummy (todos zeros)
        feats = np.zeros(router._v1.get("n_features", 11), dtype=np.float32)
        try:
            pred, version = router.predict(feats)
            print(f"\nPredict dummy v1: status={pred} ({STATUS_INT_TO_CODE.get(pred)}), version={version}")
        except Exception as e:
            print(f"\nFalha no predict v1: {e}")

    if router.has_v2:
        feats_v1 = np.zeros(11, dtype=np.float32)
        feats_v2 = np.zeros(17, dtype=np.float32)
        try:
            pred, version = router.predict(feats_v1, feats_v2)
            print(f"\nPredict dummy v2: status={pred} ({STATUS_INT_TO_CODE.get(pred)}), version={version}")
        except Exception as e:
            print(f"\nFalha no predict v2: {e}")
    else:
        print("\n(v2 não carregado — só v1 ativo)")
