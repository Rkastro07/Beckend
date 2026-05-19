"""Features v2: 11 features antigas (geométricas) + 6 novas (Sonata-aware).

As 11 antigas vêm de `_extrair_features_ml` em `app_obb.py:2666`. Pra evitar
duplicação, importamos a função existente e adicionamos 6 features extras
em cima.

Features novas (Sonata-aware):
  [11] class_purity       — frac de pts no OBB com classe esperada
  [12] class_confidence   — média da confiança Sonata pros pts da classe certa
  [13] bbox_size_ratio    — log(scan_volume / ifc_volume), clipped
  [14] centroid_offset_x  — desvio espacial em X normalizado pela bbox IFC
  [15] centroid_offset_y  — desvio espacial em Y normalizado
  [16] centroid_offset_z  — desvio espacial em Z normalizado

Essas features só fazem sentido quando há `scan_inst` matched a um `ifc_obj`
(via Frente C). Pra IFC objects sem match (AUSENTE), retornamos None
e o orchestrator decide o status sem RF.

Pra IFC objects de tipos sem cobertura Sonata (Column/Beam/etc.), os pontos
não têm classificação útil; voltamos pra features v1 puras (11 antigas).
"""

import math
from typing import Optional
import numpy as np

from .class_mapping import ifc_types_for_class, IFC_TO_SCANNET, CLASS_NAMES


# ============================================================
# Schema das features
# ============================================================
N_FEATS_V1 = 11
N_FEATS_V2 = 17
NEW_FEATURE_NAMES = (
    "class_purity",
    "class_confidence",
    "bbox_size_ratio",
    "centroid_offset_x",
    "centroid_offset_y",
    "centroid_offset_z",
)


# ============================================================
# Features novas (Sonata-aware)
# ============================================================
def compute_sonata_features(
    scan_inst: dict,
    ifc_obj: dict,
    sonata_pts: np.ndarray,
    sonata_pred: np.ndarray,
    sonata_conf: np.ndarray,
    obb_margin: float = 0.3,
) -> np.ndarray:
    """6 features Sonata-aware pra um match scan↔IFC.

    Args:
        scan_inst: instance do sonata_serve.py (post reclassify_ceiling)
        ifc_obj: objeto do extrair_objetos_por_pavimento
        sonata_pts: nuvem global voxelizada (pts_voxel do Sonata)
        sonata_pred: predição por ponto (class_id ScanNet)
        sonata_conf: confiança por ponto
        obb_margin: margem em metros pra considerar pts "dentro" do bbox IFC

    Returns:
        np.ndarray (6,) float32
    """
    # 1. Filtra pontos dentro do bbox IFC + margem
    bbox = ifc_obj["bbox"]
    mask = (
        (sonata_pts[:, 0] >= bbox["xmin"] - obb_margin) &
        (sonata_pts[:, 0] <= bbox["xmax"] + obb_margin) &
        (sonata_pts[:, 1] >= bbox["ymin"] - obb_margin) &
        (sonata_pts[:, 1] <= bbox["ymax"] + obb_margin) &
        (sonata_pts[:, 2] >= bbox["zmin"] - obb_margin) &
        (sonata_pts[:, 2] <= bbox["zmax"] + obb_margin)
    )
    pts_in_bbox = sonata_pts[mask]
    pred_in_bbox = sonata_pred[mask]
    conf_in_bbox = sonata_conf[mask]

    n_total = len(pts_in_bbox)

    # 2. Classes Sonata compatíveis com este IFC type
    # IFC type → classe Sonata esperada (inverso do mapping)
    expected_classes = IFC_TO_SCANNET.get(ifc_obj["tipo"], [])

    if not expected_classes or n_total == 0:
        # Sem classe esperada (não deveria acontecer aqui — caller já filtrou)
        # ou sem pontos no bbox
        class_purity = 0.0
        class_conf_mean = 0.0
    else:
        # Mask de pts com class certa via lookup de id no CLASS_NAMES
        expected_class_ids = [CLASS_NAMES.index(c) for c in expected_classes if c in CLASS_NAMES]
        # 'ceiling' não está em CLASS_NAMES — sonata classifica como 'floor'
        # então pra ceiling, expected_class_ids também inclui id de 'floor' (=1)
        if "ceiling" in expected_classes and CLASS_NAMES.index("floor") not in expected_class_ids:
            expected_class_ids.append(CLASS_NAMES.index("floor"))

        if expected_class_ids:
            mask_correct = np.isin(pred_in_bbox, expected_class_ids)
            n_correct = int(mask_correct.sum())
            class_purity = n_correct / n_total
            class_conf_mean = (
                float(conf_in_bbox[mask_correct].mean()) if n_correct > 0 else 0.0
            )
        else:
            class_purity = 0.0
            class_conf_mean = 0.0

    # 3. bbox_size_ratio: volume scan vs volume IFC (log, clipped)
    v_scan = max(float(scan_inst["volume"]), 1e-6)
    v_ifc  = max(
        float(ifc_obj.get("volume_ifc") or (
            (bbox["xmax"] - bbox["xmin"]) *
            (bbox["ymax"] - bbox["ymin"]) *
            (bbox["zmax"] - bbox["zmin"])
        )),
        1e-6,
    )
    bbox_size_ratio = max(-2.0, min(2.0, math.log(v_scan / v_ifc)))

    # 4. Centroid offsets normalizados pela bbox IFC
    bw = max(bbox["xmax"] - bbox["xmin"], 1e-6)
    bd = max(bbox["ymax"] - bbox["ymin"], 1e-6)
    bh = max(bbox["zmax"] - bbox["zmin"], 1e-6)
    scan_c = np.asarray(scan_inst["centroid"], dtype=np.float64)
    ifc_cx = (bbox["xmin"] + bbox["xmax"]) * 0.5
    ifc_cy = (bbox["ymin"] + bbox["ymax"]) * 0.5
    ifc_cz = (bbox["zmin"] + bbox["zmax"]) * 0.5

    # Offset normalizado: 0 = perfeitamente alinhado, ±1 = na borda da bbox
    offset_x = float((scan_c[0] - ifc_cx) / bw)
    offset_y = float((scan_c[1] - ifc_cy) / bd)
    offset_z = float((scan_c[2] - ifc_cz) / bh)

    return np.array([
        class_purity,
        class_conf_mean,
        bbox_size_ratio,
        offset_x,
        offset_y,
        offset_z,
    ], dtype=np.float32)


def compute_features_v2(
    scan_inst: dict,
    ifc_obj: dict,
    features_v1: np.ndarray,
    sonata_pts: np.ndarray,
    sonata_pred: np.ndarray,
    sonata_conf: np.ndarray,
    obb_margin: float = 0.3,
) -> np.ndarray:
    """Features completas v2 (17 = 11 v1 + 6 novas).

    Args:
        scan_inst: instance Sonata matched
        ifc_obj: IFC obj matched
        features_v1: features 11-d do `_extrair_features_ml` antigo
        sonata_pts/pred/conf: nuvem global Sonata

    Returns:
        np.ndarray (17,) float32, na ordem:
          [0-10]: features v1 (completeness_r, is_empty, height_fill, ...)
          [11-16]: features novas (class_purity, class_confidence, ...)
    """
    if features_v1.shape != (N_FEATS_V1,):
        raise ValueError(f"features_v1 deve ser shape ({N_FEATS_V1},), got {features_v1.shape}")

    feats_new = compute_sonata_features(
        scan_inst, ifc_obj,
        sonata_pts, sonata_pred, sonata_conf,
        obb_margin=obb_margin,
    )
    return np.concatenate([features_v1, feats_new]).astype(np.float32)


def get_feature_names_v2() -> list[str]:
    """Nomes ordenados das 17 features v2 (debug / análise)."""
    return [
        # v1
        "completeness_r", "is_empty", "height_fill",
        "z_bottom_norm", "z_top_norm", "centroid_z_norm",
        "xy_spread_norm", "density_area",
        "tipo", "eixo", "bh",
        # v2
        *NEW_FEATURE_NAMES,
    ]


if __name__ == "__main__":
    # Smoke test
    import numpy as np

    scan_inst = {
        "class_name": "wall",
        "centroid": np.array([5.0, 3.0, 1.5]),
        "bbox": {"xmin": 0, "xmax": 10, "ymin": 2.8, "ymax": 3.2, "zmin": 0, "zmax": 3.0},
        "volume": 12.0,
        "n_pts": 1500,
        "mean_conf": 0.85,
    }
    ifc_obj = {
        "tipo": "IfcWall",
        "guid": "w1",
        "nome": "Wall",
        "bbox": {"xmin": 0.1, "xmax": 9.9, "ymin": 2.85, "ymax": 3.15, "zmin": 0, "zmax": 3.0},
        "volume_ifc": 11.5,
    }

    # Simula Sonata: 200 pts dentro do bbox, classificados como wall (id=0)
    rng = np.random.default_rng(42)
    pts_in = rng.uniform(
        low=[0.1, 2.85, 0],
        high=[9.9, 3.15, 3.0],
        size=(200, 3),
    )
    pts_out = rng.uniform(low=[20, 20, 0], high=[30, 30, 3], size=(50, 3))  # ruído
    sonata_pts = np.vstack([pts_in, pts_out]).astype(np.float32)
    sonata_pred = np.zeros(250, dtype=np.int32)  # tudo 'wall'
    sonata_conf = np.full(250, 0.85, dtype=np.float32)

    feats_v1 = np.array([0.5, 0.0, 0.8, 0.0, 1.0, 0.5, 0.1, 50.0, 0.0, 0.0, 3.0], dtype=np.float32)
    feats_v2 = compute_features_v2(
        scan_inst, ifc_obj, feats_v1, sonata_pts, sonata_pred, sonata_conf
    )

    print(f"features_v2 shape: {feats_v2.shape}")
    print(f"Feature names + values:")
    for name, val in zip(get_feature_names_v2(), feats_v2):
        print(f"  {name:20s} = {val:.4f}")

    # Validações
    assert feats_v2.shape == (17,), f"shape errado: {feats_v2.shape}"
    assert feats_v2[11] > 0.5, "class_purity deveria ser alta (>0.5)"  # 200 wall / (200 dentro)
    assert feats_v2[12] > 0.8, "class_confidence deveria ser alta"
    assert abs(feats_v2[14]) < 0.1, "centroid_offset_x deveria estar perto de 0"
    print("\nValidações OK")
