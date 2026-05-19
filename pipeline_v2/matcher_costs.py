"""Funções de custo pra matching scan-instance ↔ ifc-object.

Custo combinado de 3 componentes:
  1. Compatibilidade de classe (penalidade alta se incompatível)
  2. Distância de centroide (preferência por objetos próximos)
  3. Diferença de tamanho (volume / dimensões — preferência por similares)

O Hungarian (`matcher_hungarian.py`) usa essas funções pra montar a matriz
de custo e minimizar o total.

Schemas esperados:
  scan_inst: dict do sonata_serve.py com keys:
    - class_name (str)
    - centroid (np.ndarray (3,))
    - bbox (dict xmin/xmax/ymin/ymax/zmin/zmax)
    - volume (float)
    - n_pts (int)
    - mean_conf (float)

  ifc_obj: dict do extrair_objetos_por_pavimento com keys:
    - tipo (str, ex: "IfcWall")
    - bbox (dict xmin/xmax/ymin/ymax/zmin/zmax)
    - volume_ifc (float)
    - guid, nome
"""

import math
import numpy as np

from .class_mapping import is_compatible, ifc_types_for_class


# Constantes de pesos — calibradas por inspeção visual.
W_DIST  = 1.0    # peso da distância de centroide (em metros)
W_SIZE  = 2.0    # peso da diferença de tamanho (em log-units)
W_CLASS = 1000.0 # penalidade brutal pra classe incompatível
                 # qualquer par com classe diferente fica >= 1000 → fora do match


def _bbox_center(bbox: dict) -> np.ndarray:
    """Centro do bbox AABB. Funciona tanto pra scan_inst quanto ifc_obj."""
    return np.array([
        (bbox["xmin"] + bbox["xmax"]) * 0.5,
        (bbox["ymin"] + bbox["ymax"]) * 0.5,
        (bbox["zmin"] + bbox["zmax"]) * 0.5,
    ], dtype=np.float64)


def _bbox_volume(bbox: dict) -> float:
    """Volume do bbox AABB."""
    dx = max(bbox["xmax"] - bbox["xmin"], 1e-6)
    dy = max(bbox["ymax"] - bbox["ymin"], 1e-6)
    dz = max(bbox["zmax"] - bbox["zmin"], 1e-6)
    return dx * dy * dz


def cost_class_compat(scan_class: str, ifc_type: str) -> float:
    """Componente CLASSE: 0 se compatível, W_CLASS senão."""
    # 'ceiling' é virtual (do reclassify_ceiling) — compatível com slab/covering
    if scan_class == "ceiling":
        return 0.0 if ifc_type in ifc_types_for_class("ceiling") else W_CLASS
    return 0.0 if is_compatible(scan_class, ifc_type) else W_CLASS


def cost_centroid_dist(scan_inst: dict, ifc_obj: dict) -> float:
    """Componente DISTÂNCIA: euclidiana 3D em metros entre centroides."""
    scan_c = np.asarray(scan_inst["centroid"], dtype=np.float64)
    ifc_c  = _bbox_center(ifc_obj["bbox"])
    return float(np.linalg.norm(scan_c - ifc_c))


def cost_size_diff(scan_inst: dict, ifc_obj: dict) -> float:
    """Componente TAMANHO: log-ratio entre volumes, clipped.

    log(v1/v2) é simétrico — não importa se scan é maior ou menor.
    Clip em [0, 2] limita influência de outliers (volumes 100× diferentes).
    """
    v_scan = max(float(scan_inst["volume"]), 1e-6)
    v_ifc  = max(float(ifc_obj.get("volume_ifc") or _bbox_volume(ifc_obj["bbox"])), 1e-6)
    log_ratio = abs(math.log(v_scan / v_ifc))
    return min(log_ratio, 2.0)


def total_cost(scan_inst: dict, ifc_obj: dict,
               w_dist: float = W_DIST,
               w_size: float = W_SIZE,
               w_class: float = W_CLASS) -> float:
    """Custo combinado pra um par (scan, ifc).

    Quanto menor, melhor o match. Pares com classe incompatível têm custo
    >= w_class (1000 default), praticamente impossíveis.

    Retorna float; matriz de custos pro Hungarian é montada chamando isso
    pra cada par.
    """
    c_cls = cost_class_compat(scan_inst["class_name"], ifc_obj["tipo"])
    if c_cls >= w_class:
        # Otimização: se classe incompatível, não precisa calcular o resto
        return c_cls

    c_dist = cost_centroid_dist(scan_inst, ifc_obj)
    c_size = cost_size_diff(scan_inst, ifc_obj)

    return w_dist * c_dist + w_size * c_size


def build_cost_matrix(scan_instances: list[dict],
                       ifc_objects: list[dict]) -> np.ndarray:
    """Constrói matriz de custos N_scan × N_ifc.

    Returns:
        np.ndarray shape (n_scan, n_ifc) com custos pra cada par.
    """
    n_scan = len(scan_instances)
    n_ifc  = len(ifc_objects)
    if n_scan == 0 or n_ifc == 0:
        return np.empty((n_scan, n_ifc), dtype=np.float64)

    C = np.empty((n_scan, n_ifc), dtype=np.float64)
    for i, si in enumerate(scan_instances):
        for j, oi in enumerate(ifc_objects):
            C[i, j] = total_cost(si, oi)
    return C


if __name__ == "__main__":
    # Smoke test
    scan_inst = {
        "class_name": "wall",
        "centroid": np.array([5.0, 3.0, 1.5]),
        "bbox": {"xmin": 0, "xmax": 10, "ymin": 2.8, "ymax": 3.2, "zmin": 0, "zmax": 3.0},
        "volume": 12.0,
        "n_pts": 1500,
        "mean_conf": 0.85,
    }

    ifc_wall = {
        "tipo": "IfcWall",
        "guid": "test_wall",
        "nome": "Test Wall",
        "bbox": {"xmin": 0.1, "xmax": 9.9, "ymin": 2.85, "ymax": 3.15, "zmin": 0, "zmax": 3.0},
        "volume_ifc": 11.5,
    }

    ifc_wall_far = {
        "tipo": "IfcWall",
        "guid": "test_wall_far",
        "nome": "Test Wall Far",
        "bbox": {"xmin": 50, "xmax": 60, "ymin": 50, "ymax": 51, "zmin": 0, "zmax": 3.0},
        "volume_ifc": 30.0,
    }

    ifc_door = {
        "tipo": "IfcDoor",
        "guid": "test_door",
        "nome": "Test Door",
        "bbox": {"xmin": 5.0, "xmax": 5.9, "ymin": 2.9, "ymax": 3.1, "zmin": 0, "zmax": 2.1},
        "volume_ifc": 0.4,
    }

    print("Custos:")
    print(f"  wall ~ wall próximo: {total_cost(scan_inst, ifc_wall):.3f}  (esperado: baixo)")
    print(f"  wall ~ wall longe:   {total_cost(scan_inst, ifc_wall_far):.3f}  (esperado: alto)")
    print(f"  wall ~ porta:        {total_cost(scan_inst, ifc_door):.3f}  (esperado: >= 1000)")

    print()
    print("Componentes do match bom:")
    print(f"  class:    {cost_class_compat(scan_inst['class_name'], ifc_wall['tipo'])}")
    print(f"  centroid: {cost_centroid_dist(scan_inst, ifc_wall):.3f}m")
    print(f"  size:     {cost_size_diff(scan_inst, ifc_wall):.3f}")
