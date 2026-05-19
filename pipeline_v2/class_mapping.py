"""Mapeamento entre classes ScanNet (Sonata) e tipos IFC do nosso domínio.

ScanNet 20 classes vs IFC types:
  - Algumas batem direto (wall→IfcWall, door→IfcDoor)
  - Outras têm fallback geométrico (floor/ceiling diferenciados por altura Z)
  - Tipos IFC sem cobertura no ScanNet (Column, Beam, Stair, etc.) seguem
    pipeline V1 (features geométricas + RF v1) — não passam pelo Hungarian.

Estratégia:
  1. SCANNET_TO_IFC: classe Sonata → lista de IFC types compatíveis
  2. IFC_TO_SCANNET: lookup inverso pra checagem de compatibilidade
  3. IFC_SEM_COBERTURA_SONATA: tipos que pulam o Hungarian
  4. reclassify_ceiling: heurística pra distinguir floor vs ceiling pelo Z
"""

import numpy as np
from typing import Iterable, Optional


# ============================================================
# ScanNet 20 — ordem oficial (índice = class_id que o Sonata produz)
# Espelho do que está em experiments/sonata/sonata_serve.py CLASS_NAMES.
# Replicado aqui pra evitar import cross-venv (sonata_serve só roda em venv_sonata).
# ============================================================
CLASS_NAMES: tuple[str, ...] = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower_curtain", "toilet", "sink", "bathtub",
    "otherfurniture",
)


# ============================================================
# Mapeamento principal
# ============================================================
SCANNET_TO_IFC: dict[str, list[str]] = {
    "wall":   ["IfcWall", "IfcRailing"],      # railing também é vertical fino
    "floor":  ["IfcSlab", "IfcCovering"],     # forro = placa horizontal igual laje
    "door":   ["IfcDoor"],
    "window": ["IfcWindow"],
    # Classes ScanNet sem mapeamento direto:
    # "ceiling" → derivado de "floor" via reclassify_ceiling
    # "cabinet", "bed", "chair", "sofa", "table", "bookshelf", "picture",
    # "counter", "desk", "curtain", "refrigerator", "shower_curtain",
    # "toilet", "sink", "bathtub", "otherfurniture" → IFC não tem equivalentes diretos
    # (esses pontos são ruído, ignorar no matching)
}

# Classes Sonata que mapeiam pra algum IFC (úteis pro matching)
SCANNET_MATCHABLE = set(SCANNET_TO_IFC.keys())


# Lookup inverso: IFC type → classes Sonata compatíveis
def _build_inverse() -> dict[str, list[str]]:
    inv: dict[str, list[str]] = {}
    for sn_cls, ifc_types in SCANNET_TO_IFC.items():
        for t in ifc_types:
            inv.setdefault(t, []).append(sn_cls)
    return inv


IFC_TO_SCANNET: dict[str, list[str]] = _build_inverse()


# ============================================================
# Tipos IFC sem cobertura no Sonata
# Esses NÃO entram no Hungarian — pipeline V1 cuida.
# ============================================================
IFC_SEM_COBERTURA_SONATA: set[str] = {
    "IfcColumn",
    "IfcBeam",
    "IfcStair",
    "IfcRoof",
    "IfcMember",
    "IfcPlate",
    "IfcSanitaryTerminal",
}


# ============================================================
# Helpers de compatibilidade
# ============================================================
def is_compatible(sonata_class: str, ifc_type: str) -> bool:
    """True se a classe Sonata casa com o tipo IFC."""
    return ifc_type in SCANNET_TO_IFC.get(sonata_class, [])


def is_matchable_ifc(ifc_type: str) -> bool:
    """True se esse tipo IFC entra no Hungarian (tem cobertura Sonata)."""
    return ifc_type not in IFC_SEM_COBERTURA_SONATA and ifc_type in IFC_TO_SCANNET


def compatible_ifc_types(sonata_class: str) -> list[str]:
    """Lista de tipos IFC compatíveis com a classe Sonata."""
    return SCANNET_TO_IFC.get(sonata_class, [])


# ============================================================
# Heurística floor vs ceiling
# ScanNet 20 não tem 'ceiling' — Sonata classifica forros como 'floor'.
# Pra distinguir, usamos a altura Z relativa ao bbox do prédio.
# ============================================================
def reclassify_ceiling(
    instances: list[dict],
    scene_bbox_z: tuple[float, float],
    ceiling_z_threshold: float = 0.65,
) -> list[dict]:
    """Re-classifica instâncias 'floor' como 'ceiling' quando estão no alto.

    Args:
        instances: lista do output do sonata_serve.py
        scene_bbox_z: (z_min, z_max) do bbox global da cena
        ceiling_z_threshold: fração da altura acima da qual 'floor' → 'ceiling'
            default 0.65 = 65% da altura (qualquer plano horizontal nessa zona
            é mais provavelmente teto/forro)

    Returns:
        Lista nova de instâncias (não muta in-place). Mantém class_id original
        no campo `class_id_original` e adiciona `class_name = 'ceiling'` quando
        re-classificado.
    """
    z_min, z_max = scene_bbox_z
    z_range = max(z_max - z_min, 1e-6)
    z_cutoff = z_min + z_range * ceiling_z_threshold

    result = []
    for inst in instances:
        inst_new = dict(inst)
        if inst.get("class_name") == "floor":
            centroid_z = float(inst["centroid"][2])
            if centroid_z >= z_cutoff:
                inst_new["class_name"] = "ceiling"
                inst_new["class_id_original"] = inst["class_id"]
                # ceiling mapeia pros mesmos IFC types que floor (Slab/Covering)
                # Note: class_id continua o mesmo (do ScanNet) — só o nome muda.
        result.append(inst_new)

    return result


# ============================================================
# Mapeamento estendido com ceiling derivada
# Após reclassify_ceiling, alguns inst têm class_name='ceiling'.
# Esses casam com os mesmos IFC types que floor.
# ============================================================
def ifc_types_for_class(class_name: str) -> list[str]:
    """Versão estendida que entende 'ceiling' (não nativa ScanNet)."""
    if class_name == "ceiling":
        return SCANNET_TO_IFC["floor"]  # Slab/Covering
    return SCANNET_TO_IFC.get(class_name, [])


# ============================================================
# Filtragem de instâncias relevantes pro matching
# ============================================================
def filter_matchable_instances(instances: list[dict]) -> list[dict]:
    """Mantém só instâncias com classe que tem mapeamento IFC.

    Descarta cadeiras, mesas, sofás, etc. (ruído pra nosso domínio BIM).
    Aceita 'ceiling' derivada.
    """
    return [
        inst for inst in instances
        if inst.get("class_name") in SCANNET_MATCHABLE or inst.get("class_name") == "ceiling"
    ]


def filter_matchable_ifc_objects(ifc_objects: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separa objetos IFC entre 'vai pro Hungarian' e 'fica pro pipeline V1'.

    Returns:
        (matchable, sem_cobertura) — matchable entra no Hungarian; sem_cobertura
        é processado separado pelo pipeline V1 (features geométricas + RF v1).
    """
    matchable = []
    sem_cobertura = []
    for obj in ifc_objects:
        tipo = obj.get("tipo", "")
        if tipo in IFC_SEM_COBERTURA_SONATA:
            sem_cobertura.append(obj)
        elif tipo in IFC_TO_SCANNET:
            matchable.append(obj)
        else:
            # Tipo IFC desconhecido — joga em sem_cobertura por segurança
            sem_cobertura.append(obj)
    return matchable, sem_cobertura


if __name__ == "__main__":
    # Smoke test / inspeção
    print("=== SCANNET → IFC ===")
    for sn_cls, ifcs in SCANNET_TO_IFC.items():
        print(f"  {sn_cls:15s} → {ifcs}")
    print()
    print("=== IFC sem cobertura Sonata (vão pro pipeline V1) ===")
    for t in sorted(IFC_SEM_COBERTURA_SONATA):
        print(f"  {t}")
    print()
    print("=== Testes ===")
    print(f"is_compatible('wall', 'IfcWall')      = {is_compatible('wall', 'IfcWall')}")
    print(f"is_compatible('wall', 'IfcDoor')      = {is_compatible('wall', 'IfcDoor')}")
    print(f"is_compatible('floor', 'IfcCovering') = {is_compatible('floor', 'IfcCovering')}")
    print(f"is_matchable_ifc('IfcWall')           = {is_matchable_ifc('IfcWall')}")
    print(f"is_matchable_ifc('IfcColumn')         = {is_matchable_ifc('IfcColumn')}")
    print(f"compatible_ifc_types('floor')         = {compatible_ifc_types('floor')}")
    print(f"ifc_types_for_class('ceiling')        = {ifc_types_for_class('ceiling')}")
