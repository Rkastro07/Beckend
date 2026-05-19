"""Hungarian matching scan-instances ↔ ifc-objects com threshold.

Estratégia:
  1. Filtra IFC pra tirar objetos sem cobertura Sonata (vão pro pipeline V1)
  2. Constrói matriz de custo (matcher_costs.build_cost_matrix)
  3. linear_sum_assignment (Hungarian/Munkres) → atribuição 1:1 ótima global
  4. Aplica threshold por tipo IFC — pares com custo alto viram unmatched
  5. Categoriza resultados:
     - matches[]:        pares válidos
     - unmatched_scan[]: instâncias scan sem IFC → status ADICAO
     - unmatched_ifc[]:  objetos IFC sem scan → status AUSENTE

Hungarian é polinomial (~O(n³)) e robusto. scipy.optimize.linear_sum_assignment
roda em milissegundos pra centenas de objetos.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .class_mapping import filter_matchable_ifc_objects, filter_matchable_instances
from .matcher_costs import build_cost_matrix, W_CLASS


# Threshold máximo de custo aceitável por tipo IFC.
# Pares com custo acima viram unmatched (não match real).
# Calibrado por inspeção: paredes grandes toleram offset maior; portas/janelas pequenas, menor.
THRESHOLDS_MAX = {
    "IfcWall":      3.0,    # parede pode estar até 3m fora do esperado
    "IfcSlab":      4.0,    # lajes grandes toleram mais
    "IfcCovering":  4.0,    # forros idem
    "IfcRailing":   2.0,
    "IfcDoor":      0.8,    # porta pequena: 80cm de offset = match perdido
    "IfcWindow":    0.8,
}
# Fallback pra tipos não listados
DEFAULT_THRESHOLD = 2.0


def _threshold_for(ifc_obj: dict) -> float:
    return THRESHOLDS_MAX.get(ifc_obj.get("tipo", ""), DEFAULT_THRESHOLD)


def match(scan_instances: list[dict],
          ifc_objects: list[dict]) -> dict:
    """Atribui instâncias scan a objetos IFC via Hungarian.

    Args:
        scan_instances: lista do sonata_serve.py, com `class_name`, `centroid`,
            `bbox`, `volume`, `n_pts`, `mean_conf`. JÁ DEVE TER PASSADO POR
            reclassify_ceiling e filter_matchable_instances.
        ifc_objects: lista de extrair_objetos_por_pavimento. Internamente
            filtra os sem cobertura Sonata.

    Returns:
        dict com:
          'matches': list[dict] — cada item tem:
              { 'scan_inst': dict, 'ifc_obj': dict, 'cost': float }
          'unmatched_scan': list[dict] — scan_inst sem par (status ADICAO)
          'unmatched_ifc':  list[dict] — ifc_obj sem par (status AUSENTE)
          'ifc_sem_cobertura': list[dict] — IFC types fora do Sonata (Column, etc.)
              vão pro pipeline V1 separadamente
          'stats': {
              'n_scan_total', 'n_ifc_total', 'n_ifc_matchable',
              'n_matches', 'n_unmatched_scan', 'n_unmatched_ifc',
              'avg_match_cost', 'max_match_cost'
          }
    """
    # 1. Separa IFC matchable (vai pro Hungarian) vs sem cobertura (pipeline V1)
    ifc_matchable, ifc_sem_cobertura = filter_matchable_ifc_objects(ifc_objects)

    n_scan = len(scan_instances)
    n_ifc  = len(ifc_matchable)

    stats = {
        "n_scan_total":      n_scan,
        "n_ifc_total":       len(ifc_objects),
        "n_ifc_matchable":   n_ifc,
        "n_ifc_sem_cobertura": len(ifc_sem_cobertura),
        "n_matches":         0,
        "n_unmatched_scan":  0,
        "n_unmatched_ifc":   0,
        "avg_match_cost":    0.0,
        "max_match_cost":    0.0,
    }

    # Casos degenerados
    if n_scan == 0:
        return {
            "matches": [],
            "unmatched_scan": [],
            "unmatched_ifc": list(ifc_matchable),
            "ifc_sem_cobertura": ifc_sem_cobertura,
            "stats": {**stats, "n_unmatched_ifc": n_ifc},
        }
    if n_ifc == 0:
        return {
            "matches": [],
            "unmatched_scan": list(scan_instances),
            "unmatched_ifc": [],
            "ifc_sem_cobertura": ifc_sem_cobertura,
            "stats": {**stats, "n_unmatched_scan": n_scan},
        }

    # 2. Matriz de custo
    C = build_cost_matrix(scan_instances, ifc_matchable)

    # 3. Hungarian
    # linear_sum_assignment minimiza C[rows[k], cols[k]] somado.
    # Trabalha com matriz retangular: emparelha min(n_scan, n_ifc) pares.
    rows, cols = linear_sum_assignment(C)

    matches = []
    matched_scan_idx: set[int] = set()
    matched_ifc_idx:  set[int] = set()

    # 4. Filtra por threshold
    for i, j in zip(rows, cols):
        cost = float(C[i, j])
        thr  = _threshold_for(ifc_matchable[j])

        # Compatibilidade de classe é "infinita" se incompatível (>= W_CLASS),
        # então cost >= W_CLASS já significa rejeitar.
        # Threshold por tipo cuida do resto.
        if cost < min(thr, W_CLASS - 1):
            matches.append({
                "scan_inst": scan_instances[i],
                "ifc_obj":   ifc_matchable[j],
                "cost":      cost,
            })
            matched_scan_idx.add(i)
            matched_ifc_idx.add(j)
        # senão: par é descartado, ambos viram unmatched

    # 5. Unmatched: o que sobrou
    unmatched_scan = [
        scan_instances[i] for i in range(n_scan) if i not in matched_scan_idx
    ]
    unmatched_ifc = [
        ifc_matchable[j] for j in range(n_ifc) if j not in matched_ifc_idx
    ]

    # 6. Stats
    costs = [m["cost"] for m in matches]
    stats.update({
        "n_matches":         len(matches),
        "n_unmatched_scan":  len(unmatched_scan),
        "n_unmatched_ifc":   len(unmatched_ifc),
        "avg_match_cost":    round(float(np.mean(costs)), 3) if costs else 0.0,
        "max_match_cost":    round(float(max(costs)), 3)   if costs else 0.0,
    })

    return {
        "matches":            matches,
        "unmatched_scan":     unmatched_scan,
        "unmatched_ifc":      unmatched_ifc,
        "ifc_sem_cobertura":  ifc_sem_cobertura,
        "stats":              stats,
    }


def match_and_summarize(scan_instances: list[dict],
                         ifc_objects: list[dict]) -> tuple[dict, str]:
    """Wrapper de conveniência: roda match() e devolve junto com um sumário pra log.

    Returns:
        (result_dict, summary_string)
    """
    result = match(scan_instances, ifc_objects)
    s = result["stats"]

    summary_lines = [
        f"Hungarian matching:",
        f"  Scan:  {s['n_scan_total']} instâncias",
        f"  IFC:   {s['n_ifc_total']} objetos ({s['n_ifc_matchable']} matchable, {s['n_ifc_sem_cobertura']} fora-Sonata)",
        f"  ✓ Matches:           {s['n_matches']:3d}  (avg cost {s['avg_match_cost']:.2f}, max {s['max_match_cost']:.2f})",
        f"  ⚠ Unmatched scan:    {s['n_unmatched_scan']:3d}  → ADICAO (construído fora do plano)",
        f"  ✗ Unmatched IFC:     {s['n_unmatched_ifc']:3d}  → AUSENTE (não construído)",
        f"  ↳ IFC sem cobertura: {s['n_ifc_sem_cobertura']:3d}  → pipeline V1 (colunas, vigas, etc.)",
    ]
    return result, "\n".join(summary_lines)


# ============================================================
# Helpers de validação (uso em testes unitários e debug)
# ============================================================
def validate_matches(matches: list[dict]) -> list[str]:
    """Retorna lista de problemas encontrados nos matches (pra debug)."""
    problems = []
    seen_scan = set()
    seen_ifc = set()
    for k, m in enumerate(matches):
        s_id = id(m["scan_inst"])
        i_id = id(m["ifc_obj"])
        if s_id in seen_scan:
            problems.append(f"match[{k}]: scan_inst repetido (já apareceu em outro match)")
        if i_id in seen_ifc:
            problems.append(f"match[{k}]: ifc_obj repetido")
        seen_scan.add(s_id)
        seen_ifc.add(i_id)
        if m["cost"] >= W_CLASS - 1:
            problems.append(f"match[{k}]: cost={m['cost']} sugere classe incompatível passou pelo filtro")
    return problems


if __name__ == "__main__":
    # Smoke test com dados sintéticos
    import numpy as np

    scan = [
        {
            "class_name": "wall",
            "centroid": np.array([5.0, 3.0, 1.5]),
            "bbox": {"xmin": 0, "xmax": 10, "ymin": 2.8, "ymax": 3.2, "zmin": 0, "zmax": 3.0},
            "volume": 12.0, "n_pts": 1500, "mean_conf": 0.85,
        },
        {
            "class_name": "floor",
            "centroid": np.array([5.0, 0.0, 0.0]),
            "bbox": {"xmin": 0, "xmax": 10, "ymin": -5, "ymax": 5, "zmin": -0.1, "zmax": 0.1},
            "volume": 10.0, "n_pts": 5000, "mean_conf": 0.80,
        },
        {
            "class_name": "wall",
            "centroid": np.array([100.0, 100.0, 1.5]),   # bem longe — sem par bom
            "bbox": {"xmin": 95, "xmax": 105, "ymin": 99, "ymax": 101, "zmin": 0, "zmax": 3},
            "volume": 60.0, "n_pts": 200, "mean_conf": 0.7,
        },
    ]

    ifc = [
        {"tipo": "IfcWall", "guid": "w1", "nome": "Wall A",
         "bbox": {"xmin": 0.1, "xmax": 9.9, "ymin": 2.85, "ymax": 3.15, "zmin": 0, "zmax": 3.0},
         "volume_ifc": 11.5},
        {"tipo": "IfcSlab", "guid": "s1", "nome": "Slab A",
         "bbox": {"xmin": 0, "xmax": 10, "ymin": -5, "ymax": 5, "zmin": 0, "zmax": 0.1},
         "volume_ifc": 5.0},
        {"tipo": "IfcWall", "guid": "w2", "nome": "Wall B (sem par)",
         "bbox": {"xmin": 200, "xmax": 210, "ymin": 200, "ymax": 201, "zmin": 0, "zmax": 3.0},
         "volume_ifc": 30.0},
        {"tipo": "IfcColumn", "guid": "c1", "nome": "Pilar (fora-Sonata)",
         "bbox": {"xmin": 1, "xmax": 1.5, "ymin": 1, "ymax": 1.5, "zmin": 0, "zmax": 3.0},
         "volume_ifc": 0.75},
    ]

    result, summary = match_and_summarize(scan, ifc)
    print(summary)
    print()
    for m in result["matches"]:
        print(f"  MATCH  {m['scan_inst']['class_name']:8s} <-> {m['ifc_obj']['nome']}  cost={m['cost']:.2f}")
    for u in result["unmatched_scan"]:
        print(f"  ADICAO   {u['class_name']:8s}  centro=({u['centroid'][0]:.1f}, {u['centroid'][1]:.1f})")
    for u in result["unmatched_ifc"]:
        print(f"  AUSENTE  {u['tipo']:10s}  {u['nome']}")
    for u in result["ifc_sem_cobertura"]:
        print(f"  V1       {u['tipo']:10s}  {u['nome']}")

    problems = validate_matches(result["matches"])
    if problems:
        print("\n!!! Problemas:")
        for p in problems:
            print(f"  {p}")
    else:
        print("\nNenhum problema de validação.")
