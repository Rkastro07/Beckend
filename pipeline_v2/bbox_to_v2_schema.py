"""Adapta o JSON do bbox_features.py para o schema esperado pelo front v2.

Front v2 espera:
  - estatisticas: {total, completos, parciais, ausentes, adicoes, iniciados, progresso_geral}
  - resultados: [{guid, nome, tipo, status, bbox, obb_corners, n_pts, match_info, ...}]
  - adicoes: [...] (vazio nesse pipeline)
  - meta, global_cloud, etc

Mapeamento:
  verdict_simples  -> status
    construido     -> COMPLETO
    divergencia    -> PARCIAL
    nao_construido -> AUSENTE

Pos-processamento: PROPAGACAO REGRESSIVA
  Apos verdicts diretos, AUSENTE cujo bbox esta >=80% contido dentro de
  um COMPLETO direto vira COMPLETO (inherited). Cascateia ate estabilizar.
  Direcao: grande -> pequeno (so pequenos herdam dos grandes que os contem).
"""
from typing import Dict, List

import numpy as np


# Mapping verdict bbox_features -> status object esperado pelo front
VERDICT_TO_STATUS = {
    "construido":     {"code": "COMPLETO", "texto": "Construído",     "emoji": "✅", "cor": "#22c55e"},
    "divergencia":    {"code": "PARCIAL",  "texto": "Divergência",    "emoji": "⚠️", "cor": "#f97316"},
    "nao_construido": {"code": "AUSENTE",  "texto": "Não construído", "emoji": "❌", "cor": "#ef4444"},
}
# Variante "inherited" pra elementos que herdaram COMPLETO via propagacao
STATUS_INHERITED = {
    "code": "COMPLETO", "texto": "Construído (inferido)",
    "emoji": "🔗", "cor": "#10b981",  # verde-esmeralda (um pouco diferente)
}

# Limiar de containment: vol(intersect)/vol(menor) >= 0.80 -> "esta dentro"
CONTAINMENT_THRESHOLD = 0.80


def _corners_ifc_to_threejs(corners_ifc):
    """Converte 8 cantos IFC (Z-up) para Three.js (Y-up + flip X)."""
    cantos = np.asarray(corners_ifc, dtype=float)
    if cantos.size == 0:
        return None
    tj = np.empty_like(cantos)
    tj[:, 0] = -cantos[:, 0]
    tj[:, 1] = cantos[:, 2]
    tj[:, 2] = -cantos[:, 1]
    return tj.tolist()


def _bbox_volume(b):
    return max(0.0, b["xmax"] - b["xmin"]) * \
           max(0.0, b["ymax"] - b["ymin"]) * \
           max(0.0, b["zmax"] - b["zmin"])


def _bbox_intersect_volume(a, b):
    """Volume da interseccao AABB de a e b (0 se nao se tocam)."""
    dx = max(0.0, min(a["xmax"], b["xmax"]) - max(a["xmin"], b["xmin"]))
    dy = max(0.0, min(a["ymax"], b["ymax"]) - max(a["ymin"], b["ymin"]))
    dz = max(0.0, min(a["zmax"], b["zmax"]) - max(a["zmin"], b["zmin"]))
    return dx * dy * dz


def _propagar_regressivamente(resultados):
    """Propaga COMPLETO direto pra AUSENTE cujo bbox esta contido (>=80%).

    Itera ate estabilizar (cascata). Marca 'inherited_from' no status.
    Direcao: grande -> pequeno (so promove se vol_pai > vol_filho).

    Modifica `resultados` in-place.
    Retorna (n_promoted, n_iterations).
    """
    # Pre-calcula volumes pra evitar recomputar
    vols = []
    for r in resultados:
        b = r.get("bbox") or {}
        vols.append(_bbox_volume(b) if b else 0.0)

    n_promoted_total = 0
    iteration = 0
    MAX_ITER = 10  # safety, geralmente converge em 2-3

    while iteration < MAX_ITER:
        iteration += 1
        promoted_this_iter = 0

        # Lista atual de "fontes de propagacao" (COMPLETO, inclusive os inherited)
        completos_idx = [i for i, r in enumerate(resultados)
                         if r["status"]["code"] == "COMPLETO"]

        for i, r in enumerate(resultados):
            if r["status"]["code"] != "AUSENTE":
                continue
            bbox_filho = r.get("bbox") or {}
            vol_filho = vols[i]
            if vol_filho <= 0:
                continue

            # Procura algum pai que contenha esse filho
            for j in completos_idx:
                if j == i:
                    continue
                vol_pai = vols[j]
                if vol_pai <= vol_filho:
                    continue  # so grande -> pequeno
                bbox_pai = resultados[j].get("bbox") or {}
                if not bbox_pai:
                    continue
                inter = _bbox_intersect_volume(bbox_filho, bbox_pai)
                if inter / vol_filho >= CONTAINMENT_THRESHOLD:
                    # Promove
                    new_status = dict(STATUS_INHERITED)
                    new_status["inherited_from"] = resultados[j].get("guid", "")
                    new_status["inherited_from_tipo"] = resultados[j].get("tipo", "")
                    r["status"] = new_status
                    promoted_this_iter += 1
                    break

        n_promoted_total += promoted_this_iter
        if promoted_this_iter == 0:
            break  # estabilizou

    return n_promoted_total, iteration


def adaptar(result_bbox: dict) -> dict:
    """Recebe dict do bbox_features (carregado do JSON), retorna schema v2."""
    elementos = result_bbox.get("elementos", [])
    stats_glob = result_bbox.get("stats_globais", {})

    # 1. Monta resultados com verdicts diretos
    resultados: List[Dict] = []
    for el in elementos:
        verdict = el.get("verdict_simples", "nao_construido")
        status = dict(VERDICT_TO_STATUS.get(verdict, VERDICT_TO_STATUS["nao_construido"]))

        corners_tj = _corners_ifc_to_threejs(el.get("corners"))
        bbox = el.get("bbox") or {}

        match_info = {
            "classe_dominante":  el.get("classe_dominante"),
            "classes_aceitas":   el.get("classes_aceitas", []),
            "pct_classes":       el.get("pct_classes", {}),
            "n_pts_dentro":      el.get("n_pts_dentro", 0),
            "n_pts_min":         el.get("n_pts_min", 0),
            "vol_obb_m3":        el.get("vol_obb_m3", 0),
            "verdict":           verdict,
        }

        resultados.append({
            "guid":             el.get("guid", ""),
            "nome":             el.get("name") or el.get("tipo", ""),
            "tipo":             el.get("tipo", ""),
            "status":           status,
            "bbox":             bbox,
            "bbox_normalized":  None,
            "obb_corners":      corners_tj,
            "rf_version":       "sonata_bbox_v1",
            "match_info":       match_info,
            "json_file":        el.get("json_file"),
            "n_pts":            int(el.get("n_pts_dentro", 0)),
        })

    # 2. Propagacao regressiva (grande COMPLETO -> pequeno AUSENTE dentro)
    n_inherited, n_iter = _propagar_regressivamente(resultados)
    print(f"[adapter] propagacao: {n_inherited} elementos herdados COMPLETO "
          f"em {n_iter} iteracoes (containment>={CONTAINMENT_THRESHOLD})")

    # 3. Estatisticas finais (depois da propagacao)
    n_completo = sum(1 for r in resultados if r["status"]["code"] == "COMPLETO")
    n_parcial  = sum(1 for r in resultados if r["status"]["code"] == "PARCIAL")
    n_ausente  = sum(1 for r in resultados if r["status"]["code"] == "AUSENTE")
    total = len(resultados)

    estatisticas = {
        "total":           total,
        "completos":       n_completo,
        "parciais":        n_parcial,
        "ausentes":        n_ausente,
        "adicoes":         0,
        "iniciados":       0,
        "inherited":       n_inherited,  # NOVO: pra UI poder mostrar separado
        "progresso_geral": round(
            (n_completo + n_parcial * 0.5) / max(total, 1) * 100, 1
        ),
    }

    meta = {
        "pipeline":          "sonata_bbox+propagacao",
        "tempo_sonata_s":    stats_glob.get("tempo_sonata_s"),
        "subprocess_seconds": result_bbox.get("_subprocess_seconds"),
        "ply":               stats_glob.get("ply"),
        "ifc":               stats_glob.get("ifc"),
        "pavimento":         stats_glob.get("pavimento"),
        "verdicts_raw":      stats_glob.get("verdicts", {}),
        "output_dir":        result_bbox.get("_output_dir"),
        "global_cloud":      stats_glob.get("global_cloud"),
        "propagacao":        {
            "n_inherited":          n_inherited,
            "n_iter":               n_iter,
            "containment_threshold": CONTAINMENT_THRESHOLD,
        },
    }

    return {
        "estatisticas": estatisticas,
        "resultados":   resultados,
        "adicoes":      [],
        "meta":         meta,
    }
