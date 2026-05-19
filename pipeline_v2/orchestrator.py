"""Pipeline v2 — orquestrador que cola Frentes B, C e D.

Entrada:
    ply_path, ifc_path, pavimento

Pipeline:
    1. Sonata (cached)                                  [B]
    2. reclassify_ceiling + filter_matchable            [B]
    3. extrair_objetos_por_pavimento (app_obb)          [existente]
    4. Hungarian matching scan↔IFC                      [C]
    5. Pra cada IFC matched: features v1 + v2 + RF      [D]
       Pra cada IFC sem cobertura Sonata: features v1 + RF v1
       Pra cada unmatched_scan: status ADICAO direto
       Pra cada unmatched_ifc: status AUSENTE direto

Saída:
    dict com {
        'objetos':  [...],     # 1 por IFC obj com status final
        'adicoes':  [...],     # construído não planejado
        'stats':    {...},
        'meta':     {...},
    }

Esse dict é convertido em JSON pelo endpoint `/api/analisar_ai_v2`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# pipeline_v2 internos
from . import sonata_runner, class_mapping, matcher_hungarian, rf_router
from .features_v2 import compute_sonata_features

# Importa app_obb (raiz) pra reutilizar extração IFC + features v1
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# Status codes (espelha _ML_STATUS de app_obb + adiciona ADICAO)
# ============================================================
STATUS_TABLE = {
    "COMPLETO": {"code": "COMPLETO", "texto": "Executado",                 "cor": "#22c55e"},
    "PARCIAL":  {"code": "PARCIAL",  "texto": "Parcialmente Executado",    "cor": "#f59e0b"},
    "AUSENTE":  {"code": "AUSENTE",  "texto": "Nao Executado",             "cor": "#ef4444"},
    "ADICAO":   {"code": "ADICAO",   "texto": "Construido Fora do Plano",  "cor": "#3b82f6"},
}
INT_TO_CODE = {0: "COMPLETO", 1: "PARCIAL", 2: "AUSENTE"}


# ============================================================
# Helpers
# ============================================================
def _load_ply(ply_path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Lê PLY → (pts, colors|None)."""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    colors = None
    if len(pcd.colors) > 0:
        colors = np.asarray(pcd.colors, dtype=np.float32)
    return pts, colors


def _status_dict(code: str) -> dict:
    """Lookup safe pra STATUS_TABLE."""
    return dict(STATUS_TABLE.get(code, STATUS_TABLE["AUSENTE"]))


# Cor RGB [0-1] por status (consumido pelo viewer Three.js)
_STATUS_COLOR_RGB = {
    "COMPLETO": [0.13, 0.77, 0.37],   # green-500
    "PARCIAL":  [0.96, 0.62, 0.04],   # amber-500
    "AUSENTE":  [0.94, 0.27, 0.27],   # red-500
    "ADICAO":   [0.23, 0.51, 0.96],   # sky-500
}


def _write_obj_cloud(
    pts_obj: "np.ndarray",
    status_code: str,
    out_dir: "Path",
    filename: str,
    max_pts_viewer: int = 50_000,
) -> Optional[str]:
    """Salva nuvem do objeto como JSON pro viewer Three.js.

    Returns path relativo (`<session_id>/<filename>`) ou None se não havia pts.
    """
    if pts_obj is None or len(pts_obj) == 0:
        return None
    # Subsample pra viewer (Three.js engasga > 100k pts/obj)
    if len(pts_obj) > max_pts_viewer:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(pts_obj), max_pts_viewer, replace=False)
        pts_view = pts_obj[idx]
    else:
        pts_view = pts_obj
    import app_obb
    pts_threejs = app_obb.converter_pontos_ifc_para_threejs(pts_view)
    payload = {
        "positions":   pts_threejs.flatten().tolist(),
        "color":       _STATUS_COLOR_RGB.get(status_code, [0.5, 0.5, 0.5]),
        "count":       int(len(pts_view)),
        "count_total": int(len(pts_obj)),
    }
    import json as _json
    out_path = out_dir / filename
    with open(out_path, "w") as f:
        _json.dump(payload, f)
    return f"{out_dir.name}/{filename}"


def _objeto_resultado(obj: dict, status_code: str, *,
                       version: Optional[str] = None,
                       match_info: Optional[dict] = None,
                       features: Optional[np.ndarray] = None,
                       json_file: Optional[str] = None,
                       n_pts: int = 0) -> dict:
    """Monta dict de saída pra um objeto IFC."""
    import app_obb
    bbox = obj.get("bbox") or {}
    out = {
        "guid":             obj.get("guid"),
        "tipo":             obj.get("tipo"),
        "nome":             obj.get("nome"),
        "bbox":             bbox,
        "bbox_normalized":  app_obb.converter_ifc_para_threejs(bbox) if bbox else None,
        "obb_corners":      app_obb.calcular_obb_corners_threejs(obj) if obj else None,
        "n_pts":            int(n_pts),
        "status":           _status_dict(status_code),
        "rf_version":       version,
        "match_info":       match_info,
        "json_file":        json_file,
    }
    if features is not None:
        out["features"] = [float(x) for x in features.tolist()]
    return out


# ============================================================
# Pipeline principal
# ============================================================
def run(
    ply_path: str,
    ifc_path: str,
    pavimento: str = "__TODOS__",
    *,
    sonata_voxel: float = 0.05,    # densidade ~10x maior que 0.15 (≈450k pts no Faro)
    sonata_dbscan_eps: Optional[float] = None,
    sonata_dbscan_min: int = 30,
    include_features_in_output: bool = False,
    output_dir: Optional[Path] = None,
    session_id: Optional[str] = None,
    max_pts_viewer: int = 50_000,
    max_pts_global: int = 200_000,
    verbose: bool = False,
) -> dict:
    """Roda o pipeline v2 completo.

    Args:
        ply_path: caminho do PLY.
        ifc_path: caminho do IFC.
        pavimento: nome do pavimento ou "__TODOS__".
        sonata_voxel: voxel size pro Sonata (default 0.15m).
        sonata_dbscan_eps: eps do DBSCAN; None = adaptativo.
        sonata_dbscan_min: min points por instância.
        include_features_in_output: se True, devolve as features no JSON
            (útil pra debug, mas pesado).
        verbose: logs por etapa.

    Returns:
        dict com chaves: 'objetos', 'adicoes', 'stats', 'meta'.
    """
    t_global = time.time()
    timings = {}
    # Prepara diretório de saída (pra escrever JSONs do viewer)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    from werkzeug.utils import secure_filename as _secure_filename

    # ---------- 1. Extrair objetos IFC ----------
    # Primeiro precisamos do IFC pra alinhar a nuvem (Sonata é coord-invariant
    # mas o matching Hungarian compara centroides — sem alinhar = 0 matches).
    t = time.time()
    import app_obb
    ifc_objects = app_obb.extrair_objetos_por_pavimento(
        ifc_path,
        pavimento_alvo=pavimento,
        incluir_estrutura_cruzando=False,
    )
    timings["ifc_extract_s"] = round(time.time() - t, 2)
    if verbose:
        print(f"[v2] (1/6) IFC: {len(ifc_objects)} objetos (pavimento={pavimento})")

    # ---------- 2. Alinhar PLY ao IFC ----------
    # Reusa o alinhador legado (alinhar_nuvem_com_ifc) — não é perfeito, mas
    # entrega coords aproximadamente IFC pra Hungarian conseguir matchar.
    # Frente A (GeoTransformer) trocará isso por AI alignment quando estiver pronta.
    t = time.time()
    pts_raw, colors_raw = _load_ply(ply_path)
    if verbose:
        print(f"[v2] (2/6) Alinhando {len(pts_raw):,} pts ao IFC...")
    pts_aligned, align_dbg = app_obb.alinhar_nuvem_com_ifc(pts_raw, ifc_objects)
    timings["alignment_s"] = round(time.time() - t, 2)
    if verbose:
        print(f"[v2]     align OK em {timings['alignment_s']}s")

    # Nuvem global (subsample p/ viewer) — escrita só se output_dir foi passado
    global_cloud_path: Optional[str] = None
    if output_dir is not None and len(pts_aligned) > 0:
        if len(pts_aligned) > max_pts_global:
            rng = np.random.default_rng(seed=42)
            idx_g = rng.choice(len(pts_aligned), max_pts_global, replace=False)
            pts_g = pts_aligned[idx_g]
        else:
            pts_g = pts_aligned
        pts_g_threejs = app_obb.converter_pontos_ifc_para_threejs(pts_g)
        import json as _json
        with open(output_dir / "_global.json", "w") as f:
            _json.dump({
                "positions":   pts_g_threejs.flatten().tolist(),
                "count":       int(len(pts_g)),
                "count_total": int(len(pts_aligned)),
                "color":       [0.45, 0.5, 0.55],   # cinza-azulado neutro
            }, f)
        global_cloud_path = f"{output_dir.name}/_global.json"
        if verbose:
            print(f"[v2]     _global.json salvo ({len(pts_g):,}/{len(pts_aligned):,} pts)")

    # Salva PLY alinhado num temp file pra alimentar o Sonata
    import tempfile
    import open3d as o3d
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(pts_aligned.astype(np.float64))
    if colors_raw is not None:
        aligned_pcd.colors = o3d.utility.Vector3dVector(colors_raw.astype(np.float64))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply",
                                       prefix=f"{Path(ply_path).stem}_aligned_")
    tmp.close()
    aligned_ply_path = tmp.name
    o3d.io.write_point_cloud(aligned_ply_path, aligned_pcd, write_ascii=False)

    # ---------- 3. Sonata sobre PLY alinhado ----------
    t = time.time()
    if verbose:
        print(f"[v2] (3/6) Sonata sobre PLY alinhado (voxel={sonata_voxel})")
    try:
        sonata_result = sonata_runner.run_sonata(
            aligned_ply_path,
            voxel=sonata_voxel,
            dbscan_eps=sonata_dbscan_eps,
            dbscan_min=sonata_dbscan_min,
            verbose=verbose,
        )
    finally:
        # Limpa PLY temp (cache da Sonata já guardou o resultado pelo hash)
        try:
            Path(aligned_ply_path).unlink(missing_ok=True)
        except Exception:
            pass

    pts_voxel   = sonata_result["pts_voxel"]
    sonata_pred = sonata_result["pred"]
    sonata_conf = sonata_result["confidence"]
    instances   = sonata_result["instances"]
    timings["sonata_s"] = round(time.time() - t, 2)

    # ---------- 4. Reclassify + filter ----------
    scene_bbox_z = (float(pts_voxel[:, 2].min()), float(pts_voxel[:, 2].max()))
    instances_rec = class_mapping.reclassify_ceiling(instances, scene_bbox_z)
    instances_match = class_mapping.filter_matchable_instances(instances_rec)
    if verbose:
        print(f"[v2]     Reclassify+filter: {len(instances)} → {len(instances_match)} matchable")

    # ---------- 5. Hungarian matching ----------
    t = time.time()
    match_result = matcher_hungarian.match(instances_match, ifc_objects)
    timings["hungarian_s"] = round(time.time() - t, 2)
    matches            = match_result["matches"]
    unmatched_scan     = match_result["unmatched_scan"]   # → ADICAO
    unmatched_ifc      = match_result["unmatched_ifc"]    # → AUSENTE
    ifc_sem_cobertura  = match_result["ifc_sem_cobertura"]  # → pipeline V1

    if verbose:
        s = match_result["stats"]
        print(f"[v2] (4/6) Matches: {s['n_matches']}, "
              f"ADICAO: {s['n_unmatched_scan']}, "
              f"AUSENTE: {s['n_unmatched_ifc']}, "
              f"V1 (sem cobertura): {s['n_ifc_sem_cobertura']}")

    # Indice rápido scan_inst pra recuperar via id
    match_by_ifc_id: dict[int, dict] = {id(m["ifc_obj"]): m for m in matches}

    # ---------- 6. RF por objeto IFC ----------
    # Usa pts_aligned (mesmo espaço que IFC, mesmo que Sonata processou).
    t = time.time()
    router = rf_router.get_router(verbose=verbose)
    objetos_out: list[dict] = []
    n_v1_used = n_v2_used = 0
    n_completo = n_parcial = n_ausente = 0

    def _json_filename_for(obj: dict) -> str:
        nome = _secure_filename(obj.get("nome", "obj"))[:30] or "obj"
        gid  = (obj.get("guid", "") or "")[:8]
        return f"{nome}_{gid}.json"

    # 5a. IFC matched → features v1 + v2 → RF (prefer v2)
    for m in matches:
        ifc_obj   = m["ifc_obj"]
        scan_inst = m["scan_inst"]

        feats_v1, pts_obj, _cls = app_obb._extrair_features_ml(pts_aligned, ifc_obj)
        try:
            feats_v2_new = compute_sonata_features(
                scan_inst, ifc_obj, pts_voxel, sonata_pred, sonata_conf,
            )
            feats_v2 = np.concatenate([feats_v1, feats_v2_new]).astype(np.float32)
        except Exception as e:
            if verbose:
                print(f"[v2] feats v2 falhou em {ifc_obj.get('guid')}: {e}; usando v1")
            feats_v2 = None

        try:
            status_int, version = router.predict(feats_v1, feats_v2, prefer="auto")
            code = INT_TO_CODE[status_int]
        except Exception as e:
            if verbose:
                print(f"[v2] RF falhou em {ifc_obj.get('guid')}: {e}; marcando AUSENTE")
            code, version = "AUSENTE", None

        if version == "v2": n_v2_used += 1
        elif version == "v1": n_v1_used += 1

        match_info = {
            "cost":           round(float(m["cost"]), 3),
            "scan_class":     scan_inst.get("class_name"),
            "scan_centroid":  [float(x) for x in scan_inst.get("centroid", [0,0,0])],
            "scan_n_pts":     int(scan_inst.get("n_pts", 0)),
            "scan_conf":      round(float(scan_inst.get("mean_conf", 0)), 3),
            "matched_by":     "hungarian",
        }
        feats_to_save = (feats_v2 if feats_v2 is not None else feats_v1) if include_features_in_output else None
        json_file = None
        if output_dir is not None:
            json_file = _write_obj_cloud(pts_obj, code, output_dir,
                                          _json_filename_for(ifc_obj), max_pts_viewer)
        objetos_out.append(_objeto_resultado(
            ifc_obj, code, version=version, match_info=match_info,
            features=feats_to_save, json_file=json_file, n_pts=len(pts_obj),
        ))
        if code == "COMPLETO": n_completo += 1
        elif code == "PARCIAL": n_parcial += 1
        else: n_ausente += 1

    # 5b. IFC sem match (Hungarian rejeitou) → AUSENTE direto
    for ifc_obj in unmatched_ifc:
        # Mesmo AUSENTE, salva nuvem (vai colorida de vermelho — pode ter pts dentro)
        _f, pts_obj, _c = app_obb._extrair_features_ml(pts_aligned, ifc_obj)
        json_file = None
        if output_dir is not None:
            json_file = _write_obj_cloud(pts_obj, "AUSENTE", output_dir,
                                          _json_filename_for(ifc_obj), max_pts_viewer)
        objetos_out.append(_objeto_resultado(
            ifc_obj, "AUSENTE", version=None, json_file=json_file, n_pts=len(pts_obj),
        ))
        n_ausente += 1

    # 5c. IFC sem cobertura Sonata (colunas, vigas, etc.) → features v1 + RF v1
    for ifc_obj in ifc_sem_cobertura:
        feats_v1, pts_obj, _cls = app_obb._extrair_features_ml(pts_aligned, ifc_obj)
        try:
            status_int, version = router.predict(feats_v1, prefer="v1")
            code = INT_TO_CODE[status_int]
        except Exception as e:
            if verbose:
                print(f"[v2] RF v1 falhou em {ifc_obj.get('guid')}: {e}; AUSENTE")
            code, version = "AUSENTE", None
        if version == "v1": n_v1_used += 1
        feats_to_save = feats_v1 if include_features_in_output else None
        json_file = None
        if output_dir is not None:
            json_file = _write_obj_cloud(pts_obj, code, output_dir,
                                          _json_filename_for(ifc_obj), max_pts_viewer)
        objetos_out.append(_objeto_resultado(
            ifc_obj, code, version=version, features=feats_to_save,
            json_file=json_file, n_pts=len(pts_obj),
        ))
        if code == "COMPLETO": n_completo += 1
        elif code == "PARCIAL": n_parcial += 1
        else: n_ausente += 1

    timings["rf_inference_s"] = round(time.time() - t, 2)

    # ---------- 5d. ADICAO (instancias scan sem par IFC) ----------
    adicoes_out = []
    for k, u in enumerate(unmatched_scan):
        c = u.get("centroid", [0,0,0])
        bbox = u.get("bbox") or {}
        # Pts do scan dentro da bbox da instancia → JSON azul
        json_file = None
        n_pts_view = 0
        if output_dir is not None and bbox:
            m_margem = 0.2
            mask = (
                (pts_aligned[:, 0] >= bbox["xmin"] - m_margem) &
                (pts_aligned[:, 0] <= bbox["xmax"] + m_margem) &
                (pts_aligned[:, 1] >= bbox["ymin"] - m_margem) &
                (pts_aligned[:, 1] <= bbox["ymax"] + m_margem) &
                (pts_aligned[:, 2] >= bbox["zmin"] - m_margem) &
                (pts_aligned[:, 2] <= bbox["zmax"] + m_margem)
            )
            pts_ad = pts_aligned[mask]
            n_pts_view = int(len(pts_ad))
            fname = f"adicao_{k:03d}_{u.get('class_name','')}.json"
            json_file = _write_obj_cloud(pts_ad, "ADICAO", output_dir, fname, max_pts_viewer)
        adicoes_out.append({
            "scan_class":    u.get("class_name"),
            "centroid":      [float(x) for x in c],
            "n_pts":         int(u.get("n_pts", 0)),
            "mean_conf":     round(float(u.get("mean_conf", 0)), 3),
            "volume":        round(float(u.get("volume", 0)), 3),
            "bbox":          bbox or None,
            "bbox_normalized": app_obb.converter_ifc_para_threejs(bbox) if bbox else None,
            "status":        _status_dict("ADICAO"),
            "json_file":     json_file,
            "n_pts_view":    n_pts_view,
        })

    # ---------- Stats e meta ----------
    timings["total_s"] = round(time.time() - t_global, 2)
    stats = {
        "n_ifc_total":          len(ifc_objects),
        "n_ifc_matched":        len(matches),
        "n_ifc_unmatched":      len(unmatched_ifc),
        "n_ifc_sem_cobertura":  len(ifc_sem_cobertura),
        "n_adicoes":            len(unmatched_scan),
        "n_completo":           n_completo,
        "n_parcial":            n_parcial,
        "n_ausente":            n_ausente,
        "n_rf_v1_used":         n_v1_used,
        "n_rf_v2_used":         n_v2_used,
        "scan_instances_total":     len(instances),
        "scan_instances_matchable": len(instances_match),
    }
    meta = {
        "pipeline_version":  "v2",
        "ply":               Path(ply_path).name,
        "ifc":               Path(ifc_path).name,
        "pavimento":         pavimento,
        "sonata_voxel":      sonata_voxel,
        "timings":           timings,
        "rf_router_info":    router.info,
        "from_sonata_cache": bool(sonata_result.get("from_cache", False)),
        "global_cloud":      global_cloud_path,
    }

    if verbose:
        print(f"[v2] (6/6) Pronto em {timings['total_s']}s — "
              f"COMPLETO={n_completo}, PARCIAL={n_parcial}, "
              f"AUSENTE={n_ausente}, ADICAO={len(adicoes_out)}")

    return {
        "objetos": objetos_out,
        "adicoes": adicoes_out,
        "stats":   stats,
        "meta":    meta,
    }


# ============================================================
# CLI / smoke test
# ============================================================
if __name__ == "__main__":
    import json as _json
    if len(sys.argv) < 3:
        print("Uso: python -m pipeline_v2.orchestrator <ply> <ifc> [pavimento]")
        sys.exit(1)
    ply  = sys.argv[1]
    ifc  = sys.argv[2]
    pav  = sys.argv[3] if len(sys.argv) > 3 else "__TODOS__"

    result = run(ply, ifc, pav, verbose=True)
    print()
    print("=" * 70)
    print("RESUMO")
    print("=" * 70)
    print(_json.dumps(result["stats"], indent=2, ensure_ascii=False))
    print()
    print("Meta:")
    print(_json.dumps(result["meta"], indent=2, ensure_ascii=False))
    print()
    print(f"Top 5 objetos (de {len(result['objetos'])}):")
    for o in result["objetos"][:5]:
        print(f"  {o['tipo']:13s} {o['status']['code']:9s} "
              f"rf={o['rf_version']}  {(o.get('nome') or o['guid'])[:50]}")
    if result["adicoes"]:
        print(f"\nTop 3 adicoes (de {len(result['adicoes'])}):")
        for a in result["adicoes"][:3]:
            c = a["centroid"]
            print(f"  {a['scan_class']:9s} "
                  f"centro=({c[0]:7.2f},{c[1]:7.2f},{c[2]:6.2f}) "
                  f"n_pts={a['n_pts']}")
