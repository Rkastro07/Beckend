# -*- coding: utf-8 -*-
"""
RUN BATCH INSTANCES — Gerador de Dataset com Labels de Instância
=================================================================
Igual ao run_batch.py mas gera .npz com 3 arrays:
  pts       : (N, 6) float32 — xyz + normais
  labels    : (N,)   uint8   — classe semântica (0..7)
  instances : (N,)   int32   — ID de instância (0=background, 1..M)

Isso é necessário para treinar modelos de segmentação de instâncias
(PointGroup, Mask3D, SPFormer) que identificam cada objeto IFC
individualmente — resolvendo T-junctions como wall_238/wall_251.

Uso:
    python randlanet/run_batch_instances.py --dataset dataset/
    python randlanet/run_batch_instances.py --dataset dataset/ --todos-pavimentos
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app import (
        extrair_pavimentos,
        extrair_objetos_por_pavimento,
        alinhar_nuvem_com_ifc,
        corrigir_orientacao_por_pico_vertical,
        normalizar_coordenadas,
        detectar_paredes_conexao,
        marcar_conexoes_piso_teto,
    )
except ImportError as e:
    print(f"Erro ao importar app.py: {e}")
    sys.exit(1)

from randlanet.dataset_generator_instances import salvar_cena_instancias, DATA_DIR
from randlanet.run_batch import _scan_dataset, _escolher_pavimento

MIN_PONTOS_PLY        = 1_000
MIN_SCORE_ALINHAMENTO = 1e-3
INST_DIR = DATA_DIR.parent / "data_instances"
INST_DIR.mkdir(parents=True, exist_ok=True)


def _processar_pavimento_inst(stem: str, pav: str, ifc_path: Path,
                               pts: np.ndarray) -> dict:
    info = {
        "pavimento": pav,
        "npz_path": None,
        "n_pontos": 0,
        "n_instancias": 0,
        "background_ratio": 0.0,
        "avisos": [],
        "erro": None,
    }

    try:
        objetos = extrair_objetos_por_pavimento(str(ifc_path), pav)
        if not objetos:
            info["erro"] = f"SEM_OBJETOS: '{pav}'"
            return info
        print(f"   '{pav}': {len(objetos)} objetos IFC")

        objetos, _ = detectar_paredes_conexao(objetos)
        objetos = marcar_conexoes_piso_teto(objetos)

        pts_alinhado, transf = alinhar_nuvem_com_ifc(pts, objetos)

        score = transf.get("score")
        if score is not None and score < MIN_SCORE_ALINHAMENTO:
            info["avisos"].append(f"ALINHAMENTO_FRACO: score={score:.2e}")

        pts_alinhado, flipped, _ = corrigir_orientacao_por_pico_vertical(
            pts_alinhado, objetos
        )
        if flipped:
            info["avisos"].append("ORIENTACAO_CORRIGIDA")

        pts_alinhado, _, objetos_norm = normalizar_coordenadas(pts_alinhado, objetos)

        # Gera .npz com labels semânticos + instância
        nome_cena = f"{stem}__{pav.replace(' ', '_')[:20]}"
        npz_path = salvar_cena_instancias(pts_alinhado, objetos_norm, nome_cena)

        if npz_path:
            info["npz_path"] = str(npz_path)
            data = np.load(npz_path)
            info["n_pontos"]       = len(data["pts"])
            info["n_instancias"]   = int(data["instances"].max())
            info["background_ratio"] = round(
                float((data["labels"] == 0).sum()) / len(data["pts"]), 4
            )

    except Exception as e:
        info["erro"] = f"ERRO: {e}\n{traceback.format_exc()}"
        print(f"   ERRO: {e}")

    return info


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description="Gera dataset de instâncias para BIM")
    parser.add_argument("--dataset",          required=True,  help="Diretório com IFC+PLY")
    parser.add_argument("--pavimento",        default="auto", help="auto|todos|nome")
    parser.add_argument("--todos-pavimentos", action="store_true")
    args = parser.parse_args()

    dataset_dir  = Path(args.dataset)
    modo_pav     = "todos" if args.todos_pavimentos else args.pavimento

    print("=" * 60)
    print("BATCH INSTANCES — RandLA-Net Instance Dataset")
    print(f"Dataset : {dataset_dir}")
    print(f"Saida   : {INST_DIR}")
    print(f"Modo    : pavimento={modo_pav}")
    print("=" * 60)

    matched, ifc_only, ply_only = _scan_dataset(dataset_dir)
    print(f"\nPares encontrados: {len(matched)}")

    if not matched:
        print("Nenhum par encontrado. Verifique o diretório.")
        return

    debug_report = {
        "gerado_em": datetime.now().isoformat(),
        "total_pares": len(matched),
        "sem_ply": ifc_only,
        "sem_ifc": ply_only,
        "resultados": []
    }

    ok = 0
    erros = 0

    for i, (stem, (ifc_path, ply_path)) in enumerate(matched.items(), 1):
        print(f"\n[{i}/{len(matched)}] {stem}")
        print(f"  IFC: {ifc_path.name}")
        print(f"  PLY: {ply_path.name}")

        res = {"stem": stem, "status": "OK", "pavimentos": [], "erro": None}

        try:
            pcd = o3d.io.read_point_cloud(str(ply_path))
            pts = np.asarray(pcd.points, dtype=np.float64)
            if len(pts) < MIN_PONTOS_PLY:
                raise ValueError(f"PLY com apenas {len(pts)} pontos")
            pts = np.unique(pts, axis=0)

            pavimentos = extrair_pavimentos(str(ifc_path))
            if not pavimentos:
                raise ValueError("Nenhum pavimento no IFC")

            pavs = _escolher_pavimento(pavimentos, modo_pav)

            for pav in pavs:
                pav_info = _processar_pavimento_inst(stem, pav, ifc_path, pts.copy())
                res["pavimentos"].append(pav_info)

            gerados = [p for p in res["pavimentos"] if p.get("npz_path")]
            if gerados:
                ok += 1
                print(f"  OK — {len(gerados)} npz gerado(s)")
            else:
                res["status"] = "AVISO"
                print(f"  AVISO — nenhum npz gerado")

        except Exception as e:
            res["status"] = "ERRO"
            res["erro"]   = str(e)
            erros += 1
            print(f"  ERRO: {e}")

        debug_report["resultados"].append(res)

    # Relatório final
    print("\n" + "=" * 60)
    print(f"OK     : {ok}/{len(matched)}")
    print(f"Erros  : {erros}")
    print(f"Saida  : {INST_DIR}")
    print("=" * 60)

    report_path = INST_DIR / "debug_report_instances.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(debug_report, f, ensure_ascii=False, indent=2)
    print(f"Relatorio: {report_path}")


if __name__ == "__main__":
    main()
