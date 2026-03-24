# -*- coding: utf-8 -*-
"""
RUN BATCH — Gerador de Dataset em Lote
=======================================
Processa todos os pares IFC+PLY de um diretório e gera .npz rotulados
para treino do RandLA-Net.

Uso:
    python randlanet/run_batch.py --dataset dataset/
    python randlanet/run_batch.py --dataset dataset/ --pavimento auto
    python randlanet/run_batch.py --dataset dataset/ --todos-pavimentos

Debug:
    Gera randlanet/data/debug_report.json com detalhes de cada par.
"""

import sys
import os
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import open3d as o3d

# Adiciona o diretório pai ao path para importar app.py
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importa funções do app.py (sem iniciar o servidor Flask)
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
    print(f"❌ Erro ao importar app.py: {e}")
    print("   Certifique-se de rodar a partir de C:/Users/Rafael/Desktop/Beckend")
    sys.exit(1)

# Importa funções do dataset_generator
from randlanet.dataset_generator import (
    _rotular_pontos,
    _estimar_normais,
    DATA_DIR,
)

# ─────────────────────────────────────────────
# CONSTANTES DE DEBUG
# ─────────────────────────────────────────────
MIN_BACKGROUND_RATIO = 0.95   # Alerta se >95% background
MIN_SCORE_ALINHAMENTO = 1e-3  # Alinhamento fraco (abaixo disso = falhou)
MIN_PONTOS_PLY = 1_000        # PLY vazio se tiver menos que isso


# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

def _scan_dataset(dataset_dir: Path):
    """
    Escaneia o diretório e retorna pares matched + arquivos sem par.
    Suporta estrutura plana (raiz) ou com subdirs ifc/ e ply/.

    Retorna:
        matched   : dict {stem: (ifc_path, ply_path)}
        ifc_only  : list de stems com .ifc mas sem .ply
        ply_only  : list de stems com .ply mas sem .ifc
    """
    # Detecta automaticamente subdirs ifc/ e ply/
    ifc_dir = dataset_dir / "ifc" if (dataset_dir / "ifc").is_dir() else dataset_dir
    ply_dir = dataset_dir / "ply" if (dataset_dir / "ply").is_dir() else dataset_dir

    print(f"   📂 IFC dir : {ifc_dir}")
    print(f"   📂 PLY dir : {ply_dir}")

    ifc_files = {p.stem.lower(): p for p in ifc_dir.glob("*.ifc")}
    ply_files = {p.stem.lower(): p for p in ply_dir.glob("*.ply")}

    print(f"   📄 IFC encontrados : {len(ifc_files)}")
    print(f"   📄 PLY encontrados : {len(ply_files)}")

    # Match case-insensitive pelo stem
    matched_keys = set(ifc_files.keys()) & set(ply_files.keys())
    ifc_only = sorted(set(ifc_files.keys()) - matched_keys)
    ply_only = sorted(set(ply_files.keys()) - matched_keys)

    if ifc_only:
        print(f"   ⚠️  Sem PLY ({len(ifc_only)}): {', '.join(ifc_only)}")
    if ply_only:
        print(f"   ⚠️  Sem IFC ({len(ply_only)}): {', '.join(ply_only)}")

    matched = {
        k: (ifc_files[k], ply_files[k])
        for k in sorted(matched_keys)
    }

    return matched, ifc_only, ply_only


def _escolher_pavimento(pavimentos: list, modo: str) -> list:
    """
    Retorna lista de pavimentos a processar.
    modo:
        'auto'  → só o 1º pavimento
        'todos' → todos os pavimentos
        'nome'  → pavimento específico (usa como filtro)
    """
    if not pavimentos:
        return []
    if modo == "auto":
        return [pavimentos[0]]
    if modo == "todos":
        return pavimentos
    # Modo nome: filtra por substring
    filtrados = [p for p in pavimentos if modo.lower() in p.lower()]
    return filtrados if filtrados else [pavimentos[0]]


def _processar_par(stem: str, ifc_path: Path, ply_path: Path,
                   pavimentos_modo: str, output_dir: Path) -> dict:
    """
    Processa um par IFC+PLY e gera os .npz.
    Retorna dict com resultado e debug info.
    """
    resultado = {
        "stem": stem,
        "ifc": str(ifc_path),
        "ply": str(ply_path),
        "status": "OK",
        "npz_gerados": [],
        "avisos": [],
        "erro": None,
        "pavimentos_processados": [],
    }

    # ── 1. Carrega PLY ──────────────────────────────────────
    try:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points, dtype=np.float64)
        if len(pts) < MIN_PONTOS_PLY:
            resultado["status"] = "ERRO"
            resultado["erro"] = f"ERRO_PLY: apenas {len(pts)} pontos (mínimo {MIN_PONTOS_PLY})"
            return resultado
        # Deduplicação
        pts = np.unique(pts, axis=0)
        print(f"   📦 PLY: {len(pts):,} pontos (após dedup)")
    except Exception as e:
        resultado["status"] = "ERRO"
        resultado["erro"] = f"ERRO_PLY: {e}"
        return resultado

    # ── 2. Extrai pavimentos do IFC ─────────────────────────
    try:
        pavimentos = extrair_pavimentos(str(ifc_path))
        if not pavimentos:
            resultado["status"] = "ERRO"
            resultado["erro"] = "ERRO_IFC: nenhum pavimento encontrado"
            return resultado
        print(f"   🏗️  Pavimentos: {pavimentos}")
    except Exception as e:
        resultado["status"] = "ERRO"
        resultado["erro"] = f"ERRO_IFC: {e}"
        return resultado

    # ── 3. Seleciona pavimentos ─────────────────────────────
    pavs_a_processar = _escolher_pavimento(pavimentos, pavimentos_modo)
    if not pavs_a_processar:
        resultado["status"] = "AVISO"
        resultado["avisos"].append(f"Nenhum pavimento correspondeu ao filtro '{pavimentos_modo}'")
        pavs_a_processar = [pavimentos[0]]

    # ── 4. Processa cada pavimento ──────────────────────────
    for pav in pavs_a_processar:
        pav_resultado = _processar_pavimento(
            stem, pav, ifc_path, pts.copy(), output_dir
        )
        resultado["pavimentos_processados"].append(pav_resultado)

        if pav_resultado.get("npz_path"):
            resultado["npz_gerados"].append(pav_resultado["npz_path"])
        if pav_resultado.get("avisos"):
            resultado["avisos"].extend(pav_resultado["avisos"])

    # Status final
    if not resultado["npz_gerados"]:
        resultado["status"] = "ERRO" if not resultado["avisos"] else "AVISO"

    return resultado


def _processar_pavimento(stem: str, pav: str, ifc_path: Path,
                         pts: np.ndarray, output_dir: Path) -> dict:
    """
    Processa um pavimento específico e gera o .npz.
    """
    info = {
        "pavimento": pav,
        "npz_path": None,
        "n_pontos": 0,
        "label_counts": {},
        "background_ratio": 0.0,
        "avisos": [],
        "erro": None,
    }

    try:
        # ── Extrai objetos ──────────────────────────────────
        objetos = extrair_objetos_por_pavimento(str(ifc_path), pav)
        if not objetos:
            info["erro"] = f"SEM_OBJETOS: '{pav}' sem objetos IFC"
            return info
        print(f"   📐 '{pav}': {len(objetos)} objetos IFC")

        # ── Detecta conexões ────────────────────────────────
        objetos, _ = detectar_paredes_conexao(objetos)
        objetos = marcar_conexoes_piso_teto(objetos)

        # ── Alinhamento ─────────────────────────────────────
        pts_alinhado, transf = alinhar_nuvem_com_ifc(pts, objetos)

        # Valida alinhamento
        score = transf.get("score", None)
        if score is not None and score < MIN_SCORE_ALINHAMENTO:
            info["avisos"].append(
                f"ALINHAMENTO_FRACO: score={score:.2e} (mínimo {MIN_SCORE_ALINHAMENTO})"
            )
            print(f"   ⚠️  Alinhamento fraco (score={score:.2e})")
        else:
            print(f"   ✅ Alinhado (escala={transf.get('scale', 1.0)})")

        # ── Orientação ──────────────────────────────────────
        pts_alinhado, flipped, _ = corrigir_orientacao_por_pico_vertical(
            pts_alinhado, objetos
        )
        if flipped:
            info["avisos"].append("ORIENTACAO_CORRIGIDA: nuvem estava invertida em Z")
            print("   🔄 Orientação corrigida (flip Z aplicado)")

        # ── Normalização ────────────────────────────────────
        pts_alinhado, _, objetos_norm = normalizar_coordenadas(pts_alinhado, objetos)

        # ── Rótulos ─────────────────────────────────────────
        labels = _rotular_pontos(pts_alinhado, objetos_norm)

        n_total = len(labels)
        n_back = int((labels == 0).sum())
        n_label = int((labels > 0).sum())
        bg_ratio = n_back / n_total if n_total > 0 else 1.0

        info["n_pontos"] = n_total
        info["background_ratio"] = round(bg_ratio, 4)

        # Contagem por classe
        nomes = {0: "background", 1: "IfcWall", 2: "IfcSlab", 3: "IfcColumn",
                 4: "IfcBeam", 5: "IfcStair", 6: "IfcRoof", 7: "IfcSanitaryTerminal"}
        for i in range(8):
            cnt = int((labels == i).sum())
            if cnt > 0:
                info["label_counts"][nomes[i]] = cnt

        print(f"   🏷️  Labels: {n_label:,} rotulados | {n_back:,} background ({bg_ratio:.1%})")

        # Alerta se quase tudo é background
        if bg_ratio > MIN_BACKGROUND_RATIO:
            info["avisos"].append(
                f"SEM_PONTOS_ROTULADOS: {bg_ratio:.1%} background — alinhamento pode ter falhado"
            )
            print(f"   ⚠️  AVISO: {bg_ratio:.1%} background — verifique o alinhamento")

        # ── Normais ─────────────────────────────────────────
        print("   🔢 Calculando normais...")
        normais = _estimar_normais(pts_alinhado)

        # ── Salva .npz ──────────────────────────────────────
        pav_safe = pav.replace(" ", "_").replace("/", "-")[:30]
        npz_name = f"{stem}__{pav_safe}.npz"
        npz_path = output_dir / npz_name

        features = np.hstack([
            pts_alinhado.astype(np.float32),
            normais.astype(np.float32)
        ])

        np.savez_compressed(
            str(npz_path),
            pts=features,
            labels=labels,
        )

        info["npz_path"] = str(npz_path)
        print(f"   💾 Salvo: {npz_path.name} ({n_total:,} pts)")

    except Exception as e:
        info["erro"] = f"EXCECAO: {traceback.format_exc()}"
        print(f"   ❌ Exceção: {e}")

    return info


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gera dataset .npz a partir de pares IFC+PLY"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="dataset",
        help="Diretório com os pares IFC+PLY (default: dataset/)"
    )
    parser.add_argument(
        "--pavimento", "-p",
        type=str,
        default="auto",
        help="Modo de seleção de pavimento: 'auto' (1º), 'todos', ou nome parcial"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Diretório de saída para .npz (default: randlanet/data/)"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output) if args.output else DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        print(f"❌ Diretório não encontrado: {dataset_dir}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🚀 BIM BATCH GENERATOR — RandLA-Net Dataset")
    print("=" * 70)
    print(f"📁 Dataset: {dataset_dir.resolve()}")
    print(f"💾 Output:  {output_dir.resolve()}")
    print(f"🏗️  Pavimento: {args.pavimento}")
    print("=" * 70 + "\n")

    # ── 1. Scan ──────────────────────────────────────────────────
    matched, ifc_only, ply_only = _scan_dataset(dataset_dir)

    print(f"📊 Arquivos encontrados:")
    print(f"   ✅ Pares matched:    {len(matched)}")
    print(f"   ⚠️  .ifc sem .ply:   {len(ifc_only)}")
    print(f"   ⚠️  .ply sem .ifc:   {len(ply_only)}")

    if ifc_only:
        print(f"\n⚠️  IFC sem PLY correspondente:")
        for s in ifc_only:
            print(f"   - {s}.ifc")

    if ply_only:
        print(f"\n⚠️  PLY sem IFC correspondente:")
        for s in ply_only:
            print(f"   - {s}.ply")

    if not matched:
        print("\n❌ Nenhum par encontrado. Verifique se os arquivos têm o mesmo nome.")
        sys.exit(1)

    # ── 2. Processa cada par ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"Processando {len(matched)} pares...")
    print("=" * 70)

    resultados = []
    n_ok = 0
    n_erro = 0
    n_aviso = 0

    for i, (stem, (ifc_path, ply_path)) in enumerate(matched.items(), 1):
        print(f"\n[{i}/{len(matched)}] {stem}")
        print(f"   IFC: {ifc_path.name}")
        print(f"   PLY: {ply_path.name}")

        resultado = _processar_par(
            stem, ifc_path, ply_path, args.pavimento, output_dir
        )
        resultados.append(resultado)

        if resultado["status"] == "OK":
            n_ok += 1
            print(f"   ✅ OK — {len(resultado['npz_gerados'])} .npz gerado(s)")
        elif resultado["status"] == "AVISO":
            n_aviso += 1
            print(f"   ⚠️  AVISO — {resultado['avisos']}")
        else:
            n_erro += 1
            print(f"   ❌ ERRO — {resultado['erro']}")

    # ── 3. Relatório final ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("📊 RESUMO FINAL")
    print("=" * 70)
    print(f"   ✅ OK:      {n_ok}/{len(matched)}")
    print(f"   ⚠️  Avisos:  {n_aviso}/{len(matched)}")
    print(f"   ❌ Erros:   {n_erro}/{len(matched)}")

    # Lista .npz gerados
    todos_npz = [npz for r in resultados for npz in r["npz_gerados"]]
    print(f"\n💾 Total de .npz gerados: {len(todos_npz)}")

    # Resumo de erros
    erros = [(r["stem"], r["erro"]) for r in resultados if r["erro"]]
    if erros:
        print(f"\n❌ Pares com erro:")
        for stem, err in erros:
            print(f"   {stem}: {err[:120]}")

    # Resumo de avisos
    avisos = [(r["stem"], r["avisos"]) for r in resultados if r["avisos"]]
    if avisos:
        print(f"\n⚠️  Pares com avisos:")
        for stem, avs in avisos:
            for av in avs:
                print(f"   {stem}: {av}")

    # ── 4. Salva debug_report.json ───────────────────────────────
    debug_report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_dir": str(dataset_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "pavimento_modo": args.pavimento,
        "resumo": {
            "total_pares": len(matched),
            "ok": n_ok,
            "avisos": n_aviso,
            "erros": n_erro,
            "npz_gerados": len(todos_npz),
        },
        "sem_par": {
            "ifc_sem_ply": ifc_only,
            "ply_sem_ifc": ply_only,
        },
        "detalhes": resultados,
    }

    report_path = output_dir / "debug_report.json"
    with open(str(report_path), "w", encoding="utf-8") as f:
        json.dump(debug_report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n📋 Relatório de debug salvo em: {report_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
