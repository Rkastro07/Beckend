# -*- coding: utf-8 -*-
"""
BATCH GENERATOR — Gera dataset de treino sem precisar do frontend.
===================================================================
Uso:
  python randlanet/batch_generator.py --dataset C:/caminho/para/dataset

Estrutura esperada do dataset (qualquer uma das duas):

  Opção A — pares por nome:
    dataset/
      obra1.ifc
      obra1.ply
      obra2.ifc
      obra2.ply

  Opção B — subpastas:
    dataset/
      obra1/
        modelo.ifc
        nuvem.ply
      obra2/
        modelo.ifc
        nuvem.ply

O script processa todos os pares, roda o alinhamento + análise completa
e salva um .npz rotulado em randlanet/data/ para cada par.
"""

import sys
import argparse
import traceback
from pathlib import Path

# Adiciona o backend ao path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np

# Importa funções do backend diretamente (sem Flask)
from app import (
    extrair_objetos_por_pavimento,
    extrair_pavimentos,
    detectar_paredes_conexao,
    marcar_conexoes_piso_teto,
    alinhar_nuvem_com_ifc,
    corrigir_orientacao_por_pico_vertical,
    normalizar_coordenadas,
)
import open3d as o3d
from dataset_generator import salvar_cena, estatisticas_dataset


# =========================
# DESCOBERTA DE PARES
# =========================
def descobrir_pares(dataset_dir: Path):
    """Encontra todos os pares (IFC, PLY) no diretório."""
    pares = []

    # Opção A: pares por nome na raiz
    ifcs = {f.stem: f for f in dataset_dir.glob("*.ifc")}
    plys = {f.stem: f for f in dataset_dir.glob("*.ply")}
    for nome, ifc in ifcs.items():
        if nome in plys:
            pares.append((ifc, plys[nome], nome))

    # Opção B: subpastas
    for subdir in sorted(dataset_dir.iterdir()):
        if not subdir.is_dir():
            continue
        ifc_files = list(subdir.glob("*.ifc"))
        ply_files = list(subdir.glob("*.ply"))
        if ifc_files and ply_files:
            pares.append((ifc_files[0], ply_files[0], subdir.name))

    return pares


# =========================
# PROCESSAMENTO DE UM PAR
# =========================
def processar_par(ifc_path: Path, ply_path: Path, nome: str) -> bool:
    """
    Roda o pipeline completo em um par IFC+PLY e salva o .npz de treino.
    Retorna True se sucesso.
    """
    print(f"\n{'='*60}")
    print(f"📁 Par: {nome}")
    print(f"   IFC: {ifc_path.name}")
    print(f"   PLY: {ply_path.name}")

    # 1. Descobre pavimentos
    try:
        pavimentos = extrair_pavimentos(str(ifc_path))
    except Exception as e:
        print(f"   ❌ Erro ao extrair pavimentos: {e}")
        return False

    if not pavimentos:
        print("   ⚠️ Nenhum pavimento encontrado.")
        return False

    print(f"   🏗️ Pavimentos: {pavimentos}")

    # 2. Carrega nuvem UMA vez (pesado)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts_bruto = np.asarray(pcd.points, dtype=float)
    pts_bruto = np.unique(pts_bruto, axis=0)
    print(f"   ✓ {len(pts_bruto):,} pontos únicos")

    # 3. Processa cada pavimento
    sucesso = False
    for pav in pavimentos:
        print(f"\n   🔍 Pavimento: {pav}")
        try:
            objetos = extrair_objetos_por_pavimento(str(ifc_path), pav)
            if not objetos:
                print(f"      ⚠️ Sem objetos no pavimento {pav}")
                continue

            print(f"      📦 {len(objetos)} objetos IFC")

            # Pipeline de alinhamento
            objetos, _ = detectar_paredes_conexao(objetos)
            objetos     = marcar_conexoes_piso_teto(objetos)
            pts, _      = alinhar_nuvem_com_ifc(pts_bruto.copy(), objetos)
            pts, _, _   = corrigir_orientacao_por_pico_vertical(pts, objetos)
            pts, _, objetos = normalizar_coordenadas(pts, objetos)

            # Salva cena rotulada
            session_id = f"{nome}_{pav}".replace(" ", "_").replace("/", "-")[:40]
            out = salvar_cena(pts, objetos, session_id=session_id)
            if out:
                sucesso = True

        except Exception as e:
            print(f"      ❌ Erro no pavimento {pav}: {e}")
            traceback.print_exc()

    return sucesso


# =========================
# ENTRY POINT
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Gera dataset RandLA-Net a partir de pares IFC+PLY"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Pasta com os pares IFC+PLY"
    )
    parser.add_argument(
        "--limite", type=int, default=None,
        help="Máximo de pares a processar (útil para testes)"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"❌ Diretório não encontrado: {dataset_dir}")
        sys.exit(1)

    pares = descobrir_pares(dataset_dir)
    if not pares:
        print(f"❌ Nenhum par IFC+PLY encontrado em: {dataset_dir}")
        print("   Verifique se os arquivos têm o mesmo nome (obra.ifc + obra.ply)")
        sys.exit(1)

    if args.limite:
        pares = pares[:args.limite]

    print(f"🔎 {len(pares)} pares encontrados")

    ok = 0
    erros = 0
    for ifc, ply, nome in pares:
        if processar_par(ifc, ply, nome):
            ok += 1
        else:
            erros += 1

    print(f"\n{'='*60}")
    print(f"✅ Processados: {ok}  |  ❌ Erros: {erros}")
    print()
    estatisticas_dataset()


if __name__ == "__main__":
    main()
