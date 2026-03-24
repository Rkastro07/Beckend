# -*- coding: utf-8 -*-
"""
VISUALIZADOR DE DATASET — Verifica qualidade dos .npz gerados
================================================================
Abre cada cena no Open3D com cores por classe.

Uso:
    python randlanet/visualizar_dataset.py                    # todos
    python randlanet/visualizar_dataset.py --filtro model_18  # um específico
    python randlanet/visualizar_dataset.py --invertidos       # só os que tiveram flip

Controles na janela 3D:
    Q / ESC  → fecha e vai pro próximo
    Scroll   → zoom
    Mouse    → rotação
"""

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

DATA_DIR = Path(__file__).parent / "data"

# Cores por classe (RGB 0-1)
CORES_CLASSE = {
    0: [0.5, 0.5, 0.5],    # background — cinza
    1: [0.2, 0.6, 1.0],    # IfcWall — azul
    2: [0.8, 0.8, 0.2],    # IfcSlab — amarelo
    3: [1.0, 0.3, 0.3],    # IfcColumn — vermelho
    4: [0.3, 1.0, 0.3],    # IfcBeam — verde
    5: [1.0, 0.5, 0.0],    # IfcStair — laranja
    6: [0.6, 0.2, 0.8],    # IfcRoof — roxo
    7: [0.0, 0.8, 0.8],    # IfcSanitaryTerminal — ciano
}

NOMES_CLASSE = {
    0: "background",
    1: "IfcWall",
    2: "IfcSlab",
    3: "IfcColumn",
    4: "IfcBeam",
    5: "IfcStair",
    6: "IfcRoof",
    7: "IfcSanitaryTerminal",
}


def carregar_e_colorir(npz_path: Path):
    """Carrega .npz e retorna PointCloud colorido por classe + stats."""
    data = np.load(str(npz_path))
    pts = data["pts"][:, :3]  # xyz (ignora normais)
    labels = data["labels"]

    # Estatísticas
    stats = {}
    for i in range(8):
        count = int((labels == i).sum())
        if count > 0:
            stats[NOMES_CLASSE[i]] = count

    total = len(labels)
    n_bg = int((labels == 0).sum())
    bg_pct = n_bg / total * 100 if total > 0 else 0

    # Colorir pontos
    cores = np.zeros((len(pts), 3))
    for label_id, cor in CORES_CLASSE.items():
        mask = labels == label_id
        cores[mask] = cor

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cores)

    return pcd, stats, total, bg_pct


def imprimir_stats(nome: str, stats: dict, total: int, bg_pct: float):
    """Imprime estatísticas no terminal."""
    print(f"\n{'='*60}")
    print(f"📊 {nome}")
    print(f"{'='*60}")
    print(f"   Total de pontos: {total:,}")
    print(f"   Background: {bg_pct:.1f}%")
    print(f"   {'─'*40}")

    for classe, count in sorted(stats.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        barra = "█" * int(pct / 2)
        print(f"   {classe:<22} {count:>8,} ({pct:>5.1f}%) {barra}")

    print(f"   {'─'*40}")


def imprimir_legenda():
    """Imprime legenda de cores."""
    print("\n🎨 LEGENDA DE CORES:")
    for label_id, nome in NOMES_CLASSE.items():
        cor = CORES_CLASSE[label_id]
        cor_hex = f"#{int(cor[0]*255):02x}{int(cor[1]*255):02x}{int(cor[2]*255):02x}"
        print(f"   {label_id}: {nome:<22} {cor_hex}")


def main():
    parser = argparse.ArgumentParser(description="Visualiza dataset .npz gerado")
    parser.add_argument("--filtro", "-f", type=str, default=None,
                        help="Filtro por nome (ex: model_18, casa)")
    parser.add_argument("--invertidos", "-i", action="store_true",
                        help="Mostra apenas modelos que tiveram flip Z")
    parser.add_argument("--stats-only", "-s", action="store_true",
                        help="Só imprime estatísticas, sem abrir visualizador 3D")
    parser.add_argument("--bg-min", type=float, default=None,
                        help="Filtra modelos com background%% acima deste valor")
    args = parser.parse_args()

    # Carrega debug report para saber quais foram invertidos
    report_path = DATA_DIR / "debug_report.json"
    invertidos = set()
    if report_path.exists():
        with open(str(report_path), "r", encoding="utf-8") as f:
            report = json.load(f)
        for det in report.get("detalhes", []):
            for pav in det.get("pavimentos_processados", []):
                avisos = pav.get("avisos", [])
                for av in avisos:
                    if "ORIENTACAO_CORRIGIDA" in av:
                        invertidos.add(det["stem"])

    # Encontra .npz
    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print("❌ Nenhum .npz encontrado em", DATA_DIR)
        sys.exit(1)

    # Filtra
    if args.filtro:
        npz_files = [f for f in npz_files if args.filtro.lower() in f.stem.lower()]
    if args.invertidos:
        npz_files = [f for f in npz_files if any(inv in f.stem for inv in invertidos)]

    if not npz_files:
        print("❌ Nenhum arquivo corresponde ao filtro")
        sys.exit(1)

    imprimir_legenda()
    print(f"\n📁 {len(npz_files)} cenas para visualizar")
    print("   (Feche a janela 3D com Q/ESC para ir ao próximo)\n")

    # Primeira passada: coleta stats de todos
    resultados = []
    for npz_path in npz_files:
        pcd, stats, total, bg_pct = carregar_e_colorir(npz_path)
        stem = npz_path.stem
        foi_invertido = any(inv in stem for inv in invertidos)
        resultados.append((npz_path, pcd, stats, total, bg_pct, foi_invertido))

    # Filtra por background se pedido
    if args.bg_min is not None:
        resultados = [(p, pcd, s, t, bg, inv) for p, pcd, s, t, bg, inv in resultados
                      if bg >= args.bg_min]

    if not resultados:
        print("❌ Nenhum arquivo corresponde ao filtro de background")
        sys.exit(1)

    # Mostra
    for i, (npz_path, pcd, stats, total, bg_pct, foi_invertido) in enumerate(resultados):
        nome = npz_path.stem
        tag = " 🔄 INVERTIDO" if foi_invertido else ""
        imprimir_stats(f"[{i+1}/{len(resultados)}] {nome}{tag}", stats, total, bg_pct)

        if not args.stats_only:
            print(f"\n   👁️  Abrindo visualizador... (Q/ESC para próximo)")
            o3d.visualization.draw_geometries(
                [pcd],
                window_name=f"{nome} ({i+1}/{len(resultados)}){tag}",
                width=1200,
                height=800,
                point_show_normal=False,
            )

    # Resumo final
    print(f"\n{'='*60}")
    print("📊 RESUMO DO DATASET")
    print(f"{'='*60}")
    total_pts = sum(t for _, _, _, t, _, _ in resultados)
    bgs = [bg for _, _, _, _, bg, _ in resultados]
    n_inv = sum(1 for _, _, _, _, _, inv in resultados if inv)
    print(f"   Cenas:         {len(resultados)}")
    print(f"   Total pontos:  {total_pts:,}")
    print(f"   Background:    {np.mean(bgs):.1f}% média (min {np.min(bgs):.1f}%, max {np.max(bgs):.1f}%)")
    print(f"   Invertidos:    {n_inv}")

    # Distribuição global de classes
    print(f"\n   Distribuição global:")
    global_stats = {}
    for _, _, stats, _, _, _ in resultados:
        for classe, count in stats.items():
            global_stats[classe] = global_stats.get(classe, 0) + count
    for classe, count in sorted(global_stats.items(), key=lambda x: -x[1]):
        pct = count / total_pts * 100
        print(f"   {classe:<22} {count:>10,} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
