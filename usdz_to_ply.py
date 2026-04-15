#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converte USDZ -> OBJ -> PLY (nuvem de pontos densa via sampling da malha).

Passo 1: USDZ -> OBJ (extrai toda a geometria como mesh)
Passo 2: OBJ  -> PLY (sampling uniforme da superficie = nuvem de pontos real)

Uso:
  python usdz_to_ply.py <entrada.usdz> [densidade_pts_por_m2]
"""

import sys
from pathlib import Path
import numpy as np

from pxr import Usd, UsdGeom, Gf
import open3d as o3d


# Filtro de nomes: RoomPlan marca mesh de estrutura com nomes especificos.
# Mobilia geralmente vem como "Mesh_###" sem prefixo de estrutura.
ESTRUTURA_KEYWORDS = (
    "wall", "floor", "ceiling", "door", "window", "stair",
    "opening", "railing", "column", "beam",
)

DENSIDADE_PADRAO = 500  # pontos por m^2


def coletar_meshes(stage, apenas_estrutura=True):
    """Percorre o stage USD e coleta (verts, faces, nome) de cada Mesh."""
    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        nome = prim.GetName().lower()
        path = str(prim.GetPath()).lower()

        if apenas_estrutura:
            if not any(k in nome or k in path for k in ESTRUTURA_KEYWORDS):
                continue

        mesh = UsdGeom.Mesh(prim)

        pts_attr = mesh.GetPointsAttr().Get()
        if pts_attr is None or len(pts_attr) == 0:
            continue

        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if face_counts is None or face_indices is None:
            continue

        verts = np.array([[p[0], p[1], p[2]] for p in pts_attr], dtype=np.float64)

        # Transformacao de mundo (posiciona no espaco global)
        xform_cache = UsdGeom.XformCache()
        M = xform_cache.GetLocalToWorldTransform(prim)
        M_np = np.array([[M[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
        # USD usa row-major com vetor a esquerda: p' = p * M
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        verts = (verts_h @ M_np)[:, :3]

        # Triangulariza faces (pode ter quads/ngons)
        tris = []
        idx = 0
        for c in face_counts:
            face = list(face_indices[idx:idx + c])
            # fan triangulation
            for k in range(1, c - 1):
                tris.append([face[0], face[k], face[k + 1]])
            idx += c
        tris = np.array(tris, dtype=np.int32)
        if len(tris) == 0:
            continue

        meshes.append((verts, tris, prim.GetName()))
    return meshes


def salvar_obj(meshes, path_obj):
    """Salva todas as meshes concatenadas num unico OBJ."""
    with open(path_obj, 'w', encoding='utf-8') as f:
        f.write("# Exportado de USDZ\n")
        vert_offset = 1  # OBJ e 1-indexed
        for verts, tris, nome in meshes:
            f.write(f"o {nome}\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for t in tris:
                a, b, c = t[0] + vert_offset, t[1] + vert_offset, t[2] + vert_offset
                f.write(f"f {a} {b} {c}\n")
            vert_offset += len(verts)
    print(f"  [OBJ] {path_obj}")


def obj_para_ply(path_obj, path_ply, densidade):
    """Carrega OBJ no open3d, faz sampling uniforme, salva PLY."""
    mesh = o3d.io.read_triangle_mesh(str(path_obj))
    mesh.compute_triangle_normals()

    area_total = mesh.get_surface_area()
    n_pts = max(int(area_total * densidade), 10_000)
    n_pts = min(n_pts, 5_000_000)

    print(f"  Area total: {area_total:.2f} m^2")
    print(f"  Amostrando {n_pts:,} pontos (densidade={densidade}/m^2)")

    pcd = mesh.sample_points_uniformly(number_of_points=n_pts)
    o3d.io.write_point_cloud(str(path_ply), pcd)
    print(f"  [PLY] {path_ply}  ({len(pcd.points):,} pts)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    usdz_path = Path(sys.argv[1])
    if not usdz_path.exists():
        print(f"Arquivo nao encontrado: {usdz_path}")
        sys.exit(1)

    densidade = int(sys.argv[2]) if len(sys.argv) >= 3 else DENSIDADE_PADRAO

    obj_path = usdz_path.with_suffix('.obj')
    ply_path = usdz_path.with_suffix('.ply')

    print(f"Abrindo {usdz_path}")
    stage = Usd.Stage.Open(str(usdz_path))
    if stage is None:
        print("Falha abrindo USDZ")
        sys.exit(1)

    print("Coletando meshes de estrutura...")
    meshes = coletar_meshes(stage, apenas_estrutura=True)
    print(f"  {len(meshes)} meshes estruturais encontradas")

    if not meshes:
        print("  Nenhuma mesh estrutural. Tentando TODAS as meshes...")
        meshes = coletar_meshes(stage, apenas_estrutura=False)
        print(f"  {len(meshes)} meshes no total")

    if not meshes:
        print("Nada a exportar.")
        sys.exit(1)

    print(f"\nPasso 1: USDZ -> OBJ")
    salvar_obj(meshes, obj_path)

    print(f"\nPasso 2: OBJ -> PLY (sampling)")
    obj_para_ply(obj_path, ply_path, densidade)

    print(f"\nPronto!")
    print(f"  OBJ: {obj_path}")
    print(f"  PLY: {ply_path}")


if __name__ == '__main__':
    main()
