# -*- coding: utf-8 -*-
"""
PLANTA -> 3D  (v0: extrusao bruta pra validacao visual)
========================================================
Le um DXF de planta baixa e extruda o desenho pra 3D:
  - Linhas das layers de parede (Wall-*) -> quads verticais com pe-direito cheio
  - Blocos (portas/janelas/loucas) e demais layers -> extrusao baixa (1m),
    so pra aparecerem no chao como referencia
  - Cotas (DIMENSION) e textos sao ignorados

Saida: OBJ (abre no CloudCompare / Blender / visualizador 3D do Windows).

NAO detecta espessura de parede nem gera IFC — isso eh o v1.
Objetivo aqui: conferir visualmente que a leitura do DXF esta correta
("a planta de pe").

Uso:
  python dxf_to_3d_v0.py Drawing1.dxf
  python dxf_to_3d_v0.py Drawing1.dxf --altura 3.0 --out saida.obj
"""
import sys
import math
import argparse
from pathlib import Path

import ezdxf

# Layers tratadas como PAREDE (extrusao com pe-direito cheio).
# Casamento por prefixo, case-insensitive: "Wall-Ext", "wall_int", "PAREDE"...
WALL_PREFIXES = ("wall", "parede")

# Entidades ignoradas (anotacao, nao geometria)
SKIP_TYPES = {"DIMENSION", "TEXT", "MTEXT", "LEADER", "MULTILEADER", "HATCH"}

ARC_SEGS = 24  # segmentos pra discretizar arcos/circulos

# Cores por layer (RGB 0-255) — casamento por prefixo, case-insensitive
LAYER_COLORS = {
    "wall-ext":  (230, 126,  34),   # laranja forte
    "wall-int":  (243, 187, 119),   # laranja claro
    "wall":      (230, 126,  34),
    "parede":    (230, 126,  34),
    "door":      ( 46, 204, 113),   # verde
    "porta":     ( 46, 204, 113),
    "window":    ( 52, 152, 219),   # azul
    "janela":    ( 52, 152, 219),
    "sanitary":  (155,  89, 182),   # roxo
    "furniture": (149, 165, 166),   # cinza
    "0":         ( 90,  90,  90),   # cinza escuro
}
COR_DEFAULT = (160, 160, 160)


def cor_da_layer(layer: str):
    l = layer.lower()
    # match exato primeiro, depois por prefixo
    if l in LAYER_COLORS:
        return LAYER_COLORS[l]
    for pref, cor in LAYER_COLORS.items():
        if l.startswith(pref):
            return cor
    return COR_DEFAULT


def eh_parede(layer: str) -> bool:
    l = layer.lower()
    return any(l.startswith(p) for p in WALL_PREFIXES)


def detectar_escala(xs, ys) -> float:
    """Extensao > 1000 => desenho em milimetros => converte pra metros."""
    if not xs:
        return 1.0
    ext = max(max(xs) - min(xs), max(ys) - min(ys))
    if ext > 1000:
        return 0.001   # mm -> m
    if ext > 100:
        return 0.01    # cm -> m
    return 1.0         # ja em metros


def coletar_segmentos(msp):
    """Extrai segmentos 2D (x1,y1,x2,y2, layer) do modelspace.

    INSERTs sao expandidos (virtual_entities) pra pegar o desenho dos blocos.
    Arcos/circulos viram polilinhas discretizadas.
    """
    segs = []

    def add_entity(e, layer_override=None):
        t = e.dxftype()
        if t in SKIP_TYPES:
            return
        layer = layer_override or e.dxf.layer

        if t == "LINE":
            s, en = e.dxf.start, e.dxf.end
            segs.append((s.x, s.y, en.x, en.y, layer))

        elif t == "LWPOLYLINE":
            pts = list(e.get_points("xy"))
            fechado = e.closed
            for i in range(len(pts) - 1):
                segs.append((pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], layer))
            if fechado and len(pts) > 2:
                segs.append((pts[-1][0], pts[-1][1], pts[0][0], pts[0][1], layer))

        elif t == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            for i in range(len(pts) - 1):
                segs.append((pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], layer))

        elif t in ("CIRCLE", "ARC"):
            cx, cy, r = e.dxf.center.x, e.dxf.center.y, e.dxf.radius
            if t == "ARC":
                a0, a1 = math.radians(e.dxf.start_angle), math.radians(e.dxf.end_angle)
                if a1 <= a0:
                    a1 += 2 * math.pi
            else:
                a0, a1 = 0.0, 2 * math.pi
            n = max(4, int(ARC_SEGS * (a1 - a0) / (2 * math.pi)))
            ang = [a0 + (a1 - a0) * i / n for i in range(n + 1)]
            pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in ang]
            for i in range(len(pts) - 1):
                segs.append((pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], layer))

        elif t == "INSERT":
            # Expande o bloco: entidades viram geometria no espaco do modelo.
            # Layer do INSERT prevalece (blocos costumam desenhar na layer 0).
            try:
                for sub in e.virtual_entities():
                    add_entity(sub, layer_override=e.dxf.layer)
            except Exception:
                pass

    for e in msp:
        add_entity(e)

    return segs


def escrever_ply(segs, escala, altura_parede, altura_ref, out_path):
    """Cada segmento vira um quad vertical (2 triangulos), com cor por layer.

    PLY ASCII com vertex colors — abre colorido no CloudCompare.
    """
    por_layer = {}
    for x1, y1, x2, y2, layer in segs:
        por_layer.setdefault(layer, []).append((x1, y1, x2, y2))

    verts = []   # (x, y, z, r, g, b)
    faces = []   # (a, b, c) 0-indexed

    for layer in sorted(por_layer):
        h = altura_parede if eh_parede(layer) else altura_ref
        r, g, b = cor_da_layer(layer)
        for x1, y1, x2, y2 in por_layer[layer]:
            if abs(x2 - x1) < 1e-9 and abs(y2 - y1) < 1e-9:
                continue  # segmento degenerado
            ax, ay = x1 * escala, y1 * escala
            bx, by = x2 * escala, y2 * escala
            base = len(verts)
            verts.append((ax, ay, 0.0, r, g, b))
            verts.append((bx, by, 0.0, r, g, b))
            verts.append((bx, by, h,   r, g, b))
            verts.append((ax, ay, h,   r, g, b))
            faces.append((base, base + 1, base + 2))
            faces.append((base, base + 2, base + 3))

    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("comment planta extrudada (v0) - dxf_to_3d_v0.py\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in verts:
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
        for a, b_, c in faces:
            f.write(f"3 {a} {b_} {c}\n")

    return len(verts), len(faces)


def main():
    ap = argparse.ArgumentParser(description="Planta DXF -> 3D (extrusao v0)")
    ap.add_argument("dxf", help="caminho do .dxf")
    ap.add_argument("--altura", type=float, default=2.80,
                    help="pe-direito das paredes em metros (default 2.80)")
    ap.add_argument("--altura_ref", type=float, default=1.00,
                    help="altura das layers nao-parede (default 1.00)")
    ap.add_argument("--out", help="PLY de saida (default: <dxf>_3d.ply)")
    args = ap.parse_args()

    dxf_path = Path(args.dxf)
    out_path = Path(args.out) if args.out else dxf_path.with_name(dxf_path.stem + "_3d.ply")

    print(f"Lendo {dxf_path.name} ...")
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    segs = coletar_segmentos(msp)
    print(f"  {len(segs)} segmentos coletados")

    xs = [s[0] for s in segs] + [s[2] for s in segs]
    ys = [s[1] for s in segs] + [s[3] for s in segs]
    escala = detectar_escala(xs, ys)
    print(f"  Escala detectada: {escala} (extensao "
          f"{(max(xs)-min(xs))*escala:.1f}m x {(max(ys)-min(ys))*escala:.1f}m)")

    layers = sorted({s[4] for s in segs})
    paredes = [l for l in layers if eh_parede(l)]
    print(f"  Layers parede (pe-direito {args.altura}m): {paredes}")
    print(f"  Layers referencia ({args.altura_ref}m): {[l for l in layers if l not in paredes]}")

    nv, nf = escrever_ply(segs, escala, args.altura, args.altura_ref, out_path)
    print(f"\nSalvo: {out_path}")
    print(f"  {nv} vertices, {nf} triangulos (colorido por layer)")
    print("Abre no CloudCompare e confere se eh a planta de pe.")


if __name__ == "__main__":
    main()
