# -*- coding: utf-8 -*-
"""Diagnostico: onde esta o bloco da porta no DXF vs onde o algoritmo colocou."""
import numpy as np
import ezdxf

DXF = r"C:\Users\Rafael\Desktop\Beckend\plantatobim\Drawing1.dxf"
doc = ezdxf.readfile(DXF)
msp = doc.modelspace()
ESC = 0.001

for e in msp:
    if e.dxftype() != "INSERT":
        continue
    if not e.dxf.layer.lower().startswith("door"):
        continue

    print("=" * 60)
    print(f"BLOCO PORTA  layer={e.dxf.layer}  block={e.dxf.name}")
    ins = e.dxf.insert
    print(f"  insert point (DXF): ({ins.x*ESC:.2f}, {ins.y*ESC:.2f}) m")
    print(f"  rotation: {e.dxf.rotation}  scale: ({e.dxf.xscale},{e.dxf.yscale})")

    # --- coleta como o codigo atual faz (bbox de arcos) ---
    pts_atual = []
    pts_so_linhas = []
    print("  --- virtual_entities ---")
    for sub in e.virtual_entities():
        t = sub.dxftype()
        if t == "LINE":
            s, en = sub.dxf.start, sub.dxf.end
            pts_atual += [(s.x, s.y), (en.x, en.y)]
            pts_so_linhas += [(s.x, s.y), (en.x, en.y)]
            print(f"    LINE  ({s.x*ESC:.2f},{s.y*ESC:.2f}) -> ({en.x*ESC:.2f},{en.y*ESC:.2f})")
        elif t == "LWPOLYLINE":
            for p in sub.get_points("xy"):
                pts_atual.append((p[0], p[1])); pts_so_linhas.append((p[0], p[1]))
            print(f"    LWPOLYLINE {len(list(sub.get_points('xy')))} pts")
        elif t in ("ARC", "CIRCLE"):
            c, r = sub.dxf.center, sub.dxf.radius
            pts_atual += [(c.x - r, c.y - r), (c.x + r, c.y + r)]  # como o codigo faz
            print(f"    {t}  centro=({c.x*ESC:.2f},{c.y*ESC:.2f}) r={r*ESC:.2f}", end="")
            if t == "ARC":
                print(f"  ang [{sub.dxf.start_angle:.0f},{sub.dxf.end_angle:.0f}]")
            else:
                print()
        else:
            print(f"    {t} (ignorado)")

    pa = np.array(pts_atual) * ESC
    pl = np.array(pts_so_linhas) * ESC if pts_so_linhas else pa
    print(f"  CENTRO (metodo atual, com bbox de arco): "
          f"({pa.mean(0)[0]:.2f}, {pa.mean(0)[1]:.2f})")
    print(f"  CENTRO (so linhas, sem arco):            "
          f"({pl.mean(0)[0]:.2f}, {pl.mean(0)[1]:.2f})")
    print(f"  insert point seria:                      "
          f"({ins.x*ESC:.2f}, {ins.y*ESC:.2f})")
