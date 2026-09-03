# -*- coding: utf-8 -*-
"""
MODELO EDITAVEL DE PLANTA -> IFC
================================
Pipeline:
  1. Importa IFC/IFCZIP semanticamente ou DXF/SVG como vetores
  2. PAREAMENTO: casa pares de linhas paralelas proximas (5-40cm)
     -> cada par vira uma PAREDE com eixo + espessura reais
  3. Laje: contorno convexo das paredes externas (ou todas)
  4. Gera IFC4 (IfcProject > IfcSite > IfcBuilding > IfcBuildingStorey
     > IfcWall / IfcSlab com IfcExtrudedAreaSolid)
  5. Preview: malha o IFC gerado (ifcopenshell.geom) e salva PLY colorido
     pra conferir no CloudCompare

Limitacoes documentadas:
  - O editor 2D representa paredes por eixos retos; curvas sao tesselladas
  - CAD/SVG inferem semantica pelos nomes de layers/grupos
  - IFC e editado um pavimento por vez

Uso:
  python planta_to_ifc_v1.py Drawing1.dxf
  python planta_to_ifc_v1.py Drawing1.dxf --altura 3.0 --pavimento "Terreo"
"""
import math
import argparse
from pathlib import Path

import numpy as np
import ezdxf
import ifcopenshell
import ifcopenshell.api as api
import ifcopenshell.geom

# ---------------- parametros do pareamento ----------------
WALL_PREFIXES   = ("wall", "parede")
ESP_MIN         = 0.05   # espessura minima de parede (m)
ESP_MAX         = 0.40   # espessura maxima
OVERLAP_MIN     = 0.25   # sobreposicao longitudinal minima entre as 2 faces (m)
ANG_TOL         = math.radians(2.0)   # tolerancia de paralelismo
SEG_MIN         = 0.05   # ignora segmentos menores que isso
LEFTOVER_WARN   = 0.50   # segmento de parede nao-pareado maior que isso gera aviso
MERGE_PERP_TOL  = 0.03   # tolerancia perpendicular pra considerar 2 segs na MESMA face
MERGE_GAP_MAX   = 3.00   # gap longitudinal maximo pra fundir (cobre vaos de porta/janela)
SINGLE_LINE_FRAC = 0.30  # se o pareamento cobre menos que isso das faces, a planta
                         # e desenhada com parede de LINHA UNICA (eixo, sem espessura)

CORES = {  # RGB 0-255 pro preview
    "IfcWall":   (230, 126, 34),
    "IfcColumn": (139, 92, 246),
    "IfcSlab":   (52, 152, 219),
    "IfcCovering": (241, 196, 15),
    "IfcDoor":   (46, 204, 113),
    "IfcWindow": (93, 173, 226),
}

# Layers de esquadrias no DXF
DOOR_PREFIXES   = ("door", "porta")
WINDOW_PREFIXES = ("window", "janela")

# ---------------- reconhecimento de layer (v2: multi-idioma + AIA) ----------------
# Cada escritorio nomeia as layers do seu jeito. Em vez de prefixo fixo, casamos
# por TOKENS contidos no nome (PT/EN/ES/NO + padrao AIA A-WALL/A-DOOR/...). A layer
# pode vir com prefixo de xref: "xref-Casa-08$0$A-WALL" -> normaliza pro trecho
# apos o ultimo separador de xref.
WALL_TOKENS    = ("wall", "parede", "pared", "muro", "vegg")
DOOR_TOKENS    = ("door", "porta", "puerta")
WINDOW_TOKENS  = ("window", "janela", "ventana", "glaz")
# aberturas genericas ou combinadas (ex "PUERYVENTA" = puertas y ventanas):
# esquadria de tipo indefinido — decide porta/janela pelo tamanho do vao depois
OPENING_TOKENS = ("opening", "abertura", "pueryventa", "vano", "vao")


def _arco_tres_pontos(A, C, B):
    """Retorna a geometria circular de P1 -> C -> P2, ou None se colinear."""
    ax, ay = map(float, A)
    cx, cy = map(float, C)
    bx, by = map(float, B)
    denominator = 2.0 * (
        ax * (cy - by) + cx * (by - ay) + bx * (ay - cy)
    )
    chord = float(np.linalg.norm(np.asarray(B) - np.asarray(A)))
    if chord < 1e-6 or abs(denominator) < chord * chord * 1e-7:
        return None
    a2, c2, b2 = ax * ax + ay * ay, cx * cx + cy * cy, bx * bx + by * by
    ox = (a2 * (cy - by) + c2 * (by - ay) + b2 * (ay - cy)) / denominator
    oy = (a2 * (bx - cx) + c2 * (ax - bx) + b2 * (cx - ax)) / denominator
    radius = math.hypot(ax - ox, ay - oy)
    start = math.atan2(ay - oy, ax - ox)
    control = math.atan2(cy - oy, cx - ox)
    end = math.atan2(by - oy, bx - ox)
    full_turn = 2.0 * math.pi
    ccw_sweep = (end - start) % full_turn
    ccw_control = (control - start) % full_turn
    sweep = ccw_sweep if ccw_control <= ccw_sweep + 1e-7 else ccw_sweep - full_turn
    length = abs(sweep) * radius
    if not math.isfinite(length) or length < 1e-6:
        return None
    return {
        "centro": np.array([ox, oy], dtype=float),
        "raio": radius,
        "angulo_inicio": start,
        "varredura": sweep,
        "comprimento": length,
    }


def _frame_parede(parede, s):
    """Ponto, tangente e normal no comprimento longitudinal da parede."""
    arco = parede.get("arco")
    comprimento = float(parede["comprimento"])
    s = max(0.0, min(comprimento, float(s)))
    if arco:
        t = s / comprimento
        angle = arco["angulo_inicio"] + arco["varredura"] * t
        direction = 1.0 if arco["varredura"] >= 0 else -1.0
        tangent = np.array([-math.sin(angle) * direction,
                            math.cos(angle) * direction], dtype=float)
        point = arco["centro"] + arco["raio"] * np.array(
            [math.cos(angle), math.sin(angle)], dtype=float)
    else:
        A, B = parede["eixo"]
        tangent = (B - A) / np.linalg.norm(B - A)
        point = A + tangent * s
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    return point, tangent, normal


def _contorno_parede_curva(parede, max_segmento=0.12):
    comprimento = float(parede["comprimento"])
    passos = max(8, min(512, int(math.ceil(comprimento / max_segmento))))
    half_width = float(parede["espessura"]) / 2.0
    frames = [_frame_parede(parede, comprimento * i / passos)
              for i in range(passos + 1)]
    esquerda = [point + normal * half_width for point, _, normal in frames]
    direita = [point - normal * half_width for point, _, normal in reversed(frames)]
    return esquerda + direita


def _norm_layer(nome):
    """lower + remove prefixo de xref (trecho apos ultimo $ ou |)."""
    n = nome.lower()
    for sep in ("$", "|"):
        if sep in n:
            n = n.rsplit(sep, 1)[-1]
    return n


def classificar_layer(nome):
    """Retorna 'wall'|'door'|'window'|'opening'|None pra qualquer convencao."""
    n = _norm_layer(nome)
    if any(t in n for t in WALL_TOKENS):       # parede e mais especifica primeiro
        return "wall"
    if any(t in n for t in OPENING_TOKENS):
        return "opening"
    if any(t in n for t in DOOR_TOKENS):
        return "door"
    if any(t in n for t in WINDOW_TOKENS):
        return "window"
    return None

# Escala: candidatos unidade->metro. A heuristica testa cada um e escolhe o que
# faz mais faces parearem (ou seja, o que poe as espessuras na faixa de parede).
ESCALAS_CAND = (1.0, 0.01, 0.001, 0.0001)


def _segs_em_escala(brutos, escala):
    """(x1,y1,x2,y2,layer) brutos -> [(P1,P2,layer)] em metros, descartando
    segmentos degenerados."""
    segs = []
    for x1, y1, x2, y2, layer in brutos:
        P1 = np.array([x1, y1]) * escala
        P2 = np.array([x2, y2]) * escala
        if np.linalg.norm(P2 - P1) >= SEG_MIN:
            segs.append((P1, P2, layer))
    return segs


def detectar_escala_auto(brutos, ext):
    """Escolhe a escala testando o pareamento: a escala correta e a unica que
    poe as espessuras das faces em [ESP_MIN, ESP_MAX], entao pareia a maior
    FRACAO do comprimento das faces. Fracao e adimensional -> comparavel entre
    escalas. Cai na heuristica de extensao se nada parear."""
    melhor = (-1.0, None)
    for esc in ESCALAS_CAND:
        if not (1.0 <= ext * esc <= 1000.0):   # extensao plausivel de edificacao
            continue
        segs = _segs_em_escala(brutos, esc)
        if len(segs) < 2:
            continue
        L_seg = sum(float(np.linalg.norm(P2 - P1)) for P1, P2, _ in segs)
        if L_seg <= 0:
            continue
        paredes, _ = parear_paredes(mesclar_colineares(segs))
        # cada parede consome ~2 faces do seu comprimento -> fracao pareada
        frac = min(1.0, 2.0 * sum(p["comprimento"] for p in paredes) / L_seg)
        if frac > melhor[0]:
            melhor = (frac, esc)
    if melhor[1] is not None and melhor[0] > 0.05:
        return melhor[1]
    return 0.001 if ext > 1000 else (0.01 if ext > 100 else 1.0)


# ============================================================
# 1) LEITURA DO DXF
# ============================================================
def _segmentos_entidade_dxf(entidade):
    """Transforma uma entidade CAD finita em segmentos 2D brutos.

    Polylines com bulge, arcos, circulos, elipses e splines sao tessellados.
    INSERT e tratado pelo chamador para preservar a layer herdada do bloco.
    """
    tipo = entidade.dxftype()
    if tipo == "LINE":
        inicio, fim = entidade.dxf.start, entidade.dxf.end
        return [(inicio.x, inicio.y, fim.x, fim.y)]
    if tipo == "LWPOLYLINE":
        # virtual_entities preserva os arcos definidos por bulge.
        try:
            virtuais = list(entidade.virtual_entities())
            if virtuais:
                return [s for sub in virtuais for s in _segmentos_entidade_dxf(sub)]
        except Exception:
            pass
        pontos = list(entidade.get_points("xy"))
        segmentos = [
            (a[0], a[1], b[0], b[1]) for a, b in zip(pontos, pontos[1:])
        ]
        if entidade.closed and len(pontos) > 2:
            a, b = pontos[-1], pontos[0]
            segmentos.append((a[0], a[1], b[0], b[1]))
        return segmentos
    if tipo == "POLYLINE":
        pontos = [(v.dxf.location.x, v.dxf.location.y) for v in entidade.vertices]
        segmentos = [
            (a[0], a[1], b[0], b[1]) for a, b in zip(pontos, pontos[1:])
        ]
        if getattr(entidade, "is_closed", False) and len(pontos) > 2:
            a, b = pontos[-1], pontos[0]
            segmentos.append((a[0], a[1], b[0], b[1]))
        return segmentos
    if tipo in ("ARC", "CIRCLE"):
        centro, raio = entidade.dxf.center, float(entidade.dxf.radius)
        if tipo == "ARC":
            a0 = math.radians(float(entidade.dxf.start_angle))
            a1 = math.radians(float(entidade.dxf.end_angle))
            if a1 <= a0:
                a1 += 2 * math.pi
        else:
            a0, a1 = 0.0, 2 * math.pi
        passos = max(8, int(math.ceil((a1 - a0) / math.radians(10))))
        pontos = [
            (centro.x + raio * math.cos(a0 + (a1 - a0) * i / passos),
             centro.y + raio * math.sin(a0 + (a1 - a0) * i / passos))
            for i in range(passos + 1)
        ]
        return [(a[0], a[1], b[0], b[1]) for a, b in zip(pontos, pontos[1:])]
    if tipo in ("ELLIPSE", "SPLINE"):
        try:
            pontos = [(float(p[0]), float(p[1])) for p in entidade.flattening(0.01)]
        except Exception:
            pontos = []
        return [(a[0], a[1], b[0], b[1]) for a, b in zip(pontos, pontos[1:])]
    if tipo in ("SOLID", "TRACE", "3DFACE"):
        pontos = []
        for nome in ("vtx0", "vtx1", "vtx2", "vtx3"):
            try:
                p = getattr(entidade.dxf, nome)
                pontos.append((float(p.x), float(p.y)))
            except Exception:
                pass
        pontos = list(dict.fromkeys(pontos))
        if len(pontos) >= 3:
            return [
                (pontos[i][0], pontos[i][1],
                 pontos[(i + 1) % len(pontos)][0], pontos[(i + 1) % len(pontos)][1])
                for i in range(len(pontos))
            ]
    return []


def ler_segmentos_parede(dxf_path, escala_forcada=None):
    """Retorna (segs, escala): segs = lista de (P1, P2, layer) em METROS."""
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    brutos = []

    def coletar(entidade, layer_herdada=None):
        layer_propria = str(getattr(entidade.dxf, "layer", "") or "")
        layer = (layer_herdada if layer_propria in ("", "0") and layer_herdada
                 else layer_propria)
        if entidade.dxftype() == "INSERT":
            role = classificar_layer(layer)
            try:
                for sub in entidade.virtual_entities():
                    coletar(sub, layer if role == "wall" else layer_herdada)
            except Exception:
                pass
            return
        if classificar_layer(layer) != "wall":
            return
        for x1, y1, x2, y2 in _segmentos_entidade_dxf(entidade):
            brutos.append((x1, y1, x2, y2, layer))

    for entidade in msp:
        coletar(entidade)

    if not brutos:
        raise SystemExit("Nenhum segmento em layers de parede "
                         "(wall/parede/muro/a-wall/...).")

    xs = [v for s in brutos for v in (s[0], s[2])]
    ys = [v for s in brutos for v in (s[1], s[3])]
    ext = max(max(xs) - min(xs), max(ys) - min(ys))
    if escala_forcada is not None:
        escala = escala_forcada
    else:
        escala = detectar_escala_auto(brutos, ext)

    segs = []
    for x1, y1, x2, y2, layer in brutos:
        P1 = np.array([x1, y1]) * escala
        P2 = np.array([x2, y2]) * escala
        if np.linalg.norm(P2 - P1) >= SEG_MIN:
            segs.append((P1, P2, layer))
    return segs, escala


# ============================================================
# 1b) FUSAO DE COLINEARES (reconecta faces cortadas por vaos)
# ============================================================
def mesclar_colineares(segs, perp_tol=MERGE_PERP_TOL, gap_max=MERGE_GAP_MAX):
    """Funde segmentos colineares da MESMA layer (mesma face de parede) que
    o DXF desenhou partidos por causa de um vao de porta/janela no meio.

    Sem isso, a face vira 2+ segmentos curtos e o pareamento gera paredes
    fragmentadas — a esquadria acaba casada com o fragmento mais proximo
    (que termina bem onde ela esta) em vez da parede inteira, dando
    posicao/largura erradas.
    """
    n = len(segs)
    thetas = []
    for P1, P2, _ in segs:
        d = P2 - P1
        thetas.append(math.atan2(d[1], d[0]) % math.pi)

    usados = [False] * n
    resultado = []
    for i in range(n):
        if usados[i]:
            continue
        P1, P2, layer = segs[i]
        d = P2 - P1
        u = d / np.linalg.norm(d)
        nrm = np.array([-u[1], u[0]])
        c0 = float(nrm @ P1)

        grupo = [i]
        usados[i] = True
        changed = True
        while changed:
            changed = False
            ts = [v for k in grupo for v in
                  (float(u @ segs[k][0]), float(u @ segs[k][1]))]
            tmin, tmax = min(ts), max(ts)
            for j in range(n):
                if usados[j] or segs[j][2] != layer:
                    continue
                dt = abs(thetas[i] - thetas[j])
                dt = min(dt, math.pi - dt)
                if dt > ANG_TOL:
                    continue
                Pj1, Pj2, _ = segs[j]
                if abs(float(nrm @ Pj1) - c0) > perp_tol:
                    continue
                tj1, tj2 = sorted((float(u @ Pj1), float(u @ Pj2)))
                gap = max(0.0, max(tj1 - tmax, tmin - tj2))
                if gap <= gap_max:
                    grupo.append(j)
                    usados[j] = True
                    changed = True

        ts = [v for k in grupo for v in
              (float(u @ segs[k][0]), float(u @ segs[k][1]))]
        tmin, tmax = min(ts), max(ts)
        A = u * tmin + nrm * c0
        B = u * tmax + nrm * c0
        resultado.append((A, B, layer))
    return resultado


# ============================================================
# 2) PAREAMENTO  (faces paralelas -> parede com espessura)
# ============================================================
def parear_paredes(segs):
    """Casa pares de segmentos paralelos proximos. Retorna (paredes, sobras).

    parede = dict(eixo=(A,B), espessura, comprimento, layer)
    """
    n = len(segs)
    # direcao de cada segmento (mod pi)
    thetas = []
    for P1, P2, _ in segs:
        d = P2 - P1
        thetas.append(math.atan2(d[1], d[0]) % math.pi)

    # agrupa por direcao (com wraparound em pi)
    grupos = []   # lista de listas de indices
    usados_grupo = [False] * n
    for i in range(n):
        if usados_grupo[i]:
            continue
        grupo = [i]
        usados_grupo[i] = True
        for j in range(i + 1, n):
            if usados_grupo[j]:
                continue
            dt = abs(thetas[i] - thetas[j])
            dt = min(dt, math.pi - dt)
            if dt <= ANG_TOL:
                grupo.append(j)
                usados_grupo[j] = True
        grupos.append(grupo)

    paredes = []
    pareado = [False] * n

    for grupo in grupos:
        if len(grupo) < 2:
            continue
        # direcao de referencia = a do segmento mais longo do grupo
        ref = max(grupo, key=lambda k: np.linalg.norm(segs[k][1] - segs[k][0]))
        d = segs[ref][1] - segs[ref][0]
        u = d / np.linalg.norm(d)
        nrm = np.array([-u[1], u[0]])

        # projecoes: offset perpendicular c e intervalo longitudinal [t1,t2]
        info = {}
        for k in grupo:
            P1, P2, _ = segs[k]
            c = float(nrm @ P1)
            t1, t2 = sorted((float(u @ P1), float(u @ P2)))
            info[k] = (c, t1, t2)

        # candidatos (i<j): espessura plausivel + sobreposicao suficiente
        cands = []
        for ii in range(len(grupo)):
            for jj in range(ii + 1, len(grupo)):
                a, b = grupo[ii], grupo[jj]
                ca, t1a, t2a = info[a]
                cb, t1b, t2b = info[b]
                esp = abs(ca - cb)
                if not (ESP_MIN <= esp <= ESP_MAX):
                    continue
                ov1, ov2 = max(t1a, t1b), min(t2a, t2b)
                if ov2 - ov1 < OVERLAP_MIN:
                    continue
                cands.append((ov2 - ov1, a, b, esp, ov1, ov2, (ca + cb) / 2))

        # greedy: maior sobreposicao primeiro, cada face usada 1x
        cands.sort(key=lambda c: -c[0])
        for ov_len, a, b, esp, ov1, ov2, c_eixo in cands:
            if pareado[a] or pareado[b]:
                continue
            pareado[a] = pareado[b] = True
            A = u * ov1 + nrm * c_eixo
            B = u * ov2 + nrm * c_eixo
            paredes.append({
                "eixo": (A, B),
                "espessura": esp,
                "comprimento": ov_len,
                "layer": segs[a][2],
            })

    sobras = [segs[k] for k in range(n)
              if not pareado[k]
              and np.linalg.norm(segs[k][1] - segs[k][0]) >= LEFTOVER_WARN]
    return paredes, sobras


def fracao_pareada(paredes, segs):
    """Fracao do comprimento das faces que virou parede (cada parede ~2 faces).
    Baixa = planta single-line (paredes desenhadas como eixo, sem duas faces)."""
    L_seg = sum(float(np.linalg.norm(P2 - P1)) for P1, P2, _ in segs)
    if L_seg <= 0:
        return 0.0
    return min(1.0, 2.0 * sum(p["comprimento"] for p in paredes) / L_seg)


def paredes_single_line(segs, esp_default):
    """Modo SINGLE-LINE: cada segmento JA e o eixo da parede -> vira uma parede
    com espessura default (a espessura nao esta desenhada, so anotada em texto)."""
    paredes = []
    for P1, P2, layer in segs:
        L = float(np.linalg.norm(P2 - P1))
        if L < OVERLAP_MIN:          # ignora tracinhos/ruido
            continue
        paredes.append({"eixo": (P1.astype(float), P2.astype(float)),
                        "espessura": float(esp_default), "comprimento": L,
                        "layer": layer})
    return paredes


# ============================================================
# 2a) COSTURA DE CANTOS (estende pontas ate a parede vizinha)
# ============================================================
def costurar_cantos(paredes, tol_canto=0.45, ang_min=math.radians(20.0)):
    """Estende cada ponta de parede ate o eixo da parede vizinha que ela
    encontra num canto (junçao L/T), fechando as frestas. Mover a ponta ao
    longo do proprio eixo nao altera a reta, entao a direçao se mantem."""
    eixos = [[p["eixo"][0].astype(float).copy(),
              p["eixo"][1].astype(float).copy()] for p in paredes]
    dirs = [(B - A) / np.linalg.norm(B - A) for A, B in eixos]
    cos_min = math.cos(ang_min)
    n_cost = 0

    def intersec(Ai, ui, Aj, uj):
        det = ui[0] * (-uj[1]) - ui[1] * (-uj[0])
        if abs(det) < 1e-9:
            return None
        rhs = Aj - Ai
        t = (rhs[0] * (-uj[1]) - rhs[1] * (-uj[0])) / det
        s = (ui[0] * rhs[1] - ui[1] * rhs[0]) / det
        return t, s

    for i in range(len(eixos)):
        A0, ui = eixos[i][0].copy(), dirs[i]
        for ponta in (0, 1):
            P = eixos[i][ponta]
            melhor = None
            for j in range(len(eixos)):
                if j == i or abs(float(ui @ dirs[j])) > cos_min:
                    continue
                Aj, Bj = eixos[j]
                Lj = float(np.linalg.norm(Bj - Aj))
                r = intersec(A0, ui, Aj, dirs[j])
                if r is None:
                    continue
                _, s = r
                X = A0 + r[0] * ui
                dist = float(np.linalg.norm(X - P))
                if dist <= tol_canto and -0.3 <= s <= Lj + 0.3:
                    if melhor is None or dist < melhor[0]:
                        melhor = (dist, X.copy())
            if melhor:
                eixos[i][ponta][:] = melhor[1]
                n_cost += 1

    novas = []
    for i, p in enumerate(paredes):
        A, B = eixos[i]
        q = dict(p)
        q["eixo"] = (A, B)
        q["comprimento"] = float(np.linalg.norm(B - A))
        novas.append(q)
    return novas, n_cost


# ============================================================
# 2b) ESQUADRIAS (blocos de porta/janela -> aberturas nas paredes)
# ============================================================
def _amostrar_entidade(sub):
    """Pontos (x,y) brutos de uma entidade de esquadria (LINE/POLYLINE/ARC/SPLINE/
    CIRCLE). ARC/SPLINE sao amostrados; CIRCLE vira so o centro (simbolo pontual)."""
    t = sub.dxftype()
    if t == "LINE":
        s, en = sub.dxf.start, sub.dxf.end
        return [(s.x, s.y), (en.x, en.y)]
    if t == "LWPOLYLINE":
        return [(p[0], p[1]) for p in sub.get_points("xy")]
    if t == "POLYLINE":
        return [(v.dxf.location.x, v.dxf.location.y) for v in sub.vertices]
    if t == "CIRCLE":
        c = sub.dxf.center
        return [(c.x, c.y)]
    if t == "ARC":
        c, r = sub.dxf.center, sub.dxf.radius
        a0, a1 = math.radians(sub.dxf.start_angle), math.radians(sub.dxf.end_angle)
        if a1 <= a0:
            a1 += 2 * math.pi
        return [(c.x + r * math.cos(a0 + (a1 - a0) * k / 8),
                 c.y + r * math.sin(a0 + (a1 - a0) * k / 8)) for k in range(9)]
    if t == "SPLINE":
        try:
            return [(p[0], p[1]) for p in sub.flattening(0.05)]
        except Exception:
            try:
                return [(p[0], p[1]) for p in sub.control_points]
            except Exception:
                return []
    return []


def _clusterizar(itens, raio):
    """itens: [(pts(N,2) em metros, tipo)]. Une por proximidade (mesmo tipo,
    distancia min entre pontos <= raio) via union-find. Retorna [[indices]]."""
    n = len(itens)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if itens[i][1] != itens[j][1]:
                continue
            # distancia minima entre os dois conjuntos de pontos
            d = np.min(np.linalg.norm(
                itens[i][0][:, None, :] - itens[j][0][None, :, :], axis=2))
            if d <= raio:
                parent[find(i)] = find(j)

    from collections import defaultdict
    grupos = defaultdict(list)
    for i in range(n):
        grupos[find(i)].append(i)
    return list(grupos.values())


def ler_blocos_esquadrias(dxf_path, escala, raio_cluster=0.40):
    """Retorna [{tipo:'door'|'window', pts:(N,2) em metros}].

    Suporta os 2 jeitos de desenhar esquadria:
      - bloco INSERT  -> expande virtual_entities (cada INSERT = 1 esquadria)
      - geometria SOLTA (LWPOLYLINE/LINE/SPLINE nas layers Door/Window/Opening,
        sem bloco) -> agrupa por proximidade (cada cluster = 1 esquadria)
    """
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    blocos = []
    soltos = []  # (pts(N,2) em metros, tipo) — geometria nao-INSERT

    for e in msp:
        cls = classificar_layer(e.dxf.layer)
        if cls not in ("door", "window", "opening"):
            continue
        tipo = "window" if cls == "window" else "door"  # opening -> door por ora
        if e.dxftype() == "INSERT":
            pts = []
            try:
                for sub in e.virtual_entities():
                    if sub.dxftype() != "ARC":  # arco = leque, nao faz parte do vao
                        pts += _amostrar_entidade(sub)
            except Exception:
                pass
            if not pts:
                p = e.dxf.insert
                pts = [(p.x, p.y)]
            blocos.append({"tipo": tipo, "pts": np.array(pts) * escala})
        else:
            pts = _amostrar_entidade(e)
            if pts:
                soltos.append((np.array(pts) * escala, tipo))

    # geometria solta -> clusters -> esquadrias
    for grupo in _clusterizar(soltos, raio_cluster):
        allpts = np.vstack([soltos[i][0] for i in grupo])
        blocos.append({"tipo": soltos[grupo[0]][1], "pts": allpts})
    return blocos


def casar_esquadrias_com_paredes(blocos, paredes,
                                 larg_min=0.40, larg_max=3.00, tol_perp=0.20,
                                 larg_default_door=0.80, larg_default_window=1.00,
                                 long_margin=1.50):
    """Pra cada bloco: acha a parede dona e mede o VAO usando APENAS os pontos
    que estao SOBRE a parede (dentro da espessura + tol_perp).

    Isso descarta a folha aberta e o leque de abertura (que se projetam pra
    dentro do comodo) — eram eles que inflavam a largura (vao + arco = 2x) e
    deslocavam o centro pra fora da parede.

    Retorna: {parede_idx, tipo, s_centro, largura}
    """
    aberturas = []
    for bl in blocos:
        pts = bl["pts"]
        candidate_max_width = max(
            float(larg_max),
            float(bl.get("max_width", larg_max)),
            float(bl.get("declared_width", 0.0) or 0.0) * 1.10,
        )
        melhor = None  # (chave, idx, s_centro, largura)
        for i, p in enumerate(paredes):
            A, B = p["eixo"]
            d = B - A
            L = np.linalg.norm(d)
            if L < 1e-9:
                continue
            u = d / L
            nrm = np.array([-u[1], u[0]])

            rel = pts - A
            s_all = rel @ u               # posicao longitudinal de cada ponto
            d_all = np.abs(rel @ nrm)     # distancia perpendicular de cada ponto

            # pontos "sobre a parede": colados nela E dentro da extensao
            # (margem generosa pra cobrir vao-de-canto, onde a parede
            # termina um pouco antes/depois do vao real da esquadria)
            sobre = ((d_all <= p["espessura"] / 2 + tol_perp) &
                     (s_all >= -long_margin) & (s_all <= L + long_margin))
            if int(sobre.sum()) < 2:
                continue

            s_sub = s_all[sobre]
            larg = float(s_sub.max() - s_sub.min())
            if larg < larg_min:  # vao degenerado -> usa default por tipo
                larg = float(
                    bl.get("declared_width")
                    or (
                        larg_default_door
                        if bl["tipo"] == "door"
                        else larg_default_window
                    )
                )
            declared_width = bl.get("declared_width")
            if declared_width is not None and abs(larg - float(declared_width)) <= max(
                0.20, float(declared_width) * 0.30,
            ):
                larg = float(declared_width)
            larg = float(np.clip(larg, larg_min, candidate_max_width))
            # NAO forca o centro pra dentro da parede: se o vao real fica na
            # ponta/fora da parede (vao-de-canto), a posicao certa eh essa.
            s_centro = float((s_sub.max() + s_sub.min()) / 2)

            # melhor parede = a com MAIS pontos sobre ela (desempate: mais colada)
            chave = (int(sobre.sum()), -float(d_all[sobre].mean()))
            if melhor is None or chave > melhor[0]:
                melhor = (chave, i, s_centro, larg)

        if melhor:
            _, i, s_centro, larg = melhor
            abertura = {
                "parede_idx": i,
                "tipo": bl["tipo"],
                "s_centro": s_centro,
                "largura": larg,
            }
            for campo in (
                "origem", "confidence", "source_layer", "block_name",
                "source_text", "semantic_subtype", "semantic_reason",
            ):
                if bl.get(campo) is not None:
                    abertura[campo] = bl[campo]
            for campo in ("declared_width", "declared_height"):
                if bl.get(campo) is not None:
                    abertura[campo] = float(bl[campo])
            aberturas.append(abertura)
    return aberturas


# ============================================================
# 3) LAJE (contorno convexo das paredes)
# ============================================================
def hull_convexo(pontos):
    """Monotone chain. pontos: (N,2) -> hull CCW sem repetir o 1o."""
    pts = sorted({(round(p[0], 4), round(p[1], 4)) for p in pontos})
    if len(pts) < 3:
        return [np.array(p) for p in pts]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [np.array(p) for p in (lower[:-1] + upper[:-1])]


# ============================================================
# 4) GERACAO DO IFC
# ============================================================
def construir_ifc(paredes, laje_poly, altura, esp_laje, pavimento, projeto, out_path,
                  aberturas=None, cobertura=True,
                  porta_altura=2.10, janela_altura=1.20, janela_peitoril=1.00,
                  esquadria_detalhada=False,
                  piso_ativo=True, piso_esp=None, teto_esp=None, spaces=None,
                  forro=None, esquadria_sobreposicao=0.02):
    f = api.run("project.create_file", version="IFC4")
    proj = api.run("root.create_entity", f, ifc_class="IfcProject", name=projeto)
    # METROS explicitos (o default do assign_unit eh MILIMETRO, o que faria
    # todas as dimensoes escritas aqui — em metros — virarem mm no arquivo)
    metro = api.run("unit.add_si_unit", f, unit_type="LENGTHUNIT")
    api.run("unit.assign_unit", f, units=[metro])
    ctx = api.run("context.add_context", f, context_type="Model")
    body = api.run("context.add_context", f, context_type="Model",
                   context_identifier="Body", target_view="MODEL_VIEW", parent=ctx)

    site = api.run("root.create_entity", f, ifc_class="IfcSite", name="Site")
    bld = api.run("root.create_entity", f, ifc_class="IfcBuilding", name="Edificio")
    storey = api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name=pavimento)
    storey.Elevation = 0.0
    api.run("aggregate.assign_object", f, products=[site], relating_object=proj)
    api.run("aggregate.assign_object", f, products=[bld], relating_object=site)
    api.run("aggregate.assign_object", f, products=[storey], relating_object=bld)

    def _placement(prod, x, y, z, ang):
        M = np.eye(4)
        c, s = math.cos(ang), math.sin(ang)
        M[:2, :2] = [[c, -s], [s, c]]
        M[:3, 3] = [x, y, z]
        api.run("geometry.edit_object_placement", f, product=prod, matrix=M)

    guids_preservados = set()

    def _rep_extrusao(profile, depth):
        solid = f.create_entity(
            "IfcExtrudedAreaSolid", SweptArea=profile,
            Position=f.create_entity(
                "IfcAxis2Placement3D",
                Location=f.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))),
            ExtrudedDirection=f.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            Depth=float(depth))
        return f.create_entity(
            "IfcShapeRepresentation", ContextOfItems=body,
            RepresentationIdentifier="Body", RepresentationType="SweptSolid",
            Items=[solid])

    def _aplicar_metadados(prod, meta):
        """Preserva identidade/proveniencia quando o modelo veio de outro IFC."""
        if not meta:
            return prod
        guid = meta.get("guid")
        guid_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
        guid_valido = (isinstance(guid, str) and len(guid) == 22
                       and all(char in guid_chars for char in guid))
        if guid_valido and guid not in guids_preservados:
            try:
                prod.GlobalId = str(guid)
                guids_preservados.add(guid)
            except Exception:
                pass
        origem = meta.get("origem")
        classe_origem = meta.get("ifc_class")
        nivel = meta.get("nivel")
        descricao = " | ".join(
            parte for parte in (
                f"origem={origem}" if origem else "",
                f"classe={classe_origem}" if classe_origem else "",
                f"nivel={nivel}" if nivel else "",
            ) if parte
        )
        if descricao:
            try:
                prod.Description = descricao
            except Exception:
                pass
        return prod

    def _caixa(prod_classe, nome, larg, esp, alt, x, y, z, ang, meta=None):
        """Cria produto com caixa extrudada (perfil larg x esp, altura alt)."""
        prod = api.run("root.create_entity", f, ifc_class=prod_classe, name=nome)
        _aplicar_metadados(prod, meta)
        prof = f.create_entity(
            "IfcRectangleProfileDef", ProfileType="AREA",
            Position=f.create_entity(
                "IfcAxis2Placement2D",
                Location=f.create_entity("IfcCartesianPoint",
                                         Coordinates=(larg / 2.0, 0.0))),
            XDim=float(larg), YDim=float(esp))
        rep = _rep_extrusao(prof, alt)
        api.run("geometry.assign_representation", f, product=prod, representation=rep)
        _placement(prod, x, y, z, ang)
        return prod

    def _parede_curva(prod_classe, nome, parede, alt, z, meta=None):
        """Extruda a faixa circular como um único IfcWall semanticamente selecionável."""
        prod = api.run("root.create_entity", f, ifc_class=prod_classe, name=nome)
        _aplicar_metadados(prod, meta)
        vertices = _contorno_parede_curva(parede)
        points = [
            f.create_entity(
                "IfcCartesianPoint",
                Coordinates=(float(point[0]), float(point[1])),
            )
            for point in vertices
        ]
        points.append(points[0])
        outer_curve = f.create_entity("IfcPolyline", Points=points)
        profile = f.create_entity(
            "IfcArbitraryClosedProfileDef",
            ProfileType="AREA",
            OuterCurve=outer_curve,
        )
        rep = _rep_extrusao(profile, alt)
        api.run("geometry.assign_representation", f, product=prod, representation=rep)
        _placement(prod, 0.0, 0.0, z, 0.0)
        return prod

    def _solid_box(x0, x1, y0, y1, z0, z1):
        """Caixa extrudada no frame LOCAL da esquadria (x=largura, y=espessura,
        z=altura). Varios _solid_box compoem o desenho da porta/janela."""
        prof = f.create_entity(
            "IfcRectangleProfileDef", ProfileType="AREA",
            Position=f.create_entity(
                "IfcAxis2Placement2D",
                Location=f.create_entity("IfcCartesianPoint",
                                         Coordinates=((x0 + x1) / 2.0, (y0 + y1) / 2.0))),
            XDim=float(x1 - x0), YDim=float(y1 - y0))
        return f.create_entity(
            "IfcExtrudedAreaSolid", SweptArea=prof,
            Position=f.create_entity(
                "IfcAxis2Placement3D",
                Location=f.create_entity("IfcCartesianPoint",
                                         Coordinates=(0.0, 0.0, float(z0)))),
            ExtrudedDirection=f.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            Depth=float(z1 - z0))

    def _esquadria_multi(prod_classe, nome, solids, x, y, z, ang, meta=None):
        """Produto com varios solids (batente+folha / moldura+vidro)."""
        prod = api.run("root.create_entity", f, ifc_class=prod_classe, name=nome)
        _aplicar_metadados(prod, meta)
        rep = f.create_entity(
            "IfcShapeRepresentation", ContextOfItems=body,
            RepresentationIdentifier="Body", RepresentationType="SweptSolid",
            Items=solids)
        api.run("geometry.assign_representation", f, product=prod, representation=rep)
        _placement(prod, x, y, z, ang)
        return prod

    def _porta_detalhada(nome, larg, esp, alt, x, y, z, ang, meta=None):
        tl = min(0.05, larg / 4)   # batente
        tp = min(0.04, esp / 2)    # folha
        ov = max(0.0, min(float(esquadria_sobreposicao), larg / 4))
        solids = [
            _solid_box(-ov, tl, -esp/2, esp/2, 0, alt + ov),
            _solid_box(larg - tl, larg + ov, -esp/2, esp/2, 0, alt + ov),
            _solid_box(-ov, larg + ov, -esp/2, esp/2, alt - tl, alt + ov),
            _solid_box(tl, larg - tl, -tp/2, tp/2, 0, alt - tl),
        ]
        return _esquadria_multi("IfcDoor", nome, solids, x, y, z, ang, meta)

    def _janela_detalhada(nome, larg, esp, alt, x, y, z, ang, meta=None):
        tl = min(0.05, larg / 6)   # moldura
        tg = 0.02                  # vidro fino
        ov = max(0.0, min(float(esquadria_sobreposicao), larg / 6))
        solids = [
            _solid_box(-ov, tl, -esp/2, esp/2, -ov, alt + ov),
            _solid_box(larg - tl, larg + ov, -esp/2, esp/2, -ov, alt + ov),
            _solid_box(-ov, larg + ov, -esp/2, esp/2, -ov, tl),
            _solid_box(-ov, larg + ov, -esp/2, esp/2, alt - tl, alt + ov),
            _solid_box(larg/2 - tl/2, larg/2 + tl/2, -esp/2, esp/2, tl, alt - tl),  # travessa
            _solid_box(tl, larg - tl, -tg/2, tg/2, tl, alt - tl),  # vidro
        ]
        return _esquadria_multi("IfcWindow", nome, solids, x, y, z, ang, meta)

    # ---- paredes ----
    walls = []
    for i, p in enumerate(paredes, 1):
        A, B = p["eixo"]
        ang = math.atan2(*(B - A)[::-1])  # atan2(dy, dx)
        wall_altura = float(
            p.get("altura")
            or p.get("altura_observada")
            or altura
        )
        wall_z = float(p.get("elevacao", 0.0) or 0.0)
        is_column = p.get("ifc_class") == "IfcColumn" or p.get("tipo") == "column"
        product_class = "IfcColumn" if is_column else "IfcWall"
        default_name = f"Pilar-{i:03d}" if is_column else f"Parede-{i:03d}"
        if p.get("arco") and not is_column:
            wall = _parede_curva(
                product_class, p.get("nome") or default_name,
                p, wall_altura, wall_z, p,
            )
        else:
            wall = _caixa(product_class, p.get("nome") or default_name,
                          p["comprimento"], p["espessura"], wall_altura,
                          float(A[0]), float(A[1]), wall_z, ang, p)
        api.run("spatial.assign_container", f, products=[wall], relating_structure=storey)
        walls.append(wall)

    # ---- aberturas (portas/janelas) ----
    n_p = n_j = 0
    for ab in (aberturas or []):
        p = paredes[ab["parede_idx"]]
        wall = walls[ab["parede_idx"]]
        A, B = p["eixo"]
        center, u, _normal = _frame_parede(p, ab["s_centro"])
        ang = math.atan2(u[1], u[0])
        larg = ab["largura"]
        if ab["tipo"] == "door":
            peitoril = ab.get("peitoril")
            z0 = float(peitoril) if peitoril is not None else 0.0
            alt = float(ab.get("altura") or porta_altura)
            n_p += 1
            classe, nome = "IfcDoor", ab.get("nome") or f"Porta-{n_p:03d}"
        else:
            peitoril = ab.get("peitoril")
            z0 = float(peitoril) if peitoril is not None else janela_peitoril
            alt = float(ab.get("altura") or janela_altura)
            n_j += 1
            classe, nome = "IfcWindow", ab.get("nome") or f"Janela-{n_j:03d}"
        wall_base = float(p.get("elevacao", 0.0) or 0.0)
        z0 += wall_base
        wall_top = wall_base + float(
            p.get("altura")
            or p.get("altura_observada")
            or altura
        )
        if z0 < wall_base:
            alt -= wall_base - z0
            z0 = wall_base
        alt = min(alt, wall_top - z0)
        if alt <= 0.05:
            continue
        # A esquadria é reta e tangente ao eixo no centro, mesmo em parede curva.
        P0 = center - u * (larg / 2)
        curvature_allowance = 0.0
        if p.get("arco"):
            radius = max(float(p["arco"]["raio"]), larg / 2 + 1e-6)
            curvature_allowance = 2.0 * (
                radius - math.sqrt(max(0.0, radius * radius - (larg / 2) ** 2))
            )
        # opening (um pouco mais "gordo" que a parede pra cortar limpo)
        opening = _caixa("IfcOpeningElement", f"Vao-{nome}",
                         larg, p["espessura"] + 0.02 + curvature_allowance, alt,
                         float(P0[0]), float(P0[1]), z0, ang)
        api.run("feature.add_feature", f, feature=opening, element=wall)
        # elemento que preenche (porta/janela)
        if esquadria_detalhada:
            if ab["tipo"] == "door":
                fill = _porta_detalhada(nome, larg, p["espessura"], alt,
                                        float(P0[0]), float(P0[1]), z0, ang, ab)
            else:
                fill = _janela_detalhada(nome, larg, p["espessura"], alt,
                                         float(P0[0]), float(P0[1]), z0, ang, ab)
        else:
            fill = _caixa(classe, nome, larg, p["espessura"], alt,
                          float(P0[0]), float(P0[1]), z0, ang, ab)
        # Atributos proprios de IfcDoor/IfcWindow (IFC4) — sem eles viewers
        # exigentes (Bonsai/Blender) nao renderizam a esquadria.
        fill.PredefinedType = "DOOR" if ab["tipo"] == "door" else "WINDOW"
        fill.OverallHeight = float(alt)
        fill.OverallWidth = float(larg)
        api.run("feature.add_filling", f, opening=opening, element=fill)
        api.run("spatial.assign_container", f, products=[fill], relating_structure=storey)

        # VERGA: se a esquadria fica (parcialmente) FORA da parede dona — caso
        # vao-de-canto —, nao ha parede acima do vao. Cria um trecho de parede
        # de z=topo_da_esquadria ate o pe-direito pra fechar o vao aberto.
        L_par = float(p["comprimento"])
        s0, s1 = ab["s_centro"] - larg / 2, ab["s_centro"] + larg / 2
        dentro = max(0.0, min(s1, L_par) - max(s0, 0.0))
        wall_top = wall_base + float(
            p.get("altura")
            or p.get("altura_observada")
            or altura
        )
        if (larg - dentro) > 0.05 and (z0 + alt) < wall_top - 1e-3:
            verga = _caixa("IfcWall", f"Verga-{nome}",
                           larg, p["espessura"], wall_top - (z0 + alt),
                           float(P0[0]), float(P0[1]), z0 + alt, ang)
            api.run("spatial.assign_container", f, products=[verga], relating_structure=storey)

    # ---- ambientes fechados + forro por ambiente ----
    # O editor calcula os ciclos pelas paredes antes da exportacao. Usar esses
    # mesmos contornos aqui garante que o PNG aprovado e o IFC descrevam os
    # mesmos comodos, sem uma segunda inferencia geometrica no exportador.
    forro_config = dict(forro or {})
    forro_ativo = bool(forro_config.get("ativo", False))
    forro_espessura = float(forro_config.get("espessura", 0.03))
    forro_altura = float(
        forro_config.get(
            "altura",
            max(0.0, float(altura) - forro_espessura),
        )
    )
    if forro_espessura <= 0:
        raise ValueError("espessura do forro deve ser positiva")
    if forro_altura <= 0:
        raise ValueError("altura do forro deve ser positiva")
    if forro_altura + forro_espessura > float(altura) + 1e-6:
        raise ValueError("forro ultrapassa a altura das paredes")
    space_height = forro_altura if forro_ativo else float(altura)

    for i, space_data in enumerate(spaces or [], 1):
        vertices = [
            (float(value[0]), float(value[1]))
            for value in space_data.get("contorno", [])
        ]
        if len(vertices) > 1 and vertices[0] == vertices[-1]:
            vertices.pop()
        if len(vertices) < 3:
            continue
        signed_area = 0.5 * sum(
            ax * by - bx * ay
            for (ax, ay), (bx, by) in zip(vertices, vertices[1:] + vertices[:1])
        )
        if abs(signed_area) <= 1e-6:
            continue
        if signed_area < 0:
            vertices.reverse()

        points = [
            f.create_entity(
                "IfcCartesianPoint",
                Coordinates=(float(x), float(y)),
            )
            for x, y in vertices
        ]
        points.append(points[0])
        curve = f.create_entity("IfcPolyline", Points=points)
        profile = f.create_entity(
            "IfcArbitraryClosedProfileDef",
            ProfileType="AREA",
            OuterCurve=curve,
        )
        space = api.run(
            "root.create_entity",
            f,
            ifc_class="IfcSpace",
            name=space_data.get("id") or f"SPACE-{i:03d}",
            predefined_type="INTERNAL",
        )
        space.LongName = space_data.get("nome") or f"Comodo {i:03d}"
        if space_data.get("area") is not None:
            space.Description = f"area={float(space_data['area']):.3f} m2"
        representation = _rep_extrusao(profile, space_height)
        api.run(
            "geometry.assign_representation",
            f,
            product=space,
            representation=representation,
        )
        _placement(space, 0.0, 0.0, 0.0, 0.0)
        api.run(
            "aggregate.assign_object",
            f,
            products=[space],
            relating_object=storey,
        )
        if forro_ativo:
            ceiling = api.run(
                "root.create_entity",
                f,
                ifc_class="IfcCovering",
                name=f"Forro-{space.Name}",
                predefined_type="CEILING",
            )
            ceiling.Description = (
                f"Forro por ambiente | cota inferior={forro_altura:.3f} m | "
                f"espessura={forro_espessura:.3f} m"
            )
            ceiling_representation = _rep_extrusao(profile, forro_espessura)
            api.run(
                "geometry.assign_representation",
                f,
                product=ceiling,
                representation=ceiling_representation,
            )
            _placement(ceiling, 0.0, 0.0, forro_altura, 0.0)
            api.run(
                "spatial.assign_container",
                f,
                products=[ceiling],
                relating_structure=space,
            )

    # ---- lajes (piso + cobertura), com espessura propria cada ----
    def _laje(nome, z_base, esp, predefined_type):
        slab = api.run(
            "root.create_entity",
            f,
            ifc_class="IfcSlab",
            name=nome,
            predefined_type=predefined_type,
        )
        pts = [f.create_entity("IfcCartesianPoint",
                               Coordinates=(float(p[0]), float(p[1])))
               for p in laje_poly]
        pts.append(pts[0])  # fecha
        poly = f.create_entity("IfcPolyline", Points=pts)
        prof = f.create_entity("IfcArbitraryClosedProfileDef",
                               ProfileType="AREA", OuterCurve=poly)
        rep = _rep_extrusao(prof, esp)
        api.run("geometry.assign_representation", f, product=slab, representation=rep)
        _placement(slab, 0.0, 0.0, z_base, 0.0)
        api.run("spatial.assign_container", f, products=[slab], relating_structure=storey)

    _piso_esp = piso_esp if piso_esp is not None else esp_laje
    _teto_esp = teto_esp if teto_esp is not None else esp_laje
    if len(laje_poly) >= 3:
        if piso_ativo:
            _laje(
                "Laje-Estrutural-Piso",
                -_piso_esp,
                _piso_esp,
                "FLOOR",
            )
        if cobertura:
            _laje(
                "Laje-Estrutural-Superior",
                altura,
                _teto_esp,
                "FLOOR",
            )

    f.write(str(out_path))
    return f


# ============================================================
# 5) PREVIEW: malha o IFC -> PLY colorido
# ============================================================
def ifc_para_ply(ifc_path, ply_path):
    from ifcopenshell.util.placement import get_local_placement

    ifc = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()

    verts_all, faces_all, cores_all = [], [], []
    n_el = 0
    for el in ifc.by_type("IfcProduct"):
        if not el.Representation:
            continue
        if el.is_a("IfcOpeningElement"):
            continue  # vaos sao subtracao, nao elemento visivel
        try:
            shape = ifcopenshell.geom.create_shape(settings, el)
        except Exception:
            continue
        v = np.array(shape.geometry.verts).reshape(-1, 3)
        fc = np.array(shape.geometry.faces).reshape(-1, 3)
        # Aplica o placement do objeto MANUALMENTE (nao confia no
        # use-world-coords do iterator, que varia entre versoes).
        M = get_local_placement(el.ObjectPlacement)
        v = (v @ M[:3, :3].T) + M[:3, 3]
        cor = CORES.get(el.is_a(), (160, 160, 160))
        base = len(verts_all)
        verts_all.extend(v.tolist())
        faces_all.extend((fc + base).tolist())
        cores_all.extend([cor] * len(v))
        n_el += 1
        if n_el <= 3 or el.is_a() == "IfcSlab":
            mn, mx = v.min(axis=0), v.max(axis=0)
            print(f"        {el.Name:<12} bbox x[{mn[0]:.1f},{mx[0]:.1f}] "
                  f"y[{mn[1]:.1f},{mx[1]:.1f}] z[{mn[2]:.1f},{mx[2]:.1f}]")

    with open(ply_path, "w") as fp:
        fp.write("ply\nformat ascii 1.0\n")
        fp.write(f"element vertex {len(verts_all)}\n")
        fp.write("property float x\nproperty float y\nproperty float z\n")
        fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fp.write(f"element face {len(faces_all)}\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(verts_all, cores_all):
            fp.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
        for a, b, c in faces_all:
            fp.write(f"3 {a} {b} {c}\n")
    return n_el, len(faces_all)


# ============================================================
# 6) MODELO EDITAVEL (parse -> dict -> IFC)  [usado pelo editor no front]
# ============================================================
def contorno_laje(paredes):
    """Contorno (hull convexo) das paredes externas -> polígono da laje. Ponto de
    partida editavel no front (o usuario ajusta pra forma real da planta)."""
    ext_pts = [pt for p in paredes if p["layer"].lower().startswith("wall-ext")
               for pt in p["eixo"]]
    if not ext_pts:
        ext_pts = [pt for p in paredes for pt in p["eixo"]]
    return hull_convexo(np.array(ext_pts)) if ext_pts else []


def parse_dxf(
    dxf_path,
    escala_forcada=None,
    esp_default=0.15,
    layer_map=None,
    cad_region=None,
    linked_image=None,
):
    """Le o DXF e devolve o MODELO estruturado (paredes + aberturas + laje), SEM
    gerar IFC. E a etapa de leitura do editor: o front mostra/edita esse modelo e
    so depois pede a geracao do IFC."""
    from cad_detection_v2 import parse_dxf_v2

    return parse_dxf_v2(
        dxf_path,
        escala_forcada=escala_forcada,
        esp_default=esp_default,
        layer_map=layer_map,
        cad_region=cad_region,
        linked_image=linked_image,
    )


def referencia_vetorial(
    segmentos,
    *,
    crop_bbox=None,
    label="Planta original",
    max_segments=40000,
):
    """Serializa a geometria-fonte como underlay leve e alinhado ao editor.

    A referencia nunca participa da modelagem/IFC: ela preserva as linhas que
    deram origem ao reconhecimento para que o usuario compare e corrija o BIM.
    ``crop_bbox`` evita que outras plantas/pavimentos distantes do CAD deixem a
    referencia atual minuscule no editor.
    """
    normalized = []
    crop = None
    if crop_bbox:
        try:
            xmin = float(crop_bbox["xmin"])
            ymin = float(crop_bbox["ymin"])
            xmax = float(crop_bbox["xmax"])
            ymax = float(crop_bbox["ymax"])
            span = max(xmax - xmin, ymax - ymin, 0.1)
            pad = max(0.50, span * 0.08)
            crop = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)
        except (KeyError, TypeError, ValueError):
            crop = None

    for segment in segmentos or []:
        if len(segment) < 2:
            continue
        a, b = np.asarray(segment[0], dtype=float), np.asarray(segment[1], dtype=float)
        if a.size < 2 or b.size < 2:
            continue
        x1, y1 = float(a[0]), float(a[1])
        x2, y2 = float(b[0]), float(b[1])
        if not np.isfinite([x1, y1, x2, y2]).all():
            continue
        if math.hypot(x2 - x1, y2 - y1) < 1e-5:
            continue
        if crop is not None:
            cx0, cy0, cx1, cy1 = crop
            if max(x1, x2) < cx0 or min(x1, x2) > cx1:
                continue
            if max(y1, y2) < cy0 or min(y1, y2) > cy1:
                continue
        normalized.append([x1, y1, x2, y2])

    source_count = len(normalized)
    limit = max(1, int(max_segments))
    if source_count > limit:
        # Amostragem uniforme preserva todas as regioes/layers melhor que reter
        # somente as linhas mais longas (que apagaria arcos e esquadrias).
        indexes = np.linspace(0, source_count - 1, limit, dtype=np.int64)
        normalized = [normalized[int(index)] for index in indexes]
    if not normalized:
        return None

    xs = [value for segment in normalized for value in (segment[0], segment[2])]
    ys = [value for segment in normalized for value in (segment[1], segment[3])]
    return {
        "kind": "vector",
        "label": str(label),
        "bounds": [
            round(min(xs), 6), round(min(ys), 6),
            round(max(xs), 6), round(max(ys), 6),
        ],
        "segments": [
            [round(value, 6) for value in segment]
            for segment in normalized
        ],
        "source_count": source_count,
        "truncated": source_count > len(normalized),
    }


def modelo_para_dict(modelo):
    """Modelo interno -> dict JSON-serializavel (ids + coords absolutas em m)."""
    paredes = []
    for i, p in enumerate(modelo["paredes"]):
        A, B = p["eixo"]
        parede = {
            "id": p.get("id") or f"w{i}",
            "ax": round(float(A[0]), 4), "ay": round(float(A[1]), 4),
            "bx": round(float(B[0]), 4), "by": round(float(B[1]), 4),
            "espessura": round(float(p["espessura"]), 4),
            "layer": p.get("layer", ""),
        }
        for campo in ("nome", "guid", "tipo", "ifc_class", "nivel", "origem"):
            if p.get(campo) is not None:
                parede[campo] = p[campo]
        for campo in ("altura", "elevacao", "confidence"):
            if p.get(campo) is not None:
                parede[campo] = round(float(p[campo]), 4)
        paredes.append(parede)
    aberturas = []
    for j, ab in enumerate(modelo["aberturas"]):
        abertura = {
            "id": ab.get("id") or f"o{j}",
            "parede_id": paredes[ab["parede_idx"]]["id"],
            "tipo": ab["tipo"],
            "s_centro": round(float(ab["s_centro"]), 4),
            "largura": round(float(ab["largura"]), 4),
        }
        for campo in (
            "nome", "guid", "origem", "source_layer", "block_name",
            "source_text", "semantic_subtype", "semantic_reason",
        ):
            if ab.get(campo) is not None:
                abertura[campo] = ab[campo]
        for campo in (
            "altura", "peitoril", "confidence",
            "declared_width", "declared_height",
        ):
            if ab.get(campo) is not None:
                abertura[campo] = round(float(ab[campo]), 4)
        aberturas.append(abertura)
    xs = [c for p in paredes for c in (p["ax"], p["bx"])]
    ys = [c for p in paredes for c in (p["ay"], p["by"])]
    bbox = ({"xmin": min(xs), "ymin": min(ys), "xmax": max(xs), "ymax": max(ys)}
            if xs else {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1})
    contorno = [[round(float(p[0]), 4), round(float(p[1]), 4)]
                for p in modelo.get("laje_contorno", [])]
    faces = modelo.get("laje_faces", {})
    laje = {"contorno": contorno,
            "piso": dict(faces.get("piso", {"ativo": True, "espessura": 0.12})),
            "teto": dict(faces.get("teto", {"ativo": True, "espessura": 0.12}))}
    return {
        "escala": modelo.get("escala"),
        "single_line": bool(modelo.get("single_line")),
        "source": modelo.get("source"),
        "reference": modelo.get("reference"),
        "warnings": list(modelo.get("warnings", [])),
        "diagnostico": {
            "sobras": modelo.get("n_sobras", 0),
            "cantos_costurados": modelo.get("n_cantos", 0),
            "blocos_esquadria": modelo.get("n_blocos_esq", 0),
            "elementos_lidos": modelo.get("n_elementos",
                                          len(paredes) + len(aberturas)),
            "geometrias_aproximadas": modelo.get("n_aproximados", 0),
        },
        "bbox": bbox,
        "paredes": paredes,
        "aberturas": aberturas,
        "laje": laje,
    }


def dict_para_modelo(d):
    """dict (possivelmente editado no front) -> formato interno pro construir_ifc.
    Ignora paredes/aberturas degeneradas ou orfas."""
    id2idx, paredes = {}, []
    for w in d.get("paredes", []):
        A = np.array([float(w["ax"]), float(w["ay"])], dtype=float)
        B = np.array([float(w["bx"]), float(w["by"])], dtype=float)
        arco = None
        if w.get("geometria") == "arco" and isinstance(w.get("curva"), dict):
            C = np.array([
                float(w["curva"]["x"]),
                float(w["curva"]["y"]),
            ], dtype=float)
            arco = _arco_tres_pontos(A, C, B)
        L = float(arco["comprimento"] if arco else np.linalg.norm(B - A))
        if L < 1e-3:
            continue  # parede degenerada
        id2idx[w["id"]] = len(paredes)
        parede = {
            "eixo": (A, B),
            "espessura": float(w["espessura"]),
            "comprimento": L,
            "layer": w.get("layer", ""),
            "nome": w.get("nome") or str(w["id"]),
        }
        if arco:
            parede["geometria"] = "arco"
            parede["curva"] = C
            parede["arco"] = arco
        for campo in ("nome", "guid", "tipo", "ifc_class", "nivel", "origem"):
            if w.get(campo) is not None:
                parede[campo] = w[campo]
        for campo in (
            "altura", "altura_observada", "elevacao", "confidence",
        ):
            if w.get(campo) is not None:
                parede[campo] = float(w[campo])
        paredes.append(parede)
    aberturas = []
    for o in d.get("aberturas", []):
        if o.get("parede_id") not in id2idx:
            continue  # abertura orfa (parede foi apagada)
        parent = paredes[id2idx[o["parede_id"]]]
        if parent.get("ifc_class") == "IfcColumn" or parent.get("tipo") == "column":
            continue  # pilares não hospedam portas ou janelas
        abertura = {
            "parede_idx": id2idx[o["parede_id"]],
            "tipo": o["tipo"],
            "s_centro": float(o["s_centro"]),
            "largura": float(o["largura"]),
            "nome": o.get("nome") or str(o["id"]),
        }
        for campo in (
            "nome", "guid", "origem", "source_layer", "block_name",
            "source_text", "semantic_subtype", "semantic_reason",
        ):
            if o.get(campo) is not None:
                abertura[campo] = o[campo]
        for campo in (
            "altura", "peitoril", "confidence",
            "declared_width", "declared_height",
        ):
            if o.get(campo) is not None:
                abertura[campo] = float(o[campo])
        aberturas.append(abertura)
    spaces = []
    for index, space in enumerate(d.get("spaces", []), 1):
        contorno = [
            [float(value[0]), float(value[1])]
            for value in space.get("contorno", [])
        ]
        if len(contorno) < 3:
            continue
        spaces.append({
            "id": str(space.get("id") or f"SPACE-{index:03d}"),
            "nome": space.get("nome"),
            "contorno": contorno,
            "area": (
                float(space["area"])
                if space.get("area") is not None
                else None
            ),
        })
    laje = d.get("laje")
    return {
        "paredes": paredes,
        "aberturas": aberturas,
        "laje": laje,
        "spaces": spaces,
    }


def gerar_ifc_do_modelo(
    paredes,
    aberturas,
    out_path,
    config=None,
    laje=None,
    spaces=None,
):
    """paredes+aberturas (formato interno) -> IFC no out_path. `laje` (opcional):
    {contorno:[[x,y]...], piso:{ativo,espessura}, teto:{ativo,espessura}}. Sem ela,
    recalcula o contorno pelo hull e usa piso+teto default."""
    config = config or {}
    # contorno: usa o editado no front, senao recalcula
    if laje and laje.get("contorno"):
        laje_poly = [np.array([float(p[0]), float(p[1])]) for p in laje["contorno"]]
    else:
        laje_poly = contorno_laje(paredes)

    piso = (laje or {}).get("piso", {})
    teto = (laje or {}).get("teto", {})
    piso_ativo = piso.get("ativo", True)
    teto_ativo = teto.get("ativo", config.get("cobertura", True))

    return construir_ifc(
        paredes, laje_poly,
        config.get("altura", 2.80), config.get("esp_laje", 0.12),
        config.get("pavimento", "Terreo"), config.get("projeto", "Planta2BIM"),
        out_path,
        aberturas=aberturas, cobertura=teto_ativo,
        porta_altura=config.get("porta_altura", 2.10),
        janela_altura=config.get("janela_altura", 1.20),
        janela_peitoril=config.get("janela_peitoril", 1.00),
        esquadria_detalhada=config.get("esquadria_detalhada", False),
        piso_ativo=piso_ativo,
        piso_esp=piso.get("espessura"), teto_esp=teto.get("espessura"),
        spaces=spaces,
        forro=config.get("forro"),
        esquadria_sobreposicao=config.get("esquadria_sobreposicao", 0.02))


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Planta DXF -> IFC (v1)")
    ap.add_argument("dxf")
    ap.add_argument("--altura", type=float, default=2.80, help="pe-direito (m)")
    ap.add_argument("--esp_laje", type=float, default=0.12, help="espessura da laje (m)")
    ap.add_argument("--esp_default", type=float, default=0.15,
                    help="espessura de parede no modo single-line (default 0.15m)")
    ap.add_argument("--pavimento", default="Terreo")
    ap.add_argument("--projeto", default="Planta2BIM")
    ap.add_argument("--sem_cobertura", action="store_true",
                    help="nao gerar a laje de cobertura (teto)")
    ap.add_argument("--porta_altura", type=float, default=2.10)
    ap.add_argument("--janela_altura", type=float, default=1.20)
    ap.add_argument("--janela_peitoril", type=float, default=1.00)
    ap.add_argument("--esquadria_detalhada", action="store_true",
                    help="desenha porta (batente+folha) e janela (moldura+vidro) "
                         "em vez do bloco solido simples")
    ap.add_argument("--escala", type=float, default=None,
                    help="fator de escala unidade->metro (forca; ex 0.001 mm, "
                         "0.0001 se a heuristica errar). default: auto pela extensao")
    args = ap.parse_args()

    dxf_path = Path(args.dxf)
    ifc_path = dxf_path.with_suffix(".ifc")
    ply_path = dxf_path.with_name(dxf_path.stem + "_ifc_preview.ply")

    print(f"[1/4] Lendo e interpretando {dxf_path.name} ...")
    modelo = parse_dxf(dxf_path, escala_forcada=args.escala, esp_default=args.esp_default)
    paredes, aberturas = modelo["paredes"], modelo["aberturas"]
    print(f"      escala {modelo['escala']}  "
          f"{'SINGLE-LINE' if modelo['single_line'] else 'parede dupla'}")
    esps = sorted({round(p['espessura'], 3) for p in paredes})
    print(f"[2/4] {len(paredes)} paredes | espessuras: {esps} | "
          f"cantos costurados: {modelo['n_cantos']}")
    n_portas = sum(1 for a in aberturas if a["tipo"] == "door")
    n_jan = sum(1 for a in aberturas if a["tipo"] == "window")
    print(f"[3/4] Esquadrias: {modelo['n_blocos_esq']} blocos -> "
          f"{n_portas} portas + {n_jan} janelas")

    config = {"altura": args.altura, "esp_laje": args.esp_laje,
              "pavimento": args.pavimento, "projeto": args.projeto,
              "cobertura": not args.sem_cobertura,
              "porta_altura": args.porta_altura, "janela_altura": args.janela_altura,
              "janela_peitoril": args.janela_peitoril,
              "esquadria_detalhada": args.esquadria_detalhada}
    gerar_ifc_do_modelo(paredes, aberturas, ifc_path, config)
    print(f"      Salvo: {ifc_path}")

    print("[4/4] Preview do IFC -> PLY ...")
    n_el, n_tri = ifc_para_ply(ifc_path, ply_path)
    print(f"      {n_el} elementos malhados, {n_tri} triangulos")
    print(f"      Salvo: {ply_path}")

    print("\nPRONTO.")
    print(f"  IFC:     {ifc_path}")
    print(f"  Preview: {ply_path}  (abre no CloudCompare)")


if __name__ == "__main__":
    main()
