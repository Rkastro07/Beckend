"""Standalone: Sonata + features semanticas por OBB do IFC.

Lê PLY + IFC, roda Sonata 1x, extrai pra cada elemento IFC:
  - quantos pontos caem dentro da OBB
  - distribuicao de classes ScanNet
  - verdict simples (construido / nao_construido / divergencia)

Saida: JSON consumivel pelo frontend via endpoint cached do app_obb.

Uso:
  python bbox_features.py <scan.ply> <model.ifc> [output.json] [pavimento]
"""
import sys
import time
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import open3d as o3d

import ifcopenshell
import ifcopenshell.geom

import sonata
from alinhamento_simples import alinhar_nuvem_com_ifc


# =====================================================================
# CONFIG
# =====================================================================
TIPOS_INTERESSE = (
    "IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcStair", "IfcRoof",
    "IfcSanitaryTerminal", "IfcDoor", "IfcWindow",
    "IfcCovering", "IfcPlate", "IfcMember", "IfcRailing",
)

CLASS_NAMES = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "desk",
    "curtain", "refrigerator", "shower_curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
)
NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# Mapeamento IfcType -> classes ScanNet aceitas como "este elemento existe"
# Permissivo pra tipos que ScanNet nao cobre (viga/pilar/escada): aceita
# qualquer estrutura.
ESTRUTURAIS = {"wall", "floor", "otherfurniture"}
TIPO_TO_CLASSES_ACEITAS = {
    "IfcWall":    {"wall"},
    "IfcRailing": {"wall", "otherfurniture"},
    "IfcSlab":    {"floor"},  # pode ser piso ou teto; ambos OK
    "IfcCovering": {"floor", "otherfurniture"},
    "IfcPlate":   {"floor", "wall"},
    "IfcRoof":    {"floor"},  # ScanNet nao tem ceiling, slab cobre
    "IfcDoor":    {"door", "wall"},
    "IfcWindow":  {"window", "wall"},
    "IfcStair":   ESTRUTURAIS,
    "IfcBeam":    ESTRUTURAIS,        # ScanNet nao tem viga
    "IfcMember":  ESTRUTURAIS,
    "IfcColumn":  ESTRUTURAIS,        # ScanNet nao tem pilar
    "IfcSanitaryTerminal": {"toilet", "sink", "bathtub", "otherfurniture"},
}

# Threshold de pontos por m3 dentro da OBB (abaixo disso → nao construido)
DENSIDADE_MIN_PTS_M3 = 20
N_PTS_MIN_ABS = 15

# Cores ScanNet (RGB 0-1)
CLASS_COLORS = {
    "wall":           (0.40, 0.50, 0.95),  # azul
    "floor":          (0.30, 0.85, 0.40),  # verde
    "ceiling":        (0.95, 0.55, 0.85),  # rosa
    "door":           (0.95, 0.65, 0.20),  # laranja
    "window":         (0.40, 0.90, 0.95),  # ciano
    "otherfurniture": (0.75, 0.75, 0.40),  # amarelo escuro
}
DEFAULT_CLASS_COLOR = (0.55, 0.55, 0.55)  # cinza

# Cores verdict
VERDICT_COLORS = {
    "construido":     (0.10, 0.85, 0.20),  # verde
    "divergencia":    (0.95, 0.85, 0.10),  # amarelo
    "nao_construido": (0.95, 0.15, 0.10),  # vermelho
}


# =====================================================================
# Sonata
# =====================================================================
class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
    def forward(self, x):
        return self.seg_head(x)


def carregar_sonata():
    print("[SONATA] carregando modelo + head ScanNet...")
    t0 = time.time()
    custom_config = dict(
        enc_patch_size=[256 for _ in range(5)],
        enable_flash=False,
    )
    model = sonata.load("sonata", repo_id="facebook/sonata",
                        custom_config=custom_config,
                        download_root="./checkpoints").cuda().eval()
    ckpt = sonata.load("sonata_linear_prob_head_sc", repo_id="facebook/sonata",
                       ckpt_only=True, download_root="./checkpoints")
    head = SegHead(**ckpt["config"]).cuda().eval()
    head.load_state_dict(ckpt["state_dict"])
    print(f"[SONATA] pronto em {time.time()-t0:.1f}s")
    return model, head


def carregar_ply(ply_path):
    """Le PLY, retorna pts_original (densidade total)."""
    print(f"[PLY] lendo {Path(ply_path).name}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points).astype(np.float32)
    print(f"[PLY] {len(pts):,} pts")
    return pts


def rodar_sonata(model, head, pts_original, fsize_mb_hint=None):
    """Roda Sonata em pts_original (ja em Z-up). Voxeliza, infere, KNN propaga.

    Retorna: (classes_full, sonata_time). classes_full tem mesma len de pts_original.
    """
    # Voxel adaptativo
    n = len(pts_original)
    if fsize_mb_hint and fsize_mb_hint > 500:
        voxel = 0.30
    elif n > 5_000_000 or (fsize_mb_hint and fsize_mb_hint > 100):
        voxel = 0.20
    else:
        voxel = 0.15
    print(f"[SONATA] voxel {voxel}m")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_original)
    pcd_d = pcd.voxel_down_sample(voxel_size=voxel)
    pcd_d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.3, max_nn=30))

    pts = np.asarray(pcd_d.points).astype(np.float32)
    normals = np.asarray(pcd_d.normals).astype(np.float32)
    cols = (np.asarray(pcd_d.colors) if len(pcd_d.colors) > 0
            else np.ones_like(pts) * 0.5).astype(np.float32)
    print(f"[SONATA] sparse pra Sonata: {len(pts):,} pts")

    # Sonata transform
    point = {
        "coord": pts.copy(), "color": cols, "normal": normals,
        "segment": np.zeros(len(pts), dtype=np.int64),
    }
    point = sonata.transform.default()(point)

    inverse_to_sparse = None
    if "inverse" in point:
        inv = point["inverse"]
        inverse_to_sparse = inv.cpu().numpy() if isinstance(inv, torch.Tensor) else np.asarray(inv)

    print("[SONATA] inferencia...")
    t0 = time.time()
    with torch.inference_mode(), torch.cuda.amp.autocast():
        for k in point:
            if isinstance(point[k], torch.Tensor):
                point[k] = point[k].cuda(non_blocking=True)
        out = model(point)
        while "pooling_parent" in out.keys():
            parent = out.pop("pooling_parent")
            inv = out.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, out.feat[inv]], dim=-1)
            out = parent
        pred = head(out.feat).argmax(dim=-1).cpu().numpy()
    torch.cuda.synchronize()
    sonata_time = time.time() - t0
    print(f"[SONATA] inferencia em {sonata_time:.1f}s, {len(pred):,} pontos")

    # KNN propaga classes pra densidade total
    print("[KNN] propagando classes pra densidade total...")
    from scipy.spatial import cKDTree
    if inverse_to_sparse is not None and len(inverse_to_sparse) == len(pts):
        sparse_classes = pred[inverse_to_sparse]
    else:
        n = min(len(pts), len(pred))
        pts = pts[:n]
        sparse_classes = pred[:n]
    tree = cKDTree(pts)
    _, idx_nn = tree.query(pts_original, k=1, workers=-1)
    classes_full = sparse_classes[idx_nn]

    # Stats
    print("[SONATA] distribuicao na densidade total:")
    for c, n in sorted(Counter(classes_full).items(), key=lambda x: -x[1]):
        nm = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"id_{c}"
        print(f"   {nm:18s} {n:>10,} ({100*n/len(classes_full):5.1f}%)")

    return classes_full, sonata_time


# =====================================================================
# IFC -> OBB
# =====================================================================
def extrair_obbs_ifc(ifc_path, pavimento_alvo=None):
    """Extrai OBB de cada IfcProduct dos TIPOS_INTERESSE.

    Retorna: lista de dicts {guid, tipo, corners (8x3), center, extent, R}.
    Coords em IFC nativo (Z-up).
    """
    print(f"[IFC] abrindo {Path(ifc_path).name}...")
    f = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    def _norm(s):
        if not s:
            return ""
        s = s.strip()
        if s.upper().startswith("OSSO "):
            s = s[5:].strip()
        return s.upper()

    alvo = _norm(pavimento_alvo) if pavimento_alvo else None
    todos_pavs = alvo is None or alvo == "__TODOS__"

    elementos = []
    n_skip_geom = 0
    n_skip_pav = 0

    for product in f.by_type("IfcProduct"):
        tipo = next((t for t in TIPOS_INTERESSE if product.is_a(t)), None)
        if tipo is None:
            continue

        if not todos_pavs:
            pav = None
            for rel in getattr(product, "ContainedInStructure", []):
                if rel.RelatingStructure.is_a("IfcBuildingStorey"):
                    pav = rel.RelatingStructure.Name
                    break
            if _norm(pav) != alvo:
                n_skip_pav += 1
                continue

        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            verts = np.array(shape.geometry.verts).reshape(-1, 3)
            if verts.size == 0:
                n_skip_geom += 1
                continue
        except Exception:
            n_skip_geom += 1
            continue

        # OBB via Open3D fit
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        try:
            obb = pcd.get_oriented_bounding_box()
        except RuntimeError:
            n_skip_geom += 1
            continue

        corners = np.asarray(obb.get_box_points())  # (8, 3)
        # AABB world-space (usado pelo alinhamento)
        aabb_min = verts.min(axis=0)
        aabb_max = verts.max(axis=0)
        # Subsample dos verts (alvo de ICP semantico, max 200 pts por elemento)
        n_keep = min(200, len(verts))
        if len(verts) > n_keep:
            idx_sub = np.linspace(0, len(verts)-1, n_keep).astype(np.int64)
            verts_sub = verts[idx_sub]
        else:
            verts_sub = verts
        elementos.append({
            "guid": product.GlobalId,
            "tipo": tipo,
            "name": product.Name or "",
            "center": np.asarray(obb.center).tolist(),
            "extent": np.asarray(obb.extent).tolist(),
            "R": np.asarray(obb.R).tolist(),
            "corners": corners.tolist(),
            "bbox": {
                "xmin": float(aabb_min[0]), "xmax": float(aabb_max[0]),
                "ymin": float(aabb_min[1]), "ymax": float(aabb_max[1]),
                "zmin": float(aabb_min[2]), "zmax": float(aabb_max[2]),
            },
            "_verts_sub": verts_sub.astype(np.float32),  # interno (nao vai pro JSON)
        })

    print(f"[IFC] {len(elementos)} elementos extraidos "
          f"(skip geom={n_skip_geom}, skip pav={n_skip_pav})")
    return elementos


# =====================================================================
# ALINHAMENTO SEMANTICO (opcional)
# =====================================================================
# Mapeamento tipo IFC -> grupo semantico (qual T aplicar no momento do verdict)
# Z-relativo decide pra IfcSlab/Covering/Plate (alto = ceiling, baixo = floor)
TIPO_TO_SEMANTIC = {
    "IfcWall":    "wall",
    "IfcRailing": "wall",
    "IfcColumn":  "wall",
    "IfcBeam":    "wall",
    "IfcMember":  "wall",
    "IfcDoor":    "wall",
    "IfcWindow":  "wall",
    "IfcRoof":    "ceiling",
    "IfcStair":   "floor",
    # Decididos por Z relativo no momento da execucao:
    "IfcSlab":     None,
    "IfcCovering": None,
    "IfcPlate":    None,
    "IfcSanitaryTerminal": "floor",
}


def _tipo_to_class(tipo, z_center, z_mid):
    """Decide grupo semantico (wall/floor/ceiling) pra um elemento IFC."""
    g = TIPO_TO_SEMANTIC.get(tipo)
    if g is not None:
        return g
    # Tipos finos horizontais (slab/covering/plate): Z decide
    return "ceiling" if z_center > z_mid else "floor"


def _run_icp(source_pts, target_pts, voxel=0.10, max_iter=50):
    """ICP point-to-point. Retorna T (4x4) e fitness.

    Source/target ja devem estar grosseiramente alinhados (mesmo sistema).
    Voxel pra baixar densidade e estabilizar.
    """
    if len(source_pts) < 50 or len(target_pts) < 50:
        return np.eye(4), 0.0
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source_pts.astype(np.float64))
    src = src.voxel_down_sample(voxel)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target_pts.astype(np.float64))
    tgt = tgt.voxel_down_sample(voxel)
    if len(src.points) < 30 or len(tgt.points) < 30:
        return np.eye(4), 0.0
    # Threshold = ~3x voxel (max distancia de correspondencia)
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, max_correspondence_distance=voxel * 3.0,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    return np.asarray(result.transformation), float(result.fitness)


def alinhamento_semantico(pts_aligned, classes, elementos, max_rotation_deg=15.0):
    """Calcula T por classe semantica via ICP scan_class <-> ifc_class.

    Pra cada classe (wall/floor/ceiling):
      1. Subset do scan: pts com classe Sonata correspondente
      2. Subset do IFC:  verts dos elementos cujo tipo mapeia pra essa classe
      3. ICP -> T
      4. Sanity check: se rotacao > max_rotation_deg, rejeita (volta pra I)

    Retorna dict {"wall": T_4x4, "floor": T_4x4, "ceiling": T_4x4}.
    """
    print("[SEM-ALIGN] particionando pts e calculando T por classe...")
    t0 = time.time()

    wall_id   = NAME_TO_ID["wall"]
    floor_id  = NAME_TO_ID["floor"]

    z_min, z_max = pts_aligned[:, 2].min(), pts_aligned[:, 2].max()
    z_mid = (z_min + z_max) / 2.0

    # Particiona SCAN
    mask_wall  = classes == wall_id
    mask_floor_all = classes == floor_id
    # ScanNet nao tem "ceiling" — heuristica: floor com Z alto
    mask_ceiling   = mask_floor_all & (pts_aligned[:, 2] > z_mid)
    mask_floor     = mask_floor_all & (pts_aligned[:, 2] <= z_mid)

    scan_subsets = {
        "wall":    pts_aligned[mask_wall],
        "floor":   pts_aligned[mask_floor],
        "ceiling": pts_aligned[mask_ceiling],
    }

    # Particiona IFC (concat verts dos elementos por grupo)
    ifc_subsets = {"wall": [], "floor": [], "ceiling": []}
    for el in elementos:
        v = el.get("_verts_sub")
        if v is None or len(v) == 0:
            continue
        zc = float(el["center"][2])
        grp = _tipo_to_class(el["tipo"], zc, z_mid)
        ifc_subsets.setdefault(grp, []).append(v)
    for k in list(ifc_subsets.keys()):
        ifc_subsets[k] = np.vstack(ifc_subsets[k]) if ifc_subsets[k] else np.empty((0, 3))

    # ICP por classe
    T_map = {}
    for cls in ("wall", "floor", "ceiling"):
        src = scan_subsets[cls]
        tgt = ifc_subsets[cls]
        n_src, n_tgt = len(src), len(tgt)
        if n_src < 100 or n_tgt < 100:
            print(f"[SEM-ALIGN] {cls:8s}: skip (scan={n_src}, ifc={n_tgt})")
            T_map[cls] = np.eye(4)
            continue
        T, fitness = _run_icp(src, tgt)
        # Sanity: rotacao pequena (ja viemos do rough align)
        R = T[:3, :3]
        # angulo = arccos((trace(R)-1)/2)
        tr = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
        ang_deg = float(np.degrees(np.arccos(tr)))
        translation_norm = float(np.linalg.norm(T[:3, 3]))
        if ang_deg > max_rotation_deg:
            print(f"[SEM-ALIGN] {cls:8s}: REJEITADO  rot={ang_deg:.1f}°  "
                  f"trans={translation_norm:.2f}m  fitness={fitness:.3f}  "
                  f"(scan={n_src}, ifc={n_tgt})")
            T_map[cls] = np.eye(4)
        else:
            print(f"[SEM-ALIGN] {cls:8s}: T  rot={ang_deg:.2f}°  "
                  f"trans={translation_norm:.3f}m  fitness={fitness:.3f}  "
                  f"(scan={n_src}, ifc={n_tgt})")
            T_map[cls] = T

    T_map["_z_mid"] = z_mid
    print(f"[SEM-ALIGN] feito em {time.time()-t0:.1f}s")
    return T_map


def _apply_T(pts, T):
    """Aplica T 4x4 a Nx3 pts."""
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (h @ T.T)[:, :3]


# =====================================================================
# Features por OBB + verdict
# =====================================================================
def features_por_obb(elementos, pts, classes, T_map=None):
    """Pra cada elemento, conta pontos+classes dentro da OBB.

    Se T_map for fornecido (alinhamento semantico), aplica T_map[grupo] aos
    pts antes de filtrar pela OBB do elemento. Cada elemento usa seu T
    correspondente baseado no tipo (wall/floor/ceiling).
    """
    print(f"[FEATURES] processando {len(elementos)} elementos"
          f"{' [SEM-ALIGN ON]' if T_map else ''}...")
    t0 = time.time()
    pcd_full_default = o3d.geometry.PointCloud()
    pcd_full_default.points = o3d.utility.Vector3dVector(pts)

    z_mid = T_map.get("_z_mid") if T_map else None

    # Cache de nuvens transformadas por classe (evita recomputar pra cada elemento)
    pcd_cache = {"identity": pcd_full_default}
    if T_map:
        for cls in ("wall", "floor", "ceiling"):
            T = T_map.get(cls, np.eye(4))
            if np.allclose(T, np.eye(4)):
                pcd_cache[cls] = pcd_full_default
            else:
                pts_t = _apply_T(pts, T)
                p = o3d.geometry.PointCloud()
                p.points = o3d.utility.Vector3dVector(pts_t)
                pcd_cache[cls] = p

    out = []
    for el in elementos:
        # Seleciona PCD correta baseado no tipo
        if T_map:
            zc = float(el["center"][2])
            grp = _tipo_to_class(el["tipo"], zc, z_mid)
            pcd_full = pcd_cache.get(grp, pcd_full_default)
        else:
            pcd_full = pcd_full_default

        obb = o3d.geometry.OrientedBoundingBox(
            center=np.array(el["center"]),
            R=np.array(el["R"]),
            extent=np.array(el["extent"]),
        )
        idx = obb.get_point_indices_within_bounding_box(pcd_full.points)
        if len(idx) == 0:
            cls_dist = {}
            dom = None
            n_pts = 0
        else:
            cls_arr = classes[np.array(idx, dtype=np.int64)]
            n_pts = len(cls_arr)
            ctr = Counter(cls_arr.tolist())
            cls_dist = {CLASS_NAMES[c]: round(n / n_pts, 3) for c, n in ctr.items()}
            dom_id = ctr.most_common(1)[0][0]
            dom = CLASS_NAMES[dom_id]

        # verdict simples
        vol = max(0.001, np.prod(el["extent"]))
        n_min = max(N_PTS_MIN_ABS, int(vol * DENSIDADE_MIN_PTS_M3 * 0.1))
        # 0.1 = fator empirico — bbox tem volume cheio mas pts so na superficie
        aceitas = TIPO_TO_CLASSES_ACEITAS.get(el["tipo"], ESTRUTURAIS)
        if n_pts < n_min:
            verdict = "nao_construido"
        elif dom in aceitas:
            verdict = "construido"
        else:
            # aceita se SOMA das classes aceitas > 30%
            soma_aceitas = sum(cls_dist.get(c, 0) for c in aceitas)
            verdict = "construido" if soma_aceitas > 0.30 else "divergencia"

        out.append({
            **el,
            "n_pts_dentro": n_pts,
            "n_pts_min": n_min,
            "vol_obb_m3": round(vol, 3),
            "pct_classes": cls_dist,
            "classe_dominante": dom,
            "classes_aceitas": sorted(aceitas),
            "verdict_simples": verdict,
        })
    print(f"[FEATURES] feito em {time.time()-t0:.1f}s")
    return out


# =====================================================================
# Visualizacao
# =====================================================================
def _obb_to_mesh(center, R, extent, color):
    dx, dy, dz = extent
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=float(dx), height=float(dy), depth=float(dz))
    mesh.translate(-mesh.get_center())
    mesh.rotate(np.array(R), center=(0, 0, 0))
    mesh.translate(np.array(center))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def salvar_jsons_front(pts_aligned, classes, resultado, out_dir, session_subdir,
                       max_global_pts=150_000, max_per_obj_pts=5000):
    """Gera JSONs consumidos pelo front v2 (Three.js).

    - _global.json: {positions: [...]}  nuvem inteira subsample
    - {guid}.json:  {positions, colors} pontos dentro da OBB (cor por classe Sonata)

    pts e cores ja convertidos pra Three.js (Y-up, flip X).
    Adiciona campo 'json_file' a cada elemento de `resultado` (mutacao).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # IFC (Z-up) -> Three.js (Y-up + flip X)
    def to_threejs(pts):
        tj = np.empty_like(pts)
        tj[:, 0] = -pts[:, 0]
        tj[:, 1] = pts[:, 2]
        tj[:, 2] = -pts[:, 1]
        return tj

    # 1. Global cloud subsample
    pts_tj = to_threejs(pts_aligned.astype(np.float32))
    if len(pts_tj) > max_global_pts:
        idx = np.linspace(0, len(pts_tj) - 1, max_global_pts).astype(np.int64)
        pts_sub = pts_tj[idx]
    else:
        pts_sub = pts_tj
    global_path = out_dir / "_global.json"
    with open(global_path, "w", encoding="utf-8") as f:
        json.dump({"positions": pts_sub.flatten().tolist()}, f)
    print(f"[FRONT] {global_path.name}  ({len(pts_sub):,} pts, {global_path.stat().st_size/1024:.0f} KB)")

    # 2. Per-OBB: pts+cores dentro da bbox
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(pts_aligned)
    n_per_obj = 0
    for el in resultado:
        guid = el.get("guid", "noguid")
        # Refiltro: mesmo OBB usado em features_por_obb
        obb = o3d.geometry.OrientedBoundingBox(
            center=np.array(el["center"]),
            R=np.array(el["R"]),
            extent=np.array(el["extent"]),
        )
        idx = obb.get_point_indices_within_bounding_box(pcd_full.points)
        if len(idx) == 0:
            el["json_file"] = None
            continue
        idx = np.array(idx, dtype=np.int64)
        # subsample por elemento (cap)
        if len(idx) > max_per_obj_pts:
            sel = np.linspace(0, len(idx) - 1, max_per_obj_pts).astype(np.int64)
            idx = idx[sel]
        pts_obj = to_threejs(pts_aligned[idx].astype(np.float32))
        cls_obj = classes[idx]
        cols = np.array([
            CLASS_COLORS.get(CLASS_NAMES[c] if c < len(CLASS_NAMES) else "_",
                             DEFAULT_CLASS_COLOR)
            for c in cls_obj
        ], dtype=np.float32)
        # Filename seguro (guid pode ter caracteres especiais)
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in guid)
        json_name = f"{safe}.json"
        with open(out_dir / json_name, "w", encoding="utf-8") as f:
            json.dump({
                "positions": pts_obj.flatten().tolist(),
                "colors":    cols.flatten().tolist(),
            }, f)
        el["json_file"] = f"{session_subdir}/{json_name}"
        n_per_obj += 1
    print(f"[FRONT] {n_per_obj} JSONs por elemento")
    # Path relativo que o front vai concatenar
    return {
        "global_cloud": f"{session_subdir}/_global.json",
    }


def gerar_visualizacoes(pts_aligned, classes, resultado, out_dir, stem):
    """Gera 4 outputs pra inspecao no CloudCompare:
       - PLY denso colorido por classe Sonata
       - 3 OBJs com OBBs por verdict (construido/divergencia/nao_construido)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. PLY colorido por classe Sonata
    print(f"\n[VIZ] gerando PLY colorido por classe Sonata...")
    cols = np.array([
        CLASS_COLORS.get(CLASS_NAMES[c] if c < len(CLASS_NAMES) else "_",
                         DEFAULT_CLASS_COLOR)
        for c in classes
    ])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_aligned)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    ply_out = out_dir / f"{stem}_sonata_classes.ply"
    o3d.io.write_point_cloud(str(ply_out), pcd)
    print(f"   {ply_out.name}  ({ply_out.stat().st_size/1024**2:.1f} MB)")

    # 2-4. OBBs por verdict
    for verdict, cor in VERDICT_COLORS.items():
        combined = o3d.geometry.TriangleMesh()
        n = 0
        for el in resultado:
            if el["verdict_simples"] != verdict:
                continue
            mesh = _obb_to_mesh(el["center"], el["R"], el["extent"], cor)
            combined += mesh
            n += 1
        if n == 0:
            print(f"   (nenhum elemento {verdict})")
            continue
        obj_out = out_dir / f"{stem}_obbs_{verdict}.obj"
        o3d.io.write_triangle_mesh(str(obj_out), combined)
        print(f"   {obj_out.name}  ({n} OBBs, {obj_out.stat().st_size/1024:.1f} KB)")


# =====================================================================
# Main
# =====================================================================
def main():
    # Parse: separa flags (--xxx) de args posicionais
    args_pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    use_semantic_align = "--semantic-align" in flags

    if len(args_pos) < 2:
        sys.exit("Uso: python bbox_features.py <scan.ply> <model.ifc> "
                 "[output_dir] [pavimento] [--semantic-align]")

    ply_path = args_pos[0]
    ifc_path = args_pos[1]
    out_dir = Path(args_pos[2]) if len(args_pos) > 2 else Path(ply_path).parent
    pavimento = args_pos[3] if len(args_pos) > 3 else None
    stem = Path(ply_path).stem
    out_json = out_dir / f"{stem}_sonata_bbox.json"

    print("=" * 60)
    print("SONATA bbox-features")
    print("=" * 60)
    print(f"  PLY: {ply_path}")
    print(f"  IFC: {ifc_path}")
    print(f"  OUT: {out_json}")
    print(f"  PAV: {pavimento or '__TODOS__'}")
    print(f"  SEM-ALIGN: {use_semantic_align}")
    print()

    elementos = extrair_obbs_ifc(ifc_path, pavimento)
    if not elementos:
        sys.exit("Nenhum elemento IFC encontrado.")

    # 1. Le PLY (densidade total, coords originais — pode estar Y-up)
    pts_orig = carregar_ply(ply_path)
    fsize_mb = Path(ply_path).stat().st_size / 1024 / 1024

    # 2. ALINHA PRIMEIRO: traz pra Z-up + sistema do IFC
    #    Critico: Sonata foi treinado em ScanNet (Z-up). Y-up quebra a inferencia.
    pts_aligned, transf = alinhar_nuvem_com_ifc(pts_orig, elementos)
    print(f"[ALIGN] R det={np.linalg.det(transf['R']):.3f} | "
          f"escala={transf['scale']} | t={np.round(transf['t'], 2).tolist()}")

    # 3. SONATA roda na nuvem JA ALINHADA (Z-up igual ao treino)
    model, head = carregar_sonata()
    classes, t_sonata = rodar_sonata(model, head, pts_aligned, fsize_mb_hint=fsize_mb)

    # 3b. Alinhamento semantico opcional (T por classe via ICP)
    T_map = None
    if use_semantic_align:
        T_map = alinhamento_semantico(pts_aligned, classes, elementos)

    # 4. OBB filter (com T_map por elemento se semantic-align ativo)
    resultado = features_por_obb(elementos, pts_aligned, classes, T_map=T_map)

    # Limpa _verts_sub dos elementos (interno, nao vai pro JSON)
    for el in resultado:
        el.pop("_verts_sub", None)

    # Stats globais
    verdicts = Counter(e["verdict_simples"] for e in resultado)
    print("\n[STATS] verdict por elemento:")
    for v, n in verdicts.most_common():
        print(f"   {v:18s} {n:>5}  ({100*n/len(resultado):5.1f}%)")

    print("\n[STATS] verdict por tipo IFC:")
    por_tipo = {}
    for e in resultado:
        por_tipo.setdefault(e["tipo"], Counter())[e["verdict_simples"]] += 1
    for t, ctr in sorted(por_tipo.items()):
        total = sum(ctr.values())
        constr = ctr.get("construido", 0)
        print(f"   {t:22s} total={total:>4}  construido={constr:>4} ({100*constr/total:5.1f}%)")

    saida = {
        "elementos": resultado,
        "stats_globais": {
            "n_elementos": len(resultado),
            "verdicts": dict(verdicts),
            "tempo_sonata_s": round(t_sonata, 1),
            "ply": str(ply_path),
            "ifc": str(ifc_path),
            "pavimento": pavimento,
        },
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] JSON salvo: {out_json}")
    print(f"     Tamanho: {Path(out_json).stat().st_size/1024:.1f} KB")

    # JSONs do visualizador front (nuvem global + per-OBB)
    # session_subdir = nome da pasta (front concatena com /outputs/)
    session_subdir = Path(out_dir).name
    front_paths = salvar_jsons_front(pts_aligned, classes, resultado, out_dir, session_subdir)
    saida["stats_globais"]["global_cloud"] = front_paths.get("global_cloud")

    # Re-salva JSON principal com json_file/global_cloud preenchidos
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)

    # Visualizacao CC (PLY+OBJs)
    gerar_visualizacoes(pts_aligned, classes, resultado, out_dir, stem)
    print(f"\n[VIZ] Outputs em: {out_dir}")
    print(f"      Abre os 4 arquivos juntos no CloudCompare:")
    print(f"        - {stem}_sonata_classes.ply       (nuvem colorida)")
    print(f"        - {stem}_obbs_construido.obj      (verde)")
    print(f"        - {stem}_obbs_divergencia.obj     (amarelo)")
    print(f"        - {stem}_obbs_nao_construido.obj  (vermelho)")


if __name__ == "__main__":
    main()
