"""Wrapper HELIOS++ — gera nuvem de pontos realista a partir de meshes.

Substitui `mesh.sample_points_uniformly()` por simulação física de LiDAR
(ray tracing + occlusion + noise gaussiano). Roda via subprocess do binário
HELIOS++ instalado em conda env separado (~/miniforge/envs/helios).

Pipeline:
    1. Recebe dict {guid: TriangleMesh}
    2. Exporta como OBJ multi-grupo (cada guid vira grupo separado)
    3. Gera scene.xml + survey.xml com scanners nas posições dadas
    4. Roda HELIOS++ CLI
    5. Parseia XYZ output, mapeia hitObjectId → guid
    6. Devolve (pts, guid_per_point, hit_info)

Esse módulo só roda em WSL/Linux (HELIOS++ via conda-forge).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


# ============================================================
# Config: caminho do HELIOS++ no WSL Ubuntu
# ============================================================
# Procura primeiro pelo env conda do user; fallback pra PATH global
HELIOS_BIN_CANDIDATES = [
    Path.home() / "miniforge" / "envs" / "helios" / "bin" / "helios",
    Path("/opt/miniforge/envs/helios/bin/helios"),
    Path(shutil.which("helios") or "/usr/bin/helios"),
]

DEFAULT_SCANNER = "data/scanners_tls.xml#riegl_vz400"
DEFAULT_PLATFORM = "data/platforms.xml#tripod"


def find_helios_bin() -> Path:
    """Localiza o binário helios. Raise se não achar."""
    for cand in HELIOS_BIN_CANDIDATES:
        if cand and cand.exists():
            return cand
    raise FileNotFoundError(
        f"HELIOS++ não encontrado. Procurado em:\n" +
        "\n".join(f"  {c}" for c in HELIOS_BIN_CANDIDATES) +
        "\nInstale via:  conda install -c conda-forge helios"
    )


# ============================================================
# Material/reflectance por tipo IFC
# ============================================================
# Valores aproximados baseados em literatura (Reflectance vs LiDAR)
# Real Faro retorna intensity diretamente correlacionada com reflectance
REFLECTANCE_BY_TIPO = {
    "IfcWall":            0.40,  # parede pintada
    "IfcWallStandardCase": 0.40,
    "IfcSlab":            0.35,  # concreto/laje
    "IfcRoof":            0.30,
    "IfcCovering":        0.50,  # forro (geralmente mais claro)
    "IfcColumn":          0.35,
    "IfcBeam":            0.35,
    "IfcStair":           0.40,
    "IfcDoor":            0.50,  # madeira/metal pintado
    "IfcWindow":          0.10,  # vidro (LiDAR penetra/reflete pouco)
    "IfcMember":          0.45,
    "IfcPlate":           0.45,
    "IfcRailing":         0.55,  # metal
    "default":            0.35,
}


# ============================================================
# Export 1 OBJ por mesh (cada vira uma ScenePart no HELIOS++)
# ============================================================
def export_one_obj_per_mesh(
    meshes: dict[str, o3d.geometry.TriangleMesh],
    workdir: Path,
) -> dict[str, Path]:
    """Escreve um arquivo OBJ por guid. HELIOS++ vai tratar cada um como
    ScenePart separado quando referenciados via <part> distintos no scene.xml.

    Returns: {guid: path_to_obj}
    """
    out_paths: dict[str, Path] = {}
    for guid, mesh in meshes.items():
        if mesh is None or len(mesh.vertices) == 0:
            continue
        # Sanitiza guid pro nome do arquivo
        safe = "".join(c if c.isalnum() else "_" for c in guid)[:60]
        obj_path = workdir / f"part_{safe}.obj"
        verts = np.asarray(mesh.vertices)
        tris  = np.asarray(mesh.triangles)
        lines = [f"o {safe}"]
        for v in verts:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        # Duplica faces com normal invertida (HELIOS++ faz backface culling;
        # scanner pode estar de qualquer lado da parede)
        for t in tris:
            lines.append(f"f {t[0]+1} {t[1]+1} {t[2]+1}")        # original
            lines.append(f"f {t[0]+1} {t[2]+1} {t[1]+1}")        # invertida
        obj_path.write_text("\n".join(lines))
        out_paths[guid] = obj_path
    return out_paths


# ============================================================
# Heurística "walkable": posições de scanner sobre o piso
# ============================================================
def compute_walkable_scanners(
    ifc_objects: list[dict],
    n_scanners: int = 5,
    tripod_height: float = 1.5,
    min_dist: float = 3.0,
    seed: int = 42,
) -> list[tuple[float, float, float]]:
    """Escolhe N posições de scanner sobre os pisos do prédio.

    Estratégia:
    1. Identifica IfcSlab horizontais (lajes/pisos)
    2. Pega o de menor Z (assume = piso da cena)
    3. Amostra n_scanners pts dentro do polígono do piso, respeitando min_dist
    4. Eleva pra altura do tripé (z + tripod_height)

    Args:
        ifc_objects: lista do extrair_objetos_por_pavimento ou similar
                     (cada dict precisa ter 'bbox' e 'tipo')
        n_scanners:  quantos scanners por cena
        tripod_height: altura típica do tripé do scanner (1.5m default)
        min_dist:     distância mínima entre scanners (evita duplicação)
        seed:         RNG seed pra reproducibilidade

    Returns:
        Lista de (x, y, z) com posições do scanner
    """
    rng = np.random.default_rng(seed)

    # Filtra lajes horizontais (Z extent pequeno comparado a XY extent)
    floor_candidates = []
    for obj in ifc_objects:
        bbox = obj.get("bbox") or {}
        if not bbox:
            continue
        tipo = obj.get("tipo", "")
        if tipo not in ("IfcSlab", "IfcCovering", "IfcRoof"):
            continue
        dx = bbox["xmax"] - bbox["xmin"]
        dy = bbox["ymax"] - bbox["ymin"]
        dz = bbox["zmax"] - bbox["zmin"]
        if dz > 0.5:  # muito alto, não é piso
            continue
        if dx * dy < 4.0:  # área pequena, não é piso principal
            continue
        floor_candidates.append((dx * dy, obj, bbox))

    # Ordena por área (maior primeiro)
    floor_candidates.sort(key=lambda t: t[0], reverse=True)

    if not floor_candidates:
        # Fallback: usa bbox global de TODOS objs como "piso"
        if not ifc_objects:
            return [(0.0, 0.0, tripod_height)]
        all_x = [o["bbox"]["xmin"] for o in ifc_objects if o.get("bbox")] + \
                [o["bbox"]["xmax"] for o in ifc_objects if o.get("bbox")]
        all_y = [o["bbox"]["ymin"] for o in ifc_objects if o.get("bbox")] + \
                [o["bbox"]["ymax"] for o in ifc_objects if o.get("bbox")]
        all_z = [o["bbox"]["zmin"] for o in ifc_objects if o.get("bbox")]
        if not all_x:
            return [(0.0, 0.0, tripod_height)]
        bbox_global = {
            "xmin": min(all_x), "xmax": max(all_x),
            "ymin": min(all_y), "ymax": max(all_y),
            "zmin": min(all_z), "zmax": min(all_z),
        }
        return _sample_in_bbox(bbox_global, n_scanners, tripod_height, min_dist, rng)

    # Pega o maior piso (provavelmente o do pavimento principal)
    _, _, bbox = floor_candidates[0]
    return _sample_in_bbox(bbox, n_scanners, tripod_height, min_dist, rng)


def _sample_in_bbox(
    bbox: dict, n: int, tripod_h: float, min_dist: float,
    rng: np.random.Generator, margin: float = 1.0,
) -> list[tuple[float, float, float]]:
    """Amostra N pts dentro de um bbox 2D, respeitando min_dist."""
    xmin = bbox["xmin"] + margin
    xmax = bbox["xmax"] - margin
    ymin = bbox["ymin"] + margin
    ymax = bbox["ymax"] - margin
    z_floor = bbox["zmin"]

    # Se margem comeu a área toda, relaxa
    if xmax <= xmin or ymax <= ymin:
        xmin = bbox["xmin"]; xmax = bbox["xmax"]
        ymin = bbox["ymin"]; ymax = bbox["ymax"]

    positions: list[tuple[float, float, float]] = []
    max_tries = n * 50
    tries = 0
    while len(positions) < n and tries < max_tries:
        tries += 1
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        # Checa min_dist com posições já escolhidas
        too_close = any(
            (x - px)**2 + (y - py)**2 < min_dist**2
            for px, py, _ in positions
        )
        if too_close:
            continue
        positions.append((x, y, z_floor + tripod_h))

    # Se não conseguiu N posições, relaxa min_dist
    if len(positions) < n:
        while len(positions) < n:
            x = float(rng.uniform(xmin, xmax))
            y = float(rng.uniform(ymin, ymax))
            positions.append((x, y, z_floor + tripod_h))

    return positions


# ============================================================
# XML generators (scene.xml + survey.xml)
# ============================================================
def _build_scene_xml(obj_paths: list[Path]) -> str:
    """Cena com 1 <part> por OBJ. HELIOS++ atribui hitObjectId sequencial
    (1, 2, 3, ...) na ordem das <part>.
    """
    parts = "\n".join(
        f'    <part>\n'
        f'      <filter type="objloader">\n'
        f'        <param type="string" key="filepath" value="{p.name}" />\n'
        f'      </filter>\n'
        f'    </part>'
        for p in obj_paths
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <scene id="scene" name="scene">
{parts}
  </scene>
</document>
"""


def _build_survey_xml(
    scene_ref: str,
    scanner_ref: str,
    platform_ref: str,
    scanner_positions: list[tuple[float, float, float]],
    pulse_freq_hz: int = 100_000,
    scan_angle_deg: float = 60.0,
    scan_freq_hz: int = 100,
    head_rotate_per_sec: float = 60.0,
    head_rotate_start_deg: float = -180.0,
    head_rotate_stop_deg: float = 180.0,
) -> str:
    """Survey com uma 'leg' por scanner_position."""
    legs = []
    for x, y, z in scanner_positions:
        legs.append(f"""    <leg>
      <platformSettings x="{x:.3f}" y="{y:.3f}" z="{z:.3f}" onGround="false" />
      <scannerSettings active="true"
                       pulseFreq_hz="{pulse_freq_hz}"
                       scanAngle_deg="{scan_angle_deg}"
                       scanFreq_hz="{scan_freq_hz}"
                       headRotatePerSec_deg="{head_rotate_per_sec}"
                       headRotateStart_deg="{head_rotate_start_deg}"
                       headRotateStop_deg="{head_rotate_stop_deg}" />
    </leg>""")
    legs_xml = "\n".join(legs)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <survey name="auto_survey"
          scene="{scene_ref}"
          platform="{platform_ref}"
          scanner="{scanner_ref}">
{legs_xml}
  </survey>
</document>
"""


# ============================================================
# Runner principal
# ============================================================
def scan_meshes(
    meshes: dict[str, o3d.geometry.TriangleMesh],
    scanner_positions: list[tuple[float, float, float]],
    tipos_por_guid: Optional[dict[str, str]] = None,
    *,
    pulse_freq_hz: int = 100_000,
    scan_angle_deg: float = 60.0,
    scan_freq_hz: int = 100,
    head_rotate_per_sec: float = 60.0,
    keep_workdir: bool = False,
) -> dict:
    """Roda HELIOS++ sobre os meshes e devolve a nuvem com labels por guid.

    Args:
        meshes: {guid: TriangleMesh}
        scanner_positions: lista de (x, y, z) com posições do scanner
        tipos_por_guid: {guid: 'IfcWall'/'IfcSlab'/...} pra associar reflectance
        pulse_freq_hz, scan_angle_deg, scan_freq_hz, head_rotate_per_sec:
            parâmetros do scanner (variar entre cenas pra densidade diversa)
        keep_workdir: True pra debug (não apaga o tempdir)

    Returns:
        dict {
            'pts':       np.ndarray (N, 3) float32 — coords XYZ
            'intensity': np.ndarray (N,)   float32 — intensity por ponto
            'hit_guid':  np.ndarray (N,)   object  — guid IFC do objeto atingido
            'hit_idx':   np.ndarray (N,)   int     — índice numérico (interno)
            'guid_to_idx': dict {guid: int}        — mapa idx → guid
            'n_pts': int
        }
    """
    helios_bin = find_helios_bin()

    # Prepara materials por classe (se tipos fornecidos)
    materials = None
    if tipos_por_guid:
        materials = {g: tipos_por_guid.get(g, "default") for g in meshes}

    workdir = Path(tempfile.mkdtemp(prefix="helios_scan_"))
    try:
        scene_xml = workdir / "scene.xml"
        survey_xml = workdir / "survey.xml"

        # 1. Exporta 1 OBJ por mesh (cada vira ScenePart distinta)
        guid_to_obj = export_one_obj_per_mesh(meshes, workdir)
        # Ordem importa — HELIOS atribui hitObjectId sequencial pela ordem das <part>
        ordered_guids = list(guid_to_obj.keys())
        guid_to_idx = {g: i for i, g in enumerate(ordered_guids)}

        # 2. Escreve XMLs (1 <part> por OBJ)
        scene_xml.write_text(_build_scene_xml([guid_to_obj[g] for g in ordered_guids]))
        survey_xml.write_text(_build_survey_xml(
            scene_ref=f"{scene_xml.name}#scene",
            scanner_ref=DEFAULT_SCANNER,
            platform_ref=DEFAULT_PLATFORM,
            scanner_positions=scanner_positions,
            pulse_freq_hz=pulse_freq_hz,
            scan_angle_deg=scan_angle_deg,
            scan_freq_hz=scan_freq_hz,
            head_rotate_per_sec=head_rotate_per_sec,
        ))

        # 3. Roda HELIOS++ (espera que helios esteja em PATH do conda env)
        cmd = [
            str(helios_bin),
            survey_xml.name,
            "--rebuildScene",
        ]
        result = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"HELIOS++ falhou (rc={result.returncode}):\n"
                f"stderr:\n{result.stderr[-2000:]}\n"
                f"stdout:\n{result.stdout[-1000:]}"
            )

        # 4. Coleta XYZ outputs (1 por leg)
        out_root = workdir / "output" / "auto_survey"
        run_dirs = sorted(out_root.glob("*"))
        if not run_dirs:
            raise RuntimeError(f"HELIOS++ não gerou output em {out_root}")
        latest = run_dirs[-1]

        xyz_files = sorted(latest.glob("leg*_points.xyz"))
        if not xyz_files:
            raise RuntimeError(f"Nenhum leg*_points.xyz em {latest}")

        # Cada linha HELIOS++ XYZ: x y z intensity echoWidth returnNum numRet
        #                          fullwaveIndex hitObjectId classId timestamp...
        # (col 8 = hitObjectId = índice 0-based da ScenePart na ordem do scene.xml)
        pts_all = []
        intensity_all = []
        hit_idx_all = []
        for xyz in xyz_files:
            data = np.loadtxt(xyz, dtype=np.float32)
            if data.ndim == 1:  # caso especial: 1 ponto só
                data = data[None, :]
            pts_all.append(data[:, :3])
            intensity_all.append(data[:, 3])
            hit_idx_all.append(data[:, 8].astype(np.int32))  # ScenePart ID

        pts       = np.concatenate(pts_all)       if pts_all else np.zeros((0, 3), dtype=np.float32)
        intensity = np.concatenate(intensity_all) if intensity_all else np.zeros((0,), dtype=np.float32)
        hit_idx   = np.concatenate(hit_idx_all)   if hit_idx_all else np.zeros((0,), dtype=np.int32)

        # 5. Mapeia hit_idx → guid
        # HELIOS++ usa hitObjectId 0-based na ordem que os grupos aparecem no OBJ
        idx_to_guid = {v: k for k, v in guid_to_idx.items()}
        hit_guid = np.array([idx_to_guid.get(int(i), "") for i in hit_idx], dtype=object)

        return {
            "pts":         pts,
            "intensity":   intensity,
            "hit_guid":    hit_guid,
            "hit_idx":     hit_idx,
            "guid_to_idx": guid_to_idx,
            "n_pts":       int(len(pts)),
        }

    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    import sys
    print("HELIOS bin:", find_helios_bin())

    # Cria duas paredes lado a lado (teste de occlusion)
    wall1 = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.25, depth=3.0)
    wall1.translate([0.0, 0.0, 0.0])

    wall2 = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.25, depth=3.0)
    wall2.translate([0.0, 4.0, 0.0])   # paralela 4m atrás

    meshes = {
        "wall_front_guid": wall1,
        "wall_back_guid":  wall2,
    }
    tipos = {
        "wall_front_guid": "IfcWall",
        "wall_back_guid":  "IfcWall",
    }
    # Scanner em frente da wall1, olhando pra ela (e parede 2 fica oculta)
    scanners = [(2.0, -3.0, 1.5)]

    print(f"Rodando scan com {len(scanners)} scanner(s)...")
    out = scan_meshes(meshes, scanners, tipos_por_guid=tipos)
    print(f"Total pts: {out['n_pts']:,}")

    from collections import Counter
    dist = Counter(out["hit_guid"].tolist())
    print(f"Distribuição por guid:")
    for g, n in dist.most_common():
        print(f"  {g}: {n}")

    print(f"\nIntensity range: {out['intensity'].min():.0f} - {out['intensity'].max():.0f}")
    print(f"Esperado: ~todos os pts em wall_front_guid; wall_back_guid bloqueada por occlusion")
