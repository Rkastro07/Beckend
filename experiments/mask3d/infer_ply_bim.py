# -*- coding: utf-8 -*-
"""
MASK3D BIM - Inferencia em PLY
================================
Carrega PLY, roda Mask3D treinado pra BIM e salva:
  - PLY colorido por instancia (parede #1, parede #2, laje #1, etc)
  - PLY colorido por classe semantica

Usa NORMALS como features (nao cor), 7 classes BIM + no-object.

Uso (WSL):
  python infer_ply_bim.py <caminho.ply>
  python infer_ply_bim.py <caminho.ply> --ckpt checkpoints/best_bim.ckpt --out resultados/
  python infer_ply_bim.py <caminho.ply> --threshold 0.5
"""
import sys
import os
import time
import argparse
from pathlib import Path

MASK3D_DIR = Path("/home/rafael/Mask3D")
sys.path.insert(0, str(MASK3D_DIR))

import numpy as np
import open3d as o3d
import torch
import MinkowskiEngine as ME
from models.mask3d import Mask3D
from omegaconf import OmegaConf


# ============================================================
# CLASSES BIM
# ============================================================
BIM_NAMES = [
    "IfcWall", "IfcSlab", "IfcColumn", "IfcBeam",
    "IfcStair", "IfcRoof", "IfcSanitaryTerminal",
]
NUM_MODEL_CLASSES = len(BIM_NAMES) + 1  # 8 (7 BIM + no-object)

# Cores por classe semantica
CLASS_COLORS = {
    "IfcWall":              (0.85, 0.55, 0.20),  # laranja
    "IfcSlab":              (0.60, 0.60, 0.60),  # cinza
    "IfcColumn":            (0.90, 0.25, 0.25),  # vermelho
    "IfcBeam":              (0.20, 0.60, 0.85),  # azul
    "IfcStair":             (0.55, 0.80, 0.25),  # verde claro
    "IfcRoof":              (0.75, 0.35, 0.70),  # roxo
    "IfcSanitaryTerminal":  (0.95, 0.90, 0.30),  # amarelo
}

# Cores pra instancias (ciclicas)
INSTANCE_COLORS = [
    (0.95, 0.26, 0.21), (0.30, 0.69, 0.31), (0.13, 0.59, 0.95),
    (1.00, 0.76, 0.03), (0.61, 0.15, 0.69), (0.00, 0.74, 0.83),
    (1.00, 0.60, 0.00), (0.55, 0.76, 0.29), (0.91, 0.12, 0.39),
    (0.40, 0.23, 0.72), (0.00, 0.59, 0.53), (0.80, 0.86, 0.22),
    (0.16, 0.71, 0.96), (0.85, 0.44, 0.84), (0.38, 0.49, 0.55),
    (1.00, 0.34, 0.13), (0.47, 0.33, 0.28), (0.62, 0.62, 0.62),
    (0.96, 0.50, 0.09), (0.24, 0.32, 0.71), (0.00, 0.90, 0.46),
    (0.74, 0.48, 0.00), (0.56, 0.34, 0.01), (0.43, 0.57, 0.13),
]


# ============================================================
# MODELO
# ============================================================
def build_model(ckpt_path):
    """Constroi Mask3D BIM e carrega checkpoint."""
    backbone_cfg = OmegaConf.create({
        "backbone": {
            "_target_": "models.Res16UNet34C",
            "config": {
                "dialations": [1, 1, 1, 1],
                "conv1_kernel_size": 5,
                "bn_momentum": 0.02,
            },
            "in_channels": 3,   # normals (3 canais)
            "out_channels": 20,
            "out_fpn": True,
        }
    })

    model = Mask3D(
        config=backbone_cfg,
        hidden_dim=128,
        num_queries=150,
        num_heads=8,
        dim_feedforward=1024,
        sample_sizes=[200, 800, 3200, 12800, 51200],
        shared_decoder=True,
        num_classes=NUM_MODEL_CLASSES,  # 8
        num_decoders=3,
        dropout=0.0,
        pre_norm=False,
        positional_encoding_type="fourier",
        non_parametric_queries=True,
        train_on_segments=True,
        normalize_pos_enc=True,
        use_level_embed=False,
        scatter_type="mean",
        hlevels=[0, 1, 2, 3],
        use_np_features=False,
        voxel_size=0.02,
        max_sample_size=False,
        random_queries=False,
        gauss_scale=1.0,
        random_query_both=False,
        random_normal=False,
    )

    # Carrega checkpoint BIM
    print(f"Carregando checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = {}
    for k, v in ckpt["state_dict"].items():
        new_k = k.replace("model.", "", 1) if k.startswith("model.") else k
        state[new_k] = v

    missing, unexpected = model.load_state_dict(state, strict=False)
    epoch = ckpt.get("epoch", "?")
    val = ckpt.get("val_loss", "?")
    print(f"  Checkpoint BIM epoch {epoch} (val={val})")
    if missing:
        print(f"  {len(missing)} missing keys")
    if unexpected:
        print(f"  {len(unexpected)} unexpected keys")

    return model


# ============================================================
# PREPROCESSAMENTO
# ============================================================
def preprocess_ply(ply_path, voxel_size=0.05):
    """Carrega PLY e prepara pro Mask3D BIM (normals como features)."""
    print(f"\nCarregando PLY: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts_orig = np.asarray(pcd.points).astype(np.float32)
    print(f"  {len(pts_orig):,} pontos originais")

    # Downsample se muito grande
    if len(pts_orig) > 500_000:
        print(f"  Downsampling...")
        vox = voxel_size
        while True:
            pcd_d = pcd.voxel_down_sample(voxel_size=vox)
            if len(pcd_d.points) <= 500_000:
                break
            vox *= 1.2
        pcd = pcd_d
        pts_orig = np.asarray(pcd.points).astype(np.float32)
        print(f"  Downsampled: {len(pts_orig):,} pontos (voxel {vox:.3f}m)")

    # Estima normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    normals = np.asarray(pcd.normals).astype(np.float32)

    # Voxelizacao pra MinkowskiEngine
    coords = np.floor(pts_orig / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)

    coords = coords[unique_idx]
    pts = pts_orig[unique_idx]
    features = normals[unique_idx]

    print(f"  Apos voxelizacao ({voxel_size}m): {len(coords):,} voxels")

    # Super-voxel segments (0.15m, como no treino)
    seg_coords = np.floor(pts / 0.15).astype(np.int32)
    _, seg_inverse = np.unique(seg_coords, axis=0, return_inverse=True)

    return pts_orig, unique_idx, pts, coords, features, seg_inverse


# ============================================================
# INFERENCIA
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Mask3D BIM - Inferencia em PLY")
    parser.add_argument("ply", help="Caminho do PLY")
    parser.add_argument("--out", help="Pasta de saida (default: mesma do PLY)")
    parser.add_argument("--ckpt",
                        default=str(Path(__file__).parent / "checkpoints/best_bim.ckpt"),
                        help="Checkpoint BIM")
    parser.add_argument("--voxel", type=float, default=0.05, help="Voxel size (m)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Score minimo pra considerar instancia")
    args = parser.parse_args()

    ply_path = Path(args.ply)
    out_dir = Path(args.out) if args.out else ply_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MASK3D BIM - Instance Segmentation")
    print("=" * 60)

    # 1. Modelo
    model = build_model(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"Dispositivo: {device}")

    # 2. Preprocess
    pts_orig, orig_idx, pts, coords, features, seg_ids = preprocess_ply(
        ply_path, voxel_size=args.voxel
    )

    # 3. Inference
    print("\nRodando Mask3D BIM...")
    t0 = time.time()

    with torch.no_grad():
        coords_t = torch.from_numpy(coords).int()
        feats_t = torch.from_numpy(features).float()

        batch_idx = torch.zeros(len(coords_t), 1, dtype=torch.int32)
        coords_batch = torch.cat([batch_idx, coords_t], dim=1)

        sinput = ME.SparseTensor(
            features=feats_t.to(device),
            coordinates=coords_batch.to(device),
        )

        point2segment = [torch.from_numpy(seg_ids).long().to(device)]
        raw_coords = torch.from_numpy(pts).float().to(device)

        outputs = model(
            sinput,
            point2segment=point2segment,
            raw_coordinates=raw_coords,
            is_eval=True,
        )

    dt = time.time() - t0
    print(f"Inferencia em {dt:.1f}s")

    # 4. Extrair predicoes
    pred_logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]

    if isinstance(pred_logits, list):
        pred_logits = pred_logits[0]
    if isinstance(pred_masks, list):
        pred_masks = pred_masks[0]
    if pred_logits.dim() == 3:
        pred_logits = pred_logits[0]
    if pred_masks.dim() == 3:
        pred_masks = pred_masks[0]

    # Corrige orientacao se necessario
    Q = pred_logits.shape[0]
    if pred_masks.shape[0] != Q:
        pred_masks = pred_masks.T

    pred_probs = pred_logits.softmax(-1)        # (Q, num_classes)
    pred_masks_sig = pred_masks.sigmoid().cpu().numpy()  # (Q, N_segments)

    # Filtra por score (ignora ultima classe = no-object)
    scores, labels = pred_probs[:, :-1].max(-1)

    keep = scores > args.threshold
    keep_idx = torch.where(keep)[0]

    # Ordena por score (maior primeiro)
    order = scores[keep_idx].argsort(descending=True)
    keep_idx = keep_idx[order]

    print(f"\n{len(keep_idx)} instancias detectadas (score > {args.threshold})")

    # 5. Atribuir pontos a instancias
    N_vox = len(pts)
    point_instance = np.full(N_vox, -1, dtype=np.int32)
    point_class = np.full(N_vox, -1, dtype=np.int32)
    point_score = np.zeros(N_vox, dtype=np.float32)

    instances_info = []
    for rank, qi in enumerate(keep_idx.cpu().numpy()):
        cls_id = labels[qi].item()
        score = scores[qi].item()
        cls_name = BIM_NAMES[cls_id] if cls_id < len(BIM_NAMES) else f"cls_{cls_id}"

        # Mascara: segmentos -> pontos
        seg_mask = pred_masks_sig[qi]
        point_mask = seg_mask[seg_ids] > 0.5

        # Atribui (score mais alto ganha)
        better = point_mask & (point_score < score)
        point_instance[better] = rank
        point_class[better] = cls_id
        point_score[better] = score

        n_pts = point_mask.sum()
        instances_info.append({
            "id": rank, "class": cls_name, "score": score, "n_pts": int(n_pts)
        })
        print(f"  [{rank:2d}] {cls_name:22s} score={score:.3f} pts={n_pts:,}")

    # 6. Salvar PLYs
    stem = ply_path.stem

    # --- PLY por instancia ---
    ply_inst = out_dir / f"{stem}_bim_instances.ply"
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts)
    cols = np.ones((N_vox, 3)) * 0.3
    for i in range(len(instances_info)):
        mask = point_instance == i
        c = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        cols[mask] = c
    pcd_out.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(str(ply_inst), pcd_out)
    print(f"\nSalvo: {ply_inst}")

    # --- PLY por classe semantica ---
    ply_sem = out_dir / f"{stem}_bim_semantic.ply"
    cols_sem = np.ones((N_vox, 3)) * 0.3
    for i in range(len(instances_info)):
        cls_name = instances_info[i]["class"]
        mask = point_instance == i
        c = CLASS_COLORS.get(cls_name, (0.5, 0.5, 0.5))
        cols_sem[mask] = c
    pcd_sem = o3d.geometry.PointCloud()
    pcd_sem.points = o3d.utility.Vector3dVector(pts)
    pcd_sem.colors = o3d.utility.Vector3dVector(cols_sem)
    o3d.io.write_point_cloud(str(ply_sem), pcd_sem)
    print(f"Salvo: {ply_sem}")

    # Resumo
    print(f"\n{'='*60}")
    print(f"RESULTADO: {len(instances_info)} instancias BIM detectadas")
    print(f"{'='*60}")

    # Contagem por classe
    from collections import Counter
    cls_count = Counter(inst["class"] for inst in instances_info)
    for cls_name, count in sorted(cls_count.items()):
        print(f"  {cls_name:22s}: {count} instancias")

    print(f"\nArquivos:")
    print(f"  Instancias: {ply_inst}")
    print(f"  Semantico:  {ply_sem}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
