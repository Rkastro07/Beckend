"""Mask3D inference em PLY avulso.

Carrega um PLY, voxeliza, roda o Mask3D e salva:
  - PLY colorido por instancia
  - PLY colorido por classe semantica

Uso (dentro do WSL):
  python infer_ply.py <caminho.ply> [--out pasta_saida] [--voxel 0.05]
"""
import sys
import os
import time
import argparse
from pathlib import Path

# Adiciona Mask3D ao path
MASK3D_DIR = Path("/home/rafael/Mask3D")
sys.path.insert(0, str(MASK3D_DIR))

import numpy as np
import open3d as o3d
import torch
import MinkowskiEngine as ME
from models.mask3d import Mask3D
from omegaconf import OmegaConf


# ScanNet 20 classes (sem wall/floor que sao filtradas no treino)
# Mask3D ScanNet filtra classes 0(wall) e 1(floor) e usa label_offset=2
# Entao as 18 classes de saida mapeiam pra:
SCANNET_LABELS_20 = [
    "wall", "floor",  # filtered out (0,1) - nao previstas pelo modelo
    "cabinet", "bed", "chair", "sofa", "table", "door", "window",
    "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower_curtain", "toilet", "sink", "bathtub",
    "otherfurniture"
]

# Na verdade o Mask3D ScanNet val prevê 18 classes (sem wall/floor)
# Indice 0 do modelo = cabinet (id 2 no ScanNet)
MASK3D_CLASSES = SCANNET_LABELS_20[2:]  # 18 classes

# Cores por classe
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

CLASS_COLORS = {
    "cabinet":  (0.80, 0.60, 0.40),
    "bed":      (0.60, 0.40, 0.80),
    "chair":    (0.90, 0.30, 0.30),
    "sofa":     (0.40, 0.70, 0.40),
    "table":    (0.80, 0.80, 0.30),
    "door":     (0.90, 0.70, 0.40),
    "window":   (0.50, 0.90, 0.90),
    "bookshelf":(0.70, 0.50, 0.30),
    "picture":  (0.60, 0.60, 0.90),
    "counter":  (0.70, 0.70, 0.60),
    "desk":     (0.85, 0.75, 0.50),
    "curtain":  (0.90, 0.60, 0.70),
    "refrigerator": (0.50, 0.80, 0.50),
    "shower_curtain": (0.80, 0.50, 0.60),
    "toilet":   (0.95, 0.95, 0.95),
    "sink":     (0.85, 0.85, 0.90),
    "bathtub":  (0.80, 0.85, 0.95),
    "otherfurniture": (0.60, 0.60, 0.60),
}


def build_model(ckpt_path):
    """Constroi modelo Mask3D com config do ScanNet val."""
    print(f"Carregando checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Config ScanNet val (do YAML: indoor.yaml + mask3d.yaml + scannet_val.sh)
    # in_channels=6: colors(3) + raw_coordinates(3), sem normals
    # num_labels=20, num_targets=18 (filter_out wall+floor)
    backbone_cfg = OmegaConf.create({
        "backbone": {
            "_target_": "models.Res16UNet34C",
            "config": {"dialations": [1, 1, 1, 1], "conv1_kernel_size": 5, "bn_momentum": 0.02},
            "in_channels": 3,   # so cor normalizada
            "out_channels": 20,
            "out_fpn": True,
        }
    })

    model = Mask3D(
        config=backbone_cfg,
        hidden_dim=128,
        num_queries=150,  # scannet_val.sh override
        num_heads=8,
        dim_feedforward=1024,
        sample_sizes=[200, 800, 3200, 12800, 51200],
        shared_decoder=True,
        num_classes=19,  # 18 sem classes + 1 no-object (como no checkpoint)
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

    # Carrega pesos
    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        new_k = k.replace("model.", "", 1) if k.startswith("model.") else k
        state_dict[new_k] = v
    model.load_state_dict(state_dict, strict=False)

    return model


def preprocess_ply(ply_path, voxel_size=0.02):
    """Carrega PLY e prepara pro Mask3D."""
    print(f"Carregando PLY: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points).astype(np.float32)
    print(f"  {len(pts):,} pontos")

    # Cores (0-255 -> 0-1)
    if len(pcd.colors) > 0:
        colors = np.asarray(pcd.colors).astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
    else:
        colors = np.ones_like(pts) * 0.5

    # Normalizacao de cor (media/std do ScanNet)
    color_mean = np.array([0.47793125906962, 0.4303257521323044, 0.3749598901421883])
    color_std = np.array([0.2834475483823543, 0.27566157565723015, 0.27018971370874995])
    colors_norm = (colors - color_mean) / color_std

    # Normals
    pcd_proc = o3d.geometry.PointCloud()
    pcd_proc.points = o3d.utility.Vector3dVector(pts)
    pcd_proc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    normals = np.asarray(pcd_proc.normals).astype(np.float32)

    # Downsample se muito grande
    if len(pts) > 500_000:
        print(f"  Downsampling de {len(pts):,}...")
        vox = voxel_size
        while True:
            pcd_d = pcd_proc.voxel_down_sample(voxel_size=vox)
            if len(pcd_d.points) <= 500_000:
                break
            vox *= 1.2
        pts = np.asarray(pcd_d.points).astype(np.float32)
        pcd_d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        normals = np.asarray(pcd_d.normals).astype(np.float32)
        # Re-interpolar cores
        from scipy.spatial import cKDTree
        tree = cKDTree(np.asarray(pcd.points))
        _, idx = tree.query(pts, k=1)
        colors_norm = ((np.asarray(pcd.colors)[idx] if len(pcd.colors) > 0
                        else np.ones_like(pts) * 0.5) - color_mean) / color_std
        print(f"  Downsampled: {len(pts):,} pontos (voxel {vox:.3f}m)")

    # Features: so cor normalizada (3 canais) — backbone in_channels=3
    # raw_coordinates sao passadas separadamente ao modelo
    features = colors_norm.astype(np.float32)

    # Voxelizacao pra MinkowskiEngine
    coords = (pts / voxel_size).astype(np.int32)
    # Remover duplicatas
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    coords = coords[unique_idx]
    features = features[unique_idx]
    pts = pts[unique_idx]

    print(f"  Apos voxelizacao: {len(coords):,} voxels")

    return pts, coords, features


def main():
    parser = argparse.ArgumentParser(description="Mask3D inference em PLY")
    parser.add_argument("ply", help="Caminho do PLY")
    parser.add_argument("--out", help="Pasta de saida")
    parser.add_argument("--voxel", type=float, default=0.02, help="Voxel size (m)")
    parser.add_argument("--ckpt", default=str(MASK3D_DIR / "checkpoints/scannet/scannet_val.ckpt"))
    args = parser.parse_args()

    ply_path = Path(args.ply)
    out_dir = Path(args.out) if args.out else ply_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MASK3D - Instance Segmentation")
    print("=" * 60)

    # 1. Carrega modelo
    model = build_model(args.ckpt)
    model = model.cuda().eval()
    print("Modelo carregado!")

    # 2. Preprocess PLY
    pts, coords, features = preprocess_ply(ply_path, voxel_size=args.voxel)

    # 3. Inference
    print("\nRodando Mask3D...")
    t0 = time.time()
    with torch.no_grad():
        # Cria SparseTensor do MinkowskiEngine
        coords_t = torch.from_numpy(coords).int()
        feats_t = torch.from_numpy(features).float()

        # Adiciona batch index
        batch_idx = torch.zeros(len(coords_t), 1, dtype=torch.int32)
        coords_batch = torch.cat([batch_idx, coords_t], dim=1)

        sinput = ME.SparseTensor(
            features=feats_t.cuda(),
            coordinates=coords_batch.cuda(),
        )

        # Cria point2segment: cada ponto eh seu proprio segmento
        # (ou agrupa em super-voxels grosseiros)
        seg_voxel = 0.1  # super-voxel de 10cm
        seg_coords = (pts / seg_voxel).astype(np.int32)
        _, seg_inverse = np.unique(seg_coords, axis=0, return_inverse=True)
        point2segment = [torch.from_numpy(seg_inverse).long().cuda()]

        # raw_coordinates precisam ser coords do SparseTensor
        raw_coords = torch.from_numpy(pts).float().cuda()

        # Forward
        outputs = model(sinput, point2segment=point2segment, raw_coordinates=raw_coords, is_eval=True)

    dt = time.time() - t0
    print(f"Inferencia em {dt:.1f}s")

    # 4. Extrair predicoes
    pred_logits = outputs["pred_logits"]  # list de tensores ou tensor
    pred_masks = outputs["pred_masks"]    # list de tensores ou tensor

    # pred_logits/pred_masks podem ser lista (batch) ou tensor
    if isinstance(pred_logits, list):
        pred_logits = pred_logits[0]  # primeiro (unico) batch
    if isinstance(pred_masks, list):
        pred_masks = pred_masks[0]

    # Squeeze batch dim se existir
    if pred_logits.dim() == 3:
        pred_logits = pred_logits[0]
    if pred_masks.dim() == 3:
        pred_masks = pred_masks[0]

    # Prob de cada query ser cada classe
    pred_probs = pred_logits.softmax(-1)  # (num_queries, num_classes+1)
    # Mascara por query — pode estar no espaco de segmentos
    pred_masks_sig = pred_masks.sigmoid()  # (num_queries, N_segments)

    # Se train_on_segments, as mascaras estao no espaco de segmentos
    # Preciso expandir pro espaco de pontos usando point2segment
    seg_ids = point2segment[0].cpu().numpy()  # (N_pontos,) -> id do segmento
    N = len(pts)

    # Move pra CPU pra nao estourar VRAM
    pred_masks_cpu = pred_masks_sig.cpu().numpy()  # (num_queries, N_segments)

    # Filtra queries com score alto
    scores, labels = pred_probs[:, :-1].max(-1)  # ignora ultima classe (no-object)
    keep = scores > 0.3  # threshold mais baixo pra ver mais resultados
    keep_idx = torch.where(keep)[0]

    print(f"\n{len(keep_idx)} instancias detectadas (score > 0.3)")

    # 5. Atribui cada ponto a melhor instancia
    point_instance = np.full(N, -1, dtype=np.int32)
    point_class = np.full(N, -1, dtype=np.int32)
    point_score = np.zeros(N, dtype=np.float32)

    instances_info = []
    for rank, qi in enumerate(keep_idx.cpu().numpy()):
        cls_id = labels[qi].item()
        score = scores[qi].item()
        cls_name = MASK3D_CLASSES[cls_id] if cls_id < len(MASK3D_CLASSES) else f"cls_{cls_id}"

        # Mascara no espaco de segmentos -> expande pra pontos
        seg_mask = pred_masks_cpu[qi]  # (N_segments,)
        point_mask = seg_mask[seg_ids] > 0.5  # (N_pontos,)

        # So sobrescreve se score maior
        better = point_mask & (point_score < score)
        point_instance[better] = rank
        point_class[better] = cls_id
        point_score[better] = score

        n_pts = point_mask.sum()
        instances_info.append({"id": rank, "class": cls_name, "score": score, "n_pts": n_pts})
        print(f"  [{rank}] {cls_name:18s} score={score:.2f} pts={n_pts:,}")

    # 6. Salvar PLY por instancia
    stem = ply_path.stem
    ply_inst = out_dir / f"{stem}_mask3d_instances.ply"
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts)
    cols = np.ones((N, 3)) * 0.3  # cinza pra nao-classificados
    for i in range(len(instances_info)):
        mask = point_instance == i
        c = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        cols[mask] = c
    pcd_out.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(str(ply_inst), pcd_out)
    print(f"\nSalvo: {ply_inst}")

    # PLY por classe semantica
    ply_sem = out_dir / f"{stem}_mask3d_semantic.ply"
    cols_sem = np.ones((N, 3)) * 0.3
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

    print(f"\n{'='*60}")
    print(f"PRONTO! {len(instances_info)} instancias detectadas")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
