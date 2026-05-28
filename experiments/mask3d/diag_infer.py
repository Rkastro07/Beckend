# -*- coding: utf-8 -*-
"""Diagnostico detalhado da inferencia Mask3D BIM"""
import sys
sys.path.insert(0, "/home/rafael/Mask3D")

import numpy as np
import open3d as o3d
import torch
import MinkowskiEngine as ME
from models.mask3d import Mask3D
from omegaconf import OmegaConf
from collections import Counter

BIM_NAMES = ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam",
             "IfcStair", "IfcRoof", "IfcSanitaryTerminal"]
NUM_MODEL_CLASSES = 8

# Build model
backbone_cfg = OmegaConf.create({
    "backbone": {
        "_target_": "models.Res16UNet34C",
        "config": {"dialations": [1,1,1,1], "conv1_kernel_size": 5, "bn_momentum": 0.02},
        "in_channels": 3, "out_channels": 20, "out_fpn": True,
    }
})
model = Mask3D(
    config=backbone_cfg, hidden_dim=128, num_queries=150, num_heads=8,
    dim_feedforward=1024, sample_sizes=[200,800,3200,12800,51200],
    shared_decoder=True, num_classes=NUM_MODEL_CLASSES, num_decoders=3,
    dropout=0.0, pre_norm=False, positional_encoding_type="fourier",
    non_parametric_queries=True, train_on_segments=True,
    normalize_pos_enc=True, use_level_embed=False, scatter_type="mean",
    hlevels=[0,1,2,3], use_np_features=False, voxel_size=0.02,
    max_sample_size=False, random_queries=False, gauss_scale=1.0,
    random_query_both=False, random_normal=False,
)

ckpt = torch.load("/tmp/mask3d_bim_checkpoints/best_bim.ckpt", map_location="cpu")
state = {}
for k, v in ckpt["state_dict"].items():
    new_k = k.replace("model.", "", 1) if k.startswith("model.") else k
    state[new_k] = v
model.load_state_dict(state, strict=False)
device = torch.device("cuda")
model = model.to(device).eval()
print("Modelo carregado")

# Load PLY
ply_path = "/mnt/c/Users/Rafael/Desktop/Beckend/dataset/ply_rhpc/estagio_07_so_estrutura.ply"
pcd = o3d.io.read_point_cloud(ply_path)
pts_orig = np.asarray(pcd.points).astype(np.float32)
print("Pontos: {:,}".format(len(pts_orig)))

# Normals
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
normals = np.asarray(pcd.normals).astype(np.float32)

# Voxelize
voxel_size = 0.05
coords = np.floor(pts_orig / voxel_size).astype(np.int32)
_, unique_idx = np.unique(coords, axis=0, return_index=True)
unique_idx = np.sort(unique_idx)
coords = coords[unique_idx]
pts = pts_orig[unique_idx]
features = normals[unique_idx]

# Segments
seg_coords = np.floor(pts / 0.15).astype(np.int32)
_, seg_ids = np.unique(seg_coords, axis=0, return_inverse=True)
print("Voxels: {:,} | Segments: {}".format(len(coords), seg_ids.max()+1))

# Inference
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
        sinput, point2segment=point2segment,
        raw_coordinates=raw_coords, is_eval=True,
    )

pred_logits = outputs["pred_logits"]
pred_masks = outputs["pred_masks"]
if isinstance(pred_logits, list): pred_logits = pred_logits[0]
if isinstance(pred_masks, list): pred_masks = pred_masks[0]
if pred_logits.dim() == 3: pred_logits = pred_logits[0]
if pred_masks.dim() == 3: pred_masks = pred_masks[0]

Q = pred_logits.shape[0]
if pred_masks.shape[0] != Q:
    pred_masks = pred_masks.T

pred_probs = pred_logits.softmax(-1)
pred_masks_sig = pred_masks.sigmoid().cpu().numpy()

# Analysis
scores_all, labels_all = pred_probs[:, :-1].max(-1)
noobj_prob = pred_probs[:, -1]

print("\n" + "=" * 60)
print("ANALISE DETALHADA - DIAGNOSTICO")
print("=" * 60)
print("Queries totais: {}".format(Q))
print("Shape logits: {}".format(pred_logits.shape))
print("Shape masks: {}".format(pred_masks.shape))

# Distribution of predicted classes
print("\nDistribuicao de classes (todas as {} queries):".format(Q))
cls_dist = Counter(labels_all.cpu().numpy().tolist())
for cls_id, count in sorted(cls_dist.items()):
    name = BIM_NAMES[cls_id] if cls_id < len(BIM_NAMES) else "cls_{}".format(cls_id)
    print("  {:22s}: {} queries".format(name, count))

# no-object prob stats
print("\nProb no-object (media={:.4f}, min={:.4f}, max={:.4f})".format(
    noobj_prob.mean().item(), noobj_prob.min().item(), noobj_prob.max().item()))

# Top queries by score
print("\nTop 40 queries por score:")
order = scores_all.argsort(descending=True)
for rank, qi in enumerate(order[:40].cpu().numpy()):
    cls_id = labels_all[qi].item()
    score = scores_all[qi].item()
    noobj = noobj_prob[qi].item()
    cls_name = BIM_NAMES[cls_id] if cls_id < len(BIM_NAMES) else "cls_{}".format(cls_id)

    # Prob per class for this query
    probs_q = pred_probs[qi].cpu().numpy()
    top3_cls = np.argsort(probs_q[:-1])[::-1][:3]
    prob_str = " | ".join([
        "{}={:.3f}".format(
            BIM_NAMES[c] if c < len(BIM_NAMES) else "cls_{}".format(c),
            probs_q[c]
        ) for c in top3_cls
    ])

    seg_mask = pred_masks_sig[qi]
    point_mask = seg_mask[seg_ids] > 0.5
    n_pts = point_mask.sum()
    print("  [{:2d}] {:22s} score={:.4f} noobj={:.4f} pts={:,}  probs: {}".format(
        rank, cls_name, score, noobj, n_pts, prob_str))

# Thresholds
for thr in [0.3, 0.5, 0.7, 0.9]:
    keep = scores_all > thr
    n = keep.sum().item()
    if n > 0:
        kept_labels = labels_all[keep].cpu().numpy()
        cls_count = Counter(kept_labels.tolist())
        parts = []
        for cls_id, count in sorted(cls_count.items()):
            name = BIM_NAMES[cls_id] if cls_id < len(BIM_NAMES) else "cls_{}".format(cls_id)
            parts.append("{}={}".format(name, count))
        print("\nThreshold {}: {} instancias -> {}".format(thr, n, ", ".join(parts)))
    else:
        print("\nThreshold {}: 0 instancias".format(thr))

# Overlap analysis: how many points are claimed by multiple high-score queries?
print("\n--- OVERLAP ANALYSIS (threshold=0.5) ---")
keep05 = scores_all > 0.5
keep_idx05 = torch.where(keep05)[0].cpu().numpy()
N_vox = len(pts)
point_claims = np.zeros(N_vox, dtype=np.int32)
for qi in keep_idx05:
    seg_mask = pred_masks_sig[qi]
    point_mask = seg_mask[seg_ids] > 0.5
    point_claims += point_mask.astype(np.int32)

for nc in range(5):
    n = (point_claims == nc).sum()
    pct = 100.0 * n / N_vox
    print("  Pontos com {} instancias: {:,} ({:.1f}%)".format(nc, n, pct))

unclaimed = (point_claims == 0).sum()
print("  Pontos sem nenhuma instancia: {:,} ({:.1f}%)".format(
    unclaimed, 100.0 * unclaimed / N_vox))

print("\n" + "=" * 60)
print("FIM DIAGNOSTICO")
print("=" * 60)
