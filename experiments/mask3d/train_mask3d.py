# -*- coding: utf-8 -*-
"""
MASK3D FINE-TUNING PARA BIM
============================
Fine-tune do Mask3D (pre-treinado ScanNet) para segmentacao de
instancias de estruturas BIM.

7 classes BIM:
  0 = IfcWall        3 = IfcBeam       6 = IfcSanitaryTerminal
  1 = IfcSlab        4 = IfcStair
  2 = IfcColumn      5 = IfcRoof

O modelo preve num_queries mascaras, cada uma com classe + mascara binaria.
Ultima classe (7) = no-object (query sem instancia atribuida).

Uso (WSL):
  python train_mask3d.py
  python train_mask3d.py --epochs 30 --lr 1e-4 --freeze_backbone
  python train_mask3d.py --data /tmp/mask3d_data --epochs 50
"""

import sys
import os
import gc
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment

# Mask3D — configura via variavel de ambiente ou usa default Colab
# WSL:   export MASK3D_DIR=/home/rafael/Mask3D
# Colab: nao precisa exportar (default ja eh /content/Mask3D)
MASK3D_DIR = Path(os.environ.get("MASK3D_DIR", "/content/Mask3D"))
sys.path.insert(0, str(MASK3D_DIR))

import MinkowskiEngine as ME
from models.mask3d import Mask3D
from omegaconf import OmegaConf


# ============================================================
# CONSTANTES
# ============================================================
NUM_BIM_CLASSES = 7       # IfcWall..IfcSanitaryTerminal (indices 0-6)
NUM_MODEL_CLASSES = 8     # 7 BIM + 1 no-object
BIM_NAMES = [
    "IfcWall", "IfcSlab", "IfcColumn", "IfcBeam",
    "IfcStair", "IfcRoof", "IfcSanitaryTerminal",
]

DEFAULT_DATA = "/tmp/mask3d_data"
DEFAULT_CKPT = str(MASK3D_DIR / "checkpoints/scannet/scannet_val.ckpt")
DEFAULT_OUT  = "/tmp/mask3d_bim_checkpoints"


# ============================================================
# DATASET
# ============================================================
class BIMInstanceDataset(Dataset):
    """
    Carrega .npz gerado pelo dataset_generator_mask3d.py.
    Retorna cena voxelizada com labels de instancia no espaco de segmentos.
    """

    def __init__(self, npz_paths, voxel_size=0.05, seg_size=0.15,
                 max_voxels=80000, augment=True):
        self.paths = list(npz_paths)
        self.voxel_size = voxel_size
        self.seg_size = seg_size
        self.max_voxels = max_voxels
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        pts     = data["pts"].copy()             # (N, 3)
        normals = data["normals"].copy()          # (N, 3)
        sem     = data["semantic_labels"].copy()  # (N,)
        inst    = data["instance_labels"].copy()  # (N,)

        # ---- Data augmentation ----
        if self.augment:
            # Rotacao aleatoria em Z
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            pts = pts @ R.T
            normals = normals @ R.T

            # Jitter
            pts += np.random.normal(0, 0.005, pts.shape).astype(np.float32)

            # Flip X
            if np.random.random() > 0.5:
                pts[:, 0] *= -1
                normals[:, 0] *= -1

            # Flip Y
            if np.random.random() > 0.5:
                pts[:, 1] *= -1
                normals[:, 1] *= -1

        # ---- Voxelizacao ----
        coords = (pts / self.voxel_size).astype(np.int32)
        _, unique_idx = np.unique(coords, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)

        coords  = coords[unique_idx]
        pts     = pts[unique_idx]
        normals = normals[unique_idx]
        sem     = sem[unique_idx]
        inst    = inst[unique_idx]

        # ---- Limita voxels pra caber na VRAM ----
        if self.max_voxels and len(coords) > self.max_voxels:
            sel = np.random.choice(len(coords), self.max_voxels, replace=False)
            sel = np.sort(sel)
            coords  = coords[sel]
            pts     = pts[sel]
            normals = normals[sel]
            sem     = sem[sel]
            inst    = inst[sel]

        # ---- Super-voxel segments ----
        seg_coords = (pts / self.seg_size).astype(np.int32)
        _, seg_inverse = np.unique(seg_coords, axis=0, return_inverse=True)
        n_segs = seg_inverse.max() + 1

        # ---- Features: normais normalizadas ----
        features = normals.astype(np.float32)

        # ---- GT: mascaras por instancia no espaco de segmentos ----
        instance_ids = np.unique(inst[inst >= 0])
        gt_masks = []
        gt_classes = []

        # Pre-calcula total de pontos por segmento (evita loop O(N*S))
        seg_totals = np.bincount(seg_inverse, minlength=n_segs).astype(np.float32)
        seg_totals = np.maximum(seg_totals, 1.0)

        for iid in instance_ids:
            pt_mask = inst == iid
            if pt_mask.sum() == 0:
                continue
            cls = int(sem[pt_mask][0]) - 1  # 1-7 -> 0-6
            if cls < 0 or cls >= NUM_BIM_CLASSES:
                continue

            # Mascara no espaco de segmentos: fracao de pontos do segmento
            seg_counts = np.bincount(
                seg_inverse[pt_mask], minlength=n_segs
            ).astype(np.float32)
            seg_mask = seg_counts / seg_totals

            gt_masks.append(seg_mask)
            gt_classes.append(cls)

        if gt_masks:
            gt_masks = np.stack(gt_masks, axis=0)    # (I, S)
            gt_classes = np.array(gt_classes, dtype=np.int64)
        else:
            gt_masks = np.zeros((0, n_segs), dtype=np.float32)
            gt_classes = np.array([], dtype=np.int64)

        return {
            "coords":     coords,       # (V, 3) int32
            "features":   features,     # (V, 3) float32
            "pts":        pts,          # (V, 3) float32
            "segments":   seg_inverse,  # (V,) int64
            "gt_masks":   gt_masks,     # (I, S) float32
            "gt_classes": gt_classes,   # (I,) int64
        }


def collate_single(batch):
    """batch_size=1, retorna cena direto."""
    return batch[0]


# ============================================================
# HUNGARIAN MATCHER
# ============================================================
class HungarianMatcher:
    """
    Encontra correspondencia otima entre queries e GT via custo combinado
    (classe + BCE mascara + Dice mascara).
    """

    def __init__(self, cost_class=2.0, cost_mask=5.0, cost_dice=5.0):
        self.cost_class = cost_class
        self.cost_mask  = cost_mask
        self.cost_dice  = cost_dice

    @torch.no_grad()
    def match(self, pred_logits, pred_masks, gt_classes, gt_masks):
        """
        Args:
            pred_logits: (Q, C)  -- logits de classe por query
            pred_masks:  (Q, S)  -- logits de mascara por query
            gt_classes:  (I,)    -- classe de cada GT (0..6)
            gt_masks:    (I, S)  -- mascara de cada GT (0-1)

        Returns:
            (pred_idx, gt_idx) -- tensores matched
        """
        Q = pred_logits.shape[0]
        I = gt_classes.shape[0]
        if I == 0:
            return (torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long))

        # Custo de classe: -prob da classe correta
        probs = pred_logits.softmax(-1)       # (Q, C)
        cost_cls = -probs[:, gt_classes]      # (Q, I)

        # Custo BCE de mascara
        pred_sig = pred_masks.sigmoid()       # (Q, S)
        cost_bce = (
            F.binary_cross_entropy_with_logits(
                pred_masks.unsqueeze(1).expand(-1, I, -1),
                gt_masks.unsqueeze(0).expand(Q, -1, -1),
                reduction="none",
            ).mean(-1)
        )  # (Q, I)

        # Custo Dice
        num = 2 * (pred_sig.unsqueeze(1) * gt_masks.unsqueeze(0)).sum(-1)
        den = pred_sig.unsqueeze(1).sum(-1) + gt_masks.unsqueeze(0).sum(-1) + 1e-8
        cost_dice = 1.0 - num / den  # (Q, I)

        C = (self.cost_class * cost_cls +
             self.cost_mask  * cost_bce +
             self.cost_dice  * cost_dice)

        row, col = linear_sum_assignment(C.cpu().numpy())
        return (torch.tensor(row, dtype=torch.long),
                torch.tensor(col, dtype=torch.long))


# ============================================================
# LOSS
# ============================================================
class Mask3DBIMLoss(nn.Module):
    """
    Loss estilo DETR/Mask2Former:
      - CrossEntropy para classificacao (com peso baixo para no-object)
      - BCE + Dice para mascaras (so queries matched)
    """

    def __init__(self, num_classes=NUM_MODEL_CLASSES,
                 w_ce=0.5, w_bce=5.0, w_dice=5.0, no_obj_weight=0.1):
        super().__init__()
        self.matcher = HungarianMatcher()
        self.num_classes = num_classes
        self.w_ce   = w_ce
        self.w_bce  = w_bce
        self.w_dice = w_dice

        # Peso menor pra no-object (maioria das queries nao tem match)
        weights = torch.ones(num_classes)
        weights[-1] = no_obj_weight
        self.register_buffer("ce_weights", weights)

    def forward(self, pred_logits, pred_masks, gt_classes, gt_masks):
        """
        pred_logits: (Q, C)
        pred_masks:  (Q, S)
        gt_classes:  (I,) long  -- valores 0..6
        gt_masks:    (I, S)     -- mascaras binarizadas
        """
        device = pred_logits.device
        Q = pred_logits.shape[0]

        # Sem instancias: todas as queries sao no-object
        if gt_classes.shape[0] == 0:
            target = torch.full((Q,), self.num_classes - 1,
                                dtype=torch.long, device=device)
            loss_ce = F.cross_entropy(pred_logits, target,
                                       weight=self.ce_weights.to(device))
            z = torch.tensor(0.0, device=device)
            return {"loss": loss_ce, "ce": loss_ce, "bce": z, "dice": z}

        # Hungarian matching
        row_idx, col_idx = self.matcher.match(
            pred_logits, pred_masks, gt_classes, gt_masks
        )
        row_idx = row_idx.to(device)
        col_idx = col_idx.to(device)

        # --- Classification loss ---
        target = torch.full((Q,), self.num_classes - 1,
                            dtype=torch.long, device=device)
        target[row_idx] = gt_classes[col_idx]
        loss_ce = F.cross_entropy(pred_logits, target,
                                   weight=self.ce_weights.to(device))

        # --- Mask losses (so matched) ---
        if len(row_idx) > 0:
            m_pred = pred_masks[row_idx]     # (M, S)
            m_gt   = gt_masks[col_idx]       # (M, S)

            loss_bce = F.binary_cross_entropy_with_logits(m_pred, m_gt)

            pred_s = m_pred.sigmoid()
            num = 2 * (pred_s * m_gt).sum(-1)
            den = pred_s.sum(-1) + m_gt.sum(-1) + 1e-8
            loss_dice = (1.0 - num / den).mean()
        else:
            loss_bce  = torch.tensor(0.0, device=device)
            loss_dice = torch.tensor(0.0, device=device)

        total = (self.w_ce * loss_ce +
                 self.w_bce * loss_bce +
                 self.w_dice * loss_dice)

        return {"loss": total, "ce": loss_ce, "bce": loss_bce, "dice": loss_dice}


# ============================================================
# MODELO
# ============================================================
def build_model(ckpt_path=None, freeze_backbone=False):
    """
    Constroi Mask3D com 8 classes (7 BIM + no-object).
    Carrega pesos ScanNet (pula class_embed_head que muda de tamanho).
    """
    backbone_cfg = OmegaConf.create({
        "backbone": {
            "_target_": "models.Res16UNet34C",
            "config": {
                "dialations": [1, 1, 1, 1],
                "conv1_kernel_size": 5,
                "bn_momentum": 0.02,
            },
            "in_channels": 3,
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
        num_classes=NUM_MODEL_CLASSES,  # 8 (7 BIM + no-object)
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

    # Carrega pesos pre-treinados
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Detecta se eh checkpoint BIM (nosso) ou ScanNet (original)
        is_bim = "bim_classes" in ckpt

        if is_bim:
            # Checkpoint BIM: carrega tudo direto
            state = {}
            for k, v in ckpt["state_dict"].items():
                new_k = k.replace("model.", "", 1) if k.startswith("model.") else k
                state[new_k] = v
            model.load_state_dict(state, strict=True)
            print(f"Checkpoint BIM carregado (epoch {ckpt.get('epoch', '?')})")
        else:
            # Checkpoint ScanNet: pula class head (19 -> 8 classes)
            state = {}
            skipped = []
            for k, v in ckpt["state_dict"].items():
                new_k = k.replace("model.", "", 1) if k.startswith("model.") else k
                if "class_embed_head" in new_k:
                    skipped.append(new_k)
                    continue
                state[new_k] = v
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Checkpoint ScanNet carregado:")
            print(f"  {len(state)} pesos | {len(skipped)} pulados (class head)")
            print(f"  {len(missing)} missing | {len(unexpected)} unexpected")
    else:
        print("Sem checkpoint - treinando do zero!")

    if freeze_backbone:
        n_frozen = 0
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
                n_frozen += 1
        print(f"Backbone congelado: {n_frozen} parametros")

    total = sum(p.numel() for p in model.parameters())
    treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros: {total:,} total | {treinaveis:,} treinaveis")

    return model


# ============================================================
# STEP DE TREINO
# ============================================================
def forward_step(model, criterion, scene, device):
    """Forward pass + loss para uma cena."""

    coords   = torch.from_numpy(scene["coords"]).int()
    feats    = torch.from_numpy(scene["features"]).float()
    pts      = torch.from_numpy(scene["pts"]).float()
    segs     = torch.from_numpy(scene["segments"]).long()
    gt_masks = torch.from_numpy(scene["gt_masks"]).float().to(device)
    gt_cls   = torch.from_numpy(scene["gt_classes"]).long().to(device)

    # SparseTensor (batch_idx = 0)
    batch_idx = torch.zeros(len(coords), 1, dtype=torch.int32)
    coords_b  = torch.cat([batch_idx, coords], dim=1)

    sinput = ME.SparseTensor(
        features=feats.to(device),
        coordinates=coords_b.to(device),
    )

    point2segment = [segs.to(device)]
    raw_coords    = pts.to(device)

    # Forward
    outputs = model(
        sinput,
        point2segment=point2segment,
        raw_coordinates=raw_coords,
        is_eval=False,
    )

    # Extrai predicoes (mesmo tratamento do infer_ply.py)
    pred_logits = outputs["pred_logits"]
    pred_masks  = outputs["pred_masks"]

    if isinstance(pred_logits, list):
        pred_logits = pred_logits[0]
    if isinstance(pred_masks, list):
        pred_masks = pred_masks[0]
    if pred_logits.dim() == 3:
        pred_logits = pred_logits[0]
    if pred_masks.dim() == 3:
        pred_masks = pred_masks[0]

    # pred_masks pode vir (S, Q) ao inves de (Q, S) — corrige
    Q = pred_logits.shape[0]  # num_queries
    if pred_masks.shape[0] != Q:
        pred_masks = pred_masks.T  # (S, Q) -> (Q, S)

    # Binariza GT masks (>0.5 no espaco de segmentos)
    gt_masks_bin = (gt_masks > 0.5).float()

    # Loss
    loss_dict = criterion(pred_logits, pred_masks, gt_cls, gt_masks_bin)
    return loss_dict


# ============================================================
# METRICAS
# ============================================================
@torch.no_grad()
def compute_metrics(model, dataloader, criterion, device, max_scenes=50):
    """Calcula loss e metricas no validation set."""
    model.eval()
    losses = []
    correct_cls = 0
    total_cls = 0

    for i, scene in enumerate(dataloader):
        if i >= max_scenes:
            break
        try:
            loss_dict = forward_step(model, criterion, scene, device)
            losses.append(loss_dict["loss"].item())
        except Exception:
            continue

    return np.mean(losses) if losses else float("nan")


# ============================================================
# TREINO PRINCIPAL
# ============================================================
def treinar(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem:.1f} GB")

    # Dados
    data_dir = Path(args.data)
    npz_files = sorted(data_dir.glob("*.npz"))
    print(f"\n{len(npz_files)} cenas encontradas em {data_dir}")

    if not npz_files:
        print("Sem dados! Rode dataset_generator_mask3d.py primeiro.")
        return

    # Split train/val (85/15)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(npz_files))
    n_val = max(1, int(len(npz_files) * 0.15))
    train_files = [npz_files[i] for i in idx[n_val:]]
    val_files   = [npz_files[i] for i in idx[:n_val]]

    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    ds_train = BIMInstanceDataset(train_files, augment=True,  max_voxels=args.max_voxels)
    ds_val   = BIMInstanceDataset(val_files,   augment=False, max_voxels=args.max_voxels)

    dl_train = DataLoader(ds_train, batch_size=1, shuffle=True,
                          num_workers=2, collate_fn=collate_single)
    dl_val   = DataLoader(ds_val, batch_size=1, shuffle=False,
                          num_workers=2, collate_fn=collate_single)

    # Modelo
    print("\n" + "=" * 60)
    model = build_model(args.ckpt, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    # Loss
    criterion = Mask3DBIMLoss(num_classes=NUM_MODEL_CLASSES).to(device)

    # Optimizer: LR menor pro backbone, maior pro decoder
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": other_params,    "lr": args.lr},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Checkpoint dir
    ckpt_dir = Path(args.ckpt_out)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Mixed precision pra economizar VRAM
    scaler = torch.cuda.amp.GradScaler()
    use_amp = device.type == "cuda"

    print(f"\nCheckpoints: {ckpt_dir}")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | AMP: {use_amp}")
    print("=" * 60)

    best_val = float("inf")

    for epoch in range(args.start_epoch, args.epochs + 1):
        t0 = time.time()

        # ---- TRAIN ----
        model.train()
        train_losses = []
        train_ce = []
        train_bce = []
        train_dice = []

        consecutive_oom = 0
        max_consecutive_oom = 10

        for i, scene in enumerate(dl_train):
            try:
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
                loss_dict = forward_step(model, criterion, scene, device)
                loss = loss_dict["loss"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())
                train_ce.append(loss_dict["ce"].item())
                train_bce.append(loss_dict["bce"].item())
                train_dice.append(loss_dict["dice"].item())
                consecutive_oom = 0  # reset on success

                if (i + 1) % 100 == 0:
                    avg = np.mean(train_losses[-100:])
                    print(f"  [{i+1:4d}/{len(dl_train)}] "
                          f"loss={avg:.4f} "
                          f"ce={np.mean(train_ce[-100:]):.4f} "
                          f"bce={np.mean(train_bce[-100:]):.4f} "
                          f"dice={np.mean(train_dice[-100:]):.4f}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    del scene
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass
                    consecutive_oom += 1
                    print(f"  [OOM] Skip cena {i}")
                    if consecutive_oom >= max_consecutive_oom:
                        print(f"  [OOM] {max_consecutive_oom} consecutivos — abortando epoch")
                        break
                else:
                    print(f"  [ERR] Skip cena {i}: {e}")
                continue

        # ---- VALIDATION ----
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
        val_loss = compute_metrics(model, dl_val, criterion, device,
                                   max_scenes=len(dl_val))

        scheduler.step()

        avg_train = np.mean(train_losses) if train_losses else float("nan")
        dt = time.time() - t0

        print(
            f"\nEpoch {epoch:2d}/{args.epochs} | "
            f"train={avg_train:.4f} val={val_loss:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | {dt:.0f}s"
        )

        # Salva melhor
        if not np.isnan(val_loss) and val_loss < best_val:
            best_val = val_loss
            path = ckpt_dir / "best_bim.ckpt"
            try:
                gc.collect()
                torch.cuda.empty_cache()
                state = {f"model.{k}": v.cpu() for k, v in model.state_dict().items()}
                torch.save({
                    "epoch": epoch,
                    "state_dict": state,
                    "val_loss": val_loss,
                    "num_classes": NUM_MODEL_CLASSES,
                    "bim_classes": BIM_NAMES,
                }, path)
                del state
                model.to(device)
                print(f"  Salvo: {path} (val={val_loss:.4f})")
            except RuntimeError as e:
                print(f"  [WARN] Falha ao salvar best: {e}")
                model.to(device)

        # Salva a cada 5 epocas tambem
        if epoch % 5 == 0:
            path = ckpt_dir / f"epoch_{epoch:03d}.ckpt"
            try:
                gc.collect()
                torch.cuda.empty_cache()
                state = {f"model.{k}": v.cpu() for k, v in model.state_dict().items()}
                torch.save({
                    "epoch": epoch,
                    "state_dict": state,
                    "val_loss": val_loss,
                    "num_classes": NUM_MODEL_CLASSES,
                    "bim_classes": BIM_NAMES,
                }, path)
                del state
                model.to(device)
                print(f"  Salvo: {path}")
            except RuntimeError as e:
                print(f"  [WARN] Falha ao salvar epoch_{epoch:03d}: {e}")
                model.to(device)

        print()

    print(f"{'='*60}")
    print(f"Treino concluido! Melhor val_loss: {best_val:.4f}")
    print(f"Checkpoint: {ckpt_dir / 'best_bim.ckpt'}")
    print(f"{'='*60}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask3D BIM Training")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Pasta com .npz de treino")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT,
                        help="Checkpoint ScanNet pra fine-tune")
    parser.add_argument("--ckpt_out", default=DEFAULT_OUT,
                        help="Pasta pra salvar checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--start_epoch", type=int, default=1,
                        help="Epoch inicial (pra resumir treino)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Congela backbone (so treina decoder)")
    parser.add_argument("--max_voxels", type=int, default=80000,
                        help="Max voxels por cena (default 80000 pro A100, 20000 pra GPUs menores)")
    args = parser.parse_args()

    treinar(args)
