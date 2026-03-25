# -*- coding: utf-8 -*-
"""
TREINO — RandLA-Net Instance Segmentation
==========================================
Treina o RandLANetInstance com dois objetivos simultâneos:

  1. Segmentação semântica  : CrossEntropy  (qual tipo de objeto)
  2. Regressão de offset    : SmoothL1      (vetor → centro da instância)

O offset ground-truth é calculado na hora: para cada ponto, o vetor
que aponta do ponto até o centroide da sua instância.

Uso:
  python randlanet/train_instances.py
  python randlanet/train_instances.py --epochs 50 --batch 2 --lr 0.001
  python randlanet/train_instances.py --backbone checkpoints/best.pth

Referência: PointGroup (Jiang et al., 2020) https://arxiv.org/abs/2004.01658
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from model_instance import RandLANetInstance
from model import NUM_CLASSES

# ─────────────────────────────────────────────
CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

INST_DATA_DIR   = Path(__file__).parent / "data_instances"
NUM_POINTS      = 8192      # pontos por iteração
LAMBDA_OFFSET   = 1.0       # peso da loss de offset vs semântica


# =========================
# UTILIDADES DE DADO
# =========================

def listar_cenas_instancias(data_dir: Path = INST_DATA_DIR):
    """Retorna lista de .npz com array 'instances'."""
    cenas = sorted(data_dir.glob("*.npz"))
    validas = []
    for p in cenas:
        try:
            with np.load(p) as d:
                if 'instances' in d:
                    validas.append(p)
        except Exception:
            pass
    return validas


def _calcular_offsets_gt(pts_xyz: np.ndarray, inst_labels: np.ndarray) -> np.ndarray:
    """
    Calcula vetor offset ground-truth por ponto.

    Para cada ponto de foreground (inst_id > 0):
        offset = centroide_da_instância - posição_do_ponto

    Pontos de background (inst_id == 0) recebem offset (0, 0, 0).

    Args:
        pts_xyz    : (N, 3) float32
        inst_labels: (N,)   int32    (0 = background)

    Returns:
        offsets_gt : (N, 3) float32
    """
    N = len(pts_xyz)
    offsets_gt = np.zeros((N, 3), dtype=np.float32)

    unique_ids = np.unique(inst_labels)
    for inst_id in unique_ids:
        if inst_id == 0:
            continue  # background → offset zero
        mask = (inst_labels == inst_id)
        centroide = pts_xyz[mask].mean(axis=0)
        offsets_gt[mask] = centroide - pts_xyz[mask]

    return offsets_gt


# =========================
# DATASET
# =========================

class BIMInstanceDataset(Dataset):
    """
    Carrega .npz com pts(N,6), labels(N,), instances(N,).
    Devolve:
        pts_t      : (n_pts, 6)  tensor float32 — xyz + normais (normalizados)
        sem_t      : (n_pts,)    tensor int64   — label semântico 0..7
        offsets_t  : (n_pts, 3)  tensor float32 — offset GT para centroide
        inst_mask_t: (n_pts,)    tensor bool    — True = ponto de foreground
    """

    def __init__(self, cenas, n_pts: int = NUM_POINTS):
        self.cenas = cenas
        self.n_pts = n_pts
        self._cache: dict = {}

    def __len__(self):
        return len(self.cenas)

    def __getitem__(self, idx):
        # Cache para não reabrir o .npz a cada epoch
        if idx not in self._cache:
            data = np.load(self.cenas[idx])
            pts  = data['pts'].astype(np.float32)        # (N, 6)
            sem  = data['labels'].astype(np.int64)       # (N,)
            inst = data['instances'].astype(np.int32)    # (N,)
            self._cache[idx] = (pts, sem, inst)

        pts, sem, inst = self._cache[idx]
        N = len(pts)

        # ── Amostragem estratificada: tenta manter foreground ──────────
        fg_idx = np.where(inst > 0)[0]
        bg_idx = np.where(inst == 0)[0]

        n_fg = min(len(fg_idx), self.n_pts // 2)
        n_bg = self.n_pts - n_fg

        chosen_fg = np.random.choice(fg_idx, n_fg, replace=(len(fg_idx) < n_fg)) if n_fg > 0 else np.array([], dtype=np.int64)
        chosen_bg = np.random.choice(bg_idx, n_bg, replace=(len(bg_idx) < n_bg)) if n_bg > 0 and len(bg_idx) > 0 else np.array([], dtype=np.int64)

        # Se fg ou bg for pequeno demais, completa com amostragem global
        total = len(chosen_fg) + len(chosen_bg)
        if total < self.n_pts:
            extras = np.random.choice(N, self.n_pts - total, replace=True)
            chosen = np.concatenate([chosen_fg, chosen_bg, extras])
        else:
            chosen = np.concatenate([chosen_fg, chosen_bg])

        chosen = chosen[:self.n_pts]
        np.random.shuffle(chosen)

        pts_s  = pts[chosen]
        sem_s  = sem[chosen]
        inst_s = inst[chosen]

        # ── Normaliza XYZ ─────────────────────────────────────────────
        centro = pts_s[:, :3].mean(axis=0)
        scale  = np.abs(pts_s[:, :3] - centro).max() + 1e-8
        pts_s[:, :3] = (pts_s[:, :3] - centro) / scale

        # ── Offset GT (calculado APÓS normalização) ────────────────────
        offsets_gt = _calcular_offsets_gt(pts_s[:, :3], inst_s)
        offsets_gt /= (scale + 1e-8)   # escala igual à normalização XYZ

        inst_mask = (inst_s > 0)        # foreground mask

        return (
            torch.from_numpy(pts_s),
            torch.from_numpy(sem_s),
            torch.from_numpy(offsets_gt),
            torch.from_numpy(inst_mask),
        )


# =========================
# PESOS ANTI-DESEQUILÍBRIO
# =========================

def calcular_pesos_semanticos(cenas, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    contagem = np.zeros(num_classes, dtype=np.float64)
    for p in cenas:
        data = np.load(p)
        for i in range(num_classes):
            contagem[i] += int((data['labels'] == i).sum())
    total  = contagem.sum()
    pesos  = total / (contagem * num_classes + 1e-8)
    pesos /= pesos.mean()
    return torch.tensor(pesos, dtype=torch.float32)


# =========================
# LOSS COMBINADA
# =========================

class InstanceLoss(nn.Module):
    """
    loss = CrossEntropy(logits, sem_gt)
         + lambda_offset * SmoothL1(pred_offsets[fg], gt_offsets[fg])

    A loss de offset só é calculada nos pontos de foreground
    (inst_id > 0), pois o background não tem instância.
    """

    def __init__(self, class_weights: torch.Tensor, lambda_offset: float = LAMBDA_OFFSET):
        super().__init__()
        self.ce        = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean', beta=0.1)
        self.lam       = lambda_offset

    def forward(
        self,
        logits:     torch.Tensor,   # (B, N, C)
        offsets:    torch.Tensor,   # (B, N, 3)
        sem_gt:     torch.Tensor,   # (B, N)
        offsets_gt: torch.Tensor,   # (B, N, 3)
        inst_mask:  torch.Tensor,   # (B, N)  bool
    ):
        B, N, C = logits.shape

        # ── Semântica ──────────────────────────────────────────────────
        loss_sem = self.ce(
            logits.reshape(B * N, C),
            sem_gt.reshape(B * N)
        )

        # ── Offset (só foreground) ────────────────────────────────────
        fg_mask = inst_mask.reshape(B * N)
        if fg_mask.any():
            pred_off = offsets.reshape(B * N, 3)[fg_mask]
            gt_off   = offsets_gt.reshape(B * N, 3)[fg_mask]
            loss_off = self.smooth_l1(pred_off, gt_off)
        else:
            loss_off = torch.tensor(0.0, device=logits.device)

        total = loss_sem + self.lam * loss_off
        return total, loss_sem.item(), loss_off.item()


# =========================
# TREINO
# =========================

def treinar(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ── Dataset ────────────────────────────────────────────────────────
    cenas = listar_cenas_instancias()
    if not cenas:
        print("Nenhuma cena encontrada em randlanet/data_instances/")
        print("  Execute: python randlanet/run_batch_instances.py --dataset dataset/")
        return

    print(f"{len(cenas)} cenas de instancia encontradas")

    n_val = max(1, int(len(cenas) * 0.2))
    n_tre = len(cenas) - n_val
    dataset      = BIMInstanceDataset(cenas, n_pts=NUM_POINTS)
    ds_tre, ds_val = random_split(dataset, [n_tre, n_val],
                                  generator=torch.Generator().manual_seed(42))

    dl_tre = DataLoader(ds_tre, batch_size=args.batch, shuffle=True,  num_workers=0, pin_memory=(device.type == 'cuda'))
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
    print(f"  Treino: {len(ds_tre)} cenas | Validacao: {len(ds_val)} cenas")

    # ── Modelo ─────────────────────────────────────────────────────────
    model = RandLANetInstance(num_classes=NUM_CLASSES, d_in=6).to(device)

    if args.backbone and Path(args.backbone).exists():
        model.carregar_backbone_semantico(args.backbone, device)
        print(f"  Transfer learning do backbone semantico: {args.backbone}")
    elif args.checkpoint and Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Checkpoint de instancia carregado: {args.checkpoint}")
    else:
        print("  Treinando do zero (sem backbone pre-treinado)")
        backbone_default = CHECKPOINT_DIR / "best.pth"
        if backbone_default.exists():
            model.carregar_backbone_semantico(str(backbone_default), device)
            print(f"  Auto-detectado backbone semantico: {backbone_default}")

    # ── Loss + Optimizer ───────────────────────────────────────────────
    pesos_sem  = calcular_pesos_semanticos(cenas).to(device)
    criterion  = InstanceLoss(pesos_sem, lambda_offset=args.lam_offset)
    optimizer  = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    melhor_val = float('inf')
    ckpt_path  = CHECKPOINT_DIR / "best_instance.pth"

    print(f"\nIniciando treino: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    print(f"  Lambda offset: {args.lam_offset}")
    print("-" * 75)
    print(f"{'Epoch':>6} | {'L_tre':>7} {'L_val':>7} | "
          f"{'Lsem_t':>7} {'Loff_t':>7} | "
          f"{'Acc_t':>6} {'Acc_v':>6} | {'Tempo':>5}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── TREINO ────────────────────────────────────────────────────
        model.train()
        L_tre = L_sem_t = L_off_t = acc_t = 0.0

        for pts, sem_gt, off_gt, inst_mask in dl_tre:
            pts, sem_gt = pts.to(device), sem_gt.to(device)
            off_gt      = off_gt.to(device)
            inst_mask   = inst_mask.to(device)

            optimizer.zero_grad()
            logits, offsets = model(pts)                  # (B,N,C), (B,N,3)

            loss, ls, lo = criterion(logits, offsets, sem_gt, off_gt, inst_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            L_tre   += loss.item()
            L_sem_t += ls
            L_off_t += lo
            preds    = logits.argmax(dim=-1)
            acc_t   += (preds == sem_gt).float().mean().item()

        n_t   = max(len(dl_tre), 1)
        L_tre   /= n_t
        L_sem_t /= n_t
        L_off_t /= n_t
        acc_t   /= n_t

        # ── VALIDAÇÃO ─────────────────────────────────────────────────
        model.eval()
        L_val = acc_v = 0.0

        with torch.no_grad():
            for pts, sem_gt, off_gt, inst_mask in dl_val:
                pts, sem_gt = pts.to(device), sem_gt.to(device)
                off_gt      = off_gt.to(device)
                inst_mask   = inst_mask.to(device)

                logits, offsets = model(pts)
                loss, _, _ = criterion(logits, offsets, sem_gt, off_gt, inst_mask)
                L_val += loss.item()
                preds  = logits.argmax(dim=-1)
                acc_v += (preds == sem_gt).float().mean().item()

        n_v   = max(len(dl_val), 1)
        L_val /= n_v
        acc_v /= n_v
        scheduler.step()

        dt = time.time() - t0
        marker = " *" if L_val < melhor_val else ""
        print(
            f"{epoch:6d} | {L_tre:7.4f} {L_val:7.4f} | "
            f"{L_sem_t:7.4f} {L_off_t:7.4f} | "
            f"{acc_t:6.3f} {acc_v:6.3f} | {dt:5.1f}s{marker}"
        )

        if L_val < melhor_val:
            melhor_val = L_val
            torch.save(model.state_dict(), ckpt_path)

    print("-" * 75)
    print(f"Treino concluido. Melhor val_loss: {melhor_val:.4f}")
    print(f"Checkpoint: {ckpt_path}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina RandLA-Net para segmentacao de instancias BIM")
    parser.add_argument("--epochs",     type=int,   default=40,
                        help="Numero de epochs (default: 40)")
    parser.add_argument("--batch",      type=int,   default=2,
                        help="Batch size (default: 2)")
    parser.add_argument("--lr",         type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    parser.add_argument("--lam-offset", type=float, default=1.0,
                        dest="lam_offset",
                        help="Peso da loss de offset vs semantica (default: 1.0)")
    parser.add_argument("--backbone",   type=str,   default=None,
                        help="Checkpoint semantico para transfer learning (best.pth)")
    parser.add_argument("--checkpoint", type=str,   default=None,
                        help="Retomar checkpoint de instancia existente")
    args = parser.parse_args()
    treinar(args)
