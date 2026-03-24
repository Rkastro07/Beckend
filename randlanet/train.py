# -*- coding: utf-8 -*-
"""
FINE-TUNING RANDLA-NET PARA BIM
=================================
Uso:
  python train.py
  python train.py --epochs 50 --batch 4 --lr 0.001 --checkpoint checkpoints/best.pth

O script lê todos os .npz em randlanet/data/ e treina o modelo.
Recomendado: pelo menos 10 cenas para resultado útil.
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
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from model import RandLANetBIM, NUM_CLASSES
from dataset_generator import listar_cenas, carregar_cena

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

NUM_POINTS_POR_CENA = 8192   # pontos amostrados por iteração


# =========================
# DATASET
# =========================
class BIMDataset(Dataset):
    def __init__(self, cenas, n_pts=NUM_POINTS_POR_CENA):
        self.cenas = cenas
        self.n_pts = n_pts
        self._cache = {}

    def __len__(self):
        return len(self.cenas)

    def __getitem__(self, idx):
        if idx not in self._cache:
            pts, labels = carregar_cena(self.cenas[idx])
            self._cache[idx] = (pts, labels)
        pts, labels = self._cache[idx]

        # Amostragem aleatória
        N = len(pts)
        if N >= self.n_pts:
            chosen = np.random.choice(N, self.n_pts, replace=False)
        else:
            chosen = np.random.choice(N, self.n_pts, replace=True)

        pts_s    = pts[chosen].astype(np.float32)
        labels_s = labels[chosen].astype(np.int64)

        # Normaliza XYZ para [-1, 1]
        centro = pts_s[:, :3].mean(axis=0)
        scale  = np.abs(pts_s[:, :3] - centro).max() + 1e-8
        pts_s[:, :3] = (pts_s[:, :3] - centro) / scale

        return torch.from_numpy(pts_s), torch.from_numpy(labels_s)


# =========================
# PESOS PARA CLASSES (anti-desequilíbrio)
# =========================
def calcular_pesos_classes(cenas, num_classes=NUM_CLASSES):
    contagem = np.zeros(num_classes, dtype=np.float64)
    for p in cenas:
        _, labels = carregar_cena(p)
        for i in range(num_classes):
            contagem[i] += (labels == i).sum()
    total = contagem.sum()
    # Peso inversamente proporcional à frequência
    pesos = total / (contagem * num_classes + 1e-8)
    pesos /= pesos.mean()
    return torch.tensor(pesos, dtype=torch.float32)


# =========================
# TREINO
# =========================
def treinar(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositivo: {device}")

    cenas = listar_cenas()
    if not cenas:
        print("❌ Nenhuma cena encontrada em randlanet/data/")
        print("   Rode a análise no frontend para gerar dados de treino.")
        return

    print(f"📦 {len(cenas)} cenas encontradas")

    # Split treino/validação (80/20)
    n_val  = max(1, int(len(cenas) * 0.2))
    n_tre  = len(cenas) - n_val
    dataset = BIMDataset(cenas, n_pts=NUM_POINTS_POR_CENA)
    ds_tre, ds_val = random_split(dataset, [n_tre, n_val])

    dl_tre = DataLoader(ds_tre, batch_size=args.batch, shuffle=True,  num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=0)

    # Modelo
    model = RandLANetBIM(num_classes=NUM_CLASSES).to(device)
    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"✅ Checkpoint carregado: {args.checkpoint}")

    # Loss com pesos de classe
    pesos = calcular_pesos_classes(cenas).to(device)
    criterion = nn.CrossEntropyLoss(weight=pesos, ignore_index=-1)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    melhor_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ----- TREINO -----
        model.train()
        loss_tre = 0.0
        acc_tre  = 0.0
        for pts, labels in dl_tre:
            pts, labels = pts.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(pts)               # (B, N, C)
            B, N, C = logits.shape
            loss = criterion(logits.reshape(B * N, C), labels.reshape(B * N))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_tre += loss.item()
            preds = logits.argmax(dim=-1)
            acc_tre += (preds == labels).float().mean().item()

        loss_tre /= len(dl_tre)
        acc_tre  /= len(dl_tre)

        # ----- VALIDAÇÃO -----
        model.eval()
        loss_val = 0.0
        acc_val  = 0.0
        with torch.no_grad():
            for pts, labels in dl_val:
                pts, labels = pts.to(device), labels.to(device)
                logits = model(pts)
                B, N, C = logits.shape
                loss = criterion(logits.reshape(B * N, C), labels.reshape(B * N))
                loss_val += loss.item()
                preds = logits.argmax(dim=-1)
                acc_val += (preds == labels).float().mean().item()

        loss_val /= max(len(dl_val), 1)
        acc_val  /= max(len(dl_val), 1)
        scheduler.step()

        dt = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss tre={loss_tre:.4f} val={loss_val:.4f} | "
            f"Acc tre={acc_tre:.3f} val={acc_val:.3f} | "
            f"{dt:.1f}s"
        )

        # Salva melhor checkpoint
        if loss_val < melhor_val_loss:
            melhor_val_loss = loss_val
            ckpt = CHECKPOINT_DIR / "best.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  💾 Checkpoint salvo: {ckpt}")

    print(f"\n✅ Treino concluído. Melhor val_loss: {melhor_val_loss:.4f}")
    print(f"   Checkpoint: {CHECKPOINT_DIR / 'best.pth'}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch",      type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--checkpoint", type=str,   default=None)
    args = parser.parse_args()
    treinar(args)
