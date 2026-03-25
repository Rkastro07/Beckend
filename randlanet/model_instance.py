# -*- coding: utf-8 -*-
"""
RANDLA-NET INSTANCE — Segmentação de Instâncias BIM
=====================================================
Estende o RandLANetBIM com uma cabeça de offset (PointGroup-style).

Por ponto, o modelo prediz:
  1. semantic_logits : (B, N, 8)  — qual tipo de objeto (parede, laje, ...)
  2. offsets         : (B, N, 3)  — vetor 3D apontando para o centro da instância

Pós-processamento:
  shifted_xyz = xyz + predicted_offset
  → DBSCAN em shifted_xyz por classe → cada cluster = uma instância

Referência: PointGroup (Jiang et al., 2020) https://arxiv.org/abs/2004.01658
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import RandLANetBIM, NUM_CLASSES, SharedMLP


class RandLANetInstance(nn.Module):
    """
    RandLA-Net com cabeça semântica + cabeça de offset.

    Permite carregar pesos do checkpoint semântico já treinado
    para o backbone (transfer learning).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, d_in: int = 6):
        super().__init__()

        # ── Backbone (idêntico ao RandLANetBIM) ──────────────────────
        from model import EncoderBlock, DecoderBlock

        self.fc_in = SharedMLP([d_in, 32])

        self.enc1 = EncoderBlock(32,  64,  ratio=4)
        self.enc2 = EncoderBlock(64,  128, ratio=4)
        self.enc3 = EncoderBlock(128, 256, ratio=4)
        self.enc4 = EncoderBlock(256, 512, ratio=4)

        self.bottleneck = SharedMLP([512, 512])

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64,  64)
        self.dec1 = DecoderBlock(64,  32,  32)

        # ── Cabeça semântica (igual ao original) ─────────────────────
        self.head = nn.Sequential(
            SharedMLP([32, 64]),
            nn.Dropout(0.5),
            nn.Conv1d(64, num_classes, 1)
        )

        # ── Cabeça de offset (nova) ───────────────────────────────────
        # Prediz vetor 3D por ponto: (ponto → centro da instância)
        self.offset_head = nn.Sequential(
            nn.Conv1d(32, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 3, 1)   # 3D offset
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, 6) — xyz + normais
        returns:
            logits  : (B, N, num_classes)
            offsets : (B, N, 3)
        """
        B, N, _ = x.shape
        xyz  = x[:, :, :3].permute(0, 2, 1)   # (B, 3, N)
        feat = x.permute(0, 2, 1)              # (B, 6, N)

        f0 = self.fc_in(feat)                  # (B, 32, N)

        xyz1, f1 = self.enc1(xyz, f0,  None)
        xyz2, f2 = self.enc2(xyz1, f1, None)
        xyz3, f3 = self.enc3(xyz2, f2, None)
        xyz4, f4 = self.enc4(xyz3, f3, None)

        f4 = self.bottleneck(f4)

        f = self.dec4(f4, f3)
        f = self.dec3(f,  f2)
        f = self.dec2(f,  f1)
        f = self.dec1(f,  f0)                  # (B, 32, N)

        # Cabeça semântica
        logits  = self.head(f).permute(0, 2, 1)        # (B, N, 8)

        # Cabeça de offset
        offsets = self.offset_head(f).permute(0, 2, 1) # (B, N, 3)

        return logits, offsets

    def carregar_backbone_semantico(self, checkpoint_path: str, device: torch.device):
        """
        Carrega pesos do checkpoint semântico para o backbone.
        A cabeça de offset começa do zero (transfer learning).
        """
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Filtra só as camadas do backbone (exclui 'head')
        backbone_keys = {k: v for k, v in ckpt.items()
                         if not k.startswith('head.')}

        missing, unexpected = self.load_state_dict(backbone_keys, strict=False)

        loaded = len(backbone_keys) - len(missing)
        print(f"   Backbone carregado: {loaded}/{len(backbone_keys)} camadas")
        if missing:
            print(f"   Camadas novas (offset_head): {len(missing)}")
        return self
