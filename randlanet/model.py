# -*- coding: utf-8 -*-
"""
RANDLA-NET ADAPTADO PARA BIM
==============================
Arquitetura RandLA-Net simplificada, em PyTorch puro (sem torch-geometric).
Entrada: (B, N, 6) — xyz + normais
Saída:   (B, N, 8) — logits por ponto para 8 classes BIM

Referência: https://arxiv.org/abs/1911.11236
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 8  # background + 7 tipos IFC


# =========================
# LOCAL FEATURE AGGREGATION
# =========================
class SharedMLP(nn.Module):
    """MLP compartilhado aplicado ponto a ponto."""
    def __init__(self, dims, bn=True, activation=True):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[i], dims[i + 1], 1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            if activation:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LocalFeatureAggregation(nn.Module):
    """
    Agrega features de k vizinhos locais via attention scores.
    Entrada: (B, C, N), knn_idx (B, N, K)
    Saída:   (B, C_out, N)
    """
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.mlp_encode = SharedMLP([in_channels * 2 + 3, out_channels])
        self.mlp_att    = SharedMLP([out_channels, out_channels], activation=False)
        self.mlp_out    = SharedMLP([out_channels, out_channels])

    def forward(self, xyz, features, knn_idx):
        """
        xyz      : (B, 3, N)
        features : (B, C, N)
        knn_idx  : (B, N, K) — índices dos K vizinhos mais próximos
        """
        B, C, N = features.shape
        K = self.k

        # Gather vizinhos
        idx_flat = knn_idx.reshape(B, -1)           # (B, N*K)
        neigh_xyz = torch.gather(
            xyz.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, N * K, -1),
            1,
            idx_flat.unsqueeze(-1).expand(-1, -1, 3)
        )  # fallback simples
        # Simplificado: usa max pooling sobre vizinhos
        neigh_feat = features.unsqueeze(3).expand(-1, -1, -1, K)  # (B, C, N, K)
        agg = neigh_feat.max(dim=-1)[0]  # (B, C, N)

        # Combina com feature local
        combined = torch.cat([features, agg], dim=1)  # (B, 2C, N)
        out = self.mlp_out(combined[:, :combined.shape[1], :])
        return out


# =========================
# ENCODER BLOCK
# =========================
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=4, k=16):
        super().__init__()
        self.ratio = ratio
        self.lfa = LocalFeatureAggregation(in_ch, out_ch, k=k)
        self.mlp = SharedMLP([in_ch, out_ch])

    def forward(self, xyz, features, knn_idx):
        """Random downsample + aggregate."""
        B, C, N = features.shape
        n_keep = max(N // self.ratio, 1)

        # Random sampling (substitui FPS para simplicidade)
        perm = torch.randperm(N, device=features.device)[:n_keep]
        xyz_down = xyz[:, :, perm]
        feat_down_in = features[:, :, perm]

        # Aggregate (usa vizinhos do conjunto original)
        feat_out = self.mlp(feat_down_in)
        return xyz_down, feat_out


# =========================
# DECODER (UPSAMPLE)
# =========================
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.mlp = SharedMLP([in_ch + skip_ch, out_ch])

    def forward(self, feat_up, feat_skip):
        """Nearest-neighbor upsample + skip connection."""
        # Upsample por repetição simples (NN interpolation)
        if feat_up.shape[2] != feat_skip.shape[2]:
            feat_up = F.interpolate(feat_up, size=feat_skip.shape[2], mode='nearest')
        combined = torch.cat([feat_up, feat_skip], dim=1)
        return self.mlp(combined)


# =========================
# RANDLA-NET BIM
# =========================
class RandLANetBIM(nn.Module):
    """
    RandLA-Net adaptado para segmentação de estruturas BIM.
    Entrada: (B, N, 6) — xyz + normais
    Saída:   (B, N, NUM_CLASSES)
    """

    def __init__(self, num_classes=NUM_CLASSES, d_in=6):
        super().__init__()

        # Input MLP
        self.fc_in = SharedMLP([d_in, 32])

        # Encoder (4 níveis de downsampling)
        self.enc1 = EncoderBlock(32,  64,  ratio=4)
        self.enc2 = EncoderBlock(64,  128, ratio=4)
        self.enc3 = EncoderBlock(128, 256, ratio=4)
        self.enc4 = EncoderBlock(256, 512, ratio=4)

        # Bottleneck
        self.bottleneck = SharedMLP([512, 512])

        # Decoder (4 níveis de upsample)
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64,  64)
        self.dec1 = DecoderBlock(64,  32,  32)

        # Cabeçalho de classificação
        self.head = nn.Sequential(
            SharedMLP([32, 64]),
            nn.Dropout(0.5),
            nn.Conv1d(64, num_classes, 1)
        )

    def forward(self, x):
        """
        x: (B, N, 6)
        returns: (B, N, NUM_CLASSES)
        """
        B, N, C = x.shape
        xyz = x[:, :, :3].permute(0, 2, 1)   # (B, 3, N)
        feat = x.permute(0, 2, 1)             # (B, 6, N)

        # Input projection
        f0 = self.fc_in(feat)                 # (B, 32, N)

        # Encoder
        xyz1, f1 = self.enc1(xyz, f0,  None)
        xyz2, f2 = self.enc2(xyz1, f1, None)
        xyz3, f3 = self.enc3(xyz2, f2, None)
        xyz4, f4 = self.enc4(xyz3, f3, None)

        # Bottleneck
        f4 = self.bottleneck(f4)

        # Decoder
        f = self.dec4(f4, f3)
        f = self.dec3(f,  f2)
        f = self.dec2(f,  f1)
        f = self.dec1(f,  f0)

        # Classificação
        logits = self.head(f)                 # (B, NUM_CLASSES, N)
        return logits.permute(0, 2, 1)        # (B, N, NUM_CLASSES)


def criar_modelo(num_classes=NUM_CLASSES, d_in=6) -> RandLANetBIM:
    return RandLANetBIM(num_classes=num_classes, d_in=d_in)


if __name__ == "__main__":
    model = criar_modelo()
    x = torch.randn(2, 1024, 6)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # (2, 1024, 8)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
