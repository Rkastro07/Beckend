#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIM Status Prediction - Training Pipeline
==========================================
Treina um classificador (COMPLETO / PARCIAL / AUSENTE) por objeto IFC
usando features extraidas da nuvem de pontos + metadados IFC.

Split por edificio (nao por sample) para evitar vazamento de dados.

Uso:
  python ml/train.py
  python ml/train.py --epochs 100 --lr 0.001
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("pip install open3d")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA else "cpu")
except ImportError:
    sys.exit("pip install torch")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_sample_weight
except ImportError:
    sys.exit("pip install scikit-learn")

# ================================================================
# CONFIG
# ================================================================

SINTETICO_DIR = Path(r"C:\Users\Rafael\Desktop\Beckend\dataset\sintetico")
OUTPUT_DIR    = Path(r"C:\Users\Rafael\Desktop\Beckend\ml\models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP  = {"COMPLETO": 0, "PARCIAL": 1, "AUSENTE": 2}
LABEL_NAMES = ["COMPLETO", "PARCIAL", "AUSENTE"]

TIPO_MAP = {
    "IfcWall":   0,
    "IfcSlab":   1,
    "IfcColumn": 2,
    "IfcBeam":   3,
    "IfcStair":  4,
    "IfcRoof":   5,
    "IfcDoor":   6,
    "IfcWindow": 7,
}
EIXO_MAP = {"z_up": 0, "horiz_longest": 1, "length_axis": 2}

BBOX_MARGIN = 0.05   # metros — margem ao filtrar pts dentro da bbox
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================================================================
# FEATURE EXTRACTION
# ================================================================

def extrair_features_objeto(pts_cena, obj_label, obj_ifc):
    """
    Extrai ~18 features por objeto combinando nuvem de pontos + IFC.

    Features de ponto (o que o sensor ve):
      n_pts           — pontos encontrados na bbox
      completeness_r  — n_pts / esperado (area * 130)
      is_empty        — 1 se nenhum ponto
      height_fill     — (zmax-zmin)_pts / bbox_h
      z_bottom_norm   — onde os pontos comeam (em % da altura)
      z_top_norm      — onde os pontos terminam
      centroid_z_norm — centroide Z normalizado
      xy_spread_norm  — desvio padrao XY normalizado pelo bbox
      density_obs     — n_pts / volume_bbox

    Features IFC (sempre conhecidas):
      tipo_enc        — tipo de elemento (0-6)
      eixo_enc        — eixo construtivo (0-2)
      bbox_w          — largura (X)
      bbox_d          — profundidade (Y)
      bbox_h          — altura (Z)
      area            — area de superficie
      volume_bbox     — volume do bbox
      aspect_hw       — h/w (proporcao)
      aspect_hd       — h/d (proporcao)
    """
    bbox   = obj_ifc['bbox']
    area   = obj_ifc.get('area', 1.0)
    vol    = obj_ifc.get('volume_bbox', 1.0)
    tipo   = TIPO_MAP.get(obj_ifc.get('tipo', ''), 0)
    eixo   = EIXO_MAP.get(obj_label.get('eixo_construtivo', 'z_up'), 0)

    bw = bbox['xmax'] - bbox['xmin']
    bd = bbox['ymax'] - bbox['ymin']
    bh = bbox['zmax'] - bbox['zmin']
    bh = max(bh, 1e-6)
    bw = max(bw, 1e-6)
    bd = max(bd, 1e-6)

    expected_pts = max(area * 130, 1)

    # Filtra pontos dentro da bbox com margem
    if pts_cena is not None and len(pts_cena) > 0:
        m = BBOX_MARGIN
        mask = (
            (pts_cena[:, 0] >= bbox['xmin'] - m) & (pts_cena[:, 0] <= bbox['xmax'] + m) &
            (pts_cena[:, 1] >= bbox['ymin'] - m) & (pts_cena[:, 1] <= bbox['ymax'] + m) &
            (pts_cena[:, 2] >= bbox['zmin'] - m) & (pts_cena[:, 2] <= bbox['zmax'] + m)
        )
        pts_obj = pts_cena[mask]
    else:
        pts_obj = np.zeros((0, 3), dtype=np.float32)

    n_pts = len(pts_obj)

    if n_pts > 0:
        zpts    = pts_obj[:, 2]
        zmin_p  = float(zpts.min())
        zmax_p  = float(zpts.max())
        zmean_p = float(zpts.mean())

        height_fill    = (zmax_p - zmin_p) / bh
        z_bottom_norm  = (zmin_p - bbox['zmin']) / bh
        z_top_norm     = (zmax_p - bbox['zmin']) / bh
        centroid_z_norm = (zmean_p - bbox['zmin']) / bh

        xy_std = float(pts_obj[:, :2].std())
        xy_spread_norm = xy_std / max(bw, bd)

        density_obs    = n_pts / max(vol, 1e-6)
        completeness_r = n_pts / expected_pts
        is_empty       = 0.0
    else:
        height_fill     = 0.0
        z_bottom_norm   = 0.0
        z_top_norm      = 0.0
        centroid_z_norm = 0.0
        xy_spread_norm  = 0.0
        density_obs     = 0.0
        completeness_r  = 0.0
        is_empty        = 1.0

    # density normalizada pela area (mais estavel que n_pts/vol)
    density_area = n_pts / max(area, 1e-6)

    feat = np.array([
        # observaveis (os que realmente indicam presenca/ausencia)
        completeness_r,
        is_empty,
        height_fill,
        z_bottom_norm,
        z_top_norm,
        centroid_z_norm,
        xy_spread_norm,
        density_area,
        # IFC (contexto do elemento)
        float(tipo),
        float(eixo),
        bh,
    ], dtype=np.float32)

    return feat


def carregar_dataset():
    """
    Varre todos os samples do dataset sintetico e extrai features por objeto.
    Retorna X (N, 18), y (N,), grupos (N,) com nome do IFC para split.
    """
    samples = sorted(SINTETICO_DIR.iterdir())
    if not samples:
        sys.exit(f"Dataset vazio: {SINTETICO_DIR}")

    X_list, y_list, grupos = [], [], []
    stats = {"COMPLETO": 0, "PARCIAL": 0, "AUSENTE": 0}

    print(f"Carregando {len(samples)} samples...")
    t0 = time.time()

    for i, sample_dir in enumerate(samples):
        lf  = sample_dir / "labels.json"
        rf  = sample_dir / "ifc_ref.json"
        ply = sample_dir / "cena.ply"
        if not (lf.exists() and rf.exists() and ply.exists()):
            continue

        labels  = json.load(open(lf, encoding='utf-8'))
        ifc_ref = json.load(open(rf, encoding='utf-8'))

        # Carrega nuvem de pontos uma vez por sample
        pcd = o3d.io.read_point_cloud(str(ply))
        pts = np.asarray(pcd.points, dtype=np.float32) if len(pcd.points) > 0 else None

        # Nome do edificio = prefixo antes do _v00 etc.
        stem = sample_dir.name
        edificio = "_".join(stem.split("_")[:-1])  # remove _v00

        for guid, obj_label in labels['objetos'].items():
            obj_ifc = ifc_ref.get(guid)
            if obj_ifc is None:
                continue

            feat  = extrair_features_objeto(pts, obj_label, obj_ifc)
            label = LABEL_MAP[obj_label['status']]

            X_list.append(feat)
            y_list.append(label)
            grupos.append(edificio)
            stats[obj_label['status']] += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} samples processados...")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    grupos = np.array(grupos)

    dt = time.time() - t0
    total = sum(stats.values())
    print(f"\nDataset carregado em {dt:.1f}s")
    print(f"  Total objetos : {total:,}")
    for s, c in stats.items():
        print(f"  {s:<12}: {c:,}  ({c/total*100:.1f}%)")

    return X, y, grupos


def split_por_edificio(X, y, grupos, val_frac=0.15, test_frac=0.15):
    """
    Divide pelo edificio — garante que o mesmo IFC nao aparece em
    treino e teste (evita vazamento de dados entre variantes).
    """
    edificios = sorted(set(grupos))
    random.shuffle(edificios)

    n = len(edificios)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))

    test_edif  = set(edificios[:n_test])
    val_edif   = set(edificios[n_test:n_test + n_val])
    train_edif = set(edificios[n_test + n_val:])

    idx_train = np.where(np.isin(grupos, list(train_edif)))[0]
    idx_val   = np.where(np.isin(grupos, list(val_edif)))[0]
    idx_test  = np.where(np.isin(grupos, list(test_edif)))[0]

    print(f"\nSplit por edificio:")
    print(f"  Treino : {len(train_edif)} edificios  |  {len(idx_train):,} objetos")
    print(f"  Val    : {len(val_edif)} edificios  |  {len(idx_val):,} objetos")
    print(f"  Teste  : {len(test_edif)} edificios  |  {len(idx_test):,} objetos")

    print(f"\n  Edificios no conjunto de TESTE:")
    for e in sorted(test_edif):
        print(f"    - {e}")

    return (X[idx_train], y[idx_train],
            X[idx_val],   y[idx_val],
            X[idx_test],  y[idx_test])


# ================================================================
# MODELO PYTORCH
# ================================================================

class StatusMLP(nn.Module):
    """
    MLP para classificacao de status de elemento BIM.
    Input: 18 features (observaveis + IFC)
    Output: 3 classes (COMPLETO / PARCIAL / AUSENTE)
    """
    def __init__(self, n_features=18, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)


def treinar_mlp(X_tr, y_tr, X_val, y_val, scaler, args):
    print("\n" + "=" * 60)
    print("TREINANDO MLP (PyTorch)")
    print(f"  Device : {DEVICE}")
    print("=" * 60)

    # Normaliza
    X_tr_s  = scaler.transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    # Weighted sampler para balancear PARCIAL (minoria)
    counts  = np.bincount(y_tr, minlength=3)
    weights = 1.0 / (counts + 1)
    sample_weights = weights[y_tr]
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights).float(),
        num_samples=len(y_tr),
        replacement=True
    )

    # Class weights para loss
    class_w = torch.tensor(
        [1.0 / (counts[i] + 1) for i in range(3)],
        dtype=torch.float32
    ).to(DEVICE)
    class_w = class_w / class_w.sum() * 3  # normaliza

    ds_tr  = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr))
    ds_val = TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val))

    dl_tr  = DataLoader(ds_tr, batch_size=512, sampler=sampler)
    dl_val = DataLoader(ds_val, batch_size=1024, shuffle=False)

    model = StatusMLP(n_features=X_tr.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit  = nn.CrossEntropyLoss(weight=class_w)

    best_val_acc = 0.0
    best_state   = None
    patience_cnt = 0
    PATIENCE     = 20

    for epoch in range(1, args.epochs + 1):
        # Treino
        model.train()
        loss_sum = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out  = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        sched.step()

        # Validacao
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds   = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += yb.size(0)

        val_acc = correct / total * 100

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = opt.param_groups[0]['lr']
            print(f"  Epoch {epoch:>4}/{args.epochs}  "
                  f"loss={loss_sum/len(dl_tr):.4f}  "
                  f"val_acc={val_acc:.2f}%  "
                  f"best={best_val_acc:.2f}%  "
                  f"lr={lr_now:.5f}")

        if patience_cnt >= PATIENCE:
            print(f"  Early stop na epoch {epoch} (sem melhora por {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    print(f"\n  Melhor val_acc: {best_val_acc:.2f}%")
    return model


def avaliar(modelo_rf, modelo_mlp, scaler, X_te, y_te):
    print("\n" + "=" * 60)
    print("AVALIACAO NO CONJUNTO DE TESTE")
    print("=" * 60)

    # Random Forest
    y_pred_rf = modelo_rf.predict(X_te)
    print("\n--- Random Forest ---")
    print(classification_report(y_te, y_pred_rf,
                                  target_names=LABEL_NAMES, digits=3))
    cm = confusion_matrix(y_te, y_pred_rf)
    print("Matriz de confusao (linhas=real, colunas=previsto):")
    print("            COMPLETO  PARCIAL  AUSENTE")
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:<12}  {row[0]:>7}  {row[1]:>7}  {row[2]:>7}")

    # MLP PyTorch
    modelo_mlp.eval()
    X_te_s = scaler.transform(X_te).astype(np.float32)
    with torch.no_grad():
        logits  = modelo_mlp(torch.from_numpy(X_te_s).to(DEVICE))
        y_pred_mlp = logits.argmax(dim=1).cpu().numpy()

    print("\n--- MLP PyTorch ---")
    print(classification_report(y_te, y_pred_mlp,
                                  target_names=LABEL_NAMES, digits=3))
    cm = confusion_matrix(y_te, y_pred_mlp)
    print("Matriz de confusao (linhas=real, colunas=previsto):")
    print("            COMPLETO  PARCIAL  AUSENTE")
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:<12}  {row[0]:>7}  {row[1]:>7}  {row[2]:>7}")


def salvar_modelos(modelo_rf, modelo_mlp, scaler, X_tr):
    n_feat = X_tr.shape[1]

    # Random Forest
    rf_path = OUTPUT_DIR / "random_forest.pkl"
    with open(rf_path, 'wb') as f:
        pickle.dump({'model': modelo_rf, 'scaler': scaler}, f)
    print(f"\nRandom Forest salvo: {rf_path}")

    # MLP PyTorch
    mlp_path = OUTPUT_DIR / "mlp_bim.pt"
    torch.save({
        'model_state': modelo_mlp.state_dict(),
        'n_features':  n_feat,
        'label_map':   LABEL_MAP,
        'label_names': LABEL_NAMES,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std':  scaler.scale_.tolist(),
    }, mlp_path)
    print(f"MLP PyTorch salvo : {mlp_path}")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr',     type=float, default=5e-4)
    args = parser.parse_args()

    print("=" * 60)
    print("BIM STATUS PREDICTION - TRAINING")
    print(f"  Device : {DEVICE}  {'(GPU)' if CUDA else '(CPU)'}")
    print(f"  Dataset: {SINTETICO_DIR}")
    print("=" * 60)

    # 1. Carrega features
    X, y, grupos = carregar_dataset()

    # 2. Split por edificio
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_por_edificio(X, y, grupos)

    # 3. Normaliza (fit so no treino)
    scaler = StandardScaler()
    scaler.fit(X_tr)

    # 4. Random Forest (baseline rapido)
    print("\n" + "=" * 60)
    print("TREINANDO RANDOM FOREST (baseline)")
    print("=" * 60)
    t0 = time.time()
    sw = compute_sample_weight('balanced', y_tr)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED,
        class_weight='balanced',
    )
    rf.fit(X_tr, y_tr)
    print(f"  Treinado em {time.time()-t0:.1f}s")

    val_pred_rf = rf.predict(X_val)
    val_acc_rf  = (val_pred_rf == y_val).mean() * 100
    print(f"  Val accuracy: {val_acc_rf:.2f}%")

    # Feature importance
    feat_names = [
        'n_pts','completeness_r','is_empty','height_fill',
        'z_bottom_norm','z_top_norm','centroid_z_norm',
        'xy_spread_norm','density_obs',
        'tipo','eixo','bbox_w','bbox_d','bbox_h',
        'area','volume','aspect_hw','aspect_hd'
    ]
    imp = rf.feature_importances_
    top = sorted(zip(feat_names, imp), key=lambda x: -x[1])
    print("\n  Top features:")
    for name, v in top[:8]:
        bar = '#' * int(v * 60)
        print(f"    {name:<18} {v:.3f}  {bar}")

    # 5. MLP PyTorch
    mlp = treinar_mlp(X_tr, y_tr, X_val, y_val, scaler, args)

    # 6. Avaliacao final
    avaliar(rf, mlp, scaler, X_te, y_te)

    # 7. Salva
    salvar_modelos(rf, mlp, scaler, X_tr)

    print("\nPronto!")


if __name__ == "__main__":
    main()
