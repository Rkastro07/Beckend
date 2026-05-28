# -*- coding: utf-8 -*-
"""Diagnostico CORRETO das labels do dataset"""
import numpy as np
import os
import glob
from collections import Counter

data_dir = "/tmp/mask3d_data_filtered"
files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

# LABEL_MAP correto do dataset_generator_mask3d.py:
DATASET_LABELS = {
    0: "background",
    1: "IfcWall",
    2: "IfcSlab",
    3: "IfcColumn",
    4: "IfcBeam",
    5: "IfcStair",
    6: "IfcRoof",
    7: "IfcSanitaryTerminal",
}

keyword_map = {
    "wall": "IfcWall",
    "slab": "IfcSlab",
    "column": "IfcColumn",
    "col_": "IfcColumn",
    "beam": "IfcBeam",
    "stair": "IfcStair",
    "roof": "IfcRoof",
}

print("=== DISTRIBUICAO GLOBAL (mapeamento CORRETO) ===")
global_labels = Counter()
for f in files:
    d = np.load(f)
    sem = d["semantic_labels"]
    for lbl in np.unique(sem):
        global_labels[int(lbl)] += int((sem == lbl).sum())

total_pts = sum(global_labels.values())
for lbl in sorted(global_labels.keys()):
    name = DATASET_LABELS.get(lbl, "cls_{}".format(lbl))
    count = global_labels[lbl]
    pct = 100.0 * count / total_pts
    print("  label {} = {:22s}: {:>12,} pts ({:.1f}%)".format(lbl, name, count, pct))

# Re-check filename vs label
print("\n=== FILENAME vs LABEL (mapeamento CORRETO) ===")
mismatches = 0
correct = 0
checked = 0
mismatch_examples = []

for f in files:
    d = np.load(f)
    name = os.path.basename(f).lower()
    sem = d["semantic_labels"]
    unique_labels = np.unique(sem).tolist()

    if len(unique_labels) == 1:
        actual_label = unique_labels[0]
        actual_cls = DATASET_LABELS.get(actual_label, "?")

        for kw, expected_cls in keyword_map.items():
            if kw in name:
                checked += 1
                if expected_cls == actual_cls:
                    correct += 1
                else:
                    mismatches += 1
                    if len(mismatch_examples) < 15:
                        mismatch_examples.append(
                            "  {} -> label {}={} (esperado={})".format(
                                os.path.basename(f)[:55], actual_label,
                                actual_cls, expected_cls))
                break

print("Checados: {}".format(checked))
print("Corretos: {} ({:.0f}%)".format(correct, 100 * correct / max(checked, 1)))
print("Mismatches: {} ({:.0f}%)".format(mismatches, 100 * mismatches / max(checked, 1)))
if mismatch_examples:
    print("\nExemplos de mismatch real:")
    for ex in mismatch_examples:
        print(ex)

# Detailed look at some specific files
print("\n=== AMOSTRA DE CENAS (labels corretas) ===")
import random
random.seed(42)
samples = random.sample(files, 8)
for f in samples:
    d = np.load(f)
    sem = d["semantic_labels"]
    inst = d["instance_labels"]
    fname = os.path.basename(f)
    unique = np.unique(sem).tolist()
    n_inst = len(np.unique(inst[inst >= 0]))
    parts = []
    for lbl in unique:
        name = DATASET_LABELS.get(lbl, "?")
        count = int((sem == lbl).sum())
        parts.append("{}={:,}".format(name, count))
    print("  {} | {} inst | {}".format(fname[:50], n_inst, " | ".join(parts)))

print("\n=== CONCLUSAO ===")
print("Dataset: labels 0-7 (0=bg, 1=IfcWall, ..., 7=IfcSanitaryTerminal)")
print("Treino linha 151: cls = int(sem[pt_mask][0]) - 1  -->  1-7 vira 0-6")
print("Modelo: classes 0-6 BIM + 7 no-object")
print("OFFSET CORRETO no treino!")
