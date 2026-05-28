#!/bin/bash
set -e
source /home/rafael/mask3d_env/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/usr/local/bin:/home/rafael/mask3d_env/bin
export OMP_NUM_THREADS=3

cd /home/rafael/Mask3D
# Copia PLY pro WSL pra evitar problema de IO no /mnt/c
PLY_SRC="/mnt/c/Users/Rafael/Desktop/Beckend/dataset/ply teste/REF_RHPC_REV00_sintetico.ply"
PLY_DST="/tmp/test_scan.ply"
OUT_DIR="/mnt/c/Users/Rafael/Downloads/mask3d_output"

echo "Copiando PLY pro WSL..."
cp "$PLY_SRC" "$PLY_DST"

python /mnt/c/Users/Rafael/Desktop/Beckend/experiments/mask3d/infer_ply.py \
    "$PLY_DST" \
    --out "$OUT_DIR" \
    --voxel 0.02
