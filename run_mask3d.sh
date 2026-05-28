#!/bin/bash
set -e

source /home/rafael/mask3d_env/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/usr/local/bin:/home/rafael/mask3d_env/bin
export OMP_NUM_THREADS=3

PLY_PATH="$1"
if [ -z "$PLY_PATH" ]; then
    echo "Uso: bash run_mask3d.sh <caminho_ply>"
    exit 1
fi

echo "=== Mask3D Inference ==="
echo "PLY: $PLY_PATH"

cd /home/rafael/Mask3D

python main_instance_segmentation.py \
    general.checkpoint='checkpoints/scannet/scannet_val.ckpt' \
    general.train_mode=false \
    general.eval_on_segments=true \
    general.train_on_segments=true \
    model.num_queries=150 \
    general.topk_per_image=500 \
    general.use_dbscan=true \
    general.dbscan_eps=0.95
