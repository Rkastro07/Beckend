#!/bin/bash
set -e

source /home/rafael/mask3d_env/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/usr/local/bin:/home/rafael/mask3d_env/bin
export OMP_NUM_THREADS=3

SCRIPT_DIR="/mnt/c/Users/Rafael/Desktop/Beckend/experiments/mask3d"
DATA_SRC="/mnt/c/Users/Rafael/Desktop/Beckend/experiments/mask3d/data/train"
DATA_DST="/tmp/mask3d_data"
CKPT_OUT="/tmp/mask3d_bim_checkpoints"

# Copia dados pro filesystem WSL (IO muito mais rapido que /mnt/c)
echo "=== Copiando dados pra /tmp (evita IO lento do /mnt/c) ==="
mkdir -p "$DATA_DST"

N_SRC=$(ls "$DATA_SRC"/*.npz 2>/dev/null | wc -l)
N_DST=$(ls "$DATA_DST"/*.npz 2>/dev/null | wc -l)

if [ "$N_SRC" -ne "$N_DST" ]; then
    echo "Copiando $N_SRC arquivos..."
    cp "$DATA_SRC"/*.npz "$DATA_DST/"
    echo "Copiado!"
else
    echo "$N_DST arquivos ja copiados, pulando."
fi

echo ""
echo "=== Mask3D BIM Training ==="
echo "Data: $DATA_DST ($N_SRC cenas)"
echo "Checkpoint out: $CKPT_OUT"
echo ""

cd /home/rafael/Mask3D

python "$SCRIPT_DIR/train_mask3d.py" \
    --data "$DATA_DST" \
    --ckpt "checkpoints/scannet/scannet_val.ckpt" \
    --ckpt_out "$CKPT_OUT" \
    --epochs 30 \
    --lr 1e-4

# Copia checkpoint de volta pro Windows
echo ""
echo "=== Copiando checkpoints de volta ==="
CKPT_WIN="/mnt/c/Users/Rafael/Desktop/Beckend/experiments/mask3d/checkpoints"
mkdir -p "$CKPT_WIN"
cp "$CKPT_OUT"/*.ckpt "$CKPT_WIN/"
echo "Checkpoints salvos em: $CKPT_WIN"
echo "PRONTO!"
