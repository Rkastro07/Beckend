#!/bin/bash
set -e

source /home/rafael/mask3d_env/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/usr/local/bin:/home/rafael/mask3d_env/bin
export TORCH_CUDA_ARCH_LIST="8.6"

echo "=== 1. MinkowskiEngine ==="
cd /home/rafael/Mask3D/third_party
if [ ! -d "MinkowskiEngine" ]; then
    git clone --recursive https://github.com/NVIDIA/MinkowskiEngine
fi
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas

echo "=== 2. PointNet2 ==="
cd /home/rafael/Mask3D/third_party/pointnet2
python setup.py install

echo "=== 3. Verificando ==="
python -c "import MinkowskiEngine; print(f'MinkowskiEngine OK: {MinkowskiEngine.__version__}')"
python -c "import pointnet2; print('PointNet2 OK')"

echo "=== TUDO INSTALADO ==="
