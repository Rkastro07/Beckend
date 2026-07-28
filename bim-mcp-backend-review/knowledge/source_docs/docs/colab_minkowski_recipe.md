# Receita: MinkowskiEngine no Colab Pro A100 (CUDA 12)

**Status:** FUNCIONA — testado em 27-May-2026, ME 0.5.4 importou OK  
**Runtime:** Colab Pro, A100 40GB, Python 3.10 (conda), PyTorch 2.1.2+cu121

---

## Problema principal

O Colab tem CUDA 12.8 no sistema mas PyTorch 2.1.2 foi compilado com cu121.  
MinkowskiEngine usa Thrust headers que mudaram na CUDA 12 — precisa de patches manuais.

---

## Passo 1 — Criar conda env Python 3.10

```python
# Célula notebook (kernel Colab padrão)
import condacolab
condacolab.install()  # kernel reinicia automaticamente
```

```python
# Célula após restart
import condacolab
condacolab.check()

import subprocess
# Remove env quebrado se existir (da tentativa anterior sem pip)
subprocess.run(["conda", "env", "remove", "-n", "me", "-y"])
# ATENÇÃO: pip precisa ser explícito — conda recente não instala por default
subprocess.run([
    "conda", "create", "-n", "me", "python=3.10",
    "pip", "setuptools=69.5.1", "ninja", "-y"
], check=True)
```

---

## Passo 2 — Instalar PyTorch 2.1.2+cu121 no env me

```bash
/opt/conda/envs/me/bin/pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Passo 3 — Instalar CUDA 12.1 toolkit (nvcc)

```bash
apt-get install -y cuda-toolkit-12-1 2>&1 | tail -3
# nvcc fica em: /usr/local/cuda-12.1/bin/nvcc
```

---

## Passo 4 — Patch no cpp_extension.py (silenciar erro de versão)

```bash
sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE/warnings.warn(CUDA_MISMATCH_MESSAGE/' \
  /opt/conda/envs/me/lib/python3.10/site-packages/torch/utils/cpp_extension.py
```

---

## Passo 5 — Clonar MinkowskiEngine e aplicar patches Thrust (issue #601)

```bash
cd /content && git clone https://github.com/NVIDIA/MinkowskiEngine.git
```

### Patch 1: `src/3rdparty/concurrent_unordered_map.cuh`
Adicionar após `#include <thrust/pair.h>`:
```cpp
#include <thrust/execution_policy.h>
```

```bash
sed -i '/#include <thrust\/pair.h>/a #include <thrust\/execution_policy.h>' \
  src/3rdparty/concurrent_unordered_map.cuh
```

### Patch 2: `src/convolution_kernel.cuh`
Adicionar após `#include <thrust/functional.h>`:
```cpp
#include <thrust/execution_policy.h>
```

```bash
sed -i '/#include <thrust\/functional.h>/a #include <thrust\/execution_policy.h>' \
  src/convolution_kernel.cuh
```

### Patch 3: `src/coordinate_map_gpu.cu`
Adicionar após `#include <thrust/sort.h>`:
```cpp
#include <thrust/unique.h>
#include <thrust/remove.h>
```

```bash
sed -i '/#include <thrust\/sort.h>/a #include <thrust\/unique.h>\n#include <thrust\/remove.h>' \
  src/coordinate_map_gpu.cu
```

### Patch 4: `src/spmm.cu` ⚠️ ATENÇÃO — anchor é `<cusparse.h>`, não `<thrust/device_vector.h>`

O arquivo NÃO tem `#include <thrust/device_vector.h>` originalmente.  
Adicionar **após** `#include <cusparse.h>`:

```bash
sed -i '/#include <cusparse.h>/a #include <thrust\/device_vector.h>\n#include <thrust\/execution_policy.h>\n#include <thrust\/sort.h>\n#include <thrust\/reduce.h>' \
  src/spmm.cu
```

Verificar que ficou correto (linhas 30-34 devem ser):
```
30:#include <cusparse.h>
31:#include <thrust/device_vector.h>
32:#include <thrust/execution_policy.h>
33:#include <thrust/sort.h>
34:#include <thrust/reduce.h>
```

---

## Passo 6 — Compilar MinkowskiEngine (~20 min no A100)

```bash
cd /content/MinkowskiEngine && \
CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=4 FORCE_CUDA=1 \
/opt/conda/envs/me/bin/python setup.py install \
--blas_include_dirs=/usr/include --blas=openblas --force_cuda 2>&1 | tail -20
echo "EXIT CODE: $?"
```

Esperado: `EXIT CODE: 0` e `Finished processing dependencies for MinkowskiEngine==0.5.4`

---

## Passo 7 — Verificar import

```bash
/opt/conda/envs/me/bin/python -c "import MinkowskiEngine as ME; print('ME version:', ME.__version__)"
# Esperado: ME version: 0.5.4
```

---

## Passo 8 — Demais dependências

```bash
/opt/conda/envs/me/bin/pip install \
  open3d omegaconf scipy hydra-core \
  pytorch-lightning==1.9.5 torchmetrics==0.11.4 wandb
```

---

## Passo 9 — Clonar Mask3D e instalar pointnet2

```bash
cd /content && git clone https://github.com/JonasSchult/Mask3D.git

cd /content/Mask3D/third_party/pointnet2 && \
CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST="8.0" \
/opt/conda/envs/me/bin/python setup.py install
```

---

## Passo 10 — Montar Google Drive (célula notebook, NÃO terminal)

```python
# Rodar como célula do notebook — abre popup OAuth
from google.colab import drive
drive.mount('/content/drive')
```

⚠️ **Não funciona do terminal** (precisa do kernel Colab/IPython).  
Se o popup for bloqueado, tentar novamente — geralmente funciona na segunda vez.

---

## Notas

- `MAX_JOBS=4` é suficiente no A100 — mais não acelera muito com ME
- `--force_cuda` força recompilação de todos os .cu files
- O tmux do Colab Terminal pode "desconectar" durante build longo — usar `tmux attach` para reconectar e ver se processo ainda roda (`0:python*` no statusbar = rodando)
- A sessão do Colab **não persiste entre restarts** — todo esse setup precisa ser refeito a cada nova sessão
- **Tempo total do setup:** ~25-30 min (ME domina com ~20 min)
