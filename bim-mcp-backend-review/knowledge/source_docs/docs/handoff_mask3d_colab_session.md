
# HANDOFF — Treino Mask3D BIM no Colab (sessão 28-May-2026)

## Contexto rápido

Estamos rodando o **Mask3D** (3D instance segmentation com sparse convolutions + transformer) para fine-tune em dados BIM sintéticos com **backbone descongelado** (39.6M params, 7 classes IFC). O ambiente local (WSL CUDA 11.8) já funciona — a missão atual é replicar isso no **Google Colab Pro A100** porque o WSL tá lento e o A100 tem 40GB VRAM.

A receita testada e funcionando tá em **`colab_minkowski_recipe.md`** (criada na sessão anterior 27-May, depois de um dia inteiro lutando com Thrust/CUDA 12 — não jogar fora!).

## Estado dos arquivos

### Repo local (commits recentes na branch `main`)
- **`e28f5fe`** — fix: `train_mask3d.py` compatível com Colab (3 mudanças):
  - `MASK3D_DIR = Path(os.environ.get("MASK3D_DIR", "/content/Mask3D"))` — antes era hardcoded `/home/rafael/Mask3D`
  - Adicionado CLI arg `--max_voxels` (default 80000 pro A100, antes era 20000 hardcoded)
  - `num_workers=0` → `num_workers=2` nos DataLoaders
- **Notebook NÃO commitado ainda**: `experiments/mask3d/Mask3D_BIM_Colab.ipynb` (versão limpa que criei nessa sessão, ainda no working tree)

### Google Drive (raiz do MyDrive) — confirmado por screenshot
| Arquivo | Tamanho |
|---------|---------|
| `train_mask3d.py` | 25 KB ✅ |
| `mask3d_dataset_bim.tar.gz` | 2.12 GB ✅ |
| `scannet_val.ckpt` | 151.4 MB ✅ |
| `treino/` (pasta vazia pra output) | — |
| `Colab Notebooks/Mask3D_BIM_Colab.ipynb` | 11 KB ✅ |

### Notebook no Colab (aberto)
URL: `https://colab.research.google.com/drive/14mPt4RF372sdzTHduPO3Gq9RAOd1-Wu5`
Runtime: A100 (Python 3) conectado
Drive: autorizado (popup aceito)

## O que rolou nessa sessão (ordem cronológica)

1. **Commitei o fix do `train_mask3d.py`** (commit `e28f5fe`) — 3 mudanças pra rodar no Colab.

2. **Criei o notebook `Mask3D_BIM_Colab.ipynb`** seguindo a receita de `colab_minkowski_recipe.md`. 16 células: condacolab → env conda → PyTorch → CUDA → patches Thrust → compila ME → deps → Mask3D/PointNet2 → mount Drive → extrai dataset → treino.

3. **Subiu o notebook + train_mask3d.py pro Drive** (user fez).

4. **Comecei a rodar no Colab**:
   - **Célula 1** (`condacolab.install()`): ✅ rodou, kernel reiniciou (esperado).
   - **Célula 2** (criar env `me`): ❌ FALHOU 2x.
     - 1ª tentativa: `--no-banner` não é flag válida em conda recente → erro `non-zero exit status 2`.
     - 2ª tentativa: removi `--no-banner` → célula passou com `[2] ✓ 33s` e print "✅ Env 'me' criado com Python 3.10", **MAS o env foi criado sem `pip`**.
   - **Células 3-10** rodadas via "Executar célula e abaixo": ❌ todas cascadearam:
     - Cell 3 (PyTorch): `/opt/conda/envs/me/bin/pip: No such file or directory`
     - Cell 4 (CUDA): pegou interrupt `^C` no meio, nvcc não foi instalado
     - Cell 5 (patch cpp_extension): `sed: can't read .../torch/utils/cpp_extension.py: No such file` (porque torch nem instalou)
     - Cells 6-10: todas falharam por falta de python/pip no env `me`
   - **Cell 11** (Drive mount): popup OAuth apareceu, **usuário autorizou** ✅
   - Interrompi tudo com Ctrl+M I.

5. **Tentei corrigir a célula 2** editando direto no browser pra:
   - Adicionar `subprocess.run(["conda", "env", "remove", "-n", "me", "-y"])` antes
   - Incluir `"pip"` no `conda create`
   - **Falhou com `IndentationError`** — typing pelo browser bagunçou indentação.

6. **Usuário pediu parar** e fazer esse handoff.

## O fix exato que precisa ser aplicado

**Editar célula 2 do notebook no Colab.** Conteúdo atual (quebrado):

```python
import condacolab
condacolab.check()

import subprocess
subprocess.run([
    "conda", "create", "-n", "me", "python=3.10",
    "setuptools=69.5.1", "ninja", "-y"
], check=True)
print("\n✅ Env 'me' criado com Python 3.10")
```

Trocar por:

```python
import condacolab
condacolab.check()

import subprocess
# Remove env quebrado se existir (da tentativa anterior sem pip)
subprocess.run(["conda", "env", "remove", "-n", "me", "-y"])
# Cria env com pip incluido
subprocess.run([
    "conda", "create", "-n", "me", "python=3.10",
    "pip", "setuptools=69.5.1", "ninja", "-y"
], check=True)
print("\n✅ Env 'me' criado com Python 3.10 + pip")
```

**Por que `pip` é necessário:** no Colab atual (28-May), `conda create` com só `python=3.10 setuptools ninja` NÃO instala pip por default. A receita antiga (27-May) deve ter incluído pip implicitamente em alguma versão anterior do conda — mas hoje precisa explícito.

**Também atualizar a receita** (`colab_minkowski_recipe.md`) pra incluir `pip` na lista — pra não cair nessa armadilha de novo.

## Como continuar do ponto atual

1. **Abrir o notebook**: `https://colab.research.google.com/drive/14mPt4RF372sdzTHduPO3Gq9RAOd1-Wu5`
2. **Verificar runtime A100 ainda conectado** (canto inferior direito)
3. **Editar célula 2** com o código acima — usar JavaScript pelo browser tool é mais confiável que digitar (typing quebra indentação):
   ```javascript
   // pseudocódigo — pegar a cell por seletor e setar .CodeMirror.value
   ```
   OU pedir pro usuário editar manualmente (mais simples).
4. **Rodar célula 2** (Shift+Enter) — deve demorar ~30-60s (conda remove + create)
5. **Rodar resto via "Executar célula e abaixo"** (Ctrl+F10) — vai compilar ME (~20 min), instalar deps, montar Drive (já autorizado), extrair dataset, e começar o treino
6. **Cell 11 (Drive mount)**: já foi autorizado, não vai pedir popup de novo
7. **Cell 14 (TREINO)**: deve começar com `--epochs 50 --lr 1e-4 --max_voxels 80000`

## Comando de treino esperado

```bash
/opt/conda/envs/me/bin/python /content/train_mask3d.py \
    --data /content/mask3d_data \
    --ckpt /content/drive/MyDrive/scannet_val.ckpt \
    --ckpt_out /content/drive/MyDrive/treino \
    --epochs 50 \
    --lr 1e-4 \
    --max_voxels 80000
```

LR diferenciado: backbone 1e-5, decoder 1e-4 (já implementado em `train_mask3d.py`).

## Regras importantes do projeto

- **Não fazer mudanças no código sem OK explícito do usuário**
- **Quando o usuário levanta dúvida no meio, a dúvida tem prioridade**
- **Receita do Colab** está em `colab_minkowski_recipe.md` — sagrada, foi um dia de trabalho

## TODO pendente

- [x] Aplicar fix da célula 2 do notebook ← FEITO (sessão 28-May-2026, segunda parte)
- [x] Atualizar `colab_minkowski_recipe.md` adicionando `pip` na lista do conda create ← FEITO
- [ ] Fazer upload do notebook atualizado pro Drive (substituir o existente)
- [ ] Rodar setup completo no Colab (~25 min)
- [ ] Iniciar treino (50 epochs)
- [ ] Commitar o notebook `Mask3D_BIM_Colab.ipynb`
