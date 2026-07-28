# HELIOS++ — Plano de geração de dataset realista pra Mask3D

## Contexto: por que HELIOS++

O dataset atual de treino do Mask3D (`dataset/sintetico/`) é gerado por `dataset/gerar_sintetico.py`, que usa `Open3D.sample_points_uniformly()` na mesh do IFC.

**Problema:** esse gerador é "perfeito demais":
- Sem oclusão (gera pontos dentro de paredes fechadas)
- Densidade uniforme (130 pts/m²) — scanner real tem densidade variável
- Sem ruído
- Eixos sempre alinhados com IFC
- Pega TODAS as faces (frente e trás de parede)

O documento `docs/JORNADA_HORIZONTE.md` registra que a maioria dos bugs em scan real (Y/Z swap, paredes com só uma face de pontos, oclusão atrás de móveis) **não apareciam no sintético antigo** — exatamente por causa dessas idealizações.

**HELIOS++** é um simulador físico de LiDAR (ray tracing + occlusion + ruído gaussiano) que substitui esse gerador idealizado por algo muito mais próximo de scan real.

| Antigo (`gerar_sintetico.py`) | Novo (HELIOS++) |
|-------------------------------|------------------|
| Open3D `sample_points_uniformly` | Ray tracing físico de scanner |
| Density uniforme (130 pts/m²) | Density depende de distância/ângulo |
| **Sem oclusão** | **Oclusão real** — só pontos da face visível |
| Sem ruído | Ruído gaussiano configurável |
| Sem posição de scanner | Scanners em poses "walkable" reais |
| Não distingue frente/trás | Backface culling |

## Estado do código no repo

### Já existe (commitado)

- **`pipeline_v2/helios_wrapper.py`** — wrapper Python completo:
  - `find_helios_bin()` — procura binário em `~/miniforge/envs/helios/bin/helios`
  - `export_one_obj_per_mesh()` — exporta 1 OBJ por GUID (cada vira ScenePart distinta no HELIOS)
  - `_build_scene_xml()`, `_build_survey_xml()` — gera config XML do HELIOS
  - `scan_meshes()` — roda HELIOS++ via subprocess, parseia XYZ output, mapeia `hitObjectId → guid`
  - `compute_walkable_scanners()` — calcula poses válidas de scanner (no piso, evitando paredes)
- **`pipeline_v2/_test_helios_e2e.py`** — POC end-to-end:
  ```python
  python -m pipeline_v2._test_helios_e2e dataset/ifc/casapequena.ifc out.ply
  ```

### NÃO existe ainda

- Binário HELIOS++ (não instalado em lugar nenhum no WSL)
- `ifcopenshell` no `mask3d_env` do WSL
- Script de geração em lote substituindo `gerar_sintetico.py` pra alimentar dataset Mask3D

## Estado do ambiente local (WSL Ubuntu-22.04)

### `mask3d_env` (venv, NÃO conda)

Localização: `/home/rafael/mask3d_env/`
Tipo: `venv` Python puro (lê de `pyvenv.cfg`, `home = /usr/bin`)
Python: 3.10.12

Pacotes relevantes (já instalados):
| Pacote | Versão |
|--------|--------|
| torch | 2.1.2+cu118 (CUDA 11.8) ✅ |
| MinkowskiEngine | 0.5.4 ✅ |
| pointnet2 | 0.0.0 (custom build) ✅ |
| open3d | 0.19.0 ✅ |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| pytorch-lightning | 1.9.5 |
| torch-scatter | 2.1.2+pt21cu118 |
| **ifcopenshell** | ❌ NÃO INSTALADO — precisa pip install |

CUDA: torch reporta CUDA disponível, device True.

### HELIOS++ no WSL

- `~/miniforge/envs/helios/` — ❌ não existe
- `which helios` — ❌ não encontrado
- Nenhum binário HELIOS no sistema

## Decisão arquitetural: aproveitar o `mask3d_env`?

**SIM, em partes.** O wrapper Python roda no `mask3d_env`; o binário HELIOS++ fica isolado em conda separado.

### Por que separar wrapper e binário

- `mask3d_env` é **venv** Python puro, sem conda
- HELIOS++ é distribuído via **conda-forge** (C++ binary)
- Wrapper só precisa de `open3d + numpy + ifcopenshell` e usar `subprocess.run()` pra chamar o binário
- `subprocess` é agnóstico ao env — não importa em qual env o binário está

### Por que NÃO levar pro Colab

- Colab A100 é caro (compute units por hora)
- HELIOS++ é CPU-only (ray tracing) — não usa GPU
- Gerar dataset é one-shot: 1× geração → N× treinos
- Setup do Colab já demora ~25min; adicionar HELIOS aumentaria
- Plano: gerar local → empacotar `.tar.gz` → subir Drive → treinar Colab

## Plano de instalação

### Passo 1 — Instalar ifcopenshell no mask3d_env

```bash
/home/rafael/mask3d_env/bin/pip install ifcopenshell
```

### Passo 2 — Instalar Miniforge (conda) pro binário HELIOS++

```bash
cd ~
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge
# -b = batch (sem prompt), -p = path
```

Verificar: `ls ~/miniforge/bin/conda`

### Passo 3 — Criar env conda só pro HELIOS++

```bash
~/miniforge/bin/conda create -n helios -c conda-forge helios -y
```

Verificar: `ls ~/miniforge/envs/helios/bin/helios`

Esse é exatamente o path que `pipeline_v2/helios_wrapper.py::HELIOS_BIN_CANDIDATES[0]` procura.

### Passo 4 — Smoke test

```bash
cd /mnt/c/Users/Rafael/Desktop/Beckend
/home/rafael/mask3d_env/bin/python -m pipeline_v2._test_helios_e2e \
  dataset/ifc/casapequena.ifc \
  /tmp/casapequena_helios.ply
```

Esperado: produz PLY colorido por tipo IFC em `/tmp/`.

### Passo 5 — Avaliar qualidade

Abrir `casapequena_helios.ply` no CloudCompare ou Open3D e comparar com:
- `dataset/sintetico/casapequena_v0/cena.ply` (gerador antigo)
- Algum scan real Faro/Polycam (se tiver à mão)

Critérios:
- Tem oclusão? (parede só com pontos de uma face)
- Densidade variável? (mais perto = mais denso)
- Ruído? (pontos levemente fora da superfície)
- Mantém labels por GUID? (verificar `out["hit_guid"]`)

### Passo 6 — Gerar dataset em lote (se POC passou)

Adaptar `dataset/gerar_sintetico.py` ou criar `dataset/gerar_helios.py` que:
1. Para cada IFC em `dataset/archive/Open IFC Model Repository.../`:
2. Gera N variantes com remoções de objetos (como hoje)
3. Pra cada variante, roda `scan_meshes()` com scanners walkable
4. Salva `cena.ply` + `ifc_ref.json` (mesmo formato do `dataset_generator_mask3d.py` consome)
5. Salva `instance_labels` direto do `hit_guid` (sem precisar do bbox-matching atual)

**Tempo estimado:** HELIOS++ leva ~30s-2min por cena (depende de complexidade). Pra 129 modelos × 4 variantes = ~500 cenas → ~8-16h overnight no WSL.

### Passo 7 — Empacotar e subir pro Drive

```bash
cd /home/rafael
tar -czf mask3d_dataset_helios.tar.gz dataset_helios/
# upload manual pro Google Drive
```

### Passo 8 — Retreinar Mask3D no Colab

Notebook `Mask3D_BIM_Colab.ipynb` já está pronto — só trocar o `mask3d_dataset_bim.tar.gz` por `mask3d_dataset_helios.tar.gz`.

## Riscos / pontos de atenção

1. **Tempo de geração**: HELIOS++ é 50-100x mais lento que sampling uniforme. Dataset de 1289 cenas pode demorar 1-2 dias.
2. **Walkable scanner positions**: `compute_walkable_scanners()` precisa achar floor planes válidos. Se a mesh do IFC tem chão furado/buggy, pode posicionar scanner errado. Validar primeiro.
3. **Memória**: cada OBJ por GUID consome RAM/disco temp. IFCs com 1000+ produtos podem ter problema.
4. **Resolução do scanner**: parâmetros default do HELIOS podem precisar tuning pra simular scanner específico (Faro Focus vs Polycam vs iPhone LiDAR).
5. **Backwards compat**: o formato `.npz` do `dataset_generator_mask3d.py` precisa ser preservado pro `train_mask3d.py` funcionar sem mudanças.

## Files de referência

- `pipeline_v2/helios_wrapper.py` — wrapper completo (~430 linhas)
- `pipeline_v2/_test_helios_e2e.py` — POC executável
- `dataset/gerar_sintetico.py` — gerador antigo (substituir/manter pra A/B)
- `experiments/mask3d/dataset_generator_mask3d.py` — converte PLY → .npz Mask3D (manter intacto)
- `docs/JORNADA_HORIZONTE.md` — registro de bugs reais não previstos no sintético antigo

## Próxima ação concreta

Rodar Passos 1-4 (instalar ifcopenshell + Miniforge + helios + smoke test). Se POC produzir PLY decente, seguir pros passos 5-8.

## Estado atual do treino (contexto cruzado)

O treino Mask3D no Colab ainda está pendente — ver `handoff_mask3d_colab_session.md`. A geração HELIOS++ pode rodar em **paralelo** com a finalização do treino atual (são WSL local + Colab, máquinas separadas).
