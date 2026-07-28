# Stack do Sistema BIM Superintendent

Resumo de bibliotecas, frameworks e ferramentas usados no projeto.

## Backend (Python — `app.py`)

### Web / API
- **Flask** — servidor HTTP (porta 8080)
- **Flask-CORS** — libera requisições cross-origin pro front

### BIM / IFC
- **ifcopenshell** — parser de arquivos IFC
  - `ifcopenshell.open()` — abre .ifc
  - `ifcopenshell.geom.create_shape()` — extrai mesh 3D de elementos
  - `ifcopenshell.geom.settings()` — configs (USE_WORLD_COORDS)
  - `ifcopenshell.util.placement.get_local_placement()` — matriz 4×4 local (OBB)
  - `product.is_a()` — checagem de subclasse (IfcWallStandardCase → IfcWall)

### Geometria / Nuvem de pontos
- **open3d** — manipulação 3D
  - `TriangleMesh` / `PointCloud`
  - `sample_points_uniformly()` — sampling denso da superfície
  - `read_point_cloud()` / `write_point_cloud()` — I/O PLY
  - `get_axis_aligned_bounding_box()` — AABB
  - `compute_triangle_normals()`
- **numpy** — vetores, matrizes, máscaras booleanas

### Machine Learning
- **scikit-learn**
  - `RandomForestClassifier` — modelo principal (97.9% test acc)
  - `train_test_split`, `classification_report`, `confusion_matrix`
  - `joblib` — serialização do `.pkl`
- **PyTorch** — baseline MLP (87.1% acc)
  - `nn.Sequential`, `Linear`, `ReLU`, `Dropout`
  - `DataLoader`, `TensorDataset`
  - `Adam`, `CrossEntropyLoss`
- **pandas** — manipulação de datasets CSV

### Deep Learning (instâncias)
- **RandLA-Net** — segmentação de instâncias em ponto (pasta `randlanet/`)

### Outros
- **pxr** (usd-core) — parse de arquivos USDZ
- **trimesh** — fallback de conversão mesh
- **scipy** — ICP/alinhamento

## Frontend (`bim-ai-superintendent/`)

### Framework / Build
- **React 19** + **TypeScript**
- **Vite 6** — dev server (porta 3000) e build
- **Tailwind CSS** — estilização

### 3D / Visualização
- **three.js** — render 3D core
- **@react-three/fiber** — bindings React pro three.js
- **@react-three/drei** — helpers (OrbitControls, Grid, etc.)
- **web-ifc** / **web-ifc-three** — carrega IFC direto no navegador

### UI
- **lucide-react** — ícones (Upload, FileText, Brain, Shapes, etc.)

### Estado / Dados
- React hooks (useState, useEffect, useMemo, useRef)

## Features do sistema

### Modos de análise (`AnalysisMode`)
- `bbox` — comparação geométrica (AABB + anti-leaking)
- `ai` — Random Forest com 11 features
- `both` — BBox vs AI lado a lado
- `instances` — RandLA-Net segmentação de objetos individuais

### 11 features do ML
- `completeness_r`, `is_empty`, `height_fill`
- `z_bottom_norm`, `z_top_norm`, `z_centroid_norm`
- `xy_spread_norm`, `density_area`
- `tipo` (one-hot), `eixo`, `bh`

### Pipeline de alinhamento PLY↔IFC
- Centroid matching
- Teste de identidade primeiro (barreira 1.30x)
- Correção de escala aproximada

### Multi-floor
- Sentinela `__TODOS__` no select de pavimentos
- Matching hierárquico (`is_a`) ao invés de nome exato

## Scripts auxiliares

- `dataset/gerar_ply_teste_especiais.py` — gera 54 PLYs de teste (19 edifícios × 3 estágios)
- `ml/train.py` — treino RF + MLP com split por edifício
- `usdz_to_ply.py` — pipeline USDZ → OBJ → PLY (sampling)
- `randlanet/` — treino e inferência RandLA-Net

## Formatos de arquivo

- **IFC** — modelo BIM paramétrico
- **PLY** — nuvem de pontos
- **USDZ** — scan do iPhone (via KIRI)
- **OBJ** — mesh intermediário
- **CSV** — cronograma opcional
- **JSON** — gabaritos e configs
- **PKL** — modelo RF serializado
- **PT** — checkpoint PyTorch

## Métricas atuais

- **Dataset**: 1290 samples de 129 edifícios
- **Split por edifício**: 91 train / 19 val / 19 test (sem leak)
- **RandomForest**: 97.9% acc, F1 macro 0.942
- **MLP PyTorch**: 87.1% acc (baseline)

## Portas

- Backend Flask: `8080`
- Frontend Vite: `3000`
