# 🏗️ BIM Analysis API - Backend

This is the core backend service for the **BIM Analysis App**, submitted to the **Google AI Studio - Vibe Coding with Gemini 3 Pro Hackathon**.

It is a robust Python/Flask application capable of auditing construction sites by comparing **As-Planned (IFC)** models against **As-Built (PLY)** point clouds.

## 🔥 Key Features

### 1. Hybrid Brute Force Alignment (Zero-Center)
Solves the "floating model" problem where scan coordinates differ from BIM coordinates.
* **Automatic Outlier Removal:** Uses statistical analysis to clean sensor noise.
* **Zero-Center Normalization:** Temporarily moves both models to `(0,0,0)` to find the optimal rotation and translation match.
* **Identity-first test:** Tries identity transform first, only replaces if another candidate is 1.30x better (avoids distorting already-aligned scans).

### 2. Anti-Leaking Protection
Prevents false positives in progress monitoring.
* **T-Junction Detection:** Identifies wall intersections to shrink bounding boxes and avoid overlapping points.
* **Dynamic Floor/Ceiling Cuts:** Automatically filters out points from slabs and roofs that "leak" into wall bounding boxes.

### 3. Machine Learning Pipeline (COMPLETO / PARCIAL / AUSENTE)
* **Random Forest classifier** (scikit-learn): 97.9% test accuracy, F1 macro 0.942
* **MLP baseline** (PyTorch): 87.1% accuracy
* **11 features** per object combining point cloud observations (completeness, height fill, Z distribution, density) + IFC metadata (type, axis, height)
* **Split by building** to prevent variant leakage across train/val/test

### 4. Instance Segmentation (RandLA-Net)
Per-point instance labels to separate individual objects in the scan.

### 5. Multi-Floor Mode
Analyze all floors at once via `__TODOS__` sentinel, with subclass-aware IFC type matching (`IfcWallStandardCase` → `IfcWall`).

### 6. AI-Powered Reporting
Integrates with **DeepSeek LLM** to generate executive summaries based on the raw technical data extracted from the scan.

---

## 🛠️ Tech Stack

See [`STACK.md`](./STACK.md) for the full list.

* **Python 3.10+** (backend)
* **Flask** + **Flask-CORS** (API server — port 8080)
* **Open3D** (point cloud processing & alignment)
* **IfcOpenShell** (BIM geometry extraction)
* **NumPy** (vector math & matrix transformations)
* **scikit-learn** + **PyTorch** (ML models)
* **React 19** + **TypeScript** + **Vite** (frontend — port 3000)
* **three.js** + **@react-three/fiber** + **web-ifc** (3D viewer)

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Rkastro07/Beckend.git
cd Beckend
```

### 2. Backend setup
```bash
pip install -r requirements.txt     # or install manually
python app.py                       # starts on http://127.0.0.1:8080
```

### 3. Frontend setup
```bash
cd bim-ai-superintendent
npm install
npm run dev                         # starts on http://localhost:3000
```

---

## ⚠️ Files NOT included in git (must be regenerated/trained)

To keep the repo lean, the following are listed in `.gitignore` and need to be recreated after cloning:

### Dataset
- `dataset/sintetico/` — synthetic training dataset (generated from IFC models)
- `dataset/ply teste/` — test PLYs

**Regenerate:** run the dataset generator scripts in `dataset/`:
```bash
python dataset/gerar_dataset_sintetico.py
python dataset/gerar_ply_teste_especiais.py    # 19 test buildings × 3 stages
```

### Trained ML models
- `ml/models/random_forest.pkl` — trained RandomForest (~55MB)
- MLP checkpoints (`.pt`)
- RandLA-Net checkpoints (`randlanet/checkpoints/`)

**Retrain:** takes ~1-2 minutes for RF+MLP on a modern CPU/GPU:
```bash
python ml/train.py
```

This produces:
- `ml/models/random_forest.pkl`
- `ml/models/mlp_bim.pt`

### Why not in git?
- **Binaries don't belong in git** — they bloat history and slow clones
- **Reproducibility** — training script is the source of truth
- **Sizes:** RF `.pkl` ≈ 55 MB, dataset ≈ several GB

### If you need to share trained models
Use **GitHub Releases** (up to 2GB per file) or **Git LFS** (`git lfs track "*.pkl"`).

---

## 📁 Project structure

```
Beckend/
├── app.py                    # Main Flask backend (production)
├── app1.py                   # OBB experiment branch
├── usdz_to_ply.py           # USDZ (KIRI/RoomPlan) → PLY converter
├── ml/
│   ├── train.py             # RF + MLP training pipeline
│   └── models/              # trained models (gitignored)
├── randlanet/               # RandLA-Net instance segmentation
├── dataset/
│   ├── sintetico/          # training data (gitignored)
│   ├── ply teste/          # test PLYs (gitignored)
│   ├── ifc/                # source IFC files
│   └── gerar_*.py          # dataset generators
├── bim-ai-superintendent/   # React frontend
│   ├── App.tsx
│   ├── components/
│   └── services/
├── docs/
│   └── RANDOM_FOREST.md    # RF deep dive
└── STACK.md                 # full stack summary
```

---

## 🧪 Quick test

1. Start backend: `python app.py`
2. Start frontend: `cd bim-ai-superintendent && npm run dev`
3. Open http://localhost:3000
4. Upload an IFC and a PLY
5. Pick analysis mode: `BBox`, `ML`, `Both`, or `Instances`
6. Click **Processar Análise**

---

## 📊 Current metrics

- **Dataset:** 1290 samples / 129 buildings (split 91/19/19 by building)
- **RandomForest:** 97.9% accuracy, F1 macro 0.942
- **MLP (PyTorch):** 87.1% accuracy
- **Classes:** COMPLETO / PARCIAL / AUSENTE
