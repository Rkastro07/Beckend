# 🏗️ BIM vs. Point Cloud Analysis API (Hybrid V2.0)

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg) ![Python](https://img.shields.io/badge/python-3.8+-green.svg) ![Status](https://img.shields.io/badge/status-stable-success.svg)

A high-performance Flask API for automated construction progress monitoring. It aligns raw 3D scans (**PLY**) with Building Information Models (**IFC**) to detect built elements, analyze deviations, and classify progress using a **Hybrid Geometric Alignment Engine** and **Anti-Leaking heuristics**.

## 🚀 Key Features

* **Hybrid Alignment 2.0:** Combines "Center of Mass" translation (high precision) with "Sampled Scoring" rotation (high speed) to register point clouds without manual markers.
* **Anti-Leaking Protection:** Dynamic bounding-box shrinking and connection detection (T-Junctions) to prevent points from floors/ceilings from "leaking" into wall analysis.
* **Vertical Slicing:** Instead of boolean collision, elements are analyzed in vertical segments to distinguish between "Started" (low wall) and "Complete" (full height).
* **AI Reporting:** Integration with **DeepSeek API** to generate executive construction reports based on the geometric analysis.

---

## 🧠 Algorithmic Logic

### 1. Hybrid Alignment Engine
The core challenge is registering a random point cloud to the IFC coordinate system. We use a two-step approach:

1.  **Translation ($t$):** We calculate the centroid of the **entire** point cloud ($C_{cloud}$) and the centroid of the IFC model ($C_{ifc}$).
    $$t = C_{ifc} - C_{cloud}$$
    *This is $O(N)$ and ensures the models overlap in space.*

2.  **Rotation ($R$):** We take a random sample (e.g., 150k points) and test orthogonal permutations of axes ($X, Y, Z$) and signs ($\pm$). The best rotation is determined by a **Scoring Function** that counts how many points fall inside IFC bounding boxes.

### 2. Anti-Leaking Heuristics
Point clouds are noisy. Points from a slab often "leak" into the wall below it. To fix this:

* **T-Junction Detection:** The system identifies walls that intersect others. Their bounding boxes are horizontally shrunk by 10% to avoid "ghost points" from neighbors.
* **Dynamic Z-Cropping:**
    * If a wall physically connects to a **Floor Slab**: Ignore bottom 10% of points.
    * If a wall physically connects to a **Ceiling Slab**: Ignore top 10% of points.

### 3. Workflow Diagram

```mermaid
graph TD
    A[Upload IFC & PLY] --> B{Geometry Engine}
    B -->|IFC| C[Extract Objects & BBox]
    B -->|PLY| D[Downsample & Center]
    C & D --> E[Hybrid Alignment]
    E --> F[Coordinate Normalization]
    F --> G[Object Iteration Loop]
    
    subgraph Analysis Per Object
    G --> H[Check Connections T/L]
    H --> I[Apply Anti-Leaking Cuts]
    I --> J[Filter Points in AABB]
    J --> K[Calculate Vertical Density]
    K --> L[Classify Status]
    end
    
    L --> M[Generate JSON/PLY Output]
🛠️ InstallationPrerequisitesPython 3.8+pipSetupClone the repository:Bashgit clone [https://github.com/your-username/bim-analysis-api.git](https://github.com/your-username/bim-analysis-api.git)
cd bim-analysis-api
Install dependencies:Bashpip install -r requirements.txt
Core requirements: flask, flask-cors, ifcopenshell, open3d, numpy, requests.(Optional) Set DeepSeek API Key for AI reports:Bashexport DEEPSEEK_API_KEY="your-key-here"
Run the server:Bashpython app.py
📖 API Documentation1. Analyze StoreyPerforms the full geometric analysis.Endpoint: POST /api/analisar_pavimentoContent-Type: multipart/form-dataParameterTypeDescriptionifc_fileFileThe .ifc project fileply_fileFileThe .ply point cloud scanpavimentoStringName of the storey (e.g., "Level 1")Response Example:JSON{
  "session_id": "550e8400-e29b...",
  "estatisticas": {
    "completos": 45,
    "parciais": 12,
    "ausentes": 5,
    "progresso_geral": 82.5
  },
  "resultados": [
    {
      "guid": "3290482309...",
      "nome": "Wall_Gen_200mm",
      "status": {
        "code": "COMPLETO",
        "emoji": "✅",
        "texto": "Completo"
      },
      "densidade": 1450.2,
      "dimensoes": {
        "executado": {"z": 2.95},
        "planejado": {"z": 3.00}
      }
    }
  ]
}
2. Generate AI ReportUses LLM to summarize the JSON statistics into a readable executive summary.Endpoint: POST /api/generate_reportBody: {"prompt": "Analyze these stats: ..."}📐 Math & Logic ReferenceTransformation MatrixThe alignment applies the following transformation to every point $P$:$$P_{aligned} = (P_{raw} \times R^T) \times S + t$$Where:$R$: Rotation Matrix (derived from axis permutation).$S$: Scale Factor (usually 1.0, 0.001, or 1000 depending on units).$t$: Translation Vector (centroid difference).Density CalculationDensity $\rho$ is calculated using a Voxel Grid approach for slabs and Slices for walls:$$\rho = \frac{N_{points}}{V_{bbox}}$$For walls, we check Vertical Coverage ($C_v$):$$C_v = \frac{\sum_{i=1}^{10} (slice_i > \text{threshold})}{10}$$📂 Project Structure.
├── app.py                 # Main Flask Application
├── requirements.txt       # Python Dependencies
├── /uploads               # Temp storage for uploaded files
└── /outputs               # Generated JSONs and PLYs
📜 LicenseMIT
