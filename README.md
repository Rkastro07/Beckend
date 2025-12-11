# 🏗️ BIM Analysis API - Backend

This is the core backend service for the **BIM Analysis App**, submitted to the **Google AI Studio - Vibe Coding with Gemini 3 Pro Hackathon**.

It is a robust Python/Flask application capable of auditing construction sites by comparing **As-Planned (IFC)** models against **As-Built (PLY)** point clouds.

## 🔥 Key Features

### 1. Hybrid Brute Force Alignment (Zero-Center)
Solves the "floating model" problem where scan coordinates differ from BIM coordinates.
* **Automatic Outlier Removal:** Uses statistical analysis to clean sensor noise.
* **Zero-Center Normalization:** Temporarily moves both models to `(0,0,0)` to find the optimal rotation and translation match.

### 2. Anti-Leaking Protection
Prevents false positives in progress monitoring.
* **T-Junction Detection:** Identifies wall intersections to shrink bounding boxes and avoid overlapping points.
* **Dynamic Floor/Ceiling Cuts:** Automatically filters out points from slabs and roofs that "leak" into wall bounding boxes.

### 3. AI-Powered Reporting
Integrates with **DeepSeek LLM** to generate executive summaries based on the raw technical data extracted from the scan.

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Flask** (API Server)
* **Open3D** (Point Cloud Processing & Alignment)
* **IfcOpenShell** (BIM Geometry Extraction)
* **NumPy** (Vector Math & Matrix Transformations)

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/SEU-USUARIO/bim-analysis-backend.git](https://github.com/SEU-USUARIO/bim-analysis-backend.git)
cd bim-analysis-backend
