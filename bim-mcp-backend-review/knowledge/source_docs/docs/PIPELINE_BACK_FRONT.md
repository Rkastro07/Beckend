# Pipeline Back ↔ JSON ↔ Front

Como os dados viajam entre upload do usuário e renderização no viewer 3D.
Cobre o fluxo do `/api/analisar_ai` (modo principal) com cores RGB do scanner.

---

## 📤 Fase 1 — Upload (Front → Back)

O usuário sobe **2 arquivos brutos**:

| O que | Tipo | Tamanho típico |
|---|---|---|
| **IFC** (planejado) | `multipart/form-data` campo `ifc_file` | 50–200 MB |
| **PLY** (executado) | campo `ply_file` | 100–500 MB |

Ou — após a primeira chamada — só o **token do IFC**:

| | |
|---|---|
| `ifc_token` | hash de 32 chars do conteúdo IFC |
| `ply_file` | sempre o arquivo (PLY não tem cache hoje) |
| `pavimento` | string ex: `"COBERTURA"` |

**Por que existe o token:** evita reupload de 200MB toda vez que troca de pavimento. O IFC fica na memória do servidor (`_IFC_CACHE`) por sessão.

---

## ⚙️ Fase 2 — Processamento backend

O backend faz, em ordem:

```
1. Cache do IFC          → ifcopenshell.open() (uma vez por sessão)
2. Extrai objetos        → walls, slabs, beams, columns do pavimento
3. Lê PLY                → pts (N×3 float64) + colors? (N×3 float em [0,1])
4. Dedup                 → np.unique(...,return_index=True) preserva ordem
5. Detecta conexões T    → marca paredes que cruzam outras
6. Marca piso/teto       → flag pra cortes anti-leaking
7. Alinha PLY ao IFC     → centroid + brute-force rotação
8. Normaliza coordenadas → translação final
9. Constrói KDTree       → índice espacial dos pontos
10. Loop de features     → 11 features por objeto (modo AI)
11. RandomForest         → predict + predict_proba (COMPLETO/PARCIAL/AUSENTE)
12. Loop de JSON         → grava 1 arquivo por objeto em /outputs/{session}/
13. Monta resposta       → 1 JSON principal com tudo
```

---

## 📥 Fase 3 — Duas respostas (back → front)

O backend produz **dois tipos de JSON diferentes**.

### 3A — Resposta direta da chamada `/api/analisar_ai`

É o JSON que volta no `fetch()` original. Pequeno, lido de uma vez.

```json
{
  "pavimento": "COBERTURA",
  "session_id": "a6237f06-...",
  "modo": "ai",
  "estatisticas": {
    "total": 85,
    "completos": 78, "parciais": 2, "ausentes": 5,
    "progresso_geral": 92.4
  },
  "resultados": [
    {
      "guid": "3lXuIwe5v5D9u4mRlpfFce",
      "nome": "Basic Wall:PI01_E010:2456961",
      "tipo": "IfcWall",
      "status": { "code": "COMPLETO", "emoji": "✅", "texto": "...", "cor": "#4caf50" },
      "confianca_ml": 0.961,
      "bbox": { "xmin":..., "xmax":..., "ymin":..., "ymax":..., "zmin":..., "zmax":... },
      "bbox_normalized": { ... },     // mesma bbox em coords Three.js
      "obb_corners": [[x,y,z]×8],     // 8 vértices do OBB orientado
      "json_file": "a6237f06-.../Basic_Wall_PI01_E010_2456961_3lXuIwe5.json",
      "n_pts": 2135                   // só metadado, não os pontos
    }
    // ... 84 outros
  ]
}
```

**Observação crítica:** essa resposta **NÃO tem os pontos da nuvem**. Só metadados.
Tamanho típico: ~50-200 KB para 100-1000 objetos. Carrega instantâneo.

### 3B — Arquivos JSON por objeto (`/outputs/{session_id}/{nome}_{guid}.json`)

Esses são gravados **no disco do servidor** durante o processamento. O front baixa
**sob demanda** quando renderiza cada objeto.

```json
{
  "positions": [x,y,z, x,y,z, ...],   // floats em coords Three.js
  "color":     [0.2, 0.8, 0.2],       // RGB único (fallback)
  "count":     2135,
  "colors":    [123,45,200, ...]      // ✨ uint8 0-255 se PLY tem RGB
}
```

| Campo | O que é | Tamanho típico |
|---|---|---|
| `positions` | array flat `[x₀,y₀,z₀, x₁,y₁,z₁, ...]` em coords Three.js (Y=altura) | 24 bytes/ponto |
| `color` | RGB único pra fallback (verde/laranja/vermelho por status) | 3 floats |
| `count` | sanity check (= `positions.length / 3`) | 4 bytes |
| `colors` | RGB do scanner como `[r₀,g₀,b₀, ...]` em **uint8** | 3 chars/ponto JSON serialize |

**Por que dividir em arquivos por objeto** em vez de mandar tudo num response só:

1. **Lazy loading no viewer** — só carrega objetos visíveis na câmera
2. **Cache do navegador** — toggle de visibilidade não recarrega
3. **Resposta principal fica leve** — não bloqueia a UI esperando 200MB
4. **Se 1 objeto falhar de baixar, os outros funcionam**

Tamanho típico de um JSON por objeto: 50KB–2MB (depende da densidade de pontos).

---

## 🎨 Fase 4 — Renderização front

```
1. App.tsx faz POST /api/analisar_ai
   ↓
2. Recebe AnalysisResult (3A)
   ↓
3. Passa pra DataView.tsx
   ↓
4. DataView itera resultados.map() → renderiza um <group> por objeto:
   ├─ <BimBoundingBox>     ← desenha o cubo wireframe (usa bbox)
   ├─ <SelectableHitbox>   ← área clicável invisível pra seleção
   └─ <ObjectPoints>       ← faz fetch do JSON 3B sob demanda
        ↓
        - GET /outputs/{session_id}/{file}.json
        - Lê positions → Float32Array → bufferAttribute "position"
        - Se colors: uint8 → float32/255 → bufferAttribute "color"
        - <pointsMaterial vertexColors={!!colors} />
```

---

## 🔄 Resumo visual

```
┌─────────────┐  IFC + PLY              ┌─────────────────────────────┐
│   FRONT     │ ──────────────────────► │           BACK              │
│             │  (uploads)              │                             │
│  App.tsx    │                         │  app_obb.py                 │
│             │ ◄────────────────────── │                             │
│  DataView   │  AnalysisResult (3A)    │  /api/analisar_ai           │
│             │  - estatisticas         │                             │
│             │  - resultados[]         │  Gera no disco:             │
│             │    - guid, status,      │  /outputs/{session}/        │
│             │      bbox, json_file    │    obj_001.json (3B)        │
│             │                         │    obj_002.json             │
│             │                         │    ... (1 por objeto)       │
│             │                         │                             │
│  ObjectPts  │ ──── GET ─────────────► │                             │
│  (lazy)     │  /outputs/.../001.json  │                             │
│             │ ◄──── JSON 3B ────────  │                             │
│             │  - positions            │                             │
│             │  - colors? (uint8)      │                             │
└─────────────┘                         └─────────────────────────────┘
```

---

## 🧠 Por que essa arquitetura

1. **Resposta principal pequena → UI responsiva.** Tabela e gráficos aparecem em <1s.
2. **Pontos por arquivo → escala.** Pavimento de 1500 objetos não trava o front; carrega só o que está visível.
3. **PLY processado uma vez, salvo em disco.** Se o usuário fechar e reabrir o site, os JSONs ainda estão lá na sessão.
4. **uint8 nos colors.** Sem isso, RGB triplica o tamanho do JSON. Com uint8, custo é ~1.5× o de positions sozinho.

---

## 📌 Pontos de extensão futuros

- **Cache de PLY** — análogo ao `_IFC_CACHE`, evita reler PLY de 500MB ao trocar de modo
- **Streaming dos pontos** — em vez de JSON com array gigante, usar binary endpoint (`.bin` + view tipada)
- **TTL do `/outputs/{session}/`** — hoje fica até reboot; cron ou middleware p/ limpar sessões antigas
- **Subsample por LOD** — versão "leve" (10% dos pontos) servida primeiro, "completa" sob demanda quando o usuário foca no objeto
