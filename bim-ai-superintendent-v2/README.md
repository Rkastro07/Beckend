# BIM AI Superintendent — v2

Front dedicado ao **pipeline v2** (Sonata + Hungarian matching + RF router).

Diferença vs `bim-ai-superintendent/` (front legado):
- Fala **apenas** com o endpoint `POST /api/analisar_ai_v2`.
- Não tem os modos BBox / AI / Both / Instances — só uma análise: v2.
- Renderiza seção "Adições" (instâncias scan sem par IFC) com cor `sky`.
- Renderiza painel de telemetria do pipeline (timings, RF disponível, cache Sonata).
- Suporta o novo status `ADICAO` na tabela.

## Como rodar

```bash
cd bim-ai-superintendent-v2
npm install
npm run dev
```

Sobe em `http://localhost:3001` (porta diferente do legado em 3000 — podem coexistir).

O backend continua sendo o `app_obb.py` na porta `8081`.

## Arquivos modificados em relação ao legado

- `services/api.ts` — só `listFloors` e `analyzeFloorV2`.
- `App.tsx` — sem `analysisMode`, sem `resultAI`.
- `components/Sidebar.tsx` — sem picker de modo de análise.
- `components/DataView.tsx` — nova seção "Adições" + painel de telemetria v2 no final.
- `package.json` — nome `bim-ai-superintendent-v2`.
- `vite.config.ts` — porta `3001`.
- `index.html` — título "v2 — Sonata + Hungarian".

Os componentes `ReportView`, `VoiceAssistant`, `Card` continuam idênticos ao legado.

## Endpoint que o front consome

```
POST /api/analisar_ai_v2
multipart/form-data:
  ifc_file: <file>        (ou ifc_token: <string>)
  ply_file: <file>
  pavimento: <string>     ("__TODOS__" pra todos)
```

Resposta esperada (recortes relevantes):

```jsonc
{
  "session_id": "...",
  "modo": "ai_v2",
  "estatisticas": {
    "total": 245,
    "completos": 87,
    "parciais": 12,
    "ausentes": 134,
    "adicoes": 12,
    "progresso_geral": 38.1
  },
  "resultados": [
    {
      "guid": "...",
      "tipo": "IfcWall",
      "nome": "Wall A",
      "status": { "code": "COMPLETO", "texto": "...", "cor": "#22c55e" },
      "rf_version": "v2",
      "match_info": {
        "cost": 1.23,
        "scan_class": "wall",
        "scan_centroid": [5.0, 3.0, 1.5],
        "scan_n_pts": 1500,
        "scan_conf": 0.85,
        "matched_by": "hungarian"
      },
      "bbox": { ... },
      "bbox_normalized": { ... }
    }
  ],
  "adicoes": [
    {
      "scan_class": "wall",
      "centroid": [12.3, 4.5, 1.5],
      "n_pts": 832,
      "mean_conf": 0.78,
      "volume": 4.2,
      "bbox": { ... },
      "status": { "code": "ADICAO", "texto": "Construido Fora do Plano", "cor": "#3b82f6" }
    }
  ],
  "meta": {
    "pipeline_version": "v2",
    "timings": { "sonata_s": 4.2, "hungarian_s": 0.1, "rf_inference_s": 0.4, "total_s": 5.1 },
    "rf_router_info": { "v1_available": true, "v2_available": false },
    "from_sonata_cache": true
  }
}
```
