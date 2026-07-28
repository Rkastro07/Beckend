# Plano: Gerador de As-Built IFC (planta → IFC → scan → as-built)

> Status: **PLANEJADO, não iniciado** (26-Jun-2026). Salvo pra retomar depois.

## Contexto / motivação

Cliente relatou que procurava extensão que transforma nuvem de pontos em análise
as-built (scan-to-BIM), mas o caminho clássico exige ML de segmentação/reconstrução
e "dava muito trampo" (mesma dor que tivemos com Mask3D/Sonata: gargalo é dataset).

**Insight (do Rafael):** em vez de o ML *inventar* a geometria a partir da nuvem,
a planta 2D fornece a geometria limpa (via planta2bim) e a nuvem só **valida o que
existe**. Scan-to-BIM "guiado por prior". Pro caso de uso de progresso/medição, o
resultado é funcionalmente o mesmo — sem o trampo de ML.

## Fluxo completo

```
Planta DWG/DXF ──planta2bim──► IFC as-planned
                                    │
Nuvem de pontos (scan) ──── match v1 (app_obb, RF) ────┐
                                    │                   │
                                    ▼                   ▼
                          guid → status            sobras de pontos
                     (COMPLETO/PARCIAL/AUSENTE      (fora das bboxes)
                        + height_fill)                  │
                                    │              Sonata (filtro semântico)
                                    │              wall/slab/column/beam
                                    │              descarta "outros"
                                    │              (entulho/móveis/andaime)
                                    │                   │
                                    │              DBSCAN por classe
                                    │              = instâncias manuais
                                    │                   │
                                    ▼                   ▼
                        ┌─────────────────────────────────────┐
                        │   TERCEIRO IFC = AS-BUILT            │
                        │   - remove AUSENTE                   │
                        │   - PARCIAL: Depth × height_fill     │
                        │   - adições: OBB → elemento TIPADO   │
                        │     (IfcWall "Adicao-Parede-001")    │
                        │   - Pset com status/percentual/data  │
                        └─────────────────────────────────────┘
```

## Componentes a construir

### 1. Gerador as-built core (novo módulo)
- Input: IFC (as-planned) + resultado da análise v1 (mapa `guid → status, height_fill`)
- Copia o IFC com ifcopenshell:
  - `AUSENTE` → remove o elemento
  - `COMPLETO` → mantém como está
  - `PARCIAL` → reduz altura: `Depth × height_fill`. Funciona direto nos IFCs do
    planta2bim (tudo é IfcExtrudedAreaSolid com Depth). Pra IFC de terceiros com
    geometria complexa: fallback = manter geometria + só marcar no Pset
- Property set em cada elemento: status, percentual, data da medição
  (qualquer viewer BIM mostra a medição junto)

### 2. Detector de adições (Sonata como FERRAMENTA, não pipeline)
- Pontos do scan fora de todas as bboxes do IFC
- **Sonata classifica** (reaproveitar `pipeline_v2/sonata_runner.py` +
  `sonata_cache.py` como BIBLIOTECA — importar funções, não o orchestrator)
- Manter só classes estruturais (wall/slab/column/beam) → mata falso-positivo de
  entulho/móvel/andaime que o DBSCAN puro geraria
- DBSCAN por classe = definição de instância manual (decisão do Rafael: Sonata é
  ruim pra instância, mas bom pra classe — instância a gente resolve na mão)
- OBB de cada cluster → entra no as-built **tipado** (IfcWall/IfcSlab "Adicao-*"),
  não proxy genérico — viewer mostra parede de verdade e o elemento fica
  comparável em scans futuros

### 3. Endpoint `/api/gerar_asbuilt` (app_obb.py)
- Amarra: análise v1 → gerador core → detector de adições → retorna IFC as-built
- Padrão dos endpoints existentes: aceita `ifc_token` (cache) ou upload, salva
  resultado em /outputs

## O que JÁ existe (verificado no código em 26-Jun)

| Peça | Onde | Nota |
|------|------|------|
| Status por guid | `app_obb.py:548` (`/api/analisar_ai` v1, RF 97.9%) | sem Sonata |
| height_fill por objeto | `app_obb.py:2717` (feature do RF) | % altura preenchida |
| Instâncias do scan | `/api/analisar_instancias` (RandLA-Net+DBSCAN) | alternativa s/ Sonata |
| Sonata runner+cache | `pipeline_v2/sonata_runner.py`, `sonata_cache.py` | usar como lib |
| Planta → IFC | `plantatobim/planta_to_ifc_v1.py` (leitor v2) | 6/6 fontes testadas |
| Escrita IFC | ifcopenshell 0.8.3 (Python Windows) | já usado no planta2bim |

## Decisões de arquitetura (tomadas nesta conversa)

1. **Pipeline v2 (orchestrator Sonata+Hungarian+RF-v2) APOSENTADO** — Rafael não
   tem interesse em continuar. O caminho de análise é o v1 (RF).
2. **Sonata rebaixado a ferramenta de filtro semântico** — só no detector de
   adições. Classe Wall dele é forte (IoU 0.77); instância é feita na mão (DBSCAN).
3. Isso esvazia também o plano HELIOS++ (existia pra melhorar treino do
   Mask3D/Sonata) — fica congelado junto.

## Caveats pra comunicar ao cliente

- As-built **nominal**, não medido: parede construída 30cm fora do lugar aparece
  na posição da PLANTA. Perfeito pra progresso/medição; NÃO substitui as-built
  cadastral/jurídico (documentação de desvios).
- Exige que o cliente tenha a planta (DWG/DXF) — premissa do planta2bim.
- Adições dependem da qualidade do Sonata no scan real (não testado ainda em
  scan de obra com entulho).
