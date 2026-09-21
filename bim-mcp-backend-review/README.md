# Plataforma BIM local

Raiz oficial dos sistemas BIM que compartilham o mesmo backend e a mesma
interface.

## Produtos ativos

- **Vistoria OBB + Random Forest**: compara IFC e nuvem PLY para classificar o
  progresso dos elementos. O backend oficial é `app_obb.py`.
- **Geradores de simulação**: criam nuvens sintéticas de estágios de obra a
  partir de IFC e geram o gabarito de status.
- **Scan/Cloud-to-BIM V2**: processa E57, PLY e XYZ, detecta geometria e produz
  PNG de aprovação antes do IFC final.
- **Planta-to-BIM + referência OBB**: o foco principal é DWG/DXF com o detector
  CAD V2, que combina layers, nomes de blocos, geometria, separação de plantas
  e mapeamento manual. IFC/IFCZIP, SVG e PDF vetorial também entram no editor.
  O modelo aprovado segue diretamente para o comparativo nuvem × IFC.
- **CAD Object Grammar V3**: associa textos, dimensões, blocos anônimos e
  geometria para reconhecer esquadrias antes do IFC, incluindo portões largos.
- **Editor e biblioteca BIM**: operações determinísticas, spaces, lajes,
  aberturas e receitas de modelagem compartilhadas pela plataforma e pelo MCP.
- **Visita 3D híbrida**: abre Gaussian Splats e nuvens E57, PLY, XYZ, PTS,
  PCD, LAS ou LAZ para navegação imersiva e captura PNG. Nuvens convencionais
  são amostradas apenas para visualização; o pipeline BIM conserva o original.
- **Conversão Pro durável**: no Cloud Run, o estado do pedido fica no Supabase
  Postgres, plantas e resultados ficam no bucket privado `plan2bim-jobs` e o
  processamento pago é disparado pelo Cloud Tasks. Sem essas variáveis, o
  ambiente local mantém o executor em processo para testes.

## Componentes retirados do runtime

Sonata, RandLA-Net e o antigo `pipeline_v2` não fazem parte da aplicação
executável. A documentação histórica continua em `knowledge/` e `docs/` porque
contém decisões e aprendizados úteis, mas não é carregada pelo backend.

## Estrutura

```text
app_obb.py                   backend Flask oficial
bim-ai-superintendent/       plataforma React com todos os produtos
simulation/                  geradores sintéticos do produto OBB
bim_editing/                 operações e revisões BIM
bim_authoring/               receitas BIM
experiments/cloud2bim/       motor atual de detecção da nuvem
plantatobim/                 importação e exportação IFC
runtime/                     launchers, diagnóstico e runtime portátil
knowledge/                   documentação consultável para o MCP
```

O modelo Random Forest ativo é empacotado em
`.runtime/models/random_forest.pkl`. O diretório `.runtime` não deve ser
versionado.

## Executar

Backend e frontend:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File runtime\run_platform.ps1
```

Somente o backend:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File runtime\run_backend.ps1
```

Pipeline Cloud-to-BIM direto:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File runtime\run_cloud2bim.ps1 `
  -InputCloud "C:\caminho\nuvem.e57"
```

Diagnóstico offline:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File runtime\doctor.ps1
```

A API usa `http://localhost:8081` e o frontend usa
`http://localhost:3000`. O processamento local não instala bibliotecas durante
a execução.

## Referência para vistoria

O painel **Modelo de referência** aceita um IFC pronto ou uma planta. O fluxo
de planta é:

```text
DXF/DWG/SVG/PDF/IFCZIP
        ↓
importação geométrica
        ↓
revisão no editor 2D
        ↓
IFC aprovado + token em cache
        ↓
comparação Cloud × IFC no OBB
```

O PDF precisa conter vetores; PDF escaneado não é tratado como geometria. A
escala inicial é 1:50 e pode ser corrigida e reprocessada no editor. DWG usa o
GNU LibreDWG 0.13.4 local, com AutoCAD Core Console apenas como fallback, para
gerar um DXF intermediário.
DWF clássico é reconhecido para diagnóstico, mas deve ser exportado como DWG,
DXF ou PDF vetorial antes da edição.

Detalhes e contrato HTTP:
[`docs/REFERENCE_INGESTION_V1.md`](docs/REFERENCE_INGESTION_V1.md).

O detector e o painel de layers CAD estão documentados em
[`docs/CAD_DETECTOR_V2.md`](docs/CAD_DETECTOR_V2.md).

O visualizador de nuvens e Gaussian Splats, incluindo formatos, protocolo
binário, limites e diagnóstico, está documentado em
[`docs/VISUALIZADOR_3D_HIBRIDO.md`](docs/VISUALIZADOR_3D_HIBRIDO.md).

## Servidor MCP BIM

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File runtime\run_mcp.ps1
```

O MCP usa o mesmo editor JSON e o mesmo gerador IFC da plataforma. O contrato
obriga a gerar PNG de aprovacao antes de aceitar `approved=true` para exportar
o IFC. As ferramentas retornam caminhos absolutos e URIs de todos os
artefatos.

A engenharia não fica implícita no prompt. O servidor publica:

- `bim://engineering/stack`: responsabilidade de IfcOpenShell, Shapely,
  Planta-to-BIM, receitas e catálogo;
- `bim://authoring/recipes`: contratos geométricos e semânticos pesquisáveis;
- `bim://ifc-library/summary`: evidência extraída de 262 modelos IFC;
- `bim://ifc-library/relationship-patterns`: relações reais de parede, abertura,
  porta e janela.

As receitas são normativas e podem ser executáveis; o catálogo de modelos é
referência somente leitura e não substitui o motor geométrico.
