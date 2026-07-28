# Revisão de escopo — BIM MCP local

Data da revisão: 24/07/2026

## Objetivo desta cópia

Esta pasta é uma cópia de trabalho **seletiva** do sistema atual. O original em
`C:\Users\Rafael\Desktop\Beckend` não foi movido nem alterado.

A pasta original tem mais de 40 GB porque contém datasets, checkpoints, nuvens,
ambientes instalados e resultados de processamento. Esta cópia contém 78 arquivos
de fonte e aproximadamente 1,48 MB de código, receitas, testes e front-end.

O primeiro produto MCP será local e cobrirá este ciclo:

1. receber uma nuvem local;
2. detectar pavimentos, paredes, pilares, aberturas, escadas e cobertura;
3. devolver modelo métrico e PNG nomeado;
4. aceitar correções por identificador;
5. validar a revisão contra a nuvem;
6. exportar IFC;
7. abrir o front-end como visualizador/editor quando necessário.

## Fica no MVP local

### Motor e integração

- `scan_endpoints.py`
  - upload e leitura da nuvem;
  - sessões e jobs;
  - preview de lajes e paredes;
  - classificação multifatia;
  - escadas;
  - geração e auditoria do IFC.
- `experiments/cloud2bim/rodar.py`
  - receita principal Cloud-to-BIM.
- `experiments/cloud2bim/cloud2bim_patched/`
  - núcleo geométrico adaptado;
  - detector de paredes V2;
  - geração IFC;
  - geração de espaços;
  - licença e atribuição do projeto de origem.
- `experiments/cloud2bim/detect_pilar.py`
  - detector de pilares usado pelo runner.
- `experiments/cloud2bim/prototipos/`
  - manter inicialmente porque o fluxo ativo chama:
    - `detect_casca_curva.py`;
    - `montar_escada_gabarito.py`;
    - `assar_geometria.py`.
  - os demais protótipos permanecem como referência até a revisão por import graph.
- `experiments/cloud2bim/configs_exemplo/`
  - receitas reproduzíveis de calibração.
- `experiments/cloud2bim/tests/`
  - testes do detector de paredes, lajes, espaços e diagnósticos.

### Modelo editável e exportação

- `plantatobim/planta_to_ifc_v1.py`
  - contrato atual de paredes, aberturas e laje;
  - reconstrução do modelo editado;
  - geração IFC detalhada;
  - base para o modelo canônico do MCP.
- `tools_endpoints.py`
  - Planta-to-BIM e conversores;
  - fica para a segunda receita, depois do Cloud-to-BIM local.
- `usdz_to_ply.py`
  - entrada futura de scans de celular.

### Visualizador

- `bim-ai-superintendent/`
  - front-end v1 sem `node_modules`, `dist`, logs ou segredos;
  - `ScanToBim.tsx` é o wizard visual do motor;
  - `PlantaEditor.tsx` é o editor humano;
  - o front continuará sendo uma interface opcional do MCP.

### Bootstrap temporário

- `app_obb.py`
  - fica temporariamente porque já registra `tools_endpoints` e
    `scan_endpoints`;
  - contém muitas rotas legadas que não devem ser expostas pelo MCP;
  - será substituído por um bootstrap local mínimo quando criarmos
    `bim_mcp/server.py`.

## Fica para fases posteriores

- Planta/DXF para BIM.
- USDZ/OBJ/ASC para PLY.
- Comparação IFC × nuvem e medição de progresso.
- Geração de relatórios.
- Design de interiores e biblioteca de objetos.
- Recursos MCP para receitas, normas, modelos e relatórios.
- Interface MCP remota, autenticação e cobrança.

Essas capacidades não devem aumentar o primeiro conjunto de ferramentas antes de
o ciclo Cloud-to-BIM local estar estável.

## Não entrou nesta cópia

### Dados e artefatos

- `dataset/` (~20 GB);
- `bim_outputs/` (~6,7 GB);
- `bim_uploads/`;
- nuvens `.ply`, `.e57`, `.xyz`, `.asc`;
- IFCs, DXFs, OBJ e ZIP de amostra;
- checkpoints e modelos treinados;
- logs e caches;
- `node_modules/` e `dist/`;
- `.git/`.

Esses itens são dados de teste, resultados ou dependências reconstruíveis. Não
fazem parte do código distribuível do servidor MCP.

### Pipelines fora do primeiro produto

- `experiments/sonata/`;
- `experiments/mask3d/`;
- `pipeline_v2/`;
- `randlanet/`;
- `ml/`;
- `bim-ai-superintendent-v2/`;
- `app.py`, `app1.py`, `app copy.py`, `app (4).py`;
- scripts de treinamento e notebooks.

Motivo: são pesquisa, versões aposentadas ou rotas que exigem datasets e
checkpoints grandes. Podem virar módulos opcionais no futuro, mas não devem
entrar no MVP Cloud-to-BIM local.

## Contratos que o MCP deverá criar

### Projeto

- `project_id`
- `source_asset`
- `status`
- `created_at`
- `floors`
- `active_revision`

### Elemento

- identificador estável, por exemplo `W-001`;
- tipo BIM;
- geometria métrica;
- pavimento;
- origem: `measured`, `inferred`, `designed` ou `confirmed`;
- cobertura da nuvem;
- residual geométrico;
- confiança;
- revisão em que foi criado ou alterado.

### Revisão

- versão imutável;
- operações aplicadas;
- autor e motivo;
- PNG correspondente;
- resultado da validação;
- IFC exportado, quando existir.

## Primeiro conjunto de ferramentas MCP

1. `cloud_to_bim_start`
2. `cloud_to_bim_status`
3. `cloud_to_bim_inspect_floor`
4. `cloud_to_bim_render_floor`
5. `cloud_to_bim_inspect_element`
6. `cloud_to_bim_propose_changes`
7. `cloud_to_bim_apply_changes`
8. `cloud_to_bim_validate`
9. `cloud_to_bim_export_ifc`
10. `open_bim_viewer`

As operações de escrita serão versionadas. Exclusões não apagarão revisões
anteriores.

## Próxima implementação

Criar `bim_mcp/` nesta cópia, sem modificar inicialmente o motor:

```text
bim_mcp/
  server.py
  contracts.py
  project_store.py
  tools/
    cloud_to_bim.py
    model_editing.py
    validation.py
    export.py
```

O primeiro adapter poderá chamar o backend local existente. Depois, as funções
serão extraídas de `scan_endpoints.py` para serviços reutilizados tanto pelo
Flask quanto pelo MCP.

## Biblioteca de engenharia de autoria BIM

A cópia agora inclui `bim_authoring/`, uma base estruturada de receitas de
modelagem. Parede básica, janela em parede e porta em parede já possuem
executores sobre a API de alto nível do IfcOpenShell. Laje por polígono, encontro
de paredes, parede em camadas e claraboia estão documentadas no mesmo contrato e
aguardam executor.

As receitas registram entidades, relações IFC, sistema de coordenadas, entradas
métricas, passos, pós-condições e falhas conhecidas. O catálogo é pesquisável e
foi exposto em `/api/bim-authoring/recipes`. Os mesmos payloads estão preparados
como resources `bim://authoring/*`, sem acoplar esta camada ao SDK MCP.

## Validação desta cópia

- Todos os arquivos Python passaram pela compilação de sintaxe.
- Foram comparados por SHA-256 os 75 arquivos copiados que também existem no
  sistema original: nenhuma divergência.
- O teste isolado do detector de paredes executou 13 casos:
  - 12 passaram;
  - 1 falhou na medição angular, retornando 32° para um esperado de 30°.
- O mesmo caso falha, com o mesmo resultado, na pasta original. Portanto, não é
  uma regressão criada pela cópia e deverá entrar no backlog do motor.
- Os testes que importam a cadeia completa não rodaram no runtime de validação
  porque ele não possui todas as dependências nativas, como `matplotlib`,
  `open3d`, `ifcopenshell` e o leitor E57. A lista necessária está em
  `requirements-cloud2bim.txt`.
- O front-end foi copiado sem `node_modules`; sua instalação e build ficam para
  a próxima etapa, dentro desta pasta.

## Evolução aplicada: entrada geométrica

O primeiro recorte do MCP não fica limitado ao DXF. O modelador da cópia agora
tem um despachante comum para IFC/IFCZIP, DXF e SVG, com catálogo de capacidade
em `GET /api/planta/formatos`. IFC preserva semântica e pavimentos; CAD/SVG
preservam vetores; malhas e nuvens são encaminhadas ao Cloud-to-BIM.

A especificação e a matriz de formatos estão em
`docs/GEOMETRY_IMPORT.md`. Esse catálogo é um bom candidato a resource MCP, e a
importação/geração são candidatas a tools separadas.

## Base de conhecimento preservada

A documentação e as receitas autorais do backend original foram espelhadas em
`knowledge/source_docs/`, mantendo os caminhos relativos. O índice temático está
em `knowledge/README.md` e o manifesto verificável em
`knowledge/manifest.json`.

O manifesto registra 24 documentos, notebooks e configurações-exemplo com
SHA-256 e tags. Essa pasta será a origem dos resources
`bim://knowledge/*` e das futuras tools de busca de receitas e decisões
técnicas.
