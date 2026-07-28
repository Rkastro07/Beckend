# Relatório técnico — BIM MCP até agora

Data de consolidação: 24/07/2026
Pasta de trabalho: `bim-mcp-backend-review`

## 1. Resumo executivo

Esta cópia evoluiu de uma seleção do backend original para a base de um produto
local de modelagem BIM assistida por linguagem natural.

O objetivo é permitir que arquitetos e engenheiros deem ordens como:

> “Iguale a parede 011 à 010, feche na frente e atrás e ajuste piso e teto.”

O sistema precisa transformar a ordem em operações geométricas verificáveis,
gerar uma nova revisão do modelo, validar relações IFC e devolver IFC, JSON e
imagens identificadas. Cloud-to-BIM fornece a observação métrica da obra; a
biblioteca de autoria fornece as regras de como o modelo deve ser construído e
editado.

Já foram entregues:

- cópia seletiva e segura do backend para o futuro MCP;
- entrada geométrica unificada para IFC/IFCZIP, DXF e SVG;
- encaminhamento correto de malhas e nuvens ao Cloud-to-BIM;
- preservação de 24 documentos e receitas técnicas;
- biblioteca pesquisável de engenharia de autoria BIM;
- executores IFC para parede, janela em parede e porta em parede;
- validação semântica de aberturas e preenchimentos;
- recursos e endpoints de descoberta preparados para MCP;
- geração real do caso `doorgrondfloor.dxf` pelo app atual;
- renderização 2D e 3D identificada;
- diagnóstico reproduzível do erro de contorno dos slabs;
- revisão R01 com paredes, piso e teto corrigidos;
- 12 testes unitários aprovados.

O servidor MCP propriamente dito e o motor genérico de edição por operações
ainda não estão concluídos.

## 2. Organização e política de segurança

O backend original permanece em:

```text
C:\Users\Rafael\Desktop\Beckend
```

A evolução para MCP acontece em:

```text
C:\Users\Rafael\Desktop\Beckend\bim-mcp-backend-review
```

O original não foi movido nem sobrescrito. DXFs e IFCs usados como referência
foram lidos da pasta original, enquanto modelos novos, PNGs e relatórios foram
gravados em `artifacts/` dentro da cópia.

Datasets, checkpoints, nuvens, ambientes virtuais, builds e outputs pesados não
foram incorporados à cópia distribuível. A seleção completa está documentada em
[MCP_SCOPE_REVIEW.md](../MCP_SCOPE_REVIEW.md).

## 3. Arquitetura pretendida

```text
DXF / SVG / IFC -------------------+
                                   |
Nuvem / malha -> Cloud-to-BIM -----+-> modelo canônico JSON
                                             |
                                             v
                                  motor genérico de edição
                                             |
                            +----------------+----------------+
                            |                |                |
                            v                v                v
                       revisão JSON         IFC          PNG nomeado
                            |                |                |
                            +----------------+----------------+
                                             |
                                             v
                                     visualizador opcional

Documentação + receitas -> catálogo pesquisável -> Resources/Tools MCP
```

Princípios:

- unidades internas em metros;
- identificadores estáveis, como `W-010` e `SLAB-E-003`;
- edições geram revisões, sem apagar silenciosamente o estado anterior;
- geometria inferida deve ser distinguida de geometria medida ou confirmada;
- toda operação precisa de pré-condições e pós-condições verificáveis;
- PNG, JSON e IFC devem representar a mesma revisão.

## 4. Entrada geométrica unificada

O arquivo [plantatobim/geometry_importers.py](../plantatobim/geometry_importers.py)
passou a separar formatos pelo nível real de informação:

| Família | Formatos | Tratamento |
|---|---|---|
| BIM | IFC, IFCZIP | preserva objetos, GUIDs, nomes, vãos, lajes e pavimentos |
| CAD/vetor | DXF, SVG | preserva vetores e infere semântica por layer/grupo |
| Malha 3D | OBJ, USDZ; futuros STL/GLB/glTF | encaminha ao Cloud-to-BIM |
| Nuvem | PLY, E57, ASC, XYZ; futuros LAS/LAZ | encaminha ao Cloud-to-BIM |
| CAD proprietário | DWG | exige conversão local verificável antes da leitura |

### DXF

O leitor considera:

- `LINE` e `LWPOLYLINE`;
- `POLYLINE` e bulges;
- blocos `INSERT` e suas transformações;
- `ARC`, `CIRCLE`, `ELLIPSE` e `SPLINE`;
- `SOLID`, `TRACE` e `3DFACE`;
- layers herdadas de blocos na layer `0`;
- nomes multilíngues para parede, porta, janela e abertura.

### IFC e IFCZIP

O importador:

- seleciona `IfcBuildingStorey`;
- lê paredes, portas, janelas, aberturas e slabs;
- mede geometria em coordenadas de mundo;
- mantém nome, GUID, classe e pavimento;
- normaliza o Z para o editor sem perder a elevação original.

### SVG

São considerados elementos vetoriais, paths, transformações de grupos e
unidades físicas. Geometria sem semântica explícita entra com aviso.

### Rotas locais

- `GET /api/planta/formatos`
- `POST /api/planta/importar`
- `POST /api/planta/parse`, mantida como alias

A especificação completa está em
[GEOMETRY_IMPORT.md](GEOMETRY_IMPORT.md).

## 5. Base de conhecimento preservada

A pasta [knowledge/](../knowledge/README.md) contém a documentação autoral,
notebooks, handoffs e configurações que não podem se perder.

O arquivo [knowledge/manifest.json](../knowledge/manifest.json) registra:

- 24 documentos;
- 139.768 bytes;
- caminho relativo;
- título;
- tags;
- hash SHA-256.

Recursos MCP planejados:

- `bim://knowledge/index`;
- `bim://knowledge/document/{path}`;
- `search_bim_knowledge(query, tags)`;
- `list_bim_knowledge(tags)`.

## 6. Biblioteca de engenharia de autoria BIM

A biblioteca está em [bim_authoring/](../bim_authoring/README.md).

Ela transforma conhecimento de modelagem em receitas versionadas com:

- entradas e unidades;
- entidades IFC;
- relações esperadas;
- regras geométricas;
- sequência operacional;
- pós-condições;
- falhas conhecidas;
- referências;
- executor Python, quando implementado.

### Receitas atuais

| Receita | Estado |
|---|---|
| `element.wall-basic` | executável |
| `assembly.window-in-wall` | executável |
| `assembly.door-in-wall` | executável |
| `element.slab-from-polygon` | documentada |
| `connection.wall-corner` | documentada |
| `type.layered-wall-type` | documentada |
| `assembly.skylight-in-slab` | documentada |

“Documentada” significa que a engenharia foi especificada, mas ainda não existe
executor pronto e testado.

### Janela e porta hospedadas

O motor não encosta um sólido visualmente na parede. Ele cria o grafo IFC:

```text
IfcWall
  └─ IfcRelVoidsElement
       └─ IfcOpeningElement
            └─ IfcRelFillsElement
                 └─ IfcWindow ou IfcDoor
```

O motor também cria representação, placement e contenção espacial, e rejeita
vãos que ultrapassem a parede.

### Componentes

- [catalog.py](../bim_authoring/catalog.py): catálogo e busca;
- [geometry.py](../bim_authoring/geometry.py): frames e regras métricas;
- [engine.py](../bim_authoring/engine.py): operações IFC;
- [validation.py](../bim_authoring/validation.py): pós-condições;
- [mcp_adapter.py](../bim_authoring/mcp_adapter.py): payloads para MCP;
- [http.py](../bim_authoring/http.py): descoberta pela API local;
- [recipe.schema.json](../bim_authoring/schemas/recipe.schema.json): contrato das receitas.

### Superfície preparada para MCP

Resources:

- `bim://authoring/recipes`;
- `bim://authoring/recipe/{recipe_id}`.

Endpoints:

- `GET /api/bim-authoring/recipes`;
- `GET /api/bim-authoring/recipes?q=janela`;
- `GET /api/bim-authoring/recipes/{recipe_id}`;
- `GET /api/bim-authoring/mcp-surface`.

Esses payloads ainda não constituem um servidor MCP completo. A camada foi
mantida neutra para poder ser reutilizada pelo Flask e pelo futuro SDK MCP.

## 7. Testes da biblioteca

Comando:

```powershell
python -m unittest discover -s bim_authoring\tests -v
```

Resultado em 24/07/2026:

```text
12 testes executados
12 aprovados
0 falhas
```

Cobertura atual:

- leitura e busca do catálogo;
- serialização dos resources MCP;
- frames ortonormais;
- offsets em eixos locais;
- rejeição de abertura fora da parede;
- sequência IFC para janela e porta;
- relações de abertura e preenchimento;
- validação de placement, geometria e contenção.

Os testes do motor usam um executor IFC injetável. O caso real descrito abaixo
também foi executado com IfcOpenShell nativo.

## 8. Caso real: `doorgrondfloor.dxf`

Fonte:

```text
C:\Users\Rafael\Desktop\Beckend\plantatobim\doorgrondfloor.dxf
```

O arquivo foi executado novamente pelo núcleo atual de
[planta_to_ifc_v1.py](../plantatobim/planta_to_ifc_v1.py). Não foi reutilizado
apenas o IFC antigo.

### Dependências usadas no teste

Foram instaladas de forma isolada em `.runtime/bim_render/`:

- IfcOpenShell 0.8.5;
- ezdxf 1.4.4;
- Shapely 2.1.2;
- dependências auxiliares.

Essa pasta é um runtime local de teste, não parte conceitual da biblioteca. As
dependências distribuíveis continuam descritas em
`requirements-cloud2bim.txt`.

### Geração inicial pelo app atual

O executor reproduzível está em
[run_plan_to_bim_case.py](../plantatobim/run_plan_to_bim_case.py).

Resultado:

| Item | Valor |
|---|---:|
| Escala detectada | `0.0001` |
| Paredes | 17 |
| Portas | 5 |
| Janelas | 0 |
| Slabs | 2 |
| Elementos malhados | 27 |
| Triângulos | 568 |
| Vértices do contorno do slab | 5 |

Artefatos:

- [IFC inicial](../artifacts/doorgrondfloor/current_app/doorgrondfloor_current_app.ifc);
- [modelo JSON](../artifacts/doorgrondfloor/current_app/doorgrondfloor_current_app_model.json);
- [resumo](../artifacts/doorgrondfloor/current_app/doorgrondfloor_current_app_summary.json);
- [vista BIM completa](../artifacts/doorgrondfloor/current_app/doorgrondfloor_current_app_bim_isometric.png);
- [vista sem slabs](../artifacts/doorgrondfloor/current_app/doorgrondfloor_current_app_bim_cutaway.png).

![BIM inicial gerado pelo app](../artifacts/doorgrondfloor/current_app/doorgrondfloor_current_app_bim_isometric.png)

### Diagnóstico encontrado

O contorno do piso e da cobertura era uma faixa diagonal:

- área aproximada: 45,53 m²;
- perímetro aproximado: 41,37 m;
- não cobria as paredes;
- o mesmo erro apareceu no IFC novo e na malha real do IfcOpenShell.

Portanto, o problema não era somente do primeiro PNG. A causa estava no
algoritmo de contorno do slab, baseado em um hull inadequado para esse conjunto
de paredes externas detectadas.

## 9. Revisão R01 solicitada

A revisão está em [artifacts/doorgrondfloor/edited_r01/](../artifacts/doorgrondfloor/edited_r01/).

Operações aplicadas:

1. `W-011` recebeu o mesmo alcance longitudinal de `W-010`;
2. `W-013` recebeu o mesmo alcance longitudinal de `W-010`;
3. `W-018` foi criada no alinhamento traseiro entre `W-010` e `W-013`;
4. `W-019` foi criada no alinhamento frontal entre `W-010` e `W-013`;
5. piso e teto foram reconstruídos pelas faces externas extremas de todas as
   paredes.

### Resultado geométrico

| Elemento | Resultado |
|---|---:|
| `W-010` | 16,4609 m |
| `W-011` | 16,4609 m |
| `W-013` | 16,4609 m |
| `W-018` | 8,84 m |
| `W-019` | 8,84 m |
| Paredes principais | 19 |
| Portas | 5 |
| Slabs | 2 |
| Elementos malhados | 29 |
| Triângulos | 584 |

### Piso e teto

| Propriedade | Valor |
|---|---:|
| Largura | 14,20255 m |
| Profundidade | 17,70275 m |
| Área | 251,42419 m² |
| Perímetro | 63,81 m |
| Arestas | 4 |

Limites:

```text
xmin = 12,5489 m
ymin = 5,5051 m
xmax = 26,75145 m
ymax = 23,20785 m
```

### Validações executadas

- `W-010`, `W-011` e `W-013` possuem exatamente o mesmo comprimento;
- `W-018` e `W-019` existem no JSON e no IFC;
- todas as faces externas das 19 paredes estão dentro do slab;
- o IFC possui 19 paredes principais, 5 portas e 2 slabs;
- piso e teto usam o mesmo contorno;
- o código Python passou por compilação de sintaxe.

### Artefatos R01

- [PNG das paredes](../artifacts/doorgrondfloor/edited_r01/doorgrondfloor_walls_annotated.png);
- [PNG do teto e arestas](../artifacts/doorgrondfloor/edited_r01/doorgrondfloor_ceiling_slab_edges_annotated.png);
- [IFC editado](../artifacts/doorgrondfloor/edited_r01/doorgrondfloor_edited_r01.ifc);
- [modelo editável](../artifacts/doorgrondfloor/edited_r01/doorgrondfloor_edited_r01_model.json);
- [códigos e metragens](../artifacts/doorgrondfloor/edited_r01/doorgrondfloor_geometry_labels.json);
- [resumo da revisão](../artifacts/doorgrondfloor/edited_r01/doorgrondfloor_edited_r01_summary.json);
- [preview PLY](../artifacts/doorgrondfloor/edited_r01/doorgrondfloor_edited_r01_preview.ply).

![Paredes da revisão R01](../artifacts/doorgrondfloor/edited_r01/doorgrondfloor_walls_annotated.png)

![Teto e arestas da revisão R01](../artifacts/doorgrondfloor/edited_r01/doorgrondfloor_ceiling_slab_edges_annotated.png)

O script [edit_doorgrondfloor_enclosure.py](../plantatobim/edit_doorgrondfloor_enclosure.py)
registra essa edição específica de forma reproduzível.

## 10. Renderizadores e ferramentas de diagnóstico criados

### Planta e slab identificados

[render_annotated_ifc_plan.py](../plantatobim/render_annotated_ifc_plan.py):

- lê a geometria STEP produzida pelo modelador;
- identifica paredes como `W-###`;
- identifica arestas como `SLAB-E-###`;
- mede comprimentos, espessuras, área e perímetro;
- gera PNG de paredes;
- gera PNG de piso/teto;
- gera JSON com coordenadas e metragens.

### Vista BIM isométrica

[render_ifc_isometric.py](../plantatobim/render_ifc_isometric.py):

- usa a malha real do IfcOpenShell;
- colore parede, porta, piso e cobertura por classe;
- cria vista com teto explodido;
- cria corte de inspeção com slabs ocultos;
- preserva o IFC original; o deslocamento do teto é somente visual.

## 11. Decisão arquitetural sobre futuras edições

Não devemos criar um novo script especializado para cada ordem do usuário.

O script R01 é uma prova reproduzível e deve virar teste do motor genérico. O
contrato desejado é uma lista declarativa de operações:

```json
{
  "base_revision": "R00",
  "operations": [
    {
      "op": "match_wall_length",
      "targets": ["W-011", "W-013"],
      "reference": "W-010",
      "anchor": "back"
    },
    {
      "op": "connect_walls",
      "from": "W-010",
      "to": "W-013",
      "side": "back",
      "new_id": "W-018"
    },
    {
      "op": "fit_slabs_to_walls",
      "targets": ["floor", "ceiling"],
      "mode": "outer_faces"
    }
  ]
}
```

O motor deverá:

1. carregar a revisão base;
2. resolver IDs estáveis;
3. validar as pré-condições;
4. aplicar as operações;
5. recalcular encontros, aberturas e slabs dependentes;
6. criar uma nova revisão;
7. validar geometria e relações IFC;
8. gerar JSON, IFC e PNG da mesma revisão.

Operações prioritárias:

- `move_wall`;
- `resize_wall`;
- `match_wall_length`;
- `add_wall`;
- `connect_walls`;
- `delete_wall`;
- `insert_opening`;
- `move_opening`;
- `set_wall_thickness`;
- `fit_slabs_to_walls`;
- `set_storey_height`;
- `validate_revision`;
- `export_ifc`;
- `render_revision`.

## 12. Estado real do MCP

### Pronto para ser adaptado

- catálogo de receitas;
- payloads JSON;
- URIs de resources;
- modelos editáveis;
- identificadores de elementos;
- geração de IFC e PNG;
- validações iniciais;
- documentação e manifesto.

### Ainda pendente

- bootstrap `bim_mcp/server.py`;
- persistência formal de projeto e revisões;
- tools MCP de leitura e escrita;
- fila de jobs do Cloud-to-BIM;
- motor genérico de operações;
- autorização e confirmação para edições destrutivas;
- autenticação, uso remoto e cobrança.

## 13. Limitações conhecidas

- O caso DXF apresentou 21 segmentos de parede não pareados.
- Foram reconhecidos 9 blocos de esquadria, mas apenas 5 portas foram associadas
  ao modelo; nenhuma janela foi gerada.
- O cálculo automático original de slab ainda precisa ser corrigido no motor,
  não apenas no caso R01.
- O contorno R01 é um retângulo pelas faces externas extremas. Ele garante
  cobertura total, mas não representa concavidades arquitetônicas.
- Quatro receitas da biblioteca ainda não possuem executor.
- A API local expõe descoberta de receitas, mas não as operações de edição.
- Ainda não há revisão imutável persistida por um `project_store`.
- Os PNGs identificados são diagnósticos; o visualizador 3D continua sendo a
  interface interativa principal planejada.

## 14. Próximos passos recomendados

1. Criar o contrato JSON das operações de edição.
2. Implementar `model_editing.py` independente de Flask e MCP.
3. Transformar a revisão R01 em teste automático do motor genérico.
4. Corrigir o perímetro de slabs com união das faces externas e suporte a
   concavidades.
5. Revalidar portas e blocos não associados do `doorgrondfloor.dxf`.
6. Criar `project_store.py` com revisões imutáveis.
7. Adaptar catálogo, conhecimento e edição ao servidor MCP local.
8. Integrar renderização e download ao visualizador existente.
9. Adicionar mais receitas executáveis: laje, encontro de paredes, materiais e
   claraboia.
10. Validar outros DXFs reais para evitar regras específicas de uma planta.

## 15. Critério de sucesso do produto

O sistema estará pronto para o primeiro uso profissional quando uma ordem em
linguagem natural puder:

- apontar elementos por código;
- ser convertida em operações explícitas;
- mostrar uma proposta antes da escrita;
- gerar uma revisão nova;
- provar as metragens;
- produzir PNG, JSON e IFC coerentes;
- manter rastreabilidade da origem medida, inferida ou projetada;
- permitir validação humana no visualizador.

O caso `doorgrondfloor` é o primeiro teste completo desse ciclo e já serve como
referência de regressão para o futuro motor de edição.
