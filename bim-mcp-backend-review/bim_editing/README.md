# Motor determinístico de edição BIM

Este pacote converte correções do cliente em operações geométricas
reutilizáveis. Nenhuma operação exige um modelo de linguagem avançado e nenhum
script novo deve ser criado para uma parede específica.

## Princípio

```text
ordem humana ou formulário
  -> JSON de operações
  -> RevisionEngine
  -> modelo R01 + relatório + índice de partes
  -> PNG geral + PNG de edição
  -> IFC após aprovação
```

O modelo de linguagem, quando usado, só transforma uma frase curta no contrato
JSON. Coordenadas, interseções, espessuras, topologia, spaces e slabs são
calculados pelo código.

## Níveis verticais da nuvem

O `ceiling_detector_v1` separa o forro regional da laje estrutural. O modelo
editável recebe a altura estrutural das paredes, a altura interna dos spaces,
a espessura das lajes e a configuração de `IfcCovering/CEILING`.

`height_band_min/max` do detector de paredes é apenas uma faixa de evidência;
ela não é usada como altura construtiva. Consulte
[`../docs/CEILING_DETECTOR_V1.md`](../docs/CEILING_DETECTOR_V1.md).

## Partes estáveis

Cada parede expõe:

```text
W-S01-005.P1
W-S01-005.P2
W-S01-005.AXIS
```

Na primeira importação, a ordem é normalizada por coordenadas. Depois disso,
`P1` e `P2` persistem mesmo quando as coordenadas mudam. Se o eixo inicial
precisar ser invertido, os offsets das portas e janelas são invertidos junto.

O arquivo `element_parts.json` registra seletores e coordenadas de cada revisão.

## Operações disponíveis

- `delete_elements`;
- `add_wall`;
- `move_wall_endpoint`;
- `connect_endpoint`;
- `move_wall`;
- `set_wall_thickness`;
- `merge_walls`;
- `add_opening` / `insert_opening`;
- `move_opening`;
- `resize_opening`;
- `set_opening_type` / `change_opening_type`;
- `copy_opening_pattern`;
- `close_wall_junctions`;
- `close_small_gaps`.

Ao converter uma janela em porta, `set_opening_type` preserva por padrão a
cota superior observada: a nova altura recebe `altura + peitoril` e o peitoril
passa a zero. Isso pode ser desligado com `preserve_head: false` ou substituído
por `height`/`sill` explícitos.

`copy_opening_pattern` projeta os centros das aberturas perpendicularmente de
uma parede de referência para uma parede paralela. `close_wall_junctions`
fecha encontros em L (duas pontas) e T (ponta contra o eixo), sempre sem mudar
o ângulo das paredes. Por padrão ele bloqueia uma união que atravesse ou caia
sobre uma porta/janela hospedada.

O schema fica em
[`schemas/revision.schema.json`](schemas/revision.schema.json).

## Exemplos

### Mover P2 até a interseção com outra parede

```json
{
  "op": "connect_endpoint",
  "selector": "W-S01-003.P2",
  "target": {
    "element": "W-S01-010",
    "mode": "axis_intersection"
  }
}
```

O eixo da parede de origem é preservado. O motor calcula a interseção das duas
retas; ele não cria uma diagonal entre os pontos mais próximos.

### Criar uma nova parede perpendicular

```json
{
  "op": "add_wall",
  "id": "W-S01-005.1",
  "from": "W-S01-005.P1",
  "direction": {
    "perpendicular_to": "W-S01-005"
  },
  "until": "W-S01-003",
  "thickness": 0.15
}
```

### Remover elementos

```json
{
  "op": "delete_elements",
  "ids": ["W-S01-002", "W-S01-009", "W-S01-017"]
}
```

Portas e janelas hospedadas nas paredes removidas também são removidas. As
outras estruturas permanecem no modelo.

### Mesclar volumes

```json
{
  "op": "merge_walls",
  "ids": ["W-S01-010", "W-S01-026"],
  "target_id": "W-S01-010"
}
```

A espessura final cobre a extensão física transversal das duas paredes. Ela
não é convertida numa linha sem espessura.

## Recalcular dependências

Uma revisão pode solicitar:

```json
{
  "recalculate": [
    "openings",
    "topology",
    "spaces",
    "slabs",
    "validation"
  ]
}
```

- aberturas órfãs são removidas;
- aberturas fora de uma parede encurtada são ajustadas ou rejeitadas;
- o grafo planar é reconstruído;
- faces fechadas viram spaces;
- piso e teto são ajustados ao hull das faces externas das paredes;
- IDs, medidas, hospedagem e cobertura do slab são validados.

O motor não inventa paredes para fechar uma planta. Se a rede continuar
aberta, o resultado informa `0 spaces`. O cliente pode então indicar
explicitamente quais pontas devem ser conectadas.

## PNG sem excesso de informação

O renderizador gera duas camadas:

- `revision_overview.png`: todas as paredes com IDs compactos;
- `revision_edit_endpoints.png`: `P1/P2` apenas nas paredes selecionadas.

Portas e janelas permanecem visíveis geometricamente. Seus IDs só aparecem
quando a abertura é selecionada.

## CLI

Importar a saída do Cloud-to-BIM:

```powershell
python -m bim_editing.cli import-cloud `
  artifacts\caso\wall_diagnostics.csv `
  artifacts\caso\base_model.json `
  --openings artifacts\caso\opening_detector_v2\opening_candidates_v2.json `
  --vertical-levels artifacts\caso\ceiling_detector_v1\vertical_levels.json
```

Aplicar uma revisão:

```powershell
python -m bim_editing.cli apply `
  artifacts\caso\base_model.json `
  bim_editing\examples\kladno_2andar_remove_non_walls.json `
  artifacts\caso\revision
```

Adicionar `--export-ifc caminho.ifc` usa o modelador Planta-to-BIM existente
depois que a revisão já foi aprovada.

## HTTP e MCP

Rotas locais:

- `GET /api/bim-editing/operations`;
- `GET /api/bim-editing/mcp-surface`;
- `POST /api/bim-editing/resolve`;
- `POST /api/bim-editing/apply`.

As descrições MCP ficam em `mcp_adapter.py`. A aplicação envia o modelo e a
revisão, recebendo um novo modelo, relatório e índice de partes. O modelo-base
não é alterado.

## Teste real da segunda nuvem

O contrato
[`examples/kladno_2andar_remove_non_walls.json`](examples/kladno_2andar_remove_non_walls.json)
aplicou as dez remoções aprovadas:

```text
28 -> 18 paredes
24 -> 20 propostas de abertura
0 spaces, pois a rede restante ainda possui extremidades abertas
revisão válida
```

O resultado foi produzido por uma única operação `delete_elements`, não por
edição manual da PNG e não por código específico para cada ID.
