# Biblioteca de autoria BIM

Esta biblioteca transforma conhecimento de modelagem em **receitas pesquisáveis,
validáveis e executáveis**. Ela é a camada de engenharia entre uma ordem como
“adicione uma janela a 1,20 m do início desta parede” e as entidades, relações,
geometria e verificações que precisam existir no IFC.

As coordenadas e dimensões do motor são sempre expressas em metros. As operações
executáveis usam a API de alto nível do IfcOpenShell; o catálogo, a busca e os
resources funcionam mesmo sem o IfcOpenShell instalado.

## O que já funciona

| Receita | Estado | Executor |
|---|---|---|
| `element.wall-basic` | executável | cria `IfcWall`, corpo, placement e containment |
| `assembly.window-in-wall` | executável | cria abertura, corte semântico e `IfcWindow` |
| `assembly.door-in-wall` | executável | cria abertura, corte semântico e `IfcDoor` |
| `element.slab-from-polygon` | especificada | geometria de laje por polígono |
| `connection.wall-corner` | especificada | encontro de duas paredes |
| `type.layered-wall-type` | especificada | tipo de parede e camadas de material |
| `assembly.skylight-in-slab` | especificada | abertura e claraboia em laje/cobertura |

“Especificada” significa que a receita já possui entradas, entidades, relações,
passos, checagens, falhas conhecidas e referências, mas ainda não tem executor.
Isso evita dizer que uma operação está pronta antes de existir código e teste.

## Modelo de engenharia

Uma janela não é apenas um sólido encostado na parede. O grafo mínimo é:

```text
IfcWall
  └─ IfcRelVoidsElement
       └─ IfcOpeningElement
            └─ IfcRelFillsElement
                 └─ IfcWindow
```

O motor também cria as representações geométricas e placements. A abertura
atravessa a espessura do hospedeiro com uma margem booleana controlada. Antes de
criar qualquer entidade, valida se largura, altura, peitoril e afastamento cabem
na parede. Depois, `validate_filling` inspeciona relações, hospedeiro,
representações, placements, dimensões e contenção espacial.

## Uso em Python

```python
from bim_authoring import IfcAuthoringEngine, load_default_catalog

catalog = load_default_catalog()
receita = catalog.get("assembly.window-in-wall")

engine = IfcAuthoringEngine(model)
wall = engine.create_wall(
    start=(0.0, 0.0),
    end=(5.0, 0.0),
    height=2.80,
    thickness=0.15,
    body_context=body_context,
    storey=storey,
)
assembly = engine.insert_window(
    wall,
    offset_from_start=1.20,
    width=1.50,
    height=1.20,
    sill_height=1.00,
)
```

O arquivo IFC ainda precisa possuir o projeto, as unidades, o contexto
geométrico e a estrutura espacial. O motor recebe `body_context` e `storey`
explicitamente para não esconder essas decisões.

## Contrato das receitas

Cada JSON em `recipes/` segue `schemas/recipe.schema.json` e registra:

- identidade, versão, status e versões IFC;
- entradas com unidade, limites e obrigatoriedade;
- entidades e relações IFC esperadas;
- regras geométricas e sistema de coordenadas;
- sequência operacional;
- pós-condições verificáveis;
- falhas conhecidas e política de recuperação;
- referências técnicas;
- executor Python, quando já implementado.

O catálogo permite filtrar por tags, estado e versão IFC, e buscar em português:

```python
from bim_authoring import load_default_catalog

catalog = load_default_catalog()
resultados = catalog.search("janela parede")
implementadas = catalog.list(status="implemented")
```

## Superfície local e MCP

O bootstrap Flask registra:

- `GET /api/bim-authoring/recipes`;
- `GET /api/bim-authoring/recipes?q=janela&status=implemented`;
- `GET /api/bim-authoring/recipes/assembly.window-in-wall`;
- `GET /api/bim-authoring/mcp-surface`.

O módulo `mcp_adapter.py` já entrega payloads JSON e os resources:

- `bim://authoring/recipes`;
- `bim://authoring/recipe/{recipe_id}`.

O futuro servidor MCP deve apenas adaptar essas funções ao SDK. Ferramentas que
alteram modelos devem receber um projeto e uma revisão, gravar uma nova revisão
e nunca sobrescrever silenciosamente a anterior.

## Estrutura

```text
bim_authoring/
  catalog.py          catálogo e busca
  geometry.py         frames e pré-condições métricas
  engine.py           operações IFC executáveis
  validation.py       pós-condições semânticas
  mcp_adapter.py      resources e contratos serializáveis
  http.py             descoberta pela API local
  schemas/            contrato JSON Schema
  recipes/            conhecimento operacional
  tests/              testes sem dependência nativa do IfcOpenShell
```

## Referências de implementação

- [IfcOpenShell — criação de geometria](https://docs.ifcopenshell.org/ifcopenshell-python/geometry_creation.html)
- [IfcOpenShell — adicionar abertura](https://docs.ifcopenshell.org/autoapi/ifcopenshell/api/feature/add_feature/index.html)
- [IfcOpenShell — preencher abertura](https://docs.ifcopenshell.org/autoapi/ifcopenshell/api/feature/add_filling/index.html)
- [buildingSMART — IfcWindow](https://ifc43-docs.standards.buildingsmart.org/IFC/RELEASE/IFC4x3/HTML/lexical/IfcWindow.htm)
- [buildingSMART — padrões openBIM](https://technical.buildingsmart.org/standards/)
- [buildingSMART Data Dictionary API](https://technical.buildingsmart.org/services/bsdd/using-the-bsdd-api/)
