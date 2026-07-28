# Entrada geométrica do modelador

O modelador não deve tratar todo arquivo como se fosse um DXF. A entrada foi
separada pelo nível de informação que o formato realmente carrega:

| Família | Formatos | Rota | O que é preservado |
|---|---|---|---|
| BIM | IFC, IFCZIP | Modelador 2D/IFC | objetos, GUIDs, nomes, vãos, lajes e pavimentos |
| CAD/vetor | DXF, SVG | Reconhecimento vetorial | coordenadas, layers/grupos, blocos e curvas tesselladas |
| Malha 3D | OBJ, USDZ; depois STL/GLB/glTF | Cloud-to-BIM | vértices, faces e transformações |
| Nuvem | PLY, E57, ASC, XYZ; depois LAS/LAZ | Cloud-to-BIM | pontos e atributos disponíveis |
| CAD proprietário | DWG | Conversor local antes do modelador | geometria CAD após conversão verificável |

Essa separação evita o erro de chamar uma face de malha de “parede BIM” sem
evidência. Malhas e nuvens têm geometria, mas precisam da visão computacional do
Cloud-to-BIM para obter semântica.

## Importadores disponíveis

### IFC e IFCZIP

- seleciona um `IfcBuildingStorey`;
- lê `IfcWall`, `IfcDoor`, `IfcWindow`, `IfcOpeningElement` e `IfcSlab`;
- usa a geometria IFC em coordenadas de mundo para medir eixo, espessura,
  altura e peitoril;
- mantém nome, classe IFC, GUID e pavimento no modelo editável;
- normaliza o Z para o piso selecionado no editor, sem perder a elevação de
  origem nos metadados;
- permite trocar de pavimento sem sair do modelador.

Uma parede curva ou complexa é reduzida a um eixo 2D para caber no editor atual.
O diagnóstico informa quantas geometrias precisaram de aproximação.

### DXF

Além de `LINE` e `LWPOLYLINE`, o leitor agora considera:

- blocos `INSERT`, inclusive a transformação aplicada;
- `POLYLINE` e bulges;
- `ARC`, `CIRCLE`, `ELLIPSE` e `SPLINE`, tessellados em segmentos;
- contornos de `SOLID`, `TRACE` e `3DFACE`;
- layers herdadas de blocos na layer `0`.

As regras multilíngues de layer continuam sendo usadas para distinguir paredes,
portas e janelas.

### SVG

O leitor considera `line`, `polyline`, `polygon`, `rect`, `circle`, `ellipse` e
`path`, incluindo transformações de grupos. Layers/grupos com nomes de parede,
porta e janela fornecem a semântica. Sem essas marcações, a geometria não
classificada é aberta no modo de parede de linha única e o modelo recebe um
aviso.

Unidades físicas (`m`, `cm`, `mm`, `in`, `ft` e `px`) e a relação
`width/viewBox` são consideradas. Quando não há unidade, aplica-se a mesma
heurística de extensão usada no CAD.

## API

### `GET /api/planta/formatos`

Devolve o catálogo completo de formatos, estado e rota correta. Esse catálogo
pode ser publicado diretamente como um resource MCP.

### `POST /api/planta/importar`

Multipart:

- `file`: IFC, IFCZIP, DXF ou SVG;
- `pavimento`: nome ou GUID de um pavimento IFC (opcional);
- `escala`: fator unidade → metro para CAD/SVG (opcional);
- `esp_default`: espessura de parede quando há apenas eixo (padrão `0.15`).

`POST /api/planta/parse` permanece como alias compatível com o front antigo.

## Contrato do modelo editável

O JSON comum contém:

- `source`: formato, família, modo, nível semântico e pavimentos;
- `paredes`: eixo 2D e espessura, mais nome/GUID/altura quando existirem;
- `aberturas`: parede dona, tipo, largura, altura e peitoril;
- `laje`: contorno, ativação e espessuras;
- `diagnostico` e `warnings`: inferências e perdas conhecidas.

Ao gerar um novo IFC, o modelador reutiliza nomes, GUIDs válidos, alturas e
peitoris importados. Elementos novos continuam recebendo identidade nova.
