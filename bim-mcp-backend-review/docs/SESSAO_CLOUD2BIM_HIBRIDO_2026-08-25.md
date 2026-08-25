# Sessão Cloud-to-BIM híbrido — 2026-08-25

## Resultado do dia

Foi integrado ao backend e ao frontend um fluxo de revisão Cloud-to-BIM que
combina geometria heurística, processamento em blocos sobrepostos, YOLO-World-M
para portas/janelas e uma rede classificadora de candidatos a parede.

O princípio do fluxo é conservador: a geometria heurística continua sendo a
fonte dos eixos e da escala; os modelos neurais acrescentam classificação e
evidência visual. Nenhuma parede suspeita é removida automaticamente antes da
confirmação no editor.

## Arquitetura implementada

```text
E57/PLY/XYZ
  -> upload e armazenamento temporário da nuvem completa em NPY float32
  -> detecção rápida de pavimentos e paredes para a prévia
  -> rota POST /api/scan/hibrido
     -> blocos XY sobrepostos (core + halo)
     -> geometria heurística de paredes por bloco
     -> histogramas verticais das paredes
     -> YOLO-World-M detectando portas/janelas diretamente nos histogramas
     -> costura determinística dos blocos
     -> classificador ML dos candidatos a parede na densidade completa
  -> PNG + decisões JSON + modelo JSON
  -> confirmação humana no editor 2D/3D
  -> IFC/DXF somente após aprovação
```

### Distinção importante

Nas paredes já existe fusão de métodos:

```text
heurística gera candidato -> ML classifica/revisa
```

Nas aberturas ainda não existe fusão completa. O fluxo atual é:

```text
heurística gera a parede -> YOLO encontra a abertura do zero no histograma
```

O detector heurístico de vãos antigo não está alimentando o YOLO. Portanto, se
o YOLO não criar uma caixa, nenhuma abertura é enviada ao editor. Uma possível
evolução, ainda não implementada, é:

```text
heurística propõe vãos -> YOLO classifica cada candidato -> regras geométricas
confirmam e associam o vão à parede-mãe
```

## Backend

- Nova rota `POST /api/scan/hibrido` em `scan_endpoints.py`.
- O job usa o mesmo mecanismo assíncrono de `/api/scan/job/<id>`.
- A sessão preserva a nuvem completa em um NPY temporário compacto para o
  processamento híbrido; o upload original continua sendo descartado.
- A limpeza automática remove tanto o cache da sessão quanto a nuvem NPY após
  expiração.
- O retorno inclui contagens, estágio, tempo, paredes com classe/probabilidade,
  aberturas e URLs dos três artefatos de revisão.
- `automatic_geometry_change` permanece `false`.

Artefatos por execução:

- `*_hibrido_ml.png`: revisão visual combinada;
- `*_hibrido_ml.json`: decisão do classificador de paredes;
- `*_hibrido_model.json`: paredes costuradas e aberturas detectadas.

## Processamento em blocos

O pipeline rápido usa blocos XY sobrepostos para manter densidade local e
reduzir custo de execução. A configuração padrão atual usa core de 12 m, halo
de 2 m e processamento paralelo de até três blocos.

Etapas principais:

1. `build_overlapping_xy_tiles.py` cria o manifesto e os blocos.
2. `cloud2entities.py`, no modo `CLOUD2BIM_GEOMETRY_ONLY=1`, produz apenas os
   diagnósticos geométricos de parede de cada bloco.
3. `prepare_tiled_wall_models.py` converte os diagnósticos em modelos locais.
4. `run_yoloworld_wall_tokens_tiled_batch.py` carrega o YOLO uma única vez e
   processa os histogramas em lote.
5. `stitch_tiled_cloud2bim.py` une eixos colineares e remapeia aberturas para as
   paredes globais.
6. `run_wall_candidate_classifier_real.py` revisa os candidatos costurados na
   nuvem completa.

## Frontend e editor

O painel `Scan -> BIM` ganhou a seção **ML + heurística em blocos**.

Fluxo validado no navegador:

1. carregar uma nuvem;
2. selecionar o pavimento detectado;
3. clicar em **Rodar ML + heurística**;
4. acompanhar o estágio do job;
5. abrir PNG/JSONs, se necessário;
6. clicar em **Confirmar híbrido no editor**;
7. revisar cada elemento antes de gerar IFC ou DXF.

Cores no editor:

- verde: parede;
- laranja: folha de porta;
- vermelho: provável falso candidato;
- cinza: incerto;
- pontos verdes: portas;
- pontos azuis: janelas.

O painel de propriedades mostra a classe ML, a probabilidade e avisa que a
decisão final é humana. Todos os candidatos geométricos são enviados ao editor,
inclusive os suspeitos.

## Pesos locais

Os pesos são artefatos de treinamento e continuam ignorados pelo Git.

| Função | Caminho esperado | Tamanho | SHA-256 |
|---|---|---:|---|
| Classificador de paredes | `artifacts/cloud2bim_wall_candidate_training_v2/best.pt` | 4.717.982 bytes | `1B90423BFB9C29949F7F22BA2EDA057202B52BF7CB21F3510B905F0B12EF8685` |
| YOLO-World-M para tokens de parede | `artifacts/cloud2bim_yoloworld_m_training/wall_tokens_m_1280_v1/weights/best.pt` | 57.151.465 bytes | `CB802ADD73B0F3181496DFC14449A3E0C70C41BE7259D6E58C29819E4DD29A1F` |

O Python da ML pode ser configurado por `CLOUD2BIM_ML_PYTHON`. No ambiente de
desenvolvimento atual existe também o fallback local
`.codex_tmp/yolo_world_zero_shot/venv/Scripts/python.exe`.

## Benchmarks independentes

Saal e RCP são nuvens de pontos diferentes. Os resultados abaixo não devem ser
misturados nem interpretados como partes da mesma edificação.

### Benchmark Saal

Nuvem:

`C:/Users/Rafael/Desktop/Beckend/dataset/RCP/kladno/kladno_saal- Cloud.e57`

IFC de referência fornecido pelo usuário:

`C:/Users/Rafael/Desktop/Beckend/dataset/RCP/kladno/65bff2e39777_5b25b8c91da6_kladno_saal-_Cloud_scan2bim.ifc`

Execução real pelo frontend:

- 1.326.472 pontos;
- 56,0 s;
- 24 candidatos de parede;
- 23 classificados como parede;
- 1 classificado como folha;
- 6 portas e 11 janelas.

Referência:

- 21 paredes;
- 7 portas;
- 9 janelas;
- 2 lajes e 4 espaços;
- caixa modelada aproximada de 18,53 x 18,66 x 3,15 m.

Comparação:

- 15 das 16 aberturas de referência tiveram correspondência de posição e
  classe;
- faltou uma porta próxima de `(-19,05; 17,92)`;
- duas janelas foram falsos positivos, próximas de `(-23,35; 19,65)` e
  `(-11,57; 20,62)`;
- as 15 paredes principais ficaram praticamente coincidentes em eixo e ângulo;
- os elementos mais ambíguos são trechos curtos e grossos entre 0,60 e 0,75 m,
  que sugerem uma classe futura `column_or_thick_wall`.

### Benchmark RCP

Nuvem independente usada no teste:

`artifacts/blind_test_rcp_voxel3cm_pipeline_v1/nuvem.xyz`

IFC de referência fornecido pelo usuário:

`C:/Users/Rafael/Desktop/Beckend/dataset/RCP/kladno/rcp.ifc`

Execução direta da rota híbrida atual:

- 1.119.465 pontos;
- 37,39 s;
- 19 candidatos de parede;
- 18 classificados como parede;
- 1 incerto/provável não-parede;
- nenhuma porta ou janela detectada.

Referência:

- 17 paredes;
- 7 portas;
- 4 janelas;
- 2 lajes e 5 espaços;
- caixa modelada aproximada de 32,69 x 11,34 x 2,25 m.

Comparação:

- 13 das 17 paredes tiveram correspondência um-para-um;
- algumas paredes longas foram truncadas durante detecção/costura;
- elementos desconectados ou com pouco suporte não produziram candidatos;
- o YOLO atual teve recall zero para as 11 aberturas desse padrão de nuvem.

O detector heurístico antigo desse caso produziu 23 propostas de abertura, mas
somente cinco ficaram próximas de uma abertura da referência e apenas quatro
tiveram a classe correta. Isso mostra dois extremos complementares: a heurística
antiga tem excesso de propostas e o YOLO atual ficou conservador demais.

## Validações executadas

- compilação Python dos módulos alterados;
- `npm run build` do frontend;
- teste real da rota híbrida com a Saal;
- teste completo pelo frontend em `http://localhost:3000`;
- abertura do resultado Saal no editor com 24 paredes e 17 esquadrias;
- teste real da rota híbrida com o caso RCP;
- auditoria geométrica dos dois IFCs de referência.

## Limitações conhecidas

1. Aberturas dependem exclusivamente das caixas produzidas pelo YOLO.
2. O padrão visual do RCP ainda não está representado adequadamente no treino.
3. A costura ainda pode truncar paredes longas ou preservar segmentos internos
   falsos.
4. Componentes desconectados precisam ser preservados explicitamente no
   manifesto/costura.
5. Colunas, pontas grossas de parede e folhas ainda não possuem taxonomia BIM
   completa.
6. Os pesos precisam de distribuição própria ou Git LFS antes de implantação.

## Próximos passos sugeridos

1. Formalizar Saal e RCP num avaliador automático por IFC, sempre separados.
2. Gerar candidatos heurísticos de vão e usar o YOLO para classificá-los, sem
   perder a detecção direta do YOLO.
3. Corrigir cobertura de componentes desconectados e continuidade de paredes
   longas entre blocos.
4. Criar classe específica para coluna/ponta grossa de parede.
5. Adicionar métricas de precisão, recall, erro de eixo, erro de comprimento e
   erro de espessura a cada execução.
6. Definir como versionar e distribuir os dois checkpoints treinados.
