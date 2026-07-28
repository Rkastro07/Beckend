# Sessão técnica — Cloud-to-BIM e revisão assistida

Data de referência: 27/07/2026  
Encerramento e último teste: madrugada de 28/07/2026  
Pasta de trabalho: `bim-mcp-backend-review`

## 1. Resumo executivo

A sessão consolidou o fluxo local:

```text
E57
  -> detecção geométrica
  -> IFC base
  -> PNG com paredes identificadas
  -> propostas de portas e janelas
  -> revisão humana
  -> IFC V2 com aberturas aprovadas
```

Também foi iniciado um detector experimental para um erro recorrente em
nuvens internas: folhas abertas de portas e janelas podem ser reconstruídas
como pequenas paredes diagonais.

O detector de aberturas apresentou bom resultado na nuvem usada para
desenvolvimento. Entretanto, o teste cego na segunda nuvem revelou uma
limitação importante: o detector automático de falsas paredes não identificou
nenhuma das dez paredes que o usuário marcou como inexistentes.

A prévia final com 18 paredes foi produzida por uma lista de aprovação humana.
Ela não representa aprendizado automático nem melhoria de recall do detector.

## 2. Nuvens utilizadas

### Nuvem de desenvolvimento

```text
C:\Users\Rafael\Desktop\Beckend\dataset\RCP\kladno\kladno_saal- Cloud.e57
```

Essa nuvem foi usada para:

- revisar paredes e junções;
- corrigir a geometria destinada ao cálculo de `IfcSpace`;
- localizar portas e janelas;
- calibrar o Opening Detector V2;
- testar a geração IFC das aberturas aprovadas;
- formular a hipótese das folhas abertas.

### Nuvem de teste cego

```text
C:\Users\Rafael\Desktop\Beckend\dataset\RCP\kladno\kladno_2andar_voxel3cm - Cloudmeiuca.e57
```

O baseline do detector de aberturas foi congelado antes dessa execução. Nenhum
ID da nova nuvem foi usado como regra do detector.

## 3. Revisão geométrica da primeira nuvem

A geometria da primeira nuvem passou por uma revisão assistida em PNG. Entre
as regras confirmadas durante a revisão:

- paredes próximas devem ser prolongadas no próprio eixo quando o intervalo
  não comporta uma porta;
- prolongar uma parede não pode alterar seu ângulo;
- uma conexão perpendicular deve ser criada como uma nova parede;
- paredes mescladas precisam conservar uma espessura física, em vez de virar
  uma linha sem volume;
- o perímetro do piso e do teto deve acompanhar o conjunto final das paredes;
- `IfcSpace` só deve ser recalculado depois que a topologia estiver fechada.

O caso específico da parede `W-005.1` confirmou a diferença entre:

- prolongamento colinear de uma parede; e
- criação de uma nova parede a 90 graus.

Os scripts dessa etapa foram mantidos como ferramentas de diagnóstico e
reprodução. As decisões específicas da obra não devem ser incorporadas como
regras genéricas do detector.

## 4. Opening Detector V2

O detector implementado em
`experiments/cloud2bim/cloud2bim_patched/opening_detector_v2.py` trabalha sem
modelo de linguagem.

Para cada parede, ele:

1. cria um sistema de coordenadas local;
2. projeta os pontos numa grade comprimento × altura;
3. mede vazios e suporte de pontos nas duas faces;
4. distingue candidatos que tocam o piso;
5. estima posição, largura, altura e evidências;
6. produz IDs estáveis para revisão;
7. renderiza a parede em X-Z para auditoria.

Foram acrescentadas duas leituras geométricas:

- `repeated_frame_family`: procura famílias repetidas de janelas por montantes,
  peitoril e verga;
- `wall_axis_topology`: procura portas nos intervalos entre o fim do eixo de
  uma parede e a interseção com outra.

Saídas principais:

```text
opening_detector_v2/opening_candidates_v2.json
opening_detector_v2/opening_candidates_v2_overview.png
opening_detector_v2/walls/W-*.png
```

## 5. Verdade de campo das aberturas

A leitura fornecida pelo usuário foi registrada em:

```text
experiments/cloud2bim/benchmarks/kladno_openings_ground_truth.json
```

Resumo:

| Parede | Verdade de campo |
|---|---|
| `W-S01-001` | 1 porta |
| `W-S01-003` | 2 portas |
| `W-S01-004` | 1 porta |
| `W-S01-012` | 3 janelas |
| `W-S01-015` | 2 janelas |
| `W-S01-016` | porta na região |
| `W-S01-027` | 2 janelas |

Portas topológicas adicionais:

- intervalo entre `W-S01-023` e `W-S01-016`;
- intervalo entre `W-S01-023` e `W-S01-015`;
- intervalo entre `W-S01-001` e `W-S01-005`.

Também foram registradas correções dimensionais:

- em `W-S01-012`, a janela 02 precisava ser reposicionada e as janelas 01 e 03
  precisavam de correção no eixo Z;
- a janela 04 de `W-S01-012` era falso positivo;
- em `W-S01-015`, havia duas janelas; a proposta de porta era falso positivo.

## 6. Baseline congelado antes do teste cego

O baseline foi salvo em:

```text
experiments/cloud2bim/benchmarks/opening_detector_v3_frozen_baseline.json
```

Resultado na nuvem de desenvolvimento, cuja geometria de paredes já havia sido
revisada:

| Métrica | Resultado |
|---|---:|
| Aberturas locais esperadas | 12 |
| Aberturas locais propostas | 12 |
| Correspondências | 12 |
| Precisão no conjunto de desenvolvimento | 1,00 |
| Recall no conjunto de desenvolvimento | 1,00 |
| Portas topológicas esperadas | 3 |
| Portas topológicas encontradas | 3 |

Esses números não são uma métrica de generalização. Eles registram o estado
congelado antes de usar uma nova nuvem.

## 7. IFC Openings Generator V2

O arquivo `experiments/cloud2bim/generate_ifc_openings_v2.py` materializa as
aberturas que já foram mostradas e aprovadas no PNG.

Após a aprovação:

- o grau de confiança deixa de excluir candidatos;
- score e status continuam gravados para auditoria;
- cada abertura cria o encadeamento IFC correto:

```text
IfcWall
  -> IfcRelVoidsElement
  -> IfcOpeningElement
  -> IfcRelFillsElement
  -> IfcDoor ou IfcWindow
```

O gerador preserva paredes, slabs, pavimentos e spaces existentes. Antes de
gravar o arquivo, valida:

- contagem e IDs;
- parede hospedeira;
- relações de vazio e preenchimento;
- representação geométrica;
- contenção espacial.

`render_ifc_openings_v2.py` reabre o IFC gerado e produz uma conferência visual
independente do JSON de propostas.

## 8. Detector experimental de folhas abertas

A hipótese investigada foi:

> portas e janelas abertas para dentro do ambiente aparecem na nuvem como
> planos verticais e podem ser classificadas incorretamente como paredes
> diagonais.

O detector foi implementado em:

```text
experiments/cloud2bim/cloud2bim_patched/wall_detector_v2.py
experiments/cloud2bim/detect_open_leaves.py
```

O candidato automático combina:

- proximidade de uma extremidade ao batente;
- comprimento comparável à largura do vão;
- extremidade oposta livre;
- espessura compatível com painel;
- ângulo articulado em relação à parede hospedeira;
- perfil vertical da nuvem.

O pipeline principal recebeu a opção:

```text
--open-leaf-review
```

Essa opção é não destrutiva. Ela gera JSON e PNG antes/depois, mas não remove
paredes do IFC sem aprovação.

## 9. Resultado do teste cego na segunda nuvem

Saída da execução:

```text
artifacts/blind_test_kladno_2andar_open_leaf_pipeline_v1/
```

### Modelo base

| Item | Resultado |
|---|---:|
| Pontos processados | 1.824.781 |
| `IfcWall` | 28 |
| `IfcSlab` | 2 |
| `IfcBuildingStorey` | 2 |
| `IfcSpace` | 1 |
| `IfcDoor` no IFC base | 0 |
| `IfcWindow` no IFC base | 11 |

### Propostas do Opening Detector V2

| Item | Resultado |
|---|---:|
| Paredes analisadas | 28 |
| Candidatos locais | 24 |
| Candidatos classificados como porta | 14 |
| Candidatos classificados como janela | 10 |
| Portas topológicas | 2 |

Esses valores são propostas para revisão; não equivalem a verdade de campo.

### Resultado automático das folhas abertas

| Item | Resultado |
|---|---:|
| Paredes antes | 28 |
| Falsas paredes encontradas automaticamente | 0 |
| Paredes removidas automaticamente | 0 |
| Paredes após a prévia automática | 28 |

Portanto, o detector experimental teve recall `0/10` frente à revisão humana
posterior.

## 10. Correção humana das paredes inexistentes

O usuário marcou como inexistentes:

```text
W-S01-002
W-S01-009
W-S01-011
W-S01-013
W-S01-017
W-S01-021
W-S01-023
W-S01-026
W-S01-027
W-S01-028
```

A verdade de campo foi salva em:

```text
experiments/cloud2bim/benchmarks/kladno_2andar_wall_ground_truth.json
```

`detect_open_leaves.py` recebeu `--approved-non-walls` para aplicar a decisão
humana de forma reproduzível. A lista:

- entra no JSON como `source: user_ground_truth`;
- recebe `status: approved`;
- produz uma prévia com 18 paredes;
- não altera o IFC de entrada.

Essa mudança evita editar pixels manualmente, mas não ensina o detector a
encontrar as paredes sozinho. Uma métrica de 1,00 calculada depois de injetar
essas aprovações mede apenas a aplicação correta da lista, não o detector.

Artefatos:

```text
artifacts/blind_test_kladno_2andar_open_leaf_pipeline_v1/
  wall_review_approved_v1/open_leaf_candidates.json
  wall_review_approved_v1/open_leaf_before_after.png
```

## 11. Diagnóstico do erro

As dez falsas paredes formam pelo menos dois grupos:

- sete segmentos compactos, entre aproximadamente 0,59 m e 1,32 m:
  `W-002`, `W-009`, `W-011`, `W-013`, `W-017`, `W-021` e `W-023`;
- três segmentos mais longos reconstruídos pelo modo single-line, entre
  aproximadamente 2,04 m e 2,31 m:
  `W-026`, `W-027` e `W-028`.

Todos receberam suporte vertical alto no detector de paredes. Isso demonstra
que persistência em altura, isoladamente, não diferencia uma parede real de
uma folha aberta ou de outro plano vertical interno.

O próximo classificador precisa combinar evidência negativa de parede:

- continuidade topológica com paredes vizinhas;
- compatibilidade com batente e vão;
- relação entre comprimento do plano e dimensão da abertura;
- presença de uma extremidade livre;
- ocupação dos pontos em torno e atrás do plano;
- espessura e suporte em ambas as faces;
- estabilidade do plano fora da região de uma abertura.

As regras devem usar geometria e topologia, nunca os IDs desta obra.

## 12. Código criado ou alterado

Principais arquivos:

```text
experiments/cloud2bim/cloud2bim_patched/opening_detector_v2.py
experiments/cloud2bim/cloud2bim_patched/wall_detector_v2.py
experiments/cloud2bim/run_opening_detector_v2.py
experiments/cloud2bim/detect_open_leaves.py
experiments/cloud2bim/generate_ifc_openings_v2.py
experiments/cloud2bim/render_ifc_openings_v2.py
experiments/cloud2bim/apply_kladno_opening_review.py
experiments/cloud2bim/evaluate_opening_proposals.py
experiments/cloud2bim/rodar.py
experiments/cloud2bim/README.md
```

Testes associados:

```text
experiments/cloud2bim/tests/test_opening_detector_v2.py
experiments/cloud2bim/tests/test_wall_detector_v2.py
```

Benchmarks:

```text
experiments/cloud2bim/benchmarks/kladno_openings_ground_truth.json
experiments/cloud2bim/benchmarks/opening_detector_v3_frozen_baseline.json
experiments/cloud2bim/benchmarks/kladno_2andar_wall_ground_truth.json
```

## 13. Estado ao encerrar a sessão

Concluído:

- detector local de aberturas auditável em PNG;
- famílias repetidas de janelas;
- portas inferidas por topologia;
- geração IFC V2 após aprovação;
- renderização independente do IFC;
- detector experimental de folhas abertas;
- integração dos dois detectores no runner;
- benchmark congelado da primeira nuvem;
- teste cego da segunda nuvem;
- registro das dez falsas paredes;
- prévia reproduzível com as dez remoções aprovadas.

Ainda pendente:

- melhorar a detecção automática das falsas paredes;
- repetir o teste cego sem passar `--approved-non-walls`;
- obter verdade de campo das portas e janelas da segunda nuvem;
- aplicar as remoções aprovadas à topologia do modelo;
- recalcular `IfcSpace`;
- gerar um novo IFC somente depois da aprovação visual;
- verificar se portas e janelas aprovadas continuam coerentes após a remoção
  das falsas paredes.

## 14. Próxima etapa recomendada

1. Extrair características geométricas das 28 paredes da segunda nuvem.
2. Comparar as dez falsas paredes com as 18 paredes confirmadas.
3. Implementar um score de “plano articulado ou objeto interno”.
4. Treinar e ajustar apenas na nuvem de desenvolvimento.
5. congelar novamente o código;
6. executar a segunda nuvem sem IDs nem lista de aprovação;
7. medir precisão e recall reais;
8. apresentar a nova PNG;
9. após aprovação, reconstruir a topologia e recalcular o IFC.

O critério de sucesso da próxima versão é detectar essas estruturas por
geometria transferível, sem depender de um modelo avançado de linguagem e sem
memorizar elementos específicos da planta.
