# Cloud2BIM adaptado — scan → IFC sem ML

Pipeline geométrico de scan-to-BIM baseado no [Cloud2BIM](https://github.com/VaclavNezerka/Cloud2BIM)
(Zbirovský & Nežerka, CTU Prague, *Automation in Construction* 177, 2025 — MIT), com
adaptações validadas nos nossos dados (scan real RCP RetroPorto + sintéticos Allplan).

## Uso

```bash
python rodar.py <nuvem.ply|.e57|.xyz> [--thr 0.3] [--saida DIR]
                [--slab-detector v1_refined] [--single-line|--sem-single-line]
                [--sem-pilar] [--sem-escada] [--opening-detector-v2-review]
```

Um comando faz tudo: conversão → autodiagnóstico de threshold → pipeline
(lajes → pavimentos → paredes → aberturas → cômodos) → detector de pilar →
detector/montador de escada →
IFC + plantas por pavimento em `<nuvem>_cloud2bim/`.

Dependências: `numpy open3d opencv-python scikit-image scipy ifcopenshell pyyaml matplotlib pandas e57 tqdm`

## O que foi adaptado em relação ao Cloud2BIM original

Tudo em `cloud2bim_patched/` (original intacto quando as env vars não são setadas):

| Adaptação | Por quê | Env var |
|---|---|---|
| Threshold de laje configurável | O 0.6 hardcoded falha em galpão: teto domina a densidade (39% dos pontos) e a régua relativa exclui o piso → 0 pavimentos | `SLAB_THR` (autodiagnosticado pelo runner) |
| Projeção de altura cheia | Original só olha 85–120% do pé-direito → perde divisória que não toca o teto | `WALL_ZLO`/`WALL_ZHI` (runner usa 0.1–0.9) |
| Resgate single-line | Parede vista de UM lado só (vidro/scan interno) era descartada; cria face sintética deslocada pro lado oposto ao centroide. RCP real: 14→53 paredes | `SINGLE_LINE=1`, `SINGLE_LINE_MINLEN` (1.0m — essencial, sem ele 1042 falsos), `SINGLE_LINE_THK` |
| Borda de bin inclusiva no histograma de laje | Superfície sintética perfeitamente plana cai na borda do bin e some com `<` estrito (0.4mm de fase mudavam o resultado) | sempre ativo |
| Lajes V1 com bordas refinadas | Preserva o footprint do V1, corrige auto-interseções e representa regiões desconectadas como sólidos válidos da mesma laje | `SLAB_DETECTOR=v1_refined` (padrão) |
| `assign_points_to_walls` com min/argmin incremental | Original materializava matriz paredes×pontos → OOM de 200GB com muitas paredes | sempre ativo |
| Fix `max2` unbound no `identify_openings` | Parede com <3 bins de altura crashava | sempre ativo |
| `usetex=False` + `plt.show→close` | Rodar headless sem LaTeX | `MPLBACKEND=Agg` |

## Detector de paredes V2

O runner e o preview usam `WALL_DETECTOR=v2` por padrão. O fluxo novo separa
evidência de geometria final:

1. rasteriza a nuvem em várias fatias de altura e conserva somente células
   persistentes;
2. vetoriza os contornos 2D sem descartar fragmentos curtos cedo demais;
3. une intervalos colineares; lacunas grandes só fecham quando alguma fatia tem
   pontos no intervalo (porta, janela ou oclusão parcialmente observada);
4. pareia faces estritamente 1:1;
5. refina direção, espessura e limites com TLS robusto nos pontos originais;
6. mede cada face como uma matriz altura × comprimento: a receita usa seis
   camadas, exige persistência em quatro e pelo menos 25% de cobertura nas duas
   camadas superiores; portas e janelas continuam válidas porque suas lacunas
   são localizadas ao longo da parede;
7. deduplica volumes paralelos sobrepostos, preferindo a parede com maior
   suporte vertical;
8. rejeita eixos finais sem suporte de pontos antes de gerar o IFC.

Para comparar com o comportamento anterior, use `WALL_DETECTOR=legacy`. Os
principais ajustes métricos do V2 são `WALL_V2_MAX_GAP`,
`WALL_V2_MAX_UNSEEN_GAP`, `WALL_V2_CLOSE_M`, `WALL_V2_MAX_SNAP`,
`WALL_V2_MIN_POINTS_PER_M`, `WALL_V2_MIN_TOP_FACE_COVERAGE` (0,25),
`WALL_V2_FACE_PERSISTENT_SLICES` (4) e
`WALL_V2_DEDUP_OVERLAP_RATIO` (0,60). A máscara inicial exige suporte em 3/6
fatias de altura; a validação final é individual por face e por trecho.
`single-line` usa pelo menos 1,5 m e a razão comprimento/espessura mínima é 2,5
(`WALL_V2_MIN_ASPECT`).

## Opening Detector V2 (revisao assistida)

`--opening-detector-v2-review` executa, depois do IFC, um detector que nao
altera o modelo. Cada parede e retificada numa grade X-Z com altura relativa ao
topo real do piso. O detector mede vazios, contato com o piso, suporte nas faces,
largura, altura e distancia da extremidade. As saidas sao:

- `opening_detector_v2/opening_candidates_v2.json`, com IDs estaveis,
  parede hospedeira, offset, dimensoes, confianca e evidencias;
- `opening_detector_v2/opening_candidates_v2_overview.png`, com propostas de
  alta confianca e candidatos que exigem revisao;
- `opening_detector_v2/walls/W-*.png`, a imagem X-Z auditavel de cada parede.

Somente candidatos aprovados devem entrar no contrato manual `aberturas` e
virar `IfcOpeningElement`, `IfcDoor` ou `IfcWindow`. O benchmark inicial fica em
`benchmarks/kladno_openings_ground_truth.json`.

O detector também possui duas leituras geométricas determinísticas, sem modelo
de linguagem e sem IDs específicos de obra:

- `repeated_frame_family`: reconhece famílias repetidas de janelas pelos
  montantes verticais, peitoril e verga na grade X-Z, inclusive quando existem
  pontos de fundo visíveis através do vidro;
- `wall_axis_topology`: reconhece um vão de porta quando o eixo de uma parede
  termina entre 0,65 m e 1,45 m antes de interceptar outra parede.

As portas topológicas são gravadas em `topology_candidates` no mesmo JSON. A
camada `apply_kladno_opening_review.py` é apenas um gabarito de desenvolvimento
e não participa da execução genérica nem do teste cego.

## IFC Openings Generator V2 (após aprovação do PNG)

`generate_ifc_openings_v2.py` materializa o JSON correspondente ao PNG
aprovado. Nesta etapa a confiança deixa de ser filtro: todos os candidatos
visíveis e aprovados viram elementos IFC, enquanto score e status permanecem
em `Pset_Cloud2BIMOpeningV2` apenas para auditoria.

O gerador substitui as aberturas antigas e cria, para cada candidato:

`IfcWall -> IfcRelVoidsElement -> IfcOpeningElement ->
IfcRelFillsElement -> IfcDoor/IfcWindow`.

Ele também preserva slabs, paredes, pavimentos e spaces do IFC de entrada. O
arquivo é validado por contagem, IDs, relacionamentos, geometria e container
espacial antes de ser gravado. `render_ifc_openings_v2.py` reabre esse IFC e
produz uma conferência visual independente do JSON.

## Detector de folhas abertas (experimental)

`--open-leaf-review` executa o `Opening Detector V2` e depois procura paredes
curtas que podem ser folhas abertas de portas ou janelas. O teste exige, em
conjunto:

- uma extremidade compatível com um dos batentes da abertura;
- comprimento compatível com a largura do vão;
- outra extremidade livre;
- espessura de painel;
- ângulo articulado em relação à parede hospedeira;
- perfil vertical da nuvem, quando ele não está contaminado por pontos de fundo.

O estágio gera `open_leaf_detector/open_leaf_candidates.json` e
`open_leaf_detector/open_leaf_before_after.png`. Ele não altera o IFC: as
supressões são propostas para a mesma aprovação humana usada no PNG de
aberturas. Após a aprovação, `keep_non_leaf_wall_indices` fornece os índices
para reconstruir a topologia antes do cálculo de `IfcSpace`.

## Detector de lajes V1 refinado

O runner usa `--slab-detector v1_refined` por padrão. Ele conserva a projeção,
o pareamento de superfícies e o footprint proposto pelo Cloud2BIM V1.03, mas:

- corrige anéis auto-intersectantes antes de criar o IFC;
- representa regiões desconectadas como sólidos distintos da mesma `IfcSlab`;
- remove somente ilhas menores que `SLAB_MIN_COMPONENT_AREA` (0,5 m²);
- simplifica o serrilhado com `SLAB_EDGE_SIMPLIFY_M` (0,12 m), limitado por
  `SLAB_EDGE_MAX_DEVIATION_M` (0,15 m).

Para comparação A/B, `--slab-detector v1` mantém o contorno V1 bruto e
`--slab-detector grouped` mantém o agrupamento experimental anterior.
O parâmetro `--slab-wall-reference v1` mantém as fachadas independentes da
limpeza da laje; `--slab-wall-reference refined` permite que o novo contorno
também auxilie a reconstrução de paredes externas.

## Detector de pilar (`detect_pilar.py`, nosso)

O Cloud2BIM não detecta pilar (só tem a casca de export comentada). O nosso usa
perfil de ocupação vertical: célula full-height + blob compacto + **seção constante
em z** + forma de pilar (lado ≥15cm, aspecto ≤2.5). Validado no RCP real: 1 pilar
existente, 1 detectado, 0 falsos (racks caem pela constância; armários pela forma).
O runner roda por pavimento e insere `IfcColumn` no IFC.

## Limitações conhecidas (mapeadas, sem detector ainda)

1. **Laje curva/abobadada**: telhado em arco vira fatia plana ARBITRÁRIA (a premissa
   "laje = pico no histograma de z" é falsa pra superfície curva, que espalha os
   pontos por toda a faixa). **Protótipo funcionando**: `prototipos/detect_casca_curva.py`
   — mapa de altura (casca = célula fina, parede = grossa) → segmentação em trechos
   de perfil constante em x → fit de arco Kasa por trecho → `IfcRoof BARREL_ROOF`
   + remoção da laje espúria. Validado no Allplan: 3 barris (2 iguais + núcleo mais
   alto), RMS 1.7–2.6 cm. Falta integrar no `rodar.py`.
2. **Escada**: lances retos e escada em U/ferradura estão integrados pelo
   `prototipos/montar_escada_gabarito.py`, incluindo o vão na laje. A receita
   padrão usa célula de 12 cm e área mínima de 0,8 m², ajustáveis por
   `--stair-cell` e `--stair-area-min`. Escada caracol continua apenas como
   protótipo em `prototipos/track_espiral.py`.
3. **Viga**: protótipo pausado (banda alta + pareamento; confusor = dutos).
4. **Split-level/mezanino**: melhora com densidade, mas a solução de fundo é
   detecção local de laje por região (não histograma global).
5. **Esquadria pobre**: janela/porta do IFC deles = chapa de 1cm. Decisão de
   arquitetura: este pipeline é só DETECÇÃO; o IFC rico sai do `plantatobim`
   (`construir_ifc`) via adapter scan→ModeloPlanta (a fazer).

## Estrutura

```
rodar.py              runner único (use este)
detect_pilar.py       detector de pilar (módulo, sem hardcode)
cloud2bim_patched/    Cloud2BIM com as adaptações acima (MIT, atribuição mantida)
prototipos/           detector/montador de escada, escada caracol, fatiador, diagnósticos
configs_exemplo/      configs usados nos testes (RCP real, sim, Allplan 4 andares)
```

## Resultados de referência

| Nuvem | Pavimentos | Paredes | Janelas | Tempo |
|---|---|---|---|---|
| RCP RetroPorto (e57 real, 32M pts) | 1 | 53 | 12 | ~5 min |
| Allplan Institute completo (1.8M pts) | 6 (incl. split-level) | 53 | 24 | ~12 s |
