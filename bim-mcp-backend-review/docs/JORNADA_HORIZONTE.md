# Jornada Horizonte — do "tudo funciona em sintético" ao "agora bate no real"

> Documento de retrospectiva: por que levamos tantos dias entre receber o primeiro
> PLY da Horizonte e ter um resultado visualmente convincente no front v2.
> Inclui os problemas reais (não previstos nas simulações) que mudaram o pipeline.

---

## TL;DR

Antes da Horizonte, a gente validava o pipeline em **scans sintéticos** gerados a partir do IFC: nuvem perfeita, sem ruído, sem oclusão, geometria casando 1:1 com o modelo. Era um teste de circuito fechado — fácil acertar quando você desenha o gabarito da própria prova.

Quando a primeira nuvem **real** (Faro da obra) chegou, vários pressupostos quebraram. Cada quebra exigiu mudança de pipeline. O resultado de hoje (89% COMPLETO no sintético da Horizonte, com Sonata + OBB + propagação regressiva) é fruto de **diagnosticar e corrigir esses descasamentos um a um** — não de um upgrade único de modelo.

---

## Linha do tempo (alto nível)

1. **Pipeline antigo funcionando em sintético** — Sonata + DBSCAN + Hungarian matching + RF router. ~80% de acerto em scans simulados perfeitos.
2. **Chegou PLY real Faro da Horizonte** — primeiros prints mostraram que o pipeline antigo desalinhava, fazia matches errados, marcava paredes existentes como ausentes.
3. **Tentativa de melhorar alinhamento (V3)** — FGR + ICP, várias iterações. Cada ajuste mexia em outro lugar. Eventualmente revertido por instabilidade.
4. **Pivô conceitual** — usar **OBB do IFC como recorte espacial** + Sonata só pra classificar dentro da OBB. Elimina o problema de "matching" (porque IFC já dá as instâncias) e desloca o problema pra "alinhar a nuvem com o IFC".
5. **Implementação `bbox_features.py` standalone** — validou conceito em sintético de outro projeto.
6. **Bug Y-up vs Z-up no Sonata** — descoberto que o Sonata classificava 99% como "wall" quando recebia coords Y-up. Inverter ordem (alinhar antes de classificar) elevou acerto de 75% → 89%.
7. **Plug no front v2** — adapter, endpoint, schema, propagação regressiva. Visualizador 3D mostrando OBBs coloridas por verdict.

Cada um desses passos foi 1-3 dias de investigação + tentativa + reversão + ajuste. Não houve "uma linha mágica que arrumou tudo".

---

## Por que demorou tanto

### 1. Problemas que NÃO existiam no sintético

O scan sintético é gerado a partir do IFC. Cada ponto **herda** geometria perfeita do modelo. Logo:

- Nuvem **alinhada por construção** com o IFC (mesma origem, mesmo sistema de coords)
- **Sem oclusão** — todo elemento tem pontos visíveis em todas as faces
- **Sem ruído** — superfícies são planos matemáticos perfeitos
- **Densidade uniforme** — distribuída homogeneamente
- **Sem elementos "a mais"** — só existe no scan o que o IFC desenhou

No real, **todos esses pressupostos falham**. Cada falha exigiu mudança no pipeline.

### 2. Investigação em loop

Um problema escondia o outro. Ex: a gente tentava melhorar o classificador (achando que era ele errando), e na verdade era o alinhamento entregando uma OBB no lugar errado, fazendo o classificador receber pontos da parede ao lado. Cada vez que tentávamos "consertar pelo fim do pipeline", o sintoma reaparecia em outro lugar.

Só depois de inverter a ordem (`alinhar → Sonata → OBB`) e separar as etapas em scripts standalone (`bbox_features.py` rodando offline com viz dump no CloudCompare) é que conseguimos isolar cada problema e atacar individualmente.

### 3. Falsos positivos infraestrutura

- Pointcept oficial: instalei tudo (CUDA toolkit, PyTorch, extensões custom) e o repo declara que **inference de instance segmentation não está pronta** — tempo perdido.
- Mask3D oficial: pesquisado, descoberto que **não tem demo.py** e exige MinkowskiEngine com CUDA 11.3 (temos 12.4) — caminho abandonado.

Esses dois desvios custaram horas e não geraram nada de útil pro pipeline final.

---

## Os problemas REAIS que a Horizonte expôs

### Problema 1: nuvem e IFC em sistemas de coordenadas diferentes

**Sintoma:** primeiro teste rodou e marcou 87% como AUSENTE. Quase tudo no vermelho.

**Causa real:** o PLY exportado estava em **Y-up** (convenção Blender/Three.js) e o IFC em **Z-up** (convenção arquitetura). Como o filtro de pontos por OBB usa coordenadas absolutas, **bbox da parede procurava altura no eixo Z mas a altura do scan estava no eixo Y** → bbox ficava vazia.

**Solução:** reusar a função `alinhar_nuvem_com_ifc` do `app_obb.py`, que testa permutações de eixos (xyz, xzy, yxz, yzx, zxy, zyx) e sinais (+/-), escolhendo a transformação que maximiza pontos dentro das bboxes. Resolveu Y↔Z em 5s. Esse era um problema **invisível no sintético** porque a simulação já gerava no mesmo sistema do IFC.

### Problema 2: Sonata precisa receber Z-up

**Sintoma:** depois de plugar o alinhamento, ainda 74% COMPLETO. Slabs (lajes) caindo 0%, mesmo sendo o piso óbvio do andar.

**Causa real:** o Sonata foi treinado em ScanNet (apartamentos com Z=altura). Quando recebia nuvem em Y-up (ANTES do alinhamento), interpretava o piso como "parede vertical" e classificava 99% dos pontos do chão como `wall`. A bbox da slab até pegava os pontos certos, mas a classe vinha errada → caía em divergência ou ausente.

**Solução:** trocar a ordem. Antes era `Sonata → alinhar → OBB`; passou a ser `alinhar → Sonata → OBB`. Sonata vê coords no sistema do treino, classifica certo. Acerto subiu de 74% → 89%.

### Problema 3: o telhado do IFC era maior que o real

**Sintoma:** rodava no Faro real e o alinhamento não estabilizava — score baixo, transformação errática.

**Causa real:** o IFC da Horizonte foi modelado com **telhado projetado maior do que o construído na obra**. O scanner via o telhado real, mas o IFC tinha geometria estendendo além. Isso fazia o algoritmo de alinhamento "esticar" a nuvem pra tentar casar com o telhado IFC, distorcendo o resto.

**Workaround atual:** **edição manual da nuvem de pontos** antes de rodar o pipeline — recortando o "excesso" de telhado pra que a geometria casse com a porção que de fato existe. Isso é gambiarra; o ideal seria o alinhamento ser robusto a geometria parcial, mas hoje não é.

### Problema 4: a escada não existia no IFC

**Sintoma:** região com pontos densos no scan que não correspondia a nenhuma OBB IFC. Esses pontos viravam ruído pro alinhamento (puxavam centróides pra lugar errado).

**Causa real:** **escada construída na obra mas omitida no IFC** (provavelmente exportada de outro arquivo, ou modelo trabalhado em estágio antes da escada existir). O pipeline assume IFC como ground truth — qualquer geometria no scan que não tem par no IFC vira "pontos órfãos".

**Workaround atual:** **recorte da escada da nuvem** antes do processamento. Mesma família de gambiarra do problema 3: a pré-edição da PLY tira o que o IFC não conhece.

**Solução pendente real:** classificar "pontos órfãos" como "ADIÇÃO" (construção fora do plano) — funcionalidade que o pipeline antigo tinha mas o novo ainda não implementou.

### Problema 5: gap nas paredes divisórias

**Sintoma:** paredes internas que dividem cômodos sendo marcadas AUSENTE mesmo quando estavam claramente construídas (visíveis no PLY).

**Causa real:** **scanner Faro fica posicionado em pontos discretos** dentro da obra (geralmente 1 scan por cômodo). Para registrar uma parede divisória entre dois cômodos, precisaria escanear de **dois lados**. Quando o operador só fez um lado, a parede aparece como **uma única face** — o "miolo" da parede e a face do outro lado ficam vazios.

A OBB do IFC pra essa parede tem espessura de ~15-25cm. Quando você corta a nuvem pela OBB, **só pega uma face fininha** (~2-5cm de pontos), o que pode cair abaixo do threshold mínimo de pontos.

**Por que NÃO previmos:** no sintético, a parede tem geometria fechada — pontos em todas as faces e dentro do volume. A oclusão de scan real (só 1 lado visível) **não existe** na simulação.

**Solução pendente:** ou (a) **margem de OBB pra elementos divisores** (inflar 5cm pra capturar a face única + margem de erro), ou (b) **threshold adaptativo** por espessura (paredes finas precisam de menos pontos), ou (c) **simular oclusão no sintético** (gerar scans virtuais com câmera em pontos específicos, herdando o problema). A solução (c) é o caminho certo a longo prazo porque torna o sintético representativo do real.

### Problema 6: o scanner é bottom-up e o teto não bate

**Sintoma:** mesmo com tudo mais funcionando, **lajes superiores (forros, tetos)** continuam tendo acerto pior que pisos.

**Causa real:** o Faro fica no tripé, a ~1.5m do chão, **escaneando de baixo pra cima**. Isso causa dois problemas pro teto:

1. **Ângulo rasante** — os feixes batem no teto quase paralelos, gerando pontos com **densidade muito menor** que no piso (que recebe feixes quase perpendiculares).
2. **Oclusão pelos próprios elementos** — vigas, dutos, luminárias bloqueiam a visada de partes do teto.

O IFC, por outro lado, modela o teto/laje/forro como geometria completa. A OBB do `IfcSlab` superior espera pontos no plano inteiro, mas o scan só tem pontos em uma fração desse plano.

**Por que NÃO previmos:** simulação gera densidade uniforme em todas as faces da geometria. Não tem conceito de "câmera/scanner está aqui, raios saem daqui" — então não tem ângulo rasante nem oclusão.

**Mitigação atual (parcial):** a **propagação regressiva** (implementada hoje) ajuda nesse caso específico — se uma laje (`IfcSlab`) grande está COMPLETO, todos os elementos pequenos contidos nela (forros, plates, members do entreteto) **herdam COMPLETO** sem precisar ter pontos próprios suficientes. Resolve uma parte dos forros que antes ficavam AUSENTE por densidade insuficiente.

**O que NÃO resolve:** se a laje em si tem poucos pontos por causa do ângulo rasante, ela própria pode cair pra AUSENTE/PARCIAL — e aí nada se propaga. Esse caso continua aberto.

---

## O que o último resultado representa

Hoje, **no Faro real da Horizonte** (com a nuvem pré-editada — telhado recortado e escada removida, conforme problemas 3 e 4), rodando o pipeline novo (Sonata + OBB + propagação regressiva):

- **89% COMPLETO** total (158 diretos + 99 herdados via propagação regressiva = 258 de 291 elementos)
- Tipos com 100% de acerto: `IfcColumn`, `IfcRailing`, `IfcWindow`, `IfcCovering`, `IfcDoor`, `IfcWall`
- Tipos com 85-90%: `IfcSlab`, `IfcMember`, `IfcPlate`
- Visualizador 3D do front v2 mostrando OBBs verdes/laranjas/vermelhas + nuvem de fundo + pontos por classe

**Importante:** esse número é **no scan real do Faro** (depois das pré-edições). Significa que o pipeline novo já entrega valor real, não só em sintético. Os 11% restantes (AUSENTE/PARCIAL) refletem em grande parte os problemas 5 e 6 que ainda não foram mitigados: gap em paredes divisórias (oclusão de 1 lado) e elementos no teto com pontos esparsos por causa do scanner bottom-up.

---

## O que ficou aprendido

1. **Sintético não substitui scan real pra validação.** Cada pressuposto que a simulação garante (alinhamento, completude, densidade) é um problema escondido esperando aparecer no real.

2. **Inverter pipeline foi mais valioso que trocar modelo.** Mudar de "DBSCAN + Hungarian" pra "IFC OBB como recorte" entregou mais do que qualquer fine-tune de modelo teria entregado, porque atacou o problema certo (matching errado) com a estrutura certa (IFC já tem instâncias prontas).

3. **Ordem das etapas importa tanto quanto a escolha das etapas.** Sonata antes vs depois do alinhamento mudou o acerto em 15 pontos percentuais com o mesmo modelo.

4. **Pré-edição manual da nuvem é gambiarra aceitável a curto prazo.** Telhado maior, escada ausente — recortar a PLY antes não escala, mas evita travar enquanto o pipeline não tem detecção de divergência de geometria.

5. **Propagação regressiva é heurística defensável.** Elementos pequenos dentro de elementos grandes herdam status do grande. Isso reflete a realidade construtiva (laje sustenta forro, parede contém porta) e mascara limitações de densidade do scan sem mentir descaradamente — o status fica marcado como "herdado" pra rastreabilidade.

---

## O que ainda falta enfrentar

- [ ] Refazer pipeline sem pré-edição manual da nuvem (alinhamento robusto a IFC com geometria divergente — telhado/escada)
- [ ] Detectar ADIÇÕES (pontos órfãos sem par IFC) — escada, elementos não modelados
- [ ] Tratar gap em paredes divisórias — margem adaptativa por espessura
- [ ] Mitigar bottom-up no teto — talvez peso menor pra elementos no Z superior
- [ ] Simular oclusão no sintético — gerar scans virtuais com posicionamento real de scanner
- [ ] Fine-tune Sonata em classes de construção (parede_alvenaria, viga, pilar) com dataset gerado do IFC
