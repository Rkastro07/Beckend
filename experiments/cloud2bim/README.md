# Cloud2BIM adaptado — scan → IFC sem ML

Pipeline geométrico de scan-to-BIM baseado no [Cloud2BIM](https://github.com/VaclavNezerka/Cloud2BIM)
(Zbirovský & Nežerka, CTU Prague, *Automation in Construction* 177, 2025 — MIT), com
adaptações validadas nos nossos dados (scan real RCP RetroPorto + sintéticos Allplan).

## Uso

```bash
python rodar.py <nuvem.ply|.e57|.xyz> [--thr 0.3] [--saida DIR] [--sem-pilar]
```

Um comando faz tudo: conversão → autodiagnóstico de threshold → pipeline
(lajes → pavimentos → paredes → aberturas → cômodos) → detector de pilar →
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
| Lajes por agrupamento de proximidade (não paridade) | Original assumia 1 superfície exposta no térreo e 2 nas demais; nuvem que vê as duas faces da laje térrea quebrava tudo | sempre ativo |
| `assign_points_to_walls` com min/argmin incremental | Original materializava matriz paredes×pontos → OOM de 200GB com muitas paredes | sempre ativo |
| Fix `max2` unbound no `identify_openings` | Parede com <3 bins de altura crashava | sempre ativo |
| `usetex=False` + `plt.show→close` | Rodar headless sem LaTeX | `MPLBACKEND=Agg` |

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
2. **Escada**: não integrada. Protótipo da caracol (fit de hélice) em
   `prototipos/track_espiral.py` — funcionou no RCP (centro/raio/avanço angular),
   limitado pela resolução do scan (~350 pts/m² na escada). Lance reto: a fazer
   (regressão linear no lugar do círculo).
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
prototipos/           escada caracol, inserção manual de pilar/escada, fatiador, diagnósticos
configs_exemplo/      configs usados nos testes (RCP real, sim, Allplan 4 andares)
```

## Resultados de referência

| Nuvem | Pavimentos | Paredes | Janelas | Tempo |
|---|---|---|---|---|
| RCP RetroPorto (e57 real, 32M pts) | 1 | 53 | 12 | ~5 min |
| Allplan Institute completo (1.8M pts) | 6 (incl. split-level) | 53 | 24 | ~12 s |
