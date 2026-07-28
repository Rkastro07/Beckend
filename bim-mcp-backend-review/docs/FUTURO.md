# Roadmap Futuro — Detecção de Problemas Estruturais

Evolução natural do sistema: de **"foi construído?"** para **"foi construído corretamente?"**.

## Tipos de problema estrutural que dá pra detectar com PLY + IFC

### Categoria 1: **Geométricos** (PLY já basta)
Comparando nuvem real vs projeto:

| Problema | Como detectar | Dificuldade |
|---|---|---|
| **Prumo** (parede inclinada) | PCA no cluster de pontos da parede → checa se o eixo vertical tem desvio > 5mm/m | ⭐ fácil |
| **Nível** (laje torta) | Fit de plano → ângulo do normal vs Z global | ⭐ fácil |
| **Desalinhamento** (parede fora do eixo do projeto) | Distância ortogonal PLY → plano teórico do IFC | ⭐ fácil |
| **Flecha em viga** (deformação elástica) | Fit de polinômio no eixo da viga → compara com reta teórica | ⭐⭐ médio |
| **Recalque** (fundação afundou) | Comparar Z do piso real vs Z do IFC em várias regiões | ⭐⭐ médio |
| **Espessura errada** de parede | Dois planos paralelos → distância real vs projetada | ⭐⭐ médio |
| **Abertura fora de posição** | Detectar hole na nuvem → posição vs IFC | ⭐⭐ médio |

### Categoria 2: **Superficiais** (precisa PLY com RGB ou imagens)
Rachaduras, manchas, desplacamento — essas **não dá só com geometria**. Precisa:
- Scanner com câmera (iPhone LiDAR + RGB, Polycam, etc.)
- Ou fotos separadas georreferenciadas ao modelo

| Problema | Como detectar | Dificuldade |
|---|---|---|
| **Rachaduras** | CV clássico (Sobel + Hough) ou **ML** (YOLO/segmentação) nas imagens | ⭐⭐⭐ |
| **Manchas/umidade** | Classificação de textura / cor | ⭐⭐⭐ |
| **Desplacamento** (concreto caindo) | Combinação geometria + textura | ⭐⭐⭐⭐ |
| **Exposição de armadura** | Detecção de padrão de ferrugem + linearidade | ⭐⭐⭐⭐ |

### Categoria 3: **Comparação temporal** (difference detection)
Dois scans do mesmo lugar em tempos diferentes:

| Problema | Como detectar |
|---|---|
| **Movimentação estrutural** | Registro ICP entre scans → deslocamento > threshold |
| **Nova fissura aparecendo** | Diff de pontos novos numa região que antes era contínua |
| **Progressão de dano** | Área afetada aumentando entre T1 e T2 |

---

## Roadmap (do mais simples ao mais ambicioso)

### Fase 1 — Checagem de **prumo e nível** (1 semana)
**Mínimo viável com o que tu já tem.**

Pra cada objeto do tipo parede/laje/coluna:
1. Extrai pontos dentro do OBB (já tá fazendo)
2. Fit de plano via RANSAC
3. Ângulo normal vs eixo esperado (Z pra pisos, XY pra paredes)
4. Flag se desvio > tolerância NBR (ex: 1/250 = 4mm/m pra paredes)

**Output no front:** nova coluna de status ao lado de COMPLETO/PARCIAL:
- `OK` / `FORA_DE_PRUMO` / `FORA_DE_NIVEL`
- Valor numérico (ex: "inclinação 1.2°")

```python
from sklearn.linear_model import RANSACRegressor
import numpy as np

def checar_prumo(pts_parede, tol_deg=0.3):
    # Fit plano: z = a*x + b*y + c
    X = pts_parede[:, :2]
    z = pts_parede[:, 2]
    ransac = RANSACRegressor(residual_threshold=0.01).fit(X, z)
    normal = np.array([-ransac.estimator_.coef_[0], -ransac.estimator_.coef_[1], 1])
    normal /= np.linalg.norm(normal)
    # Parede ideal: normal horizontal (normal·Z ≈ 0)
    angulo = np.degrees(np.arcsin(abs(normal[2])))
    return {'angulo': angulo, 'ok': angulo < tol_deg}
```

### Fase 2 — **Desalinhamento XY** (2 semanas)
Parede no lugar errado (fora do eixo de projeto).

1. Pega o plano teórico do IFC (normal + ponto)
2. Distância ortogonal de cada ponto do PLY ao plano
3. Se distância média > tolerância (ex: 2cm), flag como `DESALINHADA`
4. Reporta: "parede deslocada 3.2cm para SE"

### Fase 3 — **Flecha em vigas** (3 semanas)
Vigas deformam elasticamente. NBR aceita L/350 (viga de 10m = 28mm).

1. Extrai pontos da viga (eixo longitudinal)
2. Fit polinomial grau 2 (parábola aproxima deformação)
3. Mede flecha máxima vs L/350 da NBR
4. Flag `FLECHA_EXCESSIVA` se exceder

### Fase 4 — **Diff temporal** (1 mês)
Duas visitas → detecta o que mudou.

- Registro ICP entre nuvem T1 e T2 (ambas alinhadas ao IFC)
- Pra cada ponto de T2, busca vizinho em T1 (KDTree)
- Se distância > threshold → mudança (pode ser obra progredindo ou dano)
- Segmenta mudanças por tipo (novo concreto vs deformação)

### Fase 5 — **CV em imagens** (2+ meses, se tiver fotos)
Rachaduras, manchas, armadura exposta. Aqui sai do pipeline geométrico e entra em **deep learning de visão**:
- Treino de YOLO/Mask R-CNN em datasets públicos (SDNET2018, CCIC)
- Projetar fotos no modelo 3D (photogrammetry-BIM registration)
- Georreferenciar detecções → mostrar no IFC

---

## Como isso encaixa no pipeline atual

O sistema hoje tem um **bloco de features por objeto** que vai pro RF. Pra estrutural, expande pra:

```python
features = {
    # Existentes
    'completeness_r': ...,
    'height_fill': ...,
    # Novas estruturais
    'desvio_prumo_deg': ...,      # Fase 1
    'desvio_nivel_deg': ...,      # Fase 1
    'offset_xy_plano': ...,       # Fase 2
    'flecha_rel': ...,            # Fase 3
    'rugosidade': ...,            # desvio padrão da distância ao plano
    'buraco_detectado': ...,      # gap > threshold
}
```

E adiciona classe nova no classificador: **`PROBLEMA_ESTRUTURAL`** além de COMPLETO/PARCIAL/AUSENTE. O RF aprende a combinar esses sinais.

---

## O problema do **dataset**

Pra treinar ML estrutural precisa de **rótulos reais** de problemas. Opções:

1. **Sintético** — deforma IFCs artificialmente (inclina paredes, afunda lajes) e gera PLYs correspondentes
2. **Públicos** — SDNET2018 (rachaduras em concreto), datasets de scans de pontes com danos
3. **Manual** — escanear obras reais e rotular manualmente (caro, mas é o ouro)

Pro começo, **Fase 1 e 2 não precisam de ML** — são **regras geométricas puras** (tolerâncias da NBR). ML entra quando a coisa fica ambígua ou precisa classificar textura.

---

## Recomendação prática

**Começa pela Fase 1** (prumo + nível). Motivos:
1. **Não precisa de dataset novo** — regra geométrica
2. **Norma existente pra validar** (NBR 15575 tem tolerâncias)
3. **Impacto imediato** — cliente entende "parede fora de prumo" muito mais fácil que "objeto parcial"
4. **Reusa a infra** — mesmo loop de objetos, mesma UI, só adiciona coluna

Em 3-5 dias dá pra ter MVP funcionando no `app.py` atual sem quebrar nada.

---

## Referências / Normas

- **NBR 15575** — Desempenho de edificações (tolerâncias)
- **NBR 13753** — Execução de revestimentos
- **NBR 6118** — Projeto de estruturas de concreto (flechas limites)
- **SDNET2018** — dataset de rachaduras em concreto
- **CCIC** (Concrete Crack Images) — classificação
- **PatchBench** — benchmark de defeitos estruturais

---

## Ideias extras (brainstorm)

- **Timeline preditiva:** cruzar estágio atual (RF) com cronograma (CSV) → prever atraso
- **Volumetria de concreto:** contar m³ executado vs planejado (measurement as-built)
- **Detecção de furos/passagens elétricas** fora do projeto (adição não prevista)
- **Relatório NBR automatizado:** gerar PDF com laudos por pavimento (integrar com LLM)
- **AR overlay:** projetar o IFC sobre a câmera do celular em obra
- **Edge computing:** rodar inferência localmente no celular (quantização com TurboQuant-like, ONNX)
