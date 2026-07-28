# Aula: scikit-learn RandomForestClassifier

## 1. Por que Random Forest?

Random Forest é um **ensemble de árvores de decisão**. A ideia central: uma árvore sozinha erra muito e sofre com overfitting, mas **várias árvores votando** corrigem erros individuais.

Analogia: pedir diagnóstico pra **100 médicos diferentes** e ir com o que a maioria disse > confiar em 1 médico só.

## 2. Como uma árvore de decisão funciona

Imagina classificar paredes em `COMPLETO`, `PARCIAL`, `AUSENTE`:

```
            completeness_r >= 0.8?
              /            \
           SIM              NÃO
           /                 \
      height_fill >= 0.9?    is_empty == 1?
         /    \                /      \
       SIM    NÃO            SIM       NÃO
      COMP.  PARC.         AUSENTE    PARC.
```

Cada nó faz uma **pergunta binária** sobre uma feature. A árvore aprende automaticamente:
- **Qual feature perguntar em cada nó** (a que melhor separa as classes)
- **Qual threshold** usar (0.8? 0.75?)

O critério pra escolher "melhor separação" é geralmente **Gini impurity** ou **entropia**. Ambos medem "quão misturadas" estão as classes num nó. Quanto mais puro depois da divisão → melhor a pergunta.

## 3. O problema de uma árvore só

- **Overfitting**: árvore vai dividindo até cada folha ter 1 amostra → memoriza o treino, falha no teste
- **Instável**: muda 1 amostra do treino e a árvore inteira pode mudar
- **Viés**: se a feature X é dominante, todas as decisões giram em torno dela

## 4. O truque do Random Forest: bagging + random features

Treina **N árvores** (ex: 100), mas cada uma recebe:

### (a) **Bootstrap sample** do dataset
Se tens 1000 amostras, cada árvore recebe 1000 amostras **sorteadas com reposição** → ~63% são únicas, o resto são duplicatas. Isso significa cada árvore vê um conjunto **ligeiramente diferente**.

### (b) **Subset aleatório de features em cada split**
Se tens 11 features, em cada nó a árvore só pode escolher entre `sqrt(11) ≈ 3` features aleatórias. Isso força diversidade — sem isso todas as árvores usariam a mesma feature dominante.

### (c) **Votação final**
Pra predizer, passa a amostra em **todas** as árvores → conta os votos → classe majoritária vence.

Resultado: cada árvore individual é fraca e overfita diferente, mas **os erros se cancelam** na média.

## 5. Código básico

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# X = (N, 11) features, y = (N,) labels {COMPLETO, PARCIAL, AUSENTE}
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300,      # número de árvores
    max_depth=None,        # deixa crescer até o fim
    min_samples_leaf=2,    # mínimo por folha (evita overfitting)
    max_features='sqrt',   # features aleatórias por split
    class_weight='balanced', # compensa classes desbalanceadas
    random_state=42,
    n_jobs=-1,             # usa todos os cores
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
joblib.dump(clf, 'random_forest.pkl')
```

## 6. Hiperparâmetros importantes

| Param | O que faz | Impacto |
|---|---|---|
| `n_estimators` | número de árvores | mais = melhor, mas satura ~200-500 |
| `max_depth` | profundidade máxima | `None` = cresce até o fim; limitar reduz overfit |
| `min_samples_split` | mín. amostras pra dividir nó | maior = árvore menor, mais bias |
| `min_samples_leaf` | mín. amostras em folha | 2-5 é bom default |
| `max_features` | features por split | `sqrt` (classif.), `log2`, ou fração |
| `class_weight` | peso por classe | `'balanced'` pra classes desbalanceadas |
| `criterion` | `'gini'` ou `'entropy'` | diferença pequena, gini mais rápido |
| `bootstrap` | sample com reposição | deixa `True` (padrão) |
| `oob_score` | score out-of-bag | `True` dá validação grátis |

## 7. Por que funciona tão bem no teu caso (97.9%)

1. **Features discriminativas**: `completeness_r` e `height_fill` já quase resolvem sozinhas
2. **Dataset bem separável**: COMPLETO=1.0, PARCIAL=0.5, AUSENTE=0.0 de completude
3. **Features mistas** (numéricas + categóricas) — RF trata ambas bem, sem precisar de normalização
4. **Dataset pequeno/médio** (1290 samples) — RF brilha nessa faixa, deep learning precisaria mais dados
5. **Não-linear**: RF captura interações (ex: "se tipo=IfcWall E height_fill<0.3 → PARCIAL") sem feature engineering manual

## 8. Vantagens vs MLP (por que ganhou do PyTorch)

| Aspecto | Random Forest | MLP |
|---|---|---|
| Dados tabulares | **Excelente** | Bom |
| Features mistas | Nativo | Precisa encoding cuidadoso |
| Normalização | Não precisa | Obrigatória |
| Overfitting | Controla sozinho | Precisa dropout/regularização |
| Interpretabilidade | **`feature_importances_`** | Caixa preta |
| Tempo de treino | Segundos | Minutos |
| Dataset pequeno | **Ótimo** | Sofre |
| Dados não-lineares | Bom | Melhor |
| Imagens/áudio/texto | Ruim | **Melhor** |

**Regra prática:** dados tabulares < 100k samples → começa com RF ou XGBoost. Imagens/texto/séries temporais → deep learning.

## 9. Interpretabilidade (superpoder do RF)

```python
import pandas as pd
importances = pd.Series(clf.feature_importances_, index=feature_names)
print(importances.sort_values(ascending=False))
```

Vai sair algo tipo:
```
completeness_r      0.42
height_fill         0.28
is_empty            0.11
density_area        0.07
xy_spread_norm     0.05
...
```

Isso te diz **quais features o modelo realmente usa**. Se `tipo` aparece com 0.001, tu pode tirar. Se uma feature que tu esperava ser importante não aparece no topo, tem algo errado nela.

**`permutation_importance`** é mais robusto ainda:
```python
from sklearn.inspection import permutation_importance
r = permutation_importance(clf, X_test, y_test, n_repeats=10)
```
Mede o quanto a acurácia cai se tu embaralhar cada feature. Se cair muito → feature é crítica.

## 10. Pegadinhas comuns

### (a) Leak de dados
Se tu tem 3 variações do mesmo edifício e 2 vão pro treino e 1 pro teste, o RF **decora o edifício** e finge que aprendeu. Por isso tu fez **split por edifício** no `ml/train.py` — absolutamente correto.

### (b) Classes desbalanceadas
Se 80% é COMPLETO e 5% AUSENTE, o RF tende a prever COMPLETO sempre. Resolve com:
- `class_weight='balanced'`
- Oversampling (SMOTE) ou undersampling
- Ajustar threshold de decisão

### (c) Correlação entre features
RF lida bem com features correlacionadas (diferente de regressão linear), mas `feature_importances_` **distribui a importância** entre features correlacionadas. Se `height_fill` e `completeness_r` forem correlacionados, os dois aparecem com importância média em vez de um com alta. Use `permutation_importance` nesse caso.

### (d) Extrapolação
RF **não extrapola**. Se teu treino tem alturas de 0 a 3m e na inferência chega 10m, ele vai "capar" no máximo que viu. Predições fora do range visto são ruins.

### (e) Probabilidades mal calibradas
`predict_proba()` dá estimativas, mas são **médias de votos**, não probabilidades reais. Pra calibrar:
```python
from sklearn.calibration import CalibratedClassifierCV
cal = CalibratedClassifierCV(clf, method='isotonic', cv=5)
cal.fit(X_train, y_train)
```

## 11. Diagnóstico do teu modelo

Com 97.9% acc + F1 macro 0.942, há 3 hipóteses:

1. **Modelo realmente é bom** — features são tão fortes que o problema é fácil ✓ (provável no teu caso)
2. **Dataset sintético é fácil demais** — ruído real de scanner vai derrubar a acc
3. **Ainda tem algum leak** — split por edifício resolve variantes, mas talvez edifícios do mesmo "autor" tenham estilo similar

Pra validar #2: o teste com os 19 edifícios reais vai dizer. Se mantém 90%+ em edifícios nunca vistos e com scanner real, é modelo sólido. Se cair pra 70%, o dataset sintético não representa a realidade.

## 12. Como melhorar a partir daqui

- **XGBoost / LightGBM** — geralmente 1-3% melhor que RF em dados tabulares. Mesma API.
- **Feature engineering** — criar features novas manualmente (razões, interações)
- **Calibração de probabilidades** — pra dar "confiança" nos outputs
- **Stacking** — combinar RF + MLP + XGBoost com meta-modelo
- **Adicionar features contextuais** — pavimento, posição relativa, vizinhança
