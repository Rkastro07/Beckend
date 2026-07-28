# Features do sistema — inventário de "apzinhos"

Levantamento completo de funcionalidades existentes no código que podem ser empacotadas
como ferramentas (apzinhos) visíveis dentro do produto.

Cada item lista: o que faz, onde está hoje, status atual e esforço estimado pra virar
feature exposta no produto.

---

## 🎯 Tier 1 — Apzinhos de alto valor (cliente final usa direto)

### 1. Conversor USDZ → PLY
- **Onde está:** `usdz_to_ply.py` (CLI standalone)
- **O que faz:** pega arquivo `.usdz` do RoomPlan do iPhone, converte pra `.obj` → faz sampling → vira `.ply`
- **Por que virar app:** muito usuário tira scan com RoomPlan da Apple e não sabe converter pra usar no sistema
- **Esforço:** ⭐ Mínimo (botão "Converter USDZ" no front, sobe arquivo, baixa PLY)

### 2. Conversor OBJ → PLY com sampling
- **Onde está:** dentro de `usdz_to_ply.py` (`obj_para_ply()`)
- **O que faz:** mesh OBJ → nuvem de pontos densa via sampling uniforme
- **Por que virar app:** Kiri Engine exporta direto em OBJ (modo "AI monta modelo 3D"), e nosso sistema lê PLY. Esse conversor é o **elo perdido**.
- **Esforço:** ⭐ Mínimo

### 3. Merger de IFCs multi-disciplina
- **Onde está:** `dataset/merge_ifcs_predio.py`
- **O que faz:** pega N IFCs (ARQ, EST, HID...) com mesma estrutura de storeys e funde em 1 arquivo
- **Por que virar app:** quase todo cliente tem IFC separado por disciplina. Hoje é script com paths hardcoded.
- **Esforço:** ⭐⭐ Médio (generalizar paths + UI de upload múltiplo)

### 4. Merger de PLYs (junta scans parciais)
- **Onde está:** `dataset/juntar_predio.py`
- **O que faz:** lê múltiplos PLYs e funde em 1, opcionalmente alinhando via Z
- **Por que virar app:** **resolve diretamente o problema do iPhone LiDAR** — em vez de scanear pavimento inteiro de uma vez (estoura limite + drift), faz 3-4 scans menores e junta
- **Esforço:** ⭐⭐ Médio (precisa ICP entre eles pra qualidade boa)

### 5. Inspetor de IFC (preview antes de analisar)
- **Onde está:** `_check_storeys.py` (esboço debug) + funções `extrair_pavimentos`
- **O que faz:** sobe IFC, mostra storeys, contagem por tipo (IfcWall, IfcSlab...), tamanho do bbox, naming patterns, presença de MEP
- **Por que virar app:** usuário decide **antes** de rodar análise pesada se o IFC tem dado relevante. Filtra modelos quebrados.
- **Esforço:** ⭐ Mínimo (tudo já existe, é só GUI)

---

## 🔧 Tier 2 — Apzinhos de qualidade/pré-análise (cliente avançado ou interno)

### 6. Validador de PLY
- **Onde está:** `_ler_ply_validado` (interno do backend)
- **O que faz:** abre PLY, mostra nº de pontos, tem RGB?, densidade média, bbox, gaps detectados
- **Esforço:** ⭐ Mínimo (código existe, falta endpoint público + UI)

### 7. Diagnóstico de qualidade de scan
- **Onde está:** **NÃO existe ainda**, mas peças sim — bbox, densidade por região, etc.
- **O que faz:** classifica scan como "bom / aceitável / ruim" antes de rodar análise. Detecta cobertura parcial, drift visível, áreas vazias
- **Esforço:** ⭐⭐⭐ Médio-alto (lógica nova, mas alta utilidade pra parceiros)

### 8. Alinhamento IFC ↔ PLY isolado
- **Onde está:** `alinhar_nuvem_com_ifc` (interno)
- **O que faz:** roda só o alinhamento, devolve PLY transformado
- **Por que virar app:** parceiros de scan podem **só querer alinhar**, sem análise. Vira microserviço utilitário.
- **Esforço:** ⭐⭐ Médio

### 9. Visualizador de OBB do IFC
- **Onde está:** `calcular_obb_corners_threejs` (interno)
- **O que faz:** mostra os bounding boxes orientados de cada elemento do IFC, sem precisar de PLY
- **Por que virar app:** "pré-visualizar como o sistema vê o IFC". Cliente entende o que o sistema vai analisar.
- **Esforço:** ⭐ Mínimo (parte 3D já existe, só não tem o mode "só IFC")

---

## 🧪 Tier 3 — Apzinhos de simulação/treino (cliente que quer dataset próprio)

### 10. Gerador de PLY sintético
- **Onde está:** `dataset/gerar_sintetico.py` + variantes
- **O que faz:** pega IFC → gera N nuvens em estágios diferentes (estrutura só, metade da obra, quase pronto)
- **Por que virar app:** cliente cadastra seu IFC, recebe "obra simulada" em fases — útil pra treinar a IA com casos próprios ou pra fazer demo sem precisar escanear
- **Esforço:** ⭐⭐ Médio (script é robusto, precisa abstrair parâmetros)

### 11. Visualizador de evolução temporal
- **Onde está:** `dataset/visualizar_estagios.py`
- **O que faz:** mostra obra "evoluindo" estágio por estágio, lado a lado
- **Por que virar app:** demo poderosa pra mostrar pro cliente "olha o que o sistema enxergaria em cada fase"
- **Esforço:** ⭐⭐ Médio

### 12. Gerador de dataset estrutural só
- **Onde está:** `dataset/gerar_ply_estrutura.py`
- **O que faz:** extrai e amostra só estrutura (pilares, vigas, lajes) do IFC
- **Esforço:** ⭐ Mínimo (já roda, só falta API)

---

## 🧠 Tier 4 — Apzinhos de IA/treinamento (poder de usuário avançado)

### 13. Re-treinador de Random Forest
- **Onde está:** `ml/train.py`
- **O que faz:** retreina o modelo de classificação com dataset customizado
- **Por que virar app:** **enorme valor de produto** — cliente sobe N projetos próprios, sistema retreina pra contexto dele (residencial alto padrão, industrial, casa popular...)
- **Esforço:** ⭐⭐⭐ Alto (UX de treino assíncrono, gestão de modelos por cliente, GPU/CPU)

### 14. Re-treinador de RandLA-Net (semântico + instâncias)
- **Onde está:** `randlanet/train.py`, `train_instances.py`
- **O que faz:** retreina segmentação semântica/instância
- **Esforço:** ⭐⭐⭐⭐ Alto (treino caro, GPU obrigatório)

### 15. Gerador de dataset rotulado pra IA
- **Onde está:** `randlanet/dataset_generator.py`, `dataset_generator_instances.py`, `batch_generator.py`
- **O que faz:** pega pares IFC+PLY e gera arquivos rotulados prontos pro treino (`.npz`)
- **Esforço:** ⭐⭐ Médio

### 16. Visualizador de dataset rotulado
- **Onde está:** `randlanet/visualizar_dataset.py`
- **O que faz:** abre `.npz` rotulado, mostra cores por classe, validação visual
- **Esforço:** ⭐ Mínimo

---

## 📊 Tier 5 — Já é app, só falta refinar

| Já existe | Onde | Próximo passo |
|---|---|---|
| Análise por pavimento (BBox) | `/api/analisar_pavimento` | Já no front |
| Análise por pavimento (AI) | `/api/analisar_ai` | Já no front |
| Análise por instâncias | `/api/analisar_instancias` | Já no front |
| Análise multi-pavimento | sentinel `__TODOS__` | Já roda, falta UX consolidado |
| Relatório executivo | `/api/generate_report` | Já no front (DeepSeek) |
| Chat com obra | `/api/chat` | Já no front |
| Cronograma comparator | front (lê CSV) | Já existe, dá pra cruzar com análise |
| Visualizador 3D | front (`DataView.tsx`) | Já existe, em refinamento (RGB scanner adicionado) |
| IFC token cache | backend (`_cache_ifc`) | Já funciona, evita reupload |
| Output cache (JSON por objeto) | `/outputs/<file>` | Já funciona, falta TTL |

---

## 🎯 Roadmap sugerido de empacotamento

### Sprint curto (1 semana) — Alto impacto + baixo esforço
1. **Inspetor de IFC** (#5) — antes de analisar, ver o que tem
2. **Conversor USDZ → PLY** (#1) — destrava cliente iPhone
3. **Validador de PLY** (#6) — feedback imediato sobre qualidade

### Sprint médio (3-4 semanas) — Resolve problemas reais
4. **Merger de PLYs** (#4) — resolve problema do iPhone LiDAR
5. **Diagnóstico de qualidade de scan** (#7) — diferencial competitivo
6. **Visualizador de OBB do IFC** (#9) — demo pré-análise

### Sprint longo (futuro) — Produtos avançados
7. **Re-treinador de RF** (#13) — modo "self-service"
8. **Gerador sintético customizado** (#10) — vira ferramenta de demo + dataset privado

### Não fazer agora
- Re-treinador de RandLA-Net (#14) — custo de GPU + complexidade UX, só faz sentido em escala

---

## 📦 Estado atual em números

| Métrica | Valor |
|---|---|
| Total de scripts/módulos no projeto | ~25 |
| Endpoints REST expostos hoje | 8 |
| Funcionalidades "ocultas" prontas pra empacotar | ~16 |
| Funcionalidades em tier 1 (alto valor) | 5 |
| Funcionalidades de mínimo esforço (⭐) | 8 |

---

**Última atualização:** 2026-05-10
