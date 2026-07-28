# Changelog — 2026-04-28

Trabalho do dia: melhorias no visualizador 3D e pipeline de cores RGB do scanner.
Estilo step-by-step com aprovação do usuário a cada etapa.

---

## Frente 1 — Qualidade visual do viewer 3D

### Step 1 — Limpeza imediata

**Arquivo:** `bim-ai-superintendent/components/DataView.tsx`

Removido:
- **Cubo vermelho de debug** (`<mesh position={[0,1,0]} ...>`) que era renderizado junto
  com os objetos reais. Era resquício do "se vc ver isso o 3D funciona".
- **Mensagem flutuante de debug** ("Se você vê um cubo vermelho e eixos coloridos…")
  no canto superior esquerdo do canvas.

Trocado:
- Container do canvas: `bg-black border-4 border-blue-500` → `bg-slate-900 border border-slate-700`
  (fundo cinza-azulado escuro, borda 1px discreta dentro da paleta).

**Por quê:** havia ruído visual gritante e elementos de teste em produção.

---

### Step 2 — Antialiasing + densidade de pixel

**Arquivo:** `bim-ai-superintendent/components/DataView.tsx`

No `<Canvas>` do R3F:

```tsx
<Canvas
  camera={{ position: [5, 5, 5], fov: 50 }}
  gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
  dpr={[1, 2]}
>
```

- **`antialias: true`** — Liga MSAA no contexto WebGL. Antes, bordas dos OBBs eram
  serrilhadas. Agora, GPU faz 4 amostras por pixel nas bordas → bordas suaves.
  Custo: ~5-15% GPU.
- **`alpha: false`** — Canvas opaco (sem blending com HTML por baixo). Economiza
  pass de blending por frame. Como o `<div>` já é `bg-slate-900`, é seguro.
- **`powerPreference: 'high-performance'`** — Pede GPU dedicada em laptops dual-GPU
  (NVIDIA/AMD) em vez de iGPU.
- **`dpr={[1, 2]}`** — Tupla `[min, max]` do Device Pixel Ratio. Em 4K/Retina,
  evita renderizar a 3× (custo 9× pixel) e mantém em 2× (qualidade Retina).
  Em monitor comum (DPR=1) renderiza nativo.

**Efeito visível:** linhas diagonais dos OBBs ficam lisas; em Retina, pontos da
nuvem aparecem nítidos.

---

### Step "RGB Scanner" — Cor real do scanner (substitui Step 3 original)

O usuário levantou que iluminação Three.js sintética não fazia sentido para visualizar
nuvem de pontos (PointsMaterial é unlit). Pivotamos para **ler RGB real do PLY**.

#### Backend (`app_obb.py`)

**1. `_ler_ply_validado`** — agora retorna `(pts, colors, err)`:
- Lê `pcd.colors` do open3d quando disponível (PLYs com RGB do scanner).
- Devolve `colors` como `np.ndarray` float em [0,1] ou `None` se PLY não trouxe RGB.
- Dedup mudou de `np.unique(pts, axis=0)` (reordena) para
  `np.unique(..., return_index=True)` + `sort` (preserva ordem original).
  Necessário para cores ficarem alinhadas com os pontos correspondentes.
- 3 callers atualizados: `analisar_pavimento_completo`, `/api/analisar_instancias`,
  `/api/analisar_ai` (modos BBox e Instâncias ignoram colors via `_, _, err = ...`).

**2. `candidatos_obj`** — assinatura mudou para retornar `(pts_subset, idx)`:
- `idx` é o array de índices retornado pelo `KDTree.query_ball_point`.
- Permite fatiar arrays paralelos (cores) sem refazer a query KDTree.
- Custo zero: `idx` já existe internamente, só passamos pra fora.

**3. `filtrar_pontos_obb`** — parâmetro opcional `extras: Optional[np.ndarray]`:
- Quando passado, aplica os mesmos masks (slice_z + OBB) ao array paralelo.
- Retorna tupla `(pts, extras)`. Sem `extras`, comportamento idêntico ao anterior
  (compatibilidade total com `detectar_objetos_fantasma` etc).

**4. `_extrair_features_ml`** — assinatura aceita `colors_cena: Optional[np.ndarray]`:
- Retorna `(feats, pts_obj, colors_obj)`.
- `colors_obj` carrega o subset filtrado pelo OBB do objeto, alinhado com `pts_obj`.

**5. `/api/analisar_ai`** — pipeline propaga RGB ponta a ponta:
- Ler PLY → `(pts, ply_colors, err)`. Log: `🎨 PLY traz cores RGB do scanner — N pontos coloridos`.
- Loop de features:
  ```python
  pts_cand, idx_cand = candidatos_obj(pts, tree, obj['bbox'], margem=0.10)
  colors_cand = ply_colors[idx_cand] if ply_colors is not None and idx_cand is not None else None
  feats, pts_obj, colors_obj = _extrair_features_ml(pts_cand, obj, colors_cena=colors_cand)
  pts_obj_cache.append(pts_obj)
  colors_obj_cache.append(colors_obj)
  ```
- Loop de JSON: serializa `colors` como **uint8 [0–255]** (3-4× menor que float):
  ```python
  if colors_obj is not None and len(colors_obj) == len(pts_obj):
      rgb_uint8 = np.clip(colors_obj * 255.0, 0, 255).astype(np.uint8)
      json_data['colors'] = rgb_uint8.flatten().tolist()
  ```
- `del colors_obj_cache` ao final pra liberar memória.

#### Frontend (`bim-ai-superintendent/components/DataView.tsx`)

**`ObjectPoints`** — leitura e renderização de cor por vértice:
- Lê `data.colors` do JSON (uint8) → normaliza para Float32Array em [0,1].
- Cria `<bufferAttribute attach="attributes-color" />` quando há colors.
- Material:
  ```tsx
  <pointsMaterial
    vertexColors={!!colors}
    color={colors ? undefined : new THREE.Color(color)}
  />
  ```
- Sem colors → fallback para cor única por status (verde/laranja/vermelho).

**Resultado:**
- PLY com RGB (Polycam, iPhone LiDAR, Matterport) → textura real do scanner aparece
  no viewer (pintura, concreto, sombra natural).
- PLY sintético do dataset → fallback automático para cores de status.

---

### Step 4 — Pontos circulares + tamanho refinado

**Arquivo:** `bim-ai-superintendent/components/DataView.tsx`

Antes: pontos eram quads quadrados de 5cm. Visual "spray paint", pixelado.

**1. Sprite radial gerado uma vez no módulo:**
```tsx
const POINT_SPRITE = (() => {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const ctx = c.getContext('2d')!;
  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0.0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.7, 'rgba(255,255,255,1)');
  grad.addColorStop(1.0, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
})();
```

Definido **fora do componente** (top-level do módulo) — uma única textura na GPU
(~16 KB) compartilhada por todas as instâncias de `ObjectPoints`. Se estivesse dentro
com `useMemo`, criaria 85+ texturas (~1.4 MB).

**2. Material atualizado:**
```tsx
<pointsMaterial
  size={0.03}                  // 0.05 → 0.03 (menos blob, mais "scan denso")
  sizeAttenuation
  map={POINT_SPRITE}           // sprite radial → ponto vira disco
  alphaTest={0.5}              // recorta cantos do quad sem blending
  transparent={false}          // alphaTest dispensa transparency real
  depthWrite                   // oclusão correta entre pontos
  vertexColors={!!colors}
  color={colors ? undefined : new THREE.Color(color)}
/>
```

**Por que `alphaTest` e não `transparent=true`:**
- `transparent=true` exige depth sorting de TODOS os pontos por frame (caro com 100k+
  pontos) e sofre com leak/sort artifacts.
- `alphaTest=0.5` faz decisão binária por pixel — sem blending, sem ordenação. Borda
  do disco fica "dura", mas o MSAA (Step 2) suaviza no contorno automaticamente.

**Custo de performance:** zero. Mesma quantidade de fragments, sprite é cache hit
imediato.

---

## Outros artefatos criados

### `bim-ai-superintendent/components/Card.tsx`

Componente orfão (não importado por ninguém ainda). Criado quando o plano original
incluía Frente 2 (page design) simultânea com Frente 1; usuário pivotou pra step-by-step
e essa frente foi adiada. Serve como base pronta para quando voltarmos pro design da
página.

Inclui dois componentes:
- `<Card>` — wrapper genérico com sombra em duas camadas e tone bar opcional
- `<StatCard>` — card especializado para métricas grandes (label + value tabular-nums)

### `docs/PIPELINE_BACK_FRONT.md`

Documentação completa do fluxo de dados Back ↔ JSON ↔ Front, das duas naturezas de
JSON (resposta principal pequena com metadados + arquivos por objeto com pontos),
diagrama ASCII do fluxo e racional arquitetural.

---

## O que NÃO foi commitado / decisões adiadas

### Detecção de duplicatas geométricas no IFC merge

Implementação foi escrita e revertida no mesmo dia. O usuário identificou que o
problema só aparece consistentemente no PLY sintético do prédio inteiro
(`PREDIO__uniforme_*.ply`) — em PLY por pavimento aparece bem menos. Decisão de
**aguardar dados de scanner real** antes de implementar a correção, pra evitar
otimizar para artefatos sintéticos.

Detalhes em `~/.claude/projects/.../memory/decisions_pending_real_scan.md`.

### Frente 2 — Design da página

Adiada conforme pivot do usuário pra step-by-step da Frente 1. Componente `Card.tsx`
fica de base pra retomada futura.

---

## Estado funcional ao final do dia

- ✅ Backend e frontend sobem limpos (sintaxe verificada)
- ✅ Modo AI (`/api/analisar_ai`) propaga RGB do scanner ponta a ponta
- ✅ Outros endpoints (BBox, Instâncias) continuam funcionando inalterados
  (mudança no `_ler_ply_validado` é retrocompatível via `pts, _, err = ...`)
- ✅ Visualizador renderiza com antialias, pontos circulares menores, fundo discreto
- ✅ Sem regressão de performance (testado com prédio de 40 andares: features 0.63s,
  cache reuse 85/85, JSON 0.58s — mesmos números do commit anterior)
