# Dataset de tokens de parede para Cloud-to-BIM

Esta etapa transforma cada parede do gabarito IFC em uma elevação X-Z de escala
fixa. Ela existe para inspeção visual antes de qualquer treinamento.

## Representação

- token métrico: 5 × 5 cm;
- janela: 12,8 m de comprimento por 4,0 m de altura;
- tensor nativo: 256 × 80 tokens;
- PNG para o detector: 1280 × 640, ampliado sem interpolação;
- vermelho: densidade de pontos;
- verde: profundidade/espessura observada;
- azul: suporte simultâneo nas duas faces da parede.

As caixas de porta, janela, pilar e abertura vêm do IFC. Cada arquivo de
metadados conserva parede hospedeira, origem, tangente, normal, cota e fórmula
para converter qualquer pixel novamente em coordenadas 3D.

## Prévia atual

```powershell
.\.runtime\python\python.exe experiments\cloud2bim\build_wall_token_dataset.py `
  --sample-dir "C:\Users\Rafael\Desktop\Beckend\dataset\sintetico\model_Villa_v00" `
  --sample-dir "C:\Users\Rafael\Desktop\Beckend\dataset\sintetico\sobrado_v00" `
  --ifc-root "C:\Users\Rafael\Desktop\Beckend\dataset\ifc" `
  --output-dir "artifacts\cloud2bim_wall_tokens_preview_v1" `
  --include-negative
```

O diretório gerado contém:

- `images/preview`: imagens limpas que entrarão no detector;
- `labels/preview`: caixas YOLO normalizadas;
- `review`: imagens com caixas e régua métrica;
- `metadata`: transformação token/pixel para coordenadas 3D;
- `review_contact_sheet.png`: mosaico dos exemplos positivos;
- `review_gallery.html`: galeria de positivos e negativos;
- `manifest.json` e `summary.json`: inventário e contagens.

## Critério de aprovação antes do treino

1. O vazio de porta começa no piso e termina na verga.
2. A janela mantém parede visível abaixo e acima.
3. A caixa colorida coincide com o vazio do histograma.
4. Paredes negativas não contêm aberturas reais não rotuladas.
5. Variações do mesmo edifício ficam no mesmo split para impedir vazamento.

Depois da aprovação visual, o próximo gerador deve percorrer todas as variantes,
separar por edifício em treino/validação/teste e produzir o `dataset.yaml`.

## Dataset YOLO-World-M 1280 preparado

O gerador completo é `build_yoloworld_wall_dataset.py`. Ele usa diretamente
`labels.json` e `ifc_ref.json`, portanto inclui portas e janelas produzidas,
parciais e ausentes mesmo quando o IFC original não está disponível na máquina.

Saída atual: `artifacts/cloud2bim_yoloworld_m_dataset_v1`.

- 22 IFCs-base e 220 variantes;
- 14 famílias arquitetônicas agrupadas;
- 1.865 imagens de treino;
- 894 imagens de validação;
- 411 imagens de teste;
- classes `door` e `window`;
- resolução 1280 × 640;
- treinamento YOLO-World-M concluído.

Os splits são feitos por família arquitetônica. Modelos equivalentes de FZK
Haus, Convenience Store, Office CV2 e MiniExample nunca atravessam splits.
Objetos cuja parede hospedeira não está completa na variante são descartados,
evitando atribuição acidental à parede vizinha.

## Primeiro treinamento YOLO-World-M

O treino usa `train_yoloworld_m_wall_tokens.py` com checkpoint
`yolov8m-worldv2.pt` (28,4 milhões de parâmetros), largura 1280, batch 4 e
lotes retangulares 1280 × 640. Mosaic, alteração de cor, rotação e deformações
ficam desativados porque os canais e pixels têm significado físico. Apenas o
espelhamento horizontal da parede é permitido.

O treinamento encerrou por parada antecipada na época 15; o melhor checkpoint
é o da época 7. Em validação separada por família arquitetônica:

- geral: mAP50 0,451; mAP50-95 0,371;
- porta: mAP50 0,818; mAP50-95 0,675;
- janela: mAP50 0,0835; mAP50-95 0,0663.

No split de teste mantido fora do treino:

- geral: mAP50 0,638; mAP50-95 0,529;
- porta: mAP50 0,834; mAP50-95 0,724;
- janela: mAP50 0,443; mAP50-95 0,334.

O peso selecionado está em
`artifacts/cloud2bim_yoloworld_m_training/wall_tokens_m_1280_v1/weights/best.pt`.
A distância entre validação e teste mostra que a variedade de famílias ainda é
pequena, principalmente para janelas. O próximo dataset deve acrescentar mais
famílias arquitetônicas e exemplos de janela completa, parcial, fechada e
ocluída antes de outro fine-tuning.
