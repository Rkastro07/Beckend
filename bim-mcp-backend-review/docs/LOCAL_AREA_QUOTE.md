# Pré-análise local e orçamento da Conversão Pro

Frontend: `https://plan2bim.vercel.app`, aplicação `plan-to-bim-web/Plan2bim`.
Backend: Cloud Run `bim-api-v2`, aplicação `plan_to_bim_free_app:app`.

## Fluxo implementado

1. O upload renderiza somente a primeira página do PDF.
2. A pré-análise local usa escala impressa, portas e espessura de paredes para
   estimar a escala física e a área do pavimento. Ela não chama DeepSeek nem
   Astra e não consome API paga.
3. O backend calcula uma referência de área × R$ 0,50/m² e, separadamente,
   congela o valor fixo de lançamento aplicado ao checkout.
4. A interface mostra página incluída, área estimada, referência por m², valor
   aplicado, prazo e escopo antes do pagamento.
5. O Checkout Pro do Mercado Pago só libera o processamento depois de confirmar
   valor, moeda, referência e status aprovado diretamente no provedor.
6. Depois da aprovação, o Astra recebe a primeira página e a largura física
   estimada. Ele não recebe paredes, portas ou outros candidatos do detector;
   a geometria completa continua sendo criada pelo Astra para revisão no editor.

O valor fica congelado no job. Refresh, retomada e reinício não recalculam um
checkout já criado. Preferências já emitidas preservam o valor aceito. Se a
medição local falhar, o checkout continua disponível pelo valor fixo e a
interface informa que a área não pôde ser estimada.

## Preço atual

- `PLAN_BIM_MIN_PRICE_BRL=59.90`
- `PLAN_BIM_PRICE_PER_M2=0.50`
- unidade comercial: uma página
- páginas processadas por pedido nesta versão: uma

O valor técnico exibido é a área estimada multiplicada por R$ 0,50/m². O valor
efetivamente cobrado nesta fase é R$ 59,90 por pedido de uma página. Os dois
valores são rotulados separadamente; a referência não é apresentada como preço
anterior nem como desconto.

O pedido assistido por equipe continua sendo um serviço separado e não faz parte
desta regra da Conversão Pro automática.

## Limites explícitos

- Somente a primeira página é processada e cobrada.
- As demais páginas do PDF não entram silenciosamente no escopo.
- Ainda não existe seleção de várias páginas nem desconto por repetição.
- Quando a escala automática tiver confiança suficiente, ela substitui a
  largura informada como régua física do editor e do IFC.
- A pré-análise não executa modelagem e não consome API paga.
- O detector local é usado somente para medição. Sua geometria nunca é enviada
  à etapa Astra.

## Verificação

Backend:

`.runtime/python/python.exe -m pytest tests/test_local_area_estimator.py tests/test_area_preinspection.py tests/test_astra_local_flow.py tests/test_mercadopago_checkout.py -q`

Frontend: `npm run build:vercel` dentro de `plan-to-bim-web/Plan2bim`.
