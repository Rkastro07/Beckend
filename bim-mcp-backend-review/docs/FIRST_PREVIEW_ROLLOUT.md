# Primeira prévia 3D antes da compra

Fluxo publicado em 23/09/2026: backend Cloud Run `bim-api-v2-00056-nol` com 100% do tráfego e frontend Vercel no commit `3aa9ee0`. As duas migrações abaixo foram executadas no SQL Editor do projeto Supabase Plan to BIM e verificadas. O smoke test público confirmou API online, pagamento configurado, login obrigatório e rota gratuita antiga bloqueada. Ainda falta um teste real de primeira prévia com uma conta sem pedidos anteriores; esse teste consome a API de IA.

1. Login Google é obrigatório. A primeira planta de uma conta sem pedidos anteriores inicia uma análise visual de **uma página** sem pagamento.
2. Ao concluir, a API devolve somente sólidos 3D de visualização (paredes com vãos), nunca o JSON editável. A prévia pode ser girada e aproximada, mas não abre o editor nem oferece IFC/DXF.
3. R$ 59,90 desbloqueia **esse mesmo resultado**, sem uma segunda chamada à IA. Pagamento aprovado pelo Mercado Pago é validado no backend antes de entregar o JSON ou permitir as exportações.
4. Uma segunda planta mostra o orçamento e exige pagamento **antes** do processamento. A rota antiga de conversão gratuita fica bloqueada.

Proteções: reserva atômica de uma prévia por conta no Supabase; limite padrão de 15 novas prévias por dia no total e duas por IP por dia, além dos limites de upload existentes. Uma prévia que falhar tem uma nova tentativa sem cobrança. Os arquivos expiram após a retenção configurada (padrão: 30 dias). Prévia visual no navegador não impede captura de tela ou reconstituição aproximada da geometria; o bloqueio protege o modelo editável e as exportações, não a imagem vista pelo cliente.

O checkout da primeira prévia expira junto com os arquivos. Depois desse prazo (ou se o resultado tiver sido removido), a API bloqueia a criação e a reutilização do checkout. Se um pagamento antigo for confirmado após a remoção do resultado, o pedido fica em `review_required` para atendimento, não como entrega concluída. Uma compra aprovada enquanto os arquivos existem reinicia os 30 dias de retenção para o cliente receber o modelo.

## Configuração e verificação

1. Aplicadas `supabase/migrations/202609230001_plan_bim_first_preview.sql` e `supabase/migrations/202609230002_plan_bim_first_preview_rate_limits.sql` no projeto de produção. A tabela, a restrição e a função foram verificadas com consulta somente-leitura.
2. Cloud Run: `PLAN_BIM_FIRST_PREVIEW_ENABLED=true`, `PLAN_BIM_MIN_PRICE_BRL=59.90`, `PLAN_BIM_FIRST_PREVIEWS_PER_DAY=15` e `PLAN_BIM_FIRST_PREVIEWS_PER_IP_DAY=2`. Manter limites de gasto da OpenAI e alertas ativos.
3. Vercel: `NEXT_PUBLIC_FIRST_PREVIEW_ENABLED=true` em Production, aplicado no novo build.
4. Pendente: testar com conta nova a prévia sem pagamento; API sem `result`; exportações 403; pagar; mesmo job com `result`; edição e IFC/DXF; segunda planta aguardando pagamento. Testar recusa, aba fechada e retorno pela conta. Mercado Pago continua em produção, então uma compra de teste pode cobrar de verdade.
