# Plan2BIM — atualização operacional e pendências de lançamento

Atualizado em **21 de setembro de 2026**.

Este documento registra o estado publicado do fluxo pago, as proteções adicionadas
e o que ainda precisa ser configurado antes de anunciar o serviço publicamente.

## Estado publicado

| Área | Estado atual |
| --- | --- |
| Backend | Cloud Run `bim-api-v2`, revisão `bim-api-v2-areaquote1`, 100% do tráfego |
| URL da API | `https://bim-api-v2-f5dgdbusxq-uc.a.run.app` |
| Frontend | Publicado no projeto Vercel `plan2bim` |
| Autenticação | Google via Supabase, obrigatória para a Conversão Pro |
| Pagamento | Mercado Pago em produção (`MERCADOPAGO_USE_SANDBOX=false`) |
| Webhook | Segredos separados e URL produtiva configurada para o Cloud Run |
| Fila | Cloud Tasks `plan2bim-astra`, região `us-central1` |
| Pedido Assistido | Oculto/desativado (`PLAN_BIM_ASSISTED_ORDER_ENABLED=false`) |
| Persistência | Supabase Postgres + bucket privado `plan2bim-jobs` |
| Limites de abuso | Contadores globais e atômicos no Supabase |
| Retenção | Cloud Scheduler diário às 03:17 (`America/Sao_Paulo`) |
| Monitoramento | E-mail operacional + 3 políticas válidas no Cloud Monitoring |
| Budget Google Cloud | R$ 40/mês; alertas em 50%, 90%, 100% e 150% |
| IFC automático na Pro | Não. A Pro entrega o modelo para revisão; o IFC é exportado pelo editor após a conferência |

No momento o Cloud Run está com `maxScale=1` e `containerConcurrency=2`, em linha
com as duas threads do Gunicorn. Uma thread pode permanecer no worker de longa
duração enquanto a outra atende login, webhook, polling e health check. A fila
Cloud Tasks continua com `maxConcurrentDispatches=1`, portanto somente uma
análise Astra paga é executada por vez.

## O que foi implementado

### Proteção de uploads e pedidos

O módulo [operational_guard.py](../plantatobim/operational_guard.py) aplica limites
no servidor e devolve HTTP `429` com `Retry-After` quando o limite é atingido:

- 8 uploads por usuário por hora;
- 20 uploads por IP por hora;
- 3 novos pedidos por usuário por dia;
- 10 novos pedidos por IP por dia.

Os IPs não são armazenados em texto puro: o backend trabalha com HMAC. O proxy do
Vercel repassa o endereço de origem para o backend em `x-real-ip`. Os contadores
são consumidos pela RPC transacional `consume_plan_bim_rate_limit`; a tabela
`plan_bim_rate_limits` é inacessível aos papéis `anon` e `authenticated`. O modo
em memória permanece somente como fallback de desenvolvimento sem persistência.

### Uma conversão ativa por conta

O fluxo pago verifica os projetos persistidos no Supabase antes de criar um novo
checkout. Se já houver uma conversão `queued` ou `running`, o novo pedido é
bloqueado. Se dois pagamentos forem aprovados em uma janela muito próxima, o
segundo fica em `waiting_for_slot` e é promovido quando o primeiro termina — sem
nova cobrança.

### Retenção de arquivos

Cada pedido novo recebe uma política de retenção de 30 dias. O estado público do
pedido informa a data limite. Após a limpeza, os arquivos de entrada, imagem
preparada e resultado são removidos do bucket privado; o registro mínimo do
pedido e do pagamento permanece para suporte, prevenção de fraude e obrigações
legais.

O endpoint autenticado de manutenção já existe:

`POST /api/internal/maintenance/retention`

Ele exige a identidade OIDC da conta de serviço do worker. O job
`plan2bim-retention-daily` está habilitado para **03:17**, no horário de São
Paulo. Antes da ativação havia zero pedidos com mais de 30 dias. A execução final
em produção respondeu HTTP 200 em 18/09/2026.

### Alertas e orçamento do Google Cloud

O canal de e-mail `Plan2BIM operacional` envia notificações para o endereço de
operação. Estão habilitadas e válidas as políticas:

- `Plan2BIM - Cloud Run 5xx`;
- `Plan2BIM - falha no Cloud Tasks`;
- `Plan2BIM - fila acumulada` (mais de 3 tarefas durante 15 minutos).

O orçamento mensal já existente de R$ 40 foi preservado e ligado ao mesmo canal.
As faixas são 50%, 90%, 100% e 150%. Esse budget avisa, mas não bloqueia gastos.

### Texto apresentado ao cliente

O orçamento informa que:

- o preço corresponde a uma página nesta versão;
- o modelo fica disponível por 30 dias;
- falha técnica permite nova tentativa sem cobrança adicional;
- prazos, suporte e reembolso estão nos Termos.

As páginas [Termos](../plan-to-bim-web/Plan2bim/app/termos/page.tsx) e
[Privacidade](../plan-to-bim-web/Plan2bim/app/privacidade/page.tsx) agora descrevem
retenção, exclusão antecipada, suporte e tratamento de reembolso.

## Testes já realizados

### Pagamento aprovado (`APRO`)

O teste aprovado anterior percorreu o fluxo até o modelo ficar pronto para o
editor. O pagamento foi confirmado, a análise terminou e o projeto pôde ser
retomado pela conta Google. Esse teste ocorreu numa revisão anterior à
persistência durável e ao webhook assinado, portanto não deve ser usado como
teste da infraestrutura atual.

### Pagamento recusado (`OTHE`)

O teste informado em 17/09/2026 exibiu corretamente a tela do Mercado Pago:

> Seu cartão recusou o pagamento — use outro cartão ou outro meio de pagamento.

Esse resultado confirma que a recusa é apresentada ao cliente e não deve iniciar
o processamento Astra. O fluxo não deve cobrar novamente enquanto o cliente não
abrir uma nova tentativa de checkout.

### Proteções do backend — 18/09/2026

Foram executados 25 testes sem pagamento e sem chamada ao Astra/OpenAI. A bateria
confirmou:

- dois pagamentos simultâneos da mesma conta: somente um entra em `queued` e o
  outro fica em `waiting_for_slot`;
- conclusão do primeiro promove o segundo sem nova cobrança;
- assinatura inválida do webhook responde `401`;
- assinatura válida responde `200` e o replay é idempotente, sem tarefa duplicada;
- limite excedido responde `429` com o cabeçalho `Retry-After`;
- os contadores duráveis recebem somente a impressão HMAC do usuário/IP.

## O que ainda falta

### 1. Fazer a compra real de validação

Os testes de backend passaram e o ambiente produtivo está publicado. Ainda é
necessário concluir uma compra real controlada, fechar a aba depois do pagamento
e confirmar nos logs que a notificação `type=payment` responde `200` e inicia a
fila sem depender de `payment/sync`. Depois da validação, realizar o estorno.

O limite financeiro da OpenAI já foi configurado pelo proprietário em 18/09/2026.

## Ordem recomendada antes de divulgar

1. Fazer uma compra real controlada, fechar a aba e validar o webhook nos logs.
2. Conferir a criação do job, a entrega do resultado e realizar o estorno.
3. Opcionalmente repetir dois pedidos simultâneos pelo navegador; a proteção do
   backend já foi validada com concorrência real em duas threads.

## Verificação técnica de 18/09/2026

- migração `202609180001_plan_bim_rate_limits.sql` aplicada no Supabase;
- teste real de upload pela revisão candidata confirmou a RPC durável;
- 25 testes automatizados de proteção/pagamento do backend passaram;
- endpoint de retenção testado primeiro na candidata e depois na URL pública;
- última execução pública do Scheduler: HTTP 200;
- três políticas do Cloud Monitoring verificadas como habilitadas e válidas;
- Google OAuth confirmado como `Em produção`, com público `Externo`;
- Access Token produtivo validado como conta real e armazenado em segredo separado;
- webhook produtivo salvo com o evento `Pagamentos (legacy)`;
- revisão `bim-api-v2-mp-prod1` publicada sem erros de inicialização;
- health check público: `online`, versão `2.1.0`.

## Verificação técnica de 21/09/2026

- revisão ativa `bim-api-v2-areaquote1` com 100% do tráfego;
- `containerConcurrency=2`, `maxScale=1` e timeout de 1.800 segundos;
- fila `plan2bim-astra` em estado `RUNNING`, com
  `maxConcurrentDispatches=1`;
- 12 health checks simultâneos responderam HTTP 200, sem resposta 429;
- 55 testes locais de autenticação, pagamento, persistência, limites e
  pré-análise passaram;
- build de produção do frontend (`npm run build:vercel`) concluído.

## Variáveis de proteção

Os valores publicados na revisão `bim-api-v2-areaquote1` são:

```text
PLAN_BIM_UPLOADS_PER_USER_HOUR=8
PLAN_BIM_UPLOADS_PER_IP_HOUR=20
PLAN_BIM_ORDERS_PER_USER_DAY=3
PLAN_BIM_ORDERS_PER_IP_DAY=10
PLAN_BIM_FILE_RETENTION_DAYS=30
```

Não registrar chaves, tokens, assinaturas de webhook ou cookies neste documento.
