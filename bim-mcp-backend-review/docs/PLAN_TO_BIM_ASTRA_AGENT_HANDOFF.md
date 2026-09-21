# Plan to BIM — agente Astra e processamento assíncrono

Documento de continuidade para outro chat. Atualizado em 16/09/2026.

**Status: publicado no Cloud Run em 16/09/2026, revisão `bim-api-v2-00034-sah`.** A conversão gratuita foi preservada. A modalidade pública se chama **Conversão Pro**. O pré-processamento Pro apenas prepara o arquivo e calcula o orçamento; ele não executa detector geométrico. O Checkout Pro do Mercado Pago foi ligado ao fluxo: somente uma aprovação verificada servidor a servidor agenda a análise. O editor só abre quando a geometria devolvida foi validada. O IFC continua sendo responsabilidade do gerador determinístico do site e só é criado depois da revisão do cliente.

## Atualização de 13/09/2026 — Conversão Pro sem detector heurístico

Foi implementado um protótipo local do fluxo de produto discutido:

1. O usuário escolhe **Conversão Pro** ou mantém a **Conversão gratuita**.
2. `POST /api/astra-flow/preflight` salva a entrada, renderiza a primeira página do PDF e calcula prazo/orçamento a partir dos metadados do documento. Não detecta paredes, portas, janelas ou laje e não chama a API paga.
3. A interface pública mostra somente o preço final, o prazo e a entrega. Custos internos, modelo, provedor, detectores e versão do pipeline não são enviados ao navegador.
4. `POST /api/astra-flow/jobs/{job}/checkout` cria uma preferência única e devolve somente a URL hospedada do checkout.
5. `POST /api/astra-flow/jobs/{job}/payment/sync` consulta o pagamento diretamente no Mercado Pago para suportar o teste local, onde callbacks não podem usar `localhost`. Em produção, `POST /api/payments/mercadopago/webhook` valida a assinatura HMAC antes da mesma reconciliação.
6. Valor, moeda e referência externa precisam coincidir com o orçamento salvo. O retorno do navegador nunca libera o job.
7. `GET /api/astra-flow/jobs/{job}?token=...` permite polling e retomada no mesmo navegador. O token privado fica no `localStorage` apenas neste protótipo.
8. Depois da aprovação, o Astra recebe uma única imagem da planta, sem JSON de candidatos e sem overlay do detector. Ele devolve paredes, aberturas e contorno de laje completos sob schema estruturado.
9. O backend apenas valida e normaliza essa geometria para o contrato do editor. Nenhum IFC é produzido pela tarefa Pro.
10. Depois da revisão e aprovação do cliente, o gerador BIM já existente pode produzir IFC/DXF a partir do modelo editado.

Arquivos principais:

- `plantatobim/astra_local_flow.py`: estado persistido por tarefa, orçamento, aprovação, executor e resultado.
- `plantatobim/mercadopago_checkout.py`: preferências, consulta de pagamentos e validação HMAC de webhooks.
- `plantatobim/astra_direct_stage.py`: chamada de visão direta e adaptação validada da geometria criada pelo Astra.
- `plan_to_bim_free_app.py`: endpoints HTTP do fluxo.
- `plan-to-bim-web/Plan2bim/components/converter-upload.tsx`: seleção de modalidade, orçamento, checkout, progresso e retomada.
- `tests/test_astra_local_flow.py`: contrato ponta a ponta com estágio Astra simulado.

Os resultados anteriores da planta `INN-EST-TR-EX-006-TIP-R02-Model.pdf` que contam 28 paredes e 50 aberturas no preflight pertencem ao protótipo **híbrido legado**. Eles não descrevem o fluxo Pro atual e não devem ser usados como evidência de que o detector participa dele.

Em 16/09/2026, a Conversão Pro passou a ser cobrada **por página**. Como o pipeline atual processa somente a primeira página, todo novo pedido usa o preço mínimo configurado de **R$ 59,90**. A área construída deixou de participar do orçamento e a pré-inspeção não chama mais um modelo de visão para localizar quadro de áreas. Preferências de pagamento já emitidas preservam o valor aceito; jobs sem checkout podem ser migrados para `single-page-minimum-v1`. O pedido assistido por equipe permanece separado e continua com sua própria cobrança por m².

Ainda em 13/09/2026, o job local `d57729eb2b` concluiu uma revisão semântica real. Esse job também é **híbrido legado**: recebeu candidatos do detector, portanto não pode ser apresentado como teste do fluxo direto. Seus artefatos continuam preservados apenas como histórico. Nenhum IFC foi gerado.

O contrato `astra-editor-v2`, que recompilou aquele job usando decisões `keep/review/reject`, é igualmente legado. O contrato atual é `astra-direct-v1`: não existe candidato heurístico para aceitar ou rejeitar; existe somente geometria proposta pelo Astra e validada pelo backend.

No ambiente local, o fallback continua usando `ThreadPoolExecutor`. No Cloud Run, os pedidos pagos usam estado transacional no Supabase Postgres, arquivos privados no bucket `plan2bim-jobs` e execução pelo Cloud Tasks. Assim, uma troca de instância não apaga planta, pagamento ou resultado. O endpoint do worker valida um token OIDC emitido para a conta de serviço configurada; os caminhos privados do Storage não são enviados ao navegador.

As assinaturas do webhook do Mercado Pago são separadas por ambiente:
`MERCADOPAGO_WEBHOOK_SECRET_TEST` e
`MERCADOPAGO_WEBHOOK_SECRET_PRODUCTION`. O backend seleciona a primeira quando
`MERCADOPAGO_USE_SANDBOX=true` e a segunda em produção. A variável antiga
`MERCADOPAGO_WEBHOOK_SECRET` existe apenas como fallback local.

No protótipo local, um reinício marca tarefas `queued`/`running` como `backend_restarted` e exige confirmação explícita antes do reenvio. No Cloud Run esse comportamento é desativado: o estado permanece no Supabase e o Cloud Tasks repete apenas requisições interrompidas. Falhas tratadas pelo estágio são registradas como `failed` e continuam exigindo uma tentativa explícita, evitando consumo duplicado da API.

## Histórico experimental de 12/09/2026 — piloto estruturado

Foi validada uma simplificação importante da arquitetura: **o Astra não precisa criar ou programar o IFC**. No experimento histórico desta seção ele recebeu imagem e dados estruturados; no fluxo Pro atual recebe somente a imagem original e devolve uma especificação JSON sob schema estrito. O gerador determinístico do Plan to BIM cria o IFC4 somente depois da revisão.

Na planta `C:\Users\Rafael\Desktop\Beckend\dataset\plantas kets\INN-EST-TR-EX-006-TIP-R02-Model.pdf`, uma única chamada produziu 28 pilares, 24 vigas, 15 lajes e 19 aberturas. O IFC foi reaberto e validado sem problemas de schema, falhas geométricas ou divergências de contagem; todas as 67 malhas estruturais ficaram fechadas. Contra o gabarito local, os 86 elementos foram pareados, com IoU mediana em planta de 0,9814 e erro mediano de centroide de 2,5 mm. A hospedagem das aberturas acertou 17 de 19 casos. Os maiores desvios geométricos ficaram em `V302`, `V318` e `L307`.

A chamada levou 414,782 s, consumiu 44.214 tokens e teve custo estimado de US$ 1,677705 pelas tarifas de 12/09/2026. Não foi feita uma segunda chamada de correção. O relatório completo está em `docs/ASTRA_STRUCTURED_BIM_PILOT_2026-09-12.md` e os artefatos em `outputs/astra-structured-bim-inn-20260912/`.

O teste reforça o fluxo de produto discutido: pré-análise barata, estimativa apresentada ao cliente, pagamento/confirmação e somente então processamento assíncrono. O piloto ainda não implementa preço, checkout, fila durável nem interface do site.

### Histórico legado — Astra somente como classificador semântico

Este experimento não é mais o fluxo do site. O estágio `plantatobim/astra_semantic_stage.py` recebia geometria candidata produzida pelos detectores e devolvia classificações por ID. Ele permanece apenas para reproduzir os relatórios antigos; `astra_local_flow.py` não o usa para detectar ou classificar candidatos na Conversão Pro atual.

O estágio foi executado em seis plantas arquitetônicas da mesma pasta, uma chamada Astra por folha e nenhum IFC. As seis concluíram: 379 candidatos de parede e 513 de abertura foram classificados. O Astra rejeitou 202 falsos positivos de abertura, colocou outros 100 em revisão e identificou 15 pilares entre candidatos de parede. Todas as folhas ainda exigiram revisão humana, principalmente por candidatos geométricos compostos/duplicados e contorno de laje. O custo estimado das seis chamadas válidas foi US$ 6,027031. O relatório detalhado está em `docs/ASTRA_SEMANTIC_BATCH_PILOT_2026-09-12.md`.

## 1. Objetivo e intenção do usuário

Integrar o Astra por API ao Plan to BIM como **único autor da geometria inicial da Conversão Pro**. O Astra interpreta a imagem original e informa o que é cada elemento, com suas coordenadas, dimensões, confiança e relações. O backend valida e transforma esse JSON para o editor; ele não fornece geometria heurística ao Astra.

O Astra não precisa criar IFC, escrever scripts ou controlar o gerador BIM nessa primeira versão. O gerador determinístico já existente no site recebe o modelo após a revisão manual e produz IFC/DXF. A conversão gratuita continua separada e pode continuar usando seu detector atual.

Prioridades:

- Precisão das dimensões, cotas e geometria acima de uma prévia apenas visualmente convincente.
- Uso integral pelo navegador, sem instalar Python ou modeladores na máquina do cliente.
- Front existente na Vercel; execução pesada no Google Cloud Run.
- Não perder a conversão quando uma requisição expirar ou o cliente fechar a página.
- Medir tempo, tokens e custo por conversão, com limites explícitos.
- Preservar o motor gratuito atual como rota separada, sem misturar seus resultados na Conversão Pro.

O produto vinha sendo gratuito, com sugestão voluntária de apoio, e o usuário prefere evitar login. A Conversão Pro cobra por planta. O checkout e a verificação servidor a servidor já existem no protótipo local; a publicação ainda depende de persistência transacional, URLs públicas, segredo do webhook e controle de acesso durável.

O usuário autorizou explicitamente o piloto pago local acima. Esta autorização não se estende a novas chamadas pagas, push, deploy, mudanças em produção ou substituição do motor atual.

## 2. Local correto do projeto

| Finalidade | Caminho local |
|---|---|
| Workspace do backend | `C:\Users\Rafael\Desktop\Beckend\bim-mcp-backend-review` |
| Front correto usado pelo usuário para GitHub/Vercel | `C:\Users\Rafael\Desktop\Beckend\bim-mcp-backend-review\plan-to-bim-web\Plan2bim` |
| Este documento | `C:\Users\Rafael\Desktop\Beckend\bim-mcp-backend-review\docs\PLAN_TO_BIM_ASTRA_AGENT_HANDOFF.md` |
| Python local usado no teste | `C:\Users\Rafael\Desktop\Beckend\bim-mcp-backend-review\.runtime\python\python.exe` |

As referências de código abaixo são relativas ao workspace do backend. Não confundir o front correto com `bim-ai-superintendent` ou outras cópias antigas.

Verificação local: a raiz Git encontrada para o backend é `C:/Users/Rafael/Desktop/Beckend`; o front `Plan2bim` tem repositório próprio. Há muitas alterações preexistentes e arquivos não rastreados. Inspecionar o estado de ambos antes de editar/publicar; não limpar, sobrescrever ou incluir alterações alheias em commits. O estado publicado na nuvem não foi auditado nesta documentação.

## 3. O que realmente aconteceu no teste INN

Fonte atual: `C:\Users\Rafael\Desktop\Beckend\dataset\plantas kets\INN-EST-TR-EX-006-TIP-R02-Model.pdf`.

Foi uma reconstrução assistida, específica de uma prancha estrutural vetorial. O assistente leu a prancha e registrou explicitamente no código várias coordenadas, identificadores, seções e associações. Algumas extrações foram automáticas, mas com regras próprias dessa folha. **Não foi uma execução do YOLO, do pipeline atual do site ou de um agente autônomo de produção.**

### Código e dependências usados

- Criado para o teste: `experiments/plant2bim/build_inn_006_structural.py`.
- Testes específicos: `experiments/plant2bim/tests/test_inn_006_structural.py`.
- Reuso direto do projeto: `Projection` e `shade`, em `plantatobim/render_ifc_isometric.py`, para as prévias.
- `pdfplumber`: extração de objetos vetoriais do PDF.
- Poppler (`pdftoppm`): imagens da prancha e recortes de conferência.
- Shapely: polígonos, recortes, uniões, interseções e ajustes de encontros.
- IfcOpenShell: autoria IFC4, extração de malhas e validação.
- NumPy e Pillow: cálculos e renderização das imagens de conferência.

O script contém tabelas específicas como `CALIBRATION`, `COLUMNS`, `BEAMS`, `SPECIAL_BEAMS`, `RECESSED` e `MAIN_SLABS`. Trocar somente o caminho do PDF não o transforma em conversor genérico. Ele também usa caminhos e fontes do Windows que precisam ser adaptados em um container Linux.

### Resultado disponível

Pasta: `outputs/inn-006-structural-bim-20260912/`.

| Arquivo | Conteúdo |
|---|---|
| `INN_006_pavimento_tipo.ifc` | Um pavimento: 28 pilares, 24 vigas, 15 lajes e 19 aberturas |
| `INN_006_2ao14_pavimentos.ifc` | 13 pavimentos: 364 pilares, 312 vigas, 195 lajes e 247 aberturas |
| `conferencia_sobre_pdf.png` | Sobreposição ao desenho de origem |
| `pavimento_tipo_3d.png` | Estrutura com lajes ocultas somente na imagem |
| `pavimento_tipo_com_lajes.png` | Pavimento completo |
| `estrutura_2ao14_3d.png` | Conjunto dos pavimentos |
| `source_geometry.json` | Geometria, calibração e evidências |
| `validation.json` | Validações dos IFCs reabertos |
| `LEIA-ME.md` | Hipóteses, limites e detalhes da reconstrução |

Foram aprovados oito testes específicos na execução original. O relatório salvo registra zero problemas de esquema, geração de geometria ou divergência de volumes, com 67 e 871 malhas estruturais fechadas. Isso verifica consistência do arquivo; não certifica fidelidade completa ao projeto ou segurança estrutural.

Preservados: pilar P10 em U, lajes de 11/13 cm, rebaixos de 2 cm e vazios. A calibração por seis cotas apresentou resíduo máximo de 2,551 mm nessas referências, não uma garantia global de precisão. Há conflito de níveis na prancha: legenda +729,94/+729,92 contra título/corte +792,94. O último foi adotado provisoriamente e a divergência foi documentada.

Sem armaduras, fundações, modelo analítico de cálculo ou validação no Revit. O modelo é preliminar e exige revisão técnica.

### Consumo daquele trabalho

O registro do Codex para a montagem somou:

- Entrada nova: 100.621 tokens.
- Entrada reaproveitada de cache: 3.820.160 tokens.
- Saída, incluindo raciocínio: 36.009 tokens.
- Total acumulado entre as várias etapas: 3.956.790 tokens.

Não houve chamadas às chaves de API OpenAI/DeepSeek para essa montagem. Houve consumo do Codex para interpretar, desenvolver, testar e corrigir. Os números incluem o histórico e trabalho de desenvolvimento; não são uma medição de preço por planta na API. Cache não significa processamento gratuito, nem a sua participação em tokens equivale à participação no custo. [Referência de cache](https://developers.openai.com/api/docs/guides/prompt-caching).

## 4. Estado local do sistema que precisa ser considerado

| Componente | Evidência local e implicação |
|---|---|
| `plan_to_bim_free_app.py` | O fluxo gratuito ainda executa `pre_wall_image_to_editor_model` na própria requisição. A Conversão Pro persiste o pedido no Supabase e só é enviada ao Cloud Tasks depois da confirmação do pagamento. |
| `plan-to-bim-web/Plan2bim/app/api/processar/route.ts` | Proxy síncrono com `AbortSignal.timeout(290_000)` e `maxDuration = 300`; aceita PDF/PNG/JPG/JPEG com limite de 25 MB no código. |
| Mesma rota do front | `buildPublicResult` trabalha com o contrato atual de paredes/aberturas; `summary.walls === 0` gera erro. Um IFC estrutural válido sem paredes não cabe automaticamente nesse fluxo. |
| `plan-to-bim-web/Plan2bim/app/api/backend/[...path]/route.ts` | Outro proxy com timeout de 290 s e duração declarada de 300 s. |
| `Dockerfile` | Inicia `plan_to_bim_free_app:app` via Gunicorn com `--timeout 300`; mantém configuração do motor híbrido atual. Não é um executor de agente dedicado. |
| `requirements-plan2bim.txt` | Declara IfcOpenShell, NumPy e Pillow, entre outros; não declara explicitamente `pdfplumber`, Shapely ou SDK OpenAI. O Dockerfile não instala Poppler. Conferir dependências transitivas e fixar as necessárias no ambiente do agente. |
| `supabase_training_archive.py` | Implementa arquivamento opcional com consentimento, bucket privado padrão `plan2bim-training` e métodos `archive_initial`/`archive_final`. Não presumir que já exista uma tabela de jobs ou mecanismo de retomada. |
| `experiments/plant2bim/benchmark_astra_vs_deepseek.py` | Experimento anterior de comparação via Responses API; não é o agente que construiu o INN. Pode fornecer referências de tratamento de respostas/uso, mas não deve ser ligado ao site sem revisão. |

Os timeouts acima são os valores do código local, não uma confirmação da configuração efetiva da Vercel ou da revisão publicada no Cloud Run. Também não cobrem outros limites, como tamanho de upload imposto pela plataforma.

## 5. Arquitetura de produção proposta — ainda não implementada na nuvem

O usuário acessa tudo pelo navegador. A inferência roda no backend e o navegador não recebe chave, nome do provedor, custo interno ou detalhes do pipeline.

1. **Front/Vercel:** envia o arquivo para o preflight e recebe rapidamente preço final, prazo, `job_id` e um mecanismo privado de acompanhamento.
2. **Preflight:** valida o arquivo, renderiza somente a primeira página e registra metadados. Não cria geometria e não chama o Astra.
3. **Pagamento:** checkout cria uma intenção; somente o webhook assinado e idempotente libera o job pago.
4. **API/Cloud Run Service:** registra o job, agenda execução e oferece consultas curtas de status/resultado.
5. **Executor/Cloud Run Job:** envia a imagem original ao Astra e valida o JSON estruturado devolvido, independente da requisição do navegador.
6. **Persistência:** banco para estado/controle e armazenamento privado para entrada, imagem renderizada, JSON completo e auditoria.
7. **Editor e exportação:** o modelo validado abre no editor; após aprovação humana, o gerador BIM existente cria IFC/DXF.

Cloud Run Jobs executa tarefas de container sem servir requisições HTTP e permite configurar timeout e tentativas. Isso serve como base para uma conversão longa; ainda há limites de execução e possibilidade de falhas. [Documentação do Google](https://docs.cloud.google.com/run/docs/create-jobs).

Separar o prazo do job do timeout de cada chamada HTTP. Apenas aumentar o timeout atual ou iniciar uma thread depois da resposta não entrega durabilidade. Persistir o estado fora do disco temporário do executor e permitir retomada. A configuração de CPU e o ciclo de vida das instâncias também importam para trabalho fora de requisições. [Configuração do Cloud Run](https://docs.cloud.google.com/run/docs/configuring/billing-settings).

### Escolha técnica inicial para a integração Astra

Usar `gpt-6-astra` pela Responses API com ferramentas, conforme a documentação oficial consultada. Confirmar acesso da conta e compatibilidade do SDK antes do primeiro teste. A aplicação continua responsável por executar suas ferramentas e devolver os resultados; a chave não concede acesso automático ao filesystem ou Python. [Guia oficial do Astra](https://developers.openai.com/api/docs/guides/latest-model).

Avaliar `background: true` para respostas longas, persistindo o identificador da resposta e consultando seu estado. Isso cobre a execução da resposta na OpenAI, não a validação, revisão e exportação no nosso backend. Capturar resultados conforme a política de retenção escolhida; não usar o armazenamento da API como único checkpoint. [Execução em segundo plano](https://developers.openai.com/api/docs/guides/background).

Não escolher esforço de raciocínio, orçamento ou número de chamadas implicitamente com base em conversas antigas. Definir esses parâmetros no piloto e registrar a configuração usada.

## 6. Componentes mínimos da Conversão Pro

| Componente | Responsabilidade |
|---|---|
| Preparador do documento | Validar o tipo/tamanho e renderizar a primeira página sem detectar elementos |
| Cliente Responses API | Enviar uma imagem e exigir saída estruturada pelo schema do Plan to BIM |
| Validador geométrico | Conferir IDs, limites, espessuras, dimensões e hospedagem das aberturas |
| Adaptador do editor | Converter o JSON validado para paredes, aberturas, laje e referência visual |
| Armazenamento do job | Preservar entrada, estado privado, resposta completa e resultado publicado |
| Editor manual | Exibir somente o resultado pronto e destacar incertezas para confirmação |
| Gerador BIM existente | Criar IFC/DXF após a aprovação do modelo editado |

O executor inicial não roda Python escrito pelo modelo e não entrega ferramentas livres ao Astra. Essa simplificação reduz risco, tempo e custo e corresponde à decisão atual do produto.

## 7. Segurança da execução e dos arquivos

Uma pasta por job organiza os arquivos, mas **não constitui isolamento de segurança** para código arbitrário. Antes de permitir Python gerado pelo modelo, escolher e testar um sandbox efetivo.

- Separar o controlador que possui credenciais do ambiente que executa código gerado. Não entregar ao script chaves OpenAI/Supabase, credenciais de implantação ou acesso ao metadata server/identidade de nuvem.
- Restringir filesystem, rede, subprocessos, CPU, memória, tempo e volume de saída. Bloquear acesso a outros jobs e evitar instalações livres de pacotes.
- Não considerar análise de texto ou uma allowlist de imports, sozinhas, um sandbox suficiente.
- Tratar arquivos enviados e textos da prancha como dados não confiáveis; instruções contidas no PDF não podem alterar permissões do agente.
- Validar tipo real, quantidade de páginas, dimensões de rasterização e tamanho descomprimido; proteger contra consumo excessivo na leitura de PDFs.
- Manter chaves somente no backend/gerenciador de segredos, fora de respostas, logs, código gerado e bundle do front. Não copiar credenciais antigas do histórico para o documento ou código.
- Arquivos e status privados; downloads com autorização e expiração. Não reutilizar automaticamente rotas públicas de arquivos do motor antigo.
- Sem login: usar capability token imprevisível, com escopo e expiração. `job_id` não é autorização. Evitar vazamento do token por logs, analytics e referrer; definir recuperação e revogação.
- Definir quota, proteção contra abuso e teto de gasto antes de expor um endpoint pago publicamente.

Caso o sandbox completo ainda não esteja pronto, restringir o primeiro protótipo a funções predefinidas ou execução local supervisionada. Registrar essa limitação, sem alegar equivalência com um agente livre para programar.

## 8. Persistência, estados e retomada

Proposta: separar arquivos operacionais necessários à conversão do acervo opcional para uso futuro. O consentimento atual de arquivamento não deve ser transformado silenciosamente em consentimento de treinamento. Não reutilizar o bucket de treinamento como armazenamento universal sem rever política e retenção.

Estados sugeridos: `queued`, `running`, `needs_review`, `completed`, `failed`, `cancelled`. O campo `phase` pode informar leitura, interpretação, autoria, validação e revisão visual. Não inventar porcentagens de progresso.

Registro mínimo por job:

- Identificador, hash do arquivo, versão do pipeline, modelo/parâmetros e timestamps.
- Estado/fase, tentativa, lease/heartbeat do executor e versão do checkpoint.
- Identificadores das respostas da API e das chamadas de ferramentas.
- Manifesto dos artefatos com hashes, formatos, tamanhos e validação.
- Uso de tokens discriminado, custo estimado, orçamento e tempo gasto.
- Avisos de medidas ausentes/conflitantes, relatório final e motivo de interrupção.

Regras de execução:

1. Upload completo e registro persistido antes da confirmação de início.
2. Agendamento durável, com recuperação de registros que ficaram sem executor.
3. Claim/lease transacional para impedir dois workers ativos no mesmo job.
4. Checkpoint após cada resultado relevante; arquivos publicados por manifesto versionado.
5. Em reinício, recuperar a resposta existente antes de repetir uma chamada paga. Persistir a intenção de chamada e reconciliar resultados incertos; não prometer exatamente uma cobrança em qualquer falha distribuída.
6. Duplicações de mensagens/tentativas não podem republicar artefatos conflitantes.
7. Marcar o job Pro como `completed` quando o JSON do editor estiver salvo e validado. A exportação IFC é uma ação posterior e separada, após revisão humana.
8. Limite de tempo/tokens não vira sucesso: preservar o parcial e informar a interrupção.
9. Cancelamento deve parar o agendamento de novas operações e tentar cancelar as que estiverem em andamento.

## 9. Contrato sugerido para o front

Novas rotas separadas do motor atual, com nomes ainda sujeitos à implementação:

| Operação | Contrato sugerido |
|---|---|
| Criar conversão | `POST /api/agent-jobs` → HTTP 202, `job_id` e status privado |
| Consultar | `GET /api/agent-jobs/{id}` → estado, fase e mensagens públicas |
| Resultado | `GET /api/agent-jobs/{id}/result` → manifesto e acesso temporário aos artefatos autorizados |
| Cancelar | `POST /api/agent-jobs/{id}/cancel` |

As consultas exigem autorização vinculada ao job. Em uploads maiores, considerar envio direto ao armazenamento com autorização temporária e finalização posterior, para não depender do limite de corpo do proxy da Vercel.

O resultado público deve usar o contrato do editor e conter apenas os dados necessários à revisão. Não enviar ao navegador modelo/provedor, custo interno, estratégia de prompt, versão privada do pipeline ou resposta bruta. O IFC não faz parte da resposta do job Pro.

O acompanhamento deve sobreviver a reload e reabertura pelo acesso privado. Falha de polling não deve cancelar o processamento. Não mostrar código interno, chaves ou logs sensíveis ao usuário final.

## 10. Precisão e avaliação

- Sempre guardar a fonte da dimensão: página, recorte/texto/cota e associação ao elemento.
- Distinguir medida explícita, medida geométrica calibrada e hipótese. Não inventar cotas ausentes.
- Calibrar escala, orientação e origem; validar em referências independentes das usadas no ajuste.
- Preservar contornos especiais, rebaixos, aberturas e níveis; não preencher regiões vazias por conveniência.
- Gerar relatório de conflitos e pedir revisão quando a informação não sustentar uma escolha.
- Separar validade IFC, geometria sólida e fidelidade ao desenho. Uma dessas verificações não prova as outras.
- Não usar o modelo como cálculo/aprovação estrutural nem inferir armaduras e pavimentos não documentados.

Piloto sugerido: INN como caso de referência, sem fornecer ao agente as tabelas codificadas da solução, seguido de plantas inéditas. Testar PDF vetorial e planta rasterizada separadamente. A solução específica pode servir como referência de comparação, não como resposta disfarçada do agente.

Conjunto adicional indicado pelo usuário: pasta `C:\Users\Rafael\Desktop\Beckend\dataset\plantas kets`, arquivos `030-ARQ-PR-DE-0015-12P-R04.pdf`, `030-ARQ-PR-DE-0019-16P-R04.pdf`, `030-ARQ-PR-DE-0020-17P-R03.pdf`, `030-ARQ-PR-DE-0021-18P-R03.pdf`, `030-ARQ-PR-DE-0022-19P-R03.pdf` e `030-ARQ-PR-DE-0023-20P-R03.pdf`. Confirmar existência e páginas antes de rodar; resultados arquitetônicos não são diretamente equivalentes aos estruturais do INN.

Medir: taxa de conclusão, erro dimensional em referências conferidas, omissões/duplicações, classes corretas, tempo total, tentativas, tokens e custo. Comparar com o motor atual usando os mesmos arquivos e critérios. Não prometer preço ou precisão com base apenas no exemplo assistido.

## 11. Custo e limites

O custo por conversão será a soma de inferência, executor, armazenamento e transferências aplicáveis. A assinatura do Codex usada nesta conversa não é o orçamento da API do site.

Para reduzir uso: uma imagem por folha, instruções objetivas, schema compacto e limites de saída. Não pedir IFC como texto ao modelo; o gerador determinístico do site fará essa etapa depois da revisão.

Registrar entrada sem cache, entrada em cache, eventual escrita de cache e saída. Não somar tokens de raciocínio novamente se já estiverem incluídos na saída. Versionar a tabela de preços usada no cálculo e conferir as tarifas atuais antes do piloto; não converter os tokens desta conversa diretamente em uma promessa comercial.

Pendentes: orçamento por job, teto global diário, concorrência, tempo máximo, número de revisões e autorização do lote pago. Falhas e retries também podem consumir créditos.

## 12. Sequência de implementação recomendada

1. **Auditoria inicial:** ler este handoff e o LEIA-ME do INN; conferir Git, runtime, front correto e contratos existentes, sem revelar segredos.
2. **Protótipo local isolado:** fluxo direto Astra → JSON do editor, sem detector e sem IFC; preservar a conversão gratuita. Definir orçamento e autorização antes de chamadas pagas.
3. **Avaliação local:** INN e plantas inéditas, com revisão no editor, validação geométrica e medição de custo/tempo.
4. **Durabilidade:** persistência, estados, checkpoints, leases, duplicações, cancelamento e retomada após falha.
5. **Integração local do front:** início rápido, preço/prazo sem custo interno, status recuperável e abertura do editor somente após resultado validado.
6. **Nuvem de teste:** container Linux com dependências, sandbox verificado, API e Cloud Run Job, armazenamento privado e segredos com privilégios mínimos.
7. **Publicação controlada:** somente após autorização específica, com limite de uso e possibilidade de desativar a modalidade Astra sem afetar o motor atual.

Não começar por substituir o endpoint público e aumentar o timeout para ver se funciona.

### Critérios mínimos antes de publicar

- O preflight retorna sem iniciar a chamada paga; fechar/reabrir a página não perde o job confirmado.
- Trabalho mais longo que os 290 s atuais não depende do proxy original.
- Reinício do worker recupera o checkpoint e evita repetição indevida de operações.
- O executor não roda código livre gerado pelo modelo e não expõe credenciais ao navegador.
- Testes de token inválido, upload abusivo, duplicação, cancelamento e orçamento esgotado passam.
- Resultado incompleto ou inválido não aparece como sucesso.
- O editor recebe a mesma geometria validada que será usada posteriormente pelo gerador IFC.
- Motor gratuito atual mantém testes e comportamento esperado.
- Custo e qualidade foram medidos; política da modalidade paga/financiada foi definida.

## 13. Texto para iniciar o próximo chat

> Vamos continuar a Conversão Pro do Plan to BIM. Leia primeiro `C:\Users\Rafael\Desktop\Beckend\bim-mcp-backend-review\docs\PLAN_TO_BIM_ASTRA_AGENT_HANDOFF.md`. O Astra deve receber a imagem original sem geometria do detector e devolver todo o modelo JSON do editor. O backend apenas valida e adapta; o gerador BIM existente cria IFC somente depois da revisão do cliente. Preserve a conversão gratuita como fluxo separado. No front, use “Conversão Pro” e mostre somente preço final, prazo e entrega — nunca custo interno, modelo, provedor ou detalhes do pipeline. Preserve alterações existentes; não faça chamada paga, push ou deploy sem autorização específica.

Este documento basta para retomar o tema sem carregar todo o histórico da conversa. Os caminhos e as fontes permitem conferir as evidências quando necessário.
