# Deploy do MVP no Cloud Run — 2026-08-06

## Estado publicado

- Projeto GCP: `my-projectv2-480802`
- Regiao: `us-central1`
- Servico: `bim-api-v2`
- URL publica: <https://bim-api-v2-f5dgdbusxq-uc.a.run.app>
- Revisao ativa: `bim-api-v2-00006-jis`
- Imagem: `us-central1-docker.pkg.dev/my-projectv2-480802/cloud-run-source-deploy/bim-api-v2:launch-20260806-r5-libredwg`
- Digest: `sha256:8e388e0f03627ee9c89919d6a3a6681643699764a4d6a074a0399c608d2a4aaf`
- Build: `53a76308-f2f5-484d-94e5-ef21bad52163`
- Revisao anterior para rollback: `bim-api-v2-00004-vov`

O frontend React compilado e a API Flask sao servidos pelo mesmo container e
pela mesma origem. Isso elimina URLs `localhost` no navegador e dispensa CORS
entre frontend e backend em producao.

## Configuracao inicial do MVP

- 2 CPU
- 4 GiB de memoria
- timeout de 3.600 segundos
- concorrencia 1
- minimo de instancias 0
- maximo de instancias 1
- startup CPU boost ativo
- `SCAN_JOBS_SYNC=true`
- acesso publico (`allUsers` com `roles/run.invoker`)

Concorrencia e escala foram limitadas porque sessoes de scan e status de jobs
ainda vivem na memoria do processo. O modo sincrono impede que o Cloud Run
suspenda a CPU no meio da geracao do IFC. O limite de uma instancia tambem
protege o custo e evita que o polling caia em outra memoria.

## O que foi validado

- build do React: sucesso;
- importacao do app Flask e Random Forest: sucesso;
- 68 testes de autoria, edicao e Planta-to-BIM: sucesso;
- `/api/health`: HTTP 200;
- pagina e assets de producao: HTTP 200;
- bundle sem `localhost:8081`;
- catalogo de receitas BIM: HTTP 200;
- DWG no Linux: GNU LibreDWG 0.13.4 disponivel em `/usr/local/bin/dwgread`;
- DWG publico do corpus LibreDWG: 66 entidades DXF, 55 geometricas,
  54 paredes materializadas no IFC e download HTTP 200;
- Planta-to-BIM: quatro paredes geraram IFC e preview PLY;
- simulador: 25.722 pontos gerados a partir do IFC;
- Scan-to-BIM: upload aceito, tres niveis horizontais e quatro paredes pelo
  Detector V2.

## Build de uma nova imagem

Execute a partir da raiz `bim-mcp-backend-review`:

```powershell
gcloud builds submit `
  --project my-projectv2-480802 `
  --region us-central1 `
  --tag us-central1-docker.pkg.dev/my-projectv2-480802/cloud-run-source-deploy/bim-api-v2:TAG `
  --machine-type e2-highcpu-8 `
  --timeout 3600s `
  .
```

O `.gcloudignore` exclui datasets, nuvens, artefatos e runtimes locais, mas
inclui `.runtime/models/random_forest.pkl`. A imagem usa Python 3.12 porque o
conjunto congelado inclui NumPy 2.5.1.

O Dockerfile compila GNU LibreDWG 0.13.4 a partir do asset oficial, verifica o
SHA-256 publicado e leva apenas o runtime instalado para a imagem final. O texto
GPLv3+ acompanha a imagem em `/usr/share/doc/libredwg/COPYING`.

## Publicacao segura

Primeiro crie uma revisao sem trafego:

```powershell
gcloud run deploy bim-api-v2 `
  --project my-projectv2-480802 `
  --region us-central1 `
  --image us-central1-docker.pkg.dev/my-projectv2-480802/cloud-run-source-deploy/bim-api-v2:TAG `
  --port 8080 --cpu 2 --memory 4Gi `
  --concurrency 2 --min-instances 0 --max-instances 1 `
  --timeout 3600 --set-env-vars SCAN_JOBS_SYNC=true `
  --cpu-boost --no-traffic --tag candidate --allow-unauthenticated
```

Depois do smoke test na URL com tag, envie o trafego:

```powershell
gcloud run services update-traffic bim-api-v2 `
  --project my-projectv2-480802 `
  --region us-central1 `
  --to-latest
```

## Rollback

```powershell
gcloud run services update-traffic bim-api-v2 `
  --project my-projectv2-480802 `
  --region us-central1 `
  --to-revisions bim-api-v2-00004-vov=100
```

O rollback troca somente o trafego; nao apaga a revisao nova.

A concorrencia HTTP deve permanecer em `2`, igual ao numero de threads do
Gunicorn definido no `Dockerfile`. Uma thread pode executar o worker de longa
duracao enquanto a outra atende login, webhook, polling e health check. O
paralelismo pago nao e controlado por esse valor: a fila Cloud Tasks
`plan2bim-astra` deve continuar com `maxConcurrentDispatches=1`.

## Limites conscientes deste primeiro lancamento

1. Nao ha login nem banco, conforme a decisao do produto. O maximo de uma
   instancia limita custo, mas uma requisicao pesada pode ocupar o unico worker.
2. Sessoes, outputs e jobs sao temporarios e locais ao container. Para escalar
   horizontalmente, a proxima engenharia deve mover uploads/outputs para Cloud
   Storage e jobs para Cloud Tasks ou Cloud Run Jobs.
3. O servidor MCP local/stdio e a biblioteca de receitas estao no repositorio,
   mas este deploy publica a plataforma HTTP e suas APIs. Um MCP remoto exige
   transporte HTTP e alguma protecao de acesso antes de ser exposto na internet.
4. O audit de dependencias web de producao encontrou 1 alerta alto indireto e 2
   moderados, sem criticos. A correcao sugerida passa por atualizar
   `@react-three/drei` e deve entrar em uma revisao testada separadamente.
