# Base de conhecimento do BIM MCP

Esta pasta reúne a documentação autoral e as receitas técnicas do sistema sem
misturá-las com código executável, datasets, dependências ou outputs de testes.
Os arquivos em `source_docs/` são cópias preservadas nos mesmos caminhos
relativos em que estavam no backend original.

## Comece por aqui

### Arquitetura e visão do produto

- [Backend e capacidades gerais](source_docs/README.md)
- [Stack do sistema](source_docs/STACK.md)
- [Documentação da API de análise](source_docs/documentation.md)
- [Inventário de features](source_docs/docs/FEATURES_E_APZINHOS.md)
- [Pipeline backend, JSON e frontend](source_docs/docs/PIPELINE_BACK_FRONT.md)
- [Plano do gerador As-Built IFC](source_docs/docs/PLANO_ASBUILT_IFC.md)
- [Jornada de validação com scans reais](source_docs/docs/JORNADA_HORIZONTE.md)
- [Roadmap de problemas estruturais](source_docs/docs/FUTURO.md)
- [Changelog técnico](source_docs/docs/CHANGELOG_2026-04-28.md)

### Receitas que não podem se perder

- [Receita MinkowskiEngine no Colab/A100](source_docs/docs/colab_minkowski_recipe.md)
- [Handoff do treino Mask3D](source_docs/docs/handoff_mask3d_colab_session.md)
- [Notebook Mask3D BIM no Colab](source_docs/experiments/mask3d/Mask3D_BIM_Colab.ipynb)
- [Plano de dataset HELIOS++](source_docs/docs/helios_dataset_plan.md)
- [Receita Cloud2BIM adaptada](source_docs/experiments/cloud2bim/README.md)
- [README do Cloud2BIM original](source_docs/experiments/cloud2bim/cloud2bim_patched/README_original.md)
- [Configuração Allplan](source_docs/experiments/cloud2bim/configs_exemplo/config_allplan.yaml)
- [Configuração de scan XYZ real](source_docs/experiments/cloud2bim/configs_exemplo/config_real_xyz.yaml)
- [Configuração de simulação](source_docs/experiments/cloud2bim/configs_exemplo/config_sim.yaml)

### Machine learning e visão computacional

- [Aula e decisões do Random Forest](source_docs/docs/RANDOM_FOREST.md)
- [Sonata — referência original](source_docs/experiments/sonata/repo/README.md)
- [Como contribuir com Sonata](source_docs/experiments/sonata/repo/.github/CONTRIBUTING.md)

### Interfaces

- [Frontend atual](source_docs/bim-ai-superintendent/README.md)
- [Frontend v2](source_docs/bim-ai-superintendent-v2/README.md)

## Documentação criada especificamente para o MCP

Estes documentos não são snapshots: pertencem à evolução desta cópia.

- [Revisão de escopo do MCP](../MCP_SCOPE_REVIEW.md)
- [Entrada geométrica do modelador](../docs/GEOMETRY_IMPORT.md)
- [Biblioteca de receitas de autoria BIM](../bim_authoring/README.md)
- [Relatório consolidado do BIM MCP](../docs/RELATORIO_BIM_MCP_ATE_AGORA.md)

## Como o MCP deverá publicar esta base

O servidor pode expor:

- `bim://knowledge/index` — este índice e o manifesto;
- `bim://knowledge/document/{path}` — conteúdo integral de um documento;
- `search_bim_knowledge(query, tags)` — busca por título, caminho, tags e texto;
- `get_bim_recipe(name)` — receitas operacionais com pré-requisitos e alertas;
- `list_bim_knowledge(tags)` — descoberta sem precisar conhecer os caminhos.

As receitas estruturadas e executáveis ficam em `bim_authoring/recipes/`. Elas
complementam este arquivo histórico: a base `knowledge/` explica decisões e
experimentos; `bim_authoring/` descreve operações de modelagem com entradas,
relações IFC, geometria, sequência, validação e falhas conhecidas.

O arquivo [manifest.json](manifest.json) registra tamanho, SHA-256 e tags dos
24 arquivos copiados. Ele permite detectar alterações no backend original antes
de atualizar os resources do MCP.

## Política da cópia

Incluído:

- Markdown e referências técnicas autorais;
- notebooks que funcionam como receita reproduzível;
- configurações-exemplo usadas pelas receitas;
- documentação de projetos externos mantidos dentro dos experimentos.

Excluído:

- `node_modules`, ambientes virtuais e `site-packages`;
- worktrees auxiliares e caches;
- datasets, labels e JSONs de treino;
- outputs, logs e PDFs gerados automaticamente;
- arquivos temporários do Word e artefatos de build.
