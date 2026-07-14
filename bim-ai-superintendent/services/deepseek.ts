import { AnalysisResult } from "../types";

const BACKEND_URL = "http://localhost:8081";

export const generateExecutiveReport = async (
    analysis: AnalysisResult,
    csvContent: string | null
): Promise<string> => {
    const items = analysis.resultados || analysis.items || [];
    const stats = analysis.estatisticas || analysis.summary;

    // Compacta itens pro relatório (não precisa de guid/bbox, mas precisa de pavimento e dimensões)
    const itemsReport = items.map((i: any) => {
      const item: any = {
        nome: i.nome,
        tipo: i.tipo,
        status: i.status?.code,
      };
      if (i.pavimento) item.pavimento = i.pavimento;
      if (i.cobertura_vertical !== undefined) item.cobertura_pct = Math.round(i.cobertura_vertical * 100);
      else if (i.cobertura !== undefined) item.cobertura_pct = i.cobertura;
      if (i.dimensoes?.executado) item.altura_exec = i.dimensoes.executado.z;
      if (i.dimensoes?.planejado) item.altura_plan = i.dimensoes.planejado.z;
      if (i.volume_ifc) item.volume_m3 = i.volume_ifc;
      if (i.cross_floor) item.cross_floor = i.andares_atravessa;
      return item;
    });

    let prompt = `Você é um Engenheiro Sênior de Planejamento e Controle de Obras analisando dados de um sistema automatizado de vistoria BIM.

O sistema compara automaticamente a nuvem de pontos 3D (escaneamento laser da obra real) com o modelo BIM (IFC do projeto) para classificar cada elemento construtivo.

## DADOS DA ANÁLISE

### Estatísticas Gerais
${JSON.stringify(stats, null, 2)}

### Todos os Elementos (${itemsReport.length} itens)
${JSON.stringify(itemsReport)}`;

    if (csvContent) {
        prompt += `

### Cronograma Físico-Financeiro (CSV do planejamento)
${csvContent.slice(0, 3000)}`;
    } else {
        prompt += `

(Sem cronograma CSV — analise apenas o progresso físico detectado pelo scanner.)`;
    }

    prompt += `

## INSTRUÇÕES PARA O RELATÓRIO

Gere um **Relatório Executivo de Vistoria Digital** em Markdown com estas seções:

### 1. Resumo Executivo
- Progresso geral da obra (% e números absolutos)
- Data da vistoria digital
- Pavimento(s) analisado(s)

### 2. Progresso por Categoria
- Tabela markdown com: Tipo | Total | Completos | Parciais | Ausentes | %
- Destaque as categorias com pior desempenho

### 3. Elementos Críticos
- Liste os elementos AUSENTES e PARCIAIS mais relevantes (paredes estruturais, lajes, pilares)
- Se houver cronograma, compare real vs planejado e calcule desvio

### 4. Riscos Identificados
- Riscos de atraso com base nos elementos ausentes/parciais
- Impacto potencial no cronograma

### 5. Recomendações
- 3 a 5 ações corretivas concretas e priorizadas
- Indicar urgência (imediata / curto prazo / próxima medição)

**Regras:**
- Máximo 600 palavras
- Linguagem técnica, direta, sem floreios
- Todos os números devem vir dos dados fornecidos (não inventar)
- Use tabelas markdown quando apropriado`;

    try {
        const response = await fetch(`${BACKEND_URL}/api/generate_report`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Backend Error: ${response.status}`);
        }

        const data = await response.json();
        return data.content || "Resposta vazia do backend";
    } catch (error) {
        console.error("❌ DeepSeek Report Error:", error);

        if (error instanceof TypeError && error.message === 'Failed to fetch') {
            return `**Erro de Conexão**\n\nNão foi possível conectar ao backend ${BACKEND_URL}`;
        }

        return `Erro ao gerar relatório: ${error instanceof Error ? error.message : 'Erro desconhecido'}`;
    }
};

export const processVoiceQuery = async (
    audioBase64: string,
    analysis: AnalysisResult,
    csvContent: string | null
): Promise<string> => {
    return "Assistente de voz indisponível no momento. Use a caixa de texto abaixo para conversar via DeepSeek.";
};

// CHAT COM DEEPSEEK - DIRETO SEM BACKEND
export const processChatQuery = async (
    question: string,
    analysis: AnalysisResult,
    csvContent: string | null
): Promise<string> => {
    const items = analysis.resultados || analysis.items || [];
    const stats = analysis.estatisticas || analysis.summary;

    // Extrai todos os campos úteis de cada item pra o DeepSeek ter contexto completo
    const itemsFull = items.map((i: any) => {
      const item: any = {
        nome: i.nome,
        tipo: i.tipo,
        status: i.status?.code,
        status_texto: i.status?.texto,
        guid: i.guid,
      };
      // Pavimento / andar
      if (i.pavimento) item.pavimento = i.pavimento;
      // Cobertura
      if (i.cobertura_vertical !== undefined) item.cobertura_pct = Math.round(i.cobertura_vertical * 100);
      else if (i.cobertura !== undefined) item.cobertura_pct = i.cobertura;
      // Dimensões executado vs planejado
      if (i.dimensoes) {
        if (i.dimensoes.executado) item.dim_executado = i.dimensoes.executado;
        if (i.dimensoes.planejado) item.dim_planejado = i.dimensoes.planejado;
        if (i.dimensoes.progresso) item.dim_progresso_pct = i.dimensoes.progresso;
      }
      // Volume IFC
      if (i.volume_ifc) item.volume_ifc_m3 = i.volume_ifc;
      // Densidade de pontos
      if (i.densidade) item.densidade_pts = i.densidade;
      if (i.pontos) item.n_pontos = i.pontos;
      // Cross-floor (estrutura vertical que atravessa andares)
      if (i.cross_floor) {
        item.cross_floor = true;
        item.andares_atravessa = i.andares_atravessa;
        item.altura_total_m = i.altura_total;
      }
      // Conexão (parede-T, piso/teto)
      if (i.eh_conexao) item.eh_conexao = true;
      // Match info (v2 Sonata)
      if (i.match_info) {
        item.classe_sonata = i.match_info.classe_dominante || i.match_info.scan_class;
        item.n_pts_dentro = i.match_info.n_pts_dentro;
      }
      return item;
    });

    const prompt = `Você é o "BIM Superintendent AI", um assistente técnico especializado em acompanhamento de obra.
Você tem acesso a TODOS os dados da vistoria digital — responda com precisão e completude.
Se a pergunta pedir listas, liste TODOS os elementos relevantes, sem truncar.

## REGRAS
- Você É o sistema de vistoria BIM. Os dados fornecidos são a verdade da obra. NUNCA sugira "verificar no modelo BIM", "consultar o IFC" ou "checar manualmente" — você já tem todas as informações.
- Se algum dado não estiver disponível (ex: pavimento vazio), diga "dado não disponível na análise" em vez de mandar o usuário procurar em outro lugar.
- Use **tabelas Markdown** para listas de elementos (colunas: Nome | Tipo | Pavimento | Status | Cobertura)
- Use **negrito** para destacar números e status críticos
- Use headers ## e ### para organizar seções
- Seja direto — sem introduções longas, sem disclaimers desnecessários
- Quando listar elementos, agrupe por pavimento ou por tipo
- Nunca repita o GUID na resposta, use apenas o nome do elemento

## DADOS DA OBRA

### Estatísticas Gerais
${JSON.stringify(stats, null, 2)}

### Lista COMPLETA de Elementos (${itemsFull.length} itens)
${JSON.stringify(itemsFull)}

${csvContent ? `### Cronograma Físico-Financeiro (CSV)\n${csvContent.slice(0, 3000)}` : '(Sem cronograma CSV)'}

## PERGUNTA DO ENGENHEIRO
${question}`;

    try {
        const response = await fetch(`${BACKEND_URL}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Backend Error: ${response.status}`);
        }

        const data = await response.json();
        return data.content || "Sem resposta do backend.";

    } catch (error) {
        console.error("❌ DeepSeek Chat Error:", error);

        if (error instanceof TypeError && error.message === 'Failed to fetch') {
            return "Erro de conexão com o backend. Verifique se o deploy foi concluído.";
        }

        return `Erro: ${error instanceof Error ? error.message : 'Erro desconhecido'}`;
    }
};
