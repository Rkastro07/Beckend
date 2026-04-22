import { AnalysisResult } from "../types";

const BACKEND_URL = "http://localhost:8081";

export const generateExecutiveReport = async (
    analysis: AnalysisResult,
    csvContent: string | null
): Promise<string> => {
    const items = analysis.resultados || analysis.items || [];
    const stats = analysis.estatisticas || analysis.summary;

    let prompt = `Atue como um Engenheiro Sênior de Planejamento e Controle de Obras.
Gere um relatório executivo conciso focado em riscos e desvios.

RESUMO DO PROGRESSO FÍSICO (REAL - Via Scanner 3D/IFC):
${JSON.stringify(stats, null, 2)}

AMOSTRA DE ITENS INDIVIDUAIS (TOP 20):
${JSON.stringify(items.slice(0, 20), null, 2)}`;

    if (csvContent) {
        prompt += `

DADOS DO PLANEJAMENTO (CRONOGRAMA CSV):
${csvContent.slice(0, 2000)}`;
    } else {
        prompt += `

(Nenhum cronograma CSV foi fornecido para comparação. Analise apenas o progresso físico atual.)`;
    }

    prompt += `

Diretrizes:
1. Compare o realizado vs planejado (se disponível).
2. Destaque categorias com maior atraso.
3. Sugira 3 ações corretivas imediatas.
4. Use formatação Markdown profissional com seções claras.
5. Seja direto e objetivo, máximo 500 palavras.`;

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
    return "Assistente de voz requer Gemini API. Use a caixa de texto abaixo para conversar via DeepSeek.";
};

// CHAT COM DEEPSEEK - DIRETO SEM BACKEND
export const processChatQuery = async (
    question: string,
    analysis: AnalysisResult,
    csvContent: string | null
): Promise<string> => {
    const items = analysis.resultados || analysis.items || [];
    const stats = analysis.estatisticas || analysis.summary;

    const prompt = `Você é o "BIM Superintendent AI", um assistente técnico de obra.
Responda à pergunta do engenheiro com base ESTRITAMENTE nos dados fornecidos.
Se a resposta não estiver nos dados, diga que não sabe. Seja curto, direto e técnico (máximo 200 palavras).

DADOS DA OBRA:
Estatísticas: ${JSON.stringify(stats, null, 2)}
Amostra de Itens: ${JSON.stringify(items.slice(0, 30), null, 2)}
${csvContent ? `Cronograma (CSV): ${csvContent.slice(0, 1000)}` : '(Sem cronograma)'}

PERGUNTA DO USUÁRIO: ${question}`;

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
