import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import { AnalysisResult } from "../types";

// Helper to init the client
const getClient = () => {
  // Prioriza variável de ambiente. A chave deve vir exclusivamente de process.env.API_KEY.
  const apiKey = process.env.API_KEY;
  if (!apiKey) throw new Error("API Key not found in environment variables.");
  return new GoogleGenAI({ apiKey });
};

export const generateExecutiveReport = async (
  analysis: AnalysisResult,
  csvContent: string | null
): Promise<string> => {
  const ai = getClient();
  const model = "gemini-2.5-flash"; // Good balance of speed and reasoning

  let prompt = `
    Atue como um Engenheiro Sênior de Planejamento e Controle de Obras.
    Gere um relatório executivo conciso focado em riscos e desvios.

    RESUMO DO PROGRESSO FÍSICO (REAL - Via Scanner 3D/IFC):
    ${JSON.stringify(analysis.summary, null, 2)}
    
    AMOSTRA DE ITENS INDIVIDUAIS (TOP 20):
    ${JSON.stringify(analysis.items.slice(0, 20), null, 2)}
  `;

  if (csvContent) {
    prompt += `
    \nDADOS DO PLANEJAMENTO (CRONOGRAMA CSV):
    ${csvContent}
    `;
  } else {
    prompt += `\n(Nenhum cronograma CSV foi fornecido para comparação. Analise apenas o progresso físico atual.)`;
  }

  prompt += `
    \nDiretrizes:
    1. Compare o realizado vs planejado (se disponível).
    2. Destaque categorias com maior atraso.
    3. Sugira 3 ações corretivas imediatas.
    4. Use formatação Markdown profissional.
  `;

  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model,
      contents: prompt,
    });
    return response.text || "Não foi possível gerar o relatório.";
  } catch (error) {
    console.error("Gemini Report Error:", error);
    return "Erro ao comunicar com a IA para gerar o relatório.";
  }
};

export const processVoiceQuery = async (
  audioBase64: string,
  analysis: AnalysisResult,
  csvContent: string | null
): Promise<string> => {
  const ai = getClient();
  const model = "gemini-2.5-flash"; 

  const systemPrompt = `
    Você é o "BIM Superintendent AI". Um assistente de obra inteligente.
    Responda à pergunta de voz do engenheiro com base ESTRITAMENTE nos dados técnicos fornecidos abaixo.
    Se a resposta não estiver nos dados, diga que não sabe. Seja curto, direto e técnico.
  `;

  const contextData = `
    CONTEXTO DA OBRA:
    Resumo (JSON): ${JSON.stringify(analysis.summary)}
    Itens Detalhados (Amostra): ${JSON.stringify(analysis.items.slice(0, 50))}
    Cronograma (CSV): ${csvContent || "Não disponível"}
  `;

  try {
    const response = await ai.models.generateContent({
      model,
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: "audio/webm",
              data: audioBase64,
            },
          },
          {
            text: `${systemPrompt}\n\n${contextData}\n\nResponda ao áudio:`,
          },
        ],
      },
    });

    return response.text || "Desculpe, não entendi o áudio.";
  } catch (error) {
    console.error("Gemini Voice Error:", error);
    return "Erro ao processar o áudio. Verifique sua chave de API ou conexão.";
  }
};
// Chat de texto com Gemini
export const processChatQuery = async (
  question: string,
  analysis: AnalysisResult,
  csvContent: string | null
): Promise<string> => {
  const ai = getClient();
  const model = "gemini-2.5-flash";
  const systemPrompt = `Você é o "BIM Superintendent AI", um assistente técnico de obra.
Responda à pergunta do engenheiro com base ESTRITAMENTE nos dados fornecidos.
Se a resposta não estiver nos dados, diga que não sabe. Seja curto, direto e técnico (máximo 200 palavras).`;
  const contextData = `
DADOS DA OBRA:
Resumo: ${JSON.stringify(analysis.summary || analysis.estatisticas)}
Itens: ${JSON.stringify((analysis.items || analysis.resultados || []).slice(0, 30))}
Cronograma: ${csvContent || "Não disponível"}
PERGUNTA: ${question}`;
  try {
    const response = await ai.models.generateContent({
      model,
      contents: `${systemPrompt}\n\n${contextData}`,
    });
    return response.text || "Não foi possível gerar resposta.";
  } catch (error) {
    console.error("Gemini Chat Error:", error);
    return "Erro ao processar pergunta. Verifique sua chave de API.";
  }
};