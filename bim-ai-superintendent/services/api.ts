import { Floor, AnalysisResult, BimItem } from '../types';

const API_BASE_URL = 'http://localhost:8081';

export interface ListFloorsResult {
  floors: Floor[];
  ifcToken: string | null;  // usado pelos endpoints de análise p/ pular reupload
}

export const listFloors = async (ifcFile: File): Promise<ListFloorsResult> => {
  const formData = new FormData();
  formData.append('ifc_file', ifcFile);

  try {
    const response = await fetch(`${API_BASE_URL}/api/listar_pavimentos`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    const data = await response.json();

    if (data.pavimentos && Array.isArray(data.pavimentos)) {
      const floors = data.pavimentos.map((p: string) => ({ id: p, name: p }));
      return { floors, ifcToken: data.ifc_token ?? null };
    }

    if (Array.isArray(data)) return { floors: data, ifcToken: null };

    return { floors: [], ifcToken: null };
  } catch (error) {
    console.error("Failed to list floors:", error);
    alert("Erro ao conectar com o servidor. Verifique se o backend está ativo.");
    return { floors: [], ifcToken: null };
  }
};

export type AnalysisMode = 'bbox' | 'ai' | 'both' | 'instances';

const _doAnalysis = async (
  ifcFile: File,
  plyFile: File,
  floorId: string,
  endpoint: string,
  ifcToken: string | null
): Promise<AnalysisResult> => {
  const formData = new FormData();
  // Se temos token, o backend reusa o IFC cacheado e pulamos o reupload (pode economizar dezenas de MB)
  if (ifcToken) {
    formData.append('ifc_token', ifcToken);
  } else {
    formData.append('ifc_file', ifcFile);
  }
  formData.append('ply_file', plyFile);
  formData.append('pavimento', floorId);

  try {
    let response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
    });

    // 410 = ifc_token desconhecido (ex: backend reiniciou). Retry com upload completo.
    if (response.status === 410 && ifcToken) {
      console.warn("ifc_token expirado, refazendo upload completo do IFC");
      const retryForm = new FormData();
      retryForm.append('ifc_file', ifcFile);
      retryForm.append('ply_file', plyFile);
      retryForm.append('pavimento', floorId);
      response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        body: retryForm,
      });
    }

    if (!response.ok) {
       const errText = await response.text();
       throw new Error(`API Error: ${response.status} - ${errText}`);
    }
    
    const json = await response.json();

    // Garante que o frontend receba o formato certo
    const items: BimItem[] = json.resultados || [];
    const modo = json.modo || '';

    // Instâncias: estatísticas simplificadas (só conta detectados)
    if (modo === 'instancias') {
      const detectados = items.filter((i: any) => i.status.code === 'DETECTADO').length;
      return {
        pavimento: floorId,
        floor_id: floorId,
        session_id: json.session_id || 'unknown',
        timestamp: new Date().toISOString(),
        items, resultados: items,
        modo: 'instancias',
        estatisticas: {
          total: items.length,
          completos: 0, parciais: 0, iniciados: 0, ausentes: 0,
          detectados,
          progresso_geral: detectados > 0 ? 100 : 0
        },
        summary: {
          total_elements: items.length,
          executed_elements: detectados,
          progress_percentage: 100,
          status_by_category: {},
          risks_detected: []
        }
      };
    }

    // Calcula estatísticas para o gráfico (bbox/ai)
    const total = items.length;
    const executed = items.filter((i: any) => i.status.code === 'COMPLETO' || i.status.code === 'PARCIAL').length;
    const percentage = total > 0 ? (executed / total) * 100 : 0;
    
    const categories: any = {};
    const risks: string[] = [];

    items.forEach((i: any) => {
        if (!categories[i.tipo]) categories[i.tipo] = { total: 0, executed: 0 };
        categories[i.tipo].total++;
        if (i.status.code === 'COMPLETO') categories[i.tipo].executed++;
        
        if (i.status.code === 'AUSENTE' || i.status.code === 'PARCIAL') {
            risks.push(`Desvio detectado em ${i.nome} (${i.status.texto})`);
        }
    });

    const estatisticas = {
        total,
        completos: items.filter((i: any) => i.status.code === 'COMPLETO').length,
        parciais: items.filter((i: any) => i.status.code === 'PARCIAL').length,
        iniciados: items.filter((i: any) => i.status.code === 'INICIADO').length,
        ausentes: items.filter((i: any) => i.status.code === 'AUSENTE').length,
        progresso_geral: percentage
    };

    return {
        pavimento: floorId, // Match types.ts
        floor_id: floorId,
        session_id: json.session_id || 'unknown',
        timestamp: new Date().toISOString(),
        items: items, // Legacy support
        resultados: items, // New support
        estatisticas: estatisticas,
        summary: {
            total_elements: total,
            executed_elements: executed,
            progress_percentage: percentage,
            status_by_category: categories,
            risks_detected: risks.slice(0, 5)
        }
    };

  } catch (error) {
    console.error("Failed to analyze floor:", error);
    throw error;
  }
};

export const analyzeFloor = (
  ifcFile: File, plyFile: File, floorId: string, ifcToken: string | null = null
): Promise<AnalysisResult> => _doAnalysis(ifcFile, plyFile, floorId, '/api/analisar_pavimento', ifcToken);

export const analyzeFloorAI = (
  ifcFile: File, plyFile: File, floorId: string, ifcToken: string | null = null
): Promise<AnalysisResult> => _doAnalysis(ifcFile, plyFile, floorId, '/api/analisar_ai', ifcToken);

export const analyzeFloorInstances = (
  ifcFile: File, plyFile: File, floorId: string, ifcToken: string | null = null
): Promise<AnalysisResult> => _doAnalysis(ifcFile, plyFile, floorId, '/api/analisar_instancias', ifcToken);