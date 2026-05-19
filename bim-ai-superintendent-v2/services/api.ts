import { Floor, AnalysisResult, BimItem } from '../types';

const API_BASE_URL = 'http://localhost:8081';

export interface ListFloorsResult {
  floors: Floor[];
  ifcToken: string | null;  // usado pra pular reupload do IFC na análise
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
    alert("Erro ao conectar com o backend (porta 8081). Verifique se está ativo.");
    return { floors: [], ifcToken: null };
  }
};

/**
 * Análise via pipeline Sonata + OBB do IFC.
 * Etapas: align PLY↔IFC → Sonata classifica nuvem inteira → filtra pts por OBB
 * de cada elemento IFC → verdict por classe dominante dentro da OBB.
 * Endpoint exclusivo desse front. Não fala com /api/analisar_pavimento,
 * /api/analisar_ai ou /api/analisar_instancias (esses são do front legado).
 */
export const analyzeFloorV2 = async (
  ifcFile: File,
  plyFile: File,
  floorId: string,
  ifcToken: string | null = null
): Promise<AnalysisResult> => {
  const formData = new FormData();
  if (ifcToken) {
    formData.append('ifc_token', ifcToken);
  } else {
    formData.append('ifc_file', ifcFile);
  }
  formData.append('ply_file', plyFile);
  formData.append('pavimento', floorId);

  try {
    let response = await fetch(`${API_BASE_URL}/api/analisar_ai_v2`, {
      method: 'POST',
      body: formData,
    });

    // 410 = ifc_token desconhecido (backend reiniciou). Retry com upload completo.
    if (response.status === 410 && ifcToken) {
      console.warn("ifc_token expirado, refazendo upload completo do IFC");
      const retryForm = new FormData();
      retryForm.append('ifc_file', ifcFile);
      retryForm.append('ply_file', plyFile);
      retryForm.append('pavimento', floorId);
      response = await fetch(`${API_BASE_URL}/api/analisar_ai_v2`, {
        method: 'POST',
        body: retryForm,
      });
    }

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`API Error: ${response.status} - ${errText}`);
    }

    const json = await response.json();

    const items: BimItem[] = json.resultados || [];
    const estat = json.estatisticas || {};
    const total = items.length;
    const executed = items.filter((i: any) =>
      i.status?.code === 'COMPLETO' || i.status?.code === 'PARCIAL'
    ).length;

    // Sumário p/ ReportView (mantém forma legada usada pela aba Relatório)
    const categories: Record<string, { total: number; executed: number }> = {};
    const risks: string[] = [];
    items.forEach((i: any) => {
      if (!categories[i.tipo]) categories[i.tipo] = { total: 0, executed: 0 };
      categories[i.tipo].total++;
      if (i.status?.code === 'COMPLETO') categories[i.tipo].executed++;
      if (i.status?.code === 'AUSENTE' || i.status?.code === 'PARCIAL') {
        risks.push(`Desvio em ${i.nome} (${i.status?.texto || i.status?.code})`);
      }
    });
    // Adições viram risco também (construído fora do plano)
    (json.adicoes || []).forEach((a: any) => {
      risks.push(`Adição não planejada: ${a.scan_class} (n_pts=${a.n_pts})`);
    });

    return {
      pavimento: floorId,
      floor_id: floorId,
      session_id: json.session_id || 'unknown',
      timestamp: new Date().toISOString(),
      items,
      resultados: items,
      adicoes: json.adicoes || [],
      global_cloud: json.global_cloud,
      meta: json.meta,
      estatisticas: {
        total: estat.total ?? total,
        completos: estat.completos ?? 0,
        parciais: estat.parciais ?? 0,
        iniciados: estat.iniciados ?? 0,
        ausentes: estat.ausentes ?? 0,
        adicoes: estat.adicoes ?? (json.adicoes?.length ?? 0),
        progresso_geral: estat.progresso_geral ?? (total > 0 ? (executed / total) * 100 : 0),
      },
      summary: {
        total_elements: total,
        executed_elements: executed,
        progress_percentage: total > 0 ? (executed / total) * 100 : 0,
        status_by_category: categories,
        risks_detected: risks.slice(0, 10),
      },
    };
  } catch (error) {
    console.error("Failed to analyze (v2):", error);
    throw error;
  }
};
