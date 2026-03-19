import { Floor, AnalysisResult, BimItem } from '../types';

const API_BASE_URL = 'http://localhost:8080';

export const listFloors = async (ifcFile: File): Promise<Floor[]> => {
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
        return data.pavimentos.map((p: string) => ({ id: p, name: p }));
    }
    
    if (Array.isArray(data)) return data;
    
    return [];
  } catch (error) {
    console.error("Failed to list floors:", error);
    alert("Erro ao conectar com o servidor. Verifique se o Cloud Run está ativo.");
    return []; 
  }
};

export const analyzeFloor = async (
  ifcFile: File,
  plyFile: File,
  floorId: string
): Promise<AnalysisResult> => {
  const formData = new FormData();
  formData.append('ifc_file', ifcFile);
  formData.append('ply_file', plyFile);
  formData.append('pavimento', floorId);

  try {
    const response = await fetch(`${API_BASE_URL}/api/analisar_pavimento`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
       const errText = await response.text();
       throw new Error(`API Error: ${response.status} - ${errText}`);
    }
    
    const json = await response.json();
    
    // Garante que o frontend receba o formato certo
    const items: BimItem[] = json.resultados || [];
    
    // Calcula estatísticas para o gráfico
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