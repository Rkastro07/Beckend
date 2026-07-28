
export enum AppTab {
  VISTORIA = 'vistoria',
  RELATORIO = 'relatorio',
  VOICE = 'voice',
  FERRAMENTAS = 'ferramentas',
  PLANTA = 'planta',
  SCAN = 'scan'
}

export interface Floor {
  id: string;
  name: string;
}

export interface BBox {
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
  zmin: number;
  zmax: number;
}

export interface BimItem {
  guid: string;
  nome: string;
  tipo: string;
  pavimento?: string;
  pontos: number;
  cobertura_vertical: number;
  cobertura?: number;
  volume_ifc?: number;
  densidade?: number;
  eh_conexao?: boolean;
  status: {
    code: 'COMPLETO' | 'PARCIAL' | 'INICIADO' | 'AUSENTE' | 'ADICAO';
    emoji: string;
    texto: string;
    cor: string;
  };
  // Pipeline v2: informação de matching scan↔IFC
  rf_version?: 'v1' | 'v2' | null;
  match_info?: {
    cost: number;
    scan_class: string;
    scan_centroid: [number, number, number];
    scan_n_pts: number;
    scan_conf: number;
    matched_by: string;
  } | null;
  dimensoes: {
    progresso: { z: number };
  };
  json_file?: string;
  json_url?: string;
  ply_file?: string;
  bbox_normalized: BBox;
  obb_corners?: number[][] | null; // 8 vértices OBB em coords Three.js (null → usa AABB)
  // Cross-floor (pilar/shaft/estrutura vertical longa fatiada por andar)
  cross_floor?: boolean;
  altura_total?: number;                 // altura total do elemento (todos os andares)
  z_total?: [number, number];            // [zmin, zmax] globais
  andares_atravessa?: string[];          // nomes dos pavimentos que o elemento cruza
  storey_original_ifc?: string;          // pavimento a que o IFC atribui o elemento
  fatia_atual?: {
    pavimento: string;
    z_range:   [number, number];
    altura:    number;
  };
}

export interface AnalysisResult {
  pavimento?: string;
  floor_id?: string;
  session_id?: string;
  timestamp?: string;
  estatisticas?: {
    total: number;
    completos: number;
    parciais: number;
    iniciados: number;
    ausentes: number;
    adicoes?: number;        // v2: construído fora do plano
    progresso_geral: number;
  };
  // Pipeline v2: instâncias scan sem par IFC (ADICAO)
  adicoes?: Array<{
    scan_class: string;
    centroid: [number, number, number];
    n_pts: number;
    mean_conf: number;
    volume: number;
    bbox?: BBox;
    status: { code: 'ADICAO'; texto: string; cor: string };
  }>;
  meta?: Record<string, any>;
  // Mantém compatibilidade com código antigo
  summary?: {
    total_elements: number;
    executed_elements: number;
    progress_percentage: number;
    status_by_category?: Record<string, any>;
    risks_detected?: string[];
  };
  items?: BimItem[];
  resultados?: BimItem[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: Date;
}