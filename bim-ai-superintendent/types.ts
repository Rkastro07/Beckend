
export enum AppTab {
  VISTORIA = 'vistoria',
  RELATORIO = 'relatorio',
  VOICE = 'voice'
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
  pontos: number;
  cobertura_vertical: number;
  status: {
    code: 'COMPLETO' | 'PARCIAL' | 'INICIADO' | 'AUSENTE';
    emoji: string;
    texto: string;
    cor: string;
  };
  dimensoes: {
    progresso: { z: number };
  };
  json_file?: string;
  json_url?: string;
  ply_file?: string;
  bbox_normalized: BBox;
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
    progresso_geral: number;
  };
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