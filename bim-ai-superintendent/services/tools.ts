// Serviço das FERRAMENTAS (conversores + geradores).
// Fala com os endpoints /api/tools/* do backend (app_obb + tools_endpoints.py).

const API_BASE_URL = 'http://localhost:8081';

export interface ToolResult {
  ok: boolean;
  download_url: string;      // caminho relativo /outputs/... no backend
  n_pontos?: number;
  [k: string]: any;
}

// resolve a URL de download absoluta a partir do caminho /outputs/... do backend
export const downloadUrl = (rel: string): string => `${API_BASE_URL}${rel}`;

async function postTool(path: string, file: File,
                        params: Record<string, string> = {}): Promise<ToolResult> {
  const form = new FormData();
  form.append('file', file);
  Object.entries(params).forEach(([k, v]) => form.append(k, v));

  const resp = await fetch(`${API_BASE_URL}${path}`, { method: 'POST', body: form });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data.error) {
    throw new Error(data.error || `Erro ${resp.status}`);
  }
  return data as ToolResult;
}

// ---- Conversores ----
export const objToPly = (file: File, densidade = 130) =>
  postTool('/api/tools/obj-to-ply', file, { densidade: String(densidade) });

export const usdzToPly = (file: File, densidade = 130, apenasEstrutura = true) =>
  postTool('/api/tools/usdz-to-ply', file, {
    densidade: String(densidade),
    apenas_estrutura: String(apenasEstrutura),
  });

export const ascToPly = (file: File, subsample = 20) =>
  postTool('/api/tools/asc-to-ply', file, { subsample: String(subsample) });

// ---- Gerador de nuvem sintética ----
export interface GerarNuvemResult extends ToolResult {
  labels_url: string;
  rotulo: string;
  stats: { completo: number; parcial: number; ausente: number; total_objetos: number };
}

// modo 'estagio': estagio = '0'..'9' ou 'aleatorio'
export const gerarNuvemEstagio = (file: File, estagio: string, seed = 0) =>
  postTool('/api/tools/gerar-nuvem', file, {
    modo: 'estagio', estagio, seed: String(seed),
  }) as Promise<GerarNuvemResult>;

// modo 'manual': usuário define % ausente e % parcial (0..1)
export const gerarNuvemManual = (file: File, pctAusente: number, pctParcial: number, seed = 0) =>
  postTool('/api/tools/gerar-nuvem', file, {
    modo: 'manual',
    pct_ausente: String(pctAusente),
    pct_parcial: String(pctParcial),
    seed: String(seed),
  }) as Promise<GerarNuvemResult>;

// ====================================================================
//  PLANTA -> BIM (editor): parse (DXF -> modelo) e gerar (modelo -> IFC)
// ====================================================================
export interface Parede {
  id: string; ax: number; ay: number; bx: number; by: number;
  espessura: number; layer: string;
}
export interface Abertura {
  id: string; parede_id: string; tipo: 'door' | 'window';
  s_centro: number; largura: number;
}
export interface LajeFace { ativo: boolean; espessura: number }
export interface Laje {
  contorno: [number, number][];
  piso: LajeFace;
  teto: LajeFace;
}
export interface ModeloPlanta {
  escala: number; single_line: boolean; nome: string;
  bbox: { xmin: number; ymin: number; xmax: number; ymax: number };
  diagnostico: { sobras: number; cantos_costurados: number; blocos_esquadria: number };
  paredes: Parede[];
  aberturas: Abertura[];
  laje: Laje;
}
export interface PlantaConfig {
  altura?: number; esp_laje?: number; porta_altura?: number;
  janela_altura?: number; janela_peitoril?: number;
  cobertura?: boolean; esquadria_detalhada?: boolean;
  pavimento?: string; projeto?: string;
}

// ====================================================================
//  SCAN -> BIM (visualizador de thresholds): upload caro 1x, preview barato
// ====================================================================
export interface ScanUploadResult {
  sid: string; n_pontos: number; extent: number[];
  tabela_thr: { thr: number; grupos: number }[];
  thr_sugerido: number;
  lajes: [number, number][];
  z_hist: { zmin: number; step: number; counts: number[] };
}
export interface ScanLance {
  cx: number; cy: number; ux: number; uy: number;
  comprimento: number; largura: number;
  z0: number; z1: number; declividade: number;
  espelho_cm: number | null; degraus_vistos: number;
}
export interface ScanLajesResult {
  lajes: [number, number][];
  bandas: [number, number][];
}
export interface ScanParedesResult {
  segmentos: [number, number, number, number][];   // percepcao (contornos crus)
  n_segmentos: number;
  eixos: [number, number, number, number, number, string][];  // x1,y1,x2,y2,espessura,label — INSTANCIAS (motor real)
  n_paredes: number;
  png: string;                 // base64 (backdrop de densidade)
  bounds: [number, number, number, number];  // xmin,ymin,xmax,ymax
  contorno_teto: [number, number][];          // footprint detectada (pra editor)
}

// abertura aprovada no editor de finalização, ancorada a um eixo do preview
export interface ScanAbertura {
  eixo_idx: number; tipo: 'door' | 'window';
  s_centro: number; largura: number;
}
export interface ScanJobStatus {
  status: 'rodando' | 'pronto' | 'erro';
  etapa?: string; url?: string | null; erro?: string | null;
  detalhes?: Record<string, string>;   // resultado por etapa (pipeline/telhado/escadas/bake)
}

export const scanUpload = async (file: File): Promise<ScanUploadResult> => {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch(`${API_BASE_URL}/api/scan/upload`, { method: 'POST', body: form });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.error) throw new Error(d.error || `Erro ${r.status}`);
  return d;
};

const postJson = async (path: string, body: any) => {
  const r = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  let d: any = null;
  try { d = await r.json(); }
  catch { throw new Error(`Resposta inválida do servidor em ${path} (JSON malformado)`); }
  if (!r.ok || d.error) throw new Error(d.error || `Erro ${r.status}`);
  return d;
};

export const scanLajes = (sid: string, thr: number): Promise<ScanLajesResult> =>
  postJson('/api/scan/lajes', { sid, thr });

export const scanParedes = (sid: string, zlo: number, zhi: number,
                            zloFrac: number, zhiFrac: number,
                            minLen: number, contoursAll: boolean): Promise<ScanParedesResult> =>
  postJson('/api/scan/paredes', { sid, zlo, zhi, zlo_frac: zloFrac, zhi_frac: zhiFrac,
                                  min_len: minLen, contours_all: contoursAll });

export const scanEscadas = (sid: string, zlo: number, zhi: number,
                            areaMin: number): Promise<{ lances: ScanLance[]; n_lances: number }> =>
  postJson('/api/scan/escadas', { sid, zlo, zhi, area_min: areaMin });

export const scanGerarIfc = (sid: string, thr: number, zloFrac: number,
                             zhiFrac: number, singleMinlen: number,
                             minLen: number, contoursAll: boolean,
                             bandaIdx: number,
                             eixos: ScanParedesResult['eixos'] | null,
                             aberturas?: ScanAbertura[],
                             config?: PlantaConfig): Promise<{ job: string }> =>
  postJson('/api/scan/gerar-ifc', { sid, thr, zlo_frac: zloFrac, zhi_frac: zhiFrac,
                                    single_minlen: singleMinlen,
                                    min_len: minLen, contours_all: contoursAll,
                                    banda_idx: bandaIdx, eixos,
                                    aberturas: aberturas ?? null,
                                    config: config ?? null });

export const scanJob = async (jid: string): Promise<ScanJobStatus> => {
  const r = await fetch(`${API_BASE_URL}/api/scan/job/${jid}`);
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.error) throw new Error(d.error || `Erro ${r.status}`);
  return d;
};

export const parsePlanta = async (file: File, escala?: number,
                                  espDefault = 0.15): Promise<ModeloPlanta> => {
  const params: Record<string, string> = { esp_default: String(espDefault) };
  if (escala) params.escala = String(escala);
  const r = await postTool('/api/planta/parse', file, params);
  return r as unknown as ModeloPlanta;
};

export const gerarPlantaIfc = async (
  modelo: { paredes: Parede[]; aberturas: Abertura[]; laje?: Laje },
  config: PlantaConfig, nome: string,
): Promise<{ ifc_url: string; preview_url: string; n_paredes: number; n_aberturas: number }> => {
  const resp = await fetch(`${API_BASE_URL}/api/planta/gerar`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ modelo, config, nome }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data.error) throw new Error(data.error || `Erro ${resp.status}`);
  return data;
};
