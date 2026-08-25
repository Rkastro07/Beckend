// Serviços do núcleo atual: conversores, edição, Planta-to-BIM e Scan-to-BIM.

import { API_BASE_URL } from './config';

export interface ToolResult {
  ok: boolean;
  download_url: string;      // caminho relativo /outputs/... no backend
  n_pontos?: number;
  [k: string]: any;
}

// resolve a URL de download absoluta a partir do caminho /outputs/... do backend
export const downloadUrl = (rel: string): string => `${API_BASE_URL}${rel}`;

async function postTool(path: string, file: File,
                        params: Record<string, string> = {},
                        extraFiles: Record<string, File> = {}): Promise<ToolResult> {
  const form = new FormData();
  form.append('file', file);
  Object.entries(params).forEach(([k, v]) => form.append(k, v));
  Object.entries(extraFiles).forEach(([k, v]) => form.append(k, v));

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

// ---- Gerador de simulação sintética ----
export interface GerarNuvemResult extends ToolResult {
  labels_url: string;
  rotulo: string;
  stats: {
    completo: number;
    parcial: number;
    ausente: number;
    total_objetos: number;
  };
}

export const gerarNuvemEstagio = (
  file: File,
  estagio: string,
  seed = 0,
) => postTool('/api/tools/gerar-nuvem', file, {
  modo: 'estagio',
  estagio,
  seed: String(seed),
}) as Promise<GerarNuvemResult>;

export const gerarNuvemManual = (
  file: File,
  pctAusente: number,
  pctParcial: number,
  seed = 0,
) => postTool('/api/tools/gerar-nuvem', file, {
  modo: 'manual',
  pct_ausente: String(pctAusente),
  pct_parcial: String(pctParcial),
  seed: String(seed),
}) as Promise<GerarNuvemResult>;

// ====================================================================
//  PLANTA -> BIM: importacao geometrica -> modelo editavel -> IFC
// ====================================================================
export interface Parede {
  id: string; ax: number; ay: number; bx: number; by: number;
  espessura: number; layer: string;
  nome?: string; guid?: string; tipo?: string; ifc_class?: string;
  altura?: number; elevacao?: number; nivel?: string; origem?: string;
  confidence?: number;
  ml_status?: 'wall' | 'door_leaf' | 'non_wall' | 'uncertain';
  ml_probability?: number;
  ml_proposed_keep?: boolean;
  parts?: {
    P1: { selector: string; x: number; y: number };
    P2: { selector: string; x: number; y: number };
    AXIS: { selector: string };
  };
}
export interface Abertura {
  id: string; parede_id: string; tipo: 'door' | 'window';
  s_centro: number; largura: number;
  nome?: string; guid?: string; altura?: number; peitoril?: number; origem?: string;
  confidence?: number; source_layer?: string; block_name?: string;
  source_text?: string; semantic_subtype?: string; semantic_reason?: string;
  declared_width?: number; declared_height?: number;
}
export type CadRole = 'wall' | 'door' | 'window' | 'opening' | 'ignore';
export interface CadLayerDiagnostic {
  name: string;
  entities: number;
  segments: number;
  blocks: number;
  block_names: { name: string; count: number }[];
  entity_types: Record<string, number>;
  detected_role: CadRole | null;
  reason: 'manual' | 'layer-name' | 'block-name' | 'geometry' | 'unclassified' | string;
  confidence: number;
  included: boolean;
  wall_mode?: 'paired' | 'single-line' | null;
  wall_score: {
    confidence: number;
    pairs: number;
    paired_length: number;
    coverage: number;
    orientation: number;
  };
}
export interface LajeFace { ativo: boolean; espessura: number }
export interface Laje {
  contorno: [number, number][];
  piso: LajeFace;
  teto: LajeFace;
}
export interface PlantaVectorReference {
  kind: 'vector';
  label: string;
  bounds: [number, number, number, number];
  segments: [number, number, number, number][];
  source_count?: number;
  truncated?: boolean;
}
export interface Raster2SeqReferenceItem {
  id: string;
  category_id: number;
  label: string;
  points: [number, number][];
  area?: number;
  kind?: 'door' | 'window';
}
export interface RasterDimensionCandidate {
  id: string;
  text: string;
  line_text: string;
  value_m: number;
  confidence: number;
  assumption: string;
  kind: 'linear' | 'thickness' | 'object-size';
  position: { x: number; y: number };
  bbox: { xmin: number; ymin: number; xmax: number; ymax: number };
}
export interface RasterSliceSegment {
  id: string;
  orientation: 'vertical' | 'horizontal';
  points: [[number, number], [number, number]];
  width_px: number;
  support: number;
  strength: number;
}
export interface PlantaRaster2SeqReference {
  kind: 'raster2seq';
  engine?: 'raster2seq' | 'slices-1d' | 'morphology-2d' | 'pre-wall-yolo';
  label: string;
  bounds: [number, number, number, number];
  image_mime: 'image/png';
  image_base64: string;
  canvas_size: [number, number];
  canvas_width_m: number;
  rooms: Raster2SeqReferenceItem[];
  openings: Raster2SeqReferenceItem[];
  dimensions?: RasterDimensionCandidate[];
  slice_segments?: RasterSliceSegment[];
}
export type PlantaReference = PlantaVectorReference | PlantaRaster2SeqReference;
export interface ModeloPlanta {
  escala: number | null; single_line: boolean; nome: string;
  revision?: string;
  endpoint_order?: string;
  bbox: { xmin: number; ymin: number; xmax: number; ymax: number };
  diagnostico: {
    sobras: number; cantos_costurados: number; blocos_esquadria: number;
    elementos_lidos?: number; geometrias_aproximadas?: number;
  };
  source?: {
    format: string; family: string; mode: string; semantic_level: string;
    grammar_version?: string;
    scale_source?: string;
    pavimento?: { id: string; nome: string; elevacao: number } | null;
    pavimentos_disponiveis?: {
      id: string; nome: string; elevacao: number; n_paredes: number;
    }[];
    cad_layers?: CadLayerDiagnostic[];
    cad_summary?: {
      entities: number; layers: number; wall_layers: number;
      inferred_wall_layers: number; semantic_opening_candidates: number;
      grammar_opening_candidates?: number; semantic_text_cues?: number;
      linked_images?: number; missing_linked_images?: number;
      ocr_images?: number; ocr_lines?: number; ocr_semantic_cues?: number;
      ocr_failures?: number;
      gap_opening_candidates: number; ignored_entities: number;
      units_code: number; scale_source: string;
      regions?: number;
    };
    cad_region?: { id: string; name: string } | null;
    cad_regions?: {
      id: string; name: string; n_walls: number; total_length: number;
      bbox: { xmin: number; ymin: number; xmax: number; ymax: number };
      selected: boolean;
    }[];
    layer_map?: Record<string, CadRole>;
    cad_semantic_cues?: {
      text: string; role: string; status: string; reason?: string;
      subtype?: string; source_layer?: string; segment_length?: number;
      distance?: number; declared_width?: number; declared_height?: number;
    }[];
    cad_linked_images?: {
      layer: string; stored_path: string; resolved_path: string;
      available: boolean; handle: string; resolution_source?: string;
      insert_raw?: [number, number]; u_pixel_raw?: [number, number];
      v_pixel_raw?: [number, number]; image_size?: [number, number];
    }[];
    cad_raster_ocr?: {
      status: string; engine: string; language?: string;
      image_handle?: string; image_path?: string;
      resolution_source?: string; line_count: number;
      semantic_cue_count: number; error?: string;
      lines: {
        text: string; normalized_text: string;
        cad_position: { x: number; y: number };
        cad_bbox: { xmin: number; ymin: number; xmax: number; ymax: number };
      }[];
    }[];
  } | null;
  reference?: PlantaReference | null;
  warnings?: string[];
  paredes: Parede[];
  aberturas: Abertura[];
  laje: Laje;
  spaces?: { id: string; contorno: [number, number][]; area: number; perimetro: number }[];
  raster_2d?: {
    source_wall_segment_count: number;
    canonical_wall_count: number;
    wall_segments_absorbed: number;
    wall_gap_count: number;
    classified_wall_gaps: number;
    unclassified_wall_gaps: number;
    unmatched_openings: number;
    slab_method?: string;
    slab_area_m2?: number;
    wall_gaps?: {
      host_axis_index: number;
      orientation: 'horizontal' | 'vertical';
      fixed: number;
      start: number;
      end: number;
      classification: 'door' | 'window' | 'unknown';
      confidence: number;
    }[];
  };
  pre_wall_job?: string;
  pre_wall_preview_url?: string;
  pre_wall_candidates_url?: string;
  pre_wall?: {
    raw_detection_count?: number;
    consensus_candidate_count?: number;
    wall_geometry_source?: string;
    opening_count?: number;
    door_count?: number;
    window_count?: number;
  };
}
export interface PlantaConfig {
  altura?: number; esp_laje?: number; porta_altura?: number;
  janela_altura?: number; janela_peitoril?: number;
  cobertura?: boolean; esquadria_detalhada?: boolean;
  pavimento?: string; projeto?: string;
  forro?: { ativo: boolean; altura?: number; espessura?: number };
  finalizacao_automatica?: boolean;
}

export interface GptPlanCalibration {
  message: string;
  needs_rectification: boolean;
  applied: boolean;
  source_quad_px: [number, number][];
  main_width_m: number | null;
  main_height_m: number | null;
  right_extra_m: number;
  confidence: number;
  dimensions: {
    text: string;
    value_m: number;
    axis: 'horizontal' | 'vertical' | 'unknown';
    confidence: number;
    reason: string;
  }[];
  assumptions: string[];
}

export interface GptPlanReview {
  message: string;
  changed: boolean;
  confidence: number;
  observations: string[];
  assumptions: string[];
  unresolved: string[];
}

export interface GptPlanResult {
  ok: boolean;
  job?: string;
  assistant: string;
  changed?: boolean;
  model: ModeloPlanta;
  calibration?: GptPlanCalibration;
  review: GptPlanReview;
  rectification?: Record<string, unknown> | null;
  api?: Record<string, unknown>;
}

export type PlanAiProvider = 'openai' | 'deepseek';

export interface PlanAiProviderStatus {
  configured: boolean;
  label: string;
  model: string;
  vision_model?: string;
  reasoning_model?: string;
  workflow: string;
}

export interface PlanAiStatus {
  ok: boolean;
  configured: boolean;
  provider: PlanAiProvider;
  default_provider: PlanAiProvider;
  model: string;
  vision_model: string;
  reasoning_model: string;
  workflow: string;
  stores_responses: boolean;
  providers: Record<PlanAiProvider, PlanAiProviderStatus>;
}

export interface BimRevisionSpec {
  schema?: 'bim.edit-operations.v1';
  base_revision?: string;
  revision?: string;
  operations: Record<string, any>[];
  recalculate?: ('openings' | 'topology' | 'spaces' | 'slabs' | 'validation')[];
  policies?: Record<string, any>;
  render?: { selected?: string[] };
}

export interface BimRevisionResult {
  model: ModeloPlanta & {
    revision: string;
    spaces?: { id: string; contorno: [number, number][]; area: number; perimetro: number }[];
  };
  report: {
    revision: string;
    warnings: string[];
    recalculated?: string[];
    operation_results?: Array<{
      op: string;
      joined?: Array<{ walls: string[]; kind: string; point: number[] }>;
      moved_endpoints?: Record<string, unknown>;
      blocked_by_openings?: unknown[];
    }>;
    validation: { valid: boolean; errors: string[]; warnings: string[] };
  };
  parts: {
    revision: string;
    walls: Record<string, {
      P1: { selector: string; x: number; y: number };
      P2: { selector: string; x: number; y: number };
      AXIS: { selector: string };
    }>;
  };
}

export const aplicarRevisaoBim = async (
  model: ModeloPlanta,
  revision: BimRevisionSpec,
): Promise<BimRevisionResult> => {
  const response = await fetch(`${API_BASE_URL}/api/bim-editing/apply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, revision }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.error) throw new Error(data.error || `Erro ${response.status}`);
  return data as BimRevisionResult;
};

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
  // trechos classificados pela assinatura vertical (multi-fatia): x1,y1,x2,y2,classe
  // x1,y1,x2,y2,classe,eixo_idx: o indice ancora portas/janelas no IfcWall.
  classificacao?: [number, number, number, number, string, number][];
  png: string;                 // base64 (backdrop de densidade)
  bounds: [number, number, number, number];  // xmin,ymin,xmax,ymax
  contorno_teto: [number, number][];          // footprint detectada (pra editor)
}

// 3 faixas de altura (fracao do pe-direito) que dirigem a classificacao multi-fatia
export interface ScanFatias {
  baixa: [number, number]; media: [number, number]; alta: [number, number];
}

// abertura aprovada no editor de finalização, ancorada a um eixo do preview
export interface ScanAbertura {
  eixo_idx: number; tipo: 'door' | 'window';
  s_centro: number; largura: number;
  altura?: number; peitoril?: number;
}
export interface ScanJobStatus {
  status: 'rodando' | 'pronto' | 'erro';
  etapa?: string; url?: string | null; erro?: string | null;
  detalhes?: Record<string, string>;   // resultado por etapa (pipeline/telhado/escadas/bake)
  hybrid?: ScanHybridResult;
}

export interface ScanHybridWall {
  id: string;
  ax: number; ay: number; bx: number; by: number; espessura: number;
  ml_class: 'wall' | 'door_leaf' | 'non_wall' | 'uncertain';
  ml_predicted_class: 'wall' | 'door_leaf' | 'non_wall';
  ml_probability: number;
  wall_probability: number;
  proposed_keep: boolean;
}

export interface ScanHybridOpening {
  id: string;
  wall_id: string;
  class: 'door' | 'window';
  s_center: number;
  width: number;
  height: number;
  sill: number;
  confidence: number;
}

export interface ScanHybridResult {
  png_url: string;
  predictions_url: string;
  model_url: string;
  bounds: [number, number, number, number];
  floor_z: number;
  ceiling_z: number;
  elapsed_seconds: number;
  automatic_geometry_change: false;
  counts: {
    input_walls: number;
    proposed_keep: number;
    proposed_remove: number;
    wall: number;
    door_leaf: number;
    non_wall: number;
    doors: number;
    windows: number;
    openings: number;
  };
  walls: ScanHybridWall[];
  openings: ScanHybridOpening[];
}

export interface ScanCloudPreviewResult {
  positions: number[];        // Three.js: x, altura, y da nuvem original
  count: number;
  source_count: number;
  coordinate_order: 'x,z,y';
  vertical_base: number;      // cota original que foi convertida em Y=0
  normalized_to_ground: boolean;
  discarded_below_base?: number;
}

export interface ScanIfcRecoveryResult {
  modelo: ModeloPlanta;
  config: PlantaConfig;
  warnings: string[];
  counts: {
    walls: number;
    openings: number;
    spaces: number;
    slabs: number;
  };
}

export const scanUpload = async (file: File): Promise<ScanUploadResult> => {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch(`${API_BASE_URL}/api/scan/upload`, { method: 'POST', body: form });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.error) throw new Error(d.error || `Erro ${r.status}`);
  return d;
};

export const scanRecoverIfc = async (file: File): Promise<ScanIfcRecoveryResult> => {
  const form = new FormData();
  form.append('file', file);
  form.append('force_ceiling', 'true');
  const response = await fetch(`${API_BASE_URL}/api/scan/recover-ifc`, {
    method: 'POST',
    body: form,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.error) {
    throw new Error(data.error || `Erro ${response.status}`);
  }
  return data as ScanIfcRecoveryResult;
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

export const scanCloudPreview = (
  sid: string,
  maxPoints = 120_000,
  baseElevation?: number,
): Promise<ScanCloudPreviewResult> =>
  postJson('/api/scan/cloud-preview', {
    sid,
    max_points: maxPoints,
    ...(Number.isFinite(baseElevation) ? { base_elevation: baseElevation } : {}),
  });

export const scanParedes = (sid: string, zlo: number, zhi: number,
                            zloFrac: number, zhiFrac: number,
                            minLen: number, contoursAll: boolean,
                            fatias?: ScanFatias,
                            singleMinlen = 1.5): Promise<ScanParedesResult> =>
  postJson('/api/scan/paredes', { sid, zlo, zhi, zlo_frac: zloFrac, zhi_frac: zhiFrac,
                                  min_len: minLen, contours_all: contoursAll,
                                  fatias: fatias ?? null,
                                  single_minlen: singleMinlen });

export const scanEscadas = (sid: string, zlo: number, zhi: number,
                            areaMin: number): Promise<{ lances: ScanLance[]; n_lances: number }> =>
  postJson('/api/scan/escadas', { sid, zlo, zhi, area_min: areaMin });

export const scanHibrido = (
  sid: string,
  floorZ: number,
  ceilingZ: number,
): Promise<{ job: string }> =>
  postJson('/api/scan/hibrido', {
    sid,
    floor_z: floorZ,
    ceiling_z: ceilingZ,
  });

export const scanGerarIfc = (sid: string, thr: number, zloFrac: number,
                             zhiFrac: number, singleMinlen: number,
                             minLen: number, contoursAll: boolean,
                             stairAreaMin: number,
                             bandaIdx: number,
                             eixos: ScanParedesResult['eixos'] | null,
                             aberturas?: ScanAbertura[],
                             config?: PlantaConfig,
                             modelo?: Pick<ModeloPlanta, 'paredes' | 'aberturas' | 'laje' | 'spaces'>,
): Promise<{ job: string }> =>
  postJson('/api/scan/gerar-ifc', { sid, thr, zlo_frac: zloFrac, zhi_frac: zhiFrac,
                                    single_minlen: singleMinlen,
                                    min_len: minLen, contours_all: contoursAll,
                                    stair_area_min: stairAreaMin,
                                    banda_idx: bandaIdx, eixos,
                                    aberturas: aberturas ?? null,
                                    config: config ?? null,
                                    modelo: modelo ?? null });

export const scanJob = async (jid: string): Promise<ScanJobStatus> => {
  const r = await fetch(`${API_BASE_URL}/api/scan/job/${jid}`);
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.error) throw new Error(d.error || `Erro ${r.status}`);
  return d;
};

export const parsePlanta = async (file: File, escala?: number,
                                  espDefault = 0.15,
                                  pavimento?: string,
                                  pdfScale = 50,
                                  pagina = 0,
                                  cadLayerMap: Record<string, CadRole> = {},
                                  cadRegion?: string,
                                  linkedImage?: File): Promise<ModeloPlanta> => {
  const params: Record<string, string> = { esp_default: String(espDefault) };
  if (escala) params.escala = String(escala);
  if (pavimento) params.pavimento = pavimento;
  params.pdf_scale = String(pdfScale);
  params.pagina = String(pagina);
  if (Object.keys(cadLayerMap).length) {
    params.layer_map = JSON.stringify(cadLayerMap);
  }
  if (cadRegion) params.cad_region = cadRegion;
  const extraFiles: Record<string, File> = {};
  if (linkedImage) extraFiles.linked_image = linkedImage;
  const r = await postTool(
    '/api/referencia/importar',
    file,
    params,
    extraFiles,
  );
  return r as unknown as ModeloPlanta;
};

export const parsePlantaRaster2Seq = async (
  file: File,
  canvasWidthM = 20,
): Promise<ModeloPlanta> => {
  const result = await postTool('/api/referencia/raster2seq', file, {
    canvas_width_m: String(canvasWidthM),
  });
  return result as unknown as ModeloPlanta;
};

export const parsePlantaRasterSlices = async (
  file: File,
  canvasWidthM = 20,
): Promise<ModeloPlanta> => {
  const result = await postTool('/api/referencia/raster-slices', file, {
    canvas_width_m: String(canvasWidthM),
  });
  return result as unknown as ModeloPlanta;
};

export const parsePlantaRaster2D = async (
  file: File,
  canvasWidthM = 20,
): Promise<ModeloPlanta> => {
  const result = await postTool('/api/referencia/raster-2d', file, {
    canvas_width_m: String(canvasWidthM),
  });
  return result as unknown as ModeloPlanta;
};

export const parsePlantaPreWallYolo = async (
  file: File,
  canvasWidthM = 20,
  metricRefinement = true,
): Promise<ModeloPlanta> => {
  const result = await postTool('/api/referencia/pre-wall-yolo', file, {
    canvas_width_m: String(canvasWidthM),
    metric_refinement: String(metricRefinement),
  });
  return result as unknown as ModeloPlanta;
};

export const startGptPlan = async (
  file: File,
  canvasWidthM = 20,
  message = 'Calibre pelas cotas, combine a análise 2D com sua visão e proponha o modelo BIM.',
  provider: PlanAiProvider = 'deepseek',
): Promise<GptPlanResult> => {
  const form = new FormData();
  form.append('file', file);
  form.append('canvas_width_m', String(canvasWidthM));
  form.append('message', message);
  form.append('provider', provider);
  const response = await fetch(`${API_BASE_URL}/api/referencia/gpt-plan/start`, {
    method: 'POST',
    body: form,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.error) throw new Error(data.error || `Erro ${response.status}`);
  return data as GptPlanResult;
};

export const getPlanAiStatus = async (): Promise<PlanAiStatus> => {
  const response = await fetch(`${API_BASE_URL}/api/referencia/gpt-plan/status`);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.error) throw new Error(data.error || `Erro ${response.status}`);
  return data as PlanAiStatus;
};

export const chatGptPlan = async (
  model: ModeloPlanta,
  message: string,
  provider: PlanAiProvider = 'deepseek',
): Promise<GptPlanResult> => {
  const response = await fetch(`${API_BASE_URL}/api/referencia/gpt-plan/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, message, provider }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.error) throw new Error(data.error || `Erro ${response.status}`);
  return data as GptPlanResult;
};

export const gerarPlantaIfc = async (
  modelo: {
    paredes: Parede[];
    aberturas: Abertura[];
    laje?: Laje;
    spaces?: ModeloPlanta['spaces'];
  },
  config: PlantaConfig, nome: string,
  aprovacaoCliente?: { confirmado: boolean; confirmado_em?: string },
): Promise<{
  ifc_url: string;
  preview_url: string;
  ifc_token: string | null;
  pavimentos: string[];
  ready_for_comparison: boolean;
  n_paredes: number;
  n_aberturas: number;
}> => {
  const resp = await fetch(`${API_BASE_URL}/api/referencia/finalizar`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      modelo,
      config,
      nome,
      exigir_aprovacao_cliente: true,
      aprovacao_cliente: aprovacaoCliente,
    }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data.error) throw new Error(data.error || `Erro ${resp.status}`);
  return data;
};

export interface ResultadoDxf {
  dxf_url: string;
  units: 'meters';
  walls: number;
  doors: number;
  windows: number;
  spaces: number;
}

export const gerarPlantaDxf = async (
  modelo: ModeloPlanta,
  nome: string,
  aprovacaoCliente: { confirmado: boolean; confirmado_em?: string },
): Promise<ResultadoDxf> => {
  const resp = await fetch(`${API_BASE_URL}/api/referencia/exportar-dxf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      modelo,
      nome,
      aprovacao_cliente: aprovacaoCliente,
    }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data.error) throw new Error(data.error || `Erro ${resp.status}`);
  return data as ResultadoDxf;
};
