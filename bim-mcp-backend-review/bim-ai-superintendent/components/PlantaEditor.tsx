import React, { useRef, useState, useEffect, useMemo } from 'react';
import {
  FileUp, Loader2, Download, Trash2, Plus, DoorOpen, RectangleHorizontal,
  Sparkles, AlertCircle, CheckCircle2, RotateCcw, Move, Square, Layers, Maximize,
  ScanLine, ImagePlus, Copy, Bot, SendHorizontal, X, Ruler,
} from 'lucide-react';
import {
  parsePlanta, parsePlantaRaster2Seq, parsePlantaRasterSlices, parsePlantaRaster2D,
  parsePlantaPreWallYolo,
  gerarPlantaIfc, gerarPlantaDxf, aplicarRevisaoBim, startGptPlan, chatGptPlan,
  getPlanAiStatus,
  downloadUrl, scanRecoverIfc,
  ModeloPlanta, Parede, Abertura, PlantaConfig, Laje, LajeFace, ScanCloudPreviewResult,
  CadRole, CadLayerDiagnostic, GptPlanResult, PlanAiStatus, PlanAiProvider,
} from '../services/tools';
import { PlantaEditor3D, PlantaEditorSelection } from './PlantaEditor3D';
import {
  clampOpeningCenter,
  captureOpeningWorldAnchors,
  duplicateOpeningCenter,
  MIN_WALL_LENGTH,
  OpeningWorldAnchor,
  openingsOutsideWall,
  rasterRoomsToWallSegments,
  remapOpeningsToWall,
  scaleRasterModel,
} from './plantaEditorGeometry';

type Sel = PlantaEditorSelection;
type EditorViewMode = '2d' | '3d' | 'split';
type Drag =
  | { kind: 'wall-a' | 'wall-b' | 'wall-body'; id: string; ox: number; oy: number; ax: number; ay: number; bx: number; by: number }
  | { kind: 'opening'; id: string; ps: number }
  | { kind: 'laje-vertice'; idx: number }
  | { kind: 'raster-room-vertex'; id: string; idx: number }
  | { kind: 'pan'; px: number; py: number; vx: number; vy: number }
  | null;
type View = { x: number; y: number; w: number; h: number };
type WallEndpointEdit = {
  wallId: string;
  beforeModel: ModeloPlanta;
  anchors: OpeningWorldAnchor[];
};
type PendingOpeningRemoval = {
  wallId: string;
  beforeModel: ModeloPlanta;
  openings: Abertura[];
};
type FinalizationSummary = {
  revision: string;
  junctions: number;
  movedEndpoints: number;
  blockedByOpenings: number;
  spaces: number;
  slabAdjusted: boolean;
};
type PlanChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  text: string;
};
type MeasurementPoint = {
  x: number;
  y: number;
  snapLabel?: string;
};

const COR_PAREDE = '#e08e34';
const COR_PAREDE_ESTRUTURAL = '#b45309';
const COR_PILAR = '#8b5cf6';
const COR_ML_WALL = '#22b52a';
const COR_ML_LEAF = '#f59e0b';
const COR_ML_NON_WALL = '#ef2b2b';
const COR_ML_UNCERTAIN = '#9ca3af';
const COR_PORTA = '#2ecc71';
const COR_JANELA = '#5db4e2';
const COR_SEL = '#2563eb';
const COR_P1 = '#16a34a';
const COR_P2 = '#7c3aed';
// Tolerancia apenas do grafo de ambientes; nao move a geometria das paredes.
// Mantem editor e exportador no mesmo contrato do SPACE_MAX_SNAP do Cloud2BIM.
const SPACE_TOPOLOGY_TOLERANCE = 0.15;

const isColumnWall = (wall: Parede): boolean =>
  wall.tipo === 'column' || wall.ifc_class === 'IfcColumn';
const isStructuralWall = (wall: Parede): boolean =>
  !isColumnWall(wall) && wall.tipo === 'structural-wall';

const polygonArea = (points: [number, number][]): number => Math.abs(points.reduce(
  (sum, [x, y], index) => {
    const [nextX, nextY] = points[(index + 1) % points.length];
    return sum + x * nextY - nextX * y;
  },
  0,
)) / 2;

const modelFingerprint = (model: ModeloPlanta, cfg: PlantaConfig): string => JSON.stringify({
  paredes: model.paredes,
  aberturas: model.aberturas,
  laje: model.laje,
  spaces: model.spaces ?? [],
  cfg,
});

const proximoId = (ids: string[], prefixo: string): string => {
  const usados = new Set(ids);
  let numero = 1;
  while (usados.has(`${prefixo}-${String(numero).padStart(3, '0')}`)) numero += 1;
  return `${prefixo}-${String(numero).padStart(3, '0')}`;
};

export type ResultadoGeracao = {
  ifc_url: string;
  preview_url?: string;
  ifc_token?: string | null;
  pavimentos?: string[];
  ready_for_comparison?: boolean;
};

export interface PlantaEditorProps {
  /** Modelo inicial (ex.: vindo do Scan → BIM). Se ausente, o editor pede um DXF. */
  modeloInicial?: ModeloPlanta | null;
  /** Arquivo enviado pela entrada unificada do painel OBB. */
  arquivoInicial?: File | null;
  /** Gerador de IFC customizado; se ausente usa o gerarPlantaIfc (plantatobim/DXF). */
  onGerar?: (
    modelo: {
      paredes: Parede[];
      aberturas: Abertura[];
      laje: Laje;
      spaces?: ModeloPlanta['spaces'];
    },
    cfg: PlantaConfig,
    nome: string,
  ) => Promise<ResultadoGeracao>;
  titulo?: string;
  subtitulo?: string;
  /** O caminho do scan não produz PLY de preview do plantatobim. */
  ocultarPreviewPly?: boolean;
  /** Voltar pra etapa anterior (ex.: fase de paredes do scan) em vez de trocar planta. */
  onVoltar?: () => void;
  rotuloVoltar?: string;
  /** Mapa de densidade da nuvem (PNG base64) desenhado por baixo — o cliente
   *  edita EM CIMA do que o scanner captou. */
  backdropPng?: string;
  /** [xmin, ymin, xmax, ymax] do backdrop, em coords de mundo. */
  backdropBounds?: [number, number, number, number];
  /** Abre o comparador somente quando o usuário escolher comparar o IFC gerado. */
  onReferenciaPronta?: (resultado: ResultadoGeracao) => void;
  /** Carrega uma amostra leve da nuvem apenas quando a camada 3D for ligada. */
  loadPointCloud?: () => Promise<ScanCloudPreviewResult>;
}

export const PlantaEditor: React.FC<PlantaEditorProps> = ({
  modeloInicial, arquivoInicial, onGerar, titulo, subtitulo, ocultarPreviewPly,
  onVoltar, rotuloVoltar, backdropPng, backdropBounds, onReferenciaPronta,
  loadPointCloud,
} = {}) => {
  const initialModelHeight =
    modeloInicial?.paredes.find((parede) => parede.altura != null)?.altura ?? 2.8;
  const [modelo, setModelo] = useState<ModeloPlanta | null>(modeloInicial ?? null);
  const [arquivoFonte, setArquivoFonte] = useState<File | null>(null);
  const [imagemVinculada, setImagemVinculada] = useState<File | null>(null);
  const [sel, setSel] = useState<Sel>(null);
  const [modoLaje, setModoLaje] = useState(false);
  const [mostrarBackdrop, setMostrarBackdrop] = useState(true);
  const [mostrarAiOverlay, setMostrarAiOverlay] = useState(true);
  const [selectedAiRoomId, setSelectedAiRoomId] = useState<string | null>(null);
  const [selectedAiVertex, setSelectedAiVertex] = useState<number | null>(null);
  const [selectedDimensionId, setSelectedDimensionId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<EditorViewMode>('2d');
  const [measureMode, setMeasureMode] = useState(false);
  const [measurementStart, setMeasurementStart] = useState<MeasurementPoint | null>(null);
  const [measurementEnd, setMeasurementEnd] = useState<MeasurementPoint | null>(null);
  const [measurementCursor, setMeasurementCursor] = useState<MeasurementPoint | null>(null);
  const [selVertice, setSelVertice] = useState<number | null>(null);
  const [view, setView] = useState<View | null>(null);
  const [busy, setBusy] = useState(false);
  const [erro, setErro] = useState<string | null>(null);
  const [avisoRecuperacao, setAvisoRecuperacao] = useState<string | null>(null);
  const [resultado, setResultado] = useState<ResultadoGeracao | null>(null);
  const [finalizedFingerprint, setFinalizedFingerprint] = useState<string | null>(null);
  const [finalizationSummary, setFinalizationSummary] = useState<FinalizationSummary | null>(null);
  const [clientApprovalChecked, setClientApprovalChecked] = useState(false);
  const [clientApprovedFingerprint, setClientApprovedFingerprint] = useState<string | null>(null);
  const [clientApprovedAt, setClientApprovedAt] = useState<string | null>(null);
  const [slabEditedByUser, setSlabEditedByUser] = useState(false);
  const [pdfScale, setPdfScale] = useState(50);
  const [rasterCanvasWidth, setRasterCanvasWidth] = useState(20);
  const [preWallMetricRefinement, setPreWallMetricRefinement] = useState(true);
  const [cadLayerMap, setCadLayerMap] = useState<Record<string, CadRole>>({});
  const [planChatDraft, setPlanChatDraft] = useState(
    'Calibre pelas cotas impressas, combine o detector 2D com sua visão e revise paredes, portas, janelas e laje.',
  );
  const [planChatBusy, setPlanChatBusy] = useState(false);
  const [planChatCandidate, setPlanChatCandidate] = useState<ModeloPlanta | null>(null);
  const [planAiStatus, setPlanAiStatus] = useState<PlanAiStatus | null>(null);
  const [planAiProvider, setPlanAiProvider] = useState<PlanAiProvider>('deepseek');
  const [planChatMessages, setPlanChatMessages] = useState<PlanChatMessage[]>([{
    id: 'assistant-welcome',
    role: 'assistant',
    text: 'Envie uma planta raster. Vou ler as cotas, corrigir a perspectiva, rodar o 2D e revisar a geometria antes de propor alterações.',
  }]);
  const [cfg, setCfg] = useState<PlantaConfig>({
    altura: initialModelHeight,
    porta_altura: 2.1, janela_altura: 1.2, janela_peitoril: 1.0,
    esquadria_detalhada: false, cobertura: true,
    forro: { ativo: false, altura: Math.max(0.1, initialModelHeight - 0.1), espessura: 0.03 },
  });

  const svgRef = useRef<SVGSVGElement | null>(null);
  const recoveryInputRef = useRef<HTMLInputElement | null>(null);
  const dragRef = useRef<Drag>(null);
  const modelRef = useRef<ModeloPlanta | null>(modelo);
  const wallEndpointEditRef = useRef<WallEndpointEdit | null>(null);
  const initialFileKeyRef = useRef<string | null>(null);
  const [pendingOpeningRemoval, setPendingOpeningRemoval] =
    useState<PendingOpeningRemoval | null>(null);
  modelRef.current = modelo;

  const currentModelFingerprint = useMemo(
    () => modelo ? modelFingerprint(modelo, cfg) : null,
    [modelo, cfg],
  );

  useEffect(() => {
    setClientApprovalChecked(false);
  }, [currentModelFingerprint]);

  useEffect(() => {
    let active = true;
    void getPlanAiStatus()
      .then((status) => {
        if (!active) return;
        setPlanAiStatus(status);
        setPlanAiProvider((current) => (
          status.providers?.[current]?.configured
            ? current
            : status.default_provider
        ));
      })
      .catch(() => { if (active) setPlanAiStatus(null); });
    return () => { active = false; };
  }, []);

  const changePlanAiProvider = (provider: PlanAiProvider) => {
    if (planChatBusy) return;
    setPlanAiProvider(provider);
    setPlanChatCandidate(null);
    setErro(null);
  };

  const paredeById = useMemo(() => {
    const m: Record<string, Parede> = {};
    modelo?.paredes.forEach((p) => (m[p.id] = p));
    return m;
  }, [modelo]);
  const vectorReference = modelo?.reference?.kind === 'vector'
    ? modelo.reference
    : null;
  const rasterReference = modelo?.reference?.kind === 'raster2seq'
    ? modelo.reference
    : null;
  const isSliceReference = rasterReference?.engine === 'slices-1d';
  const isPreWallReference = rasterReference?.engine === 'pre-wall-yolo';
  const is2DReference = rasterReference?.engine === 'morphology-2d' || isPreWallReference;
  const isGeometricRasterReference = isSliceReference || is2DReference;
  const selectedAiRoom = rasterReference?.rooms.find((room) => room.id === selectedAiRoomId)
    ?? null;
  const dimensionCandidates = rasterReference?.dimensions?.filter(
    (dimension) => dimension.confidence >= 0.5 && dimension.kind === 'linear',
  ) ?? [];
  const selectedDimension = dimensionCandidates.find(
    (dimension) => dimension.id === selectedDimensionId,
  ) ?? null;
  const selectedWallForDimension = sel?.kind === 'parede' ? paredeById[sel.id] : null;
  const selectedWallIsColumn = selectedWallForDimension
    ? isColumnWall(selectedWallForDimension)
    : false;
  const selectedWallLength = selectedWallForDimension
    ? Math.hypot(
        selectedWallForDimension.bx - selectedWallForDimension.ax,
        selectedWallForDimension.by - selectedWallForDimension.ay,
      )
    : null;
  const selectedDimensionRatio = selectedDimension && selectedWallLength
    ? selectedDimension.value_m / selectedWallLength
    : null;
  const hasRasterWalls = modelo?.paredes.some(
    (wall) => wall.origem === 'raster2seq-contour'
      || wall.origem === 'raster-slices-1d'
      || wall.origem === 'raster-2d-morphology'
      || wall.origem === 'pre-wall-yolo',
  ) ?? false;
  const vectorReferencePath = useMemo(() => (
    vectorReference?.segments.map(
      ([x1, y1, x2, y2]) => `M${x1},${-y1}L${x2},${-y2}`,
    ).join(' ') ?? ''
  ), [vectorReference]);

  const appendPlanChatMessage = (role: PlanChatMessage['role'], text: string) => {
    setPlanChatMessages((current) => [
      ...current,
      { id: `${role}-${Date.now()}-${current.length}`, role, text },
    ]);
  };

  const summarizeGptResult = (result: GptPlanResult): string => {
    const calibration = result.calibration;
    const scale = calibration
      ? calibration.applied
        ? ' Retificação aplicada pelas cotas horizontal e vertical validadas.'
        : ' A escala permaneceu provisória porque faltou quadrilátero válido ou cotas em duas direções.'
      : '';
    const unresolved = result.review?.unresolved?.length
      ? ` Pendências: ${result.review.unresolved.join('; ')}.`
      : '';
    return `${result.assistant}${scale}${unresolved}`;
  };

  const onGptPlanFile = async (file: File) => {
    const prompt = planChatDraft.trim() || 'Analise e revise esta planta.';
    setArquivoFonte(file);
    setPlanChatBusy(true);
    setErro(null);
    setResultado(null);
    setPlanChatCandidate(null);
    appendPlanChatMessage('user', `${prompt}\nArquivo: ${file.name}`);
    try {
      const result = await startGptPlan(file, rasterCanvasWidth, prompt, planAiProvider);
      setModelo(result.model);
      modelRef.current = result.model;
      setMostrarBackdrop(true);
      setMostrarAiOverlay(false);
      setSel(result.model.paredes.length ? { kind: 'parede', id: result.model.paredes[0].id } : null);
      setSelectedAiRoomId(null);
      setSelectedAiVertex(null);
      setSelectedDimensionId(null);
      setFinalizedFingerprint(null);
      setFinalizationSummary(null);
      setClientApprovedFingerprint(null);
      setClientApprovedAt(null);
      setSlabEditedByUser(false);
      fitView(result.model);
      appendPlanChatMessage('assistant', summarizeGptResult(result));
    } catch (e: any) {
      const message = e.message || 'Falha na análise visual Plan-to-BIM';
      setErro(message);
      appendPlanChatMessage('assistant', `Não consegui concluir: ${message}`);
    } finally {
      setPlanChatBusy(false);
    }
  };

  const sendGptPlanMessage = async () => {
    const current = modelRef.current;
    const prompt = planChatDraft.trim();
    if (!current || !prompt || planChatBusy) return;
    setPlanChatBusy(true);
    setErro(null);
    setPlanChatCandidate(null);
    appendPlanChatMessage('user', prompt);
    setPlanChatDraft('');
    try {
      const result = await chatGptPlan(current, prompt, planAiProvider);
      appendPlanChatMessage('assistant', summarizeGptResult(result));
      if (result.changed) setPlanChatCandidate(result.model);
    } catch (e: any) {
      const message = e.message || 'Falha no chat visual Plan-to-BIM';
      setErro(message);
      appendPlanChatMessage('assistant', `Não consegui concluir: ${message}`);
    } finally {
      setPlanChatBusy(false);
    }
  };

  const applyGptPlanCandidate = () => {
    if (!planChatCandidate) return;
    setModelo(planChatCandidate);
    modelRef.current = planChatCandidate;
    setPlanChatCandidate(null);
    setSel(null);
    setSelVertice(null);
    setFinalizedFingerprint(null);
    setFinalizationSummary(null);
    setClientApprovedFingerprint(null);
    setClientApprovedAt(null);
    setSlabEditedByUser(false);
    fitView(planChatCandidate);
    appendPlanChatMessage('assistant', 'Revisão aplicada ao editor. O IFC ainda depende da sua finalização e aprovação.');
  };

  useEffect(() => {
    if (!sel) return;
    setSelectedAiRoomId(null);
    setSelectedAiVertex(null);
  }, [sel]);

  // ---------- upload / parse ----------
  const fitView = (m: ModeloPlanta) => {
    const referenceBounds = m.reference?.bounds;
    const xmin = Math.min(m.bbox.xmin, referenceBounds?.[0] ?? m.bbox.xmin);
    const ymin = Math.min(m.bbox.ymin, referenceBounds?.[1] ?? m.bbox.ymin);
    const xmax = Math.max(m.bbox.xmax, referenceBounds?.[2] ?? m.bbox.xmax);
    const ymax = Math.max(m.bbox.ymax, referenceBounds?.[3] ?? m.bbox.ymax);
    const pad = Math.max(xmax - xmin, ymax - ymin) * 0.08 + 0.5;
    setView({ x: xmin - pad, y: -(ymax + pad), w: xmax - xmin + 2 * pad, h: ymax - ymin + 2 * pad });
  };

  // Modelo injetado de fora (Scan → BIM): adota e enquadra quando muda.
  useEffect(() => {
    if (modeloInicial) {
      setModelo(modeloInicial);
      const modelHeight =
        modeloInicial.paredes.find((parede) => parede.altura != null)?.altura ?? 2.8;
      setCfg((current) => {
        const thickness = current.forro?.espessura ?? 0.03;
        const maxForroHeight = Math.max(0.1, modelHeight - thickness);
        return {
          ...current,
          altura: modelHeight,
          forro: {
            ativo: current.forro?.ativo ?? false,
            espessura: thickness,
            altura: Math.min(current.forro?.altura ?? maxForroHeight, maxForroHeight),
          },
        };
      });
      setSel(null); setSelVertice(null); setResultado(null);
      setSelectedAiRoomId(null); setSelectedAiVertex(null);
      setSelectedDimensionId(null);
      setPendingOpeningRemoval(null);
      setSlabEditedByUser(false);
      wallEndpointEditRef.current = null;
      fitView(modeloInicial);
    }
  }, [modeloInicial]);

  const carregarArquivo = async (
    file: File,
    pavimento?: string,
    layerMap: Record<string, CadRole> = cadLayerMap,
    cadRegion?: string,
    linkedImage: File | null = imagemVinculada,
  ) => {
    setBusy(true); setErro(null); setResultado(null); setSel(null); setSelVertice(null);
    setSlabEditedByUser(false);
    setSelectedAiRoomId(null); setSelectedAiVertex(null);
    setSelectedDimensionId(null);
    try {
      const m = await parsePlanta(
        file,
        undefined,
        0.15,
        pavimento,
        pdfScale,
        0,
        layerMap,
        cadRegion,
        linkedImage ?? undefined,
      );
      setModelo(m); fitView(m);
    } catch (e: any) { setErro(e.message || 'Falha ao importar geometria'); }
    finally { setBusy(false); }
  };
  const onFile = async (file: File) => {
    setArquivoFonte(file);
    setCadLayerMap({});
    await carregarArquivo(file, undefined, {});
  };
  const onRaster2SeqFile = async (file: File) => {
    setArquivoFonte(file);
    setBusy(true);
    setErro(null);
    setResultado(null);
    setSel(null);
    setSelVertice(null);
    setSelectedAiRoomId(null);
    setSelectedAiVertex(null);
    setSelectedDimensionId(null);
    setSlabEditedByUser(false);
    try {
      const imported = await parsePlantaRaster2Seq(file, rasterCanvasWidth);
      setModelo(imported);
      modelRef.current = imported;
      setMostrarBackdrop(true);
      setMostrarAiOverlay(true);
      fitView(imported);
    } catch (e: any) {
      setErro(e.message || 'Falha ao executar o Raster2Seq');
    } finally {
      setBusy(false);
    }
  };
  const onRasterSlicesFile = async (file: File) => {
    setArquivoFonte(file);
    setBusy(true);
    setErro(null);
    setResultado(null);
    setSel(null);
    setSelVertice(null);
    setSelectedAiRoomId(null);
    setSelectedAiVertex(null);
    setSelectedDimensionId(null);
    setSlabEditedByUser(false);
    try {
      const imported = await parsePlantaRasterSlices(file, rasterCanvasWidth);
      setModelo(imported);
      modelRef.current = imported;
      setMostrarBackdrop(true);
      setMostrarAiOverlay(true);
      setSel(imported.paredes.length ? { kind: 'parede', id: imported.paredes[0].id } : null);
      setAvisoRecuperacao(
        `${imported.paredes.length} parede(s) reconstruída(s) por persistência entre fatias 1D.`,
      );
      fitView(imported);
    } catch (e: any) {
      setErro(e.message || 'Falha na vetorização por fatias 1D');
    } finally {
      setBusy(false);
    }
  };
  const onRaster2DFile = async (file: File) => {
    setArquivoFonte(file);
    setBusy(true);
    setErro(null);
    setResultado(null);
    setSel(null);
    setSelVertice(null);
    setSelectedAiRoomId(null);
    setSelectedAiVertex(null);
    setSelectedDimensionId(null);
    setSlabEditedByUser(false);
    try {
      const imported = await parsePlantaRaster2D(file, rasterCanvasWidth);
      setModelo(imported);
      modelRef.current = imported;
      setMostrarBackdrop(true);
      setMostrarAiOverlay(false);
      setSel(imported.paredes.length ? { kind: 'parede', id: imported.paredes[0].id } : null);
      const columns = imported.paredes.filter(isColumnWall).length;
      const walls = imported.paredes.length - columns;
      const doors = imported.aberturas.filter((opening) => opening.tipo === 'door').length;
      const windows = imported.aberturas.filter((opening) => opening.tipo === 'window').length;
      const slab = imported.laje.piso.ativo && imported.laje.contorno.length >= 3;
      const topology = imported.raster_2d;
      const topologySummary = topology
        ? ` ${topology.source_wall_segment_count} trechos foram consolidados em `
          + `${topology.canonical_wall_count} paredes-mãe; `
          + `${topology.classified_wall_gaps}/${topology.wall_gap_count} intervalos classificados.`
        : '';
      setAvisoRecuperacao(
        `${walls} parede(s), ${columns} pilar(es), ${doors} porta(s), ${windows} janela(s)`
          + ` e ${slab ? '1 laje editável' : 'nenhuma laje confiável'} propostas pela análise 2D.`
          + topologySummary,
      );
      fitView(imported);
    } catch (e: any) {
      setErro(e.message || 'Falha na vetorização morfológica 2D');
    } finally {
      setBusy(false);
    }
  };
  const onPreWallYoloFile = async (file: File) => {
    setArquivoFonte(file);
    setBusy(true);
    setErro(null);
    setResultado(null);
    setSel(null);
    setSelVertice(null);
    setSelectedAiRoomId(null);
    setSelectedAiVertex(null);
    setSelectedDimensionId(null);
    setSlabEditedByUser(false);
    try {
      const imported = await parsePlantaPreWallYolo(
        file,
        rasterCanvasWidth,
        preWallMetricRefinement,
      );
      setModelo(imported);
      modelRef.current = imported;
      setMostrarBackdrop(true);
      setMostrarAiOverlay(false);
      setSel(imported.paredes.length ? { kind: 'parede', id: imported.paredes[0].id } : null);
      const doors = imported.aberturas.filter((opening) => opening.tipo === 'door').length;
      const windows = imported.aberturas.filter((opening) => opening.tipo === 'window').length;
      const raw = imported.pre_wall?.raw_detection_count ?? 0;
      const candidates = imported.pre_wall?.consensus_candidate_count ?? 0;
      setAvisoRecuperacao(
        `YOLO antes das paredes: ${raw} caixas brutas viraram ${candidates} candidatos; `
          + `${imported.paredes.length} parede(s), ${doors} porta(s) e ${windows} janela(s) `
          + 'foram carregadas para revisão manual.',
      );
      fitView(imported);
    } catch (e: any) {
      setErro(e.message || 'Falha no YOLO antes das paredes');
    } finally {
      setBusy(false);
    }
  };
  const recuperarIfcNoEditor = async (file: File) => {
    setBusy(true);
    setErro(null);
    setAvisoRecuperacao(null);
    setResultado(null);
    try {
      const recovered = await scanRecoverIfc(file);
      setModelo(recovered.modelo);
      modelRef.current = recovered.modelo;
      setCfg(recovered.config);
      setSel(null);
      setSelVertice(null);
      setSelectedAiRoomId(null);
      setSelectedAiVertex(null);
      setSelectedDimensionId(null);
      setFinalizedFingerprint(null);
      setFinalizationSummary(null);
      setSlabEditedByUser(false);
      setPendingOpeningRemoval(null);
      wallEndpointEditRef.current = null;
      fitView(recovered.modelo);
      setViewMode('split');
      const aligned = recovered.warnings.length
        ? ` ${recovered.warnings.join(' ')}`
        : '';
      setAvisoRecuperacao(
        `IFC recuperado: ${recovered.counts.walls} paredes, `
        + `${recovered.counts.openings} aberturas e ${recovered.counts.slabs} lajes.`
        + aligned,
      );
    } catch (e: any) {
      setErro(e.message || 'Falha ao reabrir o IFC no editor');
    } finally {
      setBusy(false);
    }
  };
  const onLinkedImage = async (file: File) => {
    setImagemVinculada(file);
    if (arquivoFonte) {
      await carregarArquivo(
        arquivoFonte,
        modelo?.source?.pavimento?.id,
        cadLayerMap,
        modelo?.source?.cad_region?.id,
        file,
      );
    }
  };
  const trocarPavimento = async (pavimento: string) => {
    if (arquivoFonte) await carregarArquivo(arquivoFonte, pavimento);
  };
  const trocarRegiaoCad = async (cadRegion: string) => {
    if (arquivoFonte) {
      await carregarArquivo(
        arquivoFonte,
        undefined,
        cadLayerMap,
        cadRegion,
      );
    }
  };

  const setCadLayerRole = (name: string, role: CadRole | 'auto') => {
    setCadLayerMap((current) => {
      const next = { ...current };
      if (role === 'auto') delete next[name];
      else next[name] = role;
      return next;
    });
  };

  useEffect(() => {
    if (!arquivoInicial) return;
    const key = [
      arquivoInicial.name,
      arquivoInicial.size,
      arquivoInicial.lastModified,
    ].join(':');
    if (initialFileKeyRef.current === key) return;
    initialFileKeyRef.current = key;
    setArquivoFonte(arquivoInicial);
    setCadLayerMap({});
    void carregarArquivo(arquivoInicial, undefined, {});
  }, [arquivoInicial]);

  // ---------- edição de dados ----------
  const patchParede = (id: string, patch: Partial<Parede>) =>
    setModelo((m) => m && { ...m, paredes: m.paredes.map((p) => p.id === id ? { ...p, ...patch } : p) });
  const patchAbertura = (id: string, patch: Partial<Abertura>) =>
    setModelo((current) => {
      if (!current) return current;
      const source = current.aberturas.find((opening) => opening.id === id);
      if (!source) return current;
      const nextOpening = { ...source, ...patch };
      const wall = current.paredes.find((item) => item.id === nextOpening.parede_id);
      if (wall) {
        const length = Math.max(MIN_WALL_LENGTH, Math.hypot(wall.bx - wall.ax, wall.by - wall.ay));
        nextOpening.largura = Math.min(Math.max(0.05, nextOpening.largura), length);
        nextOpening.s_centro = clampOpeningCenter(
          wall,
          nextOpening.largura,
          nextOpening.s_centro,
        );
      }
      const next = {
        ...current,
        aberturas: current.aberturas.map((opening) => opening.id === id ? nextOpening : opening),
      };
      modelRef.current = next;
      return next;
    });

  const patchWallGeometry = (id: string, patch: Partial<Parede>) => {
    const current = modelRef.current;
    if (!current) return;
    const wall = current.paredes.find((item) => item.id === id);
    if (!wall) return;
    const nextWall = { ...wall, ...patch };
    if (Math.hypot(nextWall.bx - nextWall.ax, nextWall.by - nextWall.ay) < 0.05) {
      setErro('A parede precisa ter pelo menos 5 cm de comprimento.');
      return;
    }
    const anchors = captureOpeningWorldAnchors(wall, current.aberturas);
    const nextOpenings = remapOpeningsToWall(current.aberturas, nextWall, anchors);
    const next = {
      ...current,
      paredes: current.paredes.map((item) => item.id === id ? nextWall : item),
      aberturas: nextOpenings,
    };
    modelRef.current = next;
    setModelo(next);
    setErro(null);
    const invalidOpenings = openingsOutsideWall(nextWall, nextOpenings);
    if (invalidOpenings.length) {
      setPendingOpeningRemoval({
        wallId: id,
        beforeModel: current,
        openings: invalidOpenings,
      });
    }
  };

  const setWallLength = (id: string, length: number) => {
    const wall = modelRef.current?.paredes.find((item) => item.id === id);
    if (!wall || !Number.isFinite(length) || length < 0.05) return;
    const currentLength = Math.hypot(wall.bx - wall.ax, wall.by - wall.ay);
    const ux = currentLength > MIN_WALL_LENGTH ? (wall.bx - wall.ax) / currentLength : 1;
    const uy = currentLength > MIN_WALL_LENGTH ? (wall.by - wall.ay) / currentLength : 0;
    patchWallGeometry(id, {
      bx: wall.ax + ux * length,
      by: wall.ay + uy * length,
    });
  };

  const setWallAngle = (id: string, degrees: number) => {
    const wall = modelRef.current?.paredes.find((item) => item.id === id);
    if (!wall || !Number.isFinite(degrees)) return;
    const length = Math.hypot(wall.bx - wall.ax, wall.by - wall.ay);
    const radians = degrees * Math.PI / 180;
    patchWallGeometry(id, {
      bx: wall.ax + Math.cos(radians) * length,
      by: wall.ay + Math.sin(radians) * length,
    });
  };

  const duplicateOpening = (id: string) => {
    const current = modelRef.current;
    if (!current) return;
    const source = current.aberturas.find((opening) => opening.id === id);
    const wall = source && current.paredes.find((item) => item.id === source.parede_id);
    if (!source || !wall) return;
    const nextId = proximoId(current.aberturas.map((opening) => opening.id), 'O-EDIT');
    const nextOpening: Abertura = {
      ...source,
      id: nextId,
      nome: source.nome ? `${source.nome} - copia` : undefined,
      guid: undefined,
      origem: 'editor-copy',
      confidence: 1,
      altura: source.altura
        ?? (source.tipo === 'door' ? cfg.porta_altura : cfg.janela_altura),
      peitoril: source.tipo === 'door'
        ? 0
        : (source.peitoril ?? cfg.janela_peitoril),
      s_centro: duplicateOpeningCenter(wall, current.aberturas, source),
    };
    const next = { ...current, aberturas: [...current.aberturas, nextOpening] };
    modelRef.current = next;
    setModelo(next);
    setSel({ kind: 'abertura', id: nextId });
  };

  const patchFace = (face: 'piso' | 'teto', patch: Partial<{ ativo: boolean; espessura: number }>) => {
    setSlabEditedByUser(true);
    setModelo((m) => m && { ...m, laje: { ...m.laje, [face]: { ...m.laje[face], ...patch } } });
  };
  const moveVertice = (idx: number, x: number, y: number) => {
    setSlabEditedByUser(true);
    setModelo((m) => {
      if (!m) return m;
      const c = m.laje.contorno.map((v, i) => i === idx ? [x, y] as [number, number] : v);
      return { ...m, laje: { ...m.laje, contorno: c } };
    });
  };
  const addVertice = (afterIdx: number) => {
    setSlabEditedByUser(true);
    setModelo((m) => {
      if (!m) return m;
      const c = [...m.laje.contorno];
      const a = c[afterIdx], b = c[(afterIdx + 1) % c.length];
      c.splice(afterIdx + 1, 0, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
      return { ...m, laje: { ...m.laje, contorno: c } };
    });
  };
  const removeVertice = (idx: number) => {
    setSlabEditedByUser(true);
    setModelo((m) => {
      if (!m || m.laje.contorno.length <= 3) return m;  // mantem poligono valido
      return { ...m, laje: { ...m.laje, contorno: m.laje.contorno.filter((_, i) => i !== idx) } };
    });
  };

  const patchRasterRoomPoints = (
    roomId: string,
    updater: (points: [number, number][]) => [number, number][],
  ) => setModelo((current) => {
    if (!current || current.reference?.kind !== 'raster2seq') return current;
    const rooms = current.reference.rooms.map((room) => {
      if (room.id !== roomId) return room;
      const points = updater(room.points);
      return { ...room, points, area: polygonArea(points) };
    });
    const next: ModeloPlanta = {
      ...current,
      reference: { ...current.reference, rooms },
    };
    modelRef.current = next;
    return next;
  });

  const moveRasterRoomVertex = (roomId: string, idx: number, x: number, y: number) =>
    patchRasterRoomPoints(
      roomId,
      (points) => points.map((point, index) => (
        index === idx ? [x, y] as [number, number] : point
      )),
    );

  const addRasterRoomVertex = (roomId: string, afterIdx: number) =>
    patchRasterRoomPoints(roomId, (points) => {
      const next = [...points];
      const first = points[afterIdx];
      const second = points[(afterIdx + 1) % points.length];
      next.splice(afterIdx + 1, 0, [
        (first[0] + second[0]) / 2,
        (first[1] + second[1]) / 2,
      ]);
      return next;
    });

  const removeRasterRoomVertex = (roomId: string, idx: number) =>
    patchRasterRoomPoints(
      roomId,
      (points) => points.length <= 3 ? points : points.filter((_, index) => index !== idx),
    );

  const converterRasterEmParedes = () => {
    const current = modelRef.current;
    if (!current || current.reference?.kind !== 'raster2seq') return;
    const pixelSize = current.reference.canvas_width_m
      / Math.max(1, current.reference.canvas_size[0]);
    const segments = rasterRoomsToWallSegments(
      current.reference.rooms.filter((room) => room.category_id !== 0),
      {
        snapTolerance: Math.max(0.08, Math.min(0.35, pixelSize * 3)),
        minimumLength: Math.max(0.18, pixelSize * 3),
      },
    );
    if (!segments.length) {
      setErro('A previsão não contém contornos longos o bastante para criar paredes.');
      return;
    }

    const previousAiWalls = new Map(
      current.paredes
        .filter((wall) => wall.origem === 'raster2seq-contour')
        .map((wall) => [wall.id, wall]),
    );
    const manualWalls = current.paredes.filter(
      (wall) => wall.origem !== 'raster2seq-contour',
    );
    const usedIds = new Set(manualWalls.map((wall) => wall.id));
    const generatedWalls = segments.map((segment, index): Parede => {
      let id = `W-AI-${String(index + 1).padStart(3, '0')}`;
      while (usedIds.has(id)) {
        id = proximoId([...usedIds], 'W-AI');
      }
      usedIds.add(id);
      const previous = previousAiWalls.get(id);
      return {
        ...previous,
        id,
        ...segment,
        espessura: previous?.espessura ?? 0.15,
        altura: previous?.altura ?? cfg.altura,
        elevacao: previous?.elevacao ?? 0,
        layer: previous?.layer ?? 'Wall-IA-Revisada',
        nome: previous?.nome ?? `Parede IA ${index + 1}`,
        origem: 'raster2seq-contour',
        confidence: previous?.confidence ?? 0.55,
      };
    });
    const validWallIds = new Set([
      ...manualWalls.map((wall) => wall.id),
      ...generatedWalls.map((wall) => wall.id),
    ]);
    const nextOpenings = current.aberturas.filter(
      (opening) => validWallIds.has(opening.parede_id),
    );
    const removedOpenings = current.aberturas.length - nextOpenings.length;
    const next: ModeloPlanta = {
      ...current,
      paredes: [...manualWalls, ...generatedWalls],
      aberturas: nextOpenings,
    };
    modelRef.current = next;
    setModelo(next);
    setSel(generatedWalls.length ? { kind: 'parede', id: generatedWalls[0].id } : null);
    setSelectedAiRoomId(null);
    setSelectedAiVertex(null);
    setModoLaje(false);
    setResultado(null);
    setFinalizedFingerprint(null);
    setFinalizationSummary(null);
    setErro(null);
    setAvisoRecuperacao(
      `${generatedWalls.length} paredes editáveis criadas a partir dos contornos da IA.`
      + (removedOpenings ? ` ${removedOpenings} esquadria(s) sem parede foram removidas.` : ''),
    );
  };

  const aplicarCotaNaParede = () => {
    if (!selectedDimension || !selectedWallForDimension) {
      setErro('Selecione uma cota OCR e a parede correspondente.');
      return;
    }
    setWallLength(selectedWallForDimension.id, selectedDimension.value_m);
    setAvisoRecuperacao(
      `Parede ${selectedWallForDimension.id} ajustada para `
      + `${selectedDimension.value_m.toFixed(3)} m pela cota “${selectedDimension.text}”.`,
    );
  };

  const calibrarPlantaComCota = () => {
    const current = modelRef.current;
    if (!current || !selectedDimension || !selectedWallForDimension) {
      setErro('Selecione uma cota OCR e a parede que ela está medindo.');
      return;
    }
    const currentLength = Math.hypot(
      selectedWallForDimension.bx - selectedWallForDimension.ax,
      selectedWallForDimension.by - selectedWallForDimension.ay,
    );
    if (currentLength < 0.05) {
      setErro('A parede selecionada é curta demais para calibrar a planta.');
      return;
    }
    const ratio = selectedDimension.value_m / currentLength;
    if (!Number.isFinite(ratio) || ratio < 0.25 || ratio > 4) {
      setErro(
        `A cota produziria um fator ${ratio.toFixed(3)}×. `
        + 'Confirme se o texto pertence realmente à parede selecionada.',
      );
      return;
    }
    try {
      const scaled = scaleRasterModel(current, ratio);
      modelRef.current = scaled;
      setModelo(scaled);
      setRasterCanvasWidth(scaled.reference?.kind === 'raster2seq'
        ? scaled.reference.canvas_width_m
        : rasterCanvasWidth);
      fitView(scaled);
      setResultado(null);
      setFinalizedFingerprint(null);
      setFinalizationSummary(null);
      setErro(null);
      setAvisoRecuperacao(
        `Planta recalibrada por ${ratio.toFixed(4)}×: `
        + `${currentLength.toFixed(3)} m → ${selectedDimension.value_m.toFixed(3)} m `
        + `usando “${selectedDimension.text}”.`,
      );
    } catch (error: any) {
      setErro(error.message || 'Não foi possível recalibrar a planta.');
    }
  };

  const apagarSel = () => {
    if (!sel) return;
    setModelo((current) => {
      if (!current) return current;
      const next = sel.kind === 'parede'
        ? {
            ...current,
            paredes: current.paredes.filter((p) => p.id !== sel.id),
            aberturas: current.aberturas.filter((a) => a.parede_id !== sel.id),
          }
        : {
            ...current,
            aberturas: current.aberturas.filter((a) => a.id !== sel.id),
          };
      modelRef.current = next;
      return next;
    });
    setSel(null);
  };

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && measureMode) {
        event.preventDefault();
        setMeasurementStart(null);
        setMeasurementEnd(null);
        setMeasurementCursor(null);
        setMeasureMode(false);
        return;
      }
      if (event.key !== 'Delete' && event.key !== 'Backspace') return;
      const target = event.target as HTMLElement | null;
      if (target?.closest('input, textarea, select, [contenteditable="true"]')) return;
      if (pendingOpeningRemoval) return;
      if (selectedAiRoomId && selectedAiVertex !== null) {
        event.preventDefault();
        removeRasterRoomVertex(selectedAiRoomId, selectedAiVertex);
        setSelectedAiVertex(null);
        return;
      }
      if (modoLaje && selVertice !== null) {
        event.preventDefault();
        removeVertice(selVertice);
        setSelVertice(null);
        return;
      }
      if (!sel) return;
      event.preventDefault();
      apagarSel();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [
    sel, selVertice, modoLaje, pendingOpeningRemoval, modelo,
    selectedAiRoomId, selectedAiVertex, measureMode,
  ]);

  const addParede = () => {
    if (!modelo) return;
    const { xmin, ymin, xmax, ymax } = modelo.bbox;
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
    const id = proximoId(modelo.paredes.map((p) => p.id), 'W-EDIT');
    setModelo({ ...modelo, paredes: [...modelo.paredes,
      { id, ax: cx - 1, ay: cy, bx: cx + 1, by: cy, espessura: 0.15,
        altura: cfg.altura, elevacao: 0, layer: 'Wall-Nova', tipo: 'wall', ifc_class: 'IfcWall' }] });
    setSel({ kind: 'parede', id });
  };

  const addPilar = () => {
    if (!modelo) return;
    const { xmin, ymin, xmax, ymax } = modelo.bbox;
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
    const id = proximoId(modelo.paredes.map((p) => p.id), 'C-EDIT');
    setModelo({ ...modelo, paredes: [...modelo.paredes,
      { id, ax: cx - 0.15, ay: cy, bx: cx + 0.15, by: cy, espessura: 0.30,
        altura: cfg.altura, elevacao: 0, layer: 'Column-Nova', nome: 'Pilar novo',
        tipo: 'column', ifc_class: 'IfcColumn', origem: 'manual-column' }] });
    setSel({ kind: 'parede', id });
  };

  const addAbertura = (tipo: 'door' | 'window') => {
    if (!modelo || sel?.kind !== 'parede') return;
    const p = paredeById[sel.id];
    if (!p || isColumnWall(p)) return;
    const L = Math.hypot(p.bx - p.ax, p.by - p.ay);
    const id = proximoId(modelo.aberturas.map((a) => a.id), 'O-EDIT');
    setModelo({ ...modelo, aberturas: [...modelo.aberturas,
      { id, parede_id: p.id, tipo, s_centro: L / 2, largura: tipo === 'door' ? 0.8 : 1.0 }] });
    setSel({ kind: 'abertura', id });
  };

  // ---------- coordenadas mundo <-> tela ----------
  const worldFromEvent = (e: React.MouseEvent): { x: number; y: number } => {
    const svg = svgRef.current!;
    const p = svg.createSVGPoint();
    p.x = e.clientX; p.y = e.clientY;
    const sp = p.matrixTransform(svg.getScreenCTM()!.inverse());
    return { x: sp.x, y: -sp.y };  // Y invertido (planta cresce pra cima)
  };

  const snapMeasurementPoint = (point: MeasurementPoint): MeasurementPoint => {
    if (!modelo) return point;
    const candidates: MeasurementPoint[] = [];
    modelo.paredes.forEach((wall) => {
      candidates.push(
        { x: wall.ax, y: wall.ay, snapLabel: `${wall.id}.P1` },
        { x: wall.bx, y: wall.by, snapLabel: `${wall.id}.P2` },
      );
    });
    modelo.aberturas.forEach((opening) => {
      const wall = paredeById[opening.parede_id];
      if (!wall) return;
      const length = Math.hypot(wall.bx - wall.ax, wall.by - wall.ay);
      if (length < MIN_WALL_LENGTH) return;
      const ux = (wall.bx - wall.ax) / length;
      const uy = (wall.by - wall.ay) / length;
      const centerX = wall.ax + ux * opening.s_centro;
      const centerY = wall.ay + uy * opening.s_centro;
      candidates.push(
        {
          x: centerX - ux * opening.largura / 2,
          y: centerY - uy * opening.largura / 2,
          snapLabel: `${opening.id}.início`,
        },
        {
          x: centerX + ux * opening.largura / 2,
          y: centerY + uy * opening.largura / 2,
          snapLabel: `${opening.id}.fim`,
        },
      );
    });
    modelo.laje.contorno.forEach(([x, y], index) => {
      candidates.push({ x, y, snapLabel: `Laje.V${index + 1}` });
    });
    const screenWidth = Math.max(1, svgRef.current?.getBoundingClientRect().width ?? 400);
    const tolerance = Math.max(0.01, (view?.w ?? 10) / screenWidth * 12);
    let best = point;
    let bestDistance = tolerance;
    candidates.forEach((candidate) => {
      const distance = Math.hypot(candidate.x - point.x, candidate.y - point.y);
      if (distance <= bestDistance) {
        best = candidate;
        bestDistance = distance;
      }
    });
    return best;
  };

  const placeMeasurementPoint = (event: React.MouseEvent) => {
    if (!measureMode || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    dragRef.current = null;
    const point = snapMeasurementPoint(worldFromEvent(event));
    if (!measurementStart || measurementEnd) {
      setMeasurementStart(point);
      setMeasurementEnd(null);
      setMeasurementCursor(point);
      return;
    }
    setMeasurementEnd(point);
    setMeasurementCursor(point);
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (measureMode && measurementStart && !measurementEnd) {
      setMeasurementCursor(snapMeasurementPoint(worldFromEvent(e)));
    }
    const d = dragRef.current;
    if (!d || !modelo) return;
    if (d.kind === 'pan') {
      const rect = svgRef.current!.getBoundingClientRect();
      const sx = view!.w / rect.width, sy = view!.h / rect.height;
      setView({ ...view!, x: d.vx - (e.clientX - d.px) * sx, y: d.vy - (e.clientY - d.py) * sy });
      return;
    }
    const w = worldFromEvent(e);
    if (d.kind === 'opening') {
      const p = paredeById[modelo.aberturas.find((a) => a.id === d.id)!.parede_id];
      const ux = p.bx - p.ax, uy = p.by - p.ay;
      const L = Math.hypot(ux, uy);
      const s = ((w.x - p.ax) * ux + (w.y - p.ay) * uy) / L;  // projeção no eixo
      const ab = modelo.aberturas.find((a) => a.id === d.id)!;
      patchAbertura(d.id, { s_centro: Math.max(ab.largura / 2, Math.min(L - ab.largura / 2, s)) });
    } else if (d.kind === 'wall-a') {
      moveWallEndpoint(d.id, { ax: w.x, ay: w.y });
    } else if (d.kind === 'wall-b') {
      moveWallEndpoint(d.id, { bx: w.x, by: w.y });
    } else if (d.kind === 'wall-body') {
      const dx = w.x - d.ox, dy = w.y - d.oy;
      patchParede(d.id, { ax: d.ax + dx, ay: d.ay + dy, bx: d.bx + dx, by: d.by + dy });
    } else if (d.kind === 'raster-room-vertex') {
      moveRasterRoomVertex(d.id, d.idx, w.x, w.y);
    } else if (d.kind === 'laje-vertice') {
      moveVertice(d.idx, w.x, w.y);
    }
  };

  const beginWallEndpointDrag = (
    kind: 'wall-a' | 'wall-b',
    wall: Parede,
  ) => {
    const current = modelRef.current;
    if (!current) return;
    wallEndpointEditRef.current = {
      wallId: wall.id,
      beforeModel: current,
      anchors: captureOpeningWorldAnchors(wall, current.aberturas),
    };
    dragRef.current = {
      kind,
      id: wall.id,
      ox: 0,
      oy: 0,
      ax: wall.ax,
      ay: wall.ay,
      bx: wall.bx,
      by: wall.by,
    };
  };

  const moveWallEndpoint = (id: string, patch: Partial<Parede>) => {
    const edit = wallEndpointEditRef.current;
    setModelo((current) => {
      if (!current) return current;
      const wall = current.paredes.find((item) => item.id === id);
      if (!wall) return current;
      const nextWall = { ...wall, ...patch };
      const nextOpenings = edit?.wallId === id
        ? remapOpeningsToWall(current.aberturas, nextWall, edit.anchors)
        : current.aberturas;
      const next = {
        ...current,
        paredes: current.paredes.map((item) => item.id === id ? nextWall : item),
        aberturas: nextOpenings,
      };
      modelRef.current = next;
      return next;
    });
  };

  const endDrag = () => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (drag?.kind !== 'wall-a' && drag?.kind !== 'wall-b') return;

    const edit = wallEndpointEditRef.current;
    wallEndpointEditRef.current = null;
    const current = modelRef.current;
    if (!edit || !current) return;
    const wall = current.paredes.find((item) => item.id === edit.wallId);
    if (!wall) return;
    const invalidOpenings = openingsOutsideWall(wall, current.aberturas);
    if (invalidOpenings.length) {
      setPendingOpeningRemoval({
        wallId: wall.id,
        beforeModel: edit.beforeModel,
        openings: invalidOpenings,
      });
    }
  };

  const cancelOpeningRemoval = () => {
    if (!pendingOpeningRemoval) return;
    modelRef.current = pendingOpeningRemoval.beforeModel;
    setModelo(pendingOpeningRemoval.beforeModel);
    setPendingOpeningRemoval(null);
  };

  const confirmOpeningRemoval = () => {
    if (!pendingOpeningRemoval) return;
    const ids = new Set(pendingOpeningRemoval.openings.map((opening) => opening.id));
    setModelo((current) => {
      if (!current) return current;
      const next = {
        ...current,
        aberturas: current.aberturas.filter((opening) => !ids.has(opening.id)),
      };
      modelRef.current = next;
      return next;
    });
    if (sel?.kind === 'abertura' && ids.has(sel.id)) setSel(null);
    setPendingOpeningRemoval(null);
  };

  // ---------- zoom (scroll) — listener NATIVO com passive:false ----------
  // (o onWheel do React e passivo -> nao permite preventDefault e nao pega
  //  bem o zoom; anexamos direto no svg pra controlar de verdade)
  const viewRef = useRef<View | null>(null);
  viewRef.current = view;
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const handler = (e: WheelEvent) => {
      const v = viewRef.current;
      if (!v) return;
      e.preventDefault();
      const p = svg.createSVGPoint();
      p.x = e.clientX; p.y = e.clientY;
      const sp = p.matrixTransform(svg.getScreenCTM()!.inverse());  // ponto svg-space sob o cursor
      const factor = e.deltaY > 0 ? 1.12 : 1 / 1.12;                // frente = aproxima
      const nw = Math.max(0.4, Math.min(500, v.w * factor));
      const nh = nw * (v.h / v.w);
      const rx = (sp.x - v.x) / v.w, ry = (sp.y - v.y) / v.h;
      setView({ x: sp.x - rx * nw, y: sp.y - ry * nh, w: nw, h: nh });
    };
    svg.addEventListener('wheel', handler, { passive: false });
    return () => svg.removeEventListener('wheel', handler);
  // O SVG desmonta no modo 3D e monta novamente ao voltar para 2D/split.
  // viewMode precisa participar do ciclo para o listener acompanhar esse DOM.
  }, [modelo === null, viewMode]);

  // ---------- aprovação do cliente + exportações ----------
  const baixarArquivo = (url: string, nome: string) => {
    const link = document.createElement('a');
    const arquivoUrl = downloadUrl(url);
    link.href = `${arquivoUrl}${arquivoUrl.includes('?') ? '&' : '?'}download=1`;
    link.download = nome.replace(/[^\w.-]+/g, '_') || 'planta';
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const baixarIfc = (resultadoIfc: ResultadoGeracao, nome: string) =>
    baixarArquivo(resultadoIfc.ifc_url, `${nome}.ifc`);

  const gerar = async (automaticamenteFinalizado: boolean) => {
    if (!modelo) return;
    if (!aprovacaoClienteAtual || !clientApprovedAt) {
      setErro('O cliente precisa revisar e confirmar este estado do projeto antes da exportação.');
      return;
    }
    setBusy(true); setErro(null); setResultado(null);
    try {
      let modeloExportacao = modelo;
      if (!automaticamenteFinalizado) {
        // "Exatamente como desenhado" preserva os eixos, mas Spaces continuam
        // sendo informacao derivada. Recalcule os ciclos atuais em vez de
        // mandar [] e deixar o exportador inventar um ambiente pelo slab.
        const derivacao = await aplicarRevisaoBim(modelo, {
          schema: 'bim.edit-operations.v1',
          operations: [],
          recalculate: ['topology', 'spaces', 'validation'],
          policies: {
            topology_snap_tolerance: SPACE_TOPOLOGY_TOLERANCE,
            minimum_space_area: 0.5,
          },
        });
        modeloExportacao = derivacao.model;
        setModelo(modeloExportacao);
        modelRef.current = modeloExportacao;
      }
      if (cfg.forro?.ativo && !(modeloExportacao.spaces ?? []).length) {
        throw new Error(
          'Forro exige ao menos um cômodo fechado. Revise os gaps das paredes; '
          + 'o sistema não cria mais Space geral pelo contorno da laje.',
        );
      }
      const payload = {
        paredes: modeloExportacao.paredes,
        aberturas: modeloExportacao.aberturas,
        laje: modeloExportacao.laje,
        spaces: modeloExportacao.spaces ?? [],
      };
      const nome = modeloExportacao.nome || 'planta';
      const configEnvio = {
        ...cfg,
        finalizacao_automatica: automaticamenteFinalizado,
      };
      const r = onGerar
        ? await onGerar(payload, configEnvio, nome)
        : await gerarPlantaIfc(payload, configEnvio, nome, {
            confirmado: true,
            confirmado_em: clientApprovedAt,
          });
      baixarIfc(r, nome);
      setResultado(r);
    } catch (e: any) { setErro(e.message || 'Falha ao gerar IFC'); }
    finally { setBusy(false); }
  };

  const gerarDxf = async () => {
    if (!modelo) return;
    if (!aprovacaoClienteAtual || !clientApprovedAt) {
      setErro('O cliente precisa revisar e confirmar este estado do projeto antes da exportação.');
      return;
    }
    setBusy(true); setErro(null);
    try {
      const nome = modelo.nome || 'planta';
      const result = await gerarPlantaDxf(modelo, nome, {
        confirmado: true,
        confirmado_em: clientApprovedAt,
      });
      baixarArquivo(result.dxf_url, `${nome}.dxf`);
      setAvisoRecuperacao(
        `DXF aprovado gerado em metros: ${result.walls} paredes, ${result.doors} portas e ${result.windows} janelas.`,
      );
    } catch (e: any) {
      setErro(e.message || 'Falha ao gerar DXF');
    } finally {
      setBusy(false);
    }
  };

  const finalizarAutomaticamente = async () => {
    if (!modelo) return;
    setBusy(true); setErro(null); setResultado(null);
    try {
      const repairRasterSlab = isGeometricRasterReference && !slabEditedByUser;
      const modeloComAlturas: ModeloPlanta = {
        ...modelo,
        paredes: modelo.paredes.map((parede) => ({
          ...parede,
          altura: parede.altura ?? cfg.altura,
        })),
      };
      const revisao = await aplicarRevisaoBim(modeloComAlturas, {
        schema: 'bim.edit-operations.v1',
        operations: [{
          op: 'close_wall_junctions',
          max_distance: 0.30,
          min_angle_deg: 25,
          protect_openings: true,
          iterations: 2,
          kinds: ['L', 'T'],
        }],
        // A laje raster ainda é uma proposta automática. Antes da aprovação,
        // amplia a envoltória para cobrir as faces externas. Se o usuário já
        // editou a laje, preserva exatamente o contorno manual.
        recalculate: [
          'openings',
          'topology',
          'spaces',
          ...(repairRasterSlab ? ['slabs' as const] : []),
          'validation',
        ],
        policies: {
          topology_snap_tolerance: SPACE_TOPOLOGY_TOLERANCE,
          minimum_space_area: 0.5,
          opening_out_of_bounds: 'clamp',
          slab_fit_mode: 'outer_faces_hull',
        },
      });
      const nextModel = revisao.model;
      const wallTopLevels = nextModel.paredes
        .map((parede) => (parede.elevacao ?? 0) + (parede.altura ?? cfg.altura))
        .filter((value) => Number.isFinite(value) && value > 0)
        .sort((a, b) => a - b);
      const middle = Math.floor(wallTopLevels.length / 2);
      const dominantWallTop = wallTopLevels.length === 0
        ? cfg.altura
        : wallTopLevels.length % 2
          ? wallTopLevels[middle]
          : (wallTopLevels[middle - 1] + wallTopLevels[middle]) / 2;
      const forroThickness = cfg.forro?.espessura ?? 0.03;
      const maxForroHeight = Math.max(0.1, dominantWallTop - forroThickness);
      const finalizedConfig: PlantaConfig = {
        ...cfg,
        altura: dominantWallTop,
        forro: {
          ativo: cfg.forro?.ativo ?? false,
          espessura: forroThickness,
          altura: Math.min(cfg.forro?.altura ?? maxForroHeight, maxForroHeight),
        },
      };
      const junctionResult = revisao.report.operation_results?.find(
        (item) => item.op === 'close_wall_junctions');
      setModelo(nextModel);
      setCfg(finalizedConfig);
      modelRef.current = nextModel;
      setFinalizedFingerprint(modelFingerprint(nextModel, finalizedConfig));
      setClientApprovedFingerprint(null);
      setClientApprovedAt(null);
      setFinalizationSummary({
        revision: revisao.report.revision,
        junctions: junctionResult?.joined?.length ?? 0,
        movedEndpoints: Object.keys(junctionResult?.moved_endpoints ?? {}).length,
        blockedByOpenings: junctionResult?.blocked_by_openings?.length ?? 0,
        spaces: nextModel.spaces?.length ?? 0,
        slabAdjusted: repairRasterSlab,
      });
      setViewMode('split');
      fitView(nextModel);
    } catch (e: any) {
      setErro(e.message || 'Falha na finalização automática');
    } finally {
      setBusy(false);
    }
  };

  // ---------- viewBox / escala dos tracos de UI (constante em pixels) ----------
  const vb = view ? `${view.x} ${view.y} ${view.w} ${view.h}` : '0 0 10 10';
  const escalaTraco = view ? view.w / 400 : 0.02;  // ~1px; escala com o zoom
  const measurementDisplayEnd = measurementEnd ?? measurementCursor;
  const measurement = measurementStart && measurementDisplayEnd ? {
    start: measurementStart,
    end: measurementDisplayEnd,
    dx: measurementDisplayEnd.x - measurementStart.x,
    dy: measurementDisplayEnd.y - measurementStart.y,
    distance: Math.hypot(
      measurementDisplayEnd.x - measurementStart.x,
      measurementDisplayEnd.y - measurementStart.y,
    ),
    complete: Boolean(measurementEnd),
  } : null;
  const finalizacaoAtual = Boolean(
    currentModelFingerprint && finalizedFingerprint === currentModelFingerprint);
  const aprovacaoClienteAtual = Boolean(
    finalizacaoAtual
      && currentModelFingerprint
      && clientApprovedFingerprint === currentModelFingerprint,
  );
  const confirmarAprovacaoCliente = () => {
    if (!currentModelFingerprint || !finalizacaoAtual || !clientApprovalChecked) return;
    setClientApprovedFingerprint(currentModelFingerprint);
    setClientApprovedAt(new Date().toISOString());
    setErro(null);
  };
  const approvalSummary = modelo ? {
    walls: modelo.paredes.filter((wall) => !isColumnWall(wall)).length,
    columns: modelo.paredes.filter(isColumnWall).length,
    doors: modelo.aberturas.filter((opening) => opening.tipo === 'door').length,
    windows: modelo.aberturas.filter((opening) => opening.tipo === 'window').length,
    slabArea: polygonArea(modelo.laje.contorno),
    scale: modelo.escala && modelo.escala > 0
      ? `1 px = ${(modelo.escala * 100).toFixed(2)} cm`
      : 'coordenadas em metros',
  } : null;

  // =====================================================================
  //  RENDER
  // =====================================================================
  if (!modelo) {
    return (
      <div className="p-8">
        <Cabecalho titulo={titulo} subtitulo={subtitulo} />
        <div className="mt-5 grid max-w-6xl gap-5 xl:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
        <div className="min-w-0">
        <label className="mt-5 flex max-w-xs items-center justify-between gap-3 text-xs text-slate-600">
          Escala para PDF vetorial: 1:
          <input
            type="number"
            min={1}
            max={1000}
            value={pdfScale}
            onChange={(e) => setPdfScale(Math.max(1, +e.target.value || 1))}
            className="w-24 rounded border border-slate-200 px-2 py-1 text-right"
          />
        </label>
        <label className="mt-6 flex flex-col items-center justify-center gap-3 h-64 max-w-2xl
                          border-2 border-dashed border-slate-300 rounded-2xl cursor-pointer
                          hover:bg-slate-50 hover:border-blue-300 transition-colors">
          {busy ? <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                : <FileUp className="w-8 h-8 text-slate-400" />}
          <span className="text-sm text-slate-600 font-medium">
            {busy ? 'Processando planta...' : 'Solte IFC, IFCZIP, DXF, DWG, SVG ou PDF'}
          </span>
          <span className="text-xs text-slate-400">
            IFC preserva objetos; CAD/SVG/PDF preservam vetores
          </span>
          <span className="text-[11px] text-slate-400">
            OBJ, USDZ, PLY e E57 entram pela aba Scan → BIM
          </span>
          <input type="file" accept=".ifc,.ifczip,.dxf,.dwg,.svg,.pdf" className="hidden"
                 onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        </label>
        <div className="mt-4 max-w-2xl rounded-2xl border border-emerald-200 bg-emerald-50/70 p-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="max-w-sm">
              <div className="flex items-center gap-2 text-sm font-semibold text-emerald-950">
                <Sparkles className="h-4 w-4" /> YOLO antes das paredes
              </div>
              <p className="mt-1 text-[11px] text-emerald-800/80">
                Recorta a edificação, detecta aberturas em 640/960/1280 px e só depois
                reconstrói as paredes para abrir tudo no editor.
              </p>
            </div>
            <div className="space-y-2 text-xs text-emerald-900">
              <label className="flex items-center justify-between gap-2">
                Largura construída
                <input
                  type="number"
                  aria-label="Largura construída YOLO m"
                  min={1}
                  max={500}
                  step={0.05}
                  value={rasterCanvasWidth}
                  onChange={(event) => setRasterCanvasWidth(
                    Math.min(500, Math.max(1, +event.target.value || 20)),
                  )}
                  className="w-20 rounded-lg border border-emerald-200 bg-white px-2 py-1.5 text-right"
                />
                m
              </label>
              <label className="flex cursor-pointer items-center justify-end gap-2 text-[10px]">
                <input
                  type="checkbox"
                  checked={preWallMetricRefinement}
                  onChange={(event) => setPreWallMetricRefinement(event.target.checked)}
                  className="rounded border-emerald-300 text-emerald-600"
                />
                Usar largura para distinguir porta/janela
              </label>
            </div>
          </div>
          <label className="mt-3 flex cursor-pointer items-center justify-center gap-2 rounded-xl
                            bg-emerald-700 px-4 py-3 text-xs font-semibold text-white
                            hover:bg-emerald-800 has-[:disabled]:cursor-wait has-[:disabled]:opacity-60">
            {busy
              ? <Loader2 className="h-4 w-4 animate-spin" />
              : <Sparkles className="h-4 w-4" />}
            {busy ? 'Rodando YOLO multiescala…' : 'Testar PNG/JPG e abrir no editor'}
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff"
              disabled={busy}
              className="hidden"
              onChange={(event) => {
                const file = event.currentTarget.files?.[0];
                event.currentTarget.value = '';
                if (file) void onPreWallYoloFile(file);
              }}
            />
          </label>
          <p className="mt-2 text-[10px] text-emerald-800/70">
            Se a largura for aproximada, desmarque a classificação métrica. O resultado continua editável.
          </p>
        </div>
        <div className="mt-4 max-w-2xl rounded-2xl border border-amber-200 bg-amber-50/70 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold text-amber-950">
                <Square className="h-4 w-4" /> Vetorizar por regiões 2D
              </div>
              <p className="mt-1 text-[11px] text-amber-800/80">
                Separa massas de parede, reconhece arcos de porta e componentes de janela.
              </p>
            </div>
            <label className="flex items-center gap-2 text-xs text-amber-900">
              Largura geométrica
              <input
                type="number"
                aria-label="Largura geométrica 2D m"
                min={1}
                max={500}
                step={0.5}
                value={rasterCanvasWidth}
                onChange={(event) => setRasterCanvasWidth(
                  Math.min(500, Math.max(1, +event.target.value || 20)),
                )}
                className="w-20 rounded-lg border border-amber-200 bg-white px-2 py-1.5 text-right"
              />
              m
            </label>
          </div>
          <label className="mt-3 flex cursor-pointer items-center justify-center gap-2 rounded-xl
                            bg-amber-600 px-4 py-3 text-xs font-semibold text-white
                            hover:bg-amber-700 has-[:disabled]:cursor-wait has-[:disabled]:opacity-60">
            {busy
              ? <Loader2 className="h-4 w-4 animate-spin" />
              : <Square className="h-4 w-4" />}
            {busy ? 'Analisando regiões...' : 'Testar PNG/JPG com detector 2D'}
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff"
              disabled={busy}
              className="hidden"
              onChange={(event) => event.target.files?.[0]
                && void onRaster2DFile(event.target.files[0])}
            />
          </label>
          <p className="mt-2 text-[10px] text-amber-800/70">
            Experimental: plantas monocromáticas e escadas ainda exigem revisão manual.
          </p>
        </div>
        <div className="mt-4 max-w-2xl rounded-2xl border border-cyan-200 bg-cyan-50/70 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold text-cyan-950">
                <ScanLine className="h-4 w-4" /> Vetorizar por fatias 1D
              </div>
              <p className="mt-1 text-[11px] text-cyan-800/80">
                Reduz faixas móveis a perfis 1D e reconstrói paredes pelas faces persistentes.
              </p>
            </div>
            <label className="flex items-center gap-2 text-xs text-cyan-900">
              Largura geométrica
              <input
                type="number"
                aria-label="Largura geométrica m"
                min={1}
                max={500}
                step={0.5}
                value={rasterCanvasWidth}
                onChange={(event) => setRasterCanvasWidth(
                  Math.min(500, Math.max(1, +event.target.value || 20)),
                )}
                className="w-20 rounded-lg border border-cyan-200 bg-white px-2 py-1.5 text-right"
              />
              m
            </label>
          </div>
          <label className="mt-3 flex cursor-pointer items-center justify-center gap-2 rounded-xl
                            bg-cyan-700 px-4 py-3 text-xs font-semibold text-white
                            hover:bg-cyan-800 has-[:disabled]:cursor-wait has-[:disabled]:opacity-60">
            {busy
              ? <Loader2 className="h-4 w-4 animate-spin" />
              : <ScanLine className="h-4 w-4" />}
            {busy ? 'Analisando fatias...' : 'Testar PNG/JPG por fatias 1D'}
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff"
              disabled={busy}
              className="hidden"
              onChange={(event) => event.target.files?.[0]
                && void onRasterSlicesFile(event.target.files[0])}
            />
          </label>
          <p className="mt-2 text-[10px] text-cyan-800/70">
            Primeira versão: paredes horizontais e verticais; diagonais entram na próxima etapa.
          </p>
        </div>
        <div className="mt-4 max-w-2xl rounded-2xl border border-violet-200 bg-violet-50/60 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold text-violet-900">
                <Sparkles className="h-4 w-4" /> Testar imagem com Raster2Seq
              </div>
              <p className="mt-1 text-[11px] text-violet-700/80">
                Mantém a imagem no fundo e sobrepõe ambientes, portas e janelas previstos.
              </p>
            </div>
            <label className="flex items-center gap-2 text-xs text-violet-800">
              Largura do canvas
              <input
                type="number"
                min={1}
                max={500}
                step={0.5}
                value={rasterCanvasWidth}
                onChange={(event) => setRasterCanvasWidth(
                  Math.min(500, Math.max(1, +event.target.value || 20)),
                )}
                className="w-20 rounded-lg border border-violet-200 bg-white px-2 py-1.5 text-right"
              />
              m
            </label>
          </div>
          <label className="mt-3 flex cursor-pointer items-center justify-center gap-2 rounded-xl
                            bg-violet-700 px-4 py-3 text-xs font-semibold text-white
                            hover:bg-violet-800 has-[:disabled]:cursor-wait has-[:disabled]:opacity-60">
            {busy
              ? <Loader2 className="h-4 w-4 animate-spin" />
              : <ImagePlus className="h-4 w-4" />}
            {busy ? 'Executando IA local...' : 'Escolher PNG, JPG, BMP ou TIFF'}
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff"
              disabled={busy}
              className="hidden"
              onChange={(event) => event.target.files?.[0]
                && void onRaster2SeqFile(event.target.files[0])}
            />
          </label>
          <p className="mt-2 text-[10px] text-violet-700/70">
            Escala experimental: a largura informada inclui também as margens brancas da imagem.
          </p>
        </div>
        <label className="mt-3 flex max-w-2xl cursor-pointer items-center justify-center gap-2
                          rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs
                          text-slate-600 hover:border-blue-300 hover:bg-slate-50">
          <ImagePlus className="h-4 w-4 text-blue-500" />
          {imagemVinculada
            ? `Raster vinculado: ${imagemVinculada.name}`
            : 'Adicionar PNG/JPG original vinculado ao DWG (opcional)'}
          <input
            type="file"
            accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && void onLinkedImage(e.target.files[0])}
          />
        </label>
        <p className="mt-1 max-w-2xl text-center text-[10px] text-slate-400">
          Selecione o raster original antes do DWG. Capturas recortadas não preservam o alinhamento CAD.
        </p>
        {erro && <ErroBox msg={erro} />}
        </div>
        <GptPlanChatPanel
          provider={planAiProvider}
          status={planAiStatus}
          onProviderChange={changePlanAiProvider}
          modelAvailable={false}
          messages={planChatMessages}
          draft={planChatDraft}
          onDraftChange={setPlanChatDraft}
          busy={planChatBusy}
          candidateAvailable={false}
          onSend={() => void sendGptPlanMessage()}
          onUpload={(file) => void onGptPlanFile(file)}
          onApply={applyGptPlanCandidate}
          onDiscard={() => setPlanChatCandidate(null)}
        />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-8 pt-6">
        <Cabecalho titulo={titulo} subtitulo={subtitulo} />
        {modelo.source && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
            <span className="rounded-full bg-slate-100 px-2 py-1 uppercase font-semibold text-slate-600">
              {modelo.source.format} · {modelo.source.mode}
            </span>
            {modelo.source.pavimento && (
              <span>Pavimento: {modelo.source.pavimento.nome}</span>
            )}
            {(modelo.source.pavimentos_disponiveis?.length ?? 0) > 1 && arquivoFonte && (
              <select
                value={modelo.source.pavimento?.id ?? ''}
                disabled={busy}
                onChange={(e) => trocarPavimento(e.target.value)}
                className="border border-slate-200 rounded-md bg-white px-2 py-1"
              >
                {modelo.source.pavimentos_disponiveis!.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.nome} · {p.n_paredes} paredes · z {p.elevacao.toFixed(2)} m
                  </option>
                ))}
              </select>
            )}
            {(modelo.source.cad_regions?.length ?? 0) > 1 && arquivoFonte && (
              <select
                value={modelo.source.cad_region?.id ?? ''}
                disabled={busy}
                onChange={(event) => void trocarRegiaoCad(event.target.value)}
                className="border border-slate-200 rounded-md bg-white px-2 py-1"
              >
                {modelo.source.cad_regions!.map((region) => (
                  <option key={region.id} value={region.id}>
                    {region.name} · {region.n_walls} paredes · {region.total_length.toFixed(1)} m
                  </option>
                ))}
              </select>
            )}
            {!!modelo.warnings?.length && (
              <span className="text-amber-600" title={modelo.warnings!.join('\n')}>
                {modelo.warnings.length} aviso(s) de importação
              </span>
            )}
            {rasterReference && (
              <span className={`rounded-md px-2 py-1 ${isPreWallReference
                ? 'bg-emerald-50 text-emerald-800'
                : is2DReference
                ? 'bg-amber-50 text-amber-800'
                : isSliceReference
                  ? 'bg-cyan-50 text-cyan-800'
                  : 'bg-violet-50 text-violet-700'}`}>
                {isPreWallReference
                  ? `YOLO pré-paredes: ${modelo.paredes.length} paredes · ${
                    modelo.aberturas.filter((item) => item.tipo === 'door').length
                  } portas · ${
                    modelo.aberturas.filter((item) => item.tipo === 'window').length
                  } janelas`
                  : is2DReference
                  ? `Regiões 2D: ${modelo.paredes.filter((item) => !isColumnWall(item)).length} paredes · ${
                    modelo.paredes.filter(isColumnWall).length
                  } pilares · ${
                    modelo.aberturas.filter((item) => item.tipo === 'door').length
                  } portas · ${
                    modelo.aberturas.filter((item) => item.tipo === 'window').length
                  } janelas`
                  : isSliceReference
                  ? `Fatias 1D: ${modelo.paredes.length} paredes · ${modelo.aberturas.length} aberturas · ${
                    rasterReference.slice_segments?.length ?? 0
                  } faces`
                  : `IA: ${rasterReference.rooms.length} ambientes · ${
                    rasterReference.openings.filter((item) => item.kind === 'door').length
                  } portas · ${
                    rasterReference.openings.filter((item) => item.kind === 'window').length
                  } janelas`}
                {dimensionCandidates.length > 0 && ` · ${dimensionCandidates.length} cotas OCR`}
              </span>
            )}
            {(modelo.source.cad_summary?.ocr_lines ?? 0) > 0 && (
              <span className="rounded-md bg-blue-50 px-2 py-1 text-blue-700">
                OCR: {modelo.source.cad_summary!.ocr_lines} linha(s) alinhada(s)
              </span>
            )}
            {arquivoFonte && (modelo.source.cad_summary?.missing_linked_images ?? 0) > 0 && (
              <label className="flex cursor-pointer items-center gap-1 rounded-md border
                                border-amber-200 bg-amber-50 px-2 py-1 text-amber-700">
                <ImagePlus className="h-3 w-3" />
                Anexar raster original
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && void onLinkedImage(e.target.files[0])}
                />
              </label>
            )}
            {modelo.source.format === 'pdf' && arquivoFonte && (
              <label className="flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1">
                Escala 1:
                <input
                  type="number"
                  min={1}
                  max={1000}
                  value={pdfScale}
                  onChange={(e) => setPdfScale(Math.max(1, +e.target.value || 1))}
                  className="w-16 text-right outline-none"
                />
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void carregarArquivo(arquivoFonte)}
                  className="ml-1 flex items-center gap-1 text-blue-600 hover:text-blue-700 disabled:opacity-40"
                >
                  <RotateCcw className="h-3 w-3" /> Reprocessar
                </button>
              </label>
            )}
          </div>
        )}
      </div>

      <div className="flex items-center justify-between px-8 pt-3">
        <div className="inline-flex rounded-xl border border-slate-200 bg-white p-1 shadow-sm">
          {([
            ['2d', 'Planta 2D'],
            ['3d', 'Modelo 3D'],
            ['split', '2D + 3D'],
          ] as const).map(([mode, label]) => (
            <button
              key={mode}
              type="button"
              onClick={() => setViewMode(mode)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                viewMode === mode
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-500 hover:bg-slate-50 hover:text-slate-800'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <span className="text-[11px] text-slate-400">
          Seleção e propriedades são compartilhadas entre as vistas
        </span>
      </div>

      <div className="flex-1 flex gap-4 px-8 pb-6 pt-3 min-h-0">
        {/* ---------- CANVAS ---------- */}
        <div className={`flex-1 min-w-0 ${viewMode === 'split' ? 'grid grid-cols-2 gap-3' : ''}`}>
        {viewMode !== '3d' && (
        <div className="h-full min-h-[430px] bg-white rounded-xl border border-slate-200 relative overflow-hidden">
          <div className="absolute top-3 left-3 right-24 z-10 flex flex-wrap gap-2">
            {!modoLaje && <button onClick={addParede}
              className="flex items-center gap-1 text-xs px-2.5 py-1.5 bg-white border border-slate-200
                         rounded-lg shadow-sm hover:bg-slate-50">
              <Plus className="w-3.5 h-3.5" /> Parede
            </button>}
            {!modoLaje && <button onClick={addPilar}
              className="flex items-center gap-1 text-xs px-2.5 py-1.5 bg-white border border-violet-200
                         text-violet-700 rounded-lg shadow-sm hover:bg-violet-50">
              <Square className="w-3.5 h-3.5" /> Pilar
            </button>}
            {!modoLaje && sel?.kind === 'parede' && !selectedWallIsColumn && <>
              <button onClick={() => addAbertura('door')}
                className="flex items-center gap-1 text-xs px-2.5 py-1.5 bg-white border border-slate-200 rounded-lg shadow-sm hover:bg-slate-50">
                <DoorOpen className="w-3.5 h-3.5 text-emerald-500" /> Porta
              </button>
              <button onClick={() => addAbertura('window')}
                className="flex items-center gap-1 text-xs px-2.5 py-1.5 bg-white border border-slate-200 rounded-lg shadow-sm hover:bg-slate-50">
                <RectangleHorizontal className="w-3.5 h-3.5 text-sky-500" /> Janela
              </button>
            </>}
            <button onClick={() => {
              setModoLaje((v) => !v);
              setMeasureMode(false);
              setSel(null); setSelVertice(null);
              setSelectedAiRoomId(null); setSelectedAiVertex(null);
            }}
              className={`flex items-center gap-1 text-xs px-2.5 py-1.5 border rounded-lg shadow-sm
                ${modoLaje ? 'bg-blue-600 text-white border-blue-600' : 'bg-white border-slate-200 hover:bg-slate-50'}`}>
              <Layers className="w-3.5 h-3.5" /> {modoLaje ? 'Editando laje' : 'Editar laje'}
            </button>
            <button type="button" onClick={() => {
              setMeasureMode((current) => {
                const next = !current;
                if (next) {
                  setModoLaje(false);
                  setSel(null);
                  setSelVertice(null);
                  setSelectedAiRoomId(null);
                  setSelectedAiVertex(null);
                }
                return next;
              });
            }}
              title="Clique em dois pontos da planta para medir"
              className={`flex items-center gap-1 text-xs px-2.5 py-1.5 border rounded-lg shadow-sm
                ${measureMode
                  ? 'bg-emerald-600 text-white border-emerald-600'
                  : 'bg-white text-emerald-700 border-emerald-200 hover:bg-emerald-50'}`}>
              <Ruler className="w-3.5 h-3.5" /> {measureMode ? 'Medindo…' : 'Medir'}
            </button>
            {(measurementStart || measurementEnd) && (
              <button type="button" onClick={() => {
                setMeasurementStart(null);
                setMeasurementEnd(null);
                setMeasurementCursor(null);
              }}
                title="Limpar medição"
                aria-label="Limpar medição"
                className="flex items-center justify-center rounded-lg border border-slate-200 bg-white p-1.5 text-slate-500 shadow-sm hover:bg-slate-50 hover:text-rose-600">
                <X className="h-3.5 w-3.5" />
              </button>
            )}
            {rasterReference ? <>
              <button onClick={() => setMostrarBackdrop((v) => !v)}
                className={`flex items-center gap-1 text-xs px-2.5 py-1.5 border rounded-lg shadow-sm
                  ${mostrarBackdrop ? 'bg-slate-700 text-white border-slate-700' : 'bg-white border-slate-200 hover:bg-slate-50'}`}>
                <ImagePlus className="w-3.5 h-3.5" /> Imagem original
              </button>
              {!is2DReference && <button onClick={() => {
                if (mostrarAiOverlay) {
                  setSelectedAiRoomId(null); setSelectedAiVertex(null);
                  setSelectedDimensionId(null);
                }
                setMostrarAiOverlay((value) => !value);
              }}
                className={`flex items-center gap-1 text-xs px-2.5 py-1.5 border rounded-lg shadow-sm
                  ${mostrarAiOverlay
                    ? isSliceReference
                      ? 'bg-cyan-700 text-white border-cyan-700'
                      : 'bg-violet-700 text-white border-violet-700'
                    : 'bg-white border-slate-200 hover:bg-slate-50'}`}>
                {isSliceReference
                  ? <ScanLine className="w-3.5 h-3.5" />
                  : <Sparkles className="w-3.5 h-3.5" />}
                {isSliceReference ? 'Faces 1D' : 'Previsão IA'}
              </button>}
              {!isGeometricRasterReference && <button onClick={converterRasterEmParedes}
                title="Esquadreja e une limites próximos; paredes manuais são preservadas"
                className="flex items-center gap-1 text-xs px-2.5 py-1.5 border border-violet-200
                           bg-white text-violet-700 rounded-lg shadow-sm hover:bg-violet-50">
                <Square className="w-3.5 h-3.5" />
                {hasRasterWalls ? 'Atualizar paredes da IA' : 'Criar paredes da IA'}
              </button>}
            </> : (backdropPng || vectorReference) && (
              <button onClick={() => setMostrarBackdrop((v) => !v)}
                className={`flex items-center gap-1 text-xs px-2.5 py-1.5 border rounded-lg shadow-sm
                  ${mostrarBackdrop ? 'bg-slate-700 text-white border-slate-700' : 'bg-white border-slate-200 hover:bg-slate-50'}`}>
                <ImagePlus className="w-3.5 h-3.5" /> {vectorReference?.label ?? 'Planta extraída'}
              </button>
            )}
          </div>

          <button onClick={() => modelo && fitView(modelo)} title="Ajustar zoom à planta"
            className="absolute top-3 right-3 z-10 flex items-center gap-1 text-xs px-2.5 py-1.5
                       bg-white border border-slate-200 rounded-lg shadow-sm hover:bg-slate-50">
            <Maximize className="w-3.5 h-3.5" /> Ajustar
          </button>

          <svg ref={svgRef} viewBox={vb} className="w-full h-full"
               style={{ cursor: measureMode
                 ? 'crosshair'
                 : dragRef.current?.kind === 'pan' ? 'grabbing' : 'default' }}
               onMouseMove={onMouseMove} onMouseUp={endDrag} onMouseLeave={endDrag}
               onMouseDownCapture={placeMeasurementPoint}
               onMouseDown={(e) => {
                 if (!measureMode && e.target === svgRef.current && view)
                   dragRef.current = { kind: 'pan', px: e.clientX, py: e.clientY, vx: view.x, vy: view.y };
               }}
               onClick={(e) => {
                 if (e.target === svgRef.current) {
                   setSel(null); setSelVertice(null);
                   setSelectedAiRoomId(null); setSelectedAiVertex(null);
                   setSelectedDimensionId(null);
                 }
               }}>
            {/* raster alinhado exatamente ao canvas quadrado visto pelo Raster2Seq */}
            {mostrarBackdrop && rasterReference && (
              <image
                href={`data:${rasterReference.image_mime};base64,${rasterReference.image_base64}`}
                x={rasterReference.bounds[0]}
                y={-rasterReference.bounds[3]}
                width={rasterReference.bounds[2] - rasterReference.bounds[0]}
                height={rasterReference.bounds[3] - rasterReference.bounds[1]}
                opacity={0.62}
                preserveAspectRatio="none"
                style={{ pointerEvents: 'none' }}
              />
            )}
            {/* linhas vetoriais originais — referencia imutavel por baixo do BIM */}
            {mostrarBackdrop && vectorReferencePath && (
              <path
                d={vectorReferencePath}
                fill="none"
                stroke="#64748b"
                strokeWidth={escalaTraco * 0.72}
                opacity={0.50}
                style={{ pointerEvents: 'none' }}
              />
            )}
            {/* mapa de densidade da nuvem captada — o cliente edita EM CIMA dela */}
            {mostrarBackdrop && backdropPng && backdropBounds && (
              <image href={`data:image/png;base64,${backdropPng}`}
                     x={backdropBounds[0]} y={-backdropBounds[3]}
                     width={backdropBounds[2] - backdropBounds[0]}
                     height={backdropBounds[3] - backdropBounds[1]}
                     opacity={0.45} preserveAspectRatio="none"
                     style={{ imageRendering: 'pixelated', pointerEvents: 'none' }} />
            )}
            {/* polígonos previstos; viram geometria BIM somente após confirmação */}
            {mostrarAiOverlay && rasterReference && (
              <g>
                <g style={{ pointerEvents: 'none' }}>
                  {rasterReference.slice_segments?.map((segment) => (
                    <line
                      key={segment.id}
                      x1={segment.points[0][0]}
                      y1={-segment.points[0][1]}
                      x2={segment.points[1][0]}
                      y2={-segment.points[1][1]}
                      stroke="#06b6d4"
                      strokeWidth={escalaTraco * 1.25}
                      strokeDasharray={`${escalaTraco * 3} ${escalaTraco * 1.8}`}
                      opacity={0.34 + 0.46 * segment.support}
                    >
                      <title>
                        Face 1D · suporte {(segment.support * 100).toFixed(0)}%
                      </title>
                    </line>
                  ))}
                </g>
                {rasterReference.rooms.map((room) => {
                  const selected = room.id === selectedAiRoomId;
                  const centerX = room.points.reduce((sum, point) => sum + point[0], 0)
                    / room.points.length;
                  const centerY = room.points.reduce((sum, point) => sum + point[1], 0)
                    / room.points.length;
                  return (
                    <g key={`r2s-room-${room.id}`}>
                      <polygon
                        points={room.points.map(([x, y]) => `${x},${-y}`).join(' ')}
                        fill={selected ? 'rgba(124,58,237,0.24)' : 'rgba(124,58,237,0.13)'}
                        stroke={selected ? '#f97316' : '#7c3aed'}
                        strokeWidth={escalaTraco * (selected ? 2.8 : 1.4)}
                        strokeDasharray={room.category_id === 0
                          ? `${escalaTraco * 4} ${escalaTraco * 2}`
                          : undefined}
                        style={{ cursor: 'pointer' }}
                        onMouseDown={(event) => {
                          event.stopPropagation();
                          setSelectedAiRoomId(room.id);
                          setSelectedAiVertex(null);
                          setSel(null);
                          setModoLaje(false);
                        }}
                      >
                        <title>{room.label}</title>
                      </polygon>
                      <text
                        x={centerX}
                        y={-centerY}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fontSize={escalaTraco * 5.5}
                        fontWeight={600}
                        fill="#5b21b6"
                        paintOrder="stroke"
                        stroke="rgba(255,255,255,0.88)"
                        strokeWidth={escalaTraco * 2.5}
                        style={{ pointerEvents: 'none' }}
                      >
                        {room.label}
                      </text>
                      {selected && room.points.map(([x, y], index) => {
                        const [nextX, nextY] = room.points[(index + 1) % room.points.length];
                        const midX = (x + nextX) / 2;
                        const midY = (y + nextY) / 2;
                        return (
                          <g key={`r2s-edit-${room.id}-${index}`}>
                            <g
                              style={{ cursor: 'copy' }}
                              onMouseDown={(event) => {
                                event.stopPropagation();
                                addRasterRoomVertex(room.id, index);
                                setSelectedAiVertex(index + 1);
                              }}
                            >
                              <circle cx={midX} cy={-midY} r={escalaTraco * 3.5}
                                fill="#10b981" opacity={0.9} />
                              <line x1={midX - escalaTraco * 1.7} y1={-midY}
                                x2={midX + escalaTraco * 1.7} y2={-midY}
                                stroke="#fff" strokeWidth={escalaTraco * 0.85} />
                              <line x1={midX} y1={-midY - escalaTraco * 1.7}
                                x2={midX} y2={-midY + escalaTraco * 1.7}
                                stroke="#fff" strokeWidth={escalaTraco * 0.85} />
                            </g>
                            <circle
                              cx={x}
                              cy={-y}
                              r={escalaTraco * (selectedAiVertex === index ? 6 : 4.8)}
                              fill={selectedAiVertex === index ? '#f97316' : '#fff'}
                              stroke="#7c3aed"
                              strokeWidth={escalaTraco * 1.8}
                              style={{ cursor: 'grab' }}
                              onMouseDown={(event) => {
                                event.stopPropagation();
                                setSelectedAiVertex(index);
                                dragRef.current = {
                                  kind: 'raster-room-vertex',
                                  id: room.id,
                                  idx: index,
                                };
                              }}
                              onDoubleClick={(event) => {
                                event.stopPropagation();
                                removeRasterRoomVertex(room.id, index);
                                setSelectedAiVertex(null);
                              }}
                            />
                          </g>
                        );
                      })}
                    </g>
                  );
                })}
                <g style={{ pointerEvents: 'none' }}>
                {rasterReference.openings.map((opening) => (
                  <polyline
                    key={`r2s-opening-${opening.id}`}
                    points={opening.points.map(([x, y]) => `${x},${-y}`).join(' ')}
                    fill="none"
                    stroke={opening.kind === 'door' ? COR_PORTA : COR_JANELA}
                    strokeWidth={escalaTraco * 4}
                    strokeLinecap="round"
                  >
                    <title>{opening.label}</title>
                  </polyline>
                ))}
                </g>
              </g>
            )}
            {/* laje (contorno de piso/teto) — por baixo das paredes */}
            {modelo.laje.contorno.length >= 3 && (
              <polygon
                points={modelo.laje.contorno.map(([x, y]) => `${x},${-y}`).join(' ')}
                fill={modoLaje ? 'rgba(37,99,235,0.10)' : 'rgba(148,163,184,0.08)'}
                stroke={modoLaje ? COR_SEL : '#94a3b8'}
                strokeWidth={escalaTraco * (modoLaje ? 2 : 1)}
                strokeDasharray={`${escalaTraco * 4} ${escalaTraco * 3}`}
                style={{ pointerEvents: 'none' }} />
            )}
            {/* spaces derivados na finalizacao automatica */}
            {!modoLaje && (modelo.spaces ?? []).filter(
              (space) => space.contorno.length >= 3).map((space) => (
              <g key={space.id} style={{ pointerEvents: 'none' }}>
                <polygon
                  points={space.contorno.map(([x, y]) => `${x},${-y}`).join(' ')}
                  fill="rgba(59,130,246,0.10)"
                  stroke="rgba(59,130,246,0.50)"
                  strokeWidth={escalaTraco}
                />
                <text
                  x={space.contorno.reduce((sum, value) => sum + value[0], 0) / space.contorno.length}
                  y={-space.contorno.reduce((sum, value) => sum + value[1], 0) / space.contorno.length}
                  textAnchor="middle"
                  fontSize={escalaTraco * 6.5}
                  fontWeight={600}
                  fill="#2563eb"
                  stroke="#fff"
                  strokeWidth={escalaTraco * 2}
                  paintOrder="stroke"
                >
                  {space.id} · {space.area.toFixed(1)} m²
                </text>
              </g>
            ))}
            {/* modo laje: marcador "+" no meio de cada aresta (adiciona ponto) */}
            {modoLaje && modelo.laje.contorno.map(([x, y], i) => {
              const [nx, ny] = modelo.laje.contorno[(i + 1) % modelo.laje.contorno.length];
              const mx = (x + nx) / 2, my = (y + ny) / 2;
              return (
                <g key={`add${i}`} style={{ cursor: 'copy' }}
                   onMouseDown={(e) => { e.stopPropagation(); addVertice(i); setSelVertice(i + 1); }}>
                  <circle cx={mx} cy={-my} r={escalaTraco * 4} fill="#10b981" opacity={0.9} />
                  <line x1={mx - escalaTraco * 2} y1={-my} x2={mx + escalaTraco * 2} y2={-my}
                        stroke="#fff" strokeWidth={escalaTraco * 0.9} />
                  <line x1={mx} y1={-my - escalaTraco * 2} x2={mx} y2={-my + escalaTraco * 2}
                        stroke="#fff" strokeWidth={escalaTraco * 0.9} />
                </g>
              );
            })}
            {/* modo laje: vertices arrastaveis (clique seleciona; duplo-clique remove) */}
            {modoLaje && modelo.laje.contorno.map(([x, y], i) => {
              const on = selVertice === i;
              return (
                <circle key={`v${i}`} cx={x} cy={-y} r={escalaTraco * (on ? 6.5 : 5)}
                        fill={on ? COR_SEL : '#fff'} stroke={COR_SEL} strokeWidth={escalaTraco * 2}
                        style={{ cursor: 'grab' }}
                        onMouseDown={(e) => { e.stopPropagation(); setSelVertice(i); dragRef.current = { kind: 'laje-vertice', idx: i }; }}
                        onDoubleClick={(e) => { e.stopPropagation(); removeVertice(i); setSelVertice(null); }} />
              );
            })}
            {/* paredes */}
            <g style={{ pointerEvents: modoLaje ? 'none' : 'auto', opacity: modoLaje ? 0.45 : 1 }}>
            {modelo.paredes.map((p) => {
              const on = sel?.kind === 'parede' && sel.id === p.id;
              const column = isColumnWall(p);
              const structural = isStructuralWall(p);
              const mlColor = p.ml_status === 'wall' ? COR_ML_WALL
                : p.ml_status === 'door_leaf' ? COR_ML_LEAF
                : p.ml_status === 'non_wall' ? COR_ML_NON_WALL
                : p.ml_status === 'uncertain' ? COR_ML_UNCERTAIN
                : null;
              return (
                <g key={p.id}>
                  <line x1={p.ax} y1={-p.ay} x2={p.bx} y2={-p.by}
                        stroke={on ? COR_SEL : (mlColor ?? (column ? COR_PILAR : (structural ? COR_PAREDE_ESTRUTURAL : COR_PAREDE)))}
                        strokeWidth={p.espessura}
                        strokeLinecap={column ? 'butt' : 'round'} opacity={column ? 0.92 : 0.85}
                        style={{ cursor: 'move' }}
                        onMouseDown={(e) => {
                          e.stopPropagation(); setSel({ kind: 'parede', id: p.id });
                          const w = worldFromEvent(e);
                          dragRef.current = { kind: 'wall-body', id: p.id, ox: w.x, oy: w.y,
                            ax: p.ax, ay: p.ay, bx: p.bx, by: p.by };
                        }} />
                  {on && [
                    { part: 'P1', kind: 'wall-a' as const, x: p.ax, y: p.ay, color: COR_P1, dy: -1 },
                    { part: 'P2', kind: 'wall-b' as const, x: p.bx, y: p.by, color: COR_P2, dy: 1 },
                  ].map(({ part, kind, x, y, color, dy }) => (
                    <g key={part}>
                      <circle cx={x} cy={-y}
                              r={escalaTraco * 5.5} fill="#fff" stroke={color} strokeWidth={escalaTraco * 2}
                              style={{ cursor: 'pointer' }}
                              onMouseDown={(e) => {
                                e.stopPropagation();
                                beginWallEndpointDrag(kind, p);
                              }} />
                      <text x={x + escalaTraco * 7} y={-y + dy * escalaTraco * 9}
                            fontSize={escalaTraco * 7} fontWeight={700} fill={color}
                            stroke="#fff" strokeWidth={escalaTraco * 2.5}
                            paintOrder="stroke" style={{ pointerEvents: 'none' }}>
                        {p.id}.{part}
                      </text>
                    </g>
                  ))}
                </g>
              );
            })}
            {/* aberturas */}
            {modelo.aberturas.map((a) => {
              const p = paredeById[a.parede_id];
              if (!p) return null;
              const L = Math.hypot(p.bx - p.ax, p.by - p.ay);
              if (L < MIN_WALL_LENGTH) return null;
              const ux = (p.bx - p.ax) / L, uy = (p.by - p.ay) / L;
              const cxw = p.ax + ux * a.s_centro, cyw = p.ay + uy * a.s_centro;
              const x1 = cxw - ux * a.largura / 2, y1 = cyw - uy * a.largura / 2;
              const x2 = cxw + ux * a.largura / 2, y2 = cyw + uy * a.largura / 2;
              const on = sel?.kind === 'abertura' && sel.id === a.id;
              const pendingRemoval = pendingOpeningRemoval?.openings.some(
                (opening) => opening.id === a.id,
              );
              return (
                <g key={a.id}>
                  <line x1={x1} y1={-y1} x2={x2} y2={-y2}
                      stroke={pendingRemoval ? '#ef4444' : (on ? COR_SEL : (a.tipo === 'door' ? COR_PORTA : COR_JANELA))}
                      strokeWidth={p.espessura * 1.25} strokeLinecap="butt"
                      strokeDasharray={pendingRemoval ? `${escalaTraco * 4} ${escalaTraco * 2}` : undefined}
                      style={{ cursor: 'grab' }}
                      onMouseDown={(e) => {
                         e.stopPropagation(); setSel({ kind: 'abertura', id: a.id });
                         dragRef.current = { kind: 'opening', id: a.id, ps: a.s_centro };
                       }} />
                  {on && (
                    <>
                      <circle cx={x1} cy={-y1} r={escalaTraco * 4}
                        fill="#fff" stroke={COR_P1} strokeWidth={escalaTraco * 1.5}
                        style={{ pointerEvents: 'none' }} />
                      <circle cx={x2} cy={-y2} r={escalaTraco * 4}
                        fill="#fff" stroke={COR_P2} strokeWidth={escalaTraco * 1.5}
                        style={{ pointerEvents: 'none' }} />
                      <text x={x1 + escalaTraco * 5} y={-y1 - escalaTraco * 6}
                        fontSize={escalaTraco * 6} fontWeight={700} fill={COR_P1}
                        stroke="#fff" strokeWidth={escalaTraco * 2} paintOrder="stroke"
                        style={{ pointerEvents: 'none' }}>INÍCIO</text>
                      <text x={x2 + escalaTraco * 5} y={-y2 + escalaTraco * 9}
                        fontSize={escalaTraco * 6} fontWeight={700} fill={COR_P2}
                        stroke="#fff" strokeWidth={escalaTraco * 2} paintOrder="stroke"
                        style={{ pointerEvents: 'none' }}>FIM</text>
                    </>
                  )}
                </g>
              );
            })}
            </g>
            {/* cotas OCR: evidência clicável, nunca aplicada sem confirmação */}
            {mostrarAiOverlay && dimensionCandidates.map((dimension) => {
              const selected = dimension.id === selectedDimensionId;
              return (
                <g
                  key={dimension.id}
                  data-ocr-dimension={dimension.id}
                  style={{ cursor: 'pointer' }}
                  onMouseDown={(event) => {
                    event.stopPropagation();
                    setSelectedDimensionId(dimension.id);
                    setSelectedAiRoomId(null);
                    setSelectedAiVertex(null);
                  }}
                >
                  <circle
                    cx={dimension.position.x}
                    cy={-dimension.position.y}
                    r={escalaTraco * (selected ? 6.5 : 5)}
                    fill={selected ? '#f97316' : '#0f766e'}
                    stroke="#fff"
                    strokeWidth={escalaTraco * 2}
                  />
                  <text
                    x={dimension.position.x + escalaTraco * 7}
                    y={-dimension.position.y - escalaTraco * 5}
                    fontSize={escalaTraco * 7}
                    fontWeight={700}
                    fill={selected ? '#c2410c' : '#0f766e'}
                    stroke="#fff"
                    strokeWidth={escalaTraco * 2.5}
                    paintOrder="stroke"
                    style={{ pointerEvents: 'none' }}
                  >
                    {dimension.value_m.toLocaleString('pt-BR', {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 3,
                    })} m
                  </text>
                  <title>{dimension.line_text}</title>
                </g>
              );
            })}
            {measurement && (
              <g style={{ pointerEvents: 'none' }}>
                <line
                  x1={measurement.start.x}
                  y1={-measurement.start.y}
                  x2={measurement.end.x}
                  y2={-measurement.start.y}
                  stroke="#10b981"
                  strokeWidth={escalaTraco}
                  strokeDasharray={`${escalaTraco * 3} ${escalaTraco * 2}`}
                  opacity={0.48}
                />
                <line
                  x1={measurement.end.x}
                  y1={-measurement.start.y}
                  x2={measurement.end.x}
                  y2={-measurement.end.y}
                  stroke="#10b981"
                  strokeWidth={escalaTraco}
                  strokeDasharray={`${escalaTraco * 3} ${escalaTraco * 2}`}
                  opacity={0.48}
                />
                <line
                  x1={measurement.start.x}
                  y1={-measurement.start.y}
                  x2={measurement.end.x}
                  y2={-measurement.end.y}
                  stroke="#059669"
                  strokeWidth={escalaTraco * 2.2}
                  strokeDasharray={measurement.complete
                    ? undefined
                    : `${escalaTraco * 5} ${escalaTraco * 2}`}
                />
                {[measurement.start, measurement.end].map((point, index) => (
                  <g key={`measurement-point-${index}`}>
                    <circle
                      cx={point.x}
                      cy={-point.y}
                      r={escalaTraco * 5}
                      fill="#fff"
                      stroke="#059669"
                      strokeWidth={escalaTraco * 2}
                    />
                    <circle
                      cx={point.x}
                      cy={-point.y}
                      r={escalaTraco * 1.5}
                      fill="#059669"
                    />
                  </g>
                ))}
                <text
                  x={(measurement.start.x + measurement.end.x) / 2}
                  y={-(measurement.start.y + measurement.end.y) / 2 - escalaTraco * 9}
                  textAnchor="middle"
                  fontSize={escalaTraco * 8}
                  fontWeight={800}
                  fill="#047857"
                  stroke="#fff"
                  strokeWidth={escalaTraco * 3}
                  paintOrder="stroke"
                >
                  {measurement.distance.toFixed(3)} m
                </text>
              </g>
            )}
          </svg>

          {measurement && (
            <div className="pointer-events-none absolute bottom-3 right-3 z-10 min-w-36 rounded-xl border border-emerald-200 bg-white/95 px-3 py-2 text-[10px] text-slate-600 shadow-md backdrop-blur-sm">
              <div className="flex items-center gap-1.5 font-semibold text-emerald-800">
                <Ruler className="h-3.5 w-3.5" />
                {measurement.complete ? 'Medição' : 'Escolha o segundo ponto'}
              </div>
              <div className="mt-1 text-base font-bold tabular-nums text-slate-900">
                {measurement.distance.toFixed(3)} m
              </div>
              <div className="mt-0.5 tabular-nums text-slate-500">
                ΔX {Math.abs(measurement.dx).toFixed(3)} m · ΔY {Math.abs(measurement.dy).toFixed(3)} m
              </div>
              {(measurement.start.snapLabel || measurement.end.snapLabel) && (
                <div className="mt-1 max-w-52 truncate text-[9px] text-emerald-700">
                  {measurement.start.snapLabel ?? 'livre'} → {measurement.end.snapLabel ?? 'livre'}
                </div>
              )}
            </div>
          )}

          <div className="absolute bottom-3 left-3 text-[11px] text-slate-400 flex items-center gap-2">
            <Move className="w-3 h-3" />
            {measureMode
              ? measurementStart && !measurementEnd
                ? 'clique no segundo ponto • snap automático em paredes, vãos e laje • Esc cancela'
                : 'clique no primeiro ponto • o terceiro clique inicia uma nova medição • Esc cancela'
              : selectedAiRoom
              ? `${selectedAiRoom.label}: arraste os pontos • “+” adiciona • duplo clique ou Delete remove`
              : modoLaje
              ? 'arraste os vértices • toque no “+” p/ adicionar • Delete remove o ponto • scroll = zoom'
              : rasterReference
                ? isSliceReference
                  ? 'linhas ciano = faces persistentes • linhas laranja = paredes BIM editáveis'
                  : isPreWallReference
                    ? 'fundo = recorte original • laranja = paredes • verde/azul = aberturas editáveis'
                  : 'clique num ambiente roxo para editar • depois use “Criar paredes da IA”'
                : 'arraste para mover • clique para selecionar • Delete = excluir • scroll = zoom • arraste o fundo p/ mover a vista'}
          </div>
        </div>
        )}
        {viewMode !== '2d' && (
          <PlantaEditor3D
            modelo={modelo}
            cfg={cfg}
            selection={sel}
            onSelectionChange={setSel}
            floorPlanPng={backdropPng ?? rasterReference?.image_base64}
            floorPlanBounds={backdropBounds ?? rasterReference?.bounds}
            floorPlanSegments={vectorReference?.segments}
            floorPlanLabel={vectorReference?.label}
            loadPointCloud={loadPointCloud}
            compact={viewMode === 'split'}
          />
        )}
        </div>

        {/* ---------- PAINEL ---------- */}
        <div className="w-72 shrink-0 flex flex-col gap-4 overflow-auto">
          <GptPlanChatPanel
            provider={planAiProvider}
            status={planAiStatus}
            onProviderChange={changePlanAiProvider}
            modelAvailable
            messages={planChatMessages}
            draft={planChatDraft}
            onDraftChange={setPlanChatDraft}
            busy={planChatBusy}
            candidateAvailable={Boolean(planChatCandidate)}
            onSend={() => void sendGptPlanMessage()}
            onUpload={(file) => void onGptPlanFile(file)}
            onApply={applyGptPlanCandidate}
            onDiscard={() => setPlanChatCandidate(null)}
          />
          {dimensionCandidates.length > 0 && (
            <div className="rounded-xl border border-teal-200 bg-teal-50/70 p-3 text-xs">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="flex items-center gap-1.5 font-semibold text-teal-900">
                  <ScanLine className="h-3.5 w-3.5" /> Cotas OCR
                </div>
                <span className="rounded-full bg-white px-2 py-0.5 text-[10px] text-teal-700">
                  {dimensionCandidates.length} candidata(s)
                </span>
              </div>
              <select
                aria-label="Cota OCR"
                value={selectedDimensionId ?? ''}
                onChange={(event) => setSelectedDimensionId(event.target.value || null)}
                className="w-full rounded-lg border border-teal-200 bg-white px-2 py-1.5 text-xs text-slate-700"
              >
                <option value="">Selecione uma cota…</option>
                {dimensionCandidates.map((dimension) => (
                  <option key={dimension.id} value={dimension.id}>
                    {dimension.value_m.toLocaleString('pt-BR', {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 3,
                    })} m · {dimension.line_text}
                  </option>
                ))}
              </select>

              {selectedDimension && (
                <div className="mt-2 space-y-2">
                  <div className="rounded-lg border border-teal-100 bg-white/80 p-2 text-[11px] text-slate-600">
                    <div className="font-medium text-teal-900">Texto: {selectedDimension.line_text}</div>
                    <div>Confiança OCR: {(selectedDimension.confidence * 100).toFixed(0)}%</div>
                  </div>
                  {selectedWallForDimension && selectedWallLength != null
                    && selectedDimensionRatio != null ? (
                    <>
                      <div className="text-[11px] text-slate-600">
                        Parede <b>{selectedWallForDimension.id}</b>: {selectedWallLength.toFixed(3)} m
                        {' → '}{selectedDimension.value_m.toFixed(3)} m
                        <br />Fator global: {selectedDimensionRatio.toFixed(4)}×
                      </div>
                      <button
                        type="button"
                        onClick={calibrarPlantaComCota}
                        className="w-full rounded-lg bg-teal-700 px-2 py-1.5 font-medium text-white hover:bg-teal-800"
                      >
                        Calibrar planta inteira
                      </button>
                      <button
                        type="button"
                        onClick={aplicarCotaNaParede}
                        className="w-full rounded-lg border border-teal-300 bg-white px-2 py-1.5 font-medium text-teal-800 hover:bg-teal-100"
                      >
                        Ajustar somente esta parede
                      </button>
                    </>
                  ) : (
                    <p className="text-[11px] leading-relaxed text-teal-800">
                      Agora selecione a parede medida por essa cota. Nada é alterado automaticamente.
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
          {modoLaje
            ? <PainelLaje laje={modelo.laje} patchFace={patchFace}
                selVertice={selVertice}
                onRemoverPonto={() => { if (selVertice !== null) { removeVertice(selVertice); setSelVertice(null); } }} />
            : <PainelSelecionado
                sel={sel} modelo={modelo} cfg={cfg} paredeById={paredeById}
                patchParede={patchParede}
                patchWallGeometry={patchWallGeometry}
                setWallLength={setWallLength}
                setWallAngle={setWallAngle}
                patchAbertura={patchAbertura}
                duplicateOpening={duplicateOpening}
                apagar={apagarSel} />}

          {modelo.source?.cad_layers && arquivoFonte && (
            <PainelCadLayers
              layers={modelo.source.cad_layers}
              summary={modelo.source.cad_summary}
              values={cadLayerMap}
              busy={busy}
              onChange={setCadLayerRole}
              onApply={() => void carregarArquivo(
                arquivoFonte,
                modelo.source?.pavimento?.id,
                cadLayerMap,
                modelo.source?.cad_region?.id,
              )}
            />
          )}

          <PainelConfig cfg={cfg} setCfg={setCfg} />

          {onGerar && ocultarPreviewPly && (
            <>
              <input
                ref={recoveryInputRef}
                type="file"
                accept=".ifc"
                className="hidden"
                onChange={(event) => {
                  const file = event.currentTarget.files?.[0];
                  event.currentTarget.value = '';
                  if (file) void recuperarIfcNoEditor(file);
                }}
              />
              <button
                type="button"
                onClick={() => recoveryInputRef.current?.click()}
                disabled={busy}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg
                           text-xs font-medium border border-blue-300 text-blue-700 bg-blue-50
                           hover:bg-blue-100 disabled:opacity-40 transition-colors">
                <FileUp className="w-3.5 h-3.5" /> Reabrir IFC no editor
              </button>
            </>
          )}

          {avisoRecuperacao && (
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-[11px] text-emerald-800">
              {avisoRecuperacao}
            </div>
          )}

          {finalizationSummary && finalizacaoAtual && (
            <div className={`rounded-xl border px-3 py-3 text-[11px] ${
              aprovacaoClienteAtual
                ? 'border-emerald-200 bg-emerald-50 text-emerald-900'
                : 'border-blue-200 bg-blue-50 text-blue-900'
            }`}>
              <div className="flex items-center gap-2 font-semibold">
                <CheckCircle2 className="h-4 w-4" />
                {aprovacaoClienteAtual ? 'Projeto confirmado pelo cliente' : 'Prévia pronta para confirmação'}
              </div>
              {approvalSummary && (
                <div className="mt-2 grid grid-cols-2 gap-1 rounded-lg bg-white/70 p-2 text-[10px]">
                  <span>{approvalSummary.walls} paredes</span>
                  <span>{approvalSummary.columns} pilares</span>
                  <span>{approvalSummary.doors} portas</span>
                  <span>{approvalSummary.windows} janelas</span>
                  <span>Laje: {approvalSummary.slabArea.toFixed(2)} m²</span>
                  <span>{approvalSummary.scale}</span>
                </div>
              )}
              <div className="mt-2 text-[10px] opacity-80">
                Prévia {finalizationSummary.revision} · {finalizationSummary.junctions} junções L/T ·{' '}
                {finalizationSummary.movedEndpoints} extremidades ajustadas ·{' '}
                {finalizationSummary.spaces} ambientes
              </div>
              {finalizationSummary.slabAdjusted && (
                <div className="mt-1 text-[10px] text-amber-700">
                  A laje automática foi ampliada para conter todas as faces das paredes; confira o contorno.
                </div>
              )}
              {!aprovacaoClienteAtual && (
                <label className="mt-3 flex cursor-pointer items-start gap-2 rounded-lg border border-blue-200 bg-white px-2.5 py-2 text-[10px] leading-4 text-slate-700">
                  <input
                    type="checkbox"
                    checked={clientApprovalChecked}
                    onChange={(event) => setClientApprovalChecked(event.target.checked)}
                    className="mt-0.5 h-3.5 w-3.5 rounded border-slate-300 text-blue-600"
                  />
                  Confirmo que revisei paredes, pilares, portas, janelas, laje e escala desta versão.
                </label>
              )}
              <div>
                {aprovacaoClienteAtual && clientApprovedAt
                  ? `Aprovado em ${new Date(clientApprovedAt).toLocaleString('pt-BR')}.`
                  : 'IFC e DXF permanecem bloqueados até a confirmação.'}
              </div>
              {finalizationSummary.blockedByOpenings > 0 && (
                <div className="mt-1 text-amber-700">
                  {finalizationSummary.blockedByOpenings} junção(ões) preservada(s) por causa de portas/janelas.
                </div>
              )}
            </div>
          )}
          <button
            onClick={() => {
              if (!finalizacaoAtual) void finalizarAutomaticamente();
              else if (!aprovacaoClienteAtual) confirmarAprovacaoCliente();
              else void gerar(true);
            }}
            disabled={busy || (finalizacaoAtual && !aprovacaoClienteAtual && !clientApprovalChecked)}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg
                       text-sm font-semibold bg-blue-600 text-white hover:bg-blue-700
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            {busy ? <Loader2 className="w-4 h-4 animate-spin" />
              : aprovacaoClienteAtual ? <Download className="w-4 h-4" />
                : <CheckCircle2 className="w-4 h-4" />}
            {busy
              ? 'Processando…'
              : !finalizacaoAtual
                ? 'Revisar geometria final'
                : aprovacaoClienteAtual
                  ? 'Gerar IFC aprovado'
                  : 'Confirmar projeto revisado'}
          </button>

          <button
            type="button"
            onClick={() => void gerar(false)}
            disabled={busy || !aprovacaoClienteAtual}
            title={aprovacaoClienteAtual ? undefined : 'Confirme a revisão do cliente primeiro'}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg
                       text-xs font-medium border border-slate-300 text-slate-600 bg-white
                       hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            Gerar IFC exatamente como desenhado
          </button>

          <button
            type="button"
            onClick={() => void gerarDxf()}
            disabled={busy || !aprovacaoClienteAtual}
            title={aprovacaoClienteAtual ? undefined : 'Confirme a revisão do cliente primeiro'}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg
                       text-xs font-medium border border-emerald-300 text-emerald-700 bg-emerald-50
                       hover:bg-emerald-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            <Download className="h-3.5 w-3.5" /> Gerar DXF aprovado
          </button>
          {finalizationSummary && !finalizacaoAtual && (
            <div className="rounded-lg bg-amber-50 px-3 py-2 text-[11px] text-amber-800">
              O modelo mudou depois da prévia final. Finalize novamente para atualizar
              junções, aberturas, spaces e forro. A laje permanece como editada.
            </div>
          )}
          {clientApprovedFingerprint && !aprovacaoClienteAtual && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-800">
              A geometria ou a configuração mudou depois da aprovação. As exportações foram
              bloqueadas até uma nova revisão e confirmação do cliente.
            </div>
          )}

          {resultado && (
            <div className="flex flex-col gap-2 bg-emerald-50 rounded-lg px-3 py-3 text-xs">
              <span className="flex items-center gap-2 text-emerald-700 font-medium">
                <CheckCircle2 className="w-4 h-4" /> IFC gerado e download iniciado
              </span>
              <button type="button"
                onClick={() => baixarIfc(resultado, modelo.nome || 'planta')}
                className="flex items-center gap-1 justify-center px-2 py-1.5 bg-emerald-600 text-white rounded-md hover:bg-emerald-700">
                <Download className="w-3.5 h-3.5" /> Baixar IFC novamente
              </button>
              {resultado.ifc_token && resultado.ready_for_comparison && onReferenciaPronta && (
                <button type="button"
                  onClick={() => onReferenciaPronta(resultado)}
                  className="flex items-center gap-1 justify-center px-2 py-1.5 bg-blue-600 text-white rounded-md hover:bg-blue-700">
                  <ScanLine className="w-3.5 h-3.5" /> Comparar com nuvem de pontos
                </button>
              )}
              {!ocultarPreviewPly && resultado.preview_url && (
                <a href={downloadUrl(resultado.preview_url)} download
                   className="flex items-center gap-1 justify-center px-2 py-1.5 bg-slate-600 text-white rounded-md hover:bg-slate-700">
                  <Download className="w-3.5 h-3.5" /> Preview PLY (CloudCompare)
                </a>
              )}
            </div>
          )}
          {erro && <ErroBox msg={erro} />}

          <button
            onClick={() => {
              if (onVoltar) { onVoltar(); return; }
              setModelo(null); setSel(null); setResultado(null);
              setFinalizedFingerprint(null); setFinalizationSummary(null);
              setClientApprovedFingerprint(null); setClientApprovedAt(null);
              setClientApprovalChecked(false);
              setSelectedAiRoomId(null); setSelectedAiVertex(null);
              setSelectedDimensionId(null);
            }}
            className="flex items-center justify-center gap-1 text-xs text-slate-400 hover:text-slate-600">
            <RotateCcw className="w-3 h-3" /> {rotuloVoltar ?? 'Trocar planta'}
          </button>
        </div>
      </div>

      {pendingOpeningRemoval && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/35 px-4"
             role="dialog" aria-modal="true" aria-labelledby="opening-removal-title">
          <div className="w-full max-w-md rounded-2xl border border-rose-200 bg-white p-5 shadow-2xl">
            <div className="flex items-start gap-3">
              <div className="rounded-full bg-rose-50 p-2 text-rose-600">
                <AlertCircle className="h-5 w-5" />
              </div>
              <div className="min-w-0 flex-1">
                <h2 id="opening-removal-title" className="font-semibold text-slate-900">
                  A parede atravessou uma abertura
                </h2>
                <p className="mt-1 text-sm text-slate-600">
                  Ao confirmar o novo comprimento de {pendingOpeningRemoval.wallId},
                  {' '}{pendingOpeningRemoval.openings.length} porta(s) ou janela(s) serão removidas.
                </p>
                <ul className="mt-3 max-h-32 space-y-1 overflow-auto rounded-lg bg-slate-50 p-2 text-xs text-slate-600">
                  {pendingOpeningRemoval.openings.map((opening) => (
                    <li key={opening.id} className="flex items-center justify-between gap-3">
                      <span className="font-medium">{opening.nome || opening.id}</span>
                      <span>{opening.tipo === 'door' ? 'Porta' : 'Janela'} · {opening.largura.toFixed(2)} m</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <div className="mt-5 flex justify-end gap-2">
              <button type="button" onClick={cancelOpeningRemoval}
                className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50">
                Cancelar alteração
              </button>
              <button type="button" onClick={confirmOpeningRemoval}
                className="flex items-center gap-1.5 rounded-lg bg-rose-600 px-3 py-2 text-sm font-semibold text-white hover:bg-rose-700">
                <Trash2 className="h-4 w-4" /> Confirmar e remover
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ---------- sub-componentes ----------
const Cabecalho: React.FC<{ titulo?: string; subtitulo?: string }> = ({ titulo, subtitulo }) => (
  <div>
    <h1 className="text-xl font-bold text-slate-800">{titulo ?? 'Geometria → BIM'}</h1>
    <p className="text-sm text-slate-500">
      {subtitulo ?? 'Importe IFC, IFCZIP, DXF, DWG, SVG ou PDF; revise e gere um IFC editável.'}
    </p>
  </div>
);

const ErroBox: React.FC<{ msg: string }> = ({ msg }) => (
  <div className="flex items-center gap-2 text-xs text-rose-600 bg-rose-50 rounded-lg px-3 py-2 max-w-2xl">
    <AlertCircle className="w-4 h-4 shrink-0" /> {msg}
  </div>
);

const GptPlanChatPanel: React.FC<{
  provider: PlanAiProvider;
  status: PlanAiStatus | null;
  onProviderChange: (provider: PlanAiProvider) => void;
  modelAvailable: boolean;
  messages: PlanChatMessage[];
  draft: string;
  onDraftChange: (value: string) => void;
  busy: boolean;
  candidateAvailable: boolean;
  onSend: () => void;
  onUpload: (file: File) => void;
  onApply: () => void;
  onDiscard: () => void;
}> = ({
  provider, status, onProviderChange, modelAvailable, messages, draft, onDraftChange, busy,
  candidateAvailable,
  onSend, onUpload, onApply, onDiscard,
}) => (
  <div className="rounded-2xl border border-indigo-200 bg-gradient-to-b from-indigo-50/80 to-white p-4 shadow-sm">
    <div className="flex items-start justify-between gap-3">
      <div>
        <div className="flex items-center gap-2 text-sm font-semibold text-indigo-950">
          <span className="rounded-lg bg-indigo-600 p-1.5 text-white">
            <Bot className="h-4 w-4" />
          </span>
          Copiloto visual Plan-to-BIM
        </div>
        <p className="mt-1.5 text-[10px] leading-4 text-indigo-800/75">
          Cotas + perspectiva + detector 2D + revisão de paredes-mãe e vãos.
        </p>
      </div>
      <span className="rounded-full border border-indigo-200 bg-white px-2 py-1 text-[9px] font-semibold text-indigo-700">
        {status?.providers?.[provider]?.configured === false ? 'Sem chave' : 'API no backend'}
      </span>
    </div>

    <div className="mt-3 grid grid-cols-2 gap-2 rounded-xl border border-indigo-100 bg-white/80 p-1.5">
      {([{
        id: 'deepseek' as const,
        label: 'DeepSeek V4',
        note: 'Visão + auditoria',
      }, {
        id: 'openai' as const,
        label: 'GPT-5.5',
        note: 'Visão original',
      }]).map((option) => {
        const configured = status?.providers?.[option.id]?.configured;
        const selected = provider === option.id;
        return (
          <button
            key={option.id}
            type="button"
            disabled={busy || configured === false}
            onClick={() => onProviderChange(option.id)}
            title={configured === false
              ? `Chave de ${option.label} não configurada no backend`
              : `Usar ${option.label}`}
            className={`rounded-lg border px-2 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-45 ${
              selected
                ? 'border-indigo-500 bg-indigo-600 text-white shadow-sm'
                : 'border-transparent bg-white text-slate-600 hover:border-indigo-200 hover:bg-indigo-50'
            }`}
          >
            <span className="block text-[10px] font-bold">{option.label}</span>
            <span className={`mt-0.5 block text-[8px] ${selected ? 'text-indigo-100' : 'text-slate-400'}`}>
              {configured === false ? 'chave ausente' : option.note}
            </span>
          </button>
        );
      })}
    </div>

    <div className={`mt-3 space-y-2 overflow-auto rounded-xl border border-indigo-100 bg-white/80 p-2 ${
      modelAvailable ? 'max-h-52' : 'max-h-64'
    }`}>
      {messages.map((message) => (
        <div key={message.id}
          className={`rounded-lg px-2.5 py-2 text-[10px] leading-4 ${
            message.role === 'user'
              ? 'ml-5 bg-indigo-600 text-white'
              : 'mr-3 border border-slate-100 bg-slate-50 text-slate-600'
          }`}>
          {message.text}
        </div>
      ))}
      {busy && (
        <div className="mr-3 flex items-center gap-2 rounded-lg bg-slate-50 px-2.5 py-2 text-[10px] text-slate-500">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-600" />
          {modelAvailable ? 'Revisando imagem e geometria…' : 'Calibrando, retificando e rodando o 2D…'}
        </div>
      )}
    </div>

    {candidateAvailable && (
      <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-2.5">
        <div className="text-[10px] font-semibold text-amber-900">Revisão pronta para comparar</div>
        <p className="mt-1 text-[9px] leading-4 text-amber-800/80">
          Nada foi exportado. Aplique a proposta para trocar a geometria do editor.
        </p>
        <div className="mt-2 flex gap-2">
          <button type="button" onClick={onApply}
            className="flex-1 rounded-lg bg-amber-600 px-2 py-1.5 text-[10px] font-semibold text-white hover:bg-amber-700">
            Aplicar revisão
          </button>
          <button type="button" onClick={onDiscard} title="Descartar proposta"
            className="rounded-lg border border-amber-200 bg-white px-2 text-amber-700 hover:bg-amber-100">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    )}

    <textarea
      value={draft}
      onChange={(event) => onDraftChange(event.target.value)}
      onKeyDown={(event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') onSend();
      }}
      rows={modelAvailable ? 3 : 4}
      maxLength={6000}
      placeholder="Ex.: faltaram duas janelas na parede W5…"
      className="mt-3 w-full resize-none rounded-xl border border-indigo-200 bg-white px-3 py-2 text-[11px] leading-4 text-slate-700 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
    />

    {modelAvailable ? (
      <button type="button" onClick={onSend} disabled={busy || !draft.trim()}
        className="mt-2 flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50">
        {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <SendHorizontal className="h-4 w-4" />}
        {busy ? 'Analisando…' : 'Enviar para revisão visual'}
      </button>
    ) : (
      <label className={`mt-2 flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl px-3 py-2.5 text-xs font-semibold text-white ${
        busy ? 'cursor-wait bg-indigo-400' : 'bg-indigo-600 hover:bg-indigo-700'
      }`}>
        {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ImagePlus className="h-4 w-4" />}
        {busy ? 'Processando a planta…' : 'Enviar planta e iniciar análise'}
        <input type="file" accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff" disabled={busy}
          className="hidden" onChange={(event) => {
            const file = event.currentTarget.files?.[0];
            event.currentTarget.value = '';
            if (file) onUpload(file);
          }} />
      </label>
    )}
    <p className="mt-2 text-[9px] leading-4 text-slate-400">
      A chave nunca é enviada ao navegador. A proposta precisa de aprovação antes do IFC/DXF.
    </p>
  </div>
);

const Campo: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <label className="flex items-center justify-between text-xs text-slate-600 gap-2">
    {label} {children}
  </label>
);
const numInput = "w-20 border border-slate-200 rounded px-2 py-1 text-right text-xs";

const MetricInput: React.FC<{
  value: number;
  onCommit: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
}> = ({ value, onCommit, min, max, step = 0.01, className = numInput }) => {
  const [draft, setDraft] = useState(String(Number(value.toFixed(4))));
  useEffect(() => setDraft(String(Number(value.toFixed(4)))), [value]);
  const commit = () => {
    let next = Number(draft.replace(',', '.'));
    if (!Number.isFinite(next)) {
      setDraft(String(Number(value.toFixed(4))));
      return;
    }
    if (min !== undefined) next = Math.max(min, next);
    if (max !== undefined) next = Math.min(max, next);
    setDraft(String(Number(next.toFixed(4))));
    onCommit(next);
  };
  return (
    <input
      type="number"
      value={draft}
      min={min}
      max={max}
      step={step}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') event.currentTarget.blur();
        if (event.key === 'Escape') {
          setDraft(String(Number(value.toFixed(4))));
          event.currentTarget.blur();
        }
      }}
      className={className}
    />
  );
};

const PainelSelecionado: React.FC<{
  sel: Sel; modelo: ModeloPlanta; cfg: PlantaConfig; paredeById: Record<string, Parede>;
  patchParede: (id: string, p: Partial<Parede>) => void;
  patchWallGeometry: (id: string, p: Partial<Parede>) => void;
  setWallLength: (id: string, length: number) => void;
  setWallAngle: (id: string, degrees: number) => void;
  patchAbertura: (id: string, p: Partial<Abertura>) => void;
  duplicateOpening: (id: string) => void;
  apagar: () => void;
}> = ({
  sel, modelo, cfg, paredeById, patchParede, patchWallGeometry,
  setWallLength, setWallAngle, patchAbertura, duplicateOpening, apagar,
}) => {
  if (!sel) return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 text-xs text-slate-500">
      <div className="flex gap-4 mb-2 font-medium text-slate-700">
        <span>{modelo.paredes.filter((item) => !isColumnWall(item)).length} paredes</span>
        <span>{modelo.paredes.filter(isColumnWall).length} pilares</span>
        <span>{modelo.aberturas.length} esquadrias</span>
      </div>
      Clique numa parede, pilar ou esquadria para editar.
      <div className="mt-2 text-[11px] text-slate-400">
        {modelo.single_line ? 'Modo linha-única (espessura default)' : 'Espessura medida do desenho'}
      </div>
    </div>
  );
  if (sel.kind === 'parede') {
    const p = paredeById[sel.id];
    if (!p) return null;
    const L = Math.hypot(p.bx - p.ax, p.by - p.ay);
    const angle = Math.atan2(p.by - p.ay, p.bx - p.ax) * 180 / Math.PI;
    const elevation = p.elevacao ?? modelo.source?.pavimento?.elevacao ?? 0;
    const column = isColumnWall(p);
    const structural = isStructuralWall(p);
    return (
      <div className={`bg-white rounded-xl border p-4 flex flex-col gap-3 ${column ? 'border-violet-300' : 'border-blue-200'}`}>
        <div className="flex items-center justify-between">
          <div>
            <span className="text-sm font-semibold text-slate-800">{p.nome || p.id}</span>
            <div className={`text-[9px] font-semibold uppercase tracking-wide ${column ? 'text-violet-600' : 'text-blue-600'}`}>
              {column ? 'Este pilar' : (structural ? 'Esta parede estrutural' : 'Esta parede')}
            </div>
          </div>
          <button onClick={apagar} title={`Excluir ${column ? 'pilar' : 'parede'} (Delete)`}
            aria-label={`Excluir ${column ? 'pilar' : 'parede'}`}
            className="text-rose-500 hover:text-rose-700"><Trash2 className="w-4 h-4" /></button>
        </div>
        <Campo label="Nome">
          <input value={p.nome ?? ''} placeholder={p.id}
            onChange={(event) => patchParede(p.id, { nome: event.target.value })}
            className="w-36 rounded border border-slate-200 px-2 py-1 text-right text-xs" />
        </Campo>
        <Campo label="Classificação">
          <select
            value={column ? 'column' : (structural ? 'structural-wall' : 'wall')}
            onChange={(event) => {
              const value = event.target.value;
              patchParede(p.id, value === 'column'
                ? { tipo: 'column', ifc_class: 'IfcColumn', layer: 'Column-Editada' }
                : {
                    tipo: value,
                    ifc_class: 'IfcWall',
                    layer: value === 'structural-wall' ? 'Wall-Structural-Editada' : 'Wall-Editada',
                  });
            }}
            className="w-36 rounded border border-slate-200 bg-white px-2 py-1 text-right text-xs"
          >
            <option value="wall">Parede</option>
            <option value="structural-wall">Parede estrutural</option>
            <option value="column" disabled={modelo.aberturas.some((item) => item.parede_id === p.id)}>
              Pilar{modelo.aberturas.some((item) => item.parede_id === p.id) ? ' (remova as esquadrias)' : ''}
            </option>
          </select>
        </Campo>
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5">
          <div className="mb-2 text-[11px] font-semibold text-slate-700">Eixo e extremidades</div>
          <div className="grid grid-cols-[30px_1fr_1fr] items-center gap-1.5 text-[10px] text-slate-500">
            <span /> <span className="text-center">X (m)</span><span className="text-center">Y (m)</span>
            <span className="font-bold text-green-700">P1</span>
            <MetricInput value={p.ax} onCommit={(value) => patchWallGeometry(p.id, { ax: value })}
              className="w-full rounded border border-green-200 bg-white px-1.5 py-1 text-right text-[10px]" />
            <MetricInput value={p.ay} onCommit={(value) => patchWallGeometry(p.id, { ay: value })}
              className="w-full rounded border border-green-200 bg-white px-1.5 py-1 text-right text-[10px]" />
            <span className="font-bold text-violet-700">P2</span>
            <MetricInput value={p.bx} onCommit={(value) => patchWallGeometry(p.id, { bx: value })}
              className="w-full rounded border border-violet-200 bg-white px-1.5 py-1 text-right text-[10px]" />
            <MetricInput value={p.by} onCommit={(value) => patchWallGeometry(p.id, { by: value })}
              className="w-full rounded border border-violet-200 bg-white px-1.5 py-1 text-right text-[10px]" />
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 border-t border-slate-200 pt-2">
            <label className="text-[10px] text-slate-500">{column ? 'Largura no eixo (m)' : 'Comprimento (m)'}
              <MetricInput value={L} min={0.05} onCommit={(value) => setWallLength(p.id, value)}
                className="mt-1 w-full rounded border border-slate-200 bg-white px-2 py-1 text-right text-xs" />
            </label>
            <label className="text-[10px] text-slate-500">Ângulo (graus)
              <MetricInput value={angle} step={0.1} onCommit={(value) => setWallAngle(p.id, value)}
                className="mt-1 w-full rounded border border-slate-200 bg-white px-2 py-1 text-right text-xs" />
            </label>
          </div>
        </div>
        <Campo label="Espessura (m)">
          <MetricInput value={p.espessura} min={0.03} max={column ? 1.5 : 0.8}
            onCommit={(value) => patchParede(p.id, { espessura: value })} />
        </Campo>
        <Campo label="Altura individual (m)">
          <MetricInput value={p.altura ?? cfg.altura} min={0.1} step={0.05}
            onCommit={(value) => patchParede(p.id, { altura: value })} />
        </Campo>
        <Campo label="Cota da base (m)">
          <MetricInput value={elevation} step={0.05}
            onCommit={(value) => patchParede(p.id, { elevacao: value })} />
        </Campo>
        <div className="text-[11px] text-slate-400">ID {p.id} · layer {p.layer || '—'}</div>
        {p.ml_status && (
          <div className={`rounded-md px-2 py-1.5 text-[10px]
            ${p.ml_status === 'wall' ? 'bg-emerald-50 text-emerald-700'
              : p.ml_status === 'door_leaf' ? 'bg-amber-50 text-amber-700'
              : p.ml_status === 'non_wall' ? 'bg-rose-50 text-rose-700'
              : 'bg-slate-100 text-slate-600'}`}>
            Revisão ML: {p.ml_status === 'wall' ? 'parede'
              : p.ml_status === 'door_leaf' ? 'possível folha de porta'
              : p.ml_status === 'non_wall' ? 'possível falso candidato'
              : 'incerto'}
            {p.ml_probability !== undefined
              ? ` · ${(p.ml_probability * 100).toFixed(0)}%`
              : ''}
            {' · confirmação humana obrigatória'}
          </div>
        )}
        {p.guid && <div className="text-[10px] text-slate-400 break-all">IFC GUID: {p.guid}</div>}
      </div>
    );
  }
  const a = modelo.aberturas.find((x) => x.id === sel.id);
  if (!a) return null;
  const wall = paredeById[a.parede_id];
  if (!wall) return null;
  const wallLength = Math.hypot(wall.bx - wall.ax, wall.by - wall.ay);
  const height = a.altura ?? (a.tipo === 'door' ? cfg.porta_altura : cfg.janela_altura);
  const sill = a.tipo === 'door' ? 0 : (a.peitoril ?? cfg.janela_peitoril);
  const start = a.s_centro - a.largura / 2;
  const end = a.s_centro + a.largura / 2;
  return (
    <div className="bg-white rounded-xl border border-blue-200 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-sm font-semibold text-slate-800">{a.nome || a.id}</span>
          <div className="text-[9px] font-semibold uppercase tracking-wide text-blue-600">Somente esta instância</div>
        </div>
        <button onClick={apagar} title="Excluir abertura (Delete)" aria-label="Excluir abertura"
          className="text-rose-500 hover:text-rose-700"><Trash2 className="w-4 h-4" /></button>
      </div>
      <Campo label="Nome">
        <input value={a.nome ?? ''} placeholder={a.id}
          onChange={(event) => patchAbertura(a.id, { nome: event.target.value })}
          className="w-36 rounded border border-slate-200 px-2 py-1 text-right text-xs" />
      </Campo>
      <div className="flex gap-2">
        {(['door', 'window'] as const).map((t) => (
          <button key={t} onClick={() => patchAbertura(a.id, { tipo: t })}
            className={`flex-1 flex items-center justify-center gap-1 px-2 py-1.5 rounded-lg text-xs border
              ${a.tipo === t ? 'bg-blue-50 border-blue-300 text-blue-700' : 'border-slate-200 text-slate-500'}`}>
            {t === 'door' ? <DoorOpen className="w-3.5 h-3.5" /> : <RectangleHorizontal className="w-3.5 h-3.5" />}
            {t === 'door' ? 'Porta' : 'Janela'}
          </button>
        ))}
      </div>
      <label className="text-[10px] font-medium text-slate-600">
        Parede hospedeira
        <select value={a.parede_id}
          onChange={(event) => patchAbertura(a.id, { parede_id: event.target.value })}
          className="mt-1 w-full rounded border border-slate-200 bg-white px-2 py-1.5 text-xs">
          {modelo.paredes.filter((item) => !isColumnWall(item)).map((item) => (
            <option key={item.id} value={item.id}>{item.nome || item.id}</option>
          ))}
        </select>
      </label>
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5">
        <div className="mb-2 text-[11px] font-semibold text-slate-700">Posição desde P1</div>
        <div className="grid grid-cols-3 gap-1.5">
          <label className="text-[9px] text-slate-500">Início (m)
            <MetricInput value={start} min={0} max={Math.max(0, wallLength - a.largura)}
              onCommit={(value) => patchAbertura(a.id, { s_centro: value + a.largura / 2 })}
              className="mt-1 w-full rounded border border-slate-200 bg-white px-1 py-1 text-right text-[10px]" />
          </label>
          <label className="text-[9px] text-slate-500">Centro (m)
            <MetricInput value={a.s_centro} min={a.largura / 2}
              max={Math.max(a.largura / 2, wallLength - a.largura / 2)}
              onCommit={(value) => patchAbertura(a.id, { s_centro: value })}
              className="mt-1 w-full rounded border border-slate-200 bg-white px-1 py-1 text-right text-[10px]" />
          </label>
          <label className="text-[9px] text-slate-500">Fim (m)
            <MetricInput value={end} min={a.largura} max={wallLength}
              onCommit={(value) => patchAbertura(a.id, { s_centro: value - a.largura / 2 })}
              className="mt-1 w-full rounded border border-slate-200 bg-white px-1 py-1 text-right text-[10px]" />
          </label>
        </div>
        <div className="mt-1.5 text-[9px] text-slate-400">
          Referência: {wall.id}.P1 → {wall.id}.P2 · parede {wallLength.toFixed(2)} m
        </div>
      </div>
      <Campo label="Largura (m)">
        <MetricInput value={a.largura} min={0.05} max={wallLength} step={0.05}
          onCommit={(value) => patchAbertura(a.id, { largura: value })} />
      </Campo>
      <Campo label="Altura individual (m)">
        <MetricInput value={height} min={0.05} step={0.05}
          onCommit={(value) => patchAbertura(a.id, { altura: value })} />
      </Campo>
      {a.tipo === 'window' && (
        <Campo label="Peitoril (m)">
          <MetricInput value={sill} min={0} step={0.05}
            onCommit={(value) => patchAbertura(a.id, { peitoril: value })} />
        </Campo>
      )}
      <Campo label="Topo do vão (m)">
        <MetricInput value={sill + height} min={sill + 0.05} step={0.05}
          onCommit={(value) => patchAbertura(a.id, { altura: value - sill })} />
      </Campo>
      <button type="button" onClick={() => duplicateOpening(a.id)}
        className="flex items-center justify-center gap-1.5 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-semibold text-blue-700 hover:bg-blue-100">
        <Copy className="h-3.5 w-3.5" /> Duplicar com estas medidas
      </button>
      <div className="text-[10px] text-slate-400">
        A cópia recebe outro ID e fica selecionada. Depois escolha a parede hospedeira e informe o novo início.
      </div>
      {a.origem && (
        <div className="rounded-md bg-slate-50 px-2 py-1.5 text-[10px] text-slate-500">
          Origem: {a.origem}
          {a.source_layer ? ` · layer ${a.source_layer}` : ''}
          {a.block_name ? ` · bloco ${a.block_name}` : ''}
          {a.confidence !== undefined
            ? ` · confiança ${(a.confidence * 100).toFixed(0)}%`
            : ''}
        </div>
      )}
      {a.guid && <div className="text-[10px] text-slate-400 break-all">IFC GUID: {a.guid}</div>}
    </div>
  );
};

const CAD_ROLE_LABELS: Record<CadRole | 'auto', string> = {
  auto: 'Automático',
  wall: 'Parede',
  door: 'Porta',
  window: 'Janela',
  opening: 'Abertura',
  ignore: 'Ignorar',
};

const PainelCadLayers: React.FC<{
  layers: CadLayerDiagnostic[];
  summary?: NonNullable<ModeloPlanta['source']>['cad_summary'];
  values: Record<string, CadRole>;
  busy: boolean;
  onChange: (name: string, role: CadRole | 'auto') => void;
  onApply: () => void;
}> = ({ layers, summary, values, busy, onChange, onApply }) => {
  const activeLayers = layers.filter((layer) => layer.entities > 0);
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 flex flex-col gap-3">
      <div>
        <span className="flex items-center gap-1.5 text-sm font-semibold text-slate-800">
          <Layers className="w-4 h-4 text-blue-600" /> Mapa CAD V3
        </span>
        <p className="mt-1 text-[10px] leading-4 text-slate-400">
          Corrija somente layers que o automático classificou errado.
        </p>
      </div>
      {summary && (
        <div className="grid grid-cols-2 gap-1 rounded-lg bg-slate-50 p-2 text-[10px] text-slate-500">
          <span>{summary.entities} entidades</span>
          <span>{summary.layers} layers</span>
          <span>{summary.wall_layers} layer(s) de parede</span>
          <span>{summary.semantic_opening_candidates} esquadrias CAD</span>
          {(summary.grammar_opening_candidates ?? 0) > 0 && (
            <span className="col-span-2 text-blue-600">
              {summary.grammar_opening_candidates} pela gramática de objetos
            </span>
          )}
          {(summary.ocr_lines ?? 0) > 0 && (
            <span className="col-span-2 text-blue-600">
              OCR local: {summary.ocr_lines} linha(s) alinhada(s) ao CAD
            </span>
          )}
          {(summary.missing_linked_images ?? 0) > 0 && (
            <span className="col-span-2 text-amber-600">
              {summary.missing_linked_images} imagem(ns) vinculada(s) ausente(s)
            </span>
          )}
        </div>
      )}
      <div className="max-h-64 space-y-1.5 overflow-auto pr-1">
        {activeLayers.map((layer) => {
          const selected = values[layer.name] ?? 'auto';
          const detected = layer.detected_role
            ? CAD_ROLE_LABELS[layer.detected_role]
            : 'Não classificado';
          const blocks = layer.block_names
            .map((block) => `${block.name} (${block.count})`)
            .join(', ');
          return (
            <div key={layer.name} className="rounded-lg border border-slate-100 p-2">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="truncate text-[10px] font-semibold text-slate-700"
                       title={layer.name}>
                    {layer.name}
                  </div>
                  <div className="text-[9px] text-slate-400">
                    {layer.entities} ent. · {layer.segments} seg. · {detected}
                    {layer.reason === 'geometry'
                      ? ` ${(layer.confidence * 100).toFixed(0)}%`
                      : ''}
                  </div>
                  {blocks && (
                    <div className="truncate text-[9px] text-slate-400" title={blocks}>
                      Blocos: {blocks}
                    </div>
                  )}
                </div>
                <select
                  value={selected}
                  onChange={(event) => onChange(
                    layer.name,
                    event.target.value as CadRole | 'auto',
                  )}
                  className="w-24 shrink-0 rounded border border-slate-200 bg-white px-1 py-1 text-[9px]"
                >
                  {(Object.keys(CAD_ROLE_LABELS) as (CadRole | 'auto')[]).map((role) => (
                    <option key={role} value={role}>{CAD_ROLE_LABELS[role]}</option>
                  ))}
                </select>
              </div>
            </div>
          );
        })}
      </div>
      <button
        type="button"
        disabled={busy}
        onClick={onApply}
        className="flex items-center justify-center gap-1.5 rounded-lg bg-slate-800 px-3 py-2 text-xs font-semibold text-white hover:bg-slate-900 disabled:opacity-40"
      >
        {busy
          ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
          : <RotateCcw className="h-3.5 w-3.5" />}
        Reprocessar layers
      </button>
    </div>
  );
};

const PainelConfig: React.FC<{ cfg: PlantaConfig; setCfg: (c: PlantaConfig) => void }> = ({ cfg, setCfg }) => (
  <div className="bg-white rounded-xl border border-slate-200 p-4 flex flex-col gap-2.5">
    <div>
      <span className="text-sm font-semibold text-slate-800">Padrões do projeto</span>
      <p className="mt-1 text-[10px] leading-4 text-slate-400">
        Usados somente quando uma parede, porta ou janela não possui medida individual.
      </p>
    </div>
    <Campo label="Pé-direito (m)">
      <input type="number" step={0.1} value={cfg.altura}
             onChange={(e) => {
               const altura = +e.target.value;
               const espessura = cfg.forro?.espessura ?? 0.03;
               const maxForroHeight = Math.max(0.1, altura - espessura);
               setCfg({
                 ...cfg,
                 altura,
                 forro: {
                   ativo: cfg.forro?.ativo ?? false,
                   espessura,
                   altura: Math.min(cfg.forro?.altura ?? maxForroHeight, maxForroHeight),
                 },
               });
             }} className={numInput} />
    </Campo>
    <Campo label="Altura porta (m)">
      <input type="number" step={0.1} value={cfg.porta_altura}
             onChange={(e) => setCfg({ ...cfg, porta_altura: +e.target.value })} className={numInput} />
    </Campo>
    <Campo label="Altura janela (m)">
      <input type="number" step={0.1} value={cfg.janela_altura}
             onChange={(e) => setCfg({ ...cfg, janela_altura: +e.target.value })} className={numInput} />
    </Campo>
    <Campo label="Peitoril (m)">
      <input type="number" step={0.1} value={cfg.janela_peitoril}
             onChange={(e) => setCfg({ ...cfg, janela_peitoril: +e.target.value })} className={numInput} />
    </Campo>
    <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer mt-1">
      <input type="checkbox" checked={cfg.esquadria_detalhada}
             onChange={(e) => setCfg({ ...cfg, esquadria_detalhada: e.target.checked })} />
      Esquadria detalhada (batente/vidro)
    </label>
    <div className="mt-1 border-t border-slate-100 pt-2.5">
      <label className="flex items-center gap-2 text-xs font-medium text-slate-700 cursor-pointer">
        <input
          type="checkbox"
          checked={Boolean(cfg.forro?.ativo)}
          onChange={(e) => {
            const espessura = cfg.forro?.espessura ?? 0.03;
            const maxForroHeight = Math.max(0.1, cfg.altura - espessura);
            setCfg({
              ...cfg,
              forro: {
                ativo: e.target.checked,
                altura: Math.min(cfg.forro?.altura ?? maxForroHeight, maxForroHeight),
                espessura,
              },
            });
          }}
        />
        Forro suspenso por ambiente
      </label>
      {cfg.forro?.ativo && (
        <div className="mt-2 flex flex-col gap-2 pl-5">
          <Campo label="Altura inferior (m)">
            <input
              type="number"
              step={0.05}
              min={0.1}
              max={Math.max(0.1, cfg.altura - (cfg.forro.espessura ?? 0.03))}
              value={cfg.forro.altura ?? Math.max(0.1, (cfg.altura ?? 2.8) - 0.1)}
              onChange={(e) => {
                const maxForroHeight = Math.max(
                  0.1,
                  cfg.altura - (cfg.forro?.espessura ?? 0.03),
                );
                setCfg({
                  ...cfg,
                  forro: {
                    ...cfg.forro!,
                    altura: Math.min(Math.max(0.1, +e.target.value), maxForroHeight),
                  },
                });
              }}
              className={numInput}
            />
          </Campo>
          <Campo label="Espessura (m)">
            <input
              type="number"
              step={0.01}
              min={0.01}
              max={0.3}
              value={cfg.forro.espessura ?? 0.03}
              onChange={(e) => {
                const espessura = Math.max(0.01, +e.target.value);
                const maxForroHeight = Math.max(0.1, cfg.altura - espessura);
                setCfg({
                  ...cfg,
                  forro: {
                    ...cfg.forro!,
                    espessura,
                    altura: Math.min(cfg.forro?.altura ?? maxForroHeight, maxForroHeight),
                  },
                });
              }}
              className={numInput}
            />
          </Campo>
        </div>
      )}
    </div>
  </div>
);

const PainelLaje: React.FC<{
  laje: Laje; patchFace: (face: 'piso' | 'teto', patch: Partial<LajeFace>) => void;
  selVertice: number | null; onRemoverPonto: () => void;
}> = ({ laje, patchFace, selVertice, onRemoverPonto }) => (
  <div className="bg-white rounded-xl border border-blue-200 p-4 flex flex-col gap-3">
    <span className="text-sm font-semibold text-slate-800 flex items-center gap-2">
      <Layers className="w-4 h-4 text-blue-600" /> Piso e teto
    </span>
    {selVertice !== null && (
      <div className="flex items-center justify-between bg-blue-50 rounded-lg px-3 py-2 text-xs">
        <span className="text-blue-700 font-medium">Ponto {selVertice + 1} selecionado</span>
        <button onClick={onRemoverPonto} disabled={laje.contorno.length <= 3}
          className="flex items-center gap-1 text-rose-600 hover:text-rose-700 disabled:opacity-30">
          <Trash2 className="w-3.5 h-3.5" /> Remover
        </button>
      </div>
    )}
    {(['piso', 'teto'] as const).map((face) => (
      <div key={face} className="flex flex-col gap-2 border-t border-slate-100 pt-2 first:border-0 first:pt-0">
        <label className="flex items-center gap-2 text-xs font-medium text-slate-700 cursor-pointer">
          <input type="checkbox" checked={laje[face].ativo}
                 onChange={(e) => patchFace(face, { ativo: e.target.checked })} />
          {face === 'piso' ? 'Piso' : 'Teto (cobertura)'}
        </label>
        {laje[face].ativo && (
          <label className="flex items-center justify-between text-xs text-slate-600 pl-6">
            Espessura (m)
            <input type="number" step={0.01} min={0.03} max={0.5} value={laje[face].espessura}
                   onChange={(e) => patchFace(face, { espessura: +e.target.value })} className={numInput} />
          </label>
        )}
      </div>
    ))}
    <p className="text-[11px] text-slate-400">
      Arraste os vértices no desenho pra ajustar o contorno. Piso e teto usam o mesmo polígono.
    </p>
  </div>
);
