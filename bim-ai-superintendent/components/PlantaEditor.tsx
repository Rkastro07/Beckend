import React, { useRef, useState, useEffect, useMemo } from 'react';
import {
  FileUp, Loader2, Download, Trash2, Plus, DoorOpen, RectangleHorizontal,
  Sparkles, AlertCircle, CheckCircle2, RotateCcw, Move, Square, Layers, Maximize,
  ScanLine,
} from 'lucide-react';
import {
  parsePlanta, gerarPlantaIfc, downloadUrl,
  ModeloPlanta, Parede, Abertura, PlantaConfig, Laje, LajeFace,
} from '../services/tools';

type Sel = { kind: 'parede' | 'abertura'; id: string } | null;
type Drag =
  | { kind: 'wall-a' | 'wall-b' | 'wall-body'; id: string; ox: number; oy: number; ax: number; ay: number; bx: number; by: number }
  | { kind: 'opening'; id: string; ps: number }
  | { kind: 'laje-vertice'; idx: number }
  | { kind: 'pan'; px: number; py: number; vx: number; vy: number }
  | null;
type View = { x: number; y: number; w: number; h: number };

const COR_PAREDE = '#e08e34';
const COR_PORTA = '#2ecc71';
const COR_JANELA = '#5db4e2';
const COR_SEL = '#2563eb';

export type ResultadoGeracao = { ifc_url: string; preview_url?: string };

export interface PlantaEditorProps {
  /** Modelo inicial (ex.: vindo do Scan → BIM). Se ausente, o editor pede um DXF. */
  modeloInicial?: ModeloPlanta | null;
  /** Gerador de IFC customizado; se ausente usa o gerarPlantaIfc (plantatobim/DXF). */
  onGerar?: (
    modelo: { paredes: Parede[]; aberturas: Abertura[]; laje: Laje },
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
}

export const PlantaEditor: React.FC<PlantaEditorProps> = ({
  modeloInicial, onGerar, titulo, subtitulo, ocultarPreviewPly,
  onVoltar, rotuloVoltar, backdropPng, backdropBounds,
} = {}) => {
  const [modelo, setModelo] = useState<ModeloPlanta | null>(modeloInicial ?? null);
  const [sel, setSel] = useState<Sel>(null);
  const [modoLaje, setModoLaje] = useState(false);
  const [mostrarBackdrop, setMostrarBackdrop] = useState(true);
  const [selVertice, setSelVertice] = useState<number | null>(null);
  const [view, setView] = useState<View | null>(null);
  const [busy, setBusy] = useState(false);
  const [erro, setErro] = useState<string | null>(null);
  const [resultado, setResultado] = useState<{ ifc_url: string; preview_url: string } | null>(null);
  const [cfg, setCfg] = useState<PlantaConfig>({
    altura: 2.8, porta_altura: 2.1, janela_altura: 1.2, janela_peitoril: 1.0,
    esquadria_detalhada: false, cobertura: true,
  });

  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<Drag>(null);

  const paredeById = useMemo(() => {
    const m: Record<string, Parede> = {};
    modelo?.paredes.forEach((p) => (m[p.id] = p));
    return m;
  }, [modelo]);

  // ---------- upload / parse ----------
  const fitView = (m: ModeloPlanta) => {
    const { xmin, ymin, xmax, ymax } = m.bbox;
    const pad = Math.max(xmax - xmin, ymax - ymin) * 0.08 + 0.5;
    setView({ x: xmin - pad, y: -(ymax + pad), w: xmax - xmin + 2 * pad, h: ymax - ymin + 2 * pad });
  };

  // Modelo injetado de fora (Scan → BIM): adota e enquadra quando muda.
  useEffect(() => {
    if (modeloInicial) {
      setModelo(modeloInicial);
      setSel(null); setSelVertice(null); setResultado(null);
      fitView(modeloInicial);
    }
  }, [modeloInicial]);

  const onFile = async (file: File) => {
    setBusy(true); setErro(null); setResultado(null); setSel(null); setSelVertice(null);
    try {
      const m = await parsePlanta(file);
      setModelo(m); fitView(m);
    } catch (e: any) { setErro(e.message || 'Falha ao interpretar DXF'); }
    finally { setBusy(false); }
  };

  // ---------- edição de dados ----------
  const patchParede = (id: string, patch: Partial<Parede>) =>
    setModelo((m) => m && { ...m, paredes: m.paredes.map((p) => p.id === id ? { ...p, ...patch } : p) });
  const patchAbertura = (id: string, patch: Partial<Abertura>) =>
    setModelo((m) => m && { ...m, aberturas: m.aberturas.map((a) => a.id === id ? { ...a, ...patch } : a) });

  const patchFace = (face: 'piso' | 'teto', patch: Partial<{ ativo: boolean; espessura: number }>) =>
    setModelo((m) => m && { ...m, laje: { ...m.laje, [face]: { ...m.laje[face], ...patch } } });
  const moveVertice = (idx: number, x: number, y: number) =>
    setModelo((m) => {
      if (!m) return m;
      const c = m.laje.contorno.map((v, i) => i === idx ? [x, y] as [number, number] : v);
      return { ...m, laje: { ...m.laje, contorno: c } };
    });
  const addVertice = (afterIdx: number) =>
    setModelo((m) => {
      if (!m) return m;
      const c = [...m.laje.contorno];
      const a = c[afterIdx], b = c[(afterIdx + 1) % c.length];
      c.splice(afterIdx + 1, 0, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
      return { ...m, laje: { ...m.laje, contorno: c } };
    });
  const removeVertice = (idx: number) =>
    setModelo((m) => {
      if (!m || m.laje.contorno.length <= 3) return m;  // mantem poligono valido
      return { ...m, laje: { ...m.laje, contorno: m.laje.contorno.filter((_, i) => i !== idx) } };
    });

  const apagarSel = () => {
    if (!sel || !modelo) return;
    if (sel.kind === 'parede') {
      setModelo({ ...modelo,
        paredes: modelo.paredes.filter((p) => p.id !== sel.id),
        aberturas: modelo.aberturas.filter((a) => a.parede_id !== sel.id) });
    } else {
      setModelo({ ...modelo, aberturas: modelo.aberturas.filter((a) => a.id !== sel.id) });
    }
    setSel(null);
  };

  const addParede = () => {
    if (!modelo) return;
    const { xmin, ymin, xmax, ymax } = modelo.bbox;
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
    const id = `w${Date.now()}`;
    setModelo({ ...modelo, paredes: [...modelo.paredes,
      { id, ax: cx - 1, ay: cy, bx: cx + 1, by: cy, espessura: 0.15, layer: 'Wall-Nova' }] });
    setSel({ kind: 'parede', id });
  };

  const addAbertura = (tipo: 'door' | 'window') => {
    if (!modelo || sel?.kind !== 'parede') return;
    const p = paredeById[sel.id];
    const L = Math.hypot(p.bx - p.ax, p.by - p.ay);
    const id = `o${Date.now()}`;
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

  const onMouseMove = (e: React.MouseEvent) => {
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
      patchParede(d.id, { ax: w.x, ay: w.y });
    } else if (d.kind === 'wall-b') {
      patchParede(d.id, { bx: w.x, by: w.y });
    } else if (d.kind === 'wall-body') {
      const dx = w.x - d.ox, dy = w.y - d.oy;
      patchParede(d.id, { ax: d.ax + dx, ay: d.ay + dy, bx: d.bx + dx, by: d.by + dy });
    } else if (d.kind === 'laje-vertice') {
      moveVertice(d.idx, w.x, w.y);
    }
  };
  const endDrag = () => { dragRef.current = null; };

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
  }, [modelo === null]);  // re-anexa quando o canvas monta (modelo carrega)

  // ---------- gerar IFC ----------
  const gerar = async () => {
    if (!modelo) return;
    setBusy(true); setErro(null); setResultado(null);
    try {
      const payload = { paredes: modelo.paredes, aberturas: modelo.aberturas, laje: modelo.laje };
      const nome = modelo.nome || 'planta';
      const r = onGerar
        ? await onGerar(payload, cfg, nome)
        : await gerarPlantaIfc(payload, cfg, nome);
      setResultado({ ifc_url: r.ifc_url, preview_url: r.preview_url ?? '' });
    } catch (e: any) { setErro(e.message || 'Falha ao gerar IFC'); }
    finally { setBusy(false); }
  };

  // ---------- viewBox / escala dos tracos de UI (constante em pixels) ----------
  const vb = view ? `${view.x} ${view.y} ${view.w} ${view.h}` : '0 0 10 10';
  const escalaTraco = view ? view.w / 400 : 0.02;  // ~1px; escala com o zoom

  // =====================================================================
  //  RENDER
  // =====================================================================
  if (!modelo) {
    return (
      <div className="p-8">
        <Cabecalho titulo={titulo} subtitulo={subtitulo} />
        <label className="mt-6 flex flex-col items-center justify-center gap-3 h-64 max-w-2xl
                          border-2 border-dashed border-slate-300 rounded-2xl cursor-pointer
                          hover:bg-slate-50 hover:border-blue-300 transition-colors">
          {busy ? <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                : <FileUp className="w-8 h-8 text-slate-400" />}
          <span className="text-sm text-slate-600 font-medium">
            {busy ? 'Interpretando planta...' : 'Solte um arquivo .dxf ou clique para escolher'}
          </span>
          <span className="text-xs text-slate-400">DWG: exporte como DXF no AutoCAD antes</span>
          <input type="file" accept=".dxf" className="hidden"
                 onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        </label>
        {erro && <ErroBox msg={erro} />}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-8 pt-6"><Cabecalho titulo={titulo} subtitulo={subtitulo} /></div>

      <div className="flex-1 flex gap-4 px-8 pb-6 pt-4 min-h-0">
        {/* ---------- CANVAS ---------- */}
        <div className="flex-1 bg-white rounded-xl border border-slate-200 relative overflow-hidden">
          <div className="absolute top-3 left-3 z-10 flex gap-2">
            {!modoLaje && <button onClick={addParede}
              className="flex items-center gap-1 text-xs px-2.5 py-1.5 bg-white border border-slate-200
                         rounded-lg shadow-sm hover:bg-slate-50">
              <Plus className="w-3.5 h-3.5" /> Parede
            </button>}
            {!modoLaje && sel?.kind === 'parede' && <>
              <button onClick={() => addAbertura('door')}
                className="flex items-center gap-1 text-xs px-2.5 py-1.5 bg-white border border-slate-200 rounded-lg shadow-sm hover:bg-slate-50">
                <DoorOpen className="w-3.5 h-3.5 text-emerald-500" /> Porta
              </button>
              <button onClick={() => addAbertura('window')}
                className="flex items-center gap-1 text-xs px-2.5 py-1.5 bg-white border border-slate-200 rounded-lg shadow-sm hover:bg-slate-50">
                <RectangleHorizontal className="w-3.5 h-3.5 text-sky-500" /> Janela
              </button>
            </>}
            <button onClick={() => { setModoLaje((v) => !v); setSel(null); setSelVertice(null); }}
              className={`flex items-center gap-1 text-xs px-2.5 py-1.5 border rounded-lg shadow-sm
                ${modoLaje ? 'bg-blue-600 text-white border-blue-600' : 'bg-white border-slate-200 hover:bg-slate-50'}`}>
              <Layers className="w-3.5 h-3.5" /> {modoLaje ? 'Editando laje' : 'Editar laje'}
            </button>
            {backdropPng && (
              <button onClick={() => setMostrarBackdrop((v) => !v)}
                className={`flex items-center gap-1 text-xs px-2.5 py-1.5 border rounded-lg shadow-sm
                  ${mostrarBackdrop ? 'bg-slate-700 text-white border-slate-700' : 'bg-white border-slate-200 hover:bg-slate-50'}`}>
                <ScanLine className="w-3.5 h-3.5" /> Nuvem captada
              </button>
            )}
          </div>

          <button onClick={() => modelo && fitView(modelo)} title="Ajustar zoom à planta"
            className="absolute top-3 right-3 z-10 flex items-center gap-1 text-xs px-2.5 py-1.5
                       bg-white border border-slate-200 rounded-lg shadow-sm hover:bg-slate-50">
            <Maximize className="w-3.5 h-3.5" /> Ajustar
          </button>

          <svg ref={svgRef} viewBox={vb} className="w-full h-full"
               style={{ cursor: dragRef.current?.kind === 'pan' ? 'grabbing' : 'default' }}
               onMouseMove={onMouseMove} onMouseUp={endDrag} onMouseLeave={endDrag}
               onMouseDown={(e) => {
                 if (e.target === svgRef.current && view)
                   dragRef.current = { kind: 'pan', px: e.clientX, py: e.clientY, vx: view.x, vy: view.y };
               }}
               onClick={(e) => { if (e.target === svgRef.current) { setSel(null); setSelVertice(null); } }}>
            {/* mapa de densidade da nuvem captada — o cliente edita EM CIMA dela */}
            {mostrarBackdrop && backdropPng && backdropBounds && (
              <image href={`data:image/png;base64,${backdropPng}`}
                     x={backdropBounds[0]} y={-backdropBounds[3]}
                     width={backdropBounds[2] - backdropBounds[0]}
                     height={backdropBounds[3] - backdropBounds[1]}
                     opacity={0.45} preserveAspectRatio="none"
                     style={{ imageRendering: 'pixelated', pointerEvents: 'none' }} />
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
              return (
                <g key={p.id}>
                  <line x1={p.ax} y1={-p.ay} x2={p.bx} y2={-p.by}
                        stroke={on ? COR_SEL : COR_PAREDE} strokeWidth={p.espessura}
                        strokeLinecap="round" opacity={0.85} style={{ cursor: 'move' }}
                        onMouseDown={(e) => {
                          e.stopPropagation(); setSel({ kind: 'parede', id: p.id });
                          const w = worldFromEvent(e);
                          dragRef.current = { kind: 'wall-body', id: p.id, ox: w.x, oy: w.y,
                            ax: p.ax, ay: p.ay, bx: p.bx, by: p.by };
                        }} />
                  {on && [['a', p.ax, p.ay], ['b', p.bx, p.by]].map(([k, x, y]) => (
                    <circle key={k as string} cx={x as number} cy={-(y as number)}
                            r={escalaTraco * 5} fill="#fff" stroke={COR_SEL} strokeWidth={escalaTraco * 2}
                            style={{ cursor: 'pointer' }}
                            onMouseDown={(e) => {
                              e.stopPropagation();
                              dragRef.current = { kind: k === 'a' ? 'wall-a' : 'wall-b', id: p.id,
                                ox: 0, oy: 0, ax: p.ax, ay: p.ay, bx: p.bx, by: p.by };
                            }} />
                  ))}
                </g>
              );
            })}
            {/* aberturas */}
            {modelo.aberturas.map((a) => {
              const p = paredeById[a.parede_id];
              if (!p) return null;
              const L = Math.hypot(p.bx - p.ax, p.by - p.ay);
              const ux = (p.bx - p.ax) / L, uy = (p.by - p.ay) / L;
              const cxw = p.ax + ux * a.s_centro, cyw = p.ay + uy * a.s_centro;
              const x1 = cxw - ux * a.largura / 2, y1 = cyw - uy * a.largura / 2;
              const x2 = cxw + ux * a.largura / 2, y2 = cyw + uy * a.largura / 2;
              const on = sel?.kind === 'abertura' && sel.id === a.id;
              return (
                <line key={a.id} x1={x1} y1={-y1} x2={x2} y2={-y2}
                      stroke={on ? COR_SEL : (a.tipo === 'door' ? COR_PORTA : COR_JANELA)}
                      strokeWidth={p.espessura * 1.25} strokeLinecap="butt"
                      style={{ cursor: 'grab' }}
                      onMouseDown={(e) => {
                        e.stopPropagation(); setSel({ kind: 'abertura', id: a.id });
                        dragRef.current = { kind: 'opening', id: a.id, ps: a.s_centro };
                      }} />
              );
            })}
            </g>
          </svg>

          <div className="absolute bottom-3 left-3 text-[11px] text-slate-400 flex items-center gap-2">
            <Move className="w-3 h-3" />
            {modoLaje
              ? 'arraste os vértices • toque no “+” p/ adicionar • duplo-clique no ponto p/ remover • scroll = zoom'
              : 'arraste para mover • clique para selecionar • scroll = zoom • arraste o fundo p/ mover a vista'}
          </div>
        </div>

        {/* ---------- PAINEL ---------- */}
        <div className="w-72 shrink-0 flex flex-col gap-4 overflow-auto">
          {modoLaje
            ? <PainelLaje laje={modelo.laje} patchFace={patchFace}
                selVertice={selVertice}
                onRemoverPonto={() => { if (selVertice !== null) { removeVertice(selVertice); setSelVertice(null); } }} />
            : <PainelSelecionado
                sel={sel} modelo={modelo} paredeById={paredeById}
                patchParede={patchParede} patchAbertura={patchAbertura} apagar={apagarSel} />}

          <PainelConfig cfg={cfg} setCfg={setCfg} />

          <button onClick={gerar} disabled={busy}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg
                       text-sm font-semibold bg-blue-600 text-white hover:bg-blue-700
                       disabled:opacity-40 transition-colors">
            {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            Gerar IFC
          </button>

          {resultado && (
            <div className="flex flex-col gap-2 bg-emerald-50 rounded-lg px-3 py-3 text-xs">
              <span className="flex items-center gap-2 text-emerald-700 font-medium">
                <CheckCircle2 className="w-4 h-4" /> IFC gerado
              </span>
              <a href={downloadUrl(resultado.ifc_url)} download
                 className="flex items-center gap-1 justify-center px-2 py-1.5 bg-emerald-600 text-white rounded-md hover:bg-emerald-700">
                <Download className="w-3.5 h-3.5" /> Baixar IFC
              </a>
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
            }}
            className="flex items-center justify-center gap-1 text-xs text-slate-400 hover:text-slate-600">
            <RotateCcw className="w-3 h-3" /> {rotuloVoltar ?? 'Trocar planta'}
          </button>
        </div>
      </div>
    </div>
  );
};

// ---------- sub-componentes ----------
const Cabecalho: React.FC<{ titulo?: string; subtitulo?: string }> = ({ titulo, subtitulo }) => (
  <div>
    <h1 className="text-xl font-bold text-slate-800">{titulo ?? 'Planta → BIM'}</h1>
    <p className="text-sm text-slate-500">
      {subtitulo ?? 'Suba um DXF, revise e edite o que o sistema entendeu, e gere o IFC.'}
    </p>
  </div>
);

const ErroBox: React.FC<{ msg: string }> = ({ msg }) => (
  <div className="flex items-center gap-2 text-xs text-rose-600 bg-rose-50 rounded-lg px-3 py-2 max-w-2xl">
    <AlertCircle className="w-4 h-4 shrink-0" /> {msg}
  </div>
);

const Campo: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <label className="flex items-center justify-between text-xs text-slate-600 gap-2">
    {label} {children}
  </label>
);
const numInput = "w-20 border border-slate-200 rounded px-2 py-1 text-right text-xs";

const PainelSelecionado: React.FC<{
  sel: Sel; modelo: ModeloPlanta; paredeById: Record<string, Parede>;
  patchParede: (id: string, p: Partial<Parede>) => void;
  patchAbertura: (id: string, p: Partial<Abertura>) => void;
  apagar: () => void;
}> = ({ sel, modelo, paredeById, patchParede, patchAbertura, apagar }) => {
  if (!sel) return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 text-xs text-slate-500">
      <div className="flex gap-4 mb-2 font-medium text-slate-700">
        <span>{modelo.paredes.length} paredes</span>
        <span>{modelo.aberturas.length} esquadrias</span>
      </div>
      Clique numa parede ou esquadria para editar.
      <div className="mt-2 text-[11px] text-slate-400">
        {modelo.single_line ? 'Modo linha-única (espessura default)' : 'Espessura medida do desenho'}
      </div>
    </div>
  );
  if (sel.kind === 'parede') {
    const p = paredeById[sel.id];
    if (!p) return null;
    const L = Math.hypot(p.bx - p.ax, p.by - p.ay);
    return (
      <div className="bg-white rounded-xl border border-blue-200 p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold text-slate-800">Parede</span>
          <button onClick={apagar} className="text-rose-500 hover:text-rose-700"><Trash2 className="w-4 h-4" /></button>
        </div>
        <Campo label="Espessura (m)">
          <input type="number" step={0.01} min={0.03} max={0.6} value={p.espessura}
                 onChange={(e) => patchParede(p.id, { espessura: +e.target.value })} className={numInput} />
        </Campo>
        <div className="text-[11px] text-slate-400">Comprimento: {L.toFixed(2)} m · layer {p.layer || '—'}</div>
      </div>
    );
  }
  const a = modelo.aberturas.find((x) => x.id === sel.id);
  if (!a) return null;
  return (
    <div className="bg-white rounded-xl border border-blue-200 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-800">Esquadria</span>
        <button onClick={apagar} className="text-rose-500 hover:text-rose-700"><Trash2 className="w-4 h-4" /></button>
      </div>
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
      <Campo label="Largura (m)">
        <input type="number" step={0.05} min={0.4} max={4} value={a.largura}
               onChange={(e) => patchAbertura(a.id, { largura: +e.target.value })} className={numInput} />
      </Campo>
      <div className="text-[11px] text-slate-400">Arraste a esquadria no desenho para reposicionar.</div>
    </div>
  );
};

const PainelConfig: React.FC<{ cfg: PlantaConfig; setCfg: (c: PlantaConfig) => void }> = ({ cfg, setCfg }) => (
  <div className="bg-white rounded-xl border border-slate-200 p-4 flex flex-col gap-2.5">
    <span className="text-sm font-semibold text-slate-800">Alturas do modelo</span>
    <Campo label="Pé-direito (m)">
      <input type="number" step={0.1} value={cfg.altura}
             onChange={(e) => setCfg({ ...cfg, altura: +e.target.value })} className={numInput} />
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
