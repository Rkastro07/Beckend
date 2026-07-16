import React, { useEffect, useRef, useState } from 'react';
import {
  FileUp, Loader2, Download, RotateCcw, ScanLine, Layers,
  CheckCircle2, AlertCircle, Sparkles, ChevronRight, TrendingUp, PencilRuler,
} from 'lucide-react';
import {
  scanUpload, scanLajes, scanParedes, scanEscadas, scanGerarIfc, scanJob, downloadUrl,
  ScanUploadResult, ScanParedesResult, ScanLance, ScanAbertura,
  ModeloPlanta, PlantaConfig,
} from '../services/tools';
import { PlantaEditor, ResultadoGeracao } from './PlantaEditor';

type Fase = 1 | 2 | 3;

/**
 * Scan -> BIM em 3 FASES (fluxo proposto pelo Rafael): cada fase mostra o que
 * o sistema esta captando e da os controles daquele detector.
 *   1. Lajes    — histograma de z + threshold -> niveis/pavimentos
 *   2. Paredes  — fatia + pareamento REAL (vermelho = instancia futura)
 *   3. Escadas  — lances detectados por banda + sensibilidade
 * "Gerar IFC" roda a cadeia completa com tudo que foi calibrado.
 */
export const ScanToBim: React.FC = () => {
  const [busy, setBusy] = useState(false);
  const [erro, setErro] = useState<string | null>(null);
  const [up, setUp] = useState<ScanUploadResult | null>(null);
  const [fase, setFase] = useState<Fase>(1);

  // fase 1 — lajes
  const [thr, setThr] = useState(0.3);
  const [lajes, setLajes] = useState<[number, number][]>([]);
  const [bandas, setBandas] = useState<[number, number][]>([]);

  // fase 2 — paredes
  const [bandaIdx, setBandaIdx] = useState(0);
  const [zloFrac, setZloFrac] = useState(0.1);
  const [zhiFrac, setZhiFrac] = useState(0.9);
  const [minLen, setMinLen] = useState(0.3);
  const [contoursAll, setContoursAll] = useState(true);
  const [preview, setPreview] = useState<ScanParedesResult | null>(null);
  const [carregando, setCarregando] = useState(false);

  // fase 3 — escadas
  const [areaMin, setAreaMin] = useState(0.8);
  const [lances, setLances] = useState<ScanLance[] | null>(null);

  // geracao
  const [job, setJob] = useState<string | null>(null);
  const [jobEtapa, setJobEtapa] = useState('');
  const [ifcUrl, setIfcUrl] = useState<string | null>(null);
  const [detalhes, setDetalhes] = useState<Record<string, string> | null>(null);

  // finalizacao opcional no editor (portas/janelas/teto)
  const [modeloEditor, setModeloEditor] = useState<ModeloPlanta | null>(null);

  // ---------- upload ----------
  const onFile = async (f: File) => {
    setBusy(true); setErro(null); setUp(null); setPreview(null);
    setIfcUrl(null); setDetalhes(null); setLances(null); setFase(1);
    try {
      const r = await scanUpload(f);
      setUp(r); setThr(r.thr_sugerido);
    } catch (e: any) { setErro(e.message || 'Falha no upload'); }
    finally { setBusy(false); }
  };

  // ---------- fase 1: reagrupa lajes quando thr muda ----------
  useEffect(() => {
    if (!up) return;
    scanLajes(up.sid, thr)
      .then((r) => {
        setLajes(r.lajes); setBandas(r.bandas);
        setBandaIdx((i) => Math.min(i, Math.max(r.bandas.length - 1, 0)));
      })
      .catch((e) => setErro(e.message));
  }, [up, thr]);

  // ---------- fase 2: preview com o motor real (debounce) ----------
  // seq: respostas fora de ordem (request antiga chegando DEPOIS da nova)
  // eram descartadas erradas — só a resposta da ÚLTIMA config vale
  const timer2 = useRef<number | null>(null);
  const seq2 = useRef(0);
  const [cfgPreview, setCfgPreview] = useState<string>('');
  useEffect(() => {
    if (!up || fase < 2 || !bandas.length) return;
    if (timer2.current) window.clearTimeout(timer2.current);
    setCarregando(true);
    timer2.current = window.setTimeout(async () => {
      const [b0, b1] = bandas[bandaIdx];
      const minha = ++seq2.current;
      const cfg = `pav ${bandaIdx + 1} · fatia ${(zloFrac * 100).toFixed(0)}–${(zhiFrac * 100).toFixed(0)}% · mín ${minLen.toFixed(2)}m`;
      try {
        const r = await scanParedes(up.sid, b0, b1, zloFrac, zhiFrac, minLen, contoursAll);
        if (seq2.current !== minha) return;      // chegou atrasada: ignora
        setPreview(r); setCfgPreview(cfg); setErro(null);
      } catch (e: any) { if (seq2.current === minha) setErro(e.message); }
      finally { if (seq2.current === minha) setCarregando(false); }
    }, 350);
    return () => { if (timer2.current) window.clearTimeout(timer2.current); };
  }, [up, fase, bandas, bandaIdx, zloFrac, zhiFrac, minLen, contoursAll]);

  // ---------- fase 3: lances (debounce + seq proprios) ----------
  const timer3 = useRef<number | null>(null);
  const seq3 = useRef(0);
  useEffect(() => {
    if (!up || fase !== 3 || !bandas.length) return;
    if (timer3.current) window.clearTimeout(timer3.current);
    timer3.current = window.setTimeout(async () => {
      const [b0, b1] = bandas[bandaIdx];
      const minha = ++seq3.current;
      setCarregando(true);
      try {
        const r = await scanEscadas(up.sid, b0, b1, areaMin);
        if (seq3.current !== minha) return;
        setLances(r.lances); setErro(null);
      } catch (e: any) { if (seq3.current === minha) setErro(e.message); }
      finally { if (seq3.current === minha) setCarregando(false); }
    }, 350);
    return () => { if (timer3.current) window.clearTimeout(timer3.current); };
  }, [up, fase, bandas, bandaIdx, areaMin]);

  // ---------- job ----------
  useEffect(() => {
    if (!job) return;
    const h = window.setInterval(async () => {
      try {
        const st = await scanJob(job);
        setJobEtapa(st.etapa || '');
        if (st.detalhes) setDetalhes(st.detalhes);
        if (st.status === 'pronto') { setIfcUrl(st.url || null); setJob(null); }
        if (st.status === 'erro') { setErro(st.erro || 'Falha na geração'); setJob(null); }
      } catch { /* tenta de novo */ }
    }, 2500);
    return () => window.clearInterval(h);
  }, [job]);

  const gerar = async () => {
    if (!up || carregando || !preview) return;
    setErro(null); setIfcUrl(null); setDetalhes(null);
    try {
      const r = await scanGerarIfc(up.sid, thr, zloFrac, zhiFrac, 1.0, minLen,
                                   contoursAll, bandaIdx, preview.eixos);
      setJob(r.job); setJobEtapa('pipeline');
    } catch (e: any) { setErro(e.message); }
  };

  // ---- finalização opcional: paredes aprovadas -> ModeloPlanta do editor ----
  const abrirEditor = () => {
    if (!preview) return;
    const [xmin, ymin, xmax, ymax] = preview.bounds;
    const paredes = preview.eixos.map((e, i) => ({
      id: `w${i}`, ax: e[0], ay: e[1], bx: e[2], by: e[3],
      espessura: e[4], layer: e[5],
    }));
    const contorno: [number, number][] =
      preview.contorno_teto && preview.contorno_teto.length >= 3
        ? preview.contorno_teto
        : [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]];
    setModeloEditor({
      escala: 1, single_line: false, nome: 'scan',
      bbox: { xmin, ymin, xmax, ymax },
      diagnostico: { sobras: 0, cantos_costurados: 0, blocos_esquadria: 0 },
      paredes, aberturas: [],
      laje: { contorno, piso: { ativo: true, espessura: 0.15 },
              teto: { ativo: false, espessura: 0.15 } },
    });
  };

  // gerador do editor: as PAREDES e as portas/janelas EDITADAS na tela viram
  // overrides na cadeia cloud2bim (o IFC = exatamente o que o cliente desenhou)
  const gerarDoEditor = async (
    modelo: { paredes: any[]; aberturas: any[] }, cfg: PlantaConfig,
  ): Promise<ResultadoGeracao> => {
    if (!up || !preview) throw new Error('sessão do scan expirada');
    if (!modelo.paredes.length) throw new Error('nenhuma parede no editor');
    // eixos = as paredes como estão AGORA no editor (movidas/adicionadas/removidas),
    // não o preview original. Índice na lista é o que ancora as aberturas.
    const eixos = modelo.paredes.map((p) =>
      [p.ax, p.ay, p.bx, p.by, p.espessura, p.layer] as
        [number, number, number, number, number, string]);
    const idxDaParede = new Map<string, number>(
      modelo.paredes.map((p, i) => [p.id, i]));
    const aberturas: ScanAbertura[] = modelo.aberturas
      .map((a) => {
        const eixo_idx = idxDaParede.get(String(a.parede_id));
        return eixo_idx === undefined ? null
          : { eixo_idx, tipo: a.tipo, s_centro: a.s_centro, largura: a.largura };
      })
      .filter((a): a is ScanAbertura => a !== null);
    const { job: jid } = await scanGerarIfc(
      up.sid, thr, zloFrac, zhiFrac, 1.0, minLen, contoursAll,
      bandaIdx, eixos, aberturas, cfg);
    setJob(null);  // este fluxo tem poll próprio (resolve a Promise do editor)
    return await new Promise((resolve, reject) => {
      const h = window.setInterval(async () => {
        try {
          const st = await scanJob(jid);
          setJobEtapa(st.etapa || '');
          if (st.detalhes) setDetalhes(st.detalhes);
          if (st.status === 'pronto') {
            window.clearInterval(h); resolve({ ifc_url: st.url || '' });
          }
          if (st.status === 'erro') {
            window.clearInterval(h); reject(new Error(st.erro || 'Falha na geração'));
          }
        } catch { /* tenta de novo */ }
      }, 2500);
    });
  };

  // =====================================================================
  if (!up) {
    return (
      <div className="p-8">
        <Cabecalho />
        <label className="mt-6 flex flex-col items-center justify-center gap-3 h-64 max-w-2xl
                          border-2 border-dashed border-slate-300 rounded-2xl cursor-pointer
                          hover:bg-slate-50 hover:border-blue-300 transition-colors">
          {busy ? <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                : <FileUp className="w-8 h-8 text-slate-400" />}
          <span className="text-sm text-slate-600 font-medium">
            {busy ? 'Processando nuvem (a parte cara roda só uma vez)...'
                  : 'Solte a nuvem de pontos (.ply, .e57, .xyz)'}
          </span>
          <input type="file" accept=".ply,.e57,.xyz" className="hidden"
                 onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        </label>
        {erro && <ErroBox msg={erro} />}
      </div>
    );
  }

  const banda = bandas[bandaIdx];

  // finalização opcional: reusa o editor da planta com as paredes aprovadas
  if (modeloEditor) {
    return (
      <PlantaEditor
        modeloInicial={modeloEditor}
        onGerar={gerarDoEditor}
        ocultarPreviewPly
        titulo="Finalizar planta do scan"
        subtitulo="Edite paredes, portas e janelas EM CIMA do mapa da nuvem captada; o tracejado é a leitura do teto. Depois gere o IFC."
        onVoltar={() => setModeloEditor(null)}
        rotuloVoltar="Voltar às paredes"
        backdropPng={preview?.png}
        backdropBounds={preview?.bounds}
      />
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-8 pt-6 flex items-end justify-between">
        <Cabecalho />
        {/* stepper */}
        <div className="flex items-center gap-1 text-xs">
          {([['1', 'Lajes'], ['2', 'Paredes'], ['3', 'Escadas']] as const).map(([n, nome], i) => (
            <React.Fragment key={n}>
              {i > 0 && <ChevronRight className="w-3.5 h-3.5 text-slate-300" />}
              <button onClick={() => setFase(+n as Fase)}
                className={`px-3 py-1.5 rounded-lg font-medium transition-colors
                  ${fase === +n ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}>
                {n}. {nome}
              </button>
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="flex-1 flex gap-4 px-8 pb-6 pt-4 min-h-0">
        {/* ================= VISUAL ================= */}
        <div className="flex-1 bg-white rounded-xl border border-slate-200 relative overflow-hidden">
          {carregando && (
            <div className="absolute top-3 right-3 z-10">
              <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
            </div>
          )}

          {fase === 1 && <HistogramaLajes zHist={up.z_hist} lajes={lajes} />}

          {fase === 2 && (preview ? (
            <SvgPlanta preview={preview} lances={null} />
          ) : (
            <Vazio msg={bandas.length ? 'Calculando preview...' : 'Nenhum pavimento — reduza o threshold na fase 1'} />
          ))}

          {fase === 3 && (preview ? (
            <SvgPlanta preview={preview} lances={lances} />
          ) : (
            <Vazio msg="Passe pela fase 2 primeiro (o fundo da planta vem de lá)" />
          ))}

          <div className="absolute bottom-3 left-3 text-[11px] text-slate-500 bg-white/85 rounded px-2 py-1">
            {fase === 1 && `${lajes.length} nível(is) de laje → ${bandas.length} pavimento(s)`}
            {fase === 2 && preview && (
              <><span className="text-red-600 font-semibold">{preview.n_paredes ?? 0} paredes</span>
                {' '}· <span className="text-slate-400">{preview.n_segmentos} contornos</span>
                {' '}· <span className="text-amber-600">âmbar = single-line</span>
                {cfgPreview && <span className="text-slate-400"> · desenho de: {cfgPreview}</span>}</>
            )}
            {fase === 3 && lances && (
              <><span className="text-emerald-700 font-semibold">{lances.length} lance(s) de escada</span>
                {cfgPreview && <span className="text-slate-400"> · fundo de: {cfgPreview}</span>}</>
            )}
          </div>
        </div>

        {/* ================= PAINEL ================= */}
        <div className="w-80 shrink-0 flex flex-col gap-4 overflow-auto">
          <div className="bg-white rounded-xl border border-slate-200 p-4 text-xs text-slate-600">
            <div className="font-semibold text-slate-800 mb-1">Nuvem carregada</div>
            {up.n_pontos.toLocaleString()} pontos · {up.extent[0]}×{up.extent[1]}×{up.extent[2]} m
          </div>

          {fase === 1 && (
            <Painel titulo="Fase 1 — Lajes / pavimentos" icone={<Layers className="w-4 h-4 text-blue-600" />}>
              <Slider label={`Threshold de laje: ${thr.toFixed(2)}`}
                      min={0.1} max={0.6} step={0.05} value={thr} onChange={setThr} />
              <div className="text-[11px] text-slate-500">
                sugerido: {up.thr_sugerido} · picos altos no gráfico = superfícies horizontais;
                o threshold decide quais viram laje
              </div>
              {lajes.map((l, i) => (
                <div key={i} className="text-[11px] text-slate-600 bg-slate-50 rounded px-2 py-1">
                  Laje {i + 1}: z {l[0].toFixed(2)} a {l[1].toFixed(2)} m
                </div>
              ))}
            </Painel>
          )}

          {fase >= 2 && bandas.length > 1 && (
            <select value={bandaIdx} onChange={(e) => setBandaIdx(+e.target.value)}
                    className="border border-slate-200 rounded-lg px-2 py-2 text-xs bg-white">
              {bandas.map((b, i) => (
                <option key={i} value={i}>Pavimento {i + 1} (z {b[0].toFixed(1)} a {b[1].toFixed(1)})</option>
              ))}
            </select>
          )}

          {fase === 2 && (
            <Painel titulo="Fase 2 — Paredes" icone={<ScanLine className="w-4 h-4 text-blue-600" />}>
              <Slider label={`Base da fatia: ${(zloFrac * 100).toFixed(0)}% do pé-direito`}
                      min={0} max={0.8} step={0.05} value={zloFrac} onChange={setZloFrac} />
              <Slider label={`Topo da fatia: ${(zhiFrac * 100).toFixed(0)}%`}
                      min={0.2} max={1} step={0.05} value={zhiFrac} onChange={setZhiFrac} />
              <p className="text-[11px] text-slate-400 -mt-1">
                Fatia alta ignora móveis; fatia cheia pega divisórias baixas.
              </p>
              <Slider label={`Comprimento mínimo: ${minLen.toFixed(2)} m`}
                      min={0.1} max={2} step={0.1} value={minLen} onChange={setMinLen} />
              <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
                <input type="checkbox" checked={contoursAll}
                       onChange={(e) => setContoursAll(e.target.checked)} />
                Contornos internos (cômodos)
              </label>
            </Painel>
          )}

          {fase === 3 && (
            <Painel titulo="Fase 3 — Escadas" icone={<TrendingUp className="w-4 h-4 text-blue-600" />}>
              <Slider label={`Sensibilidade (área mín.): ${areaMin.toFixed(1)} m²`}
                      min={0.4} max={3} step={0.2} value={areaMin} onChange={setAreaMin} />
              <p className="text-[11px] text-slate-400 -mt-1">
                Menor = acha lances menores (e mais ruído).
              </p>
              {(lances || []).map((L, i) => (
                <div key={i} className="text-[11px] text-slate-600 bg-emerald-50 rounded px-2 py-1">
                  Lance {i + 1}: {L.comprimento}m × {L.largura}m · sobe z {L.z0}→{L.z1}
                  {L.espelho_cm ? ` · espelho ${L.espelho_cm}cm` : ' · espelho: norma'}
                  {` · ${L.degraus_vistos} dg vistos`}
                </div>
              ))}
              {lances && !lances.length && (
                <div className="text-[11px] text-slate-400">nenhum lance nesta banda</div>
              )}
            </Painel>
          )}

          <button onClick={gerar} disabled={!!job || !bandas.length || carregando || !preview}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg
                       text-sm font-semibold bg-blue-600 text-white hover:bg-blue-700
                       disabled:opacity-40 transition-colors">
            {job ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {job ? `Gerando (${jobEtapa})...` : 'Gerar IFC com esses thresholds'}
          </button>

          <button onClick={abrirEditor} disabled={!!job || carregando || !preview?.eixos.length}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg
                       text-sm font-medium border border-blue-300 text-blue-700 bg-white
                       hover:bg-blue-50 disabled:opacity-40 transition-colors">
            <PencilRuler className="w-4 h-4" />
            Finalizar no editor (portas / janelas)
          </button>

          {detalhes && (
            <div className="bg-white rounded-lg border border-slate-200 px-3 py-2.5 text-xs flex flex-col gap-1">
              <span className="font-semibold text-slate-700 mb-0.5">Etapas da montagem</span>
              {Object.entries(detalhes).map(([etapa, res]) => {
                const ok = res === 'ok' || res.startsWith('ok — ');
                const msg = res === 'ok' ? '' : res.replace(/^ok — /, '');
                return (
                  <div key={etapa} className="flex items-start gap-1.5">
                    {ok ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
                        : <AlertCircle className="w-3.5 h-3.5 text-amber-500 shrink-0 mt-0.5" />}
                    <span className={ok ? 'text-slate-600' : 'text-amber-700'}>
                      <span className="font-medium capitalize">{etapa}</span>
                      {msg && <span className={ok ? 'text-emerald-700' : 'text-slate-500'}> — {msg}</span>}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {ifcUrl && (
            <div className="flex flex-col gap-2 bg-emerald-50 rounded-lg px-3 py-3 text-xs">
              <span className="flex items-center gap-2 text-emerald-700 font-medium">
                <CheckCircle2 className="w-4 h-4" /> IFC gerado
              </span>
              <a href={downloadUrl(ifcUrl)} download
                 className="flex items-center gap-1 justify-center px-2 py-1.5 bg-emerald-600 text-white rounded-md hover:bg-emerald-700">
                <Download className="w-3.5 h-3.5" /> Baixar IFC
              </a>
            </div>
          )}
          {erro && <ErroBox msg={erro} />}

          <button onClick={() => { setUp(null); setPreview(null); setIfcUrl(null); setJob(null); setLances(null); }}
            className="flex items-center justify-center gap-1 text-xs text-slate-400 hover:text-slate-600">
            <RotateCcw className="w-3 h-3" /> Trocar nuvem
          </button>
        </div>
      </div>
    </div>
  );
};

// =====================================================================
//  VISUAIS
// =====================================================================

/** Fase 1: histograma de z com os níveis de laje marcados. */
const HistogramaLajes: React.FC<{
  zHist: { zmin: number; step: number; counts: number[] };
  lajes: [number, number][];
}> = ({ zHist, lajes }) => {
  const { zmin, step, counts } = zHist;
  const max = Math.max(...counts, 1);
  const n = counts.length;
  const W = 100, H = 60;   // viewBox em unidades relativas
  return (
    <div className="w-full h-full p-6 flex flex-col">
      <div className="text-xs text-slate-500 mb-2">
        Distribuição de pontos por altura (z) — <span className="text-red-600">faixas vermelhas = lajes detectadas</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="flex-1 w-full" preserveAspectRatio="none">
        {lajes.map((l, i) => {
          const x0 = ((l[0] - zmin) / (n * step)) * W;
          const x1 = ((l[1] - zmin) / (n * step)) * W;
          return <rect key={`l${i}`} x={x0} y={0} width={Math.max(x1 - x0, 0.5)} height={H}
                       fill="#fecaca" opacity={0.7} />;
        })}
        {counts.map((c, i) => {
          const h = (Math.log1p(c) / Math.log1p(max)) * (H - 4);
          return <rect key={i} x={(i / n) * W} y={H - h} width={W / n * 0.9} height={h}
                       fill="#3b82f6" opacity={0.75} />;
        })}
      </svg>
      <div className="flex justify-between text-[10px] text-slate-400 mt-1">
        <span>z = {zmin.toFixed(1)} m</span>
        <span>z = {(zmin + n * step).toFixed(1)} m</span>
      </div>
    </div>
  );
};

/** Fases 2 e 3: planta com camadas (contornos, paredes, lances). */
const SvgPlanta: React.FC<{
  preview: ScanParedesResult; lances: ScanLance[] | null;
}> = ({ preview, lances }) => {
  const [xmin, ymin, xmax, ymax] = preview.bounds;
  return (
    <svg viewBox={`${xmin} ${-ymax} ${xmax - xmin} ${ymax - ymin}`} className="w-full h-full">
      <image href={`data:image/png;base64,${preview.png}`}
             x={xmin} y={-ymax} width={xmax - xmin} height={ymax - ymin}
             opacity={0.4} preserveAspectRatio="none"
             style={{ imageRendering: 'pixelated' }} />
      {(preview.segmentos || []).map((s, i) => (
        <line key={`p${i}`} x1={s[0]} y1={-s[1]} x2={s[2]} y2={-s[3]}
              stroke="#94a3b8" strokeWidth={(xmax - xmin) / 700} opacity={0.7} />
      ))}
      {(preview.eixos || []).map((e, i) => (
        <line key={`w${i}`} x1={e[0]} y1={-e[1]} x2={e[2]} y2={-e[3]}
              stroke={e[5] === 'single' ? '#f59e0b' : '#dc2626'}
              strokeWidth={Math.max(e[4], (xmax - xmin) / 500)}
              strokeLinecap="round" opacity={lances ? 0.35 : 0.8} />
      ))}
      {(lances || []).map((L, i) => {
        const px = -L.uy, py = L.ux;
        const c = L.comprimento / 2, w = L.largura / 2;
        const q = [
          [L.cx - L.ux * c + px * w, L.cy - L.uy * c + py * w],
          [L.cx + L.ux * c + px * w, L.cy + L.uy * c + py * w],
          [L.cx + L.ux * c - px * w, L.cy + L.uy * c - py * w],
          [L.cx - L.ux * c - px * w, L.cy - L.uy * c - py * w],
        ];
        return (
          <polygon key={`e${i}`}
                   points={q.map(([x, y]) => `${x},${-y}`).join(' ')}
                   fill="rgba(16,185,129,0.25)" stroke="#059669"
                   strokeWidth={(xmax - xmin) / 500} />
        );
      })}
    </svg>
  );
};

const Vazio: React.FC<{ msg: string }> = ({ msg }) => (
  <div className="h-full flex items-center justify-center text-sm text-slate-400">{msg}</div>
);

const Painel: React.FC<{ titulo: string; icone: React.ReactNode; children: React.ReactNode }> =
  ({ titulo, icone, children }) => (
    <div className="bg-white rounded-xl border border-slate-200 p-4 flex flex-col gap-3">
      <span className="text-sm font-semibold text-slate-800 flex items-center gap-2">
        {icone} {titulo}
      </span>
      {children}
    </div>
  );

const Cabecalho: React.FC = () => (
  <div>
    <h1 className="text-xl font-bold text-slate-800">Scan → BIM</h1>
    <p className="text-sm text-slate-500">
      Calibre por fases — lajes, paredes, escadas — vendo o que o sistema capta, e gere o IFC.
    </p>
  </div>
);

const ErroBox: React.FC<{ msg: string }> = ({ msg }) => (
  <div className="flex items-center gap-2 text-xs text-rose-600 bg-rose-50 rounded-lg px-3 py-2 max-w-2xl mt-3">
    <AlertCircle className="w-4 h-4 shrink-0" /> {msg}
  </div>
);

const Slider: React.FC<{
  label: string; min: number; max: number; step: number;
  value: number; onChange: (v: number) => void;
}> = ({ label, min, max, step, value, onChange }) => (
  <label className="flex flex-col gap-1 text-xs text-slate-600">
    {label}
    <input type="range" min={min} max={max} step={step} value={value}
           onChange={(e) => onChange(+e.target.value)} className="w-full accent-blue-600" />
  </label>
);
