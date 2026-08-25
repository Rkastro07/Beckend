import React, { useEffect, useRef, useState } from 'react';
import {
  FileUp, Loader2, Download, RotateCcw, ScanLine, Layers,
  CheckCircle2, AlertCircle, Sparkles, ChevronRight, TrendingUp, PencilRuler,
} from 'lucide-react';
import {
  scanUpload, scanLajes, scanParedes, scanEscadas, scanHibrido, scanGerarIfc, scanJob, scanCloudPreview, downloadUrl,
  ScanUploadResult, ScanParedesResult, ScanLance, ScanAbertura, ScanFatias,
  ScanHybridResult, ModeloPlanta, PlantaConfig,
} from '../services/tools';
import { PlantaEditor, ResultadoGeracao } from './PlantaEditor';

type Fase = 1 | 2 | 3;

type PerfilAutomatico = {
  id: string;
  zloFrac: number;
  zhiFrac: number;
  minLen: number;
  contoursAll: boolean;
  singleMinlen: number;
};

// Perfis internos do Detector V2. Eles cobrem nuvens com paredes completas,
// mobiliario/oclusao e plantas predominantemente desenhadas em linha unica.
// A interface nunca pede que o cliente escolha estes valores.
const PERFIS_AUTOMATICOS: PerfilAutomatico[] = [
  { id: 'padrao',   zloFrac: 0.10, zhiFrac: 0.90, minLen: 0.30, contoursAll: true,  singleMinlen: 1.50 },
  { id: 'linha',    zloFrac: 0.08, zhiFrac: 0.92, minLen: 0.20, contoursAll: true,  singleMinlen: 0.80 },
  { id: 'oclusao',  zloFrac: 0.22, zhiFrac: 0.78, minLen: 0.20, contoursAll: true,  singleMinlen: 1.00 },
  { id: 'baixo',    zloFrac: 0.04, zhiFrac: 0.58, minLen: 0.20, contoursAll: true,  singleMinlen: 0.80 },
  { id: 'envoltoria', zloFrac: 0.08, zhiFrac: 0.92, minLen: 0.25, contoursAll: false, singleMinlen: 1.00 },
];

const comprimentoTotal = (r: ScanParedesResult) => r.eixos.reduce(
  (s, e) => s + Math.hypot(e[2] - e[0], e[3] - e[1]), 0);

// Prefere cobertura arquitetonica util, sem premiar centenas de fragmentos.
const pontuarLeitura = (r: ScanParedesResult) => {
  const comprimentos = r.eixos
    .map((e) => Math.hypot(e[2] - e[0], e[3] - e[1]))
    .filter((v) => Number.isFinite(v) && v > 0);
  const uteis = comprimentos.filter((v) => v >= 0.45);
  const curtos = comprimentos.length - uteis.length;
  const excesso = Math.max(0, comprimentos.length - 220);
  return uteis.reduce((s, v) => s + Math.min(v, 20), 0)
    + Math.min(uteis.length, 120) * 0.65
    - curtos * 0.35
    - excesso * 1.5;
};

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
  // As fases continuam existindo internamente, mas o cliente entra direto na
  // aprovação visual do Detector V2. Thresholds não fazem parte do produto.
  const [fase, setFase] = useState<Fase>(2);

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
  const [perfilAutomatico, setPerfilAutomatico] = useState<PerfilAutomatico>(PERFIS_AUTOMATICOS[0]);
  const [tentativaAutomatica, setTentativaAutomatica] = useState(0);

  // fase 2 — classificacao multi-fatia (porta/janela por assinatura vertical)
  // faixas em fracao do pe-direito; defaults = joelho / meio / sob a laje
  const [multiFatia, setMultiFatia] = useState(true);
  const [fBaixa, setFBaixa] = useState<[number, number]>([0.10, 0.25]);
  const [fMedia, setFMedia] = useState<[number, number]>([0.40, 0.60]);
  const [fAlta, setFAlta] = useState<[number, number]>([0.80, 0.92]);
  const fatias: ScanFatias = { baixa: fBaixa, media: fMedia, alta: fAlta };

  // fase 3 — escadas
  const [areaMin, setAreaMin] = useState(0.8);
  const [lances, setLances] = useState<ScanLance[] | null>(null);

  // geracao
  const [job, setJob] = useState<string | null>(null);
  const [jobEtapa, setJobEtapa] = useState('');
  const [ifcUrl, setIfcUrl] = useState<string | null>(null);
  const [detalhes, setDetalhes] = useState<Record<string, string> | null>(null);

  // revisão híbrida: geometria heurística em tiles + YOLO + classificador ML
  const [hybridJob, setHybridJob] = useState<string | null>(null);
  const [hybridEtapa, setHybridEtapa] = useState('');
  const [hybridResult, setHybridResult] = useState<ScanHybridResult | null>(null);

  // finalizacao opcional no editor (portas/janelas/teto)
  const [modeloEditor, setModeloEditor] = useState<ModeloPlanta | null>(null);

  // ---------- upload ----------
  const onFile = async (f: File) => {
    setBusy(true); setErro(null); setUp(null); setPreview(null);
    setIfcUrl(null); setDetalhes(null); setLances(null); setFase(2);
    setHybridJob(null); setHybridEtapa(''); setHybridResult(null);
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
      try {
        let melhor: { resultado: ScanParedesResult; perfil: PerfilAutomatico; pontos: number } | null = null;
        const diagonal = Math.hypot(up.extent[0] || 0, up.extent[1] || 0);

        for (let i = 0; i < PERFIS_AUTOMATICOS.length; i += 1) {
          if (seq2.current !== minha) return;
          const perfil = PERFIS_AUTOMATICOS[i];
          setTentativaAutomatica(i + 1);
          const resultado = await scanParedes(
            up.sid, b0, b1,
            perfil.zloFrac, perfil.zhiFrac, perfil.minLen, perfil.contoursAll,
            multiFatia ? fatias : undefined,
            perfil.singleMinlen,
          );
          const pontos = pontuarLeitura(resultado);
          if (!melhor || pontos > melhor.pontos) melhor = { resultado, perfil, pontos };

          // Uma leitura que ja cobre uma parcela plausivel da planta nao precisa
          // pagar o custo dos perfis de recuperacao.
          if (resultado.n_paredes >= 4
              && comprimentoTotal(resultado) >= Math.max(8, diagonal * 0.65)) break;
        }

        if (seq2.current !== minha || !melhor) return;
        setPreview(melhor.resultado);
        setPerfilAutomatico(melhor.perfil);
        setCfgPreview(`Leitura automática · perfil ${melhor.perfil.id}`);
        setErro(null);
      } catch (e: any) { if (seq2.current === minha) setErro(e.message); }
      finally {
        if (seq2.current === minha) {
          setCarregando(false);
          setTentativaAutomatica(0);
        }
      }
    }, 350);
    return () => { if (timer2.current) window.clearTimeout(timer2.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [up, fase, bandas, bandaIdx, zloFrac, zhiFrac, minLen, contoursAll,
      multiFatia, fBaixa, fMedia, fAlta]);

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
        if (st.status === 'pronto') {
          setIfcUrl(st.url || null);
          setJob(null);
        }
        if (st.status === 'erro') { setErro(st.erro || 'Falha na geração'); setJob(null); }
      } catch { /* tenta de novo */ }
    }, 2500);
    return () => window.clearInterval(h);
  }, [job]);

  useEffect(() => {
    if (!hybridJob) return;
    const h = window.setInterval(async () => {
      try {
        const st = await scanJob(hybridJob);
        setHybridEtapa(st.etapa || 'Processando');
        if (st.status === 'pronto') {
          if (!st.hybrid) throw new Error('O backend não devolveu a revisão híbrida.');
          setHybridResult(st.hybrid);
          setHybridJob(null);
        }
        if (st.status === 'erro') {
          setErro(st.erro || 'Falha na análise híbrida');
          setHybridJob(null);
        }
      } catch (e: any) {
        setErro(e.message || 'Falha ao consultar a análise híbrida');
        setHybridJob(null);
      }
    }, 2000);
    return () => window.clearInterval(h);
  }, [hybridJob]);

  const rodarHibrido = async () => {
    if (!up || hybridJob) return;
    const selectedBand = bandas[bandaIdx];
    if (!selectedBand) {
      setErro('Nenhum pavimento disponível para a análise híbrida.');
      return;
    }
    setErro(null);
    setHybridResult(null);
    setHybridEtapa('Enviando para o pipeline híbrido');
    try {
      const response = await scanHibrido(up.sid, selectedBand[0], selectedBand[1]);
      setHybridJob(response.job);
    } catch (e: any) {
      setErro(e.message || 'Falha ao iniciar a análise híbrida');
    }
  };

  const gerar = async () => {
    if (!up || carregando || !preview) return;
    setErro(null); setIfcUrl(null); setDetalhes(null);
    setCarregando(true);
    try {
      const r = await scanGerarIfc(
        up.sid, thr,
        perfilAutomatico.zloFrac, perfilAutomatico.zhiFrac,
        perfilAutomatico.singleMinlen, perfilAutomatico.minLen,
        perfilAutomatico.contoursAll,
        areaMin, bandaIdx,
        preview.eixos,
      );
      setJob(r.job); setJobEtapa('pipeline');
    } catch (e: any) { setErro(e.message); }
    finally { setCarregando(false); }
  };

  // ---- finalização opcional: paredes aprovadas -> ModeloPlanta do editor ----
  const abrirEditor = () => {
    if (!preview) return;
    const [xmin, ymin, xmax, ymax] = preview.bounds;
    const [storeyBase, storeyTop] = bandas[bandaIdx] ?? [0, 2.8];
    const observedHeight = Math.max(0.1, storeyTop - storeyBase);
    const paredes = preview.eixos.map((e, i) => ({
      id: `w${i}`, ax: e[0], ay: e[1], bx: e[2], by: e[3],
      espessura: e[4], layer: e[5], altura: observedHeight,
    }));
    const aberturas = (preview.classificacao ?? [])
      .filter((trecho) => trecho[4] === 'porta' || trecho[4] === 'janela')
      .map((trecho, index) => {
        const eixoIdx = trecho[5];
        const parede = paredes[eixoIdx];
        if (!parede) return null;
        const dx = parede.bx - parede.ax;
        const dy = parede.by - parede.ay;
        const comprimento = Math.hypot(dx, dy);
        const largura = Math.hypot(trecho[2] - trecho[0], trecho[3] - trecho[1]);
        if (comprimento < 0.05 || largura < 0.40) return null;
        const ux = dx / comprimento;
        const uy = dy / comprimento;
        const centroX = (trecho[0] + trecho[2]) / 2;
        const centroY = (trecho[1] + trecho[3]) / 2;
        const sCentro = Math.min(
          comprimento - largura / 2,
          Math.max(largura / 2,
            (centroX - parede.ax) * ux + (centroY - parede.ay) * uy),
        );
        const tipo = trecho[4] === 'janela' ? 'window' as const : 'door' as const;
        const peitoril = tipo === 'window'
          ? Math.min(1.0, Math.max(0.45, observedHeight * 0.35))
          : 0;
        const altura = tipo === 'window'
          ? Math.min(1.20, Math.max(0.40, observedHeight - peitoril - 0.15))
          : Math.min(2.10, Math.max(0.60, observedHeight - 0.10));
        return {
          id: `O-SCAN-${String(index + 1).padStart(3, '0')}`,
          parede_id: parede.id,
          tipo,
          s_centro: sCentro,
          largura: Math.min(largura, comprimento),
          altura,
          peitoril,
          origem: 'scan-multifatia',
          confidence: 0.65,
        };
      })
      .filter((abertura): abertura is NonNullable<typeof abertura> => abertura !== null);
    const contorno: [number, number][] =
      preview.contorno_teto && preview.contorno_teto.length >= 3
        ? preview.contorno_teto
        : [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]];
    setModeloEditor({
      escala: 1, single_line: false, nome: 'scan',
      bbox: { xmin, ymin, xmax, ymax },
      diagnostico: { sobras: 0, cantos_costurados: 0, blocos_esquadria: 0 },
      paredes, aberturas,
      laje: { contorno, piso: { ativo: true, espessura: 0.15 },
              teto: { ativo: true, espessura: 0.15 } },
    });
  };

  const abrirEditorHibrido = () => {
    if (!hybridResult) return;
    const [xmin, ymin, xmax, ymax] = hybridResult.bounds;
    const observedHeight = Math.max(
      0.30, hybridResult.ceiling_z - hybridResult.floor_z);
    const paredes = hybridResult.walls.map((wall) => ({
      id: wall.id,
      ax: wall.ax, ay: wall.ay, bx: wall.bx, by: wall.by,
      espessura: wall.espessura,
      layer: 'scan-hybrid',
      altura: observedHeight,
      elevacao: hybridResult.floor_z,
      origem: 'scan-hybrid-ml',
      confidence: wall.wall_probability,
      ml_status: wall.ml_class,
      ml_probability: wall.ml_probability,
      ml_proposed_keep: wall.proposed_keep,
    }));
    const wallById = new Map(paredes.map((wall) => [wall.id, wall]));
    const aberturas = hybridResult.openings.flatMap((opening) => {
      const wall = wallById.get(opening.wall_id);
      if (!wall) return [];
      const length = Math.hypot(wall.bx - wall.ax, wall.by - wall.ay);
      const width = Math.min(Math.max(0.20, opening.width), length);
      const center = Math.min(
        Math.max(width / 2, opening.s_center),
        Math.max(width / 2, length - width / 2),
      );
      return [{
        id: opening.id,
        parede_id: opening.wall_id,
        tipo: opening.class,
        s_centro: center,
        largura: width,
        altura: opening.height,
        peitoril: opening.class === 'window' ? opening.sill : 0,
        origem: 'scan-hybrid-yolo',
        confidence: opening.confidence,
      }];
    });
    const contorno: [number, number][] =
      preview?.contorno_teto && preview.contorno_teto.length >= 3
        ? preview.contorno_teto
        : [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]];
    setModeloEditor({
      escala: 1,
      single_line: false,
      nome: 'scan-hybrid-review',
      bbox: { xmin, ymin, xmax, ymax },
      diagnostico: {
        sobras: hybridResult.counts.proposed_remove,
        cantos_costurados: 0,
        blocos_esquadria: hybridResult.counts.openings,
      },
      paredes,
      aberturas,
      laje: {
        contorno,
        piso: { ativo: true, espessura: 0.15 },
        teto: { ativo: true, espessura: 0.15 },
      },
    });
  };

  // gerador do editor: as PAREDES e as portas/janelas EDITADAS na tela viram
  // overrides na cadeia cloud2bim (o IFC = exatamente o que o cliente desenhou)
  const gerarDoEditor = async (
    modelo: {
      paredes: any[];
      aberturas: any[];
      laje: ModeloPlanta['laje'];
      spaces?: ModeloPlanta['spaces'];
    }, cfg: PlantaConfig,
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
          : {
              eixo_idx,
              tipo: a.tipo,
              s_centro: a.s_centro,
              largura: a.largura,
              altura: a.altura,
              peitoril: a.tipo === 'window' ? a.peitoril : 0,
            };
      })
      .filter((a): a is ScanAbertura => a !== null);
    const { job: jid } = await scanGerarIfc(
      up.sid, thr,
      perfilAutomatico.zloFrac, perfilAutomatico.zhiFrac,
      perfilAutomatico.singleMinlen, perfilAutomatico.minLen,
      perfilAutomatico.contoursAll,
      areaMin, bandaIdx, eixos, aberturas, cfg, modelo);
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
        subtitulo="Revise paredes, portas e janelas sobre a nuvem. No fluxo híbrido: verde = parede, laranja = folha, vermelho = falso candidato e cinza = incerto. Nada é removido sem sua confirmação."
        onVoltar={() => setModeloEditor(null)}
        rotuloVoltar="Voltar às paredes"
        backdropPng={preview?.png}
        backdropBounds={preview?.bounds}
        // A base visual e a face inferior da laje do pavimento. Usar banda[0]
        // zerava pelo topo da laje e deixava a propria espessura abaixo do grid.
        loadPointCloud={() => scanCloudPreview(
          up.sid,
          120_000,
          lajes[bandaIdx]?.[0] ?? banda?.[0] ?? 0,
        )}
      />
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-8 pt-6 flex items-end justify-between">
        <Cabecalho />
        <div className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium
          ${carregando || hybridJob ? 'bg-blue-50 text-blue-700'
            : preview && !preview.eixos.length ? 'bg-amber-50 text-amber-700'
            : 'bg-emerald-50 text-emerald-700'}`}>
          {carregando || hybridJob
            ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
            : preview && !preview.eixos.length
              ? <AlertCircle className="h-3.5 w-3.5" />
              : <CheckCircle2 className="h-3.5 w-3.5" />}
          {hybridJob ? `Pipeline híbrido · ${hybridEtapa}`
            : carregando ? 'Detector V2 analisando'
            : preview && !preview.eixos.length ? 'Revisão necessária'
            : hybridResult ? 'Revisão híbrida pronta'
            : 'Detector V2 automático'}
        </div>
      </div>

      <div className="flex-1 flex gap-4 px-8 pb-6 pt-4 min-h-0">
        {/* ================= VISUAL ================= */}
        <div className="flex-1 bg-white rounded-xl border border-slate-200 relative overflow-hidden">
          {(carregando || hybridJob) && (
            <div className="absolute top-3 right-3 z-10">
              <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
            </div>
          )}

          {hybridResult ? (
            <img
              src={downloadUrl(hybridResult.png_url)}
              alt="Revisão híbrida ML e heurística"
              className="h-full w-full object-contain bg-slate-50"
            />
          ) : preview ? (
            <SvgPlanta preview={preview} lances={null} />
          ) : (
            <Vazio msg={bandas.length ? 'Detector V2 analisando a planta...' : 'Nenhum pavimento detectado automaticamente'} />
          )}

          <div className="absolute bottom-3 left-3 text-[11px] text-slate-500 bg-white/85 rounded px-2 py-1">
            {hybridResult ? (
              <><span className="font-semibold text-emerald-700">ML + heurística</span>
                {' '}· {hybridResult.counts.input_walls} candidatos
                {' '}· {hybridResult.counts.openings} aberturas</>
            ) : preview ? (
              <><span className="text-red-600 font-semibold">{preview.n_paredes ?? 0} paredes</span>
                {' '}· <span className="text-slate-400">{preview.n_segmentos} contornos</span>
                {' '}· <span className="text-amber-600">âmbar = linha única</span></>
            ) : `${lajes.length} nível(is) · ${bandas.length} pavimento(s)`}
          </div>
        </div>

        {/* ================= PAINEL ================= */}
        <div className="w-80 shrink-0 flex flex-col gap-4 overflow-auto">
          <div className="bg-white rounded-xl border border-slate-200 p-4 text-xs text-slate-600">
            <div className="font-semibold text-slate-800 mb-1">Nuvem carregada</div>
            {up.n_pontos.toLocaleString()} pontos · {up.extent[0]}×{up.extent[1]}×{up.extent[2]} m
          </div>

          <Painel titulo="Detector V2 automático" icone={<ScanLine className="w-4 h-4 text-emerald-600" />}>
            <div className="rounded-lg bg-emerald-50 px-3 py-2 text-[11px] text-emerald-800">
              Lajes, paredes e candidatos a aberturas são calculados automaticamente.
              Corrija as exceções no editor 2D/3D.
            </div>
            <div className="grid grid-cols-2 gap-2 text-center">
              <div className="rounded-lg bg-slate-50 px-2 py-2">
                <div className="text-base font-semibold text-slate-800">{bandas.length}</div>
                <div className="text-[10px] text-slate-500">pavimentos</div>
              </div>
              <div className="rounded-lg bg-slate-50 px-2 py-2">
                <div className="text-base font-semibold text-slate-800">{preview?.n_paredes ?? '—'}</div>
                <div className="text-[10px] text-slate-500">paredes detectadas</div>
              </div>
            </div>
            {carregando && (
              <div className="flex items-center gap-2 text-[11px] text-blue-600">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                {tentativaAutomatica
                  ? `Validando leitura automática (${tentativaAutomatica}/${PERFIS_AUTOMATICOS.length})…`
                  : 'Atualizando leitura automática…'}
              </div>
            )}
            {!carregando && preview && (
              <div className="text-[10px] text-slate-400">{cfgPreview}</div>
            )}
            {!carregando && preview && !preview.eixos.length && (
              <div className="rounded-lg bg-amber-50 px-3 py-2 text-[11px] text-amber-800">
                Nenhuma parede passou pela validação automática. Abra o editor
                para desenhar sobre a planta captada, sem regular thresholds.
              </div>
            )}
          </Painel>

          {false && fase === 1 && (
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

          {false && fase === 2 && (
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

              <div className="border-t border-slate-200 mt-1 pt-2">
                <label className="flex items-center gap-2 text-xs font-medium text-slate-700 cursor-pointer">
                  <input type="checkbox" checked={multiFatia}
                         onChange={(e) => setMultiFatia(e.target.checked)} />
                  Detectar portas / janelas (multi-fatia)
                </label>
                {multiFatia && (
                  <div className="mt-2 flex flex-col gap-2.5">
                    <p className="text-[11px] text-slate-400">
                      3 faixas de altura (% do pé-direito). A porta é vão até o chão;
                      a janela tem peitoril embaixo. Arraste olhando o resultado.
                    </p>
                    <RangeDuplo label="Fatia baixa (peitoril)" cor="#2563eb"
                                value={fBaixa} onChange={setFBaixa} />
                    <RangeDuplo label="Fatia média (vão)" cor="#f59e0b"
                                value={fMedia} onChange={setFMedia} />
                    <RangeDuplo label="Fatia alta (verga/parede)" cor="#334155"
                                value={fAlta} onChange={setFAlta} />
                    {preview?.classificacao && (
                      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] pt-0.5">
                        <LegClass cor="#334155" txt="parede" />
                        <LegClass cor="#f59e0b" txt="porta" />
                        <LegClass cor="#2563eb" txt="janela" />
                        <LegClass cor="#dc2626" txt="oclusão" />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </Painel>
          )}

          {false && fase === 3 && (
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

          <Painel titulo="ML + heurística em blocos" icone={<Sparkles className="w-4 h-4 text-violet-600" />}>
            <div className="rounded-lg bg-violet-50 px-3 py-2 text-[11px] text-violet-800">
              A heurística preserva escala e geometria; o YOLO encontra portas e
              janelas; a ML colore paredes suspeitas para confirmação no editor.
            </div>
            {hybridResult ? (
              <>
                <div className="grid grid-cols-3 gap-1.5 text-center">
                  <div className="rounded-lg bg-emerald-50 px-1 py-2">
                    <div className="font-semibold text-emerald-700">{hybridResult.counts.wall}</div>
                    <div className="text-[9px] text-emerald-700">paredes</div>
                  </div>
                  <div className="rounded-lg bg-amber-50 px-1 py-2">
                    <div className="font-semibold text-amber-700">{hybridResult.counts.door_leaf}</div>
                    <div className="text-[9px] text-amber-700">folhas</div>
                  </div>
                  <div className="rounded-lg bg-rose-50 px-1 py-2">
                    <div className="font-semibold text-rose-700">{hybridResult.counts.non_wall}</div>
                    <div className="text-[9px] text-rose-700">não-paredes</div>
                  </div>
                </div>
                <div className="text-[10px] text-slate-500">
                  {hybridResult.counts.doors} portas · {hybridResult.counts.windows} janelas
                  {' '}· {hybridResult.elapsed_seconds.toFixed(1)} s
                </div>
                <div className="flex gap-2 text-[10px]">
                  <a href={downloadUrl(hybridResult.png_url)} download
                    className="text-violet-700 hover:underline">PNG</a>
                  <a href={downloadUrl(hybridResult.predictions_url)} download
                    className="text-violet-700 hover:underline">Decisões JSON</a>
                  <a href={downloadUrl(hybridResult.model_url)} download
                    className="text-violet-700 hover:underline">Modelo JSON</a>
                </div>
              </>
            ) : hybridJob ? (
              <div className="flex items-center gap-2 text-[11px] text-violet-700">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> {hybridEtapa}
              </div>
            ) : null}
            <button type="button" onClick={() => void rodarHibrido()}
              disabled={!!hybridJob || !!job || carregando || !bandas.length}
              className="w-full flex items-center justify-center gap-2 rounded-lg bg-violet-600 px-3 py-2 text-xs font-semibold text-white hover:bg-violet-700 disabled:opacity-40">
              {hybridJob ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
              {hybridResult ? 'Rodar novamente' : 'Rodar ML + heurística'}
            </button>
          </Painel>

          {hybridResult && (
            <button onClick={abrirEditorHibrido} disabled={!!job || carregando}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg
                         text-sm font-semibold bg-violet-600 text-white hover:bg-violet-700
                         disabled:opacity-40 transition-colors">
              <PencilRuler className="w-4 h-4" />
              Confirmar híbrido no editor
            </button>
          )}

          <button onClick={abrirEditor} disabled={!!job || carregando || !preview}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg
                       text-sm font-semibold border border-blue-200 bg-white text-blue-700 hover:bg-blue-50
                       disabled:opacity-40 transition-colors">
            <PencilRuler className="w-4 h-4" />
            Revisar detector rápido
          </button>

          <button onClick={() => void gerar()} disabled={!!job || !bandas.length || carregando || !preview?.eixos.length}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg
                       text-sm font-semibold bg-emerald-600 text-white hover:bg-emerald-700
                       disabled:opacity-40 transition-colors">
            {job ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {job ? `Gerando (${jobEtapa})...` : 'Gerar IFC'}
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

          <button onClick={() => {
            setUp(null); setPreview(null); setIfcUrl(null); setJob(null); setLances(null);
            setHybridJob(null); setHybridEtapa(''); setHybridResult(null);
          }}
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
      {(() => {
        const clas = preview.classificacao || [];
        const temClas = clas.length > 0;
        // sem classificacao: eixos crus (vermelho/laranja como sempre).
        // com classificacao: eixos viram base fina e a cor conta a historia.
        const cor: Record<string, string> = {
          parede: '#334155', porta: '#f59e0b', janela: '#2563eb', oclusao: '#dc2626',
        };
        return (
          <>
            {(preview.eixos || []).map((e, i) => (
              <line key={`w${i}`} x1={e[0]} y1={-e[1]} x2={e[2]} y2={-e[3]}
                    stroke={temClas ? '#cbd5e1' : (e[5] === 'single' ? '#f59e0b' : '#dc2626')}
                    strokeWidth={temClas ? (xmax - xmin) / 900
                                         : Math.max(e[4], (xmax - xmin) / 500)}
                    strokeLinecap="round" opacity={lances ? 0.35 : (temClas ? 0.5 : 0.8)} />
            ))}
            {clas.map((c, i) => (
              <line key={`c${i}`} x1={c[0]} y1={-c[1]} x2={c[2]} y2={-c[3]}
                    stroke={cor[c[4] as string] || '#334155'}
                    strokeWidth={(xmax - xmin) / (c[4] === 'parede' ? 320 : 260)}
                    strokeLinecap="butt" opacity={lances ? 0.4 : 0.9}
                    strokeDasharray={c[4] === 'oclusao'
                      ? `${(xmax - xmin) / 120} ${(xmax - xmin) / 240}` : undefined} />
            ))}
          </>
        );
      })()}
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
      O Detector V2 monta a planta automaticamente; revise no editor 2D/3D e gere o IFC.
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

// estilo dos dois thumbs do RangeDuplo (thumb clicavel sobre track sem eventos)
let _rdStyle = false;
const injetarEstiloRD = () => {
  if (_rdStyle || typeof document === 'undefined') return;
  _rdStyle = true;
  const st = document.createElement('style');
  st.textContent =
    '.rd-range{-webkit-appearance:none;appearance:none;background:transparent;' +
    'position:absolute;left:0;top:0;width:100%;height:100%;margin:0;pointer-events:none}' +
    '.rd-range::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:14px;' +
    'height:14px;border-radius:9999px;background:#fff;border:2px solid #475569;cursor:pointer;' +
    'pointer-events:auto;box-shadow:0 1px 2px rgba(0,0,0,.3)}' +
    '.rd-range::-moz-range-thumb{width:14px;height:14px;border-radius:9999px;background:#fff;' +
    'border:2px solid #475569;cursor:pointer;pointer-events:auto}' +
    '.rd-range::-webkit-slider-runnable-track{background:transparent}' +
    '.rd-range::-moz-range-track{background:transparent}';
  document.head.appendChild(st);
};

/** Barra de intervalo com DOIS cursores (início/fim), em fração 0..1. */
const RangeDuplo: React.FC<{
  label: string; cor: string; value: [number, number];
  onChange: (v: [number, number]) => void;
}> = ({ label, cor, value, onChange }) => {
  injetarEstiloRD();
  const [lo, hi] = value;
  return (
    <label className="flex flex-col gap-1 text-xs text-slate-600">
      <span className="flex justify-between">
        <span>{label}</span>
        <span className="tabular-nums text-slate-400">
          {(lo * 100).toFixed(0)}–{(hi * 100).toFixed(0)}%
        </span>
      </span>
      <div className="relative h-4">
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-1.5 rounded-full bg-slate-200" />
        <div className="absolute top-1/2 -translate-y-1/2 h-1.5 rounded-full"
             style={{ left: `${lo * 100}%`, right: `${(1 - hi) * 100}%`, background: cor }} />
        <input type="range" min={0} max={1} step={0.02} value={lo} className="rd-range"
               onChange={(e) => onChange([Math.min(+e.target.value, hi - 0.02), hi])} />
        <input type="range" min={0} max={1} step={0.02} value={hi} className="rd-range"
               onChange={(e) => onChange([lo, Math.max(+e.target.value, lo + 0.02)])} />
      </div>
    </label>
  );
};

const LegClass: React.FC<{ cor: string; txt: string }> = ({ cor, txt }) => (
  <span className="flex items-center gap-1 text-slate-500">
    <span className="inline-block w-3 h-1.5 rounded-sm" style={{ background: cor }} /> {txt}
  </span>
);
