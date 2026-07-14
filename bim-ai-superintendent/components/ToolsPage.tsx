import React, { useState } from 'react';
import {
  FileUp, Download, Loader2, Box, Smartphone, ScanLine,
  Cloudy, Sparkles, SlidersHorizontal, Dice5, CheckCircle2, AlertCircle,
} from 'lucide-react';
import {
  objToPly, usdzToPly, ascToPly,
  gerarNuvemEstagio, gerarNuvemManual, downloadUrl,
  ToolResult, GerarNuvemResult,
} from '../services/tools';

type Secao = 'conversores' | 'geradores';

// Estágios de obra (batem com PERFIS_VARIANTES do gerar_sintetico.py)
const ESTAGIOS = [
  { v: '0', nome: 'Vistoria final', desc: 'quase tudo pronto (0-5% ausente)' },
  { v: '1', nome: 'Acabamento', desc: 'em andamento (5-15%)' },
  { v: '2', nome: 'Instalações', desc: 'sendo feitas (15-25%)' },
  { v: '3', nome: 'Vedações', desc: 'em andamento (25-35%)' },
  { v: '4', nome: 'Alvenaria', desc: 'incompleta (35-45%)' },
  { v: '5', nome: 'Estrutura + metade', desc: 'das vedações (45-55%)' },
  { v: '6', nome: 'Só estrutura', desc: 'início de vedação (55-65%)' },
  { v: '7', nome: 'Estrutura', desc: 'quase completa (65-75%)' },
  { v: '8', nome: 'Estrutura inicial', desc: '(75-85%)' },
  { v: '9', nome: 'Fundação / lajes', desc: 'apenas (85-95%)' },
];

// ---------- helper: input de arquivo estilizado ----------
const FilePicker: React.FC<{
  accept: string; file: File | null; onPick: (f: File) => void;
}> = ({ accept, file, onPick }) => (
  <label className="flex items-center gap-2 px-3 py-2 border border-dashed border-slate-300
                    rounded-lg cursor-pointer hover:bg-slate-50 text-sm text-slate-600">
    <FileUp className="w-4 h-4 shrink-0" />
    <span className="truncate">{file ? file.name : `Escolher arquivo (${accept})`}</span>
    <input type="file" accept={accept} className="hidden"
           onChange={(e) => e.target.files?.[0] && onPick(e.target.files[0])} />
  </label>
);

// ---------- helper: card de ferramenta ----------
const ToolCard: React.FC<{
  icon: React.ReactNode; titulo: string; subtitulo: string; children: React.ReactNode;
}> = ({ icon, titulo, subtitulo, children }) => (
  <div className="bg-white rounded-xl border border-slate-200 p-5 flex flex-col gap-3">
    <div className="flex items-center gap-3">
      <div className="w-9 h-9 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
        {icon}
      </div>
      <div>
        <h3 className="text-sm font-semibold text-slate-800">{titulo}</h3>
        <p className="text-xs text-slate-500">{subtitulo}</p>
      </div>
    </div>
    {children}
  </div>
);

// ---------- helper: resultado (link de download) ----------
const ResultBox: React.FC<{ res: ToolResult | null; erro: string | null }> = ({ res, erro }) => {
  if (erro) return (
    <div className="flex items-center gap-2 text-xs text-rose-600 bg-rose-50 rounded-lg px-3 py-2">
      <AlertCircle className="w-4 h-4 shrink-0" /> {erro}
    </div>
  );
  if (!res) return null;
  return (
    <div className="flex items-center justify-between gap-2 text-xs bg-emerald-50 rounded-lg px-3 py-2">
      <span className="flex items-center gap-2 text-emerald-700">
        <CheckCircle2 className="w-4 h-4 shrink-0" />
        {res.n_pontos ? `${res.n_pontos.toLocaleString()} pontos` : 'Pronto'}
      </span>
      <a href={downloadUrl(res.download_url)} download
         className="flex items-center gap-1 px-2 py-1 bg-emerald-600 text-white rounded-md hover:bg-emerald-700">
        <Download className="w-3.5 h-3.5" /> Baixar PLY
      </a>
    </div>
  );
};

const botao = "w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm " +
              "font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 " +
              "disabled:cursor-not-allowed transition-colors";

// =====================================================================
//  CONVERSORES
// =====================================================================
const Conversores: React.FC = () => {
  const [busy, setBusy] = useState<string | null>(null);

  // OBJ->PLY
  const [objFile, setObjFile] = useState<File | null>(null);
  const [objDens, setObjDens] = useState(130);
  const [objRes, setObjRes] = useState<ToolResult | null>(null);
  const [objErr, setObjErr] = useState<string | null>(null);

  // USDZ->PLY
  const [usdzFile, setUsdzFile] = useState<File | null>(null);
  const [usdzEstru, setUsdzEstru] = useState(true);
  const [usdzRes, setUsdzRes] = useState<ToolResult | null>(null);
  const [usdzErr, setUsdzErr] = useState<string | null>(null);

  // ASC->PLY
  const [ascFile, setAscFile] = useState<File | null>(null);
  const [ascSub, setAscSub] = useState(20);
  const [ascRes, setAscRes] = useState<ToolResult | null>(null);
  const [ascErr, setAscErr] = useState<string | null>(null);

  const run = async (key: string, fn: () => Promise<ToolResult>,
                     setRes: (r: ToolResult) => void, setErr: (e: string | null) => void) => {
    setBusy(key); setErr(null); setRes(null as any);
    try { setRes(await fn()); }
    catch (e: any) { setErr(e.message || 'Falha na conversão'); }
    finally { setBusy(null); }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      <ToolCard icon={<Box className="w-5 h-5" />} titulo="OBJ → PLY"
                subtitulo="Malha (Kiri Engine, Blender) → nuvem de pontos">
        <FilePicker accept=".obj" file={objFile} onPick={(f) => { setObjFile(f); setObjRes(null); setObjErr(null); }} />
        <label className="text-xs text-slate-500 flex items-center justify-between">
          Densidade (pts/m²)
          <input type="number" min={10} max={2000} value={objDens}
                 onChange={(e) => setObjDens(+e.target.value)}
                 className="w-24 border border-slate-200 rounded px-2 py-1 text-right" />
        </label>
        <button className={botao} disabled={!objFile || busy === 'obj'}
                onClick={() => run('obj', () => objToPly(objFile!, objDens), setObjRes, setObjErr)}>
          {busy === 'obj' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Cloudy className="w-4 h-4" />}
          Converter
        </button>
        <ResultBox res={objRes} erro={objErr} />
      </ToolCard>

      <ToolCard icon={<Smartphone className="w-5 h-5" />} titulo="USDZ → PLY"
                subtitulo="Scan do RoomPlan (iPhone) → nuvem de pontos">
        <FilePicker accept=".usdz" file={usdzFile} onPick={(f) => { setUsdzFile(f); setUsdzRes(null); setUsdzErr(null); }} />
        <label className="text-xs text-slate-500 flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={usdzEstru} onChange={(e) => setUsdzEstru(e.target.checked)} />
          Apenas estrutura (ignora mobília)
        </label>
        <button className={botao} disabled={!usdzFile || busy === 'usdz'}
                onClick={() => run('usdz', () => usdzToPly(usdzFile!, 130, usdzEstru), setUsdzRes, setUsdzErr)}>
          {busy === 'usdz' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Cloudy className="w-4 h-4" />}
          Converter
        </button>
        <ResultBox res={usdzRes} erro={usdzErr} />
      </ToolCard>

      <ToolCard icon={<ScanLine className="w-5 h-5" />} titulo="ASC → PLY"
                subtitulo="Nuvem de scanner (ASC/XYZ/ZIP) com subamostragem">
        <FilePicker accept=".asc,.xyz,.txt,.zip" file={ascFile} onPick={(f) => { setAscFile(f); setAscRes(null); setAscErr(null); }} />
        <label className="text-xs text-slate-500 flex items-center justify-between">
          Subsample (1 a cada N)
          <input type="number" min={1} max={200} value={ascSub}
                 onChange={(e) => setAscSub(+e.target.value)}
                 className="w-24 border border-slate-200 rounded px-2 py-1 text-right" />
        </label>
        <button className={botao} disabled={!ascFile || busy === 'asc'}
                onClick={() => run('asc', () => ascToPly(ascFile!, ascSub), setAscRes, setAscErr)}>
          {busy === 'asc' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Cloudy className="w-4 h-4" />}
          Converter
        </button>
        <ResultBox res={ascRes} erro={ascErr} />
      </ToolCard>
    </div>
  );
};

// =====================================================================
//  GERADORES
// =====================================================================
type ModoGer = 'estagio' | 'aleatorio' | 'manual';

const Geradores: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [modo, setModo] = useState<ModoGer>('estagio');
  const [estagio, setEstagio] = useState('3');
  const [pctAusente, setPctAusente] = useState(30);
  const [pctParcial, setPctParcial] = useState(15);
  const [seed, setSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<GerarNuvemResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const gerar = async () => {
    if (!file) return;
    setBusy(true); setErr(null); setRes(null);
    try {
      let r: GerarNuvemResult;
      if (modo === 'manual') r = await gerarNuvemManual(file, pctAusente / 100, pctParcial / 100, seed);
      else r = await gerarNuvemEstagio(file, modo === 'aleatorio' ? 'aleatorio' : estagio, seed);
      setRes(r);
    } catch (e: any) { setErr(e.message || 'Falha ao gerar'); }
    finally { setBusy(false); }
  };

  const TabModo: React.FC<{ id: ModoGer; icon: React.ReactNode; label: string }> = ({ id, icon, label }) => (
    <button onClick={() => setModo(id)}
      className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm font-medium border transition-colors
        ${modo === id ? 'bg-blue-50 border-blue-300 text-blue-700' : 'border-slate-200 text-slate-600 hover:bg-slate-50'}`}>
      {icon} {label}
    </button>
  );

  return (
    <div className="max-w-2xl">
      <ToolCard icon={<Sparkles className="w-5 h-5" />} titulo="Gerar nuvem sintética a partir de IFC"
                subtitulo="Simula o scan de um estágio de obra (densidade 130 pts/m², padrão CloudCompare)">
        <FilePicker accept=".ifc" file={file} onPick={(f) => { setFile(f); setRes(null); setErr(null); }} />

        <div className="flex gap-2 mt-1">
          <TabModo id="estagio" icon={<SlidersHorizontal className="w-4 h-4" />} label="Por estágio" />
          <TabModo id="aleatorio" icon={<Dice5 className="w-4 h-4" />} label="Aleatório" />
          <TabModo id="manual" icon={<SlidersHorizontal className="w-4 h-4" />} label="Manual" />
        </div>

        {modo === 'estagio' && (
          <div className="grid grid-cols-2 gap-2 mt-1">
            {ESTAGIOS.map((e) => (
              <button key={e.v} onClick={() => setEstagio(e.v)}
                className={`text-left px-3 py-2 rounded-lg border text-xs transition-colors
                  ${estagio === e.v ? 'bg-blue-50 border-blue-300' : 'border-slate-200 hover:bg-slate-50'}`}>
                <div className="font-semibold text-slate-700">{e.nome}</div>
                <div className="text-slate-400">{e.desc}</div>
              </button>
            ))}
          </div>
        )}

        {modo === 'aleatorio' && (
          <p className="text-xs text-slate-500 bg-slate-50 rounded-lg px-3 py-2">
            Sorteia um dos 10 estágios de obra usando a seed abaixo.
          </p>
        )}

        {modo === 'manual' && (
          <div className="flex flex-col gap-2 mt-1">
            <label className="text-xs text-slate-600 flex items-center justify-between">
              % Ausente (removido) <span className="font-semibold text-slate-800">{pctAusente}%</span>
            </label>
            <input type="range" min={0} max={95} value={pctAusente} onChange={(e) => setPctAusente(+e.target.value)} />
            <label className="text-xs text-slate-600 flex items-center justify-between">
              % Parcial (dos restantes) <span className="font-semibold text-slate-800">{pctParcial}%</span>
            </label>
            <input type="range" min={0} max={50} value={pctParcial} onChange={(e) => setPctParcial(+e.target.value)} />
          </div>
        )}

        <label className="text-xs text-slate-500 flex items-center justify-between mt-1">
          Seed (reprodutível)
          <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)}
                 className="w-24 border border-slate-200 rounded px-2 py-1 text-right" />
        </label>

        <button className={botao} disabled={!file || busy} onClick={gerar}>
          {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
          Gerar nuvem
        </button>

        {res && (
          <div className="flex flex-col gap-2 bg-emerald-50 rounded-lg px-3 py-3 text-xs">
            <div className="flex items-center gap-2 text-emerald-700 font-medium">
              <CheckCircle2 className="w-4 h-4" /> {res.rotulo} — {res.n_pontos?.toLocaleString()} pontos
            </div>
            <div className="flex gap-3 text-slate-600">
              <span className="text-emerald-600">● {res.stats.completo} completos</span>
              <span className="text-amber-500">● {res.stats.parcial} parciais</span>
              <span className="text-rose-500">● {res.stats.ausente} ausentes</span>
              <span className="text-slate-400">de {res.stats.total_objetos}</span>
            </div>
            <div className="flex gap-2">
              <a href={downloadUrl(res.download_url)} download
                 className="flex items-center gap-1 px-2 py-1 bg-emerald-600 text-white rounded-md hover:bg-emerald-700">
                <Download className="w-3.5 h-3.5" /> PLY
              </a>
              <a href={downloadUrl(res.labels_url)} download
                 className="flex items-center gap-1 px-2 py-1 bg-slate-600 text-white rounded-md hover:bg-slate-700">
                <Download className="w-3.5 h-3.5" /> Gabarito (labels.json)
              </a>
            </div>
          </div>
        )}
        {err && (
          <div className="flex items-center gap-2 text-xs text-rose-600 bg-rose-50 rounded-lg px-3 py-2">
            <AlertCircle className="w-4 h-4 shrink-0" /> {err}
          </div>
        )}
      </ToolCard>
    </div>
  );
};

// =====================================================================
//  PÁGINA
// =====================================================================
export const ToolsPage: React.FC = () => {
  const [secao, setSecao] = useState<Secao>('conversores');

  const SubTab: React.FC<{ id: Secao; label: string }> = ({ id, label }) => (
    <button onClick={() => setSecao(id)}
      className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors
        ${secao === id ? 'bg-white shadow-sm text-blue-700' : 'text-slate-500 hover:text-slate-700'}`}>
      {label}
    </button>
  );

  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-xl font-bold text-slate-800">Ferramentas</h1>
        <p className="text-sm text-slate-500">Conversores de formato e geradores de nuvem de pontos.</p>
      </div>

      <div className="inline-flex gap-1 bg-slate-100 rounded-xl p-1 mb-6">
        <SubTab id="conversores" label="Conversores" />
        <SubTab id="geradores" label="Geradores" />
      </div>

      {secao === 'conversores' ? <Conversores /> : <Geradores />}
    </div>
  );
};
