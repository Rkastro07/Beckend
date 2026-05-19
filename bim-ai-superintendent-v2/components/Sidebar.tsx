import React from 'react';
import { Upload, FileText, Layers, Key, PlayCircle, Cpu } from 'lucide-react';
import { Floor } from '../types';

interface SidebarProps {
  floors: Floor[];
  selectedFloor: string;
  onFloorChange: (id: string) => void;
  onIfcUpload: (file: File) => void;
  onPlyUpload: (file: File) => void;
  onCsvUpload: (file: File) => void;
  onProcess: () => void;
  isProcessing: boolean;
  hasIfc: boolean;
  hasPly: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  floors,
  selectedFloor,
  onFloorChange,
  onIfcUpload,
  onPlyUpload,
  onCsvUpload,
  onProcess,
  isProcessing,
  hasIfc,
  hasPly,
}) => {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, callback: (f: File) => void) => {
    if (e.target.files && e.target.files[0]) {
      callback(e.target.files[0]);
    }
  };

  return (
    <aside className="w-80 bg-slate-900 text-slate-100 flex flex-col h-screen fixed left-0 top-0 overflow-y-auto z-20 shadow-xl">
      <div className="p-6 border-b border-slate-700">
        <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-300 bg-clip-text text-transparent">
          BIM Superintendent v2
        </h1>
        <p className="text-xs text-slate-400 mt-1">Sonata + OBB do IFC</p>
      </div>

      <div className="flex-1 p-6 space-y-8">
        {/* Status / pipeline info */}
        <div className="space-y-2">
          <div className="flex items-center text-sm font-semibold text-slate-300 mb-2">
            <Cpu className="w-4 h-4 mr-2" />
            <span>Pipeline</span>
          </div>
          <div className="text-xs bg-slate-800 p-2 rounded border border-slate-700 space-y-1">
            <div className="text-emerald-400">✓ Alinhamento PLY ↔ IFC</div>
            <div className="text-emerald-400">✓ Sonata (semantic seg ScanNet)</div>
            <div className="text-emerald-400">✓ Filtragem por OBB do IFC</div>
            <div className="text-slate-500 text-[10px] mt-1">
              Endpoint: <code>/api/analisar_ai_v2</code>
            </div>
          </div>
        </div>

        {/* API Key Status */}
        <div className="space-y-2">
          <div className="flex items-center text-sm font-semibold text-slate-300 mb-2">
            <Key className="w-4 h-4 mr-2" />
            <span>API Configuration</span>
          </div>
          <div className="text-xs bg-slate-800 p-2 rounded text-emerald-400 border border-slate-700">
            ✓ System API Key Active
          </div>
        </div>

        {/* Inputs */}
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-300 flex items-center">
              <Layers className="w-4 h-4 mr-2" />
              Modelo BIM (.ifc)
            </label>
            <input
              type="file"
              accept=".ifc"
              onChange={(e) => handleFileChange(e, onIfcUpload)}
              className="block w-full text-xs text-slate-400
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-xs file:font-semibold
                file:bg-slate-700 file:text-cyan-400
                hover:file:bg-slate-600 cursor-pointer"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-300 flex items-center">
              <Upload className="w-4 h-4 mr-2" />
              Nuvem de Pontos (.ply)
            </label>
            <input
              type="file"
              accept=".ply"
              onChange={(e) => handleFileChange(e, onPlyUpload)}
              className="block w-full text-xs text-slate-400
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-xs file:font-semibold
                file:bg-slate-700 file:text-cyan-400
                hover:file:bg-slate-600 cursor-pointer"
            />
          </div>
        </div>

        {/* Floor Selection */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-300">Selecionar Pavimento</label>
          <select
            value={selectedFloor}
            onChange={(e) => onFloorChange(e.target.value)}
            className="w-full bg-slate-800 border border-slate-600 rounded p-2 text-sm text-white focus:ring-2 focus:ring-cyan-500 outline-none"
            disabled={floors.length === 0}
          >
            <option value="">Selecione um andar...</option>
            <option value="__TODOS__">Todos os pavimentos</option>
            {floors.map((f) => (
              <option key={f.id} value={f.id}>
                {f.name}
              </option>
            ))}
          </select>
          {floors.length === 0 && (
            <p className="text-[10px] text-amber-500">* Carregue um IFC para listar andares</p>
          )}
        </div>

        {/* CSV Optional */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-300 flex items-center">
            <FileText className="w-4 h-4 mr-2" />
            Cronograma (.csv) <span className="ml-1 text-slate-500 text-[10px]">(Opcional)</span>
          </label>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => handleFileChange(e, onCsvUpload)}
            className="block w-full text-xs text-slate-400
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-xs file:font-semibold
              file:bg-slate-700 file:text-cyan-400
              hover:file:bg-slate-600 cursor-pointer"
          />
        </div>
      </div>

      <div className="p-6 mt-auto border-t border-slate-700">
        <button
          onClick={onProcess}
          disabled={!hasIfc || !hasPly || !selectedFloor || isProcessing}
          className={`w-full py-3 px-4 rounded-lg font-bold flex items-center justify-center space-x-2 transition-all
            ${
              !hasIfc || !hasPly || !selectedFloor || isProcessing
                ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                : 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg hover:shadow-cyan-500/25'
            }`}
        >
          {isProcessing ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
              <span>Processando v2...</span>
            </>
          ) : (
            <>
              <PlayCircle className="w-5 h-5" />
              <span>Processar (v2)</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
};
