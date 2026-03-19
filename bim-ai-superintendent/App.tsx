import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { DataView } from './components/DataView';
import { ReportView } from './components/ReportView';
import { VoiceAssistant } from './components/VoiceAssistant';
import { AppTab, Floor, AnalysisResult } from './types';
import { listFloors, analyzeFloor } from './services/api';
import { LayoutDashboard, FileText, Mic } from 'lucide-react';

export default function App() {
  // Application State
  const [activeTab, setActiveTab] = useState<AppTab>(AppTab.VISTORIA);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Data State
  const [floors, setFloors] = useState<Floor[]>([]);
  const [selectedFloorId, setSelectedFloorId] = useState<string>('');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  
  // File State
  const [ifcFile, setIfcFile] = useState<File | null>(null);
  const [plyFile, setPlyFile] = useState<File | null>(null);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [csvContent, setCsvContent] = useState<string | null>(null);

  // Handlers
  const handleIfcUpload = async (file: File) => {
    setIfcFile(file);
    // Automatically trigger floor listing when IFC is uploaded
    try {
      const floorList = await listFloors(file);
      setFloors(floorList);
    } catch (e) {
      console.error(e);
      alert("Erro ao listar andares do IFC.");
    }
  };

  const handlePlyUpload = (file: File) => setPlyFile(file);
  
  const handleCsvUpload = (file: File) => {
    setCsvFile(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      setCsvContent(text);
    };
    reader.readAsText(file);
  };

  const handleProcess = async () => {
    if (!ifcFile || !plyFile || !selectedFloorId) return;

    setIsProcessing(true);
    try {
      const result = await analyzeFloor(ifcFile, plyFile, selectedFloorId);
      setAnalysisResult(result);
      setActiveTab(AppTab.VISTORIA); // Switch to results view
    } catch (e) {
      console.error(e);
      alert("Falha ao processar a análise do pavimento.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 font-sans">
      <Sidebar
        floors={floors}
        selectedFloor={selectedFloorId}
        onFloorChange={setSelectedFloorId}
        onIfcUpload={handleIfcUpload}
        onPlyUpload={handlePlyUpload}
        onCsvUpload={handleCsvUpload}
        onProcess={handleProcess}
        isProcessing={isProcessing}
        hasIfc={!!ifcFile}
        hasPly={!!plyFile}
      />

      <main className="flex-1 ml-80 flex flex-col h-screen overflow-hidden">
        {/* Top Navigation for Tabs */}
        <header className="bg-white border-b border-slate-200 px-8 h-16 flex items-center justify-between shrink-0 z-10">
          <div className="flex space-x-1">
            <button
              onClick={() => setActiveTab(AppTab.VISTORIA)}
              className={`px-4 py-2 rounded-lg text-sm font-medium flex items-center transition-colors
                ${activeTab === AppTab.VISTORIA 
                  ? 'bg-blue-50 text-blue-700' 
                  : 'text-slate-600 hover:bg-slate-50'}`}
            >
              <LayoutDashboard className="w-4 h-4 mr-2" />
              Vistoria & Dados
            </button>
            <button
              onClick={() => setActiveTab(AppTab.RELATORIO)}
              className={`px-4 py-2 rounded-lg text-sm font-medium flex items-center transition-colors
                ${activeTab === AppTab.RELATORIO
                  ? 'bg-blue-50 text-blue-700' 
                  : 'text-slate-600 hover:bg-slate-50'}`}
            >
              <FileText className="w-4 h-4 mr-2" />
              Relatório Executivo
            </button>
            <button
              onClick={() => setActiveTab(AppTab.VOICE)}
              className={`px-4 py-2 rounded-lg text-sm font-medium flex items-center transition-colors
                ${activeTab === AppTab.VOICE 
                  ? 'bg-rose-50 text-rose-600' 
                  : 'text-slate-600 hover:bg-slate-50'}`}
            >
              <Mic className="w-4 h-4 mr-2" />
              Fale com a Obra
            </button>
          </div>
          <div className="text-xs text-slate-400">
             {analysisResult ? `Dados de: ${new Date(analysisResult.timestamp).toLocaleDateString()}` : 'Aguardando processamento'}
          </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 overflow-auto bg-slate-50">
          {activeTab === AppTab.VISTORIA && <DataView result={analysisResult} />}
          {activeTab === AppTab.RELATORIO && <ReportView result={analysisResult} csvContent={csvContent} />}
          {activeTab === AppTab.VOICE && <VoiceAssistant result={analysisResult} csvContent={csvContent} />}
        </div>
      </main>
    </div>
  );
}