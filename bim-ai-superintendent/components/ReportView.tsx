import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { AnalysisResult } from '../types';
import { generateExecutiveReport } from '../services/deepseek';
import { FileText, RefreshCw, Download } from 'lucide-react';

interface ReportViewProps {
  result: AnalysisResult | null;
  csvContent: string | null;
}

export const ReportView: React.FC<ReportViewProps> = ({ result, csvContent }) => {
  const [report, setReport] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    if (!result) return;
    setLoading(true);
    const text = await generateExecutiveReport(result, csvContent);
    setReport(text);
    setLoading(false);
  };

  if (!result) {
    return (
      <div className="flex items-center justify-center h-full text-slate-400">
        <p>Processe os dados para gerar o relatório.</p>
      </div>
    );
  }

  return (
    <div className="p-8 max-w-4xl mx-auto h-full flex flex-col">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h2 className="text-2xl font-bold text-slate-800 flex items-center">
            <FileText className="w-6 h-6 mr-2 text-blue-600" />
            Relatório Executivo IA
          </h2>
          <p className="text-slate-500 text-sm mt-1">
            Análise cruzada de Real vs Planejado gerada por DeepSeek AI
          </p>
        </div>

        <button
          onClick={handleGenerate}
          disabled={loading}
          className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-medium flex items-center transition-colors disabled:opacity-50"
        >
          {loading ? (
            <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <RefreshCw className="w-4 h-4 mr-2" />
          )}
          {report ? 'Regerar Relatório' : 'Gerar Relatório'}
        </button>
      </div>

      <div className="flex-1 bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden flex flex-col">
        {!report && !loading && (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-400 p-12 text-center">
            <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mb-4">
              <FileText className="w-8 h-8 text-slate-300" />
            </div>
            <h3 className="text-lg font-medium text-slate-600">Aguardando Geração</h3>
            <p className="max-w-md mt-2 text-sm">
              Clique em "Gerar Relatório" para que a IA analise as discrepâncias entre o modelo BIM, a nuvem de pontos e o cronograma.
            </p>
          </div>
        )}

        {loading && (
          <div className="flex-1 flex flex-col items-center justify-center p-12">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-indigo-200 border-t-indigo-600 mb-4"></div>
            <p className="text-slate-600 font-medium">Analisando dados da obra...</p>
            <p className="text-slate-400 text-sm mt-1">Identificando riscos e calculando desvios.</p>
          </div>
        )}

        {report && !loading && (
          <div className="flex-1 overflow-auto p-8 prose prose-slate max-w-none
            prose-table:w-full prose-table:text-sm prose-table:border-collapse
            prose-th:bg-slate-100 prose-th:px-4 prose-th:py-2 prose-th:text-left prose-th:border prose-th:border-slate-300 prose-th:font-semibold prose-th:text-slate-700
            prose-td:px-4 prose-td:py-2 prose-td:border prose-td:border-slate-200 prose-td:text-slate-600
            prose-tr:even:bg-slate-50
            prose-h1:text-2xl prose-h1:font-bold prose-h1:text-slate-900 prose-h1:border-b prose-h1:border-slate-200 prose-h1:pb-2
            prose-h2:text-xl prose-h2:font-bold prose-h2:text-slate-800 prose-h2:mt-8
            prose-h3:text-lg prose-h3:font-semibold prose-h3:text-slate-700
            prose-p:text-slate-600 prose-p:leading-relaxed
            prose-li:text-slate-600
            prose-strong:text-slate-900
            prose-blockquote:border-l-blue-500 prose-blockquote:bg-blue-50 prose-blockquote:py-1
            overflow-x-auto">
            <ReactMarkdown>{report}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
};