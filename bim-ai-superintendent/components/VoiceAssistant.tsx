import React, { useState, useRef, useEffect } from 'react';
import { AnalysisResult, ChatMessage } from '../types';
import { processVoiceQuery } from '../services/gemini';
import { processChatQuery } from '../services/deepseek';
import { Mic, Square, Send, User, Bot, Loader2 } from 'lucide-react';

interface VoiceAssistantProps {
  result: AnalysisResult | null;
  csvContent: string | null;
}

export const VoiceAssistant: React.FC<VoiceAssistantProps> = ({ result, csvContent }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [textInput, setTextInput] = useState('');

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
        await handleAudioProcess(audioBlob);

        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Permissão de microfone necessária para usar esta função.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleAudioProcess = async (audioBlob: Blob) => {
    if (!result) {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'model',
        text: 'Por favor, processe um pavimento antes de fazer perguntas.',
        timestamp: new Date()
      }]);
      return;
    }

    // Optimistically add user message placeholder (Audio Sent)
    const userMsgId = Date.now().toString();
    setMessages(prev => [...prev, {
      id: userMsgId,
      role: 'user',
      text: '🎤 [Áudio enviado]',
      timestamp: new Date()
    }]);

    setIsProcessing(true);

    // Convert Blob to Base64
    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    reader.onloadend = async () => {
      const base64String = reader.result as string;
      // Remove data URL prefix (e.g., "data:audio/webm;base64,")
      const base64Content = base64String.split(',')[1];

      const responseText = await processVoiceQuery(base64Content, result, csvContent);

      setIsProcessing(false);
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'model',
        text: responseText,
        timestamp: new Date()
      }]);
    };
  };
  const handleTextSubmit = async () => {
    if (!textInput.trim() || !result) return;
    // Adiciona mensagem do usuário
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      text: textInput,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMsg]);
    setTextInput('');
    setIsProcessing(true);
    // Chama DeepSeek
    const responseText = await processChatQuery(textInput, result, csvContent);
    setIsProcessing(false);
    setMessages(prev => [...prev, {
      id: (Date.now() + 1).toString(),
      role: 'model',
      text: responseText,
      timestamp: new Date()
    }]);
  };


  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-400">
        <Mic className="w-16 h-16 mb-4 text-slate-200" />
        <h3 className="text-xl font-medium text-slate-600">Fale com a Obra</h3>
        <p className="mt-2">Carregue e processe os arquivos para ativar o assistente de voz.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-slate-50 relative">
      <div className="p-6 border-b border-slate-200 bg-white shadow-sm z-10">
        <h2 className="text-xl font-bold text-slate-800 flex items-center">
          <Mic className="w-5 h-5 mr-2 text-rose-500" />
          Assistente Multimodal
        </h2>
        <p className="text-sm text-slate-500">Faça perguntas sobre o progresso, atrasos e riscos usando sua voz.</p>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full opacity-50">
            <div className="w-20 h-20 bg-slate-200 rounded-full flex items-center justify-center mb-4">
              <Bot className="w-10 h-10 text-slate-400" />
            </div>
            <p className="text-slate-500 text-center max-w-md">
              Tente perguntar: <br />
              "Qual o progresso das instalações elétricas?" <br />
              "Existe algum risco crítico reportado?"
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`flex max-w-[80%] ${msg.role === 'user'
                ? 'bg-blue-600 text-white rounded-2xl rounded-tr-sm'
                : 'bg-white border border-slate-200 text-slate-800 rounded-2xl rounded-tl-sm shadow-sm'
                } p-4`}
            >
              <div className="mr-3 mt-1 shrink-0">
                {msg.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4 text-blue-500" />}
              </div>
              <div>
                <p className="text-sm leading-relaxed">{msg.text}</p>
                <span className={`text-[10px] mt-2 block ${msg.role === 'user' ? 'text-blue-200' : 'text-slate-400'}`}>
                  {msg.timestamp.toLocaleTimeString()}
                </span>
              </div>
            </div>
          </div>
        ))}
        {isProcessing && (
          <div className="flex justify-start">
            <div className="bg-white border border-slate-200 p-4 rounded-2xl rounded-tl-sm shadow-sm flex items-center space-x-2">
              <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
              <span className="text-sm text-slate-500">Ouvindo e analisando...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Controls - Voice + Text Input */}
      <div className="p-4 bg-white border-t border-slate-200 space-y-3">
        {/* Text Input Row */}
        <div className="flex space-x-2">
          <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !isProcessing && handleTextSubmit()}
            placeholder="Digite sua pergunta sobre a obra..."
            disabled={isProcessing}
            className="flex-1 px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-slate-100 text-sm"
          />
          <button
            onClick={handleTextSubmit}
            disabled={!textInput.trim() || isProcessing}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
          >
            <Send className="w-4 h-4" />
            <span className="text-sm font-medium">Enviar</span>
          </button>
        </div>
        {/* Voice Button Row */}
        <div className="flex items-center justify-center relative pb-6">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isProcessing}
            className={`
              relative w-14 h-14 rounded-full flex items-center justify-center transition-all duration-300 shadow-lg
              ${isProcessing ? 'bg-slate-300 cursor-not-allowed' :
                isRecording
                  ? 'bg-rose-500 scale-110 shadow-rose-500/40 ring-4 ring-rose-100'
                  : 'bg-blue-600 hover:bg-blue-500 shadow-blue-500/30'
              }
            `}
          >
            {isRecording ? (
              <Square className="w-5 h-5 text-white fill-current" />
            ) : (
              <Mic className="w-6 h-6 text-white" />
            )}
            {isRecording && (
              <span className="absolute w-full h-full rounded-full bg-rose-500 opacity-20 animate-ping"></span>
            )}
          </button>
          <div className="absolute bottom-0 text-[10px] text-slate-400">
            {isRecording ? 'Gravando... Toque para parar' : 'Ou use voz (Gemini)'}
          </div>
        </div>
      </div>
    </div>
  );
};