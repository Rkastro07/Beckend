import React from 'react';

type Tone = 'neutral' | 'success' | 'warning' | 'danger' | 'info' | 'accent';

interface CardProps {
  /** Cor do realce na borda esquerda (opcional) */
  tone?: Tone;
  /** Sem padding interno (útil se o filho controla seu próprio espaçamento, ex.: tabelas) */
  flush?: boolean;
  /** Altura mínima fixa em px (útil para gráficos) */
  minH?: number;
  className?: string;
  children: React.ReactNode;
}

const TONE_BAR: Record<Tone, string> = {
  neutral: '',
  success: 'border-l-4 border-l-emerald-500',
  warning: 'border-l-4 border-l-amber-500',
  danger:  'border-l-4 border-l-rose-500',
  info:    'border-l-4 border-l-sky-500',
  accent:  'border-l-4 border-l-indigo-500',
};

/**
 * Card padrão do app — substitui o antigo `bg-white p-4 rounded shadow border border-slate-200`
 * espalhado pelo código. Sombra em duas camadas dá profundidade sutil sem ficar pesado.
 */
export const Card: React.FC<CardProps> = ({
  tone = 'neutral',
  flush = false,
  minH,
  className = '',
  children,
}) => {
  return (
    <div
      style={minH ? { minHeight: minH } : undefined}
      className={[
        'bg-white/95 backdrop-blur-sm',
        'rounded-2xl',
        flush ? '' : 'p-5',
        'border border-slate-200/70',
        'shadow-[0_1px_2px_rgba(15,23,42,0.04),0_8px_24px_-12px_rgba(15,23,42,0.12)]',
        'transition-shadow hover:shadow-[0_2px_4px_rgba(15,23,42,0.06),0_12px_32px_-12px_rgba(15,23,42,0.18)]',
        TONE_BAR[tone],
        className,
      ].join(' ')}
    >
      {children}
    </div>
  );
};

interface StatCardProps {
  label: string;
  value: string | number;
  tone?: Tone;
  /** Pequeno texto contextual (ex.: "12 de 85"), opcional */
  hint?: string;
}

/** Card especializado para métricas grandes — uniformiza os "mini stats" do dashboard. */
export const StatCard: React.FC<StatCardProps> = ({ label, value, tone = 'neutral', hint }) => {
  const valueColor: Record<Tone, string> = {
    neutral: 'text-slate-800',
    success: 'text-emerald-700',
    warning: 'text-amber-700',
    danger:  'text-rose-700',
    info:    'text-sky-700',
    accent:  'text-indigo-700',
  };
  return (
    <Card tone={tone}>
      <div className="text-[11px] uppercase tracking-wider font-semibold text-slate-500">
        {label}
      </div>
      <div className={`mt-1 text-3xl font-bold tabular-nums ${valueColor[tone]}`}>
        {value}
      </div>
      {hint && <div className="text-xs text-slate-500 mt-1">{hint}</div>}
    </Card>
  );
};
