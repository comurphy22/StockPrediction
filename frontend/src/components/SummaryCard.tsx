'use client';

import { cn } from '@/lib/utils';
import { LucideIcon } from 'lucide-react';

interface SummaryCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  variant: 'profit' | 'loss' | 'neutral' | 'info';
  delay?: number;
}

export function SummaryCard({ title, value, subtitle, icon: Icon, variant, delay = 0 }: SummaryCardProps) {
  const variants = {
    profit: 'bg-profit-500/10 border-profit-500/30 text-profit-400',
    loss: 'bg-loss-500/10 border-loss-500/30 text-loss-400',
    neutral: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
    info: 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400',
  };

  const iconVariants = {
    profit: 'text-profit-400',
    loss: 'text-loss-400',
    neutral: 'text-amber-400',
    info: 'text-cyan-400',
  };

  return (
    <div 
      className={cn(
        'rounded-xl border p-6 animate-slide-up opacity-0',
        variants[variant]
      )}
      style={{ 
        animationFillMode: 'forwards',
        animationDelay: `${delay}ms`
      }}
    >
      <div className="flex items-start justify-between mb-4">
        <h3 className="text-sm font-medium uppercase tracking-wide text-midnight-400">
          {title}
        </h3>
        <Icon className={cn('w-5 h-5', iconVariants[variant])} />
      </div>
      <p className="text-4xl font-bold tracking-tight mb-1">{value}</p>
      {subtitle && (
        <p className="text-sm text-midnight-400">{subtitle}</p>
      )}
    </div>
  );
}

