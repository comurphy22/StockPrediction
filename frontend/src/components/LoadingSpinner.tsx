'use client';

import { cn } from '@/lib/utils';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
}

export function LoadingSpinner({ size = 'md', text }: LoadingSpinnerProps) {
  const sizes = {
    sm: 'w-6 h-6',
    md: 'w-10 h-10',
    lg: 'w-16 h-16',
  };

  return (
    <div className="flex flex-col items-center justify-center gap-4">
      <div className={cn('relative', sizes[size])}>
        {/* Outer ring */}
        <div className="absolute inset-0 rounded-full border-2 border-midnight-700" />
        {/* Spinning gradient */}
        <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-profit-500 border-r-cyan-500 animate-spin" />
        {/* Inner glow */}
        <div className="absolute inset-2 rounded-full bg-gradient-to-br from-profit-500/20 to-cyan-500/20 blur-sm" />
      </div>
      {text && (
        <p className="text-midnight-400 animate-pulse">{text}</p>
      )}
    </div>
  );
}

