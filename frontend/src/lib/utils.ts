import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatPercent(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value);
}

export function formatNumber(value: number, decimals: number = 2): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

export function getSignalColor(signal: string): string {
  switch (signal) {
    case 'STRONG_BUY':
      return 'text-profit-400';
    case 'BUY':
      return 'text-profit-500';
    case 'HOLD':
      return 'text-amber-400';
    case 'SELL':
      return 'text-loss-500';
    case 'STRONG_SELL':
      return 'text-loss-400';
    default:
      return 'text-midnight-300';
  }
}

export function getSignalBgColor(signal: string): string {
  switch (signal) {
    case 'STRONG_BUY':
      return 'bg-profit-500/10 border-profit-500/30';
    case 'BUY':
      return 'bg-profit-500/10 border-profit-500/20';
    case 'HOLD':
      return 'bg-amber-500/10 border-amber-500/20';
    case 'SELL':
      return 'bg-loss-500/10 border-loss-500/20';
    case 'STRONG_SELL':
      return 'bg-loss-500/10 border-loss-500/30';
    default:
      return 'bg-midnight-800/50 border-midnight-600/20';
  }
}

export function getDirectionIcon(direction: string): string {
  return direction === 'UP' ? '↗' : '↘';
}

export function getTierLabel(tier: number): string {
  switch (tier) {
    case 1:
      return 'High Accuracy';
    case 2:
      return 'Moderate';
    case 3:
      return 'Low Accuracy';
    default:
      return 'Unknown';
  }
}

export function getTierColor(tier: number): string {
  switch (tier) {
    case 1:
      return 'text-profit-400 bg-profit-500/10';
    case 2:
      return 'text-amber-400 bg-amber-500/10';
    case 3:
      return 'text-loss-400 bg-loss-500/10';
    default:
      return 'text-midnight-400 bg-midnight-500/10';
  }
}

