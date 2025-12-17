'use client';

import { Prediction } from '@/lib/api';
import { cn, formatCurrency, formatPercent, getSignalBgColor, getSignalColor, getTierColor, getTierLabel } from '@/lib/utils';
import { TrendingUp, TrendingDown, Activity, Users, Newspaper } from 'lucide-react';

interface StockCardProps {
  prediction: Prediction;
  index?: number;
}

export function StockCard({ prediction, index = 0 }: StockCardProps) {
  const isUp = prediction.predicted_direction === 'UP';
  const signalLabel = prediction.signal.replace('_', ' ');
  
  return (
    <div 
      className={cn(
        'relative rounded-2xl border p-6 transition-all duration-300 hover:scale-[1.02] hover:shadow-xl',
        getSignalBgColor(prediction.signal),
        'animate-slide-up opacity-0',
        `stagger-${index + 1}`
      )}
      style={{ animationFillMode: 'forwards' }}
    >
      {/* Tier badge */}
      <div className="absolute top-4 right-4">
        <span className={cn(
          'text-xs font-medium px-2 py-1 rounded-full',
          getTierColor(prediction.tier)
        )}>
          Tier {prediction.tier}
        </span>
      </div>

      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-1">
          <h2 className="text-3xl font-bold tracking-tight">{prediction.ticker}</h2>
          <div className={cn(
            'p-1.5 rounded-lg',
            isUp ? 'bg-profit-500/20' : 'bg-loss-500/20'
          )}>
            {isUp ? (
              <TrendingUp className="w-5 h-5 text-profit-400" />
            ) : (
              <TrendingDown className="w-5 h-5 text-loss-400" />
            )}
          </div>
        </div>
        <p className="text-2xl font-mono text-midnight-200">
          {formatCurrency(prediction.current_price)}
        </p>
      </div>

      {/* Signal */}
      <div className={cn(
        'inline-block px-4 py-2 rounded-xl font-semibold text-lg mb-6',
        prediction.signal.includes('BUY') ? 'bg-profit-500/20 text-profit-300' :
        prediction.signal.includes('SELL') ? 'bg-loss-500/20 text-loss-300' :
        'bg-amber-500/20 text-amber-300'
      )}>
        {signalLabel}
      </div>

      {/* Confidence meter */}
      <div className="mb-6">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-midnight-400">Confidence</span>
          <span className={cn('font-mono font-medium', getSignalColor(prediction.signal))}>
            {formatPercent(prediction.confidence)}
          </span>
        </div>
        <div className="h-2 bg-midnight-800 rounded-full overflow-hidden">
          <div 
            className={cn(
              'h-full rounded-full transition-all duration-1000 ease-out',
              prediction.signal.includes('BUY') ? 'bg-gradient-to-r from-profit-600 to-profit-400' :
              prediction.signal.includes('SELL') ? 'bg-gradient-to-r from-loss-600 to-loss-400' :
              'bg-gradient-to-r from-amber-600 to-amber-400'
            )}
            style={{ width: `${prediction.confidence * 100}%` }}
          />
        </div>
      </div>

      {/* Data sources */}
      <div className="grid grid-cols-3 gap-3 pt-4 border-t border-midnight-700/50">
        <div className="text-center">
          <Activity className="w-4 h-4 mx-auto mb-1 text-midnight-500" />
          <p className="text-xs text-midnight-400">Stock Days</p>
          <p className="font-mono text-sm">{prediction.data_sources.stock_days}</p>
        </div>
        <div className="text-center">
          <Users className="w-4 h-4 mx-auto mb-1 text-midnight-500" />
          <p className="text-xs text-midnight-400">Pol. Trades</p>
          <p className="font-mono text-sm">{prediction.data_sources.politician_trades}</p>
        </div>
        <div className="text-center">
          <Newspaper className="w-4 h-4 mx-auto mb-1 text-midnight-500" />
          <p className="text-xs text-midnight-400">News</p>
          <p className="font-mono text-sm">{prediction.data_sources.news_articles}</p>
        </div>
      </div>

      {/* Prediction date */}
      <div className="mt-4 pt-4 border-t border-midnight-700/50 text-center">
        <p className="text-xs text-midnight-500">
          Prediction for <span className="text-midnight-300 font-medium">{prediction.prediction_for}</span>
        </p>
      </div>
    </div>
  );
}

