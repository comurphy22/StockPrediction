'use client';

import { useState, useEffect } from 'react';
import { Activity, Clock, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import api from '@/lib/api';

interface HeaderProps {
  onRefresh?: () => void;
  isLoading?: boolean;
}

export function Header({ onRefresh, isLoading }: HeaderProps) {
  const [marketStatus, setMarketStatus] = useState<{
    is_market_open: boolean;
    recommendation: string;
  } | null>(null);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    // Update time every second
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    
    // Fetch market status
    api.getMarketStatus()
      .then(setMarketStatus)
      .catch(console.error);

    return () => clearInterval(timer);
  }, []);

  return (
    <header className="glass-dark sticky top-0 z-50 border-b border-midnight-700/50">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo & Title */}
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-profit-500 to-cyan-500 flex items-center justify-center">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div className="absolute -inset-1 rounded-xl bg-gradient-to-br from-profit-500/50 to-cyan-500/50 blur-lg -z-10" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                <span className="gradient-text">Stock Prediction</span>
              </h1>
              <p className="text-xs text-midnight-400">ML-Powered Trading Signals</p>
            </div>
          </div>

          {/* Status & Controls */}
          <div className="flex items-center gap-6">
            {/* Market Status */}
            {marketStatus && (
              <div className="hidden md:flex items-center gap-2 text-sm">
                <div className={cn(
                  'w-2 h-2 rounded-full',
                  marketStatus.is_market_open 
                    ? 'bg-profit-500 animate-pulse' 
                    : 'bg-midnight-500'
                )} />
                <span className="text-midnight-400">
                  Market {marketStatus.is_market_open ? 'Open' : 'Closed'}
                </span>
              </div>
            )}

            {/* Current Time */}
            <div className="hidden sm:flex items-center gap-2 text-sm text-midnight-400">
              <Clock className="w-4 h-4" />
              <span className="font-mono">
                {currentTime.toLocaleTimeString('en-US', { 
                  hour: '2-digit', 
                  minute: '2-digit',
                  second: '2-digit'
                })}
              </span>
            </div>

            {/* Refresh Button */}
            {onRefresh && (
              <button
                onClick={onRefresh}
                disabled={isLoading}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg',
                  'bg-midnight-800 border border-midnight-700 hover:border-midnight-600',
                  'text-sm font-medium transition-all duration-200',
                  'hover:bg-midnight-700 active:scale-95',
                  'disabled:opacity-50 disabled:cursor-not-allowed'
                )}
              >
                <RefreshCw className={cn('w-4 h-4', isLoading && 'animate-spin')} />
                <span className="hidden sm:inline">Refresh</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

