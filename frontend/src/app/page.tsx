'use client';

import { useState, useEffect, useCallback } from 'react';
import { TrendingUp, TrendingDown, Activity, Zap, AlertTriangle, Timer } from 'lucide-react';
import { Header, StockCard, SummaryCard, LoadingSpinner } from '@/components';
import api, { Prediction, PortfolioRecommendation } from '@/lib/api';
import { cn } from '@/lib/utils';

export default function Dashboard() {
  const [recommendations, setRecommendations] = useState<PortfolioRecommendation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await api.getPortfolioRecommendations(false);
      setRecommendations(data);
      setLastUpdate(new Date());
    } catch (err) {
      console.error('Failed to fetch predictions:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch predictions');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Get all predictions sorted by confidence
  const allPredictions = recommendations 
    ? [
        ...recommendations.strong_buys,
        ...recommendations.buys,
        ...recommendations.holds,
        ...recommendations.sells,
        ...recommendations.strong_sells,
      ].sort((a, b) => b.confidence - a.confidence)
    : [];

  const buyCount = (recommendations?.strong_buys.length || 0) + (recommendations?.buys.length || 0);
  const sellCount = (recommendations?.sells.length || 0) + (recommendations?.strong_sells.length || 0);
  const holdCount = recommendations?.holds.length || 0;

  return (
    <div className="min-h-screen">
      <Header onRefresh={fetchData} isLoading={loading} />

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Hero Section */}
        <section className="mb-12 text-center animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">AI-Powered</span> Stock Predictions
          </h2>
          <p className="text-lg text-midnight-400 max-w-2xl mx-auto">
            Leveraging politician trading signals, news sentiment, and technical analysis 
            to generate actionable trading recommendations.
          </p>
        </section>

        {/* Loading State */}
        {loading && !recommendations && (
          <div className="flex items-center justify-center min-h-[400px]">
            <LoadingSpinner size="lg" text="Analyzing markets..." />
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
            <div className="p-4 rounded-full bg-loss-500/10">
              <AlertTriangle className="w-12 h-12 text-loss-400" />
            </div>
            <h3 className="text-xl font-semibold text-loss-300">Connection Error</h3>
            <p className="text-midnight-400 text-center max-w-md">
              {error}
            </p>
            <p className="text-sm text-midnight-500">
              Make sure the backend server is running on port 8000
            </p>
            <button
              onClick={fetchData}
              className="mt-4 px-6 py-2 bg-midnight-800 border border-midnight-700 rounded-lg hover:bg-midnight-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        )}

        {/* Dashboard Content */}
        {recommendations && !loading && (
          <>
            {/* Summary Cards */}
            <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-12">
              <SummaryCard
                title="Buy Signals"
                value={buyCount}
                subtitle={`${recommendations.strong_buys.length} strong`}
                icon={TrendingUp}
                variant="profit"
                delay={100}
              />
              <SummaryCard
                title="Sell Signals"
                value={sellCount}
                subtitle={`${recommendations.strong_sells.length} strong`}
                icon={TrendingDown}
                variant="loss"
                delay={200}
              />
              <SummaryCard
                title="Hold"
                value={holdCount}
                subtitle="Wait for signal"
                icon={Activity}
                variant="neutral"
                delay={300}
              />
              <SummaryCard
                title="Total Analyzed"
                value={allPredictions.length}
                subtitle="Stocks tracked"
                icon={Zap}
                variant="info"
                delay={400}
              />
            </section>

            {/* Summary Message */}
            {recommendations.summary && (
              <div className="glass rounded-xl p-4 mb-8 flex items-center gap-3 animate-slide-up" style={{ animationDelay: '500ms', animationFillMode: 'forwards', opacity: 0 }}>
                <div className="p-2 rounded-lg bg-cyan-500/10">
                  <Zap className="w-5 h-5 text-cyan-400" />
                </div>
                <p className="text-midnight-200">{recommendations.summary}</p>
              </div>
            )}

            {/* Stock Cards Grid */}
            <section className="mb-12">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold">Predictions</h3>
                {lastUpdate && (
                  <div className="flex items-center gap-2 text-sm text-midnight-500">
                    <Timer className="w-4 h-4" />
                    Updated {lastUpdate.toLocaleTimeString()}
                  </div>
                )}
              </div>

              {allPredictions.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {allPredictions.map((prediction, index) => (
                    <StockCard 
                      key={prediction.ticker} 
                      prediction={prediction}
                      index={index}
                    />
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-midnight-400">
                  <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No predictions available</p>
                </div>
              )}
            </section>

            {/* Quick Actions */}
            <section className="glass rounded-2xl p-6">
              <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <a
                  href="/backtest"
                  className={cn(
                    'flex items-center gap-3 p-4 rounded-xl',
                    'bg-midnight-800/50 border border-midnight-700/50',
                    'hover:bg-midnight-800 hover:border-midnight-600 transition-all'
                  )}
                >
                  <div className="p-2 rounded-lg bg-purple-500/10">
                    <Activity className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <p className="font-medium">Backtest</p>
                    <p className="text-xs text-midnight-400">Test strategies</p>
                  </div>
                </a>
                <a
                  href="/api/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className={cn(
                    'flex items-center gap-3 p-4 rounded-xl',
                    'bg-midnight-800/50 border border-midnight-700/50',
                    'hover:bg-midnight-800 hover:border-midnight-600 transition-all'
                  )}
                >
                  <div className="p-2 rounded-lg bg-blue-500/10">
                    <Zap className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <p className="font-medium">API Docs</p>
                    <p className="text-xs text-midnight-400">FastAPI Swagger</p>
                  </div>
                </a>
                <button
                  onClick={fetchData}
                  disabled={loading}
                  className={cn(
                    'flex items-center gap-3 p-4 rounded-xl text-left',
                    'bg-midnight-800/50 border border-midnight-700/50',
                    'hover:bg-midnight-800 hover:border-midnight-600 transition-all',
                    'disabled:opacity-50'
                  )}
                >
                  <div className="p-2 rounded-lg bg-profit-500/10">
                    <TrendingUp className="w-5 h-5 text-profit-400" />
                  </div>
                  <div>
                    <p className="font-medium">Refresh Data</p>
                    <p className="text-xs text-midnight-400">Get latest signals</p>
                  </div>
                </button>
              </div>
            </section>
          </>
        )}

        {/* Footer */}
        <footer className="mt-16 py-8 border-t border-midnight-800 text-center text-sm text-midnight-500">
          <p>
            <span className="text-midnight-400">Stock Prediction Dashboard</span> — 
            ML-powered predictions using politician trading signals
          </p>
          <p className="mt-2">
            Built with FastAPI, XGBoost, Next.js & Tailwind CSS
          </p>
          <p className="mt-4 text-xs text-midnight-600">
            ⚠️ For educational purposes only. Not financial advice.
          </p>
        </footer>
      </main>
    </div>
  );
}

