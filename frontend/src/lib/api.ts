/**
 * API Client for Stock Prediction Backend
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface DataSourceInfo {
  stock_days: number;
  politician_trades: number;
  news_articles: number;
}

export interface Prediction {
  ticker: string;
  signal: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
  confidence: number;
  predicted_direction: 'UP' | 'DOWN';
  current_price: number;
  prediction_date: string;
  prediction_for: string;
  data_sources: DataSourceInfo;
  tier: number;
}

export interface BatchPredictionResponse {
  predictions: Prediction[];
  generated_at: string;
  total_count: number;
  processing_time_seconds: number;
}

export interface PortfolioRecommendation {
  strong_buys: Prediction[];
  buys: Prediction[];
  holds: Prediction[];
  sells: Prediction[];
  strong_sells: Prediction[];
  generated_at: string;
  summary: string;
}

export interface TickerTiers {
  tier1: { stocks: string[]; description: string; recommended: boolean };
  tier2: { stocks: string[]; description: string; recommended: boolean };
  tier3: { stocks: string[]; description: string; recommended: boolean; warning?: string };
}

export interface BacktestResult {
  ticker: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_value: number;
  total_return: number;
  buy_hold_return: number;
  excess_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  accuracy: number;
}

export interface HealthCheck {
  status: string;
  models_cached: number;
  available_tickers: string[];
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_URL) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `API Error: ${response.status}`);
    }

    return response.json();
  }

  // Health check
  async healthCheck(): Promise<HealthCheck> {
    return this.fetch<HealthCheck>('/health');
  }

  // Predictions
  async getPrediction(ticker: string, lookbackDays: number = 180): Promise<Prediction> {
    return this.fetch<Prediction>(`/api/predictions/${ticker}?lookback_days=${lookbackDays}`);
  }

  async getBatchPredictions(tickers: string[], lookbackDays: number = 180): Promise<BatchPredictionResponse> {
    return this.fetch<BatchPredictionResponse>('/api/predictions/batch', {
      method: 'POST',
      body: JSON.stringify({ tickers, lookback_days: lookbackDays }),
    });
  }

  async getPortfolioRecommendations(includeTier3: boolean = false): Promise<PortfolioRecommendation> {
    return this.fetch<PortfolioRecommendation>(
      `/api/predictions/portfolio/recommendations?include_tier3=${includeTier3}`
    );
  }

  async getAvailableTickers(): Promise<TickerTiers> {
    return this.fetch<TickerTiers>('/api/predictions/tickers/available');
  }

  // Backtest
  async runBacktest(
    ticker: string,
    startDate: string,
    endDate: string,
    initialCapital: number = 10000,
    transactionCost: number = 0.001
  ): Promise<BacktestResult> {
    return this.fetch<BacktestResult>('/api/backtest/run', {
      method: 'POST',
      body: JSON.stringify({
        ticker,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        transaction_cost: transactionCost,
      }),
    });
  }

  async getBacktestSummary(): Promise<{ results: BacktestResult[]; total_stocks: number; average_sharpe: number; average_accuracy: number }> {
    return this.fetch('/api/backtest/summary');
  }

  // Data
  async getStockData(ticker: string, days: number = 30): Promise<{ ticker: string; data: any[]; count: number }> {
    return this.fetch(`/api/data/stock/${ticker}?days=${days}`);
  }

  async getPoliticianTrades(ticker: string, days: number = 90): Promise<{ ticker: string; trades: any[]; count: number }> {
    return this.fetch(`/api/data/politician-trades/${ticker}?days=${days}`);
  }

  async getValidationResults(): Promise<any> {
    return this.fetch('/api/data/validation-results');
  }

  async getMarketStatus(): Promise<{ current_time: string; is_market_open: boolean; recommendation: string }> {
    return this.fetch('/api/data/market-status');
  }
}

export const api = new ApiClient();
export default api;

