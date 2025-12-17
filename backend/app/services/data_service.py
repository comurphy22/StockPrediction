"""
Data Service
============
Provides access to stock data, politician trades, and results.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

# Add the src directory to path (for Railway deployment, src is inside backend/)
BACKEND_DIR = Path(__file__).parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
# Try local src first (Railway), then parent src (local dev)
if (BACKEND_DIR / 'src').exists():
    sys.path.insert(0, str(BACKEND_DIR / 'src'))
else:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class DataService:
    """Service for accessing stock and trading data."""
    
    def __init__(self):
        self.results_dir = PROJECT_ROOT / 'results'
        
        # Import data loading functions from existing code
        try:
            from data_loader import fetch_stock_data, fetch_politician_trades
            self.fetch_stock_data_fn = fetch_stock_data
            self.fetch_politician_trades_fn = fetch_politician_trades
        except ImportError as e:
            print(f"Warning: Could not import data_loader: {e}")
            self.fetch_stock_data_fn = None
            self.fetch_politician_trades_fn = None
    
    def get_stock_data(self, ticker: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent stock price data."""
        if not self.fetch_stock_data_fn:
            raise RuntimeError("Stock data loader not available")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = self.fetch_stock_data_fn(ticker, start_date, end_date)
        
        result = []
        for _, row in df.iterrows():
            result.append({
                'date': row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume'])
            })
        
        return result
    
    def get_politician_trades(self, ticker: str, days: int = 90) -> List[Dict[str, Any]]:
        """Get recent politician trading activity."""
        if not self.fetch_politician_trades_fn:
            raise RuntimeError("Politician trades loader not available")
        
        df = self.fetch_politician_trades_fn(ticker)
        
        if df.empty:
            return []
        
        # Filter to recent trades
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df['date'] >= cutoff]
        
        result = []
        for _, row in df.iterrows():
            result.append({
                'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row.get('date', '')),
                'politician': row.get('politician_name', 'Unknown'),
                'transaction_type': row.get('transaction_type', 'Unknown'),
                'amount': str(row.get('amount', 0)),
                'ticker': ticker
            })
        
        return result
    
    def get_validation_results(self) -> Dict[str, Any]:
        """Get historical validation results from CSV files."""
        results = {}
        
        # Multi-year validation results
        multiyear_path = self.results_dir / 'multiyear_validation_results.csv'
        if multiyear_path.exists():
            df = pd.read_csv(multiyear_path)
            results['multiyear_validation'] = df.to_dict('records')
        
        # Economic backtest results
        backtest_path = self.results_dir / 'economic_backtest_results.csv'
        if backtest_path.exists():
            df = pd.read_csv(backtest_path)
            results['economic_backtest'] = df.to_dict('records')
        
        # Feature importance
        feature_path = self.results_dir / 'feature_importance_rankings.csv'
        if feature_path.exists():
            df = pd.read_csv(feature_path)
            results['feature_importance'] = df.to_dict('records')
        
        # Model comparison
        comparison_path = self.results_dir / 'model_comparison_results.csv'
        if comparison_path.exists():
            df = pd.read_csv(comparison_path)
            results['model_comparison'] = df.to_dict('records')
        
        return results
    
    def get_daily_predictions_log(self) -> List[Dict[str, Any]]:
        """Get logged daily predictions."""
        log_path = self.results_dir / 'daily_predictions_log.csv'
        
        if not log_path.exists():
            return []
        
        df = pd.read_csv(log_path)
        return df.to_dict('records')


# Singleton instance
data_service = DataService()

