"""
Backtest Service
================
Handles backtesting of trading strategies.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Add the src directory to path (for Railway deployment, src is inside backend/)
BACKEND_DIR = Path(__file__).parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
# Try local src first (Railway), then parent src (local dev)
if (BACKEND_DIR / 'src').exists():
    sys.path.insert(0, str(BACKEND_DIR / 'src'))
else:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from app.models.schemas import BacktestResult


class BacktestService:
    """Service for backtesting trading strategies."""
    
    def __init__(self):
        self.results_dir = PROJECT_ROOT / 'results'
        self.cached_results: Dict[str, BacktestResult] = {}
        
        # Import ML modules
        try:
            from data_loader import fetch_stock_data, fetch_politician_trades
            from feature_engineering import create_features, handle_missing_values
            from model_xgboost import train_xgboost_model
            
            self.fetch_stock_data = fetch_stock_data
            self.fetch_politician_trades = fetch_politician_trades
            self.create_features = create_features
            self.handle_missing_values = handle_missing_values
            self.train_xgboost_model = train_xgboost_model
        except ImportError as e:
            print(f"Warning: Could not import ML modules: {e}")
    
    def run_backtest(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001
    ) -> BacktestResult:
        """
        Run a backtest for a trading strategy.
        
        Uses walk-forward validation: train on first 80%, test on last 20%.
        """
        ticker = ticker.upper()
        
        # Fetch data
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        stock_data = self.fetch_stock_data(ticker, start_date, end_date)
        
        try:
            trades_data = self.fetch_politician_trades(ticker)
        except Exception:
            trades_data = pd.DataFrame()
        
        # Create features
        X, y, dates = self.create_features(stock_data, pd.DataFrame(), trades_data, ticker=ticker)
        X_clean = self.handle_missing_values(X, strategy='drop')
        y_clean = y.loc[X_clean.index]
        dates_clean = dates.loc[X_clean.index] if dates is not None else None
        
        # Walk-forward split (80/20)
        split_idx = int(len(X_clean) * 0.8)
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]
        
        # Train model
        model = self.train_xgboost_model(X_train, y_train, verbose=False)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Get prices for the test period
        test_prices = stock_data.iloc[split_idx:split_idx + len(X_test)]['Close'].values
        
        # Simulate trading
        results = self._simulate_trading(
            predictions=predictions,
            prices=test_prices,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost
        )
        
        # Calculate accuracy
        accuracy = (predictions == y_test.values).mean()
        
        # Create result
        result = BacktestResult(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_value=results['final_value'],
            total_return=results['total_return'],
            buy_hold_return=results['buy_hold_return'],
            excess_return=results['excess_return'],
            sharpe_ratio=results['sharpe_ratio'],
            max_drawdown=results['max_drawdown'],
            win_rate=results['win_rate'],
            total_trades=results['total_trades'],
            accuracy=float(accuracy)
        )
        
        # Cache result
        self.cached_results[ticker] = result
        
        return result
    
    def _simulate_trading(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        initial_capital: float,
        transaction_cost: float
    ) -> Dict[str, float]:
        """Simulate a trading strategy based on predictions."""
        capital = initial_capital
        position = 0  # 0 = no position, 1 = long
        trades = 0
        wins = 0
        entry_price = 0
        
        portfolio_values = [initial_capital]
        
        for i in range(len(predictions) - 1):
            pred = predictions[i]
            current_price = prices[i]
            next_price = prices[i + 1]
            
            # Trading logic
            if pred == 1 and position == 0:  # Buy signal, no position
                position = 1
                entry_price = current_price
                capital *= (1 - transaction_cost)  # Pay transaction cost
                trades += 1
                
            elif pred == 0 and position == 1:  # Sell signal, have position
                # Calculate return
                trade_return = (current_price - entry_price) / entry_price
                capital *= (1 + trade_return)
                capital *= (1 - transaction_cost)  # Pay transaction cost
                
                if trade_return > 0:
                    wins += 1
                
                position = 0
                trades += 1
            
            # Update portfolio value
            if position == 1:
                portfolio_values.append(capital * (prices[i + 1] / current_price))
            else:
                portfolio_values.append(capital)
        
        # Close any remaining position
        if position == 1:
            trade_return = (prices[-1] - entry_price) / entry_price
            capital *= (1 + trade_return)
            capital *= (1 - transaction_cost)
            if trade_return > 0:
                wins += 1
            trades += 1
        
        final_value = capital
        
        # Calculate metrics
        total_return = (final_value - initial_capital) / initial_capital
        buy_hold_return = (prices[-1] - prices[0]) / prices[0]
        excess_return = total_return - buy_hold_return
        
        # Calculate Sharpe ratio (assuming 252 trading days, risk-free rate = 0)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Win rate
        win_rate = wins / (trades // 2) if trades > 1 else 0
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': excess_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': trades
        }
    
    def get_cached_result(self, ticker: str) -> Optional[BacktestResult]:
        """Get cached backtest result for a ticker."""
        return self.cached_results.get(ticker.upper())
    
    def get_all_results_summary(self) -> Dict[str, Any]:
        """Get summary of all available backtest results."""
        # Try to load from CSV first
        results = []
        
        backtest_path = self.results_dir / 'economic_backtest_results.csv'
        if backtest_path.exists():
            df = pd.read_csv(backtest_path)
            results = df.to_dict('records')
        
        # Add any cached results
        for ticker, result in self.cached_results.items():
            results.append({
                'ticker': result.ticker,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'accuracy': result.accuracy
            })
        
        return {
            'results': results,
            'total_stocks': len(results),
            'average_sharpe': np.mean([r.get('sharpe_ratio', 0) for r in results]) if results else 0,
            'average_accuracy': np.mean([r.get('accuracy', 0) for r in results]) if results else 0
        }


# Singleton instance
backtest_service = BacktestService()

