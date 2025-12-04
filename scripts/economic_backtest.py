"""
Economic Backtesting - Trading Strategy Performance

Tests if model predictions can generate profitable trading returns
after accounting for transaction costs.

Tests the winners:
- WFC 2018-2019 (70% and 62% accuracy)
- BABA 2019 (68% accuracy)
- PFE 2019 (62% accuracy)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
import time

from config import TRAIN_TEST_SPLIT
from data_loader import (
    fetch_stock_data,
    fetch_politician_trades,
    fetch_historical_news_kaggle,
    aggregate_daily_sentiment
)
from feature_engineering import create_features, handle_missing_values
from model_xgboost import train_xgboost_model

# Winner stocks to test
WINNER_STOCKS = [
    ('WFC', 2018),
    ('WFC', 2019),
    ('BABA', 2019),
    ('PFE', 2019)
]

# Trading parameters
TRANSACTION_COST = 0.001  # 0.1% per trade
INITIAL_CAPITAL = 10000   # $10,000 starting capital

print("="*80)
print("ECONOMIC BACKTESTING - TRADING STRATEGY PERFORMANCE")
print("="*80)
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Transaction Cost: {TRANSACTION_COST*100}% per trade")
print(f"Testing {len(WINNER_STOCKS)} stock-year combinations\n")

results = []

for stock, year in WINNER_STOCKS:
    print(f"\n{'='*80}")
    print(f"{stock} - {year}")
    print(f"{'='*80}\n")
    
    try:
        # Fetch data
        print(f"   [1/6] Fetching data...")
        stock_data = fetch_stock_data(stock, f'{year}-01-01', f'{year}-12-31')
        news_data = fetch_historical_news_kaggle(stock, f'{year}-01-01', f'{year}-12-31')
        news_sentiment = aggregate_daily_sentiment(news_data)
        trades_data = fetch_politician_trades(stock)
        print(f"   [OK] {len(stock_data)} trading days, {len(news_data)} news articles")
        
        # Create features
        print(f"   [2/6] Engineering features...")
        X, y, dates = create_features(stock_data, news_sentiment, trades_data)
        print(f"   [OK] {len(X.columns)} features")
        
        # Clean data
        print(f"   [3/6] Cleaning data...")
        X_clean = handle_missing_values(X, strategy='drop')
        y_clean = y.loc[X_clean.index]
        dates_clean = dates.loc[X_clean.index]
        
        # Get actual prices for these dates
        # Normalize ALL dates to midnight AND remove timezone for consistent comparison
        stock_data_copy = stock_data.copy()
        stock_data_copy['Date'] = pd.to_datetime(stock_data_copy['Date']).dt.tz_localize(None).dt.normalize()
        dates_clean_normalized = pd.to_datetime(dates_clean.values).normalize()
        
        # Create a mapping from dates to prices
        price_dict = dict(zip(stock_data_copy['Date'], stock_data_copy['Close']))
        
        # Filter to only keep dates that have price data available
        valid_mask = [d in price_dict for d in dates_clean_normalized]
        
        X_clean = X_clean[valid_mask]
        y_clean = y_clean[valid_mask]
        dates_clean = dates_clean[valid_mask]
        dates_clean_normalized = dates_clean_normalized[valid_mask]
        
        # Now get prices for the filtered dates
        if len(dates_clean_normalized) > 0:
            prices = np.array([price_dict[d] for d in dates_clean_normalized])
        else:
            print(f"   [WARN]  WARNING: No valid price data for any dates! Skipping this stock-year.")
            continue
        
        # Reset indices to avoid misalignment during train-test split
        X_clean = X_clean.reset_index(drop=True)
        y_clean = y_clean.reset_index(drop=True)
        dates_clean = dates_clean.reset_index(drop=True)
        
        print(f"   [OK] {len(X_clean)} samples")
        
        # Split data
        split_idx = int(len(X_clean) * TRAIN_TEST_SPLIT)
        X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
        y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
        dates_test = dates_clean[split_idx:]
        prices_test = prices[split_idx:]
        
        # Train model
        print(f"   [4/6] Training model...")
        model = train_xgboost_model(X_train, y_train, verbose=False)
        print(f"   [OK] Model trained")
        
        # Get predictions and probabilities
        print(f"   [5/6] Generating predictions...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of UP
        
        # Calculate actual returns
        actual_returns = []
        for i in range(len(prices_test) - 1):
            ret = (prices_test[i+1] - prices_test[i]) / prices_test[i]
            actual_returns.append(ret)
        actual_returns.append(0)  # Last day has no return
        actual_returns = np.array(actual_returns)
        
        print(f"   [OK] {len(y_pred)} predictions generated")
        
        # Backtest trading strategy
        print(f"   [6/6] Backtesting strategy...")
        
        capital = INITIAL_CAPITAL
        position = None  # None = cash, 'long' = holding stock
        shares = 0
        trade_count = 0
        wins = 0
        losses = 0
        
        equity_curve = [capital]
        buy_and_hold_curve = [INITIAL_CAPITAL]
        
        for i in range(len(y_pred)):
            current_price = prices_test[i]
            prediction = y_pred[i]
            confidence = y_prob[i]
            
            # Buy and hold benchmark
            if i == 0:
                bh_shares = INITIAL_CAPITAL / current_price
            buy_and_hold_value = bh_shares * current_price
            buy_and_hold_curve.append(buy_and_hold_value)
            
            # Trading strategy: Only trade with high confidence (>60%)
            if prediction == 1 and confidence > 0.6 and position is None:
                # Buy signal
                shares = capital / current_price
                cost = shares * current_price * (1 + TRANSACTION_COST)
                capital = 0
                position = 'long'
                trade_count += 1
                
            elif (prediction == 0 or confidence < 0.4) and position == 'long':
                # Sell signal
                proceeds = shares * current_price * (1 - TRANSACTION_COST)
                
                # Track win/loss
                if proceeds > cost:
                    wins += 1
                else:
                    losses += 1
                    
                capital = proceeds
                shares = 0
                position = None
            
            # Calculate current equity
            if position == 'long':
                equity = shares * current_price
            else:
                equity = capital
            
            equity_curve.append(equity)
        
        # Final liquidation if still holding
        if position == 'long':
            final_proceeds = shares * prices_test[-1] * (1 - TRANSACTION_COST)
            if final_proceeds > cost:
                wins += 1
            else:
                losses += 1
            capital = final_proceeds
        
        # Calculate metrics
        final_equity = equity_curve[-1]
        total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
        buy_hold_return = (buy_and_hold_curve[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
        
        # Sharpe ratio (annualized)
        equity_returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252) if np.std(equity_returns) > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # Store results
        results.append({
            'stock': stock,
            'year': year,
            'initial_capital': INITIAL_CAPITAL,
            'final_equity': final_equity,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': trade_count,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'test_samples': len(y_test)
        })
        
        print(f"\n    RESULTS:")
        print(f"      Strategy Return:  {total_return:>7.1%}")
        print(f"      Buy & Hold:       {buy_hold_return:>7.1%}")
        print(f"      Excess Return:    {(total_return - buy_hold_return):>7.1%}")
        print(f"      Sharpe Ratio:     {sharpe:>7.2f}")
        print(f"      Max Drawdown:     {max_drawdown:>7.1%}")
        print(f"      Trades:           {trade_count:>7}")
        print(f"      Win Rate:         {win_rate:>7.1%} ({wins}W-{losses}L)")
        print(f"      Final Value:      ${final_equity:>10,.2f}")
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        continue

# Summary
print("\n" + "="*80)
print("SUMMARY - ALL STRATEGIES")
print("="*80 + "\n")

if results:
    results_df = pd.DataFrame(results)
    
    print("Individual Results:")
    print(results_df[['stock', 'year', 'total_return', 'buy_hold_return', 
                      'excess_return', 'sharpe_ratio', 'win_rate']].to_string(index=False))
    
    # Aggregate statistics
    print(f"\n\nAggregate Statistics:")
    print(f"  Average Strategy Return:  {results_df['total_return'].mean():>7.1%}")
    print(f"  Average Buy & Hold:       {results_df['buy_hold_return'].mean():>7.1%}")
    print(f"  Average Excess Return:    {results_df['excess_return'].mean():>7.1%}")
    print(f"  Average Sharpe Ratio:     {results_df['sharpe_ratio'].mean():>7.2f}")
    print(f"  Average Win Rate:         {results_df['win_rate'].mean():>7.1%}")
    print(f"  Total Trades:             {results_df['num_trades'].sum():>7}")
    
    # Best performer
    best = results_df.loc[results_df['excess_return'].idxmax()]
    print(f"\n   Best Performer: {best['stock']} {int(best['year'])}")
    print(f"     Excess Return: {best['excess_return']:.1%}")
    print(f"     Sharpe Ratio:  {best['sharpe_ratio']:.2f}")
    
    # Save results
    output_path = 'results/economic_backtest_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n[OK] Results saved to: {output_path}")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    avg_excess = results_df['excess_return'].mean()
    if avg_excess > 0.05:
        print("[OK] PROFITABLE: Strategy significantly outperforms buy-and-hold")
        print(f"   Average excess return of {avg_excess:.1%} suggests real trading value")
    elif avg_excess > 0:
        print("[WARN]  MARGINALLY PROFITABLE: Strategy slightly beats buy-and-hold")
        print(f"   Excess return of {avg_excess:.1%} may not cover all real-world costs")
    else:
        print("[ERROR] NOT PROFITABLE: Strategy underperforms buy-and-hold")
        print(f"   Negative excess return of {avg_excess:.1%}")
    
    avg_sharpe = results_df['sharpe_ratio'].mean()
    if avg_sharpe > 1.0:
        print(f"[OK] STRONG RISK-ADJUSTED RETURNS: Sharpe ratio of {avg_sharpe:.2f}")
    elif avg_sharpe > 0.5:
        print(f"[WARN]  MODERATE RISK-ADJUSTED RETURNS: Sharpe ratio of {avg_sharpe:.2f}")
    else:
        print(f"[ERROR] WEAK RISK-ADJUSTED RETURNS: Sharpe ratio of {avg_sharpe:.2f}")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)

