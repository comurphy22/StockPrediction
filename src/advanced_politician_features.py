"""
Advanced Politician Trading Features

This module provides enhanced feature engineering for politician trading signals,
including net position indicators, temporal patterns, and rolling statistics.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def create_advanced_politician_features(
    stock_df: pd.DataFrame,
    politician_df: pd.DataFrame,
    lookback_windows=[30, 60, 90]
) -> pd.DataFrame:
    """
    Create advanced politician trading features with net indices and temporal patterns.
    
    This function enhances basic politician features (buy_count, trade_amount) with:
    - Net position indicators (buy vs. sell directional signals)
    - Temporal features (momentum, recency, rolling windows)
    - Transaction quality metrics
    
    Parameters:
    -----------
    stock_df : pd.DataFrame
        Stock price data with Date column or DatetimeIndex
    politician_df : pd.DataFrame
        Politician trading data with columns:
        - transaction_date: Date of trade
        - transaction_type: 'purchase', 'sale', 'sale (full)', 'sale (partial)', 'exchange'
        - amount: Dollar value of trade
    lookback_windows : list, default=[30, 60, 90]
        Days to look back for rolling features
    
    Returns:
    --------
    pd.DataFrame
        Stock data with additional politician feature columns:
        - net_trade_index: (-1 to +1) buy vs. sell ratio
        - net_dollar_flow: net dollars (buys - sells)
        - buy_percentage: proportion of buys
        - trades_last_Nd: count of trades in last N days
        - amount_last_Nd: dollar volume in last N days
        - trade_momentum_Nd: recent vs. historical trade velocity
        - days_since_last_trade: recency indicator
    
    Example:
    --------
    >>> politician_data = fetch_politician_trades('AAPL', '2020-01-01', '2020-12-31')
    >>> stock_data = fetch_stock_data('AAPL', '2020-01-01', '2020-12-31')
    >>> features = create_advanced_politician_features(stock_data, politician_data)
    >>> print(features['net_trade_index'].head())
    """
    
    if politician_df.empty:
        print("⚠️  No politician data provided, skipping advanced features")
        return stock_df
    
    # Ensure stock_df has Date column or DatetimeIndex
    if 'Date' not in stock_df.columns and not isinstance(stock_df.index, pd.DatetimeIndex):
        raise ValueError("stock_df must have 'Date' column or DatetimeIndex")
    
    # Work with copy to avoid modifying original
    stock_df = stock_df.copy()
    politician_df = politician_df.copy()
    
    # Normalize column names
    if 'transaction_date' in politician_df.columns:
        politician_df['date'] = pd.to_datetime(politician_df['transaction_date'])
    elif 'date' in politician_df.columns:
        politician_df['date'] = pd.to_datetime(politician_df['date'])
    else:
        raise ValueError("politician_df must have 'transaction_date' or 'date' column")
    
    print("🏛️  Creating advanced politician features...")
    
    # === PHASE 1: NET POSITION INDICATORS ===
    
    # Classify transactions as buys or sells
    politician_df['is_buy'] = politician_df['transaction_type'].str.lower() == 'purchase'
    politician_df['is_sell'] = politician_df['transaction_type'].str.lower().str.contains('sale')
    politician_df['is_full_sale'] = politician_df['transaction_type'].str.lower() == 'sale (full)'
    politician_df['is_partial_sale'] = politician_df['transaction_type'].str.lower() == 'sale (partial)'
    
    # Aggregate by date
    daily_trades = politician_df.groupby('date').agg({
        'is_buy': 'sum',
        'is_sell': 'sum',
        'is_full_sale': 'sum',
        'is_partial_sale': 'sum',
        'amount': 'sum'
    }).rename(columns={
        'is_buy': 'buy_count',
        'is_sell': 'sell_count',
        'is_full_sale': 'full_sale_count',
        'is_partial_sale': 'partial_sale_count',
        'amount': 'total_amount'
    })
    
    # Calculate buy/sell amounts separately
    buy_df = politician_df[politician_df['is_buy']].groupby('date')['amount'].sum()
    sell_df = politician_df[politician_df['is_sell']].groupby('date')['amount'].sum()
    
    daily_trades['buy_amount'] = buy_df
    daily_trades['sell_amount'] = sell_df
    daily_trades = daily_trades.fillna(0)
    
    # Net indices (key predictive features)
    daily_trades['net_trade_index'] = (
        (daily_trades['buy_count'] - daily_trades['sell_count']) /
        (daily_trades['buy_count'] + daily_trades['sell_count'] + 1)  # +1 to avoid division by zero
    )
    
    daily_trades['net_dollar_flow'] = (
        daily_trades['buy_amount'] - daily_trades['sell_amount']
    )
    
    daily_trades['buy_percentage'] = (
        daily_trades['buy_count'] /
        (daily_trades['buy_count'] + daily_trades['sell_count'] + 1)
    )
    
    print(f"   ✅ Net position indicators: net_trade_index, net_dollar_flow, buy_percentage")
    
    # === PHASE 2: TEMPORAL FEATURES ===
    
    # Ensure full date range (including days with no trades)
    if isinstance(stock_df.index, pd.DatetimeIndex):
        all_dates = stock_df.index
    else:
        all_dates = pd.to_datetime(stock_df['Date'])
    
    # Reindex to include all trading days
    daily_trades = daily_trades.reindex(all_dates, fill_value=0)
    
    # Rolling window statistics
    for window in lookback_windows:
        # Trade counts in window
        daily_trades[f'trades_last_{window}d'] = (
            (daily_trades['buy_count'] + daily_trades['sell_count'])
            .rolling(window=window, min_periods=1).sum()
        )
        
        # Dollar volume in window
        daily_trades[f'amount_last_{window}d'] = (
            daily_trades['total_amount']
            .rolling(window=window, min_periods=1).sum()
        )
        
        # Net flow in window
        daily_trades[f'net_flow_last_{window}d'] = (
            daily_trades['net_dollar_flow']
            .rolling(window=window, min_periods=1).sum()
        )
    
    # Trade momentum (compare recent to historical)
    if 30 in lookback_windows and 90 in lookback_windows:
        daily_trades['trade_momentum_30d'] = (
            daily_trades['trades_last_30d'] /
            (daily_trades['trades_last_90d'].shift(30).fillna(1) + 0.1)  # Avoid division by zero
        )
        
        daily_trades['dollar_momentum_30d'] = (
            daily_trades['amount_last_30d'] /
            (daily_trades['amount_last_90d'].shift(30).fillna(1) + 0.1)
        )
    
    print(f"   ✅ Temporal features: {len(lookback_windows)} rolling windows + momentum")
    
    # Days since last trade (recency indicator)
    trade_dates = daily_trades[daily_trades['buy_count'] + daily_trades['sell_count'] > 0].index
    daily_trades['days_since_last_trade'] = 999  # Large number for "never"
    
    for i, date in enumerate(daily_trades.index):
        if date in trade_dates:
            daily_trades.loc[date, 'days_since_last_trade'] = 0
        else:
            prev_trades = trade_dates[trade_dates < date]
            if len(prev_trades) > 0:
                days_diff = (date - prev_trades[-1]).days
                daily_trades.loc[date, 'days_since_last_trade'] = days_diff
    
    # Cap at 180 days (6 months = essentially no recent activity)
    daily_trades['days_since_last_trade'] = daily_trades['days_since_last_trade'].clip(upper=180)
    
    print(f"   ✅ Recency indicator: days_since_last_trade")
    
    # === PHASE 3: QUALITY METRICS ===
    
    # Conviction score (weight full sales higher than partial)
    daily_trades['conviction_score'] = (
        (daily_trades['full_sale_count'] * 2.0 +
         daily_trades['sell_count'] +
         daily_trades['partial_sale_count'] * 0.5 +
         daily_trades['buy_count']) /
        (daily_trades['buy_count'] + daily_trades['sell_count'] + 1)
    )
    
    print(f"   ✅ Quality metrics: conviction_score")
    
    # === MERGE WITH STOCK DATA ===
    
    # Merge on Date
    if isinstance(stock_df.index, pd.DatetimeIndex):
        result = stock_df.join(daily_trades, how='left')
    else:
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        result = stock_df.set_index('Date').join(daily_trades, how='left')
        result = result.reset_index()
    
    # Fill any remaining NaN with 0 (no politician activity)
    politician_cols = [col for col in result.columns if any(x in col.lower() for x in 
                      ['trade', 'politician', 'buy', 'sell', 'flow', 'conviction', 'momentum'])]
    result[politician_cols] = result[politician_cols].fillna(0)
    
    # Count new features
    new_features = len([col for col in result.columns if col not in stock_df.columns])
    print(f"   ✅ Added {new_features} advanced politician features")
    
    return result


# Example usage and testing
if __name__ == "__main__":
    print("Testing advanced politician features...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
    stock_data = pd.DataFrame({
        'Date': dates,
        'Close': 100 + np.random.randn(len(dates)).cumsum(),
        'Volume': 1000000 + np.random.randint(-100000, 100000, len(dates))
    })
    
    # Sample politician trades
    trade_dates = pd.to_datetime(['2020-01-15', '2020-01-20', '2020-02-10', '2020-02-25', '2020-03-15'])
    politician_data = pd.DataFrame({
        'transaction_date': trade_dates,
        'transaction_type': ['purchase', 'sale', 'purchase', 'sale (full)', 'purchase'],
        'amount': [50000, 30000, 75000, 60000, 40000]
    })
    
    # Test feature creation
    enhanced_data = create_advanced_politician_features(stock_data, politician_data)
    
    print(f"\nOriginal columns: {stock_data.shape[1]}")
    print(f"Enhanced columns: {enhanced_data.shape[1]}")
    print(f"\nNew politician features:")
    pol_features = [col for col in enhanced_data.columns if col not in stock_data.columns]
    for feat in pol_features:
        print(f"  - {feat}")
    
    print(f"\nSample data with advanced features:")
    print(enhanced_data[['Date', 'Close', 'net_trade_index', 'net_dollar_flow', 
                         'trades_last_30d', 'days_since_last_trade']].head(20))
