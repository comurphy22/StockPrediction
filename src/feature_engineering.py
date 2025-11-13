"""
Feature engineering for stock prediction model.
Creates technical indicators and prepares features for modeling.
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator


def create_features(df: pd.DataFrame, 
                   sentiment_df: pd.DataFrame = None,
                   politician_df: pd.DataFrame = None,
                   ticker: str = None,
                   add_market_features: bool = True) -> tuple:
    """
    Create features from stock data, including technical indicators, volatility,
    and market context. Optionally merge in sentiment and politician trade data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Stock data DataFrame with columns: Date, Open, High, Low, Close, Volume
    sentiment_df : pd.DataFrame, optional
        Daily sentiment scores from news
    politician_df : pd.DataFrame, optional
        Politician trading data
    ticker : str, optional
        Stock ticker symbol (needed for sector-specific features)
    add_market_features : bool, default=True
        Whether to add market context features (SPY, VIX, sector ETFs)
    
    Returns:
    --------
    tuple: (X, y, dates)
        X: Feature DataFrame
        y: Target variable (1 if next day close > today's close, 0 otherwise)
        dates: Date column for reference
    """
    # Make a copy to avoid modifying the original
    features_df = df.copy()
    
    # Ensure Date column is datetime and timezone-naive
    if 'Date' in features_df.columns:
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        # Remove timezone if present and normalize to midnight
        if hasattr(features_df['Date'].dtype, 'tz') and features_df['Date'].dtype.tz is not None:
            features_df['Date'] = features_df['Date'].dt.tz_localize(None)
        features_df['Date'] = features_df['Date'].dt.normalize()
        features_df = features_df.sort_values('Date').reset_index(drop=True)
    
    # === TECHNICAL INDICATORS ===
    
    # Simple Moving Averages
    print("Creating SMA indicators...")
    sma_10 = SMAIndicator(close=features_df['Close'], window=10)
    sma_20 = SMAIndicator(close=features_df['Close'], window=20)
    sma_50 = SMAIndicator(close=features_df['Close'], window=50)
    
    features_df['SMA_10'] = sma_10.sma_indicator()
    features_df['SMA_20'] = sma_20.sma_indicator()
    features_df['SMA_50'] = sma_50.sma_indicator()
    
    # Relative Strength Index
    print("Creating RSI indicator...")
    rsi = RSIIndicator(close=features_df['Close'], window=14)
    features_df['RSI'] = rsi.rsi()
    
    # MACD
    print("Creating MACD indicators...")
    macd = MACD(close=features_df['Close'])
    features_df['MACD'] = macd.macd()
    features_df['MACD_signal'] = macd.macd_signal()
    features_df['MACD_diff'] = macd.macd_diff()
    
    # Price-based features
    features_df['Price_change'] = features_df['Close'].pct_change()
    features_df['Volume_change'] = features_df['Volume'].pct_change()
    
    # High-Low spread
    features_df['HL_spread'] = (features_df['High'] - features_df['Low']) / features_df['Close']
    
    # Moving average crossovers
    features_df['SMA_10_20_cross'] = (features_df['SMA_10'] - features_df['SMA_20']) / features_df['Close']
    features_df['SMA_20_50_cross'] = (features_df['SMA_20'] - features_df['SMA_50']) / features_df['Close']
    
    # === VOLATILITY FEATURES ===
    print("Creating volatility features...")
    
    # Realized volatility (standard deviation of returns)
    features_df['realized_vol_5d'] = features_df['Price_change'].rolling(5).std()
    features_df['realized_vol_20d'] = features_df['Price_change'].rolling(20).std()
    
    # Average True Range (ATR) - measure of volatility
    high_low = features_df['High'] - features_df['Low']
    high_close = abs(features_df['High'] - features_df['Close'].shift(1))
    low_close = abs(features_df['Low'] - features_df['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features_df['ATR_14'] = true_range.rolling(14).mean()
    
    # Normalized ATR (ATR / Price)
    features_df['ATR_pct'] = features_df['ATR_14'] / features_df['Close']
    
    # Bollinger Band width (measure of volatility expansion/contraction)
    sma_20 = features_df['SMA_20']
    std_20 = features_df['Close'].rolling(20).std()
    features_df['BB_width'] = (2 * std_20) / sma_20
    
    # Volatility percentile (current volatility vs historical)
    # Note: needs 252 days of history, so will have NaN for first year
    features_df['vol_percentile'] = features_df['realized_vol_20d'].rolling(252, min_periods=60).rank(pct=True)
    
    # Volatility regime change (is volatility expanding?)
    features_df['vol_expanding'] = (features_df['realized_vol_5d'] > features_df['realized_vol_20d']).astype(int)
    
    # === MARKET CONTEXT FEATURES ===
    if add_market_features and 'Date' in features_df.columns:
        print("Creating market context features...")
        
        try:
            from market_data_loader import (
                fetch_all_market_indicators, 
                fetch_sector_etfs, 
                get_stock_sector
            )
            
            # Get date range for market data
            start_date = features_df['Date'].min().strftime('%Y-%m-%d')
            end_date = features_df['Date'].max().strftime('%Y-%m-%d')
            
            # Fetch market indicators (SPY, QQQ, VIX)
            market_data = fetch_all_market_indicators(start_date, end_date, use_cache=True)
            
            # Add SPY (S&P 500) features
            if 'SPY' in market_data:
                spy_df = market_data['SPY'][['Date', 'Close']].copy()
                # Ensure Date is timezone-naive
                spy_df['Date'] = pd.to_datetime(spy_df['Date']).dt.tz_localize(None).dt.normalize()
                spy_df['SPY_return'] = spy_df['Close'].pct_change()
                spy_df['SPY_return_5d'] = spy_df['Close'].pct_change(5)
                spy_df['SPY_return_20d'] = spy_df['Close'].pct_change(20)
                spy_df = spy_df[['Date', 'SPY_return', 'SPY_return_5d', 'SPY_return_20d']]
                features_df = features_df.merge(spy_df, on='Date', how='left')
                
                # Relative strength vs SPY
                features_df['rel_strength_spy'] = features_df['Price_change'] - features_df['SPY_return']
                print("      ✅ Added SPY market features")
            
            # Add QQQ (NASDAQ) features
            if 'QQQ' in market_data:
                qqq_df = market_data['QQQ'][['Date', 'Close']].copy()
                # Ensure Date is timezone-naive
                qqq_df['Date'] = pd.to_datetime(qqq_df['Date']).dt.tz_localize(None).dt.normalize()
                qqq_df['QQQ_return'] = qqq_df['Close'].pct_change()
                qqq_df = qqq_df[['Date', 'QQQ_return']]
                features_df = features_df.merge(qqq_df, on='Date', how='left')
                print("      ✅ Added QQQ market features")
            
            # Add VIX (volatility index) features
            if '^VIX' in market_data:
                vix_df = market_data['^VIX'][['Date', 'Close']].copy()
                # Ensure Date is timezone-naive
                vix_df['Date'] = pd.to_datetime(vix_df['Date']).dt.tz_localize(None).dt.normalize()
                vix_df.columns = ['Date', 'VIX']
                vix_df['VIX_change'] = vix_df['VIX'].pct_change()
                vix_df['VIX_percentile'] = vix_df['VIX'].rolling(252, min_periods=60).rank(pct=True)
                features_df = features_df.merge(vix_df, on='Date', how='left')
                print("      ✅ Added VIX volatility features")
            
            # Add sector features if ticker is provided
            if ticker:
                sector_etf = get_stock_sector(ticker)
                if sector_etf:
                    sector_data = fetch_sector_etfs(start_date, end_date, use_cache=True)
                    
                    if sector_etf in sector_data:
                        sector_df = sector_data[sector_etf][['Date', 'Close']].copy()
                        # Ensure Date is timezone-naive
                        sector_df['Date'] = pd.to_datetime(sector_df['Date']).dt.tz_localize(None).dt.normalize()
                        sector_df['sector_return'] = sector_df['Close'].pct_change()
                        sector_df['sector_return_5d'] = sector_df['Close'].pct_change(5)
                        sector_df = sector_df[['Date', 'sector_return', 'sector_return_5d']]
                        features_df = features_df.merge(sector_df, on='Date', how='left')
                        
                        # Relative strength vs sector
                        features_df['rel_strength_sector'] = features_df['Price_change'] - features_df['sector_return']
                        print(f"      ✅ Added {sector_etf} sector features")
                    
                    # Calculate rolling beta to market (if we have SPY)
                    if 'SPY_return' in features_df.columns:
                        # Beta = Cov(stock, market) / Var(market)
                        rolling_cov = features_df['Price_change'].rolling(60).cov(features_df['SPY_return'])
                        rolling_var = features_df['SPY_return'].rolling(60).var()
                        features_df['beta_60d'] = rolling_cov / rolling_var
                        
                        # Correlation to market
                        features_df['corr_spy_60d'] = features_df['Price_change'].rolling(60).corr(features_df['SPY_return'])
                        print("      ✅ Added beta and correlation features")
        
        except Exception as e:
            print(f"      ⚠️  Could not add market features: {e}")
    
    # === MERGE SENTIMENT DATA (if provided) ===
    if sentiment_df is not None and not sentiment_df.empty:
        print("Merging sentiment data...")
        # Ensure date columns are compatible
        if 'Date' in features_df.columns:
            merge_key = 'Date'
            if 'date' in sentiment_df.columns:
                sentiment_df = sentiment_df.rename(columns={'date': 'Date'})
            sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
            
            # Remove timezone info to avoid merge conflicts
            if hasattr(features_df['Date'].dtype, 'tz') and features_df['Date'].dtype.tz is not None:
                features_df['Date'] = features_df['Date'].dt.tz_localize(None)
            if hasattr(sentiment_df['Date'].dtype, 'tz') and sentiment_df['Date'].dtype.tz is not None:
                sentiment_df['Date'] = sentiment_df['Date'].dt.tz_localize(None)
            
            # Normalize to date only (remove time component)
            features_df['Date'] = features_df['Date'].dt.normalize()
            sentiment_df['Date'] = sentiment_df['Date'].dt.normalize()
        else:
            merge_key = 'date'
        
        features_df = features_df.merge(
            sentiment_df, 
            on=merge_key, 
            how='left'
        )
        
        # Fill missing sentiment values with neutral (0)
        sentiment_cols = [col for col in features_df.columns if 'sentiment' in col.lower()]
        for col in sentiment_cols:
            features_df[col] = features_df[col].fillna(0)
    else:
        print("No sentiment data provided - skipping sentiment features")
    
    # === MERGE POLITICIAN TRADE DATA (if provided) ===
    if politician_df is not None and not politician_df.empty:
        print("Merging politician trade data...")
        
        # Aggregate politician trades by date
        # Count number of buys and sells per day
        politician_df['date'] = pd.to_datetime(politician_df['date'])
        
        # Remove timezone info to avoid merge conflicts
        if hasattr(politician_df['date'].dtype, 'tz') and politician_df['date'].dtype.tz is not None:
            politician_df['date'] = politician_df['date'].dt.tz_localize(None)
        
        # Create buy/sell indicators
        politician_agg = politician_df.groupby('date').agg({
            'transaction_type': lambda x: sum(x.str.lower().str.contains('buy', na=False)),
            'amount': 'sum'
        }).reset_index()
        
        politician_agg.columns = ['Date', 'politician_buy_count', 'politician_trade_amount']
        
        # Ensure Date column in features_df is also timezone-naive
        if 'Date' in features_df.columns:
            if hasattr(features_df['Date'].dtype, 'tz') and features_df['Date'].dtype.tz is not None:
                features_df['Date'] = features_df['Date'].dt.tz_localize(None)
        
        # Merge with features
        features_df = features_df.merge(
            politician_agg,
            on='Date',
            how='left'
        )
        
        # Fill missing values with 0 (no trades)
        features_df['politician_buy_count'] = features_df['politician_buy_count'].fillna(0)
        features_df['politician_trade_amount'] = features_df['politician_trade_amount'].fillna(0)
        
        # === ADVANCED POLITICIAN FEATURES ===
        print("Creating advanced politician features...")
        try:
            from advanced_politician_features import create_advanced_politician_features
            
            # Create advanced features aligned with our date index
            # Pass features_df as stock_df (it has the Date column)
            adv_features = create_advanced_politician_features(
                features_df[['Date']] if 'Date' in features_df.columns else features_df,
                politician_df
            )
            
            # Merge advanced features
            if not adv_features.empty:
                # Merge on Date
                if 'Date' in adv_features.columns and 'Date' in features_df.columns:
                    features_df = features_df.merge(adv_features, on='Date', how='left')
                else:
                    features_df = pd.concat([features_df, adv_features], axis=1)
                print(f"      ✅ Added {len(adv_features.columns)-1 if 'Date' in adv_features.columns else len(adv_features.columns)} advanced politician features")
            else:
                print("      ⚠️  No advanced features generated (insufficient data)")
        except Exception as e:
            print(f"      ⚠️  Could not create advanced features: {e}")
    else:
        print("No politician trade data provided - skipping politician features")
    
    # === CREATE TARGET VARIABLE ===
    print("Creating target variable...")
    # Target: 1 if tomorrow's close is higher than today's close, 0 otherwise
    features_df['Next_Close'] = features_df['Close'].shift(-1)
    features_df['Target'] = (features_df['Next_Close'] > features_df['Close']).astype(int)
    
    # Drop the last row (no target available)
    features_df = features_df[:-1].copy()
    
    # === PREPARE X AND Y ===
    
    # Drop columns that shouldn't be features
    columns_to_drop = ['Date', 'Next_Close', 'Target', 'Open', 'High', 'Low', 'Close', 
                       'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
    
    # Only drop columns that actually exist
    columns_to_drop = [col for col in columns_to_drop if col in features_df.columns]
    
    X = features_df.drop(columns=columns_to_drop)
    y = features_df['Target']
    
    # Keep the Date column for reference (separate from X)
    dates = features_df['Date'] if 'Date' in features_df.columns else None
    
    print(f"\nFeature engineering complete!")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"\nTarget distribution:\n{y.value_counts()}")
    print(f"Class balance: {y.mean():.2%} positive (price increase)")
    
    return X, y, dates


def handle_missing_values(X: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in feature DataFrame.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature DataFrame
    strategy : str
        Strategy for handling missing values:
        - 'drop': Drop rows with any missing values
        - 'ffill': Forward fill missing values
        - 'mean': Fill with column mean
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values handled
    """
    print(f"\nMissing values before handling:\n{X.isnull().sum()[X.isnull().sum() > 0]}")
    
    if strategy == 'drop':
        X_clean = X.dropna()
        print(f"Dropped {len(X) - len(X_clean)} rows with missing values")
    elif strategy == 'ffill':
        X_clean = X.fillna(method='ffill').fillna(0)
        print("Forward filled missing values")
    elif strategy == 'mean':
        X_clean = X.fillna(X.mean())
        print("Filled missing values with column means")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"Missing values after handling: {X_clean.isnull().sum().sum()}")
    
    return X_clean


if __name__ == "__main__":
    # Example usage
    from data_loader import fetch_stock_data
    
    print("=== Feature Engineering Example ===\n")
    
    # Fetch sample data
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Create features (baseline - no sentiment or politician data)
    X, y, dates = create_features(stock_data)
    
    # Handle missing values
    X_clean = handle_missing_values(X, strategy='drop')
    
    # Align y with cleaned X
    y_clean = y.loc[X_clean.index]
    
    print(f"\nFinal dataset shape: X={X_clean.shape}, y={y_clean.shape}")
    print(f"\nSample features:\n{X_clean.head()}")
