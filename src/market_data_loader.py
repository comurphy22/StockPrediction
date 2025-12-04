"""
Market data loader for broad market indicators and sector ETFs.
Fetches VIX, SPY, QQQ, and sector ETFs to provide market context features.

This module helps answer: "Is the stock moving with or against the market?"
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


# Cache directory for market data
CACHE_DIR = Path(__file__).parent.parent / 'data' / 'market_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Market indicators we'll fetch
MARKET_TICKERS = {
    'SPY': 'S&P 500 ETF',
    'QQQ': 'NASDAQ 100 ETF',
    '^VIX': 'CBOE Volatility Index',
}

# Sector ETFs (SPDR Select Sector ETFs)
SECTOR_ETFS = {
    'XLF': 'Financials',
    'XLK': 'Technology',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
}

# Map stock tickers to their primary sectors (for sector-relative features)
STOCK_SECTORS = {
    'AAPL': 'XLK',   # Technology
    'MSFT': 'XLK',   # Technology
    'GOOGL': 'XLK',  # Technology
    'AMZN': 'XLY',   # Consumer Discretionary
    'NFLX': 'XLY',   # Consumer Discretionary
    'TSLA': 'XLY',   # Consumer Discretionary
    'NVDA': 'XLK',   # Technology
    'BABA': 'XLY',   # Consumer Discretionary (international, but closest fit)
    'QCOM': 'XLK',   # Technology
    'MU': 'XLK',     # Technology
}


def _get_cache_path(ticker: str, start_date: str, end_date: str) -> Path:
    """Generate cache file path for a ticker and date range."""
    cache_key = f"{ticker}_{start_date}_{end_date}.pkl"
    return CACHE_DIR / cache_key


def fetch_market_data(ticker: str, 
                     start_date: str, 
                     end_date: str,
                     use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch market data (SPY, VIX, sector ETFs) with caching.
    
    Parameters:
    -----------
    ticker : str
        Market ticker symbol (e.g., 'SPY', '^VIX', 'XLK')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    use_cache : bool
        Whether to use cached data if available
    
    Returns:
    --------
    pd.DataFrame
        Market data with columns: Date, Open, High, Low, Close, Volume
    """
    # Check cache first
    cache_path = _get_cache_path(ticker, start_date, end_date)
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                df = pickle.load(f)
                return df
        except Exception as e:
            print(f"Warning: Cache read failed for {ticker}: {e}")
    
    # Fetch from yfinance
    try:
        # Add buffer to date range to ensure we have enough data for rolling calculations
        start_dt = pd.to_datetime(start_date) - timedelta(days=120)
        end_dt = pd.to_datetime(end_date) + timedelta(days=5)
        
        data = yf.download(
            ticker,
            start=start_dt.strftime('%Y-%m-%d'),
            end=end_dt.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if data.empty:
            print(f"Warning: No data returned for {ticker}")
            return pd.DataFrame()
        
        # Flatten multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Ensure Date column exists
        if 'Date' not in data.columns:
            if 'Datetime' in data.columns:
                data = data.rename(columns={'Datetime': 'Date'})
            else:
                data['Date'] = data.index
        
        # Normalize dates (remove timezone, normalize to midnight)
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()
        
        # Sort by date
        data = data.sort_values('Date').reset_index(drop=True)
        
        # Cache the result
        if use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"Warning: Cache write failed for {ticker}: {e}")
        
        return data
    
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


def fetch_all_market_indicators(start_date: str, 
                                end_date: str,
                                use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Fetch all market indicators (SPY, QQQ, VIX).
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    use_cache : bool
        Whether to use cached data
    
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with ticker as key, DataFrame as value
    """
    results = {}
    
    print(f"Fetching market indicators for {start_date} to {end_date}...")
    for ticker, description in MARKET_TICKERS.items():
        print(f"  Fetching {ticker} ({description})...", end=' ')
        df = fetch_market_data(ticker, start_date, end_date, use_cache)
        if not df.empty:
            results[ticker] = df
            print(f"✓ ({len(df)} days)")
        else:
            print("✗ (no data)")
    
    return results


def fetch_sector_etfs(start_date: str,
                     end_date: str,
                     use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Fetch all sector ETFs.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    use_cache : bool
        Whether to use cached data
    
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with sector ticker as key, DataFrame as value
    """
    results = {}
    
    print(f"Fetching sector ETFs for {start_date} to {end_date}...")
    for ticker, description in SECTOR_ETFS.items():
        print(f"  Fetching {ticker} ({description})...", end=' ')
        df = fetch_market_data(ticker, start_date, end_date, use_cache)
        if not df.empty:
            results[ticker] = df
            print(f"✓ ({len(df)} days)")
        else:
            print("✗ (no data)")
    
    return results


def get_stock_sector(ticker: str) -> Optional[str]:
    """
    Get the primary sector ETF for a given stock ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    
    Returns:
    --------
    str or None
        Sector ETF ticker (e.g., 'XLK') or None if unknown
    """
    return STOCK_SECTORS.get(ticker)


def calculate_market_returns(market_data: pd.DataFrame, 
                            periods: List[int] = [1, 5, 20]) -> pd.DataFrame:
    """
    Calculate returns for market indicators at various periods.
    
    Parameters:
    -----------
    market_data : pd.DataFrame
        Market data with Date and Close columns
    periods : List[int]
        Periods to calculate returns for (in days)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Date and return columns
    """
    if market_data.empty or 'Close' not in market_data.columns:
        return pd.DataFrame()
    
    df = market_data[['Date', 'Close']].copy()
    
    # Calculate returns for each period
    for period in periods:
        df[f'return_{period}d'] = df['Close'].pct_change(period)
    
    return df


if __name__ == '__main__':
    """Test the market data loader."""
    print("="*80)
    print("TESTING MARKET DATA LOADER")
    print("="*80)
    print()
    
    # Test fetching market indicators
    market_data = fetch_all_market_indicators('2019-01-01', '2019-12-31')
    
    print()
    print("="*80)
    print("MARKET DATA SUMMARY")
    print("="*80)
    
    for ticker, df in market_data.items():
        print(f"\n{ticker}:")
        print(f"  Rows: {len(df)}")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  Columns: {df.columns.tolist()}")
        if 'Close' in df.columns:
            print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    # Test sector ETFs
    print()
    print("="*80)
    print("FETCHING SECTOR ETFs")
    print("="*80)
    sector_data = fetch_sector_etfs('2019-01-01', '2019-12-31')
    
    print()
    print("="*80)
    print("SECTOR DATA SUMMARY")
    print("="*80)
    
    for ticker, df in sector_data.items():
        sector_name = SECTOR_ETFS[ticker]
        print(f"\n{ticker} ({sector_name}):")
        print(f"  Rows: {len(df)}")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Test stock sector mapping
    print()
    print("="*80)
    print("STOCK SECTOR MAPPING")
    print("="*80)
    
    test_stocks = ['AAPL', 'NFLX', 'TSLA', 'NVDA', 'BABA']
    for stock in test_stocks:
        sector = get_stock_sector(stock)
        sector_name = SECTOR_ETFS.get(sector, 'Unknown') if sector else 'Unknown'
        print(f"{stock:6s} → {sector:5s} ({sector_name})")
    
    print()
    print("[OK] Market data loader test complete!")
