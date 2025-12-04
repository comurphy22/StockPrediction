"""
Data loading functions for stock data, politician trades, and news sentiment.
"""

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon for sentiment analysis (run once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch OHLCV stock data using yfinance.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
    """
    import time
    
    # Retry logic for API issues
    max_retries = 3
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            
            # Add small delay to avoid rate limiting
            if attempt > 0:
                time.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s
            
            # Try with period parameter as backup
            df = stock.history(start=start_date, end=end_date)
            
            # If empty, try alternative method
            if df.empty:
                print(f"   Retry attempt {attempt + 1}/{max_retries} for {ticker}...")
                time.sleep(1)
                continue
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Verify we got data
            if len(df) == 0:
                if attempt < max_retries - 1:
                    continue
                else:
                    raise ValueError(f"No data found for ticker {ticker}")
            
            return df
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   Attempt {attempt + 1} failed, retrying...")
                time.sleep(2)
                continue
            else:
                print(f"   Error fetching stock data for {ticker} after {max_retries} attempts: {str(e)}")
                raise ValueError(f"No data found for ticker {ticker}")


def fetch_politician_trades(ticker: str, api_key: str = None) -> pd.DataFrame:
    """
    Fetch politician (congressional) trading data from Quiver Quantitative API.
    
    Retrieves historical trading data from members of Congress for a specific ticker.
    Data includes transaction type (purchase/sale), amount, date, and politician details.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    api_key : str, optional
        Quiver Quantitative API key. If None, will try to import from config.py
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, politician_name, transaction_type, amount, ticker, etc.
    """
    # Get API key
    if api_key is None:
        try:
            from config import QUIVER_API_KEY, QUIVER_BASE_URL
            api_key = QUIVER_API_KEY
            base_url = QUIVER_BASE_URL
        except ImportError:
            print("Warning: config.py not found. Please provide api_key parameter.")
            print("Returning empty DataFrame.")
            return pd.DataFrame(columns=['date', 'politician_name', 'transaction_type', 
                                         'amount', 'ticker'])
    else:
        base_url = "https://api.quiverquant.com/beta"
    
    print(f"Fetching politician trading data for {ticker}...")
    
    try:
        # Quiver Quantitative API endpoint for congressional trading
        url = f"{base_url}/historical/congresstrading/{ticker}"
        
        # Headers with API key
        headers = {
            'Authorization': f'Bearer {api_key}',
            'accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            print(f"No politician trading data found for {ticker}")
            return pd.DataFrame(columns=['date', 'politician_name', 'transaction_type', 
                                         'amount', 'ticker'])
        
        # Transform API response to DataFrame
        df = pd.DataFrame(data)
        
        # Standardize column names (Quiver uses specific naming)
        column_mapping = {
            'ReportDate': 'date',
            'Representative': 'politician_name',
            'Transaction': 'transaction_type',
            'Range': 'amount_range',
            'Ticker': 'ticker',
            'House': 'chamber',
            'Party': 'party',
            'District': 'district'
        }
        
        # Rename columns if they exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Ensure we have the required columns
        if 'date' not in df.columns:
            print("Error: 'date' column not found in API response")
            return pd.DataFrame(columns=['date', 'politician_name', 'transaction_type', 
                                         'amount', 'ticker'])
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert amount range to numeric value (take midpoint)
        # Quiver returns ranges like "$1,001 - $15,000"
        if 'amount_range' in df.columns:
            df['amount'] = df['amount_range'].apply(_parse_amount_range)
        else:
            df['amount'] = 0
        
        # Clean transaction types
        if 'transaction_type' in df.columns:
            df['transaction_type'] = df['transaction_type'].str.lower().str.strip()
        
        # Add ticker column if not present
        if 'ticker' not in df.columns:
            df['ticker'] = ticker.upper()
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Successfully fetched {len(df)} politician trades for {ticker}")
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        # Print summary statistics
        if 'transaction_type' in df.columns:
            print(f"\nTransaction type breakdown:")
            print(df['transaction_type'].value_counts())
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching politician trades: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text[:200]}")
        print("Returning empty DataFrame")
        return pd.DataFrame(columns=['date', 'politician_name', 'transaction_type', 
                                     'amount', 'ticker'])
    except Exception as e:
        print(f"Unexpected error in fetch_politician_trades: {str(e)}")
        return pd.DataFrame(columns=['date', 'politician_name', 'transaction_type', 
                                     'amount', 'ticker'])


def _parse_amount_range(range_str: str) -> float:
    """
    Parse amount range string from Quiver API and return midpoint.
    
    Example: "$1,001 - $15,000" -> 8000.5
    
    Parameters:
    -----------
    range_str : str
        Amount range string
    
    Returns:
    --------
    float
        Midpoint of the range
    """
    try:
        if pd.isna(range_str) or not range_str:
            return 0
        
        # Remove $ and commas, split by dash
        range_str = str(range_str).replace('$', '').replace(',', '')
        
        if '-' in range_str:
            parts = range_str.split('-')
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return (low + high) / 2
        else:
            # Single value
            return float(range_str.strip())
    except Exception:
        return 0


def fetch_news_sentiment(ticker: str, days_back: int = 30, api_key: str = None) -> pd.DataFrame:
    """
    Fetch news and calculate sentiment scores using VADER.
    
    Integrates with NewsAPI (https://newsapi.org/) to fetch real news headlines
    and calculates sentiment using NLTK's VADER sentiment analyzer.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    days_back : int
        Number of days of news to fetch (default: 30, max: 30 for free tier)
    api_key : str, optional
        NewsAPI key. If None, will try to import from config.py
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, headline, sentiment_compound, etc.
    """
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Get API key
    if api_key is None:
        try:
            from config import NEWSAPI_KEY, NEWSAPI_BASE_URL, DEFAULT_NEWS_PAGE_SIZE
            api_key = NEWSAPI_KEY
            base_url = NEWSAPI_BASE_URL
            page_size = DEFAULT_NEWS_PAGE_SIZE
        except ImportError:
            print("Error: config.py not found. Please provide api_key parameter.")
            return pd.DataFrame()
    else:
        base_url = "https://newsapi.org/v2/everything"
        page_size = 100
    
    # Calculate date range (NewsAPI free tier: max 1 month back)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=min(days_back, 30))
    
    print(f"Fetching news for {ticker} from {start_date.date()} to {end_date.date()}...")
    
    try:
        # NewsAPI request parameters
        params = {
            'q': f'{ticker} OR stock',  # Search query
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': page_size,
            'apiKey': api_key
        }
        
        # Make API request
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        
        news_data = response.json()
        
        if news_data.get('status') != 'ok':
            print(f"Error from NewsAPI: {news_data.get('message', 'Unknown error')}")
            return pd.DataFrame()
        
        articles = news_data.get('articles', [])
        
        if not articles:
            print(f"No news articles found for {ticker}")
            return pd.DataFrame()
        
        print(f"Retrieved {len(articles)} articles from NewsAPI")
        
        # Calculate sentiment scores for each article
        sentiment_data = []
        for article in articles:
            # Use title and description for sentiment
            title = article.get('title', '')
            description = article.get('description', '')
            
            # Combine title and description for better sentiment analysis
            text = f"{title}. {description}" if description else title
            
            if not text or text == 'None':
                continue
            
            # Get sentiment scores using VADER
            sentiment_scores = sia.polarity_scores(text)
            
            # Parse published date
            published_at = article.get('publishedAt', '')
            try:
                date = pd.to_datetime(published_at).date()
            except:
                date = datetime.now().date()
            
            sentiment_data.append({
                'date': date,
                'headline': title,
                'description': description,
                'source': article.get('source', {}).get('name', 'Unknown'),
                'url': article.get('url', ''),
                'sentiment_compound': sentiment_scores['compound'],  # -1 to 1
                'sentiment_positive': sentiment_scores['pos'],
                'sentiment_negative': sentiment_scores['neg'],
                'sentiment_neutral': sentiment_scores['neu']
            })
        
        df = pd.DataFrame(sentiment_data)
        
        if df.empty:
            print("No valid articles with sentiment data")
            return df
        
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"Successfully calculated sentiment for {len(df)} news items")
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"Average sentiment: {df['sentiment_compound'].mean():.3f}")
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news data: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error in fetch_news_sentiment: {str(e)}")
        return pd.DataFrame()


def aggregate_daily_sentiment(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple news sentiment scores to daily level.
    
    Parameters:
    -----------
    sentiment_df : pd.DataFrame
        DataFrame with sentiment scores from fetch_news_sentiment
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with daily aggregated sentiment scores
    """
    if sentiment_df.empty:
        return pd.DataFrame(columns=['date', 'avg_sentiment', 'news_count'])
    
    # Normalize timestamps to date-only for daily aggregation
    sentiment_df = sentiment_df.copy()
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.normalize()
    
    daily_sentiment = sentiment_df.groupby('date').agg({
        'sentiment_compound': 'mean',
        'sentiment_positive': 'mean',
        'sentiment_negative': 'mean',
        'headline': 'count'
    }).reset_index()
    
    daily_sentiment.columns = ['date', 'avg_sentiment_compound', 
                               'avg_sentiment_positive', 'avg_sentiment_negative',
                               'news_count']
    
    return daily_sentiment


def fetch_historical_news_kaggle(ticker: str, start_date: str, end_date: str, 
                                  csv_path: str = None) -> pd.DataFrame:
    """
    Load historical news from Kaggle dataset and calculate sentiment.
    
    Loads from multiple CSV files in data/archive/:
    1. analyst_ratings_processed.csv - Most precise timestamps (minute-level, UTC-4)
    2. raw_analyst_ratings.csv - Benzinga analyst ratings (day-level precision)
    3. raw_partner_headlines.csv - Partner news headlines (day-level precision)
    
    The function combines all available datasets and deduplicates based on 
    headline and date to maximize coverage.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    csv_path : str, optional
        Path to specific CSV file. If None, loads all available datasets
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, headline, sentiment_compound, etc.
        
    Notes:
    ------
    - analyst_ratings_processed.csv has the most precise timestamps
    - raw_analyst_ratings.csv and raw_partner_headlines.csv only have day-level dates
    - All datasets cover primarily 2020 data (March-June 2020)
    """
    import os
    from pathlib import Path
    
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Try to find CSV files - we'll load from MULTIPLE sources and combine them
    if csv_path is None:
        # Search in data directory for all available news datasets
        data_dir = Path(__file__).parent.parent / 'data'
        archive_dir = data_dir / 'archive'
        
        # Dataset configurations with metadata
        # Priority order: Best timestamps first
        dataset_configs = [
            {
                'filename': 'analyst_ratings_processed.csv',
                'description': 'Processed analyst ratings (minute-level timestamps, UTC-4)',
                'headline_col': 'title',
                'precise_time': True
            },
            {
                'filename': 'raw_analyst_ratings.csv',
                'description': 'Raw analyst ratings from Benzinga (day-level only)',
                'headline_col': 'headline',
                'precise_time': False
            },
            {
                'filename': 'raw_partner_headlines.csv',
                'description': 'Partner news headlines (day-level only)',
                'headline_col': 'headline',
                'precise_time': False
            }
        ]
        
        dataset_files = [cfg['filename'] for cfg in dataset_configs]
        
        csv_paths = []
        
        # Check archive directory for all datasets
        for filename in dataset_files:
            test_path = archive_dir / filename
            if test_path.exists():
                csv_paths.append(str(test_path))
        
        # Also check main data directory
        if not csv_paths:
            for filename in dataset_files:
                test_path = data_dir / filename
                if test_path.exists():
                    csv_paths.append(str(test_path))
        
        if not csv_paths:
            print("[WARN]  News dataset not found in data/ or data/archive/ directory")
            print(" Looking for: analyst_ratings_processed.csv, raw_analyst_ratings.csv, or raw_partner_headlines.csv")
            print(" Expected location: /Users/tobycoleman/mining/StockPrediction/data/archive/")
            return pd.DataFrame()
        
        # If we have multiple files, we'll combine them
        csv_path = csv_paths
    
    try:
        # Handle both single file and multiple files
        if isinstance(csv_path, list):
            print(f" Loading {len(csv_path)} dataset file(s) and combining...")
            all_dfs = []
            dataset_info = []
            
            for path in csv_path:
                filename = Path(path).name
                print(f"   Loading: {filename}...", end=" ")
                file_df = pd.read_csv(path)
                print(f"{len(file_df):,} rows")
                all_dfs.append(file_df)
                dataset_info.append(f"{filename} ({len(file_df):,} rows)")
            
            # Combine all datasets
            df = pd.concat(all_dfs, ignore_index=True)
            print(f"[OK] Combined {len(csv_path)} datasets: {len(df):,} total rows")
        else:
            print(f" Loading dataset from: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f" Dataset shape: {df.shape}")
        
        print(f" Columns: {df.columns.tolist()}")
        
        # Identify ticker and date columns (dataset may have different column names)
        ticker_col = None
        date_col = None
        headline_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'ticker' in col_lower or 'symbol' in col_lower or 'stock' in col_lower:
                ticker_col = col
            elif 'date' in col_lower or 'time' in col_lower:
                date_col = col
            elif 'headline' in col_lower or 'title' in col_lower or 'news' in col_lower:
                headline_col = col
        
        if not all([ticker_col, date_col, headline_col]):
            print(f"[ERROR] Could not identify required columns")
            print(f"   Found: ticker={ticker_col}, date={date_col}, headline={headline_col}")
            return pd.DataFrame()
        
        # Create unified headline column
        # Some files use 'title', others use 'headline' - combine them
        if 'headline' in df.columns and 'title' in df.columns:
            # Prefer 'headline', fallback to 'title'
            df['unified_headline'] = df['headline'].fillna(df['title'])
            headline_col = 'unified_headline'
            print(f"[OK] Combined 'headline' and 'title' columns → '{headline_col}'")
        
        print(f"[OK] Using columns: ticker='{ticker_col}', date='{date_col}', headline='{headline_col}' from {len(df):,} total rows")
        
        # Filter by ticker
        df = df[df[ticker_col] == ticker].copy()
        print(f" Found {len(df)} news items for {ticker}")
        
        if df.empty:
            print(f"[WARN]  No news found for ticker {ticker}")
            return pd.DataFrame()
        
        # Parse dates - use utc=True to handle timezone-aware strings uniformly
        df['date'] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
        df = df.dropna(subset=['date'])
        
        # Remove timezone information to allow comparison
        # Convert all timestamps to timezone-naive (removes both UTC and other timezones)
        df['date'] = df['date'].dt.tz_localize(None)
        
        # Filter by date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        
        print(f" After date filter: {len(df)} items from {start_date} to {end_date}")
        
        if df.empty:
            print(f"[WARN]  No news in date range")
            return pd.DataFrame()
        
        # Calculate sentiment for each headline
        print(" Calculating VADER sentiment scores...")
        sentiments = []
        for headline in df[headline_col]:
            if pd.isna(headline):
                scores = {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}
            else:
                scores = sia.polarity_scores(str(headline))
            
            sentiments.append({
                'sentiment_compound': scores['compound'],
                'sentiment_positive': scores['pos'],
                'sentiment_negative': scores['neg'],
                'sentiment_neutral': scores['neu']
            })
        
        sentiment_df = pd.DataFrame(sentiments)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'date': df['date'].values,  # Keep as datetime objects
            'headline': df[headline_col].values,
            'sentiment_compound': sentiment_df['sentiment_compound'].values,
            'sentiment_positive': sentiment_df['sentiment_positive'].values,
            'sentiment_negative': sentiment_df['sentiment_negative'].values,
            'sentiment_neutral': sentiment_df['sentiment_neutral'].values
        })
        
        # Ensure date is datetime64 (not python date objects)
        result['date'] = pd.to_datetime(result['date'])
        
        # Drop rows with NaT dates
        result = result.dropna(subset=['date']).reset_index(drop=True)
        
        # Remove duplicates based on date and headline
        # (different CSV files may contain overlapping news items)
        initial_count = len(result)
        result = result.drop_duplicates(subset=['date', 'headline'], keep='first')
        duplicates_removed = initial_count - len(result)
        
        if duplicates_removed > 0:
            print(f" Removed {duplicates_removed} duplicate news items")
        
        print(f"[OK] Successfully processed {len(result)} unique news items")
        if len(result) > 0:
            print(f" Date range: {result['date'].min()} to {result['date'].max()}")
            print(f" Average sentiment: {result['sentiment_compound'].mean():.3f}")
        else:
            print("[WARN]  No valid data after cleaning")
        
        return result
        
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Error loading Kaggle dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def fetch_news_sentiment_hybrid(ticker: str, start_date: str, end_date: str, 
                                 api_key: str = None) -> pd.DataFrame:
    """
    Hybrid news sentiment fetcher:
    - Uses Kaggle dataset for historical data (> 30 days old)
    - Falls back to NewsAPI for recent data (< 30 days old)
    
    This provides the best of both worlds: 3+ years of historical coverage
    plus real-time news updates.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    api_key : str, optional
        NewsAPI key for recent news
    
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with historical + recent news sentiment
    """
    from datetime import datetime, timedelta
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    cutoff = datetime.now() - timedelta(days=30)
    
    all_data = []
    
    # Historical data from Kaggle (if date range includes historical period)
    if start < pd.to_datetime(cutoff):
        historical_end = min(end, pd.to_datetime(cutoff))
        print(f"\n Fetching historical news from Kaggle ({start_date} to {historical_end.date()})...")
        historical_df = fetch_historical_news_kaggle(ticker, start_date, str(historical_end.date()))
        
        if not historical_df.empty:
            all_data.append(historical_df)
            print(f"[OK] Got {len(historical_df)} historical news items")
    
    # Recent data from NewsAPI (if date range includes recent period)
    if end >= pd.to_datetime(cutoff):
        recent_start = max(start, pd.to_datetime(cutoff))
        days_back = (datetime.now() - recent_start).days + 1
        days_back = min(days_back, 30)  # NewsAPI limit
        
        print(f"\n🌐 Fetching recent news from NewsAPI (last {days_back} days)...")
        recent_df = fetch_news_sentiment(ticker, days_back=days_back, api_key=api_key)
        
        if not recent_df.empty:
            # Ensure date column is date type (not datetime)
            if 'date' in recent_df.columns:
                recent_df['date'] = pd.to_datetime(recent_df['date']).dt.date
            all_data.append(recent_df)
            print(f"[OK] Got {len(recent_df)} recent news items")
    
    # Combine data
    if not all_data:
        print("[WARN]  No news data found from either source")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates based on date and headline
    if 'headline' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['date', 'headline'], keep='first')
    
    # Sort by date
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    
    print(f"\n[OK] Total combined news items: {len(combined_df)}")
    print(f" Coverage: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    return combined_df


if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Fetch stock data
    print("\n=== Fetching Stock Data ===")
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    print(stock_data.head())
    
    # Fetch politician trades (placeholder)
    print("\n=== Fetching Politician Trades ===")
    politician_data = fetch_politician_trades(ticker)
    print(f"Politician trades shape: {politician_data.shape}")
    
    # Fetch news sentiment
    print("\n=== Fetching News Sentiment ===")
    news_sentiment = fetch_news_sentiment(ticker, days_back=7)
    print(news_sentiment.head())
    
    # Aggregate sentiment
    print("\n=== Aggregating Daily Sentiment ===")
    daily_sentiment = aggregate_daily_sentiment(news_sentiment)
    print(daily_sentiment.head())
