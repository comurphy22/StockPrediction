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
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        print(f"Successfully fetched {len(df)} days of data for {ticker}")
        return df
    
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {str(e)}")
        raise


def fetch_politician_trades(ticker: str, api_key: str = None) -> pd.DataFrame:
    """
    Fetch politician trading data from Finnhub or Quiver Quantitative API.
    
    NOTE: This is a placeholder implementation. You'll need to:
    1. Sign up for an API key at Finnhub (https://finnhub.io/) or 
       Quiver Quantitative (https://www.quiverquant.com/)
    2. Replace the placeholder URL and parameters with actual API endpoints
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    api_key : str, optional
        API key for the service
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, politician_name, transaction_type, amount, etc.
    """
    # Placeholder implementation
    if api_key is None:
        print("Warning: No API key provided for politician trades.")
        print("Returning empty DataFrame. Please provide an API key to fetch real data.")
        return pd.DataFrame(columns=['date', 'politician_name', 'transaction_type', 
                                     'amount', 'ticker'])
    
    try:
        # Example Finnhub API endpoint (you'll need to verify the actual endpoint)
        # url = f"https://finnhub.io/api/v1/stock/congress-trading"
        # params = {
        #     'symbol': ticker,
        #     'token': api_key
        # }
        
        # Placeholder URL - replace with actual API endpoint
        url = "https://api.example.com/politician-trades"
        params = {
            'ticker': ticker,
            'api_key': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Transform API response to DataFrame
        # This will depend on the actual API response structure
        df = pd.DataFrame(data)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        print(f"Successfully fetched {len(df)} politician trades for {ticker}")
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching politician trades: {str(e)}")
        print("Returning empty DataFrame")
        return pd.DataFrame(columns=['date', 'politician_name', 'transaction_type', 
                                     'amount', 'ticker'])


def fetch_news_sentiment(ticker: str, days_back: int = 30) -> pd.DataFrame:
    """
    Fetch news and calculate sentiment scores using VADER.
    
    NOTE: This is a simplified implementation using placeholder news data.
    For production, you would integrate with:
    - News API (https://newsapi.org/)
    - Alpha Vantage News Sentiment API
    - Finnhub News API
    - Or scrape financial news sites (with proper permissions)
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    days_back : int
        Number of days of news to fetch (default: 30)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, headline, sentiment_score
    """
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Placeholder: In production, you would fetch actual news here
    # Example using News API:
    # url = "https://newsapi.org/v2/everything"
    # params = {
    #     'q': ticker,
    #     'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
    #     'sortBy': 'publishedAt',
    #     'apiKey': 'YOUR_NEWS_API_KEY'
    # }
    
    print(f"Warning: Using placeholder news data for {ticker}")
    print("For production, integrate with a real news API")
    
    # Placeholder news headlines (replace with actual API calls)
    placeholder_news = [
        {"date": datetime.now() - timedelta(days=i), 
         "headline": f"Sample news headline about {ticker} on day {i}"}
        for i in range(min(days_back, 10))
    ]
    
    # Calculate sentiment scores for each headline
    sentiment_data = []
    for news_item in placeholder_news:
        headline = news_item['headline']
        
        # Get sentiment scores using VADER
        sentiment_scores = sia.polarity_scores(headline)
        
        sentiment_data.append({
            'date': news_item['date'],
            'headline': headline,
            'sentiment_compound': sentiment_scores['compound'],  # -1 to 1
            'sentiment_positive': sentiment_scores['pos'],
            'sentiment_negative': sentiment_scores['neg'],
            'sentiment_neutral': sentiment_scores['neu']
        })
    
    df = pd.DataFrame(sentiment_data)
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    print(f"Calculated sentiment for {len(df)} news items")
    return df


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
