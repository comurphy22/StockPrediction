"""
Enhanced data loading functions that use keyword matching for better coverage.

This module provides improved news sentiment loading that searches for company
names in article titles, not just relying on stock ticker tags.
"""

import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if needed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


# Company name to ticker mapping
COMPANY_KEYWORDS = {
    'AAPL': ['Apple', 'iPhone', 'iPad', 'Mac', 'iOS'],
    'MSFT': ['Microsoft', 'Windows', 'Azure', 'Office 365'],
    'AMZN': ['Amazon', 'AWS', 'Alexa', 'Prime'],
    'GOOGL': ['Google', 'Alphabet', 'Android', 'YouTube'],
    'TSLA': ['Tesla', 'Elon Musk'],
    'NFLX': ['Netflix'],
    'FB': ['Facebook', 'Meta'],
    'NVDA': ['Nvidia', 'NVDA'],
    'BABA': ['Alibaba', 'Jack Ma'],
    'QCOM': ['Qualcomm'],
    'MU': ['Micron Technology', 'Micron']
}


def fetch_historical_news_enhanced(ticker: str, start_date: str, end_date: str,
                                   data_dir: str = 'data/archive') -> pd.DataFrame:
    """
    Fetch historical news with enhanced keyword matching.
    
    Uses both:
    1. Direct stock ticker matching
    2. Company name/keyword matching in titles
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    data_dir : str
        Directory containing the CSV files
    
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with news articles and sentiment scores
    """
    
    print(f"📰 Enhanced news loading for {ticker}...")
    
    # Load all datasets
    datasets = [
        f'{data_dir}/analyst_ratings_processed.csv',
        f'{data_dir}/raw_analyst_ratings.csv',
        f'{data_dir}/raw_partner_headlines.csv'
    ]
    
    print(f"   Loading {len(datasets)} dataset files...")
    dfs = []
    for dataset_path in datasets:
        try:
            df = pd.read_csv(dataset_path)
            dfs.append(df)
            print(f"   ✅ Loaded: {dataset_path.split('/')[-1]} ({len(df)} rows)")
        except Exception as e:
            print(f"   ⚠️  Could not load {dataset_path}: {e}")
    
    if not dfs:
        print("   ❌ No datasets loaded!")
        return pd.DataFrame()
    
    # Combine datasets
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"   ✅ Combined {len(dfs)} datasets: {len(combined_df)} total rows")
    
    # Standardize columns
    if 'headline' in combined_df.columns and 'title' in combined_df.columns:
        combined_df['unified_headline'] = combined_df['headline'].fillna(combined_df['title'])
    elif 'title' in combined_df.columns:
        combined_df['unified_headline'] = combined_df['title']
    elif 'headline' in combined_df.columns:
        combined_df['unified_headline'] = combined_df['headline']
    else:
        print("   ❌ No headline or title column found!")
        return pd.DataFrame()
    
    # Parse dates
    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce', utc=True).dt.tz_localize(None)
    
    # Method 1: Direct ticker matching
    direct_match = combined_df[combined_df['stock'] == ticker].copy()
    print(f"   📊 Direct ticker match: {len(direct_match)} articles")
    
    # Method 2: Keyword matching
    keywords = COMPANY_KEYWORDS.get(ticker, [ticker])
    print(f"   🔍 Searching for keywords: {keywords}")
    
    # Create regex pattern for case-insensitive matching
    pattern = '|'.join([re.escape(kw) for kw in keywords])
    keyword_match = combined_df[
        combined_df['unified_headline'].str.contains(pattern, case=False, na=False, regex=True)
    ].copy()
    
    print(f"   📊 Keyword match: {len(keyword_match)} articles")
    
    # Combine both methods and remove duplicates
    all_matches = pd.concat([direct_match, keyword_match]).drop_duplicates(
        subset=['unified_headline', 'date']
    )
    
    print(f"   📊 Combined (deduplicated): {len(all_matches)} articles")
    
    # Filter by date range
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    filtered = all_matches[
        (all_matches['date'] >= start_ts) & 
        (all_matches['date'] <= end_ts)
    ].copy()
    
    print(f"   📅 After date filter ({start_date} to {end_date}): {len(filtered)} articles")
    
    if len(filtered) == 0:
        print(f"   ⚠️  No articles found for {ticker} in date range")
        return pd.DataFrame(columns=['date', 'unified_headline'])
    
    # Calculate sentiment scores using VADER
    print(f"   🤖 Calculating VADER sentiment scores...")
    sia = SentimentIntensityAnalyzer()
    
    sentiments = []
    for headline in filtered['unified_headline']:
        if pd.notna(headline):
            scores = sia.polarity_scores(str(headline))
            sentiments.append(scores)
        else:
            sentiments.append({'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0})
    
    sentiment_df = pd.DataFrame(sentiments)
    filtered = pd.concat([filtered.reset_index(drop=True), sentiment_df], axis=1)
    
    print(f"   ✅ Processed {len(filtered)} articles")
    print(f"   📊 Date range: {filtered['date'].min()} to {filtered['date'].max()}")
    print(f"   💭 Average sentiment: {filtered['compound'].mean():.3f}")
    
    return filtered[['date', 'unified_headline', 'compound', 'neg', 'neu', 'pos']]


def aggregate_daily_sentiment_enhanced(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores by day.
    
    Same as the original but works with enhanced data format.
    """
    if sentiment_df.empty:
        return pd.DataFrame(columns=[
            'date', 'avg_sentiment_compound', 'avg_sentiment_positive',
            'avg_sentiment_negative', 'news_count'
        ])
    
    # Group by date and calculate daily aggregates
    daily_sentiment = sentiment_df.groupby(sentiment_df['date'].dt.date).agg({
        'compound': 'mean',
        'pos': 'mean',
        'neg': 'mean',
        'unified_headline': 'count'
    }).reset_index()
    
    # Rename columns
    daily_sentiment.columns = [
        'date', 'avg_sentiment_compound', 'avg_sentiment_positive',
        'avg_sentiment_negative', 'news_count'
    ]
    
    # Convert date back to datetime
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    return daily_sentiment


if __name__ == "__main__":
    # Test the enhanced loader
    print("Testing Enhanced Data Loader")
    print("="*70)
    
    # Test on MSFT (which has 0 direct matches in 2019)
    news = fetch_historical_news_enhanced('MSFT', '2019-01-01', '2019-12-31')
    print(f"\nMSFT 2019: {len(news)} articles found")
    
    if len(news) > 0:
        daily = aggregate_daily_sentiment_enhanced(news)
        print(f"Daily aggregates: {len(daily)} days")
        print(f"\nSample headlines:")
        print(news['unified_headline'].head(5))
