"""
Optimized data loader with pre-indexed keyword matching for better performance.
Includes disk caching to avoid slow date parsing on every load.
"""

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os
import pickle

# Download VADER lexicon if needed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Disk cache location
CACHE_FILE = 'data/cached_indexed_articles.pkl'

# Company name to ticker mapping
COMPANY_KEYWORDS = {
    'AAPL': ['apple', 'iphone', 'ipad', 'mac', 'ios'],
    'MSFT': ['microsoft', 'windows', 'azure', 'office'],
    'AMZN': ['amazon', 'aws', 'alexa', 'bezos'],
    'GOOGL': ['google', 'alphabet', 'youtube', 'android'],
    'TSLA': ['tesla', 'elon musk', 'model 3', 'model s'],
    'NFLX': ['netflix'],
    'NVDA': ['nvidia'],
    'BABA': ['alibaba', 'baba'],
    'QCOM': ['qualcomm'],
    'MU': ['micron']
}

# Cache for loaded datasets
_DATASET_CACHE = None

def _load_and_index_datasets():
    """
    Load all datasets once and create an indexed structure for fast lookups.
    Uses disk cache to avoid slow date parsing on subsequent loads.
    """
    global _DATASET_CACHE
    
    # Check memory cache first
    if _DATASET_CACHE is not None:
        return _DATASET_CACHE
    
    # Check disk cache
    if os.path.exists(CACHE_FILE):
        print("📂 Loading from cache (fast)...")
        with open(CACHE_FILE, 'rb') as f:
            _DATASET_CACHE = pickle.load(f)
        print(f"✅ Loaded {len(_DATASET_CACHE)} cached articles")
        return _DATASET_CACHE
    
    # First-time load: read CSVs and create cache
    print("📂 Loading and indexing datasets (one-time operation, will cache)...")
    
    datasets = [
        'data/archive/analyst_ratings_processed.csv',
        'data/archive/raw_analyst_ratings.csv',
        'data/archive/raw_partner_headlines.csv'
    ]
    
    # Load all data
    all_data = []
    for path in datasets:
        df = pd.read_csv(path)
        df['source_file'] = path.split('/')[-1]
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce', utc=True).dt.tz_localize(None)
    
    # Create unified headline column
    if 'title' in combined.columns and 'headline' in combined.columns:
        combined['unified_headline'] = combined['title'].fillna('') + ' ' + combined['headline'].fillna('')
    elif 'title' in combined.columns:
        combined['unified_headline'] = combined['title'].fillna('')
    elif 'headline' in combined.columns:
        combined['unified_headline'] = combined['headline'].fillna('')
    else:
        combined['unified_headline'] = ''
    
    # Clean up
    combined['unified_headline'] = combined['unified_headline'].str.strip()
    combined = combined[combined['unified_headline'] != '']
    combined['headline_lower'] = combined['unified_headline'].str.lower()
    
    print(f"✅ Indexed {len(combined)} articles from {len(datasets)} files")
    
    # Save to disk cache for next time
    print("💾 Saving to cache...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(combined, f)
    print(f"✅ Cache saved to {CACHE_FILE}")
    
    _DATASET_CACHE = combined
    return combined


def fetch_news_with_keywords(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch news for a ticker using both direct matching and keyword matching.
    Much faster than the enhanced loader because data is loaded once.
    """
    combined = _load_and_index_datasets()
    
    # Filter by date range first (fast)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    date_filtered = combined[(combined['date'] >= start) & (combined['date'] <= end)].copy()
    
    if len(date_filtered) == 0:
        print(f"⚠️  No articles in date range {start_date} to {end_date}")
        return pd.DataFrame()
    
    # Method 1: Direct ticker match
    direct_matches = date_filtered[date_filtered['stock'] == ticker].copy()
    
    # Method 2: Keyword matching
    keyword_matches = pd.DataFrame()
    if ticker in COMPANY_KEYWORDS:
        keywords = COMPANY_KEYWORDS[ticker]
        # Create a regex pattern for all keywords
        pattern = '|'.join(keywords)
        keyword_matches = date_filtered[
            date_filtered['headline_lower'].str.contains(pattern, case=False, na=False, regex=True)
        ].copy()
    
    # Combine and deduplicate
    all_matches = pd.concat([direct_matches, keyword_matches], ignore_index=True)
    all_matches = all_matches.drop_duplicates(subset=['date', 'unified_headline'])
    
    print(f"   📊 Found {len(all_matches)} articles ({len(direct_matches)} direct + {len(keyword_matches)} keyword)")
    
    if len(all_matches) == 0:
        return pd.DataFrame()
    
    # Calculate sentiment using VADER
    print(f"   🤖 Calculating sentiment...")
    sia = SentimentIntensityAnalyzer()
    
    sentiments = []
    for _, row in all_matches.iterrows():
        headline = row['unified_headline']
        scores = sia.polarity_scores(str(headline))
        sentiments.append({
            'date': row['date'],
            'headline': headline,
            'sentiment_compound': scores['compound'],
            'sentiment_positive': scores['pos'],
            'sentiment_negative': scores['neg'],
            'sentiment_neutral': scores['neu']
        })
    
    result = pd.DataFrame(sentiments)
    print(f"   ✅ Processed {len(result)} articles with sentiment")
    
    return result


def aggregate_daily_sentiment(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores to daily level.
    """
    if sentiment_df.empty:
        return pd.DataFrame(columns=['date', 'avg_sentiment_compound', 
                                    'avg_sentiment_positive', 'avg_sentiment_negative', 'news_count'])
    
    sentiment_df = sentiment_df.copy()
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.normalize()
    
    daily = sentiment_df.groupby('date').agg({
        'sentiment_compound': 'mean',
        'sentiment_positive': 'mean',
        'sentiment_negative': 'mean',
        'headline': 'count'
    }).reset_index()
    
    daily.columns = ['date', 'avg_sentiment_compound', 'avg_sentiment_positive',
                     'avg_sentiment_negative', 'news_count']
    
    return daily


if __name__ == "__main__":
    # Test the optimized loader
    print("Testing optimized news loader...")
    print("="*70)
    
    test_stocks = ['MSFT', 'AAPL', 'AMZN']
    
    for ticker in test_stocks:
        print(f"\nTesting {ticker}:")
        news = fetch_news_with_keywords(ticker, '2019-01-01', '2019-12-31')
        print(f"Total articles: {len(news)}")
        if len(news) > 0:
            sentiment = aggregate_daily_sentiment(news)
            print(f"Daily aggregated: {len(sentiment)} days")
            print(f"Average sentiment: {sentiment['avg_sentiment_compound'].mean():.3f}")
