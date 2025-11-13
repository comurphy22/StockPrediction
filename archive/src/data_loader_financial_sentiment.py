"""
Data Loader with Financial Sentiment Classifier
Uses trained FinancialPhraseBank model instead of VADER for better sentiment scores
"""

import pandas as pd
import pickle
import re
from typing import Tuple

# Load trained sentiment model (global for performance)
_SENTIMENT_MODEL = None
_SENTIMENT_VECTORIZER = None

def _load_sentiment_model():
    """Load trained financial sentiment classifier (once)"""
    global _SENTIMENT_MODEL, _SENTIMENT_VECTORIZER
    
    if _SENTIMENT_MODEL is None:
        with open('models/financial_sentiment_classifier.pkl', 'rb') as f:
            _SENTIMENT_MODEL = pickle.load(f)
        with open('models/financial_sentiment_vectorizer.pkl', 'rb') as f:
            _SENTIMENT_VECTORIZER = pickle.load(f)
    
    return _SENTIMENT_MODEL, _SENTIMENT_VECTORIZER


def preprocess_text(text: str) -> str:
    """Preprocess text for sentiment analysis (same as training)"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s%.]', ' ', text)
    text = ' '.join(text.split())
    return text


def score_financial_sentiment(headlines: list) -> pd.DataFrame:
    """
    Score headlines using trained financial sentiment classifier
    
    Args:
        headlines: List of headline strings
        
    Returns:
        DataFrame with columns: headline, sentiment_score (-1, 0, 1), 
                                prob_negative, prob_neutral, prob_positive, compound
    """
    classifier, vectorizer = _load_sentiment_model()
    
    # Preprocess
    headlines_clean = [preprocess_text(h) for h in headlines]
    
    # Transform to TF-IDF
    headlines_tfidf = vectorizer.transform(headlines_clean)
    
    # Predict sentiment scores (-1, 0, 1)
    sentiment_scores = classifier.predict(headlines_tfidf)
    
    # Get probabilities
    probs = classifier.predict_proba(headlines_tfidf)
    
    # Create DataFrame
    results = pd.DataFrame({
        'headline': headlines,
        'sentiment_score': sentiment_scores,  # -1 (negative), 0 (neutral), 1 (positive)
        'prob_negative': probs[:, 0],         # Probability of negative
        'prob_neutral': probs[:, 1],          # Probability of neutral
        'prob_positive': probs[:, 2],         # Probability of positive
    })
    
    # Create compound score for compatibility with existing code
    # Map: -1 → -0.5, 0 → 0.0, 1 → +0.5 (scaled by confidence)
    # Use (prob_positive - prob_negative) as compound score
    results['compound'] = results['prob_positive'] - results['prob_negative']
    
    # Alternative: use sentiment_score directly scaled
    # results['compound'] = results['sentiment_score'] * 0.5
    
    return results


def fetch_news_with_financial_sentiment(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch news and score with financial sentiment classifier
    Replaces VADER sentiment with trained financial model
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with columns: date, headline, compound (and other sentiment scores)
    """
    # Import the optimized loader to get news articles WITHOUT VADER scoring
    # This returns the raw news with date, headline, but VADER scores
    from data_loader_optimized import fetch_news_with_keywords
    
    # Fetch news articles (this loads dataset once and returns filtered articles with VADER scores)
    news_df = fetch_news_with_keywords(ticker, start_date, end_date)
    
    if len(news_df) == 0:
        return pd.DataFrame(columns=['date', 'headline', 'compound', 'positive', 'negative', 'neutral'])
    
    # Score sentiment using financial classifier (replaces VADER)
    print(f"   🤖 Scoring {len(news_df)} headlines with Financial Classifier...", end=" ", flush=True)
    sentiment_results = score_financial_sentiment(news_df['headline'].tolist())
    print("✅")
    
    # Merge back with dates
    news_df = news_df.reset_index(drop=True)
    sentiment_results = sentiment_results.reset_index(drop=True)
    
    result = pd.DataFrame({
        'date': news_df['date'],
        'headline': news_df['headline'],
        'compound': sentiment_results['compound'],
        'positive': sentiment_results['prob_positive'],
        'negative': sentiment_results['prob_negative'],
        'neutral': sentiment_results['prob_neutral'],
        'sentiment_score': sentiment_results['sentiment_score']  # -1, 0, 1
    })
    
    return result


def aggregate_daily_sentiment(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores by date
    Same interface as original data_loader
    
    Args:
        sentiment_df: DataFrame with date, compound, positive, negative, neutral columns
        
    Returns:
        DataFrame with daily aggregated sentiment scores
    """
    if len(sentiment_df) == 0:
        return pd.DataFrame(columns=['date', 'avg_sentiment_compound', 'avg_sentiment_positive', 
                                     'avg_sentiment_negative', 'news_count'])
    
    # Ensure date column is datetime and normalize to date only (remove time)
    sentiment_df = sentiment_df.copy()
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.normalize()
    
    # Group by date and aggregate
    daily = sentiment_df.groupby('date').agg({
        'compound': 'mean',
        'positive': 'mean',
        'negative': 'mean',
        'headline': 'count'
    }).reset_index()
    
    daily.columns = ['date', 'avg_sentiment_compound', 'avg_sentiment_positive', 
                     'avg_sentiment_negative', 'news_count']
    
    return daily


# Test function
if __name__ == "__main__":
    print("="*70)
    print("TESTING FINANCIAL SENTIMENT CLASSIFIER ON NEWS HEADLINES")
    print("="*70)
    print()
    
    # Test on sample headlines
    test_headlines = [
        "Apple stock surges to record high on strong earnings",
        "Microsoft faces setback as sales decline sharply",
        "Amazon reports quarterly results",
        "Tesla stock plummets amid production concerns",
        "Google announces new AI breakthrough",
        "Netflix subscriber growth exceeds expectations",
        "Nvidia revenue soars on AI chip demand",
        "Boeing delays aircraft deliveries again"
    ]
    
    print("Scoring sample headlines:")
    print()
    
    results = score_financial_sentiment(test_headlines)
    
    for idx, row in results.iterrows():
        score = row['sentiment_score']
        compound = row['compound']
        emoji = "📈" if score > 0 else "📉" if score < 0 else "➡️"
        
        print(f"{emoji} [{score:+d}] Compound: {compound:+.3f}")
        print(f"   {row['headline']}")
        print(f"   Probs: Neg={row['prob_negative']:.2f}, Neu={row['prob_neutral']:.2f}, Pos={row['prob_positive']:.2f}")
        print()
    
    print("="*70)
    print("COMPARISON: Financial Classifier vs VADER")
    print("="*70)
    print()
    
    # Compare with VADER
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    
    print(f"{'Headline':<50} {'Financial':<12} {'VADER':<12}")
    print("-"*70)
    
    for headline in test_headlines:
        # Financial classifier
        fin_score = score_financial_sentiment([headline])['compound'].values[0]
        
        # VADER
        vader_score = sia.polarity_scores(headline)['compound']
        
        print(f"{headline[:47]:<50} {fin_score:+.3f}       {vader_score:+.3f}")
    
    print()
    print("✅ Financial sentiment classifier ready to use!")
