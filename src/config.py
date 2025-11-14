"""
Configuration file for Stock Prediction project.
Loads API keys from environment variables (.env file).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# ============================================================================
# API KEYS
# ============================================================================

# Politician Trading Data APIs
QUIVER_API_KEY = os.getenv('QUIVER_API_KEY', '')
QUIVER_BASE_URL = "https://api.quiverquant.com/beta"

FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# News APIs (optional - only for real-time news)
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

# Aliases for backward compatibility with data_loader.py
NEWSAPI_KEY = NEWS_API_KEY
NEWSAPI_BASE_URL = NEWS_API_BASE_URL
DEFAULT_NEWS_PAGE_SIZE = 100

# ============================================================================
# DATA PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'

# News datasets
NEWS_ARCHIVE_DIR = DATA_DIR / 'archive'
ANALYST_RATINGS_PROCESSED = NEWS_ARCHIVE_DIR / 'analyst_ratings_processed.csv'
RAW_ANALYST_RATINGS = NEWS_ARCHIVE_DIR / 'raw_analyst_ratings.csv'
RAW_PARTNER_HEADLINES = NEWS_ARCHIVE_DIR / 'raw_partner_headlines.csv'

# Sentiment model
SENTIMENT_MODEL_DIR = DATA_DIR / 'SentimentAnalysis'
FINANCIAL_SENTIMENT_MODEL = MODELS_DIR / 'financial_sentiment_classifier.pkl'
FINANCIAL_SENTIMENT_VECTORIZER = MODELS_DIR / 'financial_sentiment_vectorizer.pkl'

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# XGBoost default parameters (with STRONG regularization to combat overfitting)
# Previous attempt: Still 45-52% overfitting gap - need much stronger constraints
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'max_depth': 2,              # Further reduced from 3 (simpler trees)
    'learning_rate': 0.03,       # Further reduced from 0.05 (more careful learning)
    'n_estimators': 150,         # Increased to compensate for slower learning
    'min_child_weight': 10,      # Doubled from 5 (stronger constraint)
    'gamma': 0.5,                # Increased from 0.1 (much higher split threshold)
    'subsample': 0.7,            # Reduced from 0.8 (more aggressive sampling)
    'colsample_bytree': 0.7,     # Reduced from 0.8 (use fewer features)
    'colsample_bylevel': 0.7,    # NEW: Additional feature sampling per level
    'reg_alpha': 0.5,            # Increased L1 from 0.1 (more feature sparsity)
    'reg_lambda': 3.0,           # Increased L2 from 1.0 (stronger weight penalty)
    'scale_pos_weight': 1.0,     # Balance classes
    'random_state': 42,
    'n_jobs': -1
}

# LSTM/GRU parameters
SEQUENCE_LENGTH = 10  # Number of days to look back
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 50
LEARNING_RATE = 0.001

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

# Years for multi-year validation
VALIDATION_YEARS = [2018, 2019, 2020]

# Stocks for validation (based on data coverage analysis 2018-2020)
# Selected using analyze_stock_coverage.py for optimal news + politician trade coverage
VALIDATION_STOCKS = [
    'NFLX',   # 1980 news, 122 trades (combined score: 3200)
    'GOOGL',  # 1595 news, 124 trades (combined score: 2835)
    'NVDA',   # 1353 news, 127 trades (combined score: 2623)
    'TSLA',   # 1940 news,  67 trades (combined score: 2610)
    'PFE',    #  938 news, 144 trades (combined score: 2378)
    'WFC',    #  823 news, 153 trades (combined score: 2353)
    'FDX',    #  917 news, 124 trades (combined score: 2157)
    'BABA'    # 1135 news,  97 trades (combined score: 2105)
]

# Train/test split
TRAIN_TEST_SPLIT = 0.8

# Top features (from feature importance analysis)
TOP_FEATURES = [
    'avg_sentiment_compound', 'HL_spread', 'Price_change', 'SMA_50', 'Volume_change',
    'MACD_diff', 'SMA_10_20_cross', 'SMA_20', 'RSI', 'MACD', 'SMA_10', 'SMA_20_50_cross',
    'avg_sentiment_positive', 'MACD_signal', 'days_since_last_trade', 'avg_sentiment_negative',
    'amount_last_60d', 'net_flow_last_60d', 'net_flow_last_90d', 'news_count',
    'trades_last_60d', 'dollar_momentum_30d', 'trade_momentum_30d', 'net_flow_last_30d',
    'amount_last_30d'
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_keys():
    """Check if required API keys are configured."""
    issues = []
    
    if not QUIVER_API_KEY and not FINNHUB_API_KEY:
        issues.append("No politician trading API key found. Set QUIVER_API_KEY or FINNHUB_API_KEY in .env")
    
    if issues:
        print("\n⚠️  API KEY CONFIGURATION ISSUES:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n   Create a .env file in the project root with your API keys.")
        print("   See .env.template for the format.\n")
        return False
    
    print("✅ API keys configured successfully")
    return True

def get_politician_api():
    """Determine which politician trading API to use."""
    if QUIVER_API_KEY:
        return 'quiver'
    elif FINNHUB_API_KEY:
        return 'finnhub'
    else:
        return None

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

import logging

def setup_logging(log_file=None, level=logging.INFO):
    """Set up logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        log_path = LOGS_DIR / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=level, format=log_format)
    
    return logging.getLogger(__name__)
