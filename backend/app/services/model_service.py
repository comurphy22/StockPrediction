"""
Model Service
=============
Handles model loading, caching, and training using existing ML code.
"""

import sys
import os
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

# Add the project src directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from app.config import settings


class ModelService:
    """Service for managing ML models."""
    
    def __init__(self):
        self.model_cache: Dict[str, Dict[str, Any]] = {}
        self.models_dir = settings.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Import ML modules from existing code
        try:
            from data_loader import fetch_stock_data, fetch_politician_trades
            from feature_engineering import create_features, handle_missing_values
            from model_xgboost import train_xgboost_model, get_xgboost_feature_importance
            
            self.fetch_stock_data = fetch_stock_data
            self.fetch_politician_trades = fetch_politician_trades
            self.create_features = create_features
            self.handle_missing_values = handle_missing_values
            self.train_xgboost_model = train_xgboost_model
            self.get_xgboost_feature_importance = get_xgboost_feature_importance
            
            # Try to load TOP_FEATURES from config
            try:
                from config import TOP_FEATURES
                self.top_features = TOP_FEATURES[:20] if TOP_FEATURES else None
            except ImportError:
                self.top_features = None
                
        except ImportError as e:
            print(f"Warning: Could not import ML modules: {e}")
            raise RuntimeError(f"Failed to load ML modules from src/: {e}")
    
    def get_or_train_model(self, ticker: str, lookback_days: int = 180) -> Dict[str, Any]:
        """
        Get a cached model or train a new one.
        
        Returns:
            Dict with 'model', 'features', and 'trained_at' keys
        """
        ticker = ticker.upper()
        model_path = self.models_dir / f"{ticker.lower()}_model.pkl"
        
        # Check memory cache first
        if ticker in self.model_cache:
            cache_entry = self.model_cache[ticker]
            # Check if cache is still valid (within TTL)
            cache_age = (datetime.now() - cache_entry['trained_at']).total_seconds()
            if cache_age < settings.model_cache_ttl:
                print(f"Using cached model for {ticker} (age: {cache_age:.0f}s)")
                return cache_entry
        
        # Check disk cache
        if model_path.exists():
            try:
                print(f"Loading model for {ticker} from disk...")
                model_data = joblib.load(model_path)
                model_data['cached'] = True
                self.model_cache[ticker] = model_data
                return model_data
            except Exception as e:
                print(f"Failed to load cached model: {e}")
        
        # Train new model
        print(f"Training new model for {ticker}...")
        model, features = self._train_model(ticker, lookback_days)
        
        model_data = {
            'model': model,
            'features': features,
            'trained_at': datetime.now(),
            'lookback_days': lookback_days,
            'cached': False
        }
        
        # Save to disk and memory cache
        try:
            joblib.dump(model_data, model_path)
            print(f"Saved model to {model_path}")
        except Exception as e:
            print(f"Warning: Could not save model to disk: {e}")
        
        self.model_cache[ticker] = model_data
        return model_data
    
    def _train_model(self, ticker: str, lookback_days: int) -> tuple:
        """Train XGBoost model using existing code."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        # Fetch data using existing data_loader
        print(f"  Fetching stock data for {ticker}...")
        stock_data = self.fetch_stock_data(ticker, start_date, end_date)
        
        print(f"  Fetching politician trades for {ticker}...")
        try:
            trades_data = self.fetch_politician_trades(ticker)
        except Exception as e:
            print(f"  Warning: Could not fetch politician trades: {e}")
            trades_data = pd.DataFrame()
        
        # Create features using existing feature_engineering
        print(f"  Creating features...")
        X, y, dates = self.create_features(stock_data, pd.DataFrame(), trades_data, ticker=ticker)
        
        # Select top features if available
        if self.top_features:
            available = [f for f in self.top_features if f in X.columns]
            if available:
                X = X[available]
                print(f"  Using {len(available)} selected features")
        
        # Handle missing values
        X_clean = self.handle_missing_values(X, strategy='drop')
        y_clean = y.loc[X_clean.index]
        
        # Train model using existing model_xgboost
        print(f"  Training XGBoost model...")
        model = self.train_xgboost_model(X_clean, y_clean, verbose=False)
        
        return model, X.columns.tolist()
    
    def get_feature_importance(self, ticker: str) -> List[Dict[str, Any]]:
        """Get feature importance for a specific ticker's model."""
        model_data = self.get_or_train_model(ticker)
        model = model_data['model']
        features = model_data['features']
        
        importance_df = self.get_xgboost_feature_importance(model, features)
        
        # Categorize features
        def categorize_feature(name: str) -> str:
            name_lower = name.lower()
            if any(x in name_lower for x in ['politician', 'trade', 'congress']):
                return 'politician'
            elif any(x in name_lower for x in ['sentiment', 'news']):
                return 'sentiment'
            elif any(x in name_lower for x in ['spy', 'qqq', 'vix', 'sector', 'market', 'beta', 'corr']):
                return 'market'
            else:
                return 'technical'
        
        return [
            {
                'feature': row['feature'],
                'importance': float(row['importance']),
                'category': categorize_feature(row['feature'])
            }
            for _, row in importance_df.iterrows()
        ]
    
    def get_global_feature_importance(self) -> List[Dict[str, Any]]:
        """Get aggregated feature importance across all cached models."""
        if not self.model_cache:
            return []
        
        all_importance = []
        for ticker, model_data in self.model_cache.items():
            importance = self.get_feature_importance(ticker)
            all_importance.extend(importance)
        
        # Aggregate by feature name
        df = pd.DataFrame(all_importance)
        if df.empty:
            return []
        
        aggregated = df.groupby(['feature', 'category'])['importance'].mean().reset_index()
        aggregated = aggregated.sort_values('importance', ascending=False)
        
        return aggregated.to_dict('records')
    
    def get_model_info(self, ticker: str) -> Dict[str, Any]:
        """Get information about a model."""
        model_data = self.get_or_train_model(ticker)
        
        return {
            'ticker': ticker,
            'trained_at': model_data['trained_at'].isoformat(),
            'lookback_days': model_data.get('lookback_days', 180),
            'num_features': len(model_data['features']),
            'top_features': self.get_feature_importance(ticker)[:10],
            'cached': model_data.get('cached', False)
        }
    
    def clear_cache(self, ticker: Optional[str] = None):
        """Clear model cache."""
        if ticker:
            ticker = ticker.upper()
            if ticker in self.model_cache:
                del self.model_cache[ticker]
            model_path = self.models_dir / f"{ticker.lower()}_model.pkl"
            if model_path.exists():
                model_path.unlink()
        else:
            self.model_cache.clear()
            for path in self.models_dir.glob("*.pkl"):
                path.unlink()


# Singleton instance
model_service = ModelService()

