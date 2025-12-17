"""
Backend Configuration
=====================
Environment variables and settings for the API.
"""

import os
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    """Application settings loaded from environment."""
    
    # API Keys
    quiver_api_key: str = field(default_factory=lambda: os.getenv("QUIVER_API_KEY", ""))
    news_api_key: str = field(default_factory=lambda: os.getenv("NEWS_API_KEY", ""))
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    
    @property
    def models_dir(self) -> Path:
        return self.project_root / "models" / "trained"
    
    @property
    def src_dir(self) -> Path:
        return self.project_root / "src"
    
    # Model settings
    default_lookback_days: int = 180
    model_cache_ttl: int = 3600  # 1 hour
    
    # Stock tiers by accuracy
    tier1_stocks: List[str] = field(default_factory=lambda: ['WFC', 'PFE', 'BABA'])      # 60-70% accuracy
    tier2_stocks: List[str] = field(default_factory=lambda: ['NFLX', 'GOOGL', 'FDX'])    # 50-58% accuracy
    tier3_stocks: List[str] = field(default_factory=lambda: ['NVDA', 'TSLA'])            # 38-43% accuracy
    
    # Confidence thresholds
    strong_signal_threshold: float = 0.70
    signal_threshold: float = 0.60


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

