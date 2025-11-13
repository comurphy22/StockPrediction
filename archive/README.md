# Archive Directory

This directory contains experimental code and results from the development process. These files are preserved for reference but are not part of the current production codebase.

## Directory Structure

### `/scripts`
- **validate_mvp_v2.py** - V2 validation with VADER sentiment
- **validate_mvp_v3.py** - V3 validation with enhanced keyword matching
- **validate_mvp_v4.py** - V4 validation with FinancialPhraseBank classifier (incomplete)
- **validate_hypothesis_multiyear.py** - Multi-year validation experiments
- **fix_overfitting.py** - Initial overfitting fix attempts
- **fix_overfitting_v2.py** - Second iteration of overfitting fixes
- **quick_overfitting_test.py** - Quick test script for overfitting experiments

### `/src`
- **data_loader_enhanced.py** - Enhanced data loader with keyword matching
- **data_loader_optimized.py** - Optimized data loader with caching
- **data_loader_financial_sentiment.py** - Data loader with FinancialPhraseBank sentiment

### `/logs`
- Various validation and experiment logs from V2-V4 iterations

### `/results`
- CSV results from V2-V4 validations and confusion matrices

## Key Learnings from Archived Experiments

1. **V3 (Enhanced News Coverage)**: More news didn't help - coverage went from 0→3154 articles for AAPL but accuracy DECREASED by 0.67%
2. **V4 (Better Sentiment Model)**: 73.5% accurate FinancialPhraseBank classifier showed minimal improvement (+0.45% vs V3)
3. **Root Cause**: Overfitting was the primary bottleneck (~44% gap), not sentiment quality

These experiments led to the successful V5 implementation with L1 regularization (alpha=1.0).
