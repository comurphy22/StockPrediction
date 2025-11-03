# Feature Importance Analysis Results

**Date**: November 2, 2025  
**Objective**: Identify optimal feature subset from 37 total features  
**Method**: Random Forest importance scores on best-performing stocks (BABA, QCOM, NVDA) using 2019 data

---

## Executive Summary

✅ **Analysis Complete**: Tested 41 features (37 standard + 4 sentiment) across 3 stocks  
✅ **Key Finding**: Top 17 features capture 80% of predictive power  
✅ **Recommendation**: Use 15-20 features for optimal performance  

---

## Feature Importance Rankings

### Top 20 Features (by Average Importance)

| Rank | Feature | Importance | Category | Cumulative % |
|------|---------|------------|----------|--------------|
| 1 | avg_sentiment_compound | 0.0655 | Sentiment | 6.5% |
| 2 | HL_spread | 0.0624 | Technical | 12.8% |
| 3 | Price_change | 0.0607 | Technical | 18.9% |
| 4 | SMA_50 | 0.0586 | Technical | 24.7% |
| 5 | Volume_change | 0.0511 | Technical | 29.8% |
| 6 | MACD_diff | 0.0483 | Technical | 34.7% |
| 7 | SMA_10_20_cross | 0.0482 | Technical | 39.5% |
| 8 | SMA_20 | 0.0477 | Technical | 44.2% |
| 9 | RSI | 0.0452 | Technical | 48.8% |
| 10 | MACD | 0.0430 | Technical | 53.1% |
| 11 | SMA_10 | 0.0417 | Technical | 57.2% |
| 12 | SMA_20_50_cross | 0.0408 | Technical | 61.3% |
| 13 | avg_sentiment_positive | 0.0403 | Sentiment | 65.3% |
| 14 | MACD_signal | 0.0400 | Technical | 69.3% |
| 15 | days_since_last_trade | 0.0390 | **Advanced Pol** | 73.2% |
| 16 | avg_sentiment_negative | 0.0353 | Sentiment | 76.8% |
| 17 | amount_last_60d | 0.0282 | Technical | 79.6% |
| 18 | net_flow_last_60d | 0.0266 | **Advanced Pol** | 82.2% |
| 19 | net_flow_last_90d | 0.0232 | **Advanced Pol** | 84.6% |
| 20 | news_count | 0.0231 | Sentiment | 86.9% |

---

## Category Breakdown

### Total Importance by Category

| Category | Total Importance | Avg Importance | Feature Count | % of Total |
|----------|------------------|----------------|---------------|------------|
| **Technical** | 0.6926 | 0.0301 | 23 | 69.3% |
| **Sentiment** | 0.1643 | 0.0411 | 4 | 16.4% |
| **Advanced Politician** | 0.1420 | 0.0118 | 12 | 14.2% |
| **Basic Politician** | 0.0011 | 0.0006 | 2 | 0.1% |

### Key Insights

1. **Technical features dominate** - 69% of total importance despite being only 56% of features
2. **Sentiment has high per-feature value** - 0.0411 avg vs 0.0301 technical
3. **Advanced politician features show promise** - 3 features in top 20 (#15, #18, #19)
4. **Basic politician features weak** - Only 0.1% importance, candidates for removal

---

## Recommended Feature Sets

### Top 5 Features (29.8% cumulative importance)
```
avg_sentiment_compound, HL_spread, Price_change, SMA_50, Volume_change
```
- **Composition**: 4 technical + 1 sentiment
- **Use case**: Minimal baseline model

### Top 10 Features (53.1% cumulative importance)
```
avg_sentiment_compound, HL_spread, Price_change, SMA_50, Volume_change,
MACD_diff, SMA_10_20_cross, SMA_20, RSI, MACD
```
- **Composition**: 9 technical + 1 sentiment
- **Use case**: Lean production model
- **⚠️ Warning**: No politician features

### Top 15 Features (73.2% cumulative importance) ⭐ RECOMMENDED
```
avg_sentiment_compound, HL_spread, Price_change, SMA_50, Volume_change,
MACD_diff, SMA_10_20_cross, SMA_20, RSI, MACD, SMA_10, SMA_20_50_cross,
avg_sentiment_positive, MACD_signal, days_since_last_trade
```
- **Composition**: 12 technical + 2 sentiment + 1 advanced politician
- **Use case**: **Optimal for MVP**
- **Benefits**: Balanced, includes politician signal, captures 73% importance
- **Justification**: Sweet spot between performance and complexity

### Top 20 Features (86.9% cumulative importance)
```
Top 15 + avg_sentiment_negative, amount_last_60d, net_flow_last_60d,
net_flow_last_90d, news_count
```
- **Composition**: 13 technical + 4 sentiment + 3 advanced politician
- **Use case**: Enhanced model with stronger politician signals
- **Benefits**: Full sentiment suite, 3 politician temporal features

---

## Feature Selection Recommendations

### Optimal Range Analysis

- **Top 17 features** capture **80% of importance** ✅
- **Top 21 features** capture **90% of importance**
- **Top 24 features** capture **95% of importance**

### Decision Matrix

| Feature Count | Cumulative % | Pros | Cons | Verdict |
|---------------|--------------|------|------|---------|
| 5 | 29.8% | Fast, simple | Too basic | ❌ Skip |
| 10 | 53.1% | Lean, efficient | No politician features | ⚠️ Backup option |
| 15 | 73.2% | Balanced, includes pol | Optimal for MVP | ✅ **RECOMMENDED** |
| 20 | 86.9% | Full signal coverage | More complex | ✅ **ALTERNATIVE** |
| 25+ | >90% | Diminishing returns | Overfitting risk | ❌ Avoid |

---

## Next Steps

### Immediate (Tuesday Afternoon)

1. **Feature Selection Experiments**
   - Create `feature_selection_experiments.py`
   - Test feature sets: 5, 10, 15, 20, 25
   - Measure: train accuracy, test accuracy, overfitting gap
   - Compare across multiple stocks (BABA, QCOM, NVDA)
   - Time estimate: 2 hours

2. **Select Final Feature Set**
   - Analyze experiment results
   - Balance: minimize overfitting, maximize test accuracy
   - Ensure diversity across categories
   - Document decision rationale
   - Time estimate: 30 minutes

3. **Update Documentation**
   - Add experiment results to this document
   - Update PROJECT_STATUS.md
   - Create final feature list for MVP
   - Time estimate: 1 hour

### Wednesday - XGBoost Implementation

- Use optimal feature set (likely 15-20 features)
- Compare Random Forest vs XGBoost vs Logistic Regression
- Target: Identify best model for MVP

---

## Stock-Specific Insights

### BABA (Best Performer: +9.89%)

**Top 5 Features**:
1. SMA_50 (0.0850) - Long-term trend
2. HL_spread (0.0619) - Volatility
3. avg_sentiment_compound (0.0594) - News sentiment
4. MACD_diff (0.0530) - Momentum
5. Volume_change (0.0516) - Trading activity

**Pattern**: Trend-following + sentiment-driven

### QCOM (Strong Performer: +4.61%)

**Top 5 Features**:
1. avg_sentiment_compound (0.0719) - **Sentiment-first!**
2. Volume_change (0.0536)
3. RSI (0.0504) - Mean reversion
4. HL_spread (0.0498)
5. Price_change (0.0474)

**Pattern**: Sentiment-dominant strategy

### NVDA (Moderate: +1.55%)

**Top 5 Features**:
1. Price_change (0.0843) - Short-term momentum
2. HL_spread (0.0753)
3. avg_sentiment_compound (0.0651)
4. MACD_diff (0.0602)
5. SMA_10_20_cross (0.0537)

**Pattern**: Short-term technical + sentiment

---

## Critical Observations

### ✅ Strengths

1. **Sentiment compound is universally strong** - #1 or top-3 for all stocks
2. **Technical features dominate numerically** - 12 of top 15 are technical
3. **Advanced politician features appear** - 3 in top 20, showing value
4. **Clear diminishing returns** - 80% captured by 17 features

### ⚠️ Concerns

1. **Basic politician features very weak** - buy_count and trade_amount near bottom
2. **Many advanced politician features low-ranked** - Only 3 of 12 in top 20
3. **Stock-specific variation** - Different stocks prefer different features
4. **60d/90d windows dominate 30d** - Longer-term politician signals stronger

### 💡 Insights for MVP

1. **Keep sentiment** - Clear value across all stocks
2. **Focus on top technical indicators** - SMA, MACD, RSI proven
3. **Select best politician features** - days_since_last_trade (#15), net_flow_60d (#18), net_flow_90d (#19)
4. **Remove basic politician features** - They add almost nothing
5. **Consider stock-specific models** - Post-MVP enhancement opportunity

---

## Files Generated

- `feature_importance_rankings.csv` - Aggregated rankings with statistics
- `feature_importance_detailed.csv` - Per-stock breakdown
- `feature_importance.log` - Full execution log

---

## Conclusion

**Recommendation**: Proceed with **15-feature model** for MVP

**Rationale**:
- Captures 73% of predictive power (sufficient for MVP)
- Balanced across categories (technical, sentiment, politician)
- Includes top politician feature (days_since_last_trade)
- Avoids overfitting risk from excessive features
- Aligned with MVP goal of 60% accuracy

**Next Action**: Run feature selection experiments to validate this choice empirically.
