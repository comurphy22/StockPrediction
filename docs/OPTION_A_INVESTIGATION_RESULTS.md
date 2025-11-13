# Option A Investigation Results: The 67.4% Mystery Solved

**Date:** November 4, 2025  
**Investigation:** Why did initial experiments show 67.4% accuracy but V5 validation only achieved 55.2%?

## 🔍 Investigation Summary

We re-ran the overfitting experiments on **all 10 stocks** (instead of just 3) to validate if the results scaled. The findings are definitive.

## 📊 Critical Findings

### Initial Experiments (3 stocks) - CLAIMED
- **Test Accuracy:** 67.4%
- **Overfitting Gap:** 32.6%
- **Stocks tested:** NFLX, NVDA, BABA
- **Configuration:** Alpha=1.0

### Re-run Experiments (3 same stocks) - ACTUAL
- **Test Accuracy:** 65.2%
- **Overfitting Gap:** ~35%
- **Stocks tested:** NFLX, NVDA, BABA
- **Configuration:** Alpha=1.0

### Full Experiments (all 10 stocks) - REALITY
- **Test Accuracy:** 53.7%
- **Overfitting Gap:** 46.3%
- **Stocks tested:** All 10 stocks
- **Configuration:** Alpha=1.0

## 🎯 The Truth Revealed

### What We Discovered

1. **The 3-stock sample was BIASED**
   - NFLX with Alpha=1.0: **72.7%** accuracy (excellent!)
   - NVDA with Alpha=1.0: **55.2%** accuracy (moderate)
   - BABA with Alpha=1.0: **67.7%** accuracy (good)
   - **Average of 3:** 65.2% (close to reported 67.4%)

2. **The other 7 stocks performed WORSE**
   - MU: 32.3% (terrible)
   - MSFT: 43.9% (poor)
   - GOOGL: 50.0% (coin flip)
   - AMZN: 48.8% (below random)
   - TSLA: 48.0% (below random)
   - QCOM: 62.5% (decent)
   - AAPL: 56.1% (moderate)

3. **Alpha=1.0 improvement is MINIMAL**
   - Baseline (10 stocks): **52.4%** accuracy
   - Alpha=1.0 (10 stocks): **53.7%** accuracy
   - **Net improvement: +1.4%** (not +13% as initially thought!)

### Per-Stock Performance: Baseline vs Alpha=1.0

| Stock | Baseline | Alpha=1.0 | Change | Winner |
|-------|----------|-----------|--------|--------|
| NFLX  | 57.6% | 72.7% | **+15.2%** | ✅ Alpha wins big |
| NVDA  | 41.4% | 55.2% | **+13.8%** | ✅ Alpha wins big |
| BABA  | 67.7% | 67.7% | 0.0% | Tie |
| AAPL  | 53.7% | 56.1% | +2.4% | ✅ Alpha slightly better |
| MSFT  | 41.5% | 43.9% | +2.4% | ✅ Alpha slightly better |
| QCOM  | 66.7% | 62.5% | **-4.2%** | ❌ Baseline better |
| GOOGL | 50.0% | 50.0% | 0.0% | Tie |
| AMZN  | 51.2% | 48.8% | -2.4% | ❌ Baseline better |
| TSLA  | 52.0% | 48.0% | -4.0% | ❌ Baseline better |
| MU    | 41.9% | 32.3% | **-9.7%** | ❌ Baseline better |

**Alpha=1.0 wins:** 4/10 stocks  
**Baseline wins:** 4/10 stocks  
**Ties:** 2/10 stocks

## 💡 Key Insights

### 1. Selection Bias in Initial Experiments

The initial 3-stock experiment **accidentally chose** the 2 stocks where Alpha=1.0 helps most:
- NFLX improved by +15.2%
- NVDA improved by +13.8%

These are **outliers**, not representative of overall performance.

### 2. Alpha=1.0 is Stock-Specific

L1 regularization helps dramatically for some stocks (NFLX, NVDA) but **hurts** others (MU, TSLA). There's no universal "best" configuration.

### 3. Overfitting Remains Unsolved

- Baseline: 47.6% gap
- Alpha=1.0: 46.3% gap
- **Reduction: 1.3 percentage points** (not the 13 points initially claimed)

All configurations still achieve 100% train accuracy but only ~53% test accuracy.

### 4. The V5 Validation Was Correct

V5 validation showed:
- Test Accuracy: 55.2%
- Overfit Gap: 44.7%

This matches our re-run experiments (53.7% accuracy, 46.3% gap). V5 was **accurate**, not flawed.

## 🤔 Why Did Initial Experiments Mislead Us?

### Probable Causes

1. **Small Sample Size**
   - 3 stocks is too small to assess generalizability
   - Cherry-picked (unintentionally) the best-case stocks

2. **No Validation Set**
   - Should have tested on a hold-out validation set separate from the 3 test stocks

3. **Confirmation Bias**
   - We wanted overfitting to be "solved" so we accepted the positive result
   - Didn't question why 3 stocks showed such dramatic improvement

### What We Should Have Done

1. Test on at least 5-7 stocks initially
2. Use cross-validation across stocks
3. Check for consistency across different stock types (tech, retail, etc.)
4. Be skeptical of "too good to be true" results

## 📋 Implications

### What This Means for the Project

1. **Overfitting is NOT solved**
   - Alpha=1.0 provides minimal benefit (+1.4% average)
   - The 44% gap remains fundamentally unchanged

2. **Stock-specific tuning may be needed**
   - NFLX and NVDA benefit from Alpha=1.0
   - MU and TSLA do NOT
   - May need per-stock hyperparameter optimization

3. **The problem is deeper than regularization**
   - 100% train accuracy suggests model is memorizing patterns
   - These patterns don't generalize to test set
   - May be fundamental issues with:
     - Feature quality
     - Data leakage
     - Problem formulation (daily predictions too noisy?)
     - Model architecture (XGBoost may not be ideal)

## 🎯 Recommendations Going Forward

### Option 1: Accept Current Performance ✅ RECOMMENDED
**Status:** Most realistic  
**Reasoning:** 55-56% accuracy may be the ceiling for daily stock prediction with this approach

**Actions:**
1. Document findings thoroughly
2. Focus on other value-adds (better features, longer timeframes)
3. Consider ensemble methods
4. Accept this as a learning experience

### Option 2: Try Ensemble Methods
**Status:** Worth exploring  
**Reasoning:** Combining multiple models might capture different patterns

**Actions:**
1. Train models with different hyperparameters per stock
2. Use soft voting or stacking
3. Combine XGBoost + Random Forest + Neural Network

### Option 3: Fundamental Rethink
**Status:** Nuclear option  
**Reasoning:** The problem formulation itself may be flawed

**Actions:**
1. Switch to weekly predictions (less noise)
2. Predict price movement magnitude, not just direction
3. Try different target definitions (beat market, not absolute direction)
4. Use time-series specific models (LSTM, Transformer)

### Option 4: Per-Stock Optimization
**Status:** Labor intensive  
**Reasoning:** Each stock may need custom configuration

**Actions:**
1. Run grid search per stock
2. Save best hyperparameters for each ticker
3. Use stock-specific models in production
4. May overfit to 2019 data

## 📝 Final Verdict

**The 67.4% result was a MIRAGE caused by:**
1. Small sample size (3 stocks)
2. Selection bias (accidentally chose best-performing stocks)
3. Lack of validation

**The TRUE performance of Alpha=1.0 is:**
- **53.7% accuracy** (barely better than baseline's 52.4%)
- **46.3% overfitting gap** (barely better than baseline's 47.6%)

**Recommendation:** Accept ~55% accuracy as the realistic ceiling and either:
- Move to Option 1 (accept and pivot to other improvements)
- Move to Option 3 (fundamental rethink of problem formulation)

**Do NOT:** Continue chasing hyperparameter tuning. The experiments prove it won't solve the fundamental issue.

---

## 📈 Data Supporting This Analysis

**Files Generated:**
- `results/overfitting_experiments.csv` - Summary results
- `results/overfitting_experiments_detailed.csv` - Per-stock breakdown
- `logs/overfitting_experiments_all10.log` - Full execution log

**Experiments Run:**
- Baseline (alpha=0)
- Alpha=0.5
- Alpha=1.0
- MaxDepth=4
- Depth4+Alpha1.0

All tested on 10 stocks: NFLX, NVDA, BABA, QCOM, MU, TSLA, AAPL, MSFT, GOOGL, AMZN
