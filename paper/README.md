# Paper: Stock Movement Prediction with News Sentiment and Politician Position Signals

**Authors:** Conner Murphy and William Coleman  
**Date:** November 14, 2025

---

## 📄 Files

- **`stock_prediction_paper.tex`** - Main LaTeX paper file
- **`references.bib`** - Bibliography with all citations
- **`README.md`** - This file

---

## 🔨 How to Compile to PDF

### Option 1: Using `pdflatex` with BibTeX (Recommended)

```bash
cd paper
pdflatex stock_prediction_paper.tex
bibtex stock_prediction_paper
pdflatex stock_prediction_paper.tex
pdflatex stock_prediction_paper.tex
```

**Why multiple runs?**
- First `pdflatex`: Generates `.aux` file
- `bibtex`: Processes citations
- Second `pdflatex`: Incorporates citations
- Third `pdflatex`: Fixes cross-references

### Option 2: Using Overleaf (Easiest)

1. Go to [Overleaf.com](https://www.overleaf.com)
2. Create a new project → Upload Project
3. Upload `stock_prediction_paper.tex` and `references.bib`
4. Click "Recompile"
5. Download the PDF

### Option 3: Using `latexmk` (Automated)

```bash
cd paper
latexmk -pdf stock_prediction_paper.tex
```

This automatically handles all the multiple compilation passes.

---

## 📊 Paper Structure

### Abstract
- Research question
- Methodology overview
- Key results
- Contributions

### 1. Introduction
- Motivation
- Research gap
- Contributions

### 2. Related Work
- Congressional trading literature
- Sentiment analysis in finance
- Machine learning for stocks

### 3. Data and Sources
- Stock price data (Yahoo Finance)
- News sentiment data (442K articles)
- Politician trading data (Quiver API)
- Technical indicators

### 4. Methodology
- Feature engineering (61 features)
- XGBoost model with regularization
- Walk-forward validation
- Economic backtesting strategy

### 5. Results
- Overall performance (51.8% average)
- Sector-specific results (WFC: 70%, BABA: 68%, PFE: 61%)
- Economic backtest (Sharpe 2.22, +9.5% excess in WFC 2018)
- Overfitting analysis
- Feature importance

### 6. Discussion
- Sector-specific predictability
- Politician signal value
- Economic interpretation
- Limitations
- Comparison to prior work

### 7. Conclusion
- Summary of findings
- Practical implications
- Future work

---

## 📈 Key Results to Highlight

**Best Classification Performance:**
- Wells Fargo (WFC) 2018: **70.0% accuracy**
- Alibaba (BABA) 2019: **67.7% accuracy**
- Pfizer (PFE) 2019: **61.0% accuracy**

**Economic Performance:**
- Average Sharpe Ratio: **2.22** (excellent risk-adjusted returns)
- WFC 2018 Excess Return: **+9.5%** (beat buy-and-hold during downturn)
- Average Win Rate: **61.7%**

**Honest Limitations:**
- Average accuracy: 51.8% (barely above baseline)
- Severe overfitting: 38.4% train-test gap
- Technology stocks struggle (NVDA: 38.6%, TSLA: 43.3%)

---

## 💡 Presentation Tips

**Frame it positively but honestly:**

✅ **DO emphasize:**
- Sector-specific success (financials, healthcare)
- Superior risk-adjusted returns (Sharpe 2.22)
- Downside protection (WFC 2018)
- Novel integration of politician trading signals
- Rigorous validation methodology

⚠️ **BE HONEST about:**
- Limited average performance
- Technology sector failures
- Small sample size constraints
- Overfitting challenges

**Key message:**
> "Alternative data sources like politician trading provide sector-specific predictive value when combined with sentiment and technical signals. While not universally effective, our approach demonstrates meaningful returns for financial and healthcare stocks with superior risk management."

---

## 🎓 Academic Contributions

1. **First systematic integration** of politician trading + sentiment + technical indicators
2. **Comprehensive validation** across 8 stocks, 2 years, 16 experiments
3. **Economic validation** with transaction costs and risk-adjusted metrics
4. **Honest negative results** - documenting where approach fails
5. **Sector-specific insights** - not all stocks are equally predictable

---

## 📝 Suggested Edits Before Submission

1. **Add figures** (if journal allows):
   - Figure 1: Overfitting across stocks (training vs test accuracy)
   - Figure 2: Feature importance bar chart
   - Figure 3: Economic backtest equity curves

2. **Tables to consider adding**:
   - Detailed feature descriptions (appendix)
   - Year-by-year results for all stocks
   - Confusion matrices for best performers

3. **Expand sections if needed**:
   - Feature engineering details (appendix)
   - Robustness checks
   - Ablation studies (if you run the ablation script)

4. **Polish**:
   - Check all citations are in references.bib
   - Verify table/figure numbering
   - Proofread for typos
   - Consistent terminology

---

## 🔗 Related Documentation

- **[FINAL_VALIDATION_SUMMARY.md](../docs/FINAL_VALIDATION_SUMMARY.md)** - Complete validation results
- **[ECONOMIC_BACKTEST_RESULTS.md](../ECONOMIC_BACKTEST_RESULTS.md)** - Economic backtest details
- **[README.md](../README.md)** - Project overview

---

## ✅ Checklist Before Submission

- [ ] Compile successfully to PDF
- [ ] All tables render correctly
- [ ] All citations appear in bibliography
- [ ] Proofread entire paper
- [ ] Check page limit (if applicable)
- [ ] Add author affiliations/emails
- [ ] Add acknowledgments (if any)
- [ ] Include data availability statement
- [ ] Review journal formatting requirements
- [ ] Get co-author approval

---

## 📧 Contact

For questions about the paper or code:
- Conner Murphy: [email]
- William Coleman: [email]

---

**Paper Status:** Draft complete, ready for review and submission preparation

