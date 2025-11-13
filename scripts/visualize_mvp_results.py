"""
Create visualizations for MVP validation results.

Generates:
1. Accuracy distribution histogram
2. Performance box plot
3. Confusion matrices for top 3 stocks
4. Overfitting gap analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load results
results_df = pd.read_csv('Results/mvp_validation_results.csv')
cm_df = pd.read_csv('Results/mvp_confusion_matrices.csv')

print("Creating MVP validation visualizations...")

# Figure 1: Accuracy Distribution Histogram
fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.hist(results_df['test_acc'] * 100, bins=8, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(x=60, color='red', linestyle='--', linewidth=2, label='MVP Target (60%)')
ax1.axvline(x=results_df['test_acc'].mean() * 100, color='green', linestyle='-', 
            linewidth=2, label=f'Mean ({results_df["test_acc"].mean()*100:.1f}%)')

ax1.set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Stocks', fontsize=12, fontweight='bold')
ax1.set_title('MVP Validation: Test Accuracy Distribution (10 Stocks)', 
              fontsize=14, fontweight='bold', pad=20)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add stats text
stats_text = f'Mean: {results_df["test_acc"].mean()*100:.2f}%\n'
stats_text += f'Median: {results_df["test_acc"].median()*100:.2f}%\n'
stats_text += f'Std Dev: {results_df["test_acc"].std()*100:.2f}%\n'
stats_text += f'Min: {results_df["test_acc"].min()*100:.2f}% ({results_df.loc[results_df["test_acc"].idxmin(), "ticker"]})\n'
stats_text += f'Max: {results_df["test_acc"].max()*100:.2f}% ({results_df.loc[results_df["test_acc"].idxmax(), "ticker"]})'

ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('visualizations/mvp_accuracy_distribution.png', bbox_inches='tight')
print("✅ Saved: visualizations/mvp_accuracy_distribution.png")
plt.close()

# Figure 2: Performance Box Plot with Train vs Test
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))

# Sort by test accuracy
results_sorted = results_df.sort_values('test_acc', ascending=False)

# Plot 1: Test Accuracy by Stock
colors = ['green' if acc >= 0.60 else 'orange' if acc >= 0.55 else 'red' 
          for acc in results_sorted['test_acc']]
bars = ax2a.barh(results_sorted['ticker'], results_sorted['test_acc'] * 100, color=colors, alpha=0.7)
ax2a.axvline(x=60, color='red', linestyle='--', linewidth=2, label='MVP Target')
ax2a.set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax2a.set_ylabel('Stock Ticker', fontsize=12, fontweight='bold')
ax2a.set_title('Test Accuracy by Stock', fontsize=12, fontweight='bold')
ax2a.legend()
ax2a.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (ticker, acc) in enumerate(zip(results_sorted['ticker'], results_sorted['test_acc'])):
    ax2a.text(acc * 100 + 1, i, f'{acc*100:.1f}%', va='center', fontsize=9)

# Plot 2: Train vs Test Comparison
x = np.arange(len(results_sorted))
width = 0.35

bars1 = ax2b.bar(x - width/2, results_sorted['train_acc'] * 100, width, label='Train', 
                 color='lightblue', alpha=0.8)
bars2 = ax2b.bar(x + width/2, results_sorted['test_acc'] * 100, width, label='Test',
                 color='steelblue', alpha=0.8)

ax2b.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
ax2b.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2b.set_title('Train vs Test Accuracy (Overfitting)', fontsize=12, fontweight='bold')
ax2b.set_xticks(x)
ax2b.set_xticklabels(results_sorted['ticker'], rotation=45)
ax2b.legend()
ax2b.grid(True, alpha=0.3, axis='y')
ax2b.set_ylim([0, 105])

# Add gap annotations for top 3 overfitters
worst_overfit = results_sorted.nlargest(3, 'overfit_gap')
for idx in worst_overfit.index:
    stock_idx = list(results_sorted.index).index(idx)
    gap = results_sorted.loc[idx, 'overfit_gap'] * 100
    ax2b.annotate(f'+{gap:.0f}%', xy=(stock_idx, 50), 
                 fontsize=8, ha='center', color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/mvp_performance_comparison.png', bbox_inches='tight')
print("✅ Saved: visualizations/mvp_performance_comparison.png")
plt.close()

# Figure 3: Confusion Matrices for Top 3 Stocks
top3 = results_df.nlargest(3, 'test_acc')
top3_tickers = top3['ticker'].tolist()

fig3, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (ticker, ax) in enumerate(zip(top3_tickers, axes)):
    cm_row = cm_df[cm_df['ticker'] == ticker].iloc[0]
    cm = np.array([[cm_row['tn'], cm_row['fp']], 
                   [cm_row['fn'], cm_row['tp']]])
    
    # Normalize to percentages
    cm_pct = cm / cm.sum() * 100
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Predicted Down', 'Predicted Up'],
                yticklabels=['Actual Down', 'Actual Up'])
    
    acc = top3[top3['ticker'] == ticker]['test_acc'].values[0] * 100
    ax.set_title(f'{ticker}\nTest Accuracy: {acc:.2f}%', fontsize=12, fontweight='bold')
    
plt.suptitle('Confusion Matrices: Top 3 Performing Stocks', 
             fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('visualizations/mvp_confusion_matrices.png', bbox_inches='tight')
print("✅ Saved: visualizations/mvp_confusion_matrices.png")
plt.close()

# Figure 4: Feature Correlation with Performance
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: News Articles vs Test Accuracy
ax1 = axes[0, 0]
ax1.scatter(results_df['news_articles'], results_df['test_acc'] * 100, 
           s=100, alpha=0.6, c=results_df['test_acc'], cmap='RdYlGn', vmin=0.4, vmax=0.9)
for idx, row in results_df.iterrows():
    ax1.annotate(row['ticker'], (row['news_articles'], row['test_acc']*100), 
                fontsize=8, ha='center')
ax1.set_xlabel('Number of News Articles', fontsize=11, fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('News Coverage vs Accuracy', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=60, color='red', linestyle='--', alpha=0.5)

# Plot 2: Politician Trades vs Test Accuracy
ax2 = axes[0, 1]
ax2.scatter(results_df['politician_trades'], results_df['test_acc'] * 100,
           s=100, alpha=0.6, c=results_df['test_acc'], cmap='RdYlGn', vmin=0.4, vmax=0.9)
for idx, row in results_df.iterrows():
    ax2.annotate(row['ticker'], (row['politician_trades'], row['test_acc']*100),
                fontsize=8, ha='center')
ax2.set_xlabel('Number of Politician Trades', fontsize=11, fontweight='bold')
ax2.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Politician Trading Activity vs Accuracy', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=60, color='red', linestyle='--', alpha=0.5)

# Plot 3: Overfitting Gap
ax3 = axes[1, 0]
results_sorted = results_df.sort_values('overfit_gap')
colors = ['red' if gap > 0.5 else 'orange' if gap > 0.4 else 'yellow' 
          for gap in results_sorted['overfit_gap']]
ax3.barh(results_sorted['ticker'], results_sorted['overfit_gap'] * 100, color=colors, alpha=0.7)
ax3.set_xlabel('Overfitting Gap (%)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Stock Ticker', fontsize=11, fontweight='bold')
ax3.set_title('Overfitting Analysis (Train - Test Accuracy)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (ticker, gap) in enumerate(zip(results_sorted['ticker'], results_sorted['overfit_gap'])):
    ax3.text(gap * 100 + 1, i, f'+{gap*100:.1f}%', va='center', fontsize=8)

# Plot 4: Sample Size vs Performance
ax4 = axes[1, 1]
ax4.scatter(results_df['n_test'], results_df['test_acc'] * 100,
           s=100, alpha=0.6, c=results_df['test_acc'], cmap='RdYlGn', vmin=0.4, vmax=0.9)
for idx, row in results_df.iterrows():
    ax4.annotate(row['ticker'], (row['n_test'], row['test_acc']*100),
                fontsize=8, ha='center')
ax4.set_xlabel('Test Set Size (samples)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
ax4.set_title('Sample Size vs Accuracy', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=60, color='red', linestyle='--', alpha=0.5)

plt.suptitle('MVP Validation: Performance Analysis', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('visualizations/mvp_performance_analysis.png', bbox_inches='tight')
print("✅ Saved: visualizations/mvp_performance_analysis.png")
plt.close()

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print("Generated 4 figures:")
print("  1. visualizations/mvp_accuracy_distribution.png")
print("  2. visualizations/mvp_performance_comparison.png")
print("  3. visualizations/mvp_confusion_matrices.png")
print("  4. visualizations/mvp_performance_analysis.png")
print("="*70)
