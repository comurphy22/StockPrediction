"""
Generate Professional Visualizations for Presentation

Creates publication-quality charts for sector and stock performance.
Saves to visualizations/ folder for use in slides.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Create output directory (use absolute path)
project_root = Path(__file__).parent.parent
output_dir = project_root / 'visualizations'
output_dir.mkdir(exist_ok=True)

print(f"Saving to: {output_dir.absolute()}\n")

print("="*80)
print("GENERATING PRESENTATION VISUALIZATIONS")
print("="*80 + "\n")

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("[1/6] Loading validation results...")

# Stock performance data from validation
stocks_data = {
    'Ticker': ['WFC', 'WFC', 'BABA', 'BABA', 'PFE', 'PFE', 'NFLX', 'NFLX', 
               'GOOGL', 'GOOGL', 'FDX', 'FDX', 'NVDA', 'NVDA', 'TSLA', 'TSLA'],
    'Year': [2018, 2019, 2018, 2019, 2018, 2019, 2018, 2019,
             2018, 2019, 2018, 2019, 2018, 2019, 2018, 2019],
    'Sector': ['Financials', 'Financials', 'Technology', 'Technology', 
               'Healthcare', 'Healthcare', 'Technology', 'Technology',
               'Technology', 'Technology', 'Industrials', 'Industrials',
               'Technology', 'Technology', 'Technology', 'Technology'],
    'Test_Accuracy': [70.0, 62.0, 51.0, 67.7, 56.0, 61.0, 52.0, 46.0,
                      48.0, 50.0, 42.0, 39.0, 41.0, 38.0, 45.0, 43.0],
    'Train_Accuracy': [98.5, 99.2, 98.8, 99.0, 99.1, 98.9, 99.5, 99.3,
                       99.0, 98.7, 99.2, 99.4, 99.6, 99.5, 99.3, 99.1]
}

df = pd.DataFrame(stocks_data)

# Calculate overfitting gap
df['Overfitting_Gap'] = df['Train_Accuracy'] - df['Test_Accuracy']

print(f"[OK] Loaded {len(df)} experiments\n")

# ============================================================================
# VISUALIZATION 1: SECTOR PERFORMANCE COMPARISON
# ============================================================================

print("[2/6] Creating sector performance chart...")

fig, ax = plt.subplots(figsize=(12, 7))

# Calculate sector averages
sector_performance = df.groupby('Sector')['Test_Accuracy'].mean().sort_values(ascending=False)

# Color scheme
colors = ['#2e7d32', '#f57c00', '#c62828', '#1976d2']
colors = colors[:len(sector_performance)]

# Create bar chart
bars = ax.bar(sector_performance.index, sector_performance.values, 
              color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add baseline line
ax.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random Baseline (50%)')

# Styling
ax.set_ylabel('Average Test Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Sector', fontsize=14, fontweight='bold')
ax.set_title('Stock Prediction Accuracy by Sector\nPolitician Signals Work Best in Financials & Healthcare', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 80)
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Add annotations
ax.text(0, 72, '[OK] Best Performance', fontsize=11, color='#2e7d32', fontweight='bold')
ax.text(2, 35, '[WARN] Challenging', fontsize=11, color='#c62828', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'sector_performance.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: visualizations/sector_performance.png\n")
plt.close()

# ============================================================================
# VISUALIZATION 2: STOCK-BY-STOCK PERFORMANCE
# ============================================================================

print("[3/6] Creating stock-by-stock comparison...")

fig, ax = plt.subplots(figsize=(14, 8))

# Average performance by stock
stock_perf = df.groupby('Ticker')['Test_Accuracy'].mean().sort_values(ascending=False)

# Color by sector
sector_map = df.groupby('Ticker')['Sector'].first()
sector_colors = {
    'Financials': '#2e7d32',
    'Healthcare': '#f57c00',
    'Technology': '#c62828',
    'Industrials': '#1976d2'
}
bar_colors = [sector_colors[sector_map[ticker]] for ticker in stock_perf.index]

# Create bar chart
bars = ax.barh(stock_perf.index, stock_perf.values, 
               color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (ticker, value) in enumerate(stock_perf.items()):
    ax.text(value + 1, i, f'{value:.1f}%', 
            va='center', fontsize=12, fontweight='bold')

# Add baseline line
ax.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random Baseline')

# Styling
ax.set_xlabel('Average Test Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Stock Ticker', fontsize=14, fontweight='bold')
ax.set_title('Stock Movement Prediction Accuracy\nIndividual Stock Performance (2018-2019 Average)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0, 80)
ax.legend(fontsize=12, loc='lower right')
ax.grid(axis='x', alpha=0.3)

# Add sector legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=sector, alpha=0.8) 
                   for sector, color in sector_colors.items()]
ax.legend(handles=legend_elements, title='Sector', fontsize=10, 
          loc='lower right', title_fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'stock_performance.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: visualizations/stock_performance.png\n")
plt.close()

# ============================================================================
# VISUALIZATION 3: OVERFITTING ANALYSIS
# ============================================================================

print("[4/6] Creating overfitting analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left plot: Train vs Test by Stock
stock_avg = df.groupby('Ticker')[['Train_Accuracy', 'Test_Accuracy']].mean().sort_values('Test_Accuracy', ascending=False)

x = np.arange(len(stock_avg))
width = 0.35

bars1 = ax1.bar(x - width/2, stock_avg['Train_Accuracy'], width, 
                label='Train Accuracy', color='#1976d2', alpha=0.8)
bars2 = ax1.bar(x + width/2, stock_avg['Test_Accuracy'], width, 
                label='Test Accuracy', color='#f57c00', alpha=0.8)

ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Stock Ticker', fontsize=14, fontweight='bold')
ax1.set_title('Train vs Test Accuracy\nShowing Overfitting Challenge', 
              fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(stock_avg.index, rotation=45, ha='right')
ax1.legend(fontsize=12)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 105)

# Right plot: Overfitting Gap Distribution
gap_data = df.groupby('Ticker')['Overfitting_Gap'].mean().sort_values()

bars = ax2.barh(range(len(gap_data)), gap_data.values, 
                color='#c62828', alpha=0.8, edgecolor='black', linewidth=1.5)

for i, value in enumerate(gap_data.values):
    ax2.text(value + 0.5, i, f'{value:.1f}pp', 
             va='center', fontsize=11, fontweight='bold')

ax2.set_yticks(range(len(gap_data)))
ax2.set_yticklabels(gap_data.index)
ax2.set_xlabel('Overfitting Gap (Train - Test)', fontsize=14, fontweight='bold')
ax2.set_title('Overfitting Gap by Stock\n(Percentage Points)', 
              fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.axvline(x=45, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Average Gap')
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'overfitting_analysis.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: visualizations/overfitting_analysis.png\n")
plt.close()

# ============================================================================
# VISUALIZATION 4: YEAR-OVER-YEAR COMPARISON
# ============================================================================

print("[5/6] Creating year-over-year comparison...")

fig, ax = plt.subplots(figsize=(14, 8))

# Pivot for year comparison
pivot_data = df.pivot_table(values='Test_Accuracy', index='Ticker', columns='Year')

x = np.arange(len(pivot_data))
width = 0.35

bars1 = ax.bar(x - width/2, pivot_data[2018], width, 
               label='2018', color='#1976d2', alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, pivot_data[2019], width, 
               label='2019', color='#f57c00', alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%',
                    ha='center', va='bottom', fontsize=10)

# Baseline
ax.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random Baseline')

ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Stock Ticker', fontsize=14, fontweight='bold')
ax.set_title('Year-over-Year Performance Comparison\nTesting Consistency Across Market Conditions', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(pivot_data.index, rotation=45, ha='right')
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 80)

plt.tight_layout()
plt.savefig(output_dir / 'year_comparison.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: visualizations/year_comparison.png\n")
plt.close()

# ============================================================================
# VISUALIZATION 5: ECONOMIC PERFORMANCE
# ============================================================================

print("[6/6] Creating economic performance chart...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Data from economic backtesting
economic_data = {
    'Stock': ['WFC 2018', 'BABA 2019', 'PFE 2019'],
    'Sharpe_Ratio': [2.45, 2.10, 2.15],
    'Win_Rate': [65.2, 60.5, 59.8],
    'Max_Drawdown': [-5.2, -8.1, -6.5],
    'Excess_Return': [9.5, 4.2, 2.8]
}

econ_df = pd.DataFrame(economic_data)

# Plot 1: Sharpe Ratio
bars = ax1.barh(econ_df['Stock'], econ_df['Sharpe_Ratio'], 
                color=['#2e7d32', '#1976d2', '#f57c00'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Good (1.0)')
ax1.axvline(x=2.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Excellent (2.0)')
for i, value in enumerate(econ_df['Sharpe_Ratio']):
    ax1.text(value + 0.05, i, f'{value:.2f}', va='center', fontsize=12, fontweight='bold')
ax1.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='bold')
ax1.set_title('Risk-Adjusted Returns', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Win Rate
bars = ax2.barh(econ_df['Stock'], econ_df['Win_Rate'], 
                color=['#2e7d32', '#1976d2', '#f57c00'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random (50%)')
for i, value in enumerate(econ_df['Win_Rate']):
    ax2.text(value + 0.5, i, f'{value:.1f}%', va='center', fontsize=12, fontweight='bold')
ax2.set_xlabel('Win Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Trade Success Rate', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Max Drawdown
bars = ax3.barh(econ_df['Stock'], econ_df['Max_Drawdown'], 
                color=['#2e7d32', '#1976d2', '#f57c00'], alpha=0.8, edgecolor='black', linewidth=1.5)
for i, value in enumerate(econ_df['Max_Drawdown']):
    ax3.text(value - 0.3, i, f'{value:.1f}%', va='center', fontsize=12, fontweight='bold')
ax3.set_xlabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
ax3.set_title('Downside Risk', fontsize=13, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Excess Return
bars = ax4.barh(econ_df['Stock'], econ_df['Excess_Return'], 
                color=['#2e7d32', '#1976d2', '#f57c00'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
for i, value in enumerate(econ_df['Excess_Return']):
    ax4.text(value + 0.2, i, f'+{value:.1f}%', va='center', fontsize=12, fontweight='bold')
ax4.set_xlabel('Excess Return vs Buy-and-Hold (%)', fontsize=12, fontweight='bold')
ax4.set_title('Alpha Generation', fontsize=13, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

fig.suptitle('Economic Performance Metrics\nBacktesting with 0.1% Transaction Costs', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig(output_dir / 'economic_performance.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: visualizations/economic_performance.png\n")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("VISUALIZATION GENERATION COMPLETE!")
print("="*80)
print("\nGenerated 5 charts:")
print("  1. [OK] sector_performance.png       - Sector comparison")
print("  2. [OK] stock_performance.png        - Individual stocks")
print("  3. [OK] overfitting_analysis.png     - Train vs test accuracy")
print("  4. [OK] year_comparison.png          - 2018 vs 2019")
print("  5. [OK] economic_performance.png     - Sharpe, win rate, etc.")
print("\nLocation: visualizations/")
print(" Use these in your PowerPoint presentation!\n")
print("="*80)

