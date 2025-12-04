"""
Visualize Live Trading Predictions

Creates visualizations from daily_predictions_log.csv
showing model behavior and prediction patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Output directory
output_dir = Path(__file__).parent.parent / 'visualizations'
output_dir.mkdir(exist_ok=True)

print("="*80)
print("VISUALIZING LIVE PREDICTIONS")
print("="*80 + "\n")

# Load data
print("[1/4] Loading prediction log...")
results_file = Path(__file__).parent.parent / 'results' / 'daily_predictions_log.csv'
df = pd.read_csv(results_file)
df['prediction_date'] = pd.to_datetime(df['prediction_date'])
df['date_only'] = df['prediction_date'].dt.date

print(f"[OK] Loaded {len(df)} predictions across {df['date_only'].nunique()} days\n")

# ============================================================================
# VISUALIZATION 1: SIGNAL DISTRIBUTION & CONFIDENCE
# ============================================================================

print("[2/4] Creating signal distribution chart...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Signal counts by stock
signal_counts = df.groupby(['ticker', 'signal']).size().unstack(fill_value=0)
signal_counts.plot(kind='bar', ax=ax1, color=['#c62828', '#2e7d32', '#757575'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_title('Signal Distribution by Stock\nBUY vs SELL vs HOLD', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Signals', fontsize=12, fontweight='bold')
ax1.legend(['SELL', 'BUY', 'HOLD'], fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(axis='x', rotation=0)

# Panel 2: Average confidence by stock
avg_conf = df.groupby('ticker')['confidence'].mean().sort_values(ascending=False)
bars = ax2.barh(avg_conf.index, avg_conf.values * 100, 
                color='#1976d2', alpha=0.8, edgecolor='black', linewidth=1.5)
for i, value in enumerate(avg_conf.values * 100):
    ax2.text(value + 1, i, f'{value:.1f}%', va='center', fontsize=11, fontweight='bold')
ax2.set_xlabel('Average Confidence (%)', fontsize=12, fontweight='bold')
ax2.set_title('Model Confidence by Stock\nHow Sure is the Model?', 
              fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.set_xlim(0, 100)

# Panel 3: Confidence distribution
ax3.hist(df['confidence'] * 100, bins=20, color='#f57c00', alpha=0.7, 
         edgecolor='black', linewidth=1.5)
ax3.axvline(60, color='green', linestyle='--', linewidth=2, label='BUY Threshold (60%)')
ax3.axvline(40, color='red', linestyle='--', linewidth=2, label='SELL Threshold (40%)')
ax3.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Confidence Score Distribution\nModel Decision Thresholds', 
              fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# Panel 4: Signals over time
daily_signals = df.groupby(['date_only', 'signal']).size().unstack(fill_value=0)
if len(daily_signals) > 0:
    daily_signals.plot(kind='bar', stacked=True, ax=ax4, 
                       color=['#2e7d32', '#c62828', '#757575'], 
                       alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_title('Daily Signal Evolution\nModel Recommendations Over Time', 
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Signals', fontsize=12, fontweight='bold')
    ax4.legend(['BUY', 'SELL', 'HOLD'], fontsize=11)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)

fig.suptitle('Live Prediction Analysis\nModel Behavior & Signal Patterns', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig(output_dir / 'live_predictions_overview.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: visualizations/live_predictions_overview.png\n")
plt.close()

# ============================================================================
# VISUALIZATION 2: STOCK-BY-STOCK PREDICTION HEATMAP
# ============================================================================

print("[3/4] Creating prediction heatmap...")

fig, ax = plt.subplots(figsize=(14, 8))

# Create pivot table for heatmap
pivot_data = df.pivot_table(
    values='confidence', 
    index='ticker', 
    columns='date_only', 
    aggfunc='mean'
)

# Convert to percentages
pivot_data = pivot_data * 100

# Create heatmap
sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
            center=50, vmin=0, vmax=100, ax=ax,
            cbar_kws={'label': 'Confidence (%)'},
            linewidths=2, linecolor='white')

ax.set_title('Daily Prediction Confidence Heatmap\nGreen = Bullish (BUY), Red = Bearish (SELL)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Prediction Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Stock Ticker', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'live_predictions_heatmap.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: visualizations/live_predictions_heatmap.png\n")
plt.close()

# ============================================================================
# VISUALIZATION 3: DETAILED SIGNAL TIMELINE
# ============================================================================

print("[4/4] Creating detailed timeline...")

fig, ax = plt.subplots(figsize=(16, 10))

# Create timeline plot
tickers = df['ticker'].unique()
colors = {'BUY': '#2e7d32', 'SELL': '#c62828', 'HOLD': '#757575'}
markers = {'BUY': '^', 'SELL': 'v', 'HOLD': 'o'}

for i, ticker in enumerate(tickers):
    ticker_data = df[df['ticker'] == ticker].copy()
    ticker_data = ticker_data.sort_values('prediction_date')
    
    for signal in ['BUY', 'SELL', 'HOLD']:
        signal_data = ticker_data[ticker_data['signal'] == signal]
        if len(signal_data) > 0:
            ax.scatter(signal_data['prediction_date'], 
                      [i] * len(signal_data),
                      c=colors[signal], 
                      marker=markers[signal],
                      s=signal_data['confidence'] * 500,  # Size by confidence
                      alpha=0.7,
                      edgecolors='black',
                      linewidth=2,
                      label=signal if i == 0 else "")

ax.set_yticks(range(len(tickers)))
ax.set_yticklabels(tickers)
ax.set_xlabel('Prediction Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Stock Ticker', fontsize=12, fontweight='bold')
ax.set_title('Prediction Timeline\nMarker Size = Confidence Level', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
ax.legend(title='Signal', fontsize=11, title_fontsize=12, loc='upper right')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(output_dir / 'live_predictions_timeline.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: visualizations/live_predictions_timeline.png\n")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("="*80)
print("SUMMARY STATISTICS")
print("="*80 + "\n")

print("Signal Distribution:")
print(df['signal'].value_counts())
print()

print("Average Confidence by Signal:")
print(df.groupby('signal')['confidence'].mean() * 100)
print()

print("Most Bullish Stock (avg confidence):")
bullish = df[df['predicted_direction'] == 'UP'].groupby('ticker')['confidence'].mean().sort_values(ascending=False)
if len(bullish) > 0:
    print(f"  {bullish.index[0]}: {bullish.values[0]*100:.1f}%")
print()

print("Most Bearish Stock (avg confidence):")
bearish = df[df['predicted_direction'] == 'DOWN'].groupby('ticker')['confidence'].mean().sort_values(ascending=False)
if len(bearish) > 0:
    print(f"  {bearish.index[0]}: {bearish.values[0]*100:.1f}%")
print()

print("="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print("\nGenerated 3 charts:")
print("  1. [OK] live_predictions_overview.png  - 4-panel analysis")
print("  2. [OK] live_predictions_heatmap.png   - Confidence heatmap")
print("  3. [OK] live_predictions_timeline.png  - Signal timeline")
print("\nLocation: visualizations/")
print(" Perfect for showing model behavior in presentation!\n")
print("="*80)

