"""
Feature Importance Visualization

Creates visual charts for:
1. Top 25 features by importance (bar chart)
2. Category breakdown (pie chart)
3. Feature count vs accuracy (line chart)
4. Stock-specific feature importance heatmap
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("FEATURE IMPORTANCE VISUALIZATION")
print("="*70)

# Load data
print("\n[1/6] Loading data...")
importance_df = pd.read_csv('feature_importance_rankings.csv')
detailed_df = pd.read_csv('feature_importance_detailed.csv')
selection_df = pd.read_csv('feature_selection_results.csv')

print(f"      [OK] Rankings: {len(importance_df)} features")
print(f"      [OK] Detailed: {len(detailed_df)} records")
print(f"      [OK] Selection: {len(selection_df)} experiments")

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# ===== PLOT 1: Top 25 Features Bar Chart =====
print("\n[2/6] Creating top 25 features bar chart...")
ax1 = plt.subplot(2, 3, 1)

top_25 = importance_df.head(25).copy()
colors = top_25['category'].map({
    'Technical': '#3498db',
    'Sentiment': '#e74c3c',
    'Advanced Politician': '#2ecc71',
    'Basic Politician': '#f39c12'
})

bars = ax1.barh(range(len(top_25)), top_25['mean_importance'], color=colors)
ax1.set_yticks(range(len(top_25)))
ax1.set_yticklabels(top_25['feature'], fontsize=8)
ax1.set_xlabel('Mean Importance', fontsize=10)
ax1.set_title('Top 25 Features by Importance', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(top_25.iterrows()):
    ax1.text(row['mean_importance'], i, f" {row['mean_importance']:.4f}", 
             va='center', fontsize=7)

print("      [OK] Done")

# ===== PLOT 2: Category Breakdown Pie Chart =====
print("[3/6] Creating category breakdown pie chart...")
ax2 = plt.subplot(2, 3, 2)

category_totals = importance_df.groupby('category')['mean_importance'].sum()
colors_pie = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

wedges, texts, autotexts = ax2.pie(
    category_totals.values,
    labels=category_totals.index,
    autopct='%1.1f%%',
    colors=colors_pie,
    startangle=90
)

for text in texts:
    text.set_fontsize(10)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

ax2.set_title('Feature Importance by Category', fontsize=12, fontweight='bold')
print("      [OK] Done")

# ===== PLOT 3: Cumulative Importance =====
print("[4/6] Creating cumulative importance curve...")
ax3 = plt.subplot(2, 3, 3)

ax3.plot(range(1, len(importance_df) + 1), 
         importance_df['cumulative_pct'], 
         'b-', linewidth=2, label='Cumulative %')
ax3.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80% threshold')
ax3.axvline(x=17, color='r', linestyle='--', alpha=0.7)
ax3.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
ax3.axvline(x=21, color='orange', linestyle='--', alpha=0.7)

ax3.set_xlabel('Number of Features', fontsize=10)
ax3.set_ylabel('Cumulative Importance (%)', fontsize=10)
ax3.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)
ax3.legend(fontsize=8)

# Add annotations
ax3.annotate('17 features\n80%', xy=(17, 80), xytext=(17, 65),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
ax3.annotate('21 features\n90%', xy=(21, 90), xytext=(30, 85),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7))

print("      [OK] Done")

# ===== PLOT 4: Feature Count vs Accuracy =====
print("[5/6] Creating feature count vs accuracy chart...")
ax4 = plt.subplot(2, 3, 4)

# Group by feature count
selection_summary = selection_df.groupby('n_features').agg({
    'train_acc': 'mean',
    'test_acc': 'mean',
    'overfit_gap': 'mean'
}).reset_index()

ax4.plot(selection_summary['n_features'], selection_summary['train_acc'] * 100, 
         'o-', linewidth=2, markersize=8, label='Train Accuracy', color='#3498db')
ax4.plot(selection_summary['n_features'], selection_summary['test_acc'] * 100, 
         's-', linewidth=2, markersize=8, label='Test Accuracy', color='#e74c3c')

# Mark optimal point
optimal_idx = selection_summary['test_acc'].idxmax()
optimal_n = selection_summary.loc[optimal_idx, 'n_features']
optimal_acc = selection_summary.loc[optimal_idx, 'test_acc'] * 100

ax4.scatter([optimal_n], [optimal_acc], s=200, marker='*', 
           color='gold', edgecolor='black', linewidth=2, zorder=5,
           label=f'Optimal ({optimal_n} features)')

ax4.set_xlabel('Number of Features', fontsize=10)
ax4.set_ylabel('Accuracy (%)', fontsize=10)
ax4.set_title('Model Performance vs Feature Count', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.legend(fontsize=9)
ax4.set_ylim(40, 105)

# Add 60% MVP target line
ax4.axhline(y=60, color='green', linestyle='--', alpha=0.5, label='MVP Target (60%)')

print("      [OK] Done")

# ===== PLOT 5: Stock-Specific Heatmap =====
print("[6/6] Creating stock-specific feature importance heatmap...")
ax5 = plt.subplot(2, 3, 5)

# Pivot detailed data for top 15 features
top_15_features = importance_df.head(15)['feature'].tolist()
heatmap_data = detailed_df[detailed_df['feature'].isin(top_15_features)].pivot(
    index='feature', 
    columns='ticker', 
    values='importance'
)

# Reorder to match importance ranking
heatmap_data = heatmap_data.reindex(top_15_features)

sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
           linewidths=0.5, ax=ax5, cbar_kws={'label': 'Importance'})

ax5.set_title('Top 15 Features: Stock-Specific Importance', 
             fontsize=12, fontweight='bold')
ax5.set_xlabel('Stock', fontsize=10)
ax5.set_ylabel('Feature', fontsize=10)
plt.setp(ax5.get_yticklabels(), fontsize=8)

print("      [OK] Done")

# ===== PLOT 6: Overfitting Gap Analysis =====
ax6 = plt.subplot(2, 3, 6)

# Create bar chart of overfitting gaps
n_features_list = selection_summary['n_features'].values
gaps = selection_summary['overfit_gap'].values * 100

colors_bars = ['red' if g > 40 else 'orange' if g > 30 else 'green' for g in gaps]
bars = ax6.bar(range(len(n_features_list)), gaps, color=colors_bars, alpha=0.7)

ax6.set_xticks(range(len(n_features_list)))
ax6.set_xticklabels(n_features_list)
ax6.set_xlabel('Number of Features', fontsize=10)
ax6.set_ylabel('Overfitting Gap (%)', fontsize=10)
ax6.set_title('Overfitting Analysis by Feature Count', fontsize=12, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
ax6.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Moderate (30%)')
ax6.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='High (40%)')

# Add value labels
for i, (n, gap) in enumerate(zip(n_features_list, gaps)):
    ax6.text(i, gap + 1, f'{gap:.1f}%', ha='center', fontsize=8, fontweight='bold')

ax6.legend(fontsize=8)

print("      [OK] Done")

# Adjust layout and save
plt.tight_layout()
output_file = 'feature_importance_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n[OK] Visualization saved to: {output_file}")

# Also create a simplified version for presentations
print("\n[BONUS] Creating simplified presentation chart...")
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Top 15 features only
ax_left = axes[0]
top_15 = importance_df.head(15).copy()
colors = top_15['category'].map({
    'Technical': '#3498db',
    'Sentiment': '#e74c3c',
    'Advanced Politician': '#2ecc71'
})

bars = ax_left.barh(range(len(top_15)), top_15['mean_importance'], color=colors)
ax_left.set_yticks(range(len(top_15)))
ax_left.set_yticklabels(top_15['feature'], fontsize=11)
ax_left.set_xlabel('Mean Importance', fontsize=12, fontweight='bold')
ax_left.set_title('Top 15 Features by Importance', fontsize=14, fontweight='bold')
ax_left.invert_yaxis()
ax_left.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(top_15.iterrows()):
    ax_left.text(row['mean_importance'], i, f" {row['mean_importance']:.4f}", 
                va='center', fontsize=9, fontweight='bold')

# Add category legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', label='Technical'),
    Patch(facecolor='#e74c3c', label='Sentiment'),
    Patch(facecolor='#2ecc71', label='Advanced Politician')
]
ax_left.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Right: Feature count vs accuracy (clean)
ax_right = axes[1]

ax_right.plot(selection_summary['n_features'], selection_summary['test_acc'] * 100, 
             'o-', linewidth=3, markersize=12, label='Test Accuracy', color='#e74c3c')

# Mark optimal point
ax_right.scatter([optimal_n], [optimal_acc], s=400, marker='*', 
                color='gold', edgecolor='black', linewidth=3, zorder=5,
                label=f'Optimal: {optimal_n} features')

ax_right.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax_right.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax_right.set_title('Optimal Feature Count = 25', fontsize=14, fontweight='bold')
ax_right.grid(alpha=0.3)
ax_right.legend(fontsize=11, loc='lower right')
ax_right.set_ylim(50, 72)

# Add MVP target line
ax_right.axhline(y=60, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax_right.text(40, 61, 'MVP Target (60%)', fontsize=10, color='green', fontweight='bold')

# Add annotations for key points
for i, row in selection_summary.iterrows():
    acc = row['test_acc'] * 100
    n = row['n_features']
    ax_right.annotate(f'{acc:.1f}%', xy=(n, acc), xytext=(0, 10),
                     textcoords='offset points', ha='center', fontsize=9,
                     fontweight='bold')

plt.tight_layout()
simple_file = 'feature_importance_simple.png'
plt.savefig(simple_file, dpi=300, bbox_inches='tight')
print(f"[OK] Simple visualization saved to: {simple_file}")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print(f"\nCreated files:")
print(f"  1. {output_file} (comprehensive, 6 charts)")
print(f"  2. {simple_file} (presentation-ready, 2 charts)")
print("\nKey findings visualized:")
print("  [OK] Top 25 features ranked by importance")
print("  [OK] Category breakdown (Technical 69%, Sentiment 16%, Politician 14%)")
print("  [OK] Cumulative importance curve (80% at 17 features)")
print("  [OK] Feature count vs accuracy (25 features optimal)")
print("  [OK] Stock-specific patterns (heatmap)")
print("  [OK] Overfitting analysis")
print("="*70)

plt.show()
