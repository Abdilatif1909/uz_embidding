import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure plots directory exists
os.makedirs('results/plots', exist_ok=True)

# Load optimization results
with open('results/optimization/optimization_report_20251222_163754.json') as f:
    data = json.load(f)

# ============================
# 1. FASTTEXT COMPARISON GRAPH
# ============================
ft_data = data['fasttext_tokenizer']['results']
ft_configs = [r['config'].replace(' (', '\n(').replace(' +', '\n+') for r in ft_data]
ft_spearman = [r['spearman'] for r in ft_data]
ft_time = [r['time'] for r in ft_data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Spearman scores
colors = ['#2ecc71', '#e74c3c', '#3498db']
bars1 = ax1.bar(range(len(ft_configs)), ft_spearman, color=colors, edgecolor='black', linewidth=2)
ax1.set_xticks(range(len(ft_configs)))
ax1.set_xticklabels(['Original\n(no tokenizer)', 'Tokenizer\n+ stemming', 'Tokenizer\n(no stem)'], fontsize=10)
ax1.set_ylabel('Spearman Correlation', fontsize=12, fontweight='bold')
ax1.set_title('FastText-UZ: Tokenizer Strategies Comparison', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 0.7)

for bar, val in zip(bars1, ft_spearman):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Execution time (log scale)
bars2 = ax2.bar(range(len(ft_configs)), ft_time, color=colors, edgecolor='black', linewidth=2)
ax2.set_xticks(range(len(ft_configs)))
ax2.set_xticklabels(['Original\n(no tokenizer)', 'Tokenizer\n+ stemming', 'Tokenizer\n(no stem)'], fontsize=10)
ax2.set_ylabel('Time (seconds, log scale)', fontsize=12, fontweight='bold')
ax2.set_title('FastText-UZ: Performance Comparison', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars2, ft_time):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/plots/fasttext_comparison.png', dpi=200, bbox_inches='tight')
print('[OK] FastText comparison graph saved!')
plt.close()

# ============================
# 2. BERT POOLING COMPARISON
# ============================
bert_sim = data['bert_pooling']['similarity']
bert_anal = data['bert_pooling']['analogy']

strategies = [r['pooling'] for r in bert_sim]
spearman = [r['spearman'] for r in bert_sim]
accuracy = [r['accuracy'] for r in bert_anal]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors_bert = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']

# Similarity task
bars1 = ax1.bar(strategies, spearman, color=colors_bert, edgecolor='black', linewidth=2)
ax1.set_ylabel('Spearman Correlation', fontsize=12, fontweight='bold')
ax1.set_title('BERT: Pooling Strategies - Similarity Task', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.tick_params(axis='x', labelsize=11)

for bar, val in zip(bars1, spearman):
    height = bar.get_height()
    va_pos = 'bottom' if height >= 0 else 'top'
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va=va_pos, fontsize=11, fontweight='bold')

# Analogy task
bars2 = ax2.bar(strategies, accuracy, color=colors_bert, edgecolor='black', linewidth=2)
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('BERT: Pooling Strategies - Analogy Task', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 110)
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', labelsize=11)

for bar, val in zip(bars2, accuracy):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/plots/bert_comparison.png', dpi=200, bbox_inches='tight')
print('[OK] BERT comparison graph saved!')
plt.close()

# ============================
# 3. COMBINED SUMMARY CHART
# ============================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('UZ Embedding Platform - Optimization Results Summary', 
             fontsize=16, fontweight='bold', y=0.98)

# 1. FastText Spearman
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(range(len(ft_spearman)), ft_spearman, color=colors, edgecolor='black', linewidth=2)
ax1.set_xticks(range(len(ft_spearman)))
ax1.set_xticklabels(['Original', 'Tokenizer\n+ stem', 'Tokenizer'], fontsize=9)
ax1.set_ylabel('Spearman', fontsize=11, fontweight='bold')
ax1.set_title('1. FastText-UZ Performance', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, ft_spearman):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. FastText Time
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(range(len(ft_time)), ft_time, color=colors, edgecolor='black', linewidth=2)
ax2.set_xticks(range(len(ft_time)))
ax2.set_xticklabels(['Original', 'Tokenizer\n+ stem', 'Tokenizer'], fontsize=9)
ax2.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax2.set_title('2. Execution Time', fontsize=12, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, ft_time):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. BERT Similarity
ax3 = fig.add_subplot(gs[1, 0])
bars = ax3.bar(strategies, spearman, color=colors_bert, edgecolor='black', linewidth=2)
ax3.set_ylabel('Spearman', fontsize=11, fontweight='bold')
ax3.set_title('3. BERT Similarity (Word Pairs)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
for bar, val in zip(bars, spearman):
    height = bar.get_height()
    va_pos = 'bottom' if height >= 0 else 'top'
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va=va_pos, fontsize=10, fontweight='bold')

# 4. BERT Analogy
ax4 = fig.add_subplot(gs[1, 1])
bars = ax4.bar(strategies, accuracy, color=colors_bert, edgecolor='black', linewidth=2)
ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax4.set_title('4. BERT Analogy Task', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 110)
ax4.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, accuracy):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.savefig('results/plots/summary_dashboard.png', dpi=200, bbox_inches='tight')
print('[OK] Summary dashboard saved!')
plt.close()

print('\n' + '='*60)
print('ALL OPTIMIZATION GRAPHS SUCCESSFULLY GENERATED!')
print('='*60)
print('\nGenerated files:')
print('  1. results/plots/fasttext_comparison.png')
print('  2. results/plots/bert_comparison.png')
print('  3. results/plots/summary_dashboard.png')
print('\nYou can view these graphs now!')
print('='*60)
