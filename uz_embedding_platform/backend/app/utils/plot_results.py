import os
import matplotlib.pyplot as plt
import numpy as np

def plot_similarity_scatter(results, model_name="model"):
    """Gold va predicted similarity ni scatter plot da ko'rsatish"""
    
    gold = [item["gold"] for item in results]
    pred = [item["cosine"] for item in results]

    plt.figure(figsize=(10, 8))
    plt.scatter(gold, pred, alpha=0.6, s=100, color='steelblue', edgecolors='black')
    
    # Diagonal line qo'shish (perfect prediction)
    min_val = min(min(gold), min(pred))
    max_val = max(max(gold), max(pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    plt.xlabel("Gold o'xshashlik", fontsize=12)
    plt.ylabel("Prediction o'xshashlik", fontsize=12)
    plt.title(f"Gold vs Predicted o'xshashlik ({model_name})", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = _save_path(f"scatter_{model_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[ OK ] Scatter saqlandi → {save_path}")


def plot_model_comparison(model_scores):
    """Modellarni Spearman correlation bo'yicha solishtirish"""

    models = list(model_scores.keys())
    scores = list(model_scores.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color=['steelblue', 'coral', 'lightgreen'][:len(models)], 
                    edgecolor='black', linewidth=1.5)
    
    # Bar ustiga qiymat yozish
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel("Spearman Correlation", fontsize=12)
    plt.title("Embedding Modellar Solishtirmasi", fontsize=14, fontweight='bold')
    plt.ylim(0, max(scores) * 1.1 if scores else 1)
    plt.grid(True, alpha=0.3, axis='y')

    save_path = _save_path("model_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[ OK ] Solishtirma saqlandi → {save_path}")


def plot_metrics_comparison(all_metrics, metric_name="spearman"):
    """Barcha modellarning metrikalarini taqqoslash"""
    
    models = list(all_metrics.get("models", {}).keys())
    values = []
    
    for model in models:
        metric_value = all_metrics["models"][model].get("similarity", {}).get(metric_name, 0)
        values.append(metric_value)
    
    if not values:
        return
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color=['steelblue', 'coral', 'lightgreen'][:len(models)],
                    edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel(metric_name.upper(), fontsize=12)
    plt.title(f"Modellar - {metric_name.upper()} Koeffitsenti", fontsize=14, fontweight='bold')
    plt.ylim(0, max(values) * 1.1 if values else 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    save_path = _save_path(f"{metric_name}_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[ OK ] {metric_name} Solishtirmasi saqlandi → {save_path}")


def plot_performance_metrics(all_metrics):
    """Performance metrikalarini (vaqt va memory) ko'rsatish"""
    
    models = list(all_metrics.get("models", {}).keys())
    times = []
    memory = []
    
    for model in models:
        perf = all_metrics["models"][model].get("performance", {})
        times.append(perf.get("elapsed_seconds", 0))
        memory.append(perf.get("memory_used_mb", 0))
    
    if not times or not memory:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Execution time
    bars1 = ax1.bar(models, times, color='steelblue', edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel("Vaqt (sekundlarda)", fontsize=12)
    ax1.set_title("Execution Time", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Memory usage
    bars2 = ax2.bar(models, memory, color='coral', edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars2, memory):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}MB',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel("Xotira (MB)", fontsize=12)
    ax2.set_title("Memory Usage", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = _save_path("performance_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[ OK ] Performance metrikalar saqlandi → {save_path}")


def plot_analogy_accuracy(all_metrics):
    """Analogy test accuracy ko'rsatish"""
    
    models = list(all_metrics.get("models", {}).keys())
    accuracies = []
    
    for model in models:
        analogy = all_metrics["models"][model].get("analogy", {})
        acc = analogy.get("accuracy", 0)
        accuracies.append(acc)
    
    if not accuracies:
        return
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['steelblue', 'coral', 'lightgreen'][:len(models)],
                    edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Analogy Test Aniqlik", fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    save_path = _save_path("analogy_accuracy.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[ OK ] Analogy Accuracy saqlandi → {save_path}")


def plot_all_metrics(all_metrics):
    """Barcha metrikalar uchun grafiklar qo'llab-quvvatlash"""
    
    print("\n[ PLOTTING ] Grafiklar ishlab chiqilmoqda...")
    
    # Performance grafiklar
    plot_performance_metrics(all_metrics)
    
    # Metrika solishtirmalar
    for metric in ["spearman", "pearson", "mae", "rmse"]:
        plot_metrics_comparison(all_metrics, metric)
    
    # Analogy accuracy
    plot_analogy_accuracy(all_metrics)
    
    print("[ OK ] Barcha grafiklar yakunlandi!")


def plot_fasttext_tokenizer_comparison(results_list):
    """FastText tokenizer strategies solishtirmasi"""
    
    configs = [r['config'] for r in results_list]
    spearman = [r['spearman'] for r in results_list]
    time_vals = [r['time'] for r in results_list]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Spearman comparison
    colors = ['steelblue', 'coral', 'lightgreen']
    bars1 = ax1.bar(range(len(configs)), spearman, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=15, ha='right', fontsize=9)
    ax1.set_ylabel("Spearman Correlation", fontsize=11)
    ax1.set_title("FastText-UZ Tokenizer Strategies", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars1, spearman)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Time comparison (log scale)
    bars2 = ax2.bar(range(len(configs)), time_vals, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=15, ha='right', fontsize=9)
    ax2.set_ylabel("Execution Time (seconds, log)", fontsize=11)
    ax2.set_title("FastText-UZ Performance", fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars2, time_vals)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_path = _save_path("fasttext_tokenizer_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[ OK ] FastText comparison saqlandi -> {save_path}")


def plot_bert_pooling_comparison(similarity_results, analogy_results):
    """BERT pooling strategies solishtirmasi"""
    
    strategies = [r['pooling'] for r in similarity_results]
    spearman = [r['spearman'] for r in similarity_results]
    accuracy = [r['accuracy'] for r in analogy_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['steelblue', 'coral', 'lightgreen', 'gold']
    
    # Similarity (Spearman)
    bars1 = ax1.bar(strategies, spearman, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel("Spearman Correlation", fontsize=11)
    ax1.set_title("BERT Pooling - Similarity Task", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.tick_params(axis='x', labelsize=9)
    
    for bar, val in zip(bars1, spearman):
        height = bar.get_height()
        va_pos = 'bottom' if height >= 0 else 'top'
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va=va_pos, fontsize=9, fontweight='bold')
    
    # Analogy (Accuracy)
    bars2 = ax2.bar(strategies, accuracy, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("BERT Pooling - Analogy Task", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', labelsize=9)
    
    for bar, val in zip(bars2, accuracy):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_path = _save_path("bert_pooling_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[ OK ] BERT pooling comparison saqlandi -> {save_path}")


def plot_optimization_summary(all_results):
    """Butun optimization testlarining comprehensive summary"""
    
    # Extract FastText results
    ft_results = all_results.get("fasttext_tokenizer", {}).get("results", [])
    
    # Extract BERT results
    bert_sim = all_results.get("bert_pooling", {}).get("similarity", [])
    bert_anal = all_results.get("bert_pooling", {}).get("analogy", [])
    
    if ft_results:
        plot_fasttext_tokenizer_comparison(ft_results)
    
    if bert_sim and bert_anal:
        plot_bert_pooling_comparison(bert_sim, bert_anal)
    
    print("[ OK ] Barcha optimization grafiklar yakunlandi!")


def _save_path(filename):
    """Natija papkasiga fayl saqlash yo'li"""
    
    folder = os.path.join(
        os.path.dirname(__file__),
        "../../../results/plots"
    )
    
    os.makedirs(folder, exist_ok=True)
    
    return os.path.join(folder, filename)
