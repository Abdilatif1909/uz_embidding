import sys, os
import json
import csv
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from app.utils.evaluation import (
    evaluate_similarity_dataset,
    evaluate_analogy_dataset,
    compute_spearman,
    compute_pearson,
    compute_metrics,
    save_results_csv,
    save_results_json,
    clear_vector_cache,
    PerformanceMonitor,
    timestamp
)
from app.utils.plot_results import (
    plot_similarity_scatter,
    plot_model_comparison,
    plot_all_metrics,
)


# ============================
# UTILITY: Get absolute results path
# ============================
def get_results_dir():
    """Get absolute path to results directory"""
    # Go up from backend/app to root project folder
    app_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(app_dir)
    project_root = os.path.dirname(os.path.dirname(backend_dir))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    return results_dir


# ============================
# TEST DATASETS
# ============================

# Kichik word similarity dataset (existing)
small_similarity_dataset = [
    ("kitob", "daftar", 0.82),
    ("ona", "ota", 0.74),
    ("uy", "hovli", 0.63),
    ("it", "mushuk", 0.45),
    ("yoʻl", "koʻcha", 0.52)
]

# Katta word similarity dataset
large_similarity_dataset = [
    ("kitob", "daftar", 0.82),
    ("ona", "ota", 0.74),
    ("uy", "hovli", 0.63),
    ("it", "mushuk", 0.45),
    ("yoʻl", "koʻcha", 0.52),
    ("odam", "kishi", 0.89),
    ("sog'liq", "kasallik", 0.71),
    ("qora", "oq", 0.35),
    ("issiq", "sovuq", 0.42),
    ("katta", "kichik", 0.78),
    ("tez", "asta", 0.65),
    ("yaxshi", "yomon", 0.81),
    ("shuq", "jimjit", 0.28),
    ("yoqimli", "nafrat", 0.19),
    ("bog'", "xonadon", 0.68),
    ("ko'l", "daryo", 0.72),
    ("tog'", "tepalik", 0.61),
    ("shamol", "bahor", 0.44),
    ("yomg'ir", "qor", 0.53),
    ("quyosh", "oy", 0.48),
]

# Analogy dataset (O'zbekcha analogiyalar)
# Format: (a, b, c, d) where a:b :: c:d
analogy_dataset = [
    ("erkak", "ayol", "oʻg'il", "qiz"),  # man:woman :: boy:girl
    ("ona", "ota", "opa", "aka"),  # mother:father :: sister:brother
    ("shahar", "qishloq", "yangi", "eski"),  # city:village :: new:old
    ("yoqimli", "yomon", "katta", "kichik"),  # good:bad :: big:small
    ("kitob", "qalam", "maktab", "o'qituvchi"),  # book:pen :: school:teacher
]


# ============================
# MODELS TO TEST
# ============================
models = [
    "fasttext-uz",
    # "fasttext-en",  # Skip due to memory constraints
    "bert"
]


# ============================
# UTILITY FUNCTIONS
# ============================
def create_results_directory():
    """Barcha natija papkalarini yaratish"""
    dirs = [
        os.path.join(os.path.dirname(__file__), "../../../results"),
        os.path.join(os.path.dirname(__file__), "../../../results/plots"),
        os.path.join(os.path.dirname(__file__), "../../../results/tables"),
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs[0]


def save_summary_metrics(all_metrics, filename=None):
    """Barcha metrikalarni JSON/CSV formatda saqlash"""
    if not filename:
        filename = f"summary_metrics_{timestamp()}"
    
    results_dir = create_results_directory()
    
    # JSON formatda saqlash
    json_path = os.path.join(results_dir, f"{filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)
    print(f"[ OK ] Summary JSON saqlandi → {json_path}")
    
    # CSV formatda saqlash
    csv_path = os.path.join(results_dir, f"{filename}.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if all_metrics and "models" in all_metrics:
            writer = csv.writer(f)
            
            # Header yozish
            headers = ["Metric"]
            for model in all_metrics.get("models", {}).keys():
                headers.append(model)
            writer.writerow(headers)
            
            # Metrikalar yozish
            for metric_name in ["spearman", "pearson", "mae", "rmse", "samples"]:
                row = [metric_name]
                for model in all_metrics.get("models", {}).keys():
                    value = all_metrics["models"][model].get("similarity", {}).get(metric_name, "-")
                    row.append(value)
                writer.writerow(row)
            
            # Performance metrikalar
            writer.writerow([])
            writer.writerow(["Performance Metrics"])
            for model in all_metrics.get("models", {}).keys():
                perf = all_metrics["models"][model].get("performance", {})
                if perf:
                    writer.writerow([f"{model} - Elapsed (s)", perf.get("elapsed_seconds", "-")])
                    writer.writerow([f"{model} - Memory (MB)", perf.get("memory_used_mb", "-")])
    
    print(f"[ OK ] Summary CSV saqlandi → {csv_path}")
    return json_path, csv_path


# ============================
# MAIN EXPERIMENT RUNNER
# ============================
def run_experiments():
    """Barcha eksperimentlarni ishga tushirish"""
    
    print("\n" + "="*70)
    print(">>> UZ EMBEDDING PLATFORM - EXPERIMENT RUNNER")
    print("="*70 + "\n")
    
    results_dir = create_results_directory()
    
    # Umumiy metrikalar
    all_metrics = {
        "timestamp": timestamp(),
        "models": {},
        "datasets": {}
    }
    
    # ==================
    # 1. SIMILARITY TEST
    # ==================
    print("\n[ STAGE 1 ] WORD SIMILARITY EVALUATION")
    print("-" * 70)
    
    monitor = PerformanceMonitor()
    
    for model in models:
        print(f"\n>> Testing {model.upper()}")
        
        model_metrics = {}
        
        # Small dataset
        print("\n  [SMALL] Dataset (5 pairs):")
        monitor.start()
        small_results = evaluate_similarity_dataset(small_similarity_dataset, model_type=model)
        small_perf = monitor.stop(f"{model} - Small Dataset")
        
        small_metrics = compute_metrics(small_results)
        model_metrics["small_dataset"] = small_metrics
        
        save_results_csv(small_results, f"{model}_small_similarity.csv")
        save_results_json(small_results, f"{model}_small_similarity.json")
        plot_similarity_scatter(small_results, f"{model}_small")
        
        print(f"    * Spearman: {small_metrics.get('spearman', 'N/A'):.4f}")
        print(f"    * Pearson: {small_metrics.get('pearson', 'N/A'):.4f}")
        
        # Large dataset
        print("\n  [LARGE] Dataset (20 pairs):")
        monitor.start()
        large_results = evaluate_similarity_dataset(large_similarity_dataset, model_type=model, use_batch=True)
        large_perf = monitor.stop(f"{model} - Large Dataset")
        
        large_metrics = compute_metrics(large_results)
        model_metrics["large_dataset"] = large_metrics
        
        save_results_csv(large_results, f"{model}_large_similarity.csv")
        save_results_json(large_results, f"{model}_large_similarity.json")
        plot_similarity_scatter(large_results, f"{model}_large")
        
        print(f"    * Spearman: {large_metrics.get('spearman', 'N/A'):.4f}")
        print(f"    * Pearson: {large_metrics.get('pearson', 'N/A'):.4f}")
        
        model_metrics["similarity"] = large_metrics
        model_metrics["performance"] = large_perf
        
        all_metrics["models"][model] = model_metrics
        
        # Cache tozalash
        clear_vector_cache()
    
    # ==================
    # 2. ANALOGY TEST
    # ==================
    print("\n\n[ STAGE 2 ] ANALOGY EVALUATION")
    print("-" * 70)
    
    for model in models:
        print(f"\n>> Testing {model.upper()}")
        
        monitor.start()
        analogy_results = evaluate_analogy_dataset(analogy_dataset, model_type=model)
        analogy_perf = monitor.stop(f"{model} - Analogy Test")
        
        # Analogy accuracy
        correct = sum(1 for r in analogy_results if r.get("correct", False))
        accuracy = (correct / len(analogy_results)) * 100 if analogy_results else 0
        
        save_results_csv(analogy_results, f"{model}_analogy.csv")
        save_results_json(analogy_results, f"{model}_analogy.json")
        
        print(f"    * Accuracy: {accuracy:.1f}% ({correct}/{len(analogy_results)})")
        
        if "models" not in all_metrics:
            all_metrics["models"][model] = {}
        
        all_metrics["models"][model]["analogy"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(analogy_results)
        }
        
        # Cache tozalash
        clear_vector_cache()
    
    # ==================
    # 3. MODEL COMPARISON
    # ==================
    print("\n\n[ STAGE 3 ] MODEL COMPARISON")
    print("-" * 70)
    
    comparison_scores = {}
    for model in models:
        if model in all_metrics["models"]:
            score = all_metrics["models"][model].get("similarity", {}).get("spearman", 0)
            comparison_scores[model] = score
    
    print("\nSpearman Correlation Results:")
    for model, score in comparison_scores.items():
        print(f"  {model:20} -> {score:.4f}")
    
    plot_model_comparison(comparison_scores)
    
    # ==================
    # 4. SAVE ALL METRICS
    # ==================
    print("\n\n[ STAGE 4 ] SAVING RESULTS")
    print("-" * 70)
    
    json_path, csv_path = save_summary_metrics(all_metrics)
    
    # ==================
    # FINAL REPORT
    # ==================
    print("\n\n" + "="*70)
    print("SUCCESS: BARCHA EKSPERIMENTLAR MUVAFFAQIYATLI YAKUNLANDI")
    print("="*70)
    print(f"\nResults dir: {results_dir}")
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV: {csv_path}")
    print("\nAll plots saved to 'results/plots/'\n")
    
    return all_metrics


# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    try:
        all_metrics = run_experiments()
    except KeyboardInterrupt:
        print("\n\nWARNING: Experiment interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
