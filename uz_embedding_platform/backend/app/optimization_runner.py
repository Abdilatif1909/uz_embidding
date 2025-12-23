import sys, os
import json
import csv
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from app.utils.evaluation import (
    evaluate_similarity_dataset,
    evaluate_similarity_with_tokenizer,
    evaluate_analogy_dataset,
    evaluate_analogy_bert_pooling,
    evaluate_similarity_bert_pooling,
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
    plot_optimization_summary,
)


# ============================
# TEST DATASETS
# ============================

similarity_dataset = [
    ("kitob", "daftar", 0.82),
    ("ona", "ota", 0.74),
    ("uy", "hovli", 0.63),
    ("it", "mushuk", 0.45),
    ("yoʻl", "koʻcha", 0.52),
    ("odam", "kishi", 0.89),
    ("sog'liq", "kasallik", 0.71),
]

analogy_dataset = [
    ("erkak", "ayol", "oʻg'il", "qiz"),
    ("ona", "ota", "opa", "aka"),
    ("shahar", "qishloq", "yangi", "eski"),
]


# ============================
# UTILITY FUNCTIONS
# ============================

def create_results_dir():
    """Create results directory"""
    results_dir = os.path.join(os.path.dirname(__file__), "../../../results")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "optimization"), exist_ok=True)
    return results_dir


def save_optimization_report(all_results):
    """Save optimization comparison report"""
    results_dir = create_results_dir()
    report_file = os.path.join(results_dir, "optimization", f"optimization_report_{timestamp()}.json")
    
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"[OK] Optimization report saved: {report_file}")
    
    # Create CSV summary
    csv_file = os.path.join(results_dir, "optimization", f"optimization_summary_{timestamp()}.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Configuration", "Spearman", "Pearson", "MAE", "RMSE", "Time(s)", "Memory(MB)"])
        
        for test_name, test_data in all_results.items():
            if isinstance(test_data, dict) and "results" in test_data:
                for result in test_data.get("results", []):
                    writer.writerow([
                        test_name,
                        result.get("config", ""),
                        result.get("spearman", "-"),
                        result.get("pearson", "-"),
                        result.get("mae", "-"),
                        result.get("rmse", "-"),
                        result.get("time", "-"),
                        result.get("memory", "-")
                    ])
    
    print(f"[OK] CSV summary saved: {csv_file}")
    return report_file, csv_file


def run_optimization_tests():
    """Run all optimization tests"""
    
    print("\n" + "="*70)
    print(">>> MODEL OPTIMIZATION & COMPARISON")
    print("="*70 + "\n")
    
    all_results = {
        "timestamp": timestamp(),
        "fasttext_tokenizer": {},
        "bert_pooling": {}
    }
    
    monitor = PerformanceMonitor()
    
    # ==========================================
    # 1. FASTTEXT-UZ TOKENIZER OPTIMIZATION
    # ==========================================
    print("\n[ STAGE 1 ] FastText-UZ Tokenizer Optimization")
    print("-" * 70)
    
    results_list = []
    
    # Original (without tokenizer)
    print("\n>> Original FastText (no tokenizer)")
    monitor.start()
    orig_results = evaluate_similarity_dataset(similarity_dataset, model_type="fasttext-uz")
    perf = monitor.stop("FastText Original")
    
    orig_metrics = compute_metrics(orig_results)
    results_list.append({
        "config": "Original (no tokenizer)",
        "spearman": orig_metrics.get("spearman"),
        "pearson": orig_metrics.get("pearson"),
        "mae": orig_metrics.get("mae"),
        "rmse": orig_metrics.get("rmse"),
        "time": perf.get("elapsed_seconds"),
        "memory": perf.get("memory_used_mb")
    })
    print(f"  Spearman: {orig_metrics.get('spearman'):.4f}")
    print(f"  Time: {perf.get('elapsed_seconds'):.2f}s")
    
    clear_vector_cache()
    
    # With tokenizer + stemming
    print("\n>> FastText with tokenizer + stemming")
    monitor.start()
    stem_results = evaluate_similarity_with_tokenizer(similarity_dataset, model_type="fasttext-uz", use_stemming=True)
    perf = monitor.stop("FastText Tokenized+Stemming")
    
    stem_metrics = compute_metrics(stem_results)
    results_list.append({
        "config": "With tokenizer + stemming",
        "spearman": stem_metrics.get("spearman"),
        "pearson": stem_metrics.get("pearson"),
        "mae": stem_metrics.get("mae"),
        "rmse": stem_metrics.get("rmse"),
        "time": perf.get("elapsed_seconds"),
        "memory": perf.get("memory_used_mb")
    })
    print(f"  Spearman: {stem_metrics.get('spearman'):.4f}")
    print(f"  Time: {perf.get('elapsed_seconds'):.2f}s")
    
    clear_vector_cache()
    
    # With tokenizer, no stemming
    print("\n>> FastText with tokenizer (no stemming)")
    monitor.start()
    nostem_results = evaluate_similarity_with_tokenizer(similarity_dataset, model_type="fasttext-uz", use_stemming=False)
    perf = monitor.stop("FastText Tokenized (no stemming)")
    
    nostem_metrics = compute_metrics(nostem_results)
    results_list.append({
        "config": "With tokenizer (no stemming)",
        "spearman": nostem_metrics.get("spearman"),
        "pearson": nostem_metrics.get("pearson"),
        "mae": nostem_metrics.get("mae"),
        "rmse": nostem_metrics.get("rmse"),
        "time": perf.get("elapsed_seconds"),
        "memory": perf.get("memory_used_mb")
    })
    print(f"  Spearman: {nostem_metrics.get('spearman'):.4f}")
    print(f"  Time: {perf.get('elapsed_seconds'):.2f}s")
    
    all_results["fasttext_tokenizer"]["results"] = results_list
    
    clear_vector_cache()
    
    # ==========================================
    # 2. BERT POOLING STRATEGY COMPARISON
    # ==========================================
    print("\n\n[ STAGE 2 ] BERT Pooling Strategy Comparison")
    print("-" * 70)
    
    bert_results = {}
    pooling_strategies = ["cls", "mean", "max", "weighted"]
    
    # Similarity test
    print("\n>> Similarity Task")
    sim_results = []
    
    for strategy in pooling_strategies:
        print(f"\n  Testing pooling: {strategy}")
        monitor.start()
        results = evaluate_similarity_bert_pooling(similarity_dataset, pooling_strategy=strategy)
        perf = monitor.stop(f"BERT {strategy} pooling")
        
        metrics = compute_metrics(results)
        sim_results.append({
            "pooling": strategy,
            "spearman": metrics.get("spearman"),
            "pearson": metrics.get("pearson"),
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "time": perf.get("elapsed_seconds"),
            "memory": perf.get("memory_used_mb")
        })
        print(f"    Spearman: {metrics.get('spearman'):.4f}")
        print(f"    Time: {perf.get('elapsed_seconds'):.2f}s")
        
        clear_vector_cache()
    
    bert_results["similarity"] = sim_results
    
    # Analogy test
    print("\n>> Analogy Task")
    analogy_results = []
    
    for strategy in pooling_strategies:
        print(f"\n  Testing pooling: {strategy}")
        monitor.start()
        results = evaluate_analogy_bert_pooling(analogy_dataset, pooling_strategy=strategy)
        perf = monitor.stop(f"BERT {strategy} analogy")
        
        correct = sum(1 for r in results if r.get("correct", False))
        accuracy = (correct / len(results)) * 100 if results else 0
        
        analogy_results.append({
            "pooling": strategy,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(results),
            "time": perf.get("elapsed_seconds"),
            "memory": perf.get("memory_used_mb")
        })
        print(f"    Accuracy: {accuracy:.1f}%")
        print(f"    Time: {perf.get('elapsed_seconds'):.2f}s")
        
        clear_vector_cache()
    
    bert_results["analogy"] = analogy_results
    all_results["bert_pooling"] = bert_results
    
    # ==========================================
    # 3. FINAL REPORT
    # ==========================================
    print("\n\n[ STAGE 3 ] GENERATING REPORT")
    print("-" * 70)
    
    json_file, csv_file = save_optimization_report(all_results)
    
    print("\n" + "="*70)
    print("SUCCESS: OPTIMIZATION TESTS COMPLETED")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}\n")
    
    # Print summary
    print("\n[FASTTEXT SUMMARY]")
    for result in results_list:
        print(f"  {result['config']:30} -> Spearman: {result['spearman']:.4f}")
    
    print("\n[BERT SIMILARITY SUMMARY]")
    for result in sim_results:
        print(f"  {result['pooling']:10} -> Spearman: {result['spearman']:.4f}")
    
    print("\n[BERT ANALOGY SUMMARY]")
    for result in analogy_results:
        print(f"  {result['pooling']:10} -> Accuracy: {result['accuracy']:.1f}%\n")
    
    # ==========================================
    # 4. GENERATE OPTIMIZATION PLOTS
    # ==========================================
    print("\n\n[ STAGE 4 ] CREATING VISUALIZATION PLOTS")
    print("-" * 70)
    
    plot_optimization_summary(all_results)
    
    print("\n" + "="*70)
    
    return all_results


if __name__ == "__main__":
    try:
        all_results = run_optimization_tests()
    except KeyboardInterrupt:
        print("\n\nWARNING: Optimization interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
