import os
import csv
import json
import numpy as np
import psutil
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import lru_cache
from tqdm import tqdm
from app.utils.similarity import cosine_sim
from app.models.model_manager import load_model
from app.utils.tokenizer import uz_tokenize, uz_tokenize_no_stem, bert_preprocess

# Vector cache dictionary
_vector_cache = {}
_cache_size = 0
MAX_CACHE_SIZE = 500 * 1024 * 1024  # 500 MB limit


# ------------------------------
# Cache Management
# ------------------------------
def clear_vector_cache():
    global _vector_cache, _cache_size
    _vector_cache.clear()
    _cache_size = 0
    print("[ INFO ] Vector cache cleared")


def get_cached_vector(word, model, model_type):
    """Get cached vector or compute and cache it"""
    global _vector_cache, _cache_size
    
    cache_key = f"{model_type}_{word}"
    
    if cache_key in _vector_cache:
        return _vector_cache[cache_key]
    
    # Compute vector
    if model_type.startswith("fasttext"):
        vector = model.get_word_vector(word)
    else:
        vector = model.get_sentence_vector(word)
    
    # Convert to numpy array if it's a list
    if isinstance(vector, list):
        vector = np.array(vector)
    
    # Add to cache if space available
    vector_size = vector.nbytes
    if _cache_size + vector_size < MAX_CACHE_SIZE:
        _vector_cache[cache_key] = vector
        _cache_size += vector_size
    
    return vector


# ------------------------------
# Calculate similarity over dataset (Original - Single Process)
# ------------------------------
def evaluate_similarity_dataset(dataset, model_type="fasttext-uz", use_batch=False, batch_size=32):
    """
    Evaluate similarity for dataset
    
    Args:
        dataset: List of tuples (word1, word2, gold_score)
        model_type: Type of model to use
        use_batch: Whether to use batch processing
        batch_size: Size of batch for processing
    
    Returns:
        List of result dictionaries with metrics
    """
    model = load_model(model_type)
    results = []
    
    if use_batch and len(dataset) > batch_size:
        results = _evaluate_batched(dataset, model, model_type, batch_size)
    else:
        results = _evaluate_sequential(dataset, model, model_type)
    
    return results


def _evaluate_sequential(dataset, model, model_type):
    """Sequential evaluation with progress bar"""
    results = []
    
    for w1, w2, gold_score in tqdm(dataset, desc=f"Evaluating {model_type}", unit="pair"):
        v1 = get_cached_vector(w1, model, model_type)
        v2 = get_cached_vector(w2, model, model_type)
        
        sim = cosine_sim(v1, v2)
        
        results.append({
            "word1": w1,
            "word2": w2,
            "gold": float(gold_score),
            "cosine": float(sim)
        })
    
    return results


def _evaluate_batched(dataset, model, model_type, batch_size):
    """Batch evaluation with progress bar"""
    results = []
    batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
    
    for batch in tqdm(batches, desc=f"Evaluating {model_type} (batches)", unit="batch"):
        for w1, w2, gold_score in batch:
            v1 = get_cached_vector(w1, model, model_type)
            v2 = get_cached_vector(w2, model, model_type)
            
            sim = cosine_sim(v1, v2)
            
            results.append({
                "word1": w1,
                "word2": w2,
                "gold": float(gold_score),
                "cosine": float(sim)
            })
    
    return results


# ------------------------------
# TOKENIZER-OPTIMIZED EVALUATION
# ------------------------------
def evaluate_similarity_with_tokenizer(dataset, model_type="fasttext-uz", use_stemming=True):
    """
    Evaluate FastText similarity with tokenization preprocessing
    
    Args:
        dataset: List of tuples (word1, word2, gold_score)
        model_type: Type of model
        use_stemming: Whether to use stemming
    
    Returns:
        List of result dictionaries
    """
    model = load_model(model_type)
    results = []
    
    tokenize_func = uz_tokenize if use_stemming else uz_tokenize_no_stem
    
    for w1, w2, gold_score in tqdm(dataset, desc=f"Evaluating {model_type} (tokenized)", unit="pair"):
        # Tokenize words
        tokens1 = tokenize_func(w1)
        tokens2 = tokenize_func(w2)
        
        # Skip if tokenization resulted in empty
        if not tokens1 or not tokens2:
            continue
        
        # Get vectors (use first token or average if multiple)
        if len(tokens1) == 1:
            v1 = get_cached_vector(tokens1[0], model, model_type)
        else:
            v1 = np.mean([get_cached_vector(t, model, model_type) for t in tokens1], axis=0)
        
        if len(tokens2) == 1:
            v2 = get_cached_vector(tokens2[0], model, model_type)
        else:
            v2 = np.mean([get_cached_vector(t, model, model_type) for t in tokens2], axis=0)
        
        sim = cosine_sim(v1, v2)
        
        results.append({
            "word1": w1,
            "word2": w2,
            "tokens1": tokens1,
            "tokens2": tokens2,
            "gold": float(gold_score),
            "cosine": float(sim)
        })
    
    return results


# ------------------------------
# BERT POOLING STRATEGY EVALUATION
# ------------------------------
def evaluate_similarity_bert_pooling(dataset, pooling_strategy="mean"):
    """
    Evaluate BERT similarity with different pooling strategies
    
    Args:
        dataset: List of tuples (word1, word2, gold_score)
        pooling_strategy: 'cls', 'mean', 'max', or 'weighted'
    
    Returns:
        List of result dictionaries
    """
    from app.models.bert_wrapper import BERTEmbedding
    
    model = BERTEmbedding(pooling_strategy=pooling_strategy)
    results = []
    
    for w1, w2, gold_score in tqdm(dataset, desc=f"Evaluating BERT ({pooling_strategy})", unit="pair"):
        # Preprocess text
        w1_processed = bert_preprocess(w1)
        w2_processed = bert_preprocess(w2)
        
        # Get vectors
        v1 = model.get_sentence_vector(w1_processed, pooling_strategy=pooling_strategy)
        v2 = model.get_sentence_vector(w2_processed, pooling_strategy=pooling_strategy)
        
        # Convert to numpy if needed
        if isinstance(v1, list):
            v1 = np.array(v1)
        if isinstance(v2, list):
            v2 = np.array(v2)
        
        sim = cosine_sim(v1, v2)
        
        results.append({
            "word1": w1,
            "word2": w2,
            "strategy": pooling_strategy,
            "gold": float(gold_score),
            "cosine": float(sim)
        })
    
    return results


def evaluate_analogy_bert_pooling(dataset, pooling_strategy="mean"):
    """
    Evaluate BERT analogy with different pooling strategies
    
    Args:
        dataset: List of tuples (a, b, c, d)
        pooling_strategy: 'cls', 'mean', 'max', or 'weighted'
    
    Returns:
        List of result dictionaries
    """
    from app.models.bert_wrapper import BERTEmbedding
    
    model = BERTEmbedding(pooling_strategy=pooling_strategy)
    results = []
    
    for a, b, c, d in tqdm(dataset, desc=f"Evaluating BERT Analogies ({pooling_strategy})", unit="analogy"):
        try:
            # Preprocess
            a_p = bert_preprocess(a)
            b_p = bert_preprocess(b)
            c_p = bert_preprocess(c)
            d_p = bert_preprocess(d)
            
            # Get vectors
            va = np.array(model.get_sentence_vector(a_p, pooling_strategy=pooling_strategy))
            vb = np.array(model.get_sentence_vector(b_p, pooling_strategy=pooling_strategy))
            vc = np.array(model.get_sentence_vector(c_p, pooling_strategy=pooling_strategy))
            vd = np.array(model.get_sentence_vector(d_p, pooling_strategy=pooling_strategy))
            
            # a:b :: c:d
            target = vb - va + vc
            target = target / np.linalg.norm(target)
            
            pred_sim = cosine_sim(target, vd)
            
            results.append({
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "strategy": pooling_strategy,
                "similarity": float(pred_sim),
                "correct": int(pred_sim > 0.5)
            })
        except Exception as e:
            results.append({
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "strategy": pooling_strategy,
                "error": str(e)
            })
    
    return results


# ------------------------------
# Analogy Test
# ------------------------------
def evaluate_analogy_dataset(dataset, model_type="fasttext-uz"):
    """
    Evaluate model on analogy task
    Format: (a, b, c, d) where a:b :: c:d
    
    Args:
        dataset: List of tuples (a, b, c, d)
        model_type: Type of model to use
    
    Returns:
        List of result dictionaries
    """
    model = load_model(model_type)
    results = []
    
    for a, b, c, d in tqdm(dataset, desc=f"Evaluating Analogies {model_type}", unit="analogy"):
        try:
            va = get_cached_vector(a, model, model_type)
            vb = get_cached_vector(b, model, model_type)
            vc = get_cached_vector(c, model, model_type)
            vd = get_cached_vector(d, model, model_type)
            
            # a:b :: c:d => vd should be closest to (vb - va + vc)
            target = vb - va + vc
            target = target / np.linalg.norm(target)
            
            pred_sim = cosine_sim(target, vd)
            
            results.append({
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "similarity": float(pred_sim),
                "correct": int(pred_sim > 0.5)  # Convert bool to int for JSON serialization
            })
        except Exception as e:
            results.append({
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "error": str(e)
            })
    
    return results


# ------------------------------
# Save results as JSON
# ------------------------------
def save_results_json(results, filename=None):

    if not filename:
        filename = f"results_{timestamp()}.json"

    folder = os.path.join(
        os.path.dirname(__file__), "../../../results"
    )
    
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"[ OK ] JSON saqlandi → {path}")
    return path


# ------------------------------
# Save results as CSV
# ------------------------------
def save_results_csv(results, filename=None):

    if not filename:
        filename = f"results_{timestamp()}.csv"

    folder = os.path.join(
        os.path.dirname(__file__), "../../../results"
    )
    
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, filename)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Determine headers based on result type
        if results and "cosine" in results[0]:
            writer.writerow(["word1", "word2", "gold_score", "cosine_sim"])
            for item in results:
                writer.writerow([item["word1"], item["word2"], item["gold"], item["cosine"]])
        elif results and "similarity" in results[0]:
            writer.writerow(["a", "b", "c", "d", "similarity", "correct"])
            for item in results:
                if "error" not in item:
                    writer.writerow([item["a"], item["b"], item["c"], item["d"], item["similarity"], item["correct"]])

    print(f"[ OK ] CSV saqlandi → {path}")
    return path


# ------------------------------
# Evaluate model using Spearman correlation
# ------------------------------
def compute_spearman(results):

    gold = np.array([item["gold"] for item in results])
    predict = np.array([item["cosine"] for item in results])

    gold_rank = gold.argsort().argsort()
    pred_rank = predict.argsort().argsort()

    spearman = np.corrcoef(gold_rank, pred_rank)[0, 1]

    return float(spearman)


# ------------------------------
# Compute Pearson correlation
# ------------------------------
def compute_pearson(results):
    """Compute Pearson correlation coefficient"""
    gold = np.array([item["gold"] for item in results])
    predict = np.array([item["cosine"] for item in results])
    
    pearson = np.corrcoef(gold, predict)[0, 1]
    return float(pearson)


# ------------------------------
# Additional metrics
# ------------------------------
def compute_metrics(results):
    """Compute comprehensive metrics for evaluation results"""
    if not results or "cosine" not in results[0]:
        return {}
    
    gold = np.array([item["gold"] for item in results])
    predict = np.array([item["cosine"] for item in results])
    
    # Correlations
    spearman = compute_spearman(results)
    pearson = compute_pearson(results)
    
    # MAE and RMSE
    mae = np.mean(np.abs(gold - predict))
    rmse = np.sqrt(np.mean((gold - predict) ** 2))
    
    return {
        "spearman": float(spearman),
        "pearson": float(pearson),
        "mae": float(mae),
        "rmse": float(rmse),
        "samples": len(results)
    }


# ------------------------------
# Memory and time profiling
# ------------------------------
class PerformanceMonitor:
    """Monitor execution time and memory usage"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.metrics = {}
    
    def start(self):
        self.start_time = time.time()
        process = psutil.Process(os.getpid())
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    def stop(self, label=""):
        end_time = time.time()
        process = psutil.Process(os.getpid())
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        elapsed = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        
        self.metrics[label] = {
            "elapsed_seconds": round(elapsed, 2),
            "memory_used_mb": round(memory_delta, 2),
            "peak_memory_mb": round(end_memory, 2)
        }
        
        print(f"[ PERF ] {label}: {elapsed:.2f}s, Memory Delta: {memory_delta:.2f}MB, Peak: {end_memory:.2f}MB")
        
        return self.metrics[label]
    
    def get_metrics(self):
        return self.metrics


# ------------------------------
# timestamp helper
# ------------------------------
def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
