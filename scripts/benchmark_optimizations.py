"""
Comprehensive Benchmark Script — Tests optimization axes for StudyHub AI.

Runs 5 benchmark tests using test datasets from tests/ folder:
  1. Flat L2 vs HNSW indexing (speed + recall accuracy)
  2. Chunk size variations (500, 1000, 1500, 2000)
  3. Embedding model comparison (MiniLM-384 vs MPNet-768)
  4. HNSW parameter tuning (M, efSearch, efConstruction)
  5. Batch vs single embedding speed

Usage:
    cd c:\\Users\\Amar\\Desktop\\SOP
    python scripts/benchmark_optimizations.py
"""
import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ──────────────────────────── Helpers ────────────────────────────

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_recall(true_ids, predicted_ids, k):
    """Compute Recall@k: fraction of true top-k neighbors found."""
    if len(true_ids) == 0:
        return 1.0
    hits = len(set(true_ids[:k]) & set(predicted_ids[:k]))
    return hits / min(k, len(true_ids))


def timed(func, *args, **kwargs):
    """Returns (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - t0


# ──────────────────────────── Test 1 ────────────────────────────

def test_flat_vs_hnsw(encoder, datasets):
    """Compare Flat L2 vs HNSW: search speed and recall accuracy."""
    print("\n" + "=" * 70)
    print("TEST 1: Flat L2  vs  HNSW — Speed & Recall Accuracy")
    print("=" * 70)

    dim = 768
    results = []

    for ds in datasets:
        texts = ds["texts"]
        queries = ds["queries"]
        n = len(texts)
        label = ds["name"]

        print(f"\n  📦 Dataset: {label}")

        # Encode all texts
        embeddings = encoder.encode(texts, show_progress_bar=False)
        emb = np.array(embeddings).astype("float32")

        # Encode queries
        query_texts = [q["query"] for q in queries]
        query_embs = np.array(encoder.encode(query_texts, show_progress_bar=False)).astype("float32")

        k = 5

        # --- Flat L2 (brute-force = ground truth) ---
        flat_index = faiss.IndexFlatL2(dim)
        flat_index.add(emb)

        flat_search_times = []
        flat_results_all = []
        for qe in query_embs:
            qe_2d = qe.reshape(1, -1)
            _, elapsed = timed(lambda: None)  # warmup
            (distances, indices), elapsed = timed(flat_index.search, qe_2d, k)
            flat_search_times.append(elapsed)
            flat_results_all.append(indices[0].tolist())

        flat_avg_ms = np.mean(flat_search_times) * 1000

        # --- HNSW ---
        hnsw_index = faiss.IndexHNSWFlat(dim, 32)
        hnsw_index.hnsw.efConstruction = 200
        hnsw_index.hnsw.efSearch = 128

        _, build_time = timed(hnsw_index.add, emb)

        hnsw_search_times = []
        hnsw_results_all = []
        for qe in query_embs:
            qe_2d = qe.reshape(1, -1)
            (distances, indices), elapsed = timed(hnsw_index.search, qe_2d, k)
            hnsw_search_times.append(elapsed)
            hnsw_results_all.append(indices[0].tolist())

        hnsw_avg_ms = np.mean(hnsw_search_times) * 1000

        # Recall: HNSW recall vs Flat (ground truth)
        recalls = []
        for flat_res, hnsw_res in zip(flat_results_all, hnsw_results_all):
            recalls.append(compute_recall(flat_res, hnsw_res, k))
        avg_recall = np.mean(recalls) * 100

        # Semantic accuracy: does top-1 match the expected relevant passage?
        flat_semantic_hits = 0
        hnsw_semantic_hits = 0
        for i, q in enumerate(queries):
            relevant = set(q["relevant_indices"])
            if flat_results_all[i][0] in relevant:
                flat_semantic_hits += 1
            if hnsw_results_all[i][0] in relevant:
                hnsw_semantic_hits += 1
        flat_accuracy = flat_semantic_hits / len(queries) * 100 if queries else 0
        hnsw_accuracy = hnsw_semantic_hits / len(queries) * 100 if queries else 0

        speedup = flat_avg_ms / hnsw_avg_ms if hnsw_avg_ms > 0 else float("inf")

        row = {
            "dataset": label,
            "n": n,
            "flat_avg_ms": round(flat_avg_ms, 4),
            "hnsw_avg_ms": round(hnsw_avg_ms, 4),
            "speedup": round(speedup, 2),
            "hnsw_recall_pct": round(avg_recall, 1),
            "flat_semantic_acc": round(flat_accuracy, 1),
            "hnsw_semantic_acc": round(hnsw_accuracy, 1),
            "hnsw_build_time_s": round(build_time, 3),
        }
        results.append(row)

        print(f"    Flat L2 avg search:  {flat_avg_ms:.4f} ms  |  Semantic Acc: {flat_accuracy:.1f}%")
        print(f"    HNSW   avg search:   {hnsw_avg_ms:.4f} ms  |  Semantic Acc: {hnsw_accuracy:.1f}%")
        print(f"    HNSW Recall@{k}: {avg_recall:.1f}%  |  Speedup: {speedup:.2f}x  |  Build: {build_time:.3f}s")

    return {"test": "Flat L2 vs HNSW", "results": results}


# ──────────────────────────── Test 2 ────────────────────────────

def test_chunk_sizes(encoder, datasets):
    """Test different chunk sizes: count, embed time, search time, accuracy."""
    print("\n" + "=" * 70)
    print("TEST 2: Chunk Size Variations")
    print("=" * 70)

    chunk_sizes = [500, 1000, 1500, 2000]
    dim = 768
    results = []

    # Use medium dataset text combined into one big doc
    ds = datasets[1]  # medium
    full_text = "\n\n".join(ds["texts"])
    queries = ds["queries"]

    for cs in chunk_sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cs, chunk_overlap=int(cs * 0.1), length_function=len
        )
        chunks = splitter.split_text(full_text)
        num_chunks = len(chunks)

        # Embed
        (embeddings, embed_time) = timed(encoder.encode, chunks, show_progress_bar=False)
        emb = np.array(embeddings).astype("float32")

        # Build HNSW
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efSearch = 128
        index.add(emb)

        # Search
        query_texts = [q["query"] for q in queries]
        query_embs = np.array(encoder.encode(query_texts, show_progress_bar=False)).astype("float32")

        search_times = []
        top_results = []
        for qe in query_embs:
            (d, idx), elapsed = timed(index.search, qe.reshape(1, -1), 5)
            search_times.append(elapsed)
            if idx[0][0] >= 0 and idx[0][0] < len(chunks):
                top_results.append(chunks[idx[0][0]][:100])

        avg_search_ms = np.mean(search_times) * 1000

        row = {
            "chunk_size": cs,
            "overlap": int(cs * 0.1),
            "num_chunks": num_chunks,
            "embed_time_s": round(embed_time, 3),
            "avg_search_ms": round(avg_search_ms, 4),
            "avg_chunk_len": round(np.mean([len(c) for c in chunks]), 0),
        }
        results.append(row)

        print(f"\n  chunk_size={cs}  overlap={int(cs*0.1)}")
        print(f"    Chunks: {num_chunks}  |  Avg length: {row['avg_chunk_len']:.0f} chars")
        print(f"    Embed time: {embed_time:.3f}s  |  Avg search: {avg_search_ms:.4f} ms")

    return {"test": "Chunk Size Variations", "results": results}


# ──────────────────────────── Test 3 ────────────────────────────

def test_embedding_models(datasets):
    """Compare MiniLM (384-dim) vs MPNet (768-dim): speed and accuracy."""
    print("\n" + "=" * 70)
    print("TEST 3: Embedding Model Comparison")
    print("=" * 70)

    models = [
        ("all-MiniLM-L6-v2", 384),
        ("all-mpnet-base-v2", 768),
    ]

    ds = datasets[1]  # medium
    texts = ds["texts"]
    queries = ds["queries"]
    results = []

    for model_name, dim in models:
        print(f"\n  🔄 Loading model: {model_name} (dim={dim})...")
        enc, load_time = timed(SentenceTransformer, model_name)

        # Encode texts
        (embeddings, encode_time) = timed(enc.encode, texts, show_progress_bar=False)
        emb = np.array(embeddings).astype("float32")

        # Build index
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efSearch = 128
        index.add(emb)

        # Encode and search queries
        query_texts = [q["query"] for q in queries]
        (q_embs, q_encode_time) = timed(enc.encode, query_texts, show_progress_bar=False)
        q_embs = np.array(q_embs).astype("float32")

        semantic_hits = 0
        search_times = []
        for i, q in enumerate(queries):
            (d, idx), elapsed = timed(index.search, q_embs[i].reshape(1, -1), 5)
            search_times.append(elapsed)
            if idx[0][0] in set(q["relevant_indices"]):
                semantic_hits += 1

        accuracy = semantic_hits / len(queries) * 100 if queries else 0
        avg_search_ms = np.mean(search_times) * 1000

        row = {
            "model": model_name,
            "dimension": dim,
            "load_time_s": round(load_time, 2),
            "encode_time_s": round(encode_time, 3),
            "avg_search_ms": round(avg_search_ms, 4),
            "semantic_accuracy_pct": round(accuracy, 1),
        }
        results.append(row)

        print(f"    Load: {load_time:.2f}s  |  Encode {len(texts)} texts: {encode_time:.3f}s")
        print(f"    Avg search: {avg_search_ms:.4f} ms  |  Semantic Accuracy: {accuracy:.1f}%")

    return {"test": "Embedding Model Comparison", "results": results}


# ──────────────────────────── Test 4 ────────────────────────────

def test_hnsw_params(encoder, datasets):
    """Tune HNSW parameters: M, efConstruction, efSearch."""
    print("\n" + "=" * 70)
    print("TEST 4: HNSW Parameter Tuning")
    print("=" * 70)

    dim = 768
    ds = datasets[2] if len(datasets) > 2 else datasets[-1]  # large
    texts = ds["texts"]
    queries = ds["queries"]

    embeddings = encoder.encode(texts, show_progress_bar=False)
    emb = np.array(embeddings).astype("float32")

    query_texts = [q["query"] for q in queries]
    q_embs = np.array(encoder.encode(query_texts, show_progress_bar=False)).astype("float32")

    # Ground truth from Flat L2
    flat = faiss.IndexFlatL2(dim)
    flat.add(emb)
    gt_results = []
    for qe in q_embs:
        _, idx = flat.search(qe.reshape(1, -1), 5)
        gt_results.append(idx[0].tolist())

    configs = [
        {"M": 16, "efConstruction": 100, "efSearch": 64},
        {"M": 16, "efConstruction": 200, "efSearch": 128},
        {"M": 32, "efConstruction": 200, "efSearch": 128},  # current config
        {"M": 32, "efConstruction": 400, "efSearch": 256},
        {"M": 64, "efConstruction": 200, "efSearch": 128},
        {"M": 64, "efConstruction": 400, "efSearch": 256},
    ]

    results = []
    for cfg in configs:
        index = faiss.IndexHNSWFlat(dim, cfg["M"])
        index.hnsw.efConstruction = cfg["efConstruction"]

        _, build_time = timed(index.add, emb)

        index.hnsw.efSearch = cfg["efSearch"]

        search_times = []
        hnsw_results = []
        for qe in q_embs:
            (d, idx), elapsed = timed(index.search, qe.reshape(1, -1), 5)
            search_times.append(elapsed)
            hnsw_results.append(idx[0].tolist())

        recalls = [compute_recall(gt, pred, 5) for gt, pred in zip(gt_results, hnsw_results)]
        avg_recall = np.mean(recalls) * 100
        avg_search_ms = np.mean(search_times) * 1000

        label = f"M={cfg['M']} efC={cfg['efConstruction']} efS={cfg['efSearch']}"
        is_current = cfg["M"] == 32 and cfg["efConstruction"] == 200 and cfg["efSearch"] == 128

        row = {
            "config": label,
            "M": cfg["M"],
            "efConstruction": cfg["efConstruction"],
            "efSearch": cfg["efSearch"],
            "build_time_s": round(build_time, 3),
            "avg_search_ms": round(avg_search_ms, 4),
            "recall_at_5_pct": round(avg_recall, 1),
            "is_current": is_current,
        }
        results.append(row)

        marker = " ◀ CURRENT" if is_current else ""
        print(f"    {label:40s}  Build: {build_time:.3f}s  Search: {avg_search_ms:.4f}ms  Recall@5: {avg_recall:.1f}%{marker}")

    return {"test": "HNSW Parameter Tuning", "results": results}


# ──────────────────────────── Test 5 ────────────────────────────

def test_batch_vs_single(encoder, datasets):
    """Compare batch encoding vs single-item encoding speed."""
    print("\n" + "=" * 70)
    print("TEST 5: Batch vs Single Embedding Speed")
    print("=" * 70)

    batch_sizes = [10, 30, 100]
    results = []

    for bs in batch_sizes:
        # Pick texts up to batch size
        ds_idx = 0
        for i, d in enumerate(datasets):
            if len(d["texts"]) >= bs:
                ds_idx = i
                break
        texts = datasets[ds_idx]["texts"][:bs]

        # Single encoding
        _, single_time = timed(lambda: [encoder.encode([t]) for t in texts])

        # Batch encoding
        _, batch_time = timed(encoder.encode, texts, show_progress_bar=False)

        speedup = single_time / batch_time if batch_time > 0 else float("inf")

        row = {
            "batch_size": bs,
            "single_time_s": round(single_time, 3),
            "batch_time_s": round(batch_time, 3),
            "speedup": round(speedup, 2),
        }
        results.append(row)

        print(f"    n={bs:4d}  |  Single: {single_time:.3f}s  |  Batch: {batch_time:.3f}s  |  Speedup: {speedup:.2f}x")

    return {"test": "Batch vs Single Embedding", "results": results}


# ──────────────────────────── Main ────────────────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_dir = os.path.join(project_root, "tests")

    # Ensure test data exists
    data_files = [
        os.path.join(tests_dir, f"dataset_{s}.json")
        for s in ["small", "medium", "large", "xlarge"]
    ]
    missing = [f for f in data_files if not os.path.exists(f)]
    if missing:
        print("⚠️  Test datasets missing. Generating them first...\n")
        os.system(f'"{sys.executable}" "{os.path.join(tests_dir, "generate_test_data.py")}"')
        print()

    # Load datasets
    datasets = []
    for f in data_files:
        if os.path.exists(f):
            datasets.append(load_dataset(f))
    
    if not datasets:
        print("❌ No datasets found! Run generate_test_data.py first.")
        return

    print("=" * 70)
    print("  StudyHub AI — Comprehensive Optimization Benchmark")
    print(f"  Datasets loaded: {len(datasets)}")
    print("=" * 70)

    # Load primary model
    print("\n🔄 Loading primary embedding model (all-mpnet-base-v2)...")
    encoder, load_time = timed(SentenceTransformer, "all-mpnet-base-v2")
    print(f"   Loaded in {load_time:.2f}s")

    all_results = {}

    # Run all tests
    all_results["test1"] = test_flat_vs_hnsw(encoder, datasets)
    all_results["test2"] = test_chunk_sizes(encoder, datasets)
    all_results["test3"] = test_embedding_models(datasets)
    all_results["test4"] = test_hnsw_params(encoder, datasets)
    all_results["test5"] = test_batch_vs_single(encoder, datasets)

    # Save results
    output_path = os.path.join(project_root, "tests", "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  ✅ All benchmarks complete! Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
