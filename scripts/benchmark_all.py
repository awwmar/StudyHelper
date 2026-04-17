"""
Comprehensive Benchmark Script for StudyHub AI
================================================
Measures REAL timing data for:
  1. FlatL2 vs HNSW search at different dataset sizes
  2. Chunk size impact on search relevance
  3. HNSW efSearch tuning (speed vs recall tradeoff)
  4. Embedding batch size impact
  5. Index save/load (persistence) overhead

Run from project root:
    python scripts/benchmark_all.py

Outputs a JSON file: benchmark_results.json
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

RESULTS = {}
ENCODER = None

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_encoder():
    global ENCODER
    if ENCODER is None:
        print("\n[SETUP] Loading all-mpnet-base-v2 (768-dim)...")
        t0 = time.time()
        ENCODER = SentenceTransformer('all-mpnet-base-v2')
        print(f"        Model ready in {time.time()-t0:.1f}s")
    return ENCODER

def random_vecs(n, d=768):
    return np.random.rand(n, d).astype('float32')

def bench_search(index, n_queries=200, k=5, d=768):
    """Run n_queries searches and return total ms."""
    queries = random_vecs(n_queries, d)
    t0 = time.time()
    for q in queries:
        index.search(q.reshape(1, -1), k)
    return (time.time() - t0) * 1000  # ms


# ─────────────────────────────────────────────
# BENCHMARK 1 — FlatL2 vs HNSW at scale
# ─────────────────────────────────────────────
def benchmark_flat_vs_hnsw():
    print("\n" + "="*60)
    print("BENCHMARK 1: FlatL2 vs HNSW Search Speed")
    print("="*60)
    sizes = [100, 500, 1000, 5000, 10000]
    results = []
    for n in sizes:
        vecs = random_vecs(n)

        flat = faiss.IndexFlatL2(768)
        flat.add(vecs)
        flat_ms = bench_search(flat)

        hnsw = faiss.IndexHNSWFlat(768, 32)
        hnsw.hnsw.efSearch = 128
        hnsw.add(vecs)
        hnsw_ms = bench_search(hnsw)

        speedup = flat_ms / hnsw_ms if hnsw_ms > 0 else float('inf')
        row = {
            "dataset_size": n,
            "flatl2_ms_200_queries": round(flat_ms, 2),
            "hnsw_ms_200_queries": round(hnsw_ms, 2),
            "speedup_x": round(speedup, 2)
        }
        results.append(row)
        print(f"  N={n:6d} | FlatL2={flat_ms:7.1f}ms | HNSW={hnsw_ms:6.1f}ms | Speedup={speedup:.1f}x")
    RESULTS["flat_vs_hnsw"] = results


# ─────────────────────────────────────────────
# BENCHMARK 2 — Chunk size impact
# ─────────────────────────────────────────────
def benchmark_chunk_sizes():
    print("\n" + "="*60)
    print("BENCHMARK 2: Chunk Size Impact on Retrieval")
    print("="*60)
    encoder = load_encoder()

    # A representative textbook paragraph (~3000 chars)
    sample_text = (
        "Data structures are specialized formats for organizing, processing, retrieving, and storing data. "
        "Different kinds of data structures are suited to different kinds of applications and some are highly "
        "specialized to specific tasks. Trees are hierarchical data structures. A Binary Search Tree is a node-based "
        "binary tree where every node follows: left child < parent < right child. B-trees are self-balancing search "
        "trees where each node can have more than two children, making them ideal for database indexing. Hash tables "
        "store key-value pairs. A good hash function distributes keys uniformly to avoid collisions. "
        "Linked lists store elements in nodes where each node contains data and a pointer to the next node. "
        "Stacks follow Last In First Out (LIFO) and queues follow First In First Out (FIFO). "
        "Heaps are complete binary trees; min-heaps have the smallest key at top. "
        "Graphs consist of vertices and edges. Graph traversal includes DFS (depth-first search) and BFS (breadth-first search). "
        "Sorting algorithms include Bubble Sort O(n^2), Merge Sort O(n log n), Quick Sort O(n log n) average. "
        "Dynamic programming solves overlapping subproblems by storing solutions in a table (memoization). "
        "The Fibonacci sequence can be computed in O(n) using DP instead of O(2^n) with naive recursion. "
    ) * 6  # ~3000 chars

    query = "What is the time complexity of Merge Sort?"
    chunk_configs = [
        {"chunk_size": 200,  "chunk_overlap": 20},
        {"chunk_size": 500,  "chunk_overlap": 50},
        {"chunk_size": 1000, "chunk_overlap": 100},  # Current default
        {"chunk_size": 2000, "chunk_overlap": 200},
    ]

    results = []
    for cfg in chunk_configs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            length_function=len
        )
        chunks = splitter.split_text(sample_text)
        n_chunks = len(chunks)

        # Embed chunks
        t0 = time.time()
        embeddings = encoder.encode(chunks, show_progress_bar=False)
        embed_time_ms = (time.time() - t0) * 1000

        # Build HNSW
        index = faiss.IndexHNSWFlat(768, 32)
        index.hnsw.efSearch = 64
        index.add(np.array(embeddings).astype('float32'))

        # Search
        q_vec = encoder.encode([query])
        t0 = time.time()
        for _ in range(100):
            distances, indices = index.search(np.array(q_vec).astype('float32'), 3)
        search_time_ms = (time.time() - t0) * 1000

        top_chunk = chunks[indices[0][0]] if indices[0][0] != -1 else ""
        relevance_hit = "merge sort" in top_chunk.lower() or "o(n log n)" in top_chunk.lower()

        row = {
            "chunk_size": cfg["chunk_size"],
            "chunk_overlap": cfg["chunk_overlap"],
            "num_chunks_generated": n_chunks,
            "embed_time_ms": round(embed_time_ms, 1),
            "search_time_100_queries_ms": round(search_time_ms, 2),
            "top_chunk_relevant": relevance_hit,
            "top_chunk_preview": top_chunk[:120].replace("\n", " ")
        }
        results.append(row)
        print(f"  ChunkSize={cfg['chunk_size']:4d} | Chunks={n_chunks:3d} | EmbedTime={embed_time_ms:6.1f}ms | "
              f"Search(100q)={search_time_ms:6.1f}ms | Relevant={'✅' if relevance_hit else '❌'}")

    RESULTS["chunk_size_impact"] = results


# ─────────────────────────────────────────────
# BENCHMARK 3 — HNSW efSearch tuning
# ─────────────────────────────────────────────
def benchmark_efsearch_tuning():
    print("\n" + "="*60)
    print("BENCHMARK 3: HNSW efSearch Tuning (Speed vs Recall)")
    print("="*60)
    n = 5000
    d = 768
    k = 5
    ef_values = [16, 32, 64, 128, 256]

    vecs = random_vecs(n, d)

    # Ground truth via flat
    flat = faiss.IndexFlatL2(d)
    flat.add(vecs)
    queries = random_vecs(50, d)
    _, gt_indices = flat.search(queries, k)

    results = []
    for ef in ef_values:
        hnsw = faiss.IndexHNSWFlat(d, 32)
        hnsw.hnsw.efSearch = ef
        hnsw.add(vecs)

        t0 = time.time()
        _, pred_indices = hnsw.search(queries, k)
        search_ms = (time.time() - t0) * 1000

        # Recall@k
        recall_scores = []
        for i in range(len(queries)):
            gt_set = set(gt_indices[i].tolist())
            pred_set = set(pred_indices[i].tolist())
            recall_scores.append(len(gt_set & pred_set) / k)
        avg_recall = round(sum(recall_scores) / len(recall_scores), 4)

        row = {
            "efSearch": ef,
            "search_time_50_queries_ms": round(search_ms, 2),
            "avg_recall_at_5": avg_recall
        }
        results.append(row)
        print(f"  efSearch={ef:3d} | Time(50q)={search_ms:6.1f}ms | Recall@5={avg_recall:.4f}")
    RESULTS["efsearch_tuning"] = results


# ─────────────────────────────────────────────
# BENCHMARK 4 — Embedding batch size impact
# ─────────────────────────────────────────────
def benchmark_embedding_batch_size():
    print("\n" + "="*60)
    print("BENCHMARK 4: Embedding Batch Size Impact")
    print("="*60)
    encoder = load_encoder()
    texts = [f"This is sample sentence number {i} about data structures and algorithms." for i in range(256)]
    batch_sizes = [1, 8, 32, 64, 128, 256]

    results = []
    for bs in batch_sizes:
        t0 = time.time()
        encoder.encode(texts[:bs], show_progress_bar=False)
        ms = (time.time() - t0) * 1000
        per_item = ms / bs
        row = {"batch_size": bs, "total_time_ms": round(ms, 1), "per_item_ms": round(per_item, 2)}
        results.append(row)
        print(f"  BatchSize={bs:3d} | Total={ms:7.1f}ms | Per-item={per_item:.2f}ms")
    RESULTS["embedding_batch_size"] = results


# ─────────────────────────────────────────────
# BENCHMARK 5 — Index persistence overhead
# ─────────────────────────────────────────────
def benchmark_index_persistence():
    print("\n" + "="*60)
    print("BENCHMARK 5: Index Save/Load Overhead")
    print("="*60)
    sizes = [500, 2000, 5000]
    index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_bench_temp.faiss")
    results = []
    for n in sizes:
        vecs = random_vecs(n)
        hnsw = faiss.IndexHNSWFlat(768, 32)
        hnsw.add(vecs)

        t0 = time.time()
        faiss.write_index(hnsw, index_path)
        save_ms = (time.time() - t0) * 1000

        t0 = time.time()
        faiss.read_index(index_path)
        load_ms = (time.time() - t0) * 1000

        file_kb = os.path.getsize(index_path) / 1024
        row = {"vectors": n, "save_ms": round(save_ms, 1), "load_ms": round(load_ms, 1), "file_size_kb": round(file_kb, 1)}
        results.append(row)
        print(f"  N={n:5d} | Save={save_ms:.1f}ms | Load={load_ms:.1f}ms | FileSize={file_kb:.1f}KB")
    os.remove(index_path)
    RESULTS["index_persistence"] = results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  StudyHub AI — Full Benchmark Suite")
    print("="*60)

    benchmark_flat_vs_hnsw()
    benchmark_chunk_sizes()
    benchmark_efsearch_tuning()
    benchmark_embedding_batch_size()
    benchmark_index_persistence()

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)

    print("\n" + "="*60)
    print(f"  ✅ All benchmarks complete!")
    print(f"  Results saved -> {out_path}")
    print("="*60)
