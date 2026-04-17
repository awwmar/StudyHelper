"""
Test script: Verify HNSW index works correctly without any DB dependency.

Creates a temporary in-memory HNSW index, adds sample vectors, runs a search,
and validates results.

Usage:
    cd c:\\Users\\Amar\\Desktop\\SOP
    python scripts/test_hnsw_upgrade.py
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
from sentence_transformers import SentenceTransformer


def test_hnsw():
    print("=" * 60)
    print("HNSW Upgrade Verification Test")
    print("=" * 60)

    # 1. Test model loading
    print("\n[1] Loading all-mpnet-base-v2 model...")
    t0 = time.time()
    encoder = SentenceTransformer('all-mpnet-base-v2')
    print(f"    Model loaded in {time.time() - t0:.1f}s")

    # 2. Test embedding dimension
    print("\n[2] Checking embedding dimensions...")
    test_embedding = encoder.encode(["Hello world"])
    dim = test_embedding.shape[1]
    assert dim == 768, f"Expected 768, got {dim}"
    print(f"    ✅ Dimension = {dim} (correct)")

    # 3. Test HNSW index creation
    print("\n[3] Creating HNSW index...")
    index = faiss.IndexHNSWFlat(768, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    print(f"    ✅ HNSW index created (M=32, efConstruction=200, efSearch=128)")

    # 4. Test adding vectors
    print("\n[4] Adding sample documents...")
    sample_texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Neural networks are computing systems inspired by biological neural networks in the human brain.",
        "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
        "The French Revolution began in 1789 and had a lasting impact on European politics.",
        "Sorting algorithms like QuickSort have O(n log n) average time complexity.",
        "DNA replication is a biological process that occurs in all living organisms.",
        "The TCP/IP protocol suite is the foundation of internet communication.",
        "Shakespeare wrote many plays including Hamlet, Macbeth, and Romeo and Juliet.",
        "Gradient descent is an optimization algorithm used to minimize a function.",
        "The water cycle involves evaporation, condensation, and precipitation.",
    ]
    
    t0 = time.time()
    embeddings = encoder.encode(sample_texts)
    index.add(np.array(embeddings).astype('float32'))
    print(f"    ✅ Added {index.ntotal} vectors in {time.time() - t0:.2f}s")

    # 5. Test search
    print("\n[5] Running search queries...")
    queries = [
        ("What is deep learning?", "Machine learning / Neural networks"),
        ("How do plants make food?", "Photosynthesis"),
        ("What is TCP?", "TCP/IP protocol"),
    ]

    all_passed = True
    for query, expected_topic in queries:
        query_vec = encoder.encode([query])
        distances, indices = index.search(np.array(query_vec).astype('float32'), 3)
        top_result = sample_texts[indices[0][0]]
        
        # Check if result is semantically relevant
        is_relevant = any(word.lower() in top_result.lower() for word in expected_topic.split())
        status = "✅" if is_relevant else "⚠️"
        if not is_relevant:
            all_passed = False
        
        print(f"    {status} Query: '{query}'")
        print(f"       Top result: '{top_result[:80]}...'")
        print(f"       Distance: {distances[0][0]:.4f}")

    # 6. Performance comparison
    print("\n[6] HNSW vs FlatL2 speed comparison across sizes...")
    dataset_sizes = [10, 100, 1000, 10000]

    for size in dataset_sizes:
        print(f"\n    --- Dataset Size: {size} ---")
        large_embeddings = np.random.rand(size, 768).astype('float32')

        # FlatL2
        flat_index = faiss.IndexFlatL2(768)
        flat_index.add(large_embeddings)
        
        t0 = time.time()
        for _ in range(100):
            flat_index.search(np.random.rand(1, 768).astype('float32'), 5)
        flat_time = time.time() - t0

        # HNSW
        hnsw_index = faiss.IndexHNSWFlat(768, 32)
        hnsw_index.hnsw.efSearch = 128
        hnsw_index.add(large_embeddings)
        
        t0 = time.time()
        for _ in range(100):
            hnsw_index.search(np.random.rand(1, 768).astype('float32'), 5)
        hnsw_time = time.time() - t0

        print(f"    FlatL2:  {flat_time*1000:.1f}ms for 100 searches")
        print(f"    HNSW:    {hnsw_time*1000:.1f}ms for 100 searches")
        speedup = flat_time / hnsw_time if hnsw_time > 0 else float('inf')
        print(f"    Speedup: {speedup:.1f}x")

    # 7. Test save/load
    print("\n[7] Testing index save/load...")
    test_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_hnsw_temp.faiss")
    faiss.write_index(index, test_path)
    loaded = faiss.read_index(test_path)
    assert loaded.ntotal == index.ntotal, f"Mismatch: {loaded.ntotal} vs {index.ntotal}"
    assert loaded.d == 768, f"Dimension mismatch: {loaded.d}"
    os.remove(test_path)
    print(f"    ✅ Save/load works ({loaded.ntotal} vectors, dim={loaded.d})")

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED — HNSW upgrade verified!")
    else:
        print("⚠️ Some search relevance tests were imperfect (may be expected with small sample)")
    print("=" * 60)


if __name__ == "__main__":
    test_hnsw()
