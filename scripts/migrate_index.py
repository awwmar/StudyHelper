"""
Migration Script: Rebuild FAISS index with new 768-dim HNSW configuration.

This script:
1. Fetches all text chunks from Supabase
2. Re-embeds them with the new all-mpnet-base-v2 model (768-dim)
3. Builds a new HNSW index
4. Updates the faiss_map table in Supabase
5. Saves the new index.faiss

Usage:
    cd c:\\Users\\Amar\\Desktop\\SOP
    python scripts/migrate_index.py
"""
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from services.client import supabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Config (must match vector.py) ---
EMBEDDING_MODEL = 'all-mpnet-base-v2'
EMBEDDING_DIMENSION = 768
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128
INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'index.faiss')


def fetch_all_chunks():
    """Fetch all chunks from Supabase, handling pagination."""
    all_chunks = []
    page_size = 1000
    offset = 0

    while True:
        response = supabase.table("chunks").select("id, text_content").range(offset, offset + page_size - 1).execute()
        if not response.data:
            break
        all_chunks.extend(response.data)
        if len(response.data) < page_size:
            break
        offset += page_size

    return all_chunks


def clear_faiss_map():
    """Clear the existing faiss_map table entries."""
    try:
        # Delete all rows — Supabase requires a filter, so we use a dummy one
        supabase.table("faiss_map").delete().gte("faiss_index", 0).execute()
        logger.info("Cleared faiss_map table.")
    except Exception as e:
        logger.warning(f"Could not clear faiss_map (may be empty): {e}")


def migrate():
    logger.info("=" * 60)
    logger.info("FAISS Index Migration: FlatL2 (384-dim) -> HNSW (768-dim)")
    logger.info("=" * 60)

    # 1. Backup old index
    if os.path.exists(INDEX_PATH):
        backup_path = INDEX_PATH + ".bak"
        import shutil
        shutil.copy2(INDEX_PATH, backup_path)
        logger.info(f"Backed up old index to: {backup_path}")

    # 2. Fetch all chunks
    logger.info("Fetching chunks from Supabase...")
    chunks = fetch_all_chunks()
    logger.info(f"Found {len(chunks)} chunks to re-index.")

    if not chunks:
        logger.info("No chunks found. Creating empty HNSW index.")
        index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, HNSW_M)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = HNSW_EF_SEARCH
        faiss.write_index(index, INDEX_PATH)
        logger.info(f"Empty HNSW index saved to: {INDEX_PATH}")
        return

    # 3. Load new encoder
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL} (this may take a minute on first load)...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    # 4. Embed all chunks
    texts = [c['text_content'] for c in chunks]
    chunk_ids = [c['id'] for c in chunks]

    logger.info(f"Encoding {len(texts)} chunks with {EMBEDDING_MODEL}...")
    embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=64)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # 5. Build HNSW index
    logger.info("Building HNSW index...")
    index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(np.array(embeddings).astype('float32'))
    logger.info(f"HNSW index built with {index.ntotal} vectors.")

    # 6. Save index
    faiss.write_index(index, INDEX_PATH)
    logger.info(f"New index saved to: {INDEX_PATH}")

    # 7. Rebuild faiss_map in Supabase
    logger.info("Rebuilding faiss_map table...")
    clear_faiss_map()

    # Insert in batches
    batch_size = 100
    for i in range(0, len(chunk_ids), batch_size):
        batch = [
            {"faiss_index": i + j, "chunk_id": chunk_ids[i + j]}
            for j in range(min(batch_size, len(chunk_ids) - i))
        ]
        supabase.table("faiss_map").insert(batch).execute()

    logger.info(f"faiss_map rebuilt with {len(chunk_ids)} entries.")

    # 8. Quick verification
    logger.info("Running quick search verification...")
    test_query = texts[0][:200]  # Use first chunk as test query
    query_vec = encoder.encode([test_query])
    distances, indices = index.search(np.array(query_vec).astype('float32'), 3)
    logger.info(f"Test search returned indices: {indices[0]}, distances: {distances[0]}")

    logger.info("=" * 60)
    logger.info("Migration complete!")
    logger.info(f"  Vectors: {index.ntotal}")
    logger.info(f"  Dimension: {EMBEDDING_DIMENSION}")
    logger.info(f"  Index type: HNSW (M={HNSW_M})")
    logger.info("=" * 60)


if __name__ == "__main__":
    migrate()
