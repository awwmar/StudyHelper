import os
import logging
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from services.db import DBService
from rank_bm25 import BM25Okapi
from config import config

logger = logging.getLogger(__name__)

# --- Configuration ---
EMBEDDING_MODEL = 'all-mpnet-base-v2'   # 768-dim, higher quality than MiniLM
EMBEDDING_DIMENSION = 768
HNSW_M = 32              # Number of neighbors per node in the HNSW graph
HNSW_EF_CONSTRUCTION = 200  # Build-time quality (higher = better recall, slower build)
HNSW_EF_SEARCH = 128        # Search-time quality (higher = better recall, slower search)


class VectorService:
    def __init__(self, index_path="index.faiss"):
        self.index_path = index_path
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.dimension = EMBEDDING_DIMENSION
        self.index = self._load_index()
        self.bm25_corpus = []
        self.bm25_chunk_ids = []
        self.bm25_index = None
        self.bm25_path = "bm25_index.pkl"
        self._load_bm25()

    def _load_bm25(self):
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25_corpus = data["corpus"]
                self.bm25_chunk_ids = data["chunk_ids"]
                if self.bm25_corpus:
                    tokenized_corpus = [doc.split(" ") for doc in self.bm25_corpus]
                    self.bm25_index = BM25Okapi(tokenized_corpus)
        else:
            self.bm25_corpus = []
            self.bm25_chunk_ids = []
            self.bm25_index = None

    def _create_hnsw_index(self):
        """Creates a new HNSW index with tuned parameters."""
        index = faiss.IndexHNSWFlat(self.dimension, 32) # Tuned M=32
        index.hnsw.efConstruction = 100 # Tuned for build quality
        index.hnsw.efSearch = 64 # Tuned search beam width: tradeoff higher recall for slightly slower query
        return index

    def _load_index(self):
        if os.path.exists(self.index_path):
            index = faiss.read_index(self.index_path)
            # Detect dimension mismatch from old 384-dim index
            if index.d != self.dimension:
                logger.warning(
                    f"Index dimension ({index.d}) != model dimension ({self.dimension}). "
                    f"Creating a new HNSW index. Run 'python scripts/migrate_index.py' to re-index existing documents."
                )
                return self._create_hnsw_index()
            logger.info(f"FAISS index loaded: {index.ntotal} vectors, dimension={index.d}")
            return index
        else:
            logger.info("No existing index found. Creating new HNSW index.")
            return self._create_hnsw_index()

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.bm25_path, 'wb') as f:
            pickle.dump({"corpus": self.bm25_corpus, "chunk_ids": self.bm25_chunk_ids}, f)

    def add_texts(self, texts: list, chunk_ids: list):
        """
        Embeds texts, adds to FAISS, and maps index -> chunk_id in DB.
        """
        if not texts:
            return

        embeddings = self.encoder.encode(texts, show_progress_bar=len(texts) > 50)
        start_idx = self.index.ntotal
        
        # Add to FAISS
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Save immediately to avoid data loss
        self.save_index()

        # Map logic IDs to Supabase Chunk IDs
        for i, chunk_id in enumerate(chunk_ids):
            faiss_id = start_idx + i
            DBService.add_faiss_map(faiss_id, chunk_id)
            
        if config.ENABLE_HYBRID_SEARCH:
            self.bm25_corpus.extend(texts)
            self.bm25_chunk_ids.extend(chunk_ids)
            tokenized_corpus = [doc.split(" ") for doc in self.bm25_corpus]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            self.save_index()
        
        logger.info(f"Added {len(texts)} vectors to HNSW index (total: {self.index.ntotal})")

    def search(self, query: str, k: int = 5):
        """
        Returns list of chunk objects from DB, retrieved via HNSW similarity search.
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on empty index, returning no results.")
            return []

        query_vec = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:
                # Retrieve chunk via DB
                chunk = DBService.get_chunk_by_faiss_index(int(idx))
                if chunk:
                    results.append(chunk)
        return results

    def bm25_search(self, query: str, k: int = 5):
        if not self.bm25_index:
            return []
            
        tokenized_query = query.split(" ")
        scores = self.bm25_index.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            if scores[idx] > 0:
                cid = self.bm25_chunk_ids[idx]
                chunk_resp = DBService.supabase.table("chunks").select("*").eq("id", cid).execute()
                if chunk_resp.data:
                    results.append(chunk_resp.data[0])
                    
        return results

# Singleton instance
vector_service = VectorService()
