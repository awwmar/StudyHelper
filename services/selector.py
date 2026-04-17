"""
Selector Agent — Re-ranks retrieved chunks using a cross-encoder model.

The cross-encoder scores each (query, chunk) pair together, providing more accurate 
relevance scores than bi-encoder (FAISS) similarity. This acts as a second-stage filter:
    1. FAISS retrieves top-K candidates (fast, approximate)
    2. Selector re-ranks them and keeps the best top-N (slow, accurate)

Based on the "Agentic AI Quiz-Based Learning" paper's Selector Agent concept.
"""
import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Cross-encoder model — small, fast, effective for re-ranking
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


class SelectorAgent:
    def __init__(self):
        logger.info(f"Loading cross-encoder model: {CROSS_ENCODER_MODEL}")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("Selector Agent ready.")

    def select(self, query: str, chunks: list, top_ratio: float = 0.5, min_chunks: int = 3) -> list:
        """
        Re-ranks retrieved chunks using the cross-encoder and returns the best ones.

        Args:
            query: The user's search query
            chunks: List of chunk dicts (must have 'text_content' key)
            top_ratio: Fraction of chunks to keep (default: top 50%)
            min_chunks: Minimum number of chunks to return (even if ratio gives fewer)

        Returns:
            Filtered and re-ranked list of chunk dicts (highest relevance first)
        """
        if not chunks:
            return []

        if len(chunks) <= min_chunks:
            # Not enough chunks to filter, return all
            return chunks

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, chunk['text_content']) for chunk in chunks]

        # Score all pairs
        scores = self.cross_encoder.predict(pairs)

        # Pair scores with original chunks
        scored_chunks = list(zip(scores, chunks))

        # Sort by score descending (higher = more relevant)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Calculate how many to keep
        top_k = max(min_chunks, int(len(chunks) * top_ratio))
        selected = scored_chunks[:top_k]

        logger.info(
            f"Selector: {len(chunks)} chunks → {len(selected)} selected "
            f"(score range: {selected[-1][0]:.3f} to {selected[0][0]:.3f})"
        )

        # Return just the chunk dicts (without scores)
        return [chunk for _, chunk in selected]

    def select_with_scores(self, query: str, chunks: list) -> list:
        """
        Returns ALL chunks with their relevance scores (for debugging/analysis).

        Returns:
            List of (score, chunk_dict) tuples, sorted by score descending
        """
        if not chunks:
            return []

        pairs = [(query, chunk['text_content']) for chunk in chunks]
        scores = self.cross_encoder.predict(pairs)

        scored_chunks = list(zip(scores.tolist(), chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        return scored_chunks


# Singleton instance
selector_agent = SelectorAgent()
