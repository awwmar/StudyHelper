import logging

logger = logging.getLogger(__name__)

class PromptOptimizationAgent:
    """Implements Corpus-in-Context positioning to mitigate 'Lost in the middle' syndrome."""

    def format_context(self, chunks: list, max_tokens_estimate: int = 6000) -> str:
        """
        Reorders chunks so the most relevant are at the absolute top and bottom.
        Assumes chunks is sorted by relevance (highest first).
        """
        logger.info("PromptOptimizationAgent: Reordering context chunks...")
        if not chunks:
            return ""

        n = len(chunks)
        reordered = [None] * n
        left = 0
        right = n - 1

        # Place highest at boundaries: #1 at start, #2 at end, #3 at start+1, #4 at end-1, etc.
        for i in range(n):
            if i % 2 == 0:
                reordered[left] = chunks[i]['text_content']
                left += 1
            else:
                reordered[right] = chunks[i]['text_content']
                right -= 1

        # Combine and naive truncate
        final_text = "\n\n---\n\n".join(reordered)
        return final_text[:max_tokens_estimate]

prompt_optimization_agent = PromptOptimizationAgent()
