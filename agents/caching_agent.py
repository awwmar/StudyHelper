import logging
from services.db import DBService

logger = logging.getLogger(__name__)

class CachingAgent:
    """Agent that persists semantic memory for cross-session queries via DBService."""

    def _generate_key(self, topics: list, difficulty: str, num_questions: int, q_type: str) -> str:
        return f"{'-'.join(sorted(topics))}_{difficulty}_{num_questions}_{q_type}"

    def get_cache(self, topics: list, difficulty: str, num_questions: int, q_type: str) -> list:
        cache_key = self._generate_key(topics, difficulty, num_questions, q_type)
        val = DBService.get_agent_cache(cache_key)
        if val:
            logger.info(f"CachingAgent: Cache HIT for {cache_key}")
            return val
        return None

    def store_cache(self, topics: list, difficulty: str, num_questions: int, q_type: str, response_json: list):
        cache_key = self._generate_key(topics, difficulty, num_questions, q_type)
        logger.info(f"CachingAgent: Storing cache for {cache_key}")
        DBService.set_agent_cache(cache_key, response_json)

caching_agent = CachingAgent()
