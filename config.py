"""
Feature flags and application configuration.
"""
import os

class Config:
    # Feature Flags
    ENABLE_KNOWLEDGE_GRAPH = True
    ENABLE_HYBRID_SEARCH = True
    ENABLE_CHUNK_LENGTH_FILTER = True
    ENABLE_LLM_CACHE = True
    ENABLE_PARALLEL_METRICS = True

config = Config()
