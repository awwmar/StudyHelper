import json
import logging
import networkx as nx
from services.llm import llm_service
from services.db import DBService

logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    def __init__(self):
        self.graph = nx.Graph()

    def extract_and_store(self, text_chunk: str, chunk_id: str, document_id: str):
        """
        Extract entities and relationships from a text chunk using the LLM and store them
        in the knowledge graph in Supabase.
        """
        data = self.extract_triplets(text_chunk)
        if not data:
            return

        try:
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])

            # Step 1: Insert/Update Nodes in DB
            for entity in entities:
                entity = entity.lower().strip()
                DBService.add_kg_node(entity, document_id, chunk_id)

            # Step 2: Insert Edges in DB
            for rel in relationships:
                src = rel.get("source", "").lower().strip()
                tgt = rel.get("target", "").lower().strip()
                lbl = rel.get("label", "").lower().strip()
                if src and tgt and lbl:
                    DBService.add_kg_edge(src, tgt, lbl)

            logger.info(f"Knowledge Graph updated with {len(entities)} entities and {len(relationships)} relationships for chunk {chunk_id}.")

        except Exception as e:
            logger.error(f"Failed to store Knowledge Graph data: {e}")

    def extract_triplets(self, text_chunk: str) -> dict:
        """
        Pure extraction logic. Returns JSON triplets without DB storage.
        Used for research benchmarking.
        """
        prompt = f"""
You are a Knowledge Graph entity extractor.
Extract the core entities and their relationships from the given text chunk.

Rules:
1. Entities should be simple, clear concepts, names, or terms (e.g., "Mitosis", "Cell Division", "DNA").
2. Relationships should be short labels (e.g., "is a type of", "depends on", "causes").
3. Return ONLY valid JSON in this format:
{{
  "entities": ["entity1", "entity2", ...],
  "relationships": [
    {{"source": "entity1", "target": "entity2", "label": "relationship_label"}}
  ]
}}

Text Chunk:
{text_chunk[:3000]}
"""
        response_text = llm_service.invoke_custom(prompt)
        try:
            clean_response = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_response)
        except Exception as e:
            logger.error(f"Failed to extract triplets: {e}")
            return None

    def extract_query_concepts(self, query: str) -> list:
        """
        Extracts key concepts from a user query to fetch related nodes from KG.
        """
        prompt = f"""
Extract 2-4 key conceptual entities from this user query.
Return ONLY a valid JSON array of strings. Example: ["concept 1", "concept 2"]

User Query: {query}
"""
        response_text = llm_service.invoke_custom(prompt)
        try:
            clean_response = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_response)
            if isinstance(data, list):
                return [d.lower().strip() for d in data]
        except Exception as e:
            logger.error(f"Failed to extract concepts from query: {e}")
        return []

knowledge_graph_service = KnowledgeGraphService()
